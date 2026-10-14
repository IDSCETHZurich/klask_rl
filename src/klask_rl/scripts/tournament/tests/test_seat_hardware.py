"""CPU tests of per-agent match hardware (each seat on its own agent's hardware), without Isaac Sim.

Runs the REAL ``evaluation._make_eval_env`` and ``_seat_hardware_chain`` (extracted from their source) and the REAL
wrappers (collision avoidance, opponent frame, learned v1 actuator model with the repository's checkpoint, actuator
lag, velocity scaling) around a fake simulator. The simulator records the peg commands it receives and returns a
seeded observation stream that does not depend on the actions (pegs near and beyond the edges, random velocities,
random episode ends), so runs with different hardware see the same observations and their columns are comparable:

  * same block on both seats: the per-seat chain is bit-identical to the shared chain (v1/v1, v2/v2);
  * mixed blocks: each seat's columns are bit-identical to the shared chain of that seat's block;
  * plan -> configuration_for_job: the block follows the agent when the seats swap between legs.

Plus the config parsing, the refusals and the manifest / metadata records. Run in the Isaac Lab container:

    CUDA_VISIBLE_DEVICES= python.sh -m pytest -q scripts/tournament/tests/test_seat_hardware.py
"""

import ast
import copy
import importlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import yaml

gym = pytest.importorskip("gymnasium")
from omegaconf import OmegaConf  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[2]
TASK = SCRIPTS.parent / "source/klask_rl/klask_rl/tasks/manager_based/klask_rl"
sys.path.insert(0, str(SCRIPTS / "tournament"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_hardware import HARDWARE_KEYS, PRESETS, V1_ACTUATOR_CHECKPOINT, training_hardware  # noqa: E402
from report import hardware_line  # noqa: E402
from test_match_hardware import _project  # noqa: E402
from worker import configuration_for_job, seat_hardware_record  # noqa: E402

LAG_MODULE = "klask_rl.tasks.manager_based.klask_rl.actuator_model.actuator_lag"
OBS_MODULE = "klask_rl.tasks.manager_based.klask_rl.env_cfg.klask_rl_observations_cfg"
# The v1 table's learned actuator model (the training checkpoint when the logs are mounted, else the model shipped
# with the source tree: same network architecture, other weights).
MODEL_CKPT = Path(V1_ACTUATOR_CHECKPOINT)
if not MODEL_CKPT.is_file():
    MODEL_CKPT = TASK / "actuator_model" / (
        "model_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states.pt"
    )
NUM_ENVS, STEPS = 6, 300


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observation_stream(steps, num_envs, seed=7):
    """Seeded observations (policy/opponent, 20 values) and episode ends, independent of the actions."""
    g = torch.Generator().manual_seed(seed)

    def uniform(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=g)

    margin = 0.01  # also beyond the edges: the collision avoidance's hard stop
    frames = []
    for _ in range(steps + 1):
        policy = torch.zeros(num_envs, 20)
        policy[:, 0] = uniform(-0.16 - margin, 0.16 + margin, num_envs)  # Peg_1 x, world frame
        policy[:, 1] = uniform(-0.21 - margin, -0.02 + margin, num_envs)  # Peg_1 y (player half)
        policy[:, 2:4] = uniform(-1.5, 1.5, num_envs, 2)  # Peg_1 velocity
        policy[:, 4] = uniform(-0.16 - margin, 0.16 + margin, num_envs)  # Peg_2 x
        policy[:, 5] = uniform(0.02 - margin, 0.21 + margin, num_envs)  # Peg_2 y (opponent half)
        policy[:, 6:8] = uniform(-1.5, 1.5, num_envs, 2)
        policy[:, 8:12] = uniform(-0.2, 0.2, num_envs, 4)
        opponent = torch.cat([-policy[:, 4:8], -policy[:, 0:4], -policy[:, 8:12], policy[:, 12:]], dim=1)
        done = torch.rand(num_envs, generator=g) < 0.03
        frames.append(({"policy": policy, "opponent": opponent}, done))
    return frames


class StreamSim(gym.Env):
    """Stands in for gym.make(task, cfg): records the 4 peg commands; returns the seeded observation stream."""

    def __init__(self, cfg, frames):
        self.cfg = cfg
        self.num_envs = cfg.scene.num_envs
        self.device = "cpu"
        self.step_dt = cfg.decimation * cfg.sim.dt
        self.single_action_space = gym.spaces.Box(-1, 1, (4,))
        self.action_manager = SimpleNamespace(_terms={"pegs": SimpleNamespace(_scale=0.4)})
        self.frames, self.t = frames, 0
        self.received, self.saturation, self.delta = [], [], []

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        obs, _ = self.frames[self.t]
        return {key: value.clone() for key, value in obs.items()}

    def reset(self, *args, **kwargs):
        self.t = 0
        return self._obs(), {}

    def step(self, actions):
        self.received.append(actions.clone())
        self.saturation.append(self._action_rate_cap_saturation.clone())
        self.delta.append(self._action_delta.clone())
        self.t += 1
        done = self.frames[self.t][1].clone()
        return self._obs(), torch.zeros(self.num_envs), done, torch.zeros_like(done), {}


class FakeEnvCfg:
    def __init__(self):
        self.sim = SimpleNamespace(device="cpu", dt=0.005, render=None)
        self.scene = SimpleNamespace(num_envs=1, camera=None)
        self.observations = SimpleNamespace(image=None, opponent_image=None)
        self.events = SimpleNamespace(reset_ball_position=SimpleNamespace(params={"pose_range": {}}))
        self.ball_reset_position_x = self.ball_reset_position_y = None
        self.terminations = SimpleNamespace(
            time_out=1, goal_scored=1, goal_conceded=1, player_in_goal=1, opponent_in_goal=1
        )
        self.decimation, self.seed, self.episode_length_s = 4, 0, 1.0


@pytest.fixture
def make_eval_env(monkeypatch):
    params = _load(SCRIPTS.parent / "source/klask_rl/klask_rl/assets/robots/klask_params.py", "klask_params")
    monkeypatch.setitem(sys.modules, "klask_rl.assets.robots.klask_params", params)
    robots = ModuleType("klask_rl.assets.robots")
    robots.klask_params = params
    monkeypatch.setitem(sys.modules, "klask_rl.assets.robots", robots)
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    training = _load(TASK / "wrappers/klask_rl_training_wrappers.py", "klask_training_wrappers")
    observation = _load(TASK / "wrappers/klask_rl_ovservation_wrappers.py", "klask_observation_wrappers")
    velocity = _load(TASK / "wrappers/velocity_scale_wrapper.py", "klask_velocity_scale_wrapper")
    model = _load(TASK / "actuator_model/actuator_model.py", "klask_actuator_model")
    monkeypatch.setitem(sys.modules, LAG_MODULE, _load(TASK / "actuator_model/actuator_lag.py", LAG_MODULE))
    obs_cfg = ModuleType(OBS_MODULE)
    obs_cfg._FastSACPlayerPolicyCfg = obs_cfg._FastSACOpponentPolicyCfg = lambda: None
    monkeypatch.setitem(sys.modules, OBS_MODULE, obs_cfg)
    env_cfg_utils = _load(SCRIPTS / "dreamer/env_cfg_utils.py", "klask_env_cfg_utils")
    frames = _observation_stream(STEPS, NUM_ENVS)
    sims = []

    def make(name, cfg):
        sims.append(StreamSim(cfg, frames))
        return sims[-1]

    fake_gym = SimpleNamespace(
        spec=lambda name: SimpleNamespace(kwargs={"env_cfg_entry_point": FakeEnvCfg}),
        make=make,
    )
    source = (SCRIPTS / "dreamer/evaluation.py").read_text()
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in ("_make_eval_env", "_seat_hardware_chain")
    ]
    assert len(functions) == 2
    namespace = {
        "gym": fake_gym,
        "importlib": importlib,
        "OmegaConf": OmegaConf,
        "RenderCfg": lambda **kwargs: kwargs,
        "apply_camera_size_to_env_cfg": env_cfg_utils.apply_camera_size_to_env_cfg,
        "configure_domain_randomization": lambda env_cfg, dr: setattr(env_cfg, "dr", dr),
        "KLASK_PARAMS": params.KLASK_PARAMS,
        "ActuatorModelWrapper": model.ActuatorModelWrapper,
        "KlaskRlCollisionAvoidanceWrapper": training.KlaskRlCollisionAvoidanceWrapper,
        "OpponentActionWrapper": observation.OpponentActionWrapper,
        "ObservationNoiseWrapper": observation.ObservationNoiseWrapper,
        "VelocityScaleWrapper": velocity.VelocityScaleWrapper,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "evaluation.py", "exec"), namespace)
    return namespace["_make_eval_env"], sims


def _block(preset):
    block = copy.deepcopy(PRESETS[preset])
    block["actuator_model"]["checkpoint"] = str(MODEL_CKPT)
    return block


def _env_config(shared=None, seats=None):
    env = {
        "task": "isaaclab_Klask-Rl-Dreamer-Sprite-v0",
        "device": "cpu",
        "seed": 0,
        "sim_dt": 0.001,
        "decimation": 20,
        "episode_length_s": 30.0,
    }
    if shared is not None:
        env.update(copy.deepcopy(shared))
    if seats is not None:
        env["seat_hardware"] = [dict(copy.deepcopy(block), agent=f"seat{i}") for i, block in enumerate(seats)]
    return OmegaConf.create(env)


ACTIONS = (torch.rand(STEPS, NUM_ENVS, 4, generator=torch.Generator().manual_seed(3)) * 2.6 - 1.3)


def _run(make_eval_env, env_config, actions=ACTIONS):
    """Drive the chain; return what the simulator received and VelocityScaleWrapper's per-step signals."""
    make, sims = make_eval_env
    env, opponent = make(env_config, NUM_ENVS, tournament=True)
    assert opponent is None
    env.reset()
    for step_actions in actions:
        env.step(step_actions.clone())
    sim = sims[-1]
    return env, {
        "received": torch.stack(sim.received),
        "saturation": torch.stack(sim.saturation),
        "delta": torch.stack(sim.delta),
    }


def _chain(env):
    out = []
    while hasattr(env, "env"):
        out.append(env)
        env = env.env
    return out


def _names(env):
    return [type(wrapper).__name__ for wrapper in _chain(env)]


def _assert_equal(a, b):
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_stream_exercises_the_edges_resets_and_the_rate_cap(make_eval_env):
    """Sanity of the test input: collision clipping, episode ends and rate-cap saturation all occur."""
    _, sims = make_eval_env
    _, unclipped = _run(make_eval_env, _env_config(shared={**_block("v2"), "collision_avoidance": {"enable": False}}))
    _, clipped = _run(make_eval_env, _env_config(shared=_block("v2")))
    assert (unclipped["received"] != clipped["received"]).float().mean() > 0.05
    assert sum(int(done.sum()) for _, done in sims[-1].frames) > 20
    assert clipped["saturation"].float().mean() > 0.05


@pytest.mark.parametrize("preset", ["v1", "v2"])
def test_same_block_on_both_seats_is_bit_identical_to_the_shared_chain(make_eval_env, preset):
    shared_env, shared = _run(make_eval_env, _env_config(shared=_block(preset)))
    seat_env, per_seat = _run(make_eval_env, _env_config(seats=[_block(preset), _block(preset)]))
    _assert_equal(per_seat, shared)
    actuator = "ActuatorModelWrapper" if preset == "v1" else "ActuatorLagWrapper"
    expected = ["VelocityScaleWrapper", actuator, "OpponentActionWrapper", "KlaskRlCollisionAvoidanceWrapper"]
    assert _names(shared_env) == _names(seat_env) == expected
    scale = _chain(seat_env)[0]
    assert scale.max_velocity == (PRESETS[preset]["max_velocity"],) * 4
    assert torch.equal(_chain(seat_env)[0]._prev_v, _chain(shared_env)[0]._prev_v)


@pytest.mark.parametrize("seat0, seat1", [("v2", "v1"), ("v1", "v2")])
def test_mixed_seats_each_get_exactly_their_own_hardware(make_eval_env, seat0, seat1):
    _, alone0 = _run(make_eval_env, _env_config(shared=_block(seat0)))
    _, alone1 = _run(make_eval_env, _env_config(shared=_block(seat1)))
    env, mixed = _run(make_eval_env, _env_config(seats=[_block(seat0), _block(seat1)]))
    for key in mixed:
        assert torch.equal(mixed[key][..., 0:2], alone0[key][..., 0:2]), key
        assert torch.equal(mixed[key][..., 2:4], alone1[key][..., 2:4]), key
    # ... and differ from the other hardware (the test would not see a swap otherwise).
    assert not torch.equal(mixed["received"][..., 0:2], alone1["received"][..., 0:2])
    assert not torch.equal(mixed["received"][..., 2:4], alone0["received"][..., 2:4])
    chain = _chain(env)
    assert _names(env) == [
        "VelocityScaleWrapper",
        "ActuatorModelWrapper",
        "ActuatorLagWrapper",
        "OpponentActionWrapper",
        "KlaskRlCollisionAvoidanceWrapper",
    ]
    v1_seat = (seat0, seat1).index("v1")
    v2_seat = 1 - v1_seat
    assert chain[1]._peg1 == (v1_seat == 0) and chain[1]._peg2 == (v1_seat == 1)
    assert chain[2].columns == (slice(0, 2), slice(2, 4))[v2_seat]
    speeds = tuple(PRESETS[s]["max_velocity"] for s in (seat0, seat1))
    assert chain[4]._peg_max_vel == speeds
    assert chain[0].max_velocity == (speeds[0], speeds[0], speeds[1], speeds[1])
    assert chain[0].max_acceleration == tuple(PRESETS[s]["max_acceleration"] for s in (seat0, seat0, seat1, seat1))
    # v1 seat peaks at its 0.4 m/s scale (model output), v2 seat reaches v2 speeds.
    assert mixed["received"][..., 2 * v2_seat : 2 * v2_seat + 2].abs().max() > 1.2


def test_uncapped_and_unclipped_seats(make_eval_env):
    """A seat without rate cap / collision avoidance is passed through like the shared chain without them."""
    free = {"max_velocity": 0.8, "collision_avoidance": {"enable": False}}
    _, alone_free = _run(make_eval_env, _env_config(shared=free))
    _, alone_v2 = _run(make_eval_env, _env_config(shared=_block("v2")))
    env, mixed = _run(make_eval_env, _env_config(seats=[_block("v2"), free]))
    assert torch.equal(mixed["received"][..., 0:2], alone_v2["received"][..., 0:2])
    assert torch.equal(mixed["received"][..., 2:4], alone_free["received"][..., 2:4])
    assert torch.equal(mixed["saturation"][..., 2:4], alone_free["saturation"][..., 2:4])
    assert not mixed["saturation"][..., 2:4].any()
    assert _chain(env)[-1]._pegs == (True, False)


def test_two_lag_settings_and_shared_keys_refusals(make_eval_env):
    slow = copy.deepcopy(_block("v2"))
    slow["actuator_lag"] = {"enable": True, "dead_time_s": 0.01, "tau_s": 0.03}
    _, alone_slow = _run(make_eval_env, _env_config(shared=slow))
    _, alone_v2 = _run(make_eval_env, _env_config(shared=_block("v2")))
    env, mixed = _run(make_eval_env, _env_config(seats=[slow, _block("v2")]))
    assert torch.equal(mixed["received"][..., 0:2], alone_slow["received"][..., 0:2])
    assert torch.equal(mixed["received"][..., 2:4], alone_v2["received"][..., 2:4])
    assert _names(env).count("ActuatorLagWrapper") == 2
    make, _ = make_eval_env
    with pytest.raises(ValueError, match="must not set both"):
        make(_env_config(shared=_block("v2"), seats=[_block("v2"), _block("v1")]), 2, tournament=True)
    both = copy.deepcopy(_block("v2"))
    both["actuator_model"]["enable"] = True
    with pytest.raises(ValueError, match="alternatives"):
        make(_env_config(seats=[_block("v1"), both]), 2, tournament=True)


# ---- plan / manifest / worker env ----


def _write_ppo_config(root, hardware):
    """A PPO-B agent.yaml with the network build_plan checks and its training env block (as rl_games saves it)."""
    path = root / "logs/ppo/agent.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    env = {"sim_dt": 0.001, "decimation": 20, **copy.deepcopy(hardware)}
    path.write_text(yaml.safe_dump({"params": {"network": {"mlp": {"units": [256, 128, 64]}}}, "env": env}))


@pytest.fixture
def plan_of(tmp_path, monkeypatch):
    """build_plan on the synthetic project of test_match_hardware, with a PPO-B config and the v1 model file."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    from run_tournament import build_plan

    counter = iter(range(100))

    def plan(hardware=None, extra=None, ppo_hardware=PRESETS["v1"]):
        args = _project(tmp_path, next(counter), hardware=hardware, extra=extra)
        _write_ppo_config(args.project_root, ppo_hardware)
        model = args.project_root / Path(V1_ACTUATOR_CHECKPOINT).relative_to("/workspace/klask_rl")
        model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MODEL_CKPT, model)
        return build_plan(args)

    return plan


OWN = {"per_agent": {"State": "v2", "PPO-B": "v1"}}


def _job_envs(plan):
    return [
        OmegaConf.to_container(configuration_for_job(plan, dict(job, device="cpu")).env) for job in plan["jobs"]
    ]


def test_per_agent_plan_records_each_agents_block(plan_of):
    plan, blockers = plan_of(hardware=OWN)
    assert not any("actuator checkpoint" in b or "hardware source" in b for b in blockers)
    env = plan["evaluation_config"]["env"]
    assert not set(HARDWARE_KEYS) & set(env)  # no shared hardware keys
    assert env["per_agent_hardware"]["State"] == PRESETS["v2"]
    v1 = env["per_agent_hardware"]["PPO-B"]
    assert {k: v for k, v in v1.items() if k != "actuator_model"} == {
        k: v for k, v in PRESETS["v1"].items() if k != "actuator_model"
    }
    assert v1["actuator_model"]["enable"] and len(v1["actuator_model"]["sha256"]) == 64  # resolved and hashed
    hw = plan["hardware"]
    assert hw["setting"] == "per_agent" and hw["seats_share_block"] is False and hw["block"] is None
    assert hw["per_agent"] == {
        "State": {"setting": "v2", "block": PRESETS["v2"]},
        "PPO-B": {"setting": "v1", "block": PRESETS["v1"]},
    }
    # Each agent's training hardware (Dreamer .hydra config, PPO agent.yaml env block) equals its seat block.
    assert set(hw["agents"]) == {"State", "PPO-B"} and hw["mismatched_agents"] == []
    assert hw["agents"]["PPO-B"]["training_hardware"] == PRESETS["v1"]
    assert hw["timing"] == {"sim_dt": 0.001, "decimation": 20}
    # The jobs name each seat's hardware: it follows the agent when the seats swap.
    assert [(j["seat0"], j["seat1"], j["seat_hardware"]) for j in plan["jobs"]] == [
        ("State", "PPO-B", ["v2", "v1"]),
        ("PPO-B", "State", ["v1", "v2"]),
    ]
    line = hardware_line(hw)
    assert "PPO-B (v1): max_velocity 0.4 m/s" in line and "State (v2): max_velocity 1.4 m/s" in line
    json.dumps(plan)


def test_seat_swap_keeps_the_hardware_with_the_agent(plan_of):
    plan, _ = plan_of(hardware=OWN)
    for job, env in zip(plan["jobs"], _job_envs(plan)):
        assert "per_agent_hardware" not in env and not set(HARDWARE_KEYS) & set(env)
        seats = env["seat_hardware"]
        assert [seat["agent"] for seat in seats] == [job["seat0"], job["seat1"]]
        for seat in seats:
            expected = PRESETS["v2" if seat["agent"] == "State" else "v1"]
            assert seat["max_velocity"] == expected["max_velocity"]
            assert seat["max_acceleration"] == expected["max_acceleration"]
            assert seat["actuator_model"]["enable"] == expected["actuator_model"]["enable"]
            assert ("actuator_lag" in seat) == ("actuator_lag" in expected)
        record = seat_hardware_record(plan, job)
        assert [(r["seat"], r["agent"], r["setting"]) for r in record] == [
            (0, job["seat0"], job["seat_hardware"][0]),
            (1, job["seat1"], job["seat_hardware"][1]),
        ]
        json.dumps(record)


def test_job_chain_follows_the_agent_across_legs(plan_of, make_eval_env):
    """Plan -> configuration_for_job -> _make_eval_env: the State agent's columns run v2 physics in both legs."""
    plan, _ = plan_of(hardware=OWN)
    _, v2_alone = _run(make_eval_env, _env_config(shared=_block("v2")))
    _, v1_alone = _run(make_eval_env, _env_config(shared=_block("v1")))
    for job in plan["jobs"]:
        cfg = configuration_for_job(plan, dict(job, device="cpu"))
        ppob = [seat for seat in cfg.env.seat_hardware if seat.agent == "PPO-B"][0]
        # The plan resolved the project's v1 model (here a copy of the source-tree checkpoint) and hashed it.
        assert ppob.actuator_model.checkpoint.endswith("train_seed0.pt") and len(ppob.actuator_model.sha256) == 64
        _, mixed = _run(make_eval_env, cfg.env)
        for seat, name in enumerate((job["seat0"], job["seat1"])):
            alone = v2_alone if name == "State" else v1_alone
            cols = slice(2 * seat, 2 * seat + 2)
            assert torch.equal(mixed["received"][..., cols], alone["received"][..., cols]), (job["id"], name)


@pytest.mark.parametrize("preset", ["v1", "v2"])
def test_same_block_for_every_player_equals_the_shared_setting(plan_of, preset):
    shared, _ = plan_of(hardware=preset)
    per_agent, _ = plan_of(hardware={"per_agent": {"State": preset, "PPO-B": preset}})
    assert per_agent["evaluation_config"]["env"] == shared["evaluation_config"]["env"]
    assert _job_envs(per_agent) == _job_envs(shared)
    hw = per_agent["hardware"]
    assert hw["seats_share_block"] is True and hw["block"] == PRESETS[preset]
    assert hw["per_agent"]["State"]["setting"] == hw["per_agent"]["PPO-B"]["setting"] == preset


def test_mixed_tournament_job_with_equal_seats_uses_the_shared_keys(plan_of):
    """Three agents, two hardware: a job whose seats carry the same block gets the shared env of that block."""
    extra = {
        "matches": [
            {"id": "v2_vs_ppob", "a": "State", "b": "PPO-B", "condition": "exact", "primary": False},
            {"id": "v1_vs_ppob", "a": "State-v1", "b": "PPO-B", "condition": "exact", "primary": False},
        ]
    }
    plan, _ = plan_of(hardware={"per_agent": {"State": "v2", "State-v1": "v1", "PPO-B": "v1"}}, extra=extra)
    shared_v1, _ = plan_of(hardware="v1", extra=extra)
    envs = dict(zip([j["id"] for j in plan["jobs"]], _job_envs(plan)))
    reference = dict(zip([j["id"] for j in shared_v1["jobs"]], _job_envs(shared_v1)))
    assert "seat_hardware" in envs["v2_vs_ppob_leg0"]
    assert envs["v1_vs_ppob_leg0"] == reference["v1_vs_ppob_leg0"]
    assert envs["v1_vs_ppob_leg1"] == reference["v1_vs_ppob_leg1"]


def test_from_agent_entries(plan_of):
    plan, _ = plan_of(hardware={"per_agent": {"State": {"from_agent": "State"}, "PPO-B": {"from_agent": "PPO-B"}}})
    hw = plan["hardware"]
    assert hw["per_agent"]["State"] == {"setting": "from_agent State", "block": PRESETS["v2"]}
    assert hw["per_agent"]["PPO-B"] == {"setting": "from_agent PPO-B", "block": PRESETS["v1"]}
    preset, _ = plan_of(hardware=OWN)
    assert plan["evaluation_config"]["env"] == preset["evaluation_config"]["env"]
    # A source that does not play is hashed into the manifest like the players.
    plan, _ = plan_of(hardware={"per_agent": {"State": {"from_agent": "State-v1"}, "PPO-B": "v1"}})
    assert plan["agents"]["State-v1"]["config_sha256"] and plan["hardware"]["seats_share_block"] is True
    assert plan["hardware"]["mismatched_agents"] == ["State"]  # a v2-trained agent on its v1 seat: recorded


@pytest.mark.parametrize(
    "setting, message",
    [
        ({"per_agent": {"State": "v2", "PPO-B": "v1", "Nobody": "v1"}}, "not in the config"),
        ({"per_agent": {"State": "v2", "PPO_B": "v1"}}, "not in the config"),
        ({"per_agent": {"State": "v2"}}, r"no entry for the playing agent\(s\) \['PPO-B'\]"),
        ({"per_agent": {"State": "v3", "PPO-B": "v1"}}, "must be a preset"),
        ({"per_agent": {"State": {"agent": "State"}, "PPO-B": "v1"}}, "must be a preset"),
        ({"per_agent": {"State": {"from_agent": "Nobody"}, "PPO-B": "v1"}}, "from_agent must name"),
        ({"per_agent": {}}, "must map agent names"),
        ({"per_agent": {"State": "v2", "PPO-B": "v1"}, "from_agent": "State"}, "must map agent names"),
    ],
)
def test_invalid_per_agent_settings_are_refused(plan_of, setting, message):
    with pytest.raises(ValueError, match=message):
        plan_of(hardware=setting)


def test_existing_settings_keep_their_manifest_layout(plan_of):
    for setting in (None, "v1", "v2", {"from_agent": "State"}):
        plan, _ = plan_of(hardware=setting)
        assert "per_agent_hardware" not in plan["evaluation_config"]["env"]
        assert all("seat_hardware" not in job for job in plan["jobs"])
        assert seat_hardware_record(plan, plan["jobs"][0]) is None
        if setting is not None:
            assert set(plan["hardware"]) == {"setting", "block", "timing", "agents", "mismatched_agents"}


def test_real_ppob_training_config_is_the_v1_preset():
    """PPO-B's saved rl_games config (when the logs are mounted) trained on exactly the v1 block and timing."""
    path = Path("/workspace/klask_rl/logs/rl_games/klask/new_agent_from_pretrained/new_AM/params/agent.yaml")
    if not path.is_file():
        pytest.skip("PPO-B training config not mounted")
    block, timing = training_hardware(OmegaConf.load(path))
    assert block == PRESETS["v1"] and timing == {"sim_dt": 0.001, "decimation": 20}


def test_new_ownhw_configs():
    """The example own-hardware config equals its *_exact_vs_ppob.yaml except for the per-agent hardware."""
    configs = SCRIPTS / "tournament/configs"
    for name, agent in (("state_v2_1ms80eff_1m", "State-V2-1m-s80-eff-1M"),):
        exact = yaml.safe_load((configs / f"{name}_exact_vs_ppob.yaml").read_text())
        own = yaml.safe_load((configs / f"{name}_ownhw_vs_ppob.yaml").read_text())
        assert own.pop("hardware") == {"per_agent": {agent: "v2", "PPO-B": "v1"}}
        assert exact.pop("hardware") == "v2"
        own_match, exact_match = own.pop("matches"), exact.pop("matches")
        assert own == exact and own["games"] == 200
        assert [dict(m, id=None) for m in own_match] == [dict(m, id=None) for m in exact_match]
        assert own_match[0]["id"].endswith("_ownhw_vs_ppob")
