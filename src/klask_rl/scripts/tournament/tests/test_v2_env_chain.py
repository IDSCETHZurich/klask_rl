"""CPU smoke of the match environment's action chain on v1 and v2 hardware, without Isaac Sim.

Isaac Sim cannot start without a GPU, so this runs the REAL ``evaluation._make_eval_env`` (extracted from its source)
and the REAL wrappers (collision avoidance, opponent frame, actuator lag, velocity scaling) around a fake simulator
that records the peg velocity commands it receives. The env config comes from build_plan -> configuration_for_job,
i.e. exactly what a tournament worker hands to _make_eval_env.
"""

import ast
import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

gym = pytest.importorskip("gymnasium")
from omegaconf import OmegaConf  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[2]
TASK = SCRIPTS.parent / "source/klask_rl/klask_rl/tasks/manager_based/klask_rl"
sys.path.insert(0, str(SCRIPTS / "tournament"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_match_hardware import _project  # noqa: E402
from worker import configuration_for_job  # noqa: E402

LAG_MODULE = "klask_rl.tasks.manager_based.klask_rl.actuator_model.actuator_lag"
OBS_MODULE = "klask_rl.tasks.manager_based.klask_rl.env_cfg.klask_rl_observations_cfg"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMANDS = torch.tensor([[1.0, 0.5, 1.0, 0.5], [-1.0, -0.5, 0.0, -0.25]])  # |y| small: no boundary clipping


class FakeSim(gym.Env):
    """Stands in for gym.make(task, cfg): 4 peg velocity commands in, peg positions out (pegs mid-half)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = cfg.scene.num_envs
        self.device = "cpu"
        self.step_dt = cfg.decimation * cfg.sim.dt
        self.single_action_space = gym.spaces.Box(-1, 1, (4,))
        self.action_manager = SimpleNamespace(_terms={"pegs": SimpleNamespace(_scale=0.4)})
        self.received = []

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        policy = torch.zeros(self.num_envs, 20)
        policy[:, 0:2] = torch.tensor([0.0, -0.12])  # seat-0 peg, world frame
        policy[:, 4:6] = torch.tensor([0.0, 0.12])  # seat-1 peg
        return {"policy": policy, "opponent": policy.clone()}

    def reset(self, *args, **kwargs):
        return self._obs(), {}

    def step(self, actions):
        self.received.append(actions.clone())
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), torch.zeros(self.num_envs), done, done.clone(), {}


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


class RecordingActuatorModel(gym.Wrapper):
    """The learned v1 model needs its checkpoint; here it only records that it was (not) built."""

    built = []

    def __init__(self, env, device=None, model_file=None):
        super().__init__(env)
        RecordingActuatorModel.built.append(model_file)


@pytest.fixture
def make_eval_env(monkeypatch):
    params = _load(SCRIPTS.parent / "source/klask_rl/klask_rl/assets/robots/klask_params.py", "klask_params")
    monkeypatch.setitem(sys.modules, "klask_rl.assets.robots.klask_params", params)
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    training = _load(TASK / "wrappers/klask_rl_training_wrappers.py", "klask_training_wrappers")
    observation = _load(TASK / "wrappers/klask_rl_ovservation_wrappers.py", "klask_observation_wrappers")
    velocity = _load(TASK / "wrappers/velocity_scale_wrapper.py", "klask_velocity_scale_wrapper")
    monkeypatch.setitem(sys.modules, LAG_MODULE, _load(TASK / "actuator_model/actuator_lag.py", LAG_MODULE))
    obs_cfg = ModuleType(OBS_MODULE)
    obs_cfg._FastSACPlayerPolicyCfg = obs_cfg._FastSACOpponentPolicyCfg = lambda: None
    monkeypatch.setitem(sys.modules, OBS_MODULE, obs_cfg)
    env_cfg_utils = _load(SCRIPTS / "dreamer/env_cfg_utils.py", "klask_env_cfg_utils")

    sims = []

    def make(name, cfg):
        sims.append(FakeSim(cfg))
        return sims[-1]

    fake_gym = SimpleNamespace(
        spec=lambda name: SimpleNamespace(kwargs={"env_cfg_entry_point": FakeEnvCfg}),
        make=make,
    )
    source = (SCRIPTS / "dreamer/evaluation.py").read_text()
    function = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "_make_eval_env"
    )
    namespace = {
        "gym": fake_gym,
        "importlib": importlib,
        "OmegaConf": OmegaConf,
        "RenderCfg": lambda **kwargs: kwargs,
        "apply_camera_size_to_env_cfg": env_cfg_utils.apply_camera_size_to_env_cfg,
        "configure_domain_randomization": lambda env_cfg, dr: setattr(env_cfg, "dr", dr),
        "KLASK_PARAMS": params.KLASK_PARAMS,
        "ActuatorModelWrapper": RecordingActuatorModel,
        "KlaskRlCollisionAvoidanceWrapper": training.KlaskRlCollisionAvoidanceWrapper,
        "OpponentActionWrapper": observation.OpponentActionWrapper,
        "ObservationNoiseWrapper": observation.ObservationNoiseWrapper,
        "VelocityScaleWrapper": velocity.VelocityScaleWrapper,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "evaluation.py", "exec"), namespace)
    RecordingActuatorModel.built = []
    return namespace["_make_eval_env"], sims


def _job_env(tmp_path, monkeypatch, hardware):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    from run_tournament import build_plan

    plan, _ = build_plan(_project(tmp_path, 0, hardware=hardware))
    job = dict(plan["jobs"][0], device="cpu")
    return configuration_for_job(plan, job).env


def _chain(env):
    out = {}
    while hasattr(env, "env"):
        out[type(env).__name__] = env
        env = env.env
    return out


def _drive(make_eval_env, sims, env_cfg, commands=COMMANDS, steps=5):
    env, opponent = make_eval_env(env_cfg, 2, tournament=True)
    assert opponent is None
    env.reset()
    for _ in range(steps):
        env.step(commands.clone())
    return env, torch.stack(sims[-1].received)


def _reference(commands, vmax, amax, dt, dead=None, tau=None):
    """Rate limit + scale (VelocityScaleWrapper), then the lag filter (actuator_lag.py docstring formula)."""
    prev_v = torch.zeros_like(commands[0])
    prev_u = torch.zeros_like(commands[0])
    lag_v = torch.zeros_like(commands[0])
    out = []
    for u in commands:
        v = prev_v + torch.clamp(u * vmax - prev_v, -amax * dt, amax * dt)
        prev_v = v
        if dead is not None:
            f, alpha = dead / dt, 1 - math.exp(-dt / tau)
            lag_v = lag_v + alpha * ((1 - f) * v + f * prev_u - lag_v)
            prev_u = v
            v = lag_v
        out.append(v.clone())
    return torch.stack(out)


def test_v2_match_env_drives_both_seats_like_v2_training(tmp_path, monkeypatch, make_eval_env):
    make, sims = make_eval_env
    env_cfg = _job_env(tmp_path, monkeypatch, "v2")
    env, received = _drive(make, sims, env_cfg)
    chain = _chain(env)
    assert list(chain) == [
        "VelocityScaleWrapper",
        "ActuatorLagWrapper",
        "OpponentActionWrapper",
        "KlaskRlCollisionAvoidanceWrapper",
    ]
    assert RecordingActuatorModel.built == []  # the learned v1 model is off
    scale, lag = chain["VelocityScaleWrapper"], chain["ActuatorLagWrapper"].lag
    assert (scale.max_velocity, scale.max_acceleration) == (1.4, 35.0)
    assert math.isclose(scale._max_dv, 0.7)  # 35 m/s^2 x 20 ms
    assert math.isclose(lag.f, 0.25) and math.isclose(lag.alpha, 1 - math.exp(-2))  # 5 ms / 10 ms at 20 ms
    assert chain["KlaskRlCollisionAvoidanceWrapper"].MAX_VEL == 1.4
    assert sims[-1].step_dt == pytest.approx(0.02) and sims[-1].action_manager._terms["pegs"]._scale == 1.0
    expected = _reference(COMMANDS.expand(5, 2, 4), 1.4, 35.0, 0.02, dead=0.005, tau=0.01)
    expected[..., 2:] *= -1  # seat 1's commands are in its own (rotated) frame
    torch.testing.assert_close(received, expected)
    assert received[1:, 0, 0].max() > 1.2  # reaches v2 speeds (v1 caps at 0.4 m/s)


def test_provider_hardware_chain_is_the_v1_chain(tmp_path, monkeypatch, make_eval_env):
    make, sims = make_eval_env
    env_cfg = _job_env(tmp_path, monkeypatch, None)
    env, received = _drive(make, sims, env_cfg)
    assert list(_chain(env)) == [
        "VelocityScaleWrapper",
        "RecordingActuatorModel",
        "OpponentActionWrapper",
        "KlaskRlCollisionAvoidanceWrapper",
    ]
    assert len(RecordingActuatorModel.built) == 1 and RecordingActuatorModel.built[0].endswith("train_seed0.pt")
    expected = _reference(COMMANDS.expand(5, 2, 4), 0.4, 10.0, 0.02)
    expected[..., 2:] *= -1
    torch.testing.assert_close(received, expected)


def test_both_actuators_enabled_is_refused_like_training(tmp_path, monkeypatch, make_eval_env):
    make, _ = make_eval_env
    env_cfg = _job_env(tmp_path, monkeypatch, "v2")
    env_cfg.actuator_model.enable = True
    with pytest.raises(ValueError, match="alternatives"):
        make(env_cfg, 2, tournament=True)


@pytest.mark.parametrize("hardware, vmax", [("v2", 1.4), (None, 0.4)])
def test_collision_avoidance_ramp_uses_the_match_speed_limit(tmp_path, monkeypatch, make_eval_env, hardware, vmax):
    """Seat-0 peg 9 cm from its back edge: a full-speed retreat is capped at vmax / 9 cm x (9 cm - 1.8 r)."""
    make, sims = make_eval_env
    retreat = torch.tensor([[0.0, -1.0, 0.0, 0.0]] * 2)
    _, received = _drive(make, sims, _job_env(tmp_path, monkeypatch, hardware), retreat)
    cap = vmax / 0.09 * (0.09 - 1.8 * 0.0075)
    assert received[-1, 0, 1].item() == pytest.approx(-cap)
