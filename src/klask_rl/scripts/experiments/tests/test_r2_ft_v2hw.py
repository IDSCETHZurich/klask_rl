"""CPU tests of the v2-hardware fine-tune of r2_pfsp/exact_eff_ft (branch r2-ft-v2hw, 2026-10-14): the hardware-speed
schedule, the training env chain that applies it (the REAL train_dreamer._make_env around a fake simulator), the
fine-tune start (full trainer state, fresh buffer, 1-member PFSP pool), the resume path, the composed config versus
exact_eff_ft's saved config, and the launcher. No GPU, no Isaac Sim. In the Isaac Lab container, from a code copy:

    cd <code directory> && CUDA_VISIBLE_DEVICES="" /workspace/isaaclab/_isaac_sim/python.sh -m pytest -q -s \\
        scripts/experiments/tests/test_r2_ft_v2hw.py

Tests that need the composed Hydra config (scripts/dreamer/r2dreamer/configs, present in the container copies) or the
real exact_eff_ft files (read only: memory-mapped checkpoint, saved .hydra/config.yaml) are skipped where missing.
"""

import ast
import importlib
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

gym = pytest.importorskip("gymnasium")
from omegaconf import OmegaConf  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]  # .../src/klask_rl or the /opt code copy
SCRIPTS = ROOT / "scripts"
DREAMER = SCRIPTS / "dreamer"
TASK = ROOT / "source/klask_rl/klask_rl/tasks/manager_based/klask_rl"
CONFIG_NAME = "config-sprite-r2-pfsp-eff-v2hw-ft"
CONFIG = DREAMER / f"{CONFIG_NAME}.yaml"
LAUNCHER = SCRIPTS / "experiments/run_scripts/r2_eff_ft_v2hw_run.sh"
SOURCE_RUN = ROOT / "logs/dreamer/training/r2_pfsp/exact_eff_ft"  # read only
START, RAMP_END, END = 10_404_928, 11_404_928, 13_404_928
sys.path.insert(0, str(DREAMER))

from pfsp_pool import LATEST, PFSPPool  # noqa: E402
from run_init import (  # noqa: E402
    StartPlan,
    buffer_dir_to_restore,
    check_finetune_checkpoint,
    checkpoint_step,
    load_trainer_state,
    restored_entry,
    start_pfsp_pool,
    start_plan,
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HW = _load(TASK / "wrappers/hardware_schedule_wrapper.py", "klask_hardware_schedule_wrapper")
LAG = _load(TASK / "actuator_model/actuator_lag.py", "klask_actuator_lag")


def _launcher_overrides(gpu=2, logdir="/workspace/klask_rl/logs/dreamer/training/r2_pfsp/eff_ft_v2hw_ramp"):
    """The Hydra overrides r2_eff_ft_v2hw_run.sh passes (default START / RAMP / RUN)."""
    return [
        "wandb_name=r2_pfsp_eff_ft_v2hw_ramp", f"logdir={logdir}", f"device=cuda:{gpu}", f"sim_device=cuda:{gpu}",
        f"train_devices=[cuda:{gpu}]", "micro_batch_size=10", f"env.steps={END}", "opponent_separation.enabled=true",
        "model.imag_opponent_recompute=true", f"finetune_from.checkpoint={logdir}/init/exact_eff_ft_latest.pt",
        f"finetune_from.expect_step={START}",
        "finetune_from.source=/workspace/klask_rl/logs/dreamer/training/r2_pfsp/exact_eff_ft/latest.pt",
        f"env.hardware_schedule.steps=[{START},{RAMP_END}]",
    ]


def _compose(overrides):
    hydra = pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if not (DREAMER / "r2dreamer/configs/model/size50M.yaml").is_file():
        pytest.skip("scripts/dreamer/r2dreamer/configs missing (run from a container code copy)")
    OmegaConf.register_new_resolver("script_dir", lambda: str(DREAMER), use_cache=True, replace=True)
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(DREAMER), version_base=None):
            return compose(config_name=CONFIG_NAME, overrides=list(overrides))
    finally:
        GlobalHydra.instance().clear()
        del hydra


# =====================================================================================================================
# 1. The schedule
# =====================================================================================================================


def _config_schedule():
    env = OmegaConf.load(CONFIG).env
    block = OmegaConf.to_container(env.hardware_schedule, resolve=True)
    return HW.HardwareSchedule.from_config(block, env.max_velocity, env.max_acceleration), env


def test_config_block_is_the_specified_ramp():
    schedule, env = _config_schedule()
    assert (schedule.start_step, schedule.end_step) == (START, RAMP_END)  # YAML 10_404_928 parses as an int
    assert schedule.max_velocity == (0.4, 1.4) and schedule.max_acceleration == (10.0, 35.0)
    assert env.max_velocity == 1.4 and env.max_acceleration == 35.0  # the hardware the run ends on (tournaments)
    assert env.actuator_model.enable is False and env.actuator_lag.enable is True
    assert (env.actuator_lag.dead_time_s, env.actuator_lag.tau_s) == (0.005, 0.010)
    assert env.collision_avoidance.enable is True and env.steps == END


@pytest.mark.parametrize(
    "step, v, a",
    [
        (0, 0.4, 10.0),  # before the ramp: start values
        (START - 16, 0.4, 10.0),
        (START, 0.4, 10.0),  # the fine-tune's first step
        (START + 16, 0.4 + 16e-6, 10.0 + 25 * 16e-6),
        (START + 250_000, 0.65, 16.25),
        (START + 500_000, 0.9, 22.5),  # middle
        (START + 750_000, 1.15, 28.75),
        (RAMP_END, 1.4, 35.0),  # end of the ramp
        (RAMP_END + 16, 1.4, 35.0),  # held
        (END, 1.4, 35.0),  # last step of the run
        (20_000_000, 1.4, 35.0),
    ],
)
def test_schedule_values(step, v, a):
    schedule, _ = _config_schedule()
    got_v, got_a = schedule.values(step)
    print(f"step {step:>10,}: max_velocity {got_v:.6f} m/s, max_acceleration {got_a:.6f} m/s^2")
    assert got_v == pytest.approx(v, abs=1e-12) and got_a == pytest.approx(a, abs=1e-9)
    # a / v = 25 1/s at both ends, so the per-step rate cap in command units is 0.5 throughout.
    assert got_a * 0.02 / got_v == pytest.approx(0.5)


def test_schedule_validation():
    ok = {"enable": True, "steps": [10, 20], "max_velocity": [0.4, 1.4], "max_acceleration": [10.0, 35.0]}
    assert HW.HardwareSchedule.from_config(None, 1.4, 35.0) is None
    assert HW.HardwareSchedule.from_config({**ok, "enable": False}, 0.4, 10.0) is None
    with pytest.raises(ValueError, match="ends at 1.4, but env.max_velocity is 0.4"):
        HW.HardwareSchedule.from_config(ok, 0.4, 35.0)
    with pytest.raises(ValueError, match="max_acceleration"):
        HW.HardwareSchedule.from_config(ok, 1.4, 10.0)
    with pytest.raises(ValueError, match="needs env.max_acceleration"):
        HW.HardwareSchedule.from_config(ok, 1.4, None)
    with pytest.raises(ValueError, match="end >= start"):
        HW.HardwareSchedule([20, 10], [0.4, 1.4], [10, 35])
    with pytest.raises(ValueError, match="pair"):
        HW.HardwareSchedule([10, 20], [1.4], [10, 35])
    step_switch = HW.HardwareSchedule([100, 100], [0.4, 1.4], [10, 35])
    assert step_switch.values(99) == (0.4, 10.0) and step_switch.values(100) == (1.4, 35.0)


# =====================================================================================================================
# 2. The REAL training env chain (train_dreamer._make_env) around a fake simulator
# =====================================================================================================================

PARAMS = _load(ROOT / "source/klask_rl/klask_rl/assets/robots/klask_params.py", "klask_params_r2ftv2hw").KLASK_PARAMS
DECEL, MIN_CLEAR = PARAMS["collision_avoidance_decel_distance"], PARAMS["peg_radius"] * PARAMS["collision_avoidance_min_clearance_factor"]
X_EDGE = PARAMS["joint_x_pos_limit"]
PEG1 = (0.0, -0.115)  # mid-board in x (no clipping), 9.5 cm from both y edges (soft zone)
PEG2 = (X_EDGE[0] + 0.036, 0.115)  # 3.6 cm from the -x edge: the world-frame -x command is clipped


def _ramp(d):
    """Collision avoidance: speed limit at distance d from an edge, in units of its MAX_VEL."""
    return (d - MIN_CLEAR) / DECEL


class FakeSim(gym.Env):
    """Stands in for gym.make(task, cfg=env_cfg): peg velocity commands (m/s, world frame) in, fixed peg positions out."""

    def __init__(self, cfg=None, render_mode=None, **kwargs):
        self.cfg = cfg
        self.num_envs = cfg.scene.num_envs
        self.device = "cpu"
        self.step_dt = cfg.decimation * cfg.sim.dt
        self.single_action_space = gym.spaces.Box(-1, 1, (4,))
        self.action_manager = SimpleNamespace(_terms={"pegs": SimpleNamespace(_scale=0.4)})
        self.received = []
        FakeSim.last = self

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        policy = torch.zeros(self.num_envs, 20)
        policy[:, 0:2] = torch.tensor(PEG1)
        policy[:, 4:6] = torch.tensor(PEG2)
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
        self.scene = SimpleNamespace(num_envs=2)
        self.observations = SimpleNamespace()
        self.events = SimpleNamespace(reset_ball_position=SimpleNamespace(params={"pose_range": {}}))
        self.ball_reset_position_x = self.ball_reset_position_y = None
        self.terminations = SimpleNamespace(time_out=1, goal_scored=1, goal_conceded=1, player_in_goal=1, opponent_in_goal=1)
        self.decimation, self.seed, self.episode_length_s = 4, 0, 1.0


GYM_ID = "FakeKlaskR2FtV2hw-v0"
if GYM_ID not in gym.registry:
    gym.register(GYM_ID, entry_point=FakeSim, kwargs={"env_cfg_entry_point": FakeEnvCfg}, disable_env_checker=True,
                 order_enforce=False)


class Passthrough(gym.Wrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env)


class RecordingActuatorModel(gym.Wrapper):
    """The learned v1 model: here it only records that it was built (it must not be)."""

    built = []

    def __init__(self, env, **kwargs):
        super().__init__(env)
        RecordingActuatorModel.built.append(kwargs.get("model_file"))


@pytest.fixture
def make_env(monkeypatch):
    """train_dreamer._make_env, extracted from its source and run with the real hardware wrappers."""
    params = _load(ROOT / "source/klask_rl/klask_rl/assets/robots/klask_params.py", "klask_params")
    monkeypatch.setitem(sys.modules, "klask_rl.assets.robots.klask_params", params)
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = lambda *a, **k: SimpleNamespace(args=a, kwargs=k)
    managers.TerminationTermCfg = lambda **k: SimpleNamespace(**k)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    training = _load(TASK / "wrappers/klask_rl_training_wrappers.py", "klask_training_wrappers_r2ftv2hw")
    observation = _load(TASK / "wrappers/klask_rl_ovservation_wrappers.py", "klask_observation_wrappers_r2ftv2hw")
    velocity = _load(TASK / "wrappers/velocity_scale_wrapper.py", "klask_velocity_scale_wrapper_r2ftv2hw")
    pkg = "klask_rl.tasks.manager_based.klask_rl"
    monkeypatch.setitem(sys.modules, f"{pkg}.actuator_model.actuator_lag", LAG)
    utils = ModuleType(f"{pkg}.utils_manager_based")
    utils.ball_stalled = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, f"{pkg}.utils_manager_based", utils)
    rewards = ModuleType(f"{pkg}.env_cfg.klask_rl_rewards_cfg")
    rewards._opponent_reward_terms = {}
    monkeypatch.setitem(sys.modules, f"{pkg}.env_cfg.klask_rl_rewards_cfg", rewards)
    opp_reward = ModuleType(f"{pkg}.wrappers.klask_rl_training_wrappers")
    opp_reward.OpponentRewardWrapper = Passthrough
    monkeypatch.setitem(sys.modules, f"{pkg}.wrappers.klask_rl_training_wrappers", opp_reward)
    RecordingActuatorModel.built.clear()

    source = (DREAMER / "train_dreamer.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_make_env")
    namespace = {
        "OmegaConf": OmegaConf, "RenderCfg": lambda **k: k, "apply_camera_size_to_env_cfg": lambda cfg, size: None,
        "configure_domain_randomization": lambda cfg, dr: None, "ActuatorModelWrapper": RecordingActuatorModel,
        "KlaskRlCollisionAvoidanceWrapper": training.KlaskRlCollisionAvoidanceWrapper,
        "OpponentActionWrapper": observation.OpponentActionWrapper, "VelocityScaleWrapper": velocity.VelocityScaleWrapper,
        "HardwareSchedule": HW.HardwareSchedule, "HardwareScheduleWrapper": HW.HardwareScheduleWrapper,
        "InitializationWrapper": training.InitializationWrapper, "CurriculumWrapper": Passthrough,
        "DreamerSelfPlayWrapper": Passthrough, "KlaskRlRandomOpponentWrapper": Passthrough,
        "EpisodeTagTracker": Passthrough, "EventFlagWrapper": Passthrough, "EpisodeMetricsWrapper": Passthrough,
        "RewardWeightLogWrapper": Passthrough, "_make_reward_term_bool_checker": lambda term: None,
        "IsaacLabVecEnv": lambda env, simulation_app=None: env, "simulation_app": None,
        "_self_play_wrapper": None, "_tag_tracker": None, "_hw_schedule_wrapper": None,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "train_dreamer._make_env", "exec"), namespace)

    def build(cfg):
        env = namespace["_make_env"](
            cfg.env, GYM_ID, None, self_play=True, self_play_config={}, prioritized_cfg=cfg.buffer.prioritized,
            trajectory_mirroring=True,
        )
        return env, namespace["_hw_schedule_wrapper"], FakeSim.last

    return build


def _chain(env):
    out = []
    while isinstance(env, gym.Wrapper):
        out.append(env)
        env = env.env
    return out


def _find(env, name):
    return [w for w in _chain(env) if type(w).__name__ == name]


def _expected_commands(values, dt=0.02):
    """Independent re-implementation: +1 command -> rate cap -> dead time + first-order lag, per step (m/s)."""
    f, alpha = 0.005 / dt, 1.0 - math.exp(-dt / 0.010)
    v_prev = lag_u = lag_v = 0.0
    out = []
    for v_max, a_max in values:
        v_cmd = v_prev + max(-a_max * dt, min(a_max * dt, v_max - v_prev))
        v_prev = v_cmd
        lag_v = lag_v + alpha * ((1 - f) * v_cmd + f * lag_u - lag_v)
        lag_u = v_cmd
        out.append(lag_v)
    return out


def test_env_chain_receives_the_scheduled_hardware(make_env):
    short_end = START + 16 * 50  # 50 vector steps of ramp (the launcher's override path, shortened)
    cfg = _compose(_launcher_overrides()[:-1] + [f"env.hardware_schedule.steps=[{START},{short_end}]"])
    env, hw, sim = make_env(cfg)
    names = [type(w).__name__ for w in _chain(env)]
    print("chain (outer -> inner):", " -> ".join(names))
    i = names.index
    # The schedule sits directly outside the velocity scaling; v2 actuator: lag on, learned v1 model never built.
    assert i("HardwareScheduleWrapper") + 1 == i("VelocityScaleWrapper") < i("ActuatorLagWrapper") < i(
        "OpponentActionWrapper") < i("KlaskRlCollisionAvoidanceWrapper")
    assert "RecordingActuatorModel" not in names and RecordingActuatorModel.built == []
    lag = _find(env, "ActuatorLagWrapper")[0].lag
    assert lag.f == pytest.approx(0.25) and lag.alpha == pytest.approx(1 - math.exp(-2))
    vs, ca = _find(env, "VelocityScaleWrapper")[0], _find(env, "KlaskRlCollisionAvoidanceWrapper")[0]
    assert hw is _find(env, "HardwareScheduleWrapper")[0]

    # As train_dreamer: positioned at the start step after the checkpoint restore, then bound to the trainer's step.
    hw.set_step(START)
    assert (vs.max_velocity, vs.max_acceleration, ca.MAX_VEL) == (0.4, 10.0, 0.4)  # v1 speed at the start
    trainer = {"step": START - 16 * 10}  # 10 steps before the ramp, then the ramp, then held
    hw.bind_step(lambda: trainer["step"])
    env.reset()
    steps, values = [], []
    for _ in range(80):
        trainer["step"] += 16  # the trainer counts the step before stepping the env
        env.step(torch.ones(sim.num_envs, 4))
        v, a = hw.schedule.values(trainer["step"])
        steps.append(trainer["step"])
        values.append((v, a))
        # The wrappers hold the scheduled values at this step.
        assert vs.max_velocity == pytest.approx(v) and vs.max_acceleration == pytest.approx(a)
        assert vs._max_dv == pytest.approx(a * 0.02) and ca.MAX_VEL == pytest.approx(v)
        assert sim.unwrapped._action_max_dv_norm == pytest.approx(0.5)
    expected = torch.tensor(_expected_commands(values))
    got = torch.stack(sim.received)[:, 0, :]  # env 0, (steps, 4) world-frame m/s
    vmax = torch.tensor([v for v, _ in values])
    torch.testing.assert_close(got[:, 0], expected, atol=1e-6, rtol=0)  # peg 1 x: free
    torch.testing.assert_close(got[:, 1], torch.minimum(expected, _ramp(0.095) * vmax), atol=1e-6, rtol=0)  # y soft zone
    torch.testing.assert_close(got[:, 2], torch.maximum(-expected, -_ramp(0.036) * vmax), atol=1e-6, rtol=0)  # near -x edge
    torch.testing.assert_close(got[:, 3], torch.maximum(-expected, -_ramp(0.095) * vmax), atol=1e-6, rtol=0)
    for k in (0, 9, 10, 35, 59, 60, 79):
        print(f"step {steps[k]:>10,}: v_max {values[k][0]:.3f} a_max {values[k][1]:.2f} | sim got "
              f"{[round(x, 4) for x in got[k].tolist()]} (CA speed {values[k][0]:.3f})")
    assert values[9] == (0.4, 10.0) and values[60] == (1.4, 35.0) and values[-1] == (1.4, 35.0)
    assert got[-1, 0].item() == pytest.approx(1.4, abs=1e-3)  # full v2 speed reached on the free axis
    assert got[-1, 2].item() == pytest.approx(-_ramp(0.036) * 1.4, abs=1e-6)  # wall ramp at the v2 speed


def test_restart_continues_the_schedule_at_the_checkpoint_step(make_env):
    """A crash-restart rebuilds the chain at the construction values and is then positioned at the restored trainer
    step: the hardware continues mid-ramp, it does not start over at v1."""
    cfg = _compose(_launcher_overrides())
    env, hw, sim = make_env(cfg)
    vs, ca = _find(env, "VelocityScaleWrapper")[0], _find(env, "KlaskRlCollisionAvoidanceWrapper")[0]
    resumed = START + 500_000  # step of the run's own latest.pt
    hw.set_step(resumed)
    trainer = {"step": resumed}
    hw.bind_step(lambda: trainer["step"])
    assert (vs.max_velocity, vs.max_acceleration, ca.MAX_VEL) == pytest.approx((0.9, 22.5, 0.9))
    env.reset()
    trainer["step"] += 16
    env.step(torch.ones(sim.num_envs, 4))
    assert vs.max_velocity == pytest.approx(0.9 + 16e-6) and hw.current["step"] == resumed + 16
    hw.set_step(0)  # the own counter is ignored once bound to the trainer
    assert hw.step_count == resumed + 16


def test_constant_hardware_without_a_schedule(make_env):
    """exact_eff_ft's own config (no hardware_schedule): no schedule wrapper, values as before (regression)."""
    saved = SOURCE_RUN / ".hydra/config.yaml"
    if not saved.is_file():
        pytest.skip("exact_eff_ft saved config not available")
    cfg = OmegaConf.load(saved)
    env, hw, sim = make_env(cfg)
    assert hw is None and not _find(env, "HardwareScheduleWrapper")
    vs = _find(env, "VelocityScaleWrapper")[0]
    assert (vs.max_velocity, vs.max_acceleration) == (0.4, 10.0)
    assert RecordingActuatorModel.built and not _find(env, "ActuatorLagWrapper")  # v1: the learned model


# =====================================================================================================================
# 3. Start plan: fine-tune vs resume
# =====================================================================================================================


def _copy_layout(tmp_path):
    logdir = tmp_path / "eff_ft_v2hw_ramp"
    (logdir / "init").mkdir(parents=True)
    copy = logdir / "init/exact_eff_ft_latest.pt"
    copy.write_bytes(b"x")
    return logdir, copy


def test_start_plan_finetune_then_resume_from_own_latest(tmp_path):
    logdir, copy = _copy_layout(tmp_path)
    plan = start_plan(logdir, copy)
    assert plan == StartPlan("finetune", copy, restore_buffer=False, restore_pool=False, restore_scores=False)
    # A saved buffer next to it (e.g. left by something else) is still not loaded by a fine-tune start.
    (logdir / "replay_buffer").mkdir()
    assert buffer_dir_to_restore(plan, logdir) is None
    # After the first save the run resumes from its OWN latest.pt (+ its buffer), though finetune_from is still set.
    (logdir / "latest.pt").write_bytes(b"y")
    (logdir / "replay_buffer.latest").write_text("replay_buffer")
    plan = start_plan(logdir, copy)
    assert plan.mode == "resume" and plan.checkpoint == logdir / "latest.pt"
    assert plan.restore_buffer and plan.restore_pool and plan.restore_scores
    assert buffer_dir_to_restore(plan, logdir) == logdir / "replay_buffer"
    (logdir / "replay_buffer_5000000").mkdir()
    (logdir / "replay_buffer.latest").write_text("replay_buffer_5000000")
    assert buffer_dir_to_restore(plan, logdir) == logdir / "replay_buffer_5000000"


def test_start_plan_refusals(tmp_path):
    logdir, copy = _copy_layout(tmp_path)
    outside = tmp_path / "exact_eff_ft_latest.pt"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="copy inside the logdir"):
        start_plan(logdir, outside)  # e.g. the source run's own latest.pt
    with pytest.raises(FileNotFoundError):
        start_plan(logdir, logdir / "init/missing.pt")
    with pytest.raises(ValueError, match="alternatives"):
        start_plan(logdir, copy, init_checkpoint=copy)
    assert start_plan(logdir).mode == "scratch"
    assert start_plan(logdir, init_checkpoint=outside).mode == "init"
    with pytest.raises(ValueError, match="at step 5, expected 10404928"):
        check_finetune_checkpoint({"agent_state_dict": {}, "optims_state_dict": {}, "step": 5}, START)
    with pytest.raises(ValueError, match="missing"):
        check_finetune_checkpoint({"agent_state_dict": {}, "step": START}, START)
    assert check_finetune_checkpoint({"agent_state_dict": {}, "optims_state_dict": {}, "step": START}, START) == START


def test_restored_entries_come_from_a_resume_only():
    ck = {"pfsp_state": {"pool": 1}, "selfplay_score_buffer": [1.0]}
    ft = StartPlan("finetune", Path("x"), False, False, False)
    rs = StartPlan("resume", Path("y"), True, True, True)
    assert restored_entry(ft, ck, "pfsp_state") is None and restored_entry(ft, ck, "selfplay_score_buffer") is None
    assert restored_entry(rs, ck, "pfsp_state") == {"pool": 1} and restored_entry(rs, ck, "selfplay_score_buffer") == [1.0]
    with pytest.raises(KeyError):
        restored_entry(rs, ck, "agent_state_dict")


# =====================================================================================================================
# 4. Fine-tune start: full trainer state, fresh pool
# =====================================================================================================================


class FakeAgent(torch.nn.Module):
    """The parts of Dreamer the trainer state touches: modules, an optimizer (+ the scheduler's reference to it),
    LR scheduler, GradScaler, slow-critic counter, inference-copy sync."""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder, self.rssm, self.actor = torch.nn.Linear(3, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)
        self._optimizer = torch.optim.Adam(self.parameters(), lr=4e-5)
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lambda s: min(1.0, (s + 1) / 1000))
        self._scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** (13 + 3 * seed))
        self._slow_value_updates = 0
        self.synced = 0

    def _sync_inference_copies(self):
        self.synced += 1

    def train_steps(self, n):
        for _ in range(n):
            loss = self.actor(self.rssm(self.encoder(torch.randn(5, 3)))).pow(2).mean()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            self._slow_value_updates += 1


def _tools():
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "third_party/r2dreamer"))  # worktree layout
    sys.path.insert(0, str(DREAMER / "r2dreamer"))  # container copies
    return importlib.import_module("tools")


def _trainer_checkpoint(agent, tools, step):
    """Same items as train_dreamer._save_checkpoint (self-play part included)."""
    return {
        "agent_state_dict": agent.state_dict(), "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "step": step, "curriculum_step": 12_252_928, "init_step": 12_252_928,
        "scheduler_state_dict": agent._scheduler.state_dict(), "scaler_state_dict": agent._scaler.state_dict(),
        "slow_value_updates": agent._slow_value_updates, "selfplay_score_buffer": [1.0, 0.0, -1.0],
        "pfsp_state": {"pool": {"members": [{"file": "member_000_step0.pt"}] * 38}, "active": LATEST, "last_snapshot": 10_230_000},
    }


def test_finetune_start_restores_weights_and_optimizer(tmp_path):
    tools = _tools()
    source = FakeAgent(seed=0)
    source.train_steps(7)
    path = tmp_path / "exact_eff_ft_latest.pt"
    torch.save(_trainer_checkpoint(source, tools, START), path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert check_finetune_checkpoint(ck, START) == START

    agent = FakeAgent(seed=1)  # fresh weights, empty optimizer state
    assert not torch.equal(agent.actor.weight, source.actor.weight)
    restored = load_trainer_state(agent, ck, tools.recursively_load_optim_state_dict)
    assert restored["step"] == START and restored["curriculum_step"] == 12_252_928
    assert restored["missing"] == [] and restored["unexpected"] == []
    for (name, a), (_, b) in zip(agent.state_dict().items(), source.state_dict().items()):
        assert torch.equal(a, b), name
    for pa, pb in zip(agent._optimizer.state.values(), source._optimizer.state.values()):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(pa[key], pb[key]), key
    assert agent._scheduler.last_epoch == source._scheduler.last_epoch == 7
    assert agent._scheduler.get_last_lr() == source._scheduler.get_last_lr()
    assert agent._scaler.get_scale() == source._scaler.get_scale() and agent._slow_value_updates == 7
    # The next update is the one the source would have made (optimizer moments and LR warmup continue).
    torch.manual_seed(5)
    source.train_steps(1)
    torch.manual_seed(5)
    agent.train_steps(1)
    for a, b in zip(agent.parameters(), source.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_finetune_start_has_an_empty_buffer_and_a_one_member_pool(tmp_path):
    """train_dreamer's fine-tune path: no buffer to restore, the source's 38-member pool and score window ignored,
    a fresh pool whose only member is the start weights at the start step."""
    tools = _tools()
    source = FakeAgent(seed=0)
    ck = _trainer_checkpoint(source, tools, START)
    logdir, copy = _copy_layout(tmp_path)
    plan = start_plan(logdir, copy)
    assert buffer_dir_to_restore(plan, logdir) is None
    assert restored_entry(plan, ck, "pfsp_state") is None and restored_entry(plan, ck, "selfplay_score_buffer") is None
    pool = PFSPPool(logdir / "pfsp", p_latest=0.5, power=2.0, floor=0.02, prior_games=2.0, seed=0,
                    score_mode="decisive", selection="thompson")
    snapshot = {n: getattr(source, n).state_dict() for n in ("encoder", "rssm", "actor")}
    state, restored = start_pfsp_pool(pool, restored_entry(plan, ck, "pfsp_state"), True, START, lambda: snapshot)
    assert not restored and state == {"active": 0, "last_snapshot": START}
    assert len(pool.members) == 1 and pool.members[0]["step"] == START
    assert sorted(p.name for p in (logdir / "pfsp").iterdir()) == ["member_000_step10404928.pt", "pool.json"]
    member = pool.load(0)
    assert all(torch.equal(member["actor"][k], v) for k, v in source.actor.state_dict().items())
    picks = {pool.choose() for _ in range(200)}
    assert picks == {LATEST, 0}  # the start weights or the learner itself

    # Resume of the same run later: its own pool comes back from its checkpoint.
    pool.add(START + 260_000, snapshot)
    saved = {"pool": pool.state_dict(), "active": 1, "last_snapshot": START + 260_000}
    pool2 = PFSPPool(logdir / "pfsp", seed=0, score_mode="decisive", selection="thompson")
    resumed = StartPlan("resume", logdir / "latest.pt", True, True, True)
    state, restored = start_pfsp_pool(pool2, restored_entry(resumed, {"pfsp_state": saved}, "pfsp_state"), True,
                                      START + 300_000, lambda: snapshot)
    assert restored and state == {"active": 1, "last_snapshot": START + 260_000} and len(pool2.members) == 2


def test_real_source_checkpoint(tmp_path):
    """exact_eff_ft/latest.pt (memory-mapped, read only): a full trainer checkpoint at the configured start step; its
    38-member pool and score window are NOT used by the fine-tune start."""
    src = SOURCE_RUN / "latest.pt"
    if not src.is_file():
        pytest.skip("exact_eff_ft/latest.pt not available")
    before = (src.stat().st_size, src.stat().st_mtime_ns)
    ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    assert check_finetune_checkpoint(ck, START) == START == OmegaConf.load(CONFIG).finetune_from.expect_step
    assert checkpoint_step(src) == START  # the launcher's check
    for key in ("scheduler_state_dict", "scaler_state_dict", "slow_value_updates", "curriculum_step", "init_step"):
        assert key in ck, key
    assert set(ck["optims_state_dict"]) == {"_optimizer", "_scheduler.optimizer"}
    assert len(ck["pfsp_state"]["pool"]["members"]) == 38 and len(ck["selfplay_score_buffer"]) > 0
    logdir, copy = _copy_layout(tmp_path)
    plan = start_plan(logdir, copy)
    pool = PFSPPool(logdir / "pfsp", seed=0, score_mode="decisive", selection="thompson")
    tiny = {"encoder": {}, "rssm": {}, "actor": {}}  # the real snapshot is the learner's weights (124 MB)
    state, restored = start_pfsp_pool(pool, restored_entry(plan, ck, "pfsp_state"), True, ck["step"], lambda: tiny)
    assert not restored and len(pool.members) == 1 and state["active"] == 0
    print(f"source: step {ck['step']}, curriculum/init step {ck['curriculum_step']}/{ck['init_step']}, "
          f"scheduler last_epoch {ck['scheduler_state_dict']['last_epoch']}, scaler {ck['scaler_state_dict']['scale']}, "
          f"source pool {len(ck['pfsp_state']['pool']['members'])} members -> fine-tune pool {len(pool.members)}")
    del ck
    assert (src.stat().st_size, src.stat().st_mtime_ns) == before


# =====================================================================================================================
# 5. The composed config versus exact_eff_ft's saved config
# =====================================================================================================================


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


INTENDED = {
    "wandb_name", "logdir", "finetune_from.checkpoint", "finetune_from.expect_step", "finetune_from.source", "env.steps",
    "env.max_velocity", "env.max_acceleration", "env.actuator_model.enable", "env.actuator_lag.enable",
    "env.actuator_lag.dead_time_s", "env.actuator_lag.tau_s", "env.hardware_schedule.enable",
    "env.hardware_schedule.steps", "env.hardware_schedule.max_velocity", "env.hardware_schedule.max_acceleration",
}


# Model options added after exact_eff_ft ran (absent from its saved config), at their defaults: no behaviour change.
NEW_DEFAULTS = {"model.actor_grad": "reinforce", "model.actor_grad_mix": 0.5, "model.actor_grad_checkpoint": True,
                "model.imag_opponent_world": "separate", "model.amp_dtype": "float16"}


def test_composed_config_differs_from_exact_eff_ft_only_in_the_intended_keys():
    saved_path = SOURCE_RUN / ".hydra/config.yaml"
    if not saved_path.is_file():
        pytest.skip("exact_eff_ft saved config not available")
    saved = _flat(OmegaConf.to_container(OmegaConf.load(saved_path), resolve=False))
    for gpu in (2, 0):  # exact_eff_ft ran on GPU 2; on another GPU the device keys differ as well
        new = _flat(OmegaConf.to_container(_compose(_launcher_overrides(gpu=gpu)), resolve=False))
        diff = {k for k in saved.keys() | new.keys() if saved.get(k, "<absent>") != new.get(k, "<absent>")}
        print(f"\n--- composed {CONFIG_NAME} (GPU {gpu}) vs exact_eff_ft/.hydra/config.yaml: {len(diff)} keys differ")
        for k in sorted(diff):
            print(f"  {k}: {saved.get(k, '<absent>')!r} -> {new.get(k, '<absent>')!r}")
        device_keys = {"device", "sim_device", "train_devices"} if gpu != 2 else set()
        assert diff == INTENDED | set(NEW_DEFAULTS) | device_keys
        for k, v in NEW_DEFAULTS.items():
            assert k not in saved and new[k] == v, (k, new.get(k))
    assert new["env.hardware_schedule.steps"] == [START, RAMP_END] and new["env.steps"] == END
    assert new["finetune_from.checkpoint"].endswith("r2_pfsp/eff_ft_v2hw_ramp/init/exact_eff_ft_latest.pt")


def test_config_defaults_equal_the_launcher_values():
    """Running the config without the launcher's overrides gives the same run (apart from devices/logdir paths)."""
    plain = _flat(OmegaConf.to_container(_compose([]), resolve=False))
    launched = _flat(OmegaConf.to_container(_compose(_launcher_overrides(gpu=0)), resolve=False))
    diff = {k for k in plain if plain[k] != launched.get(k)}
    assert diff <= {"logdir", "device", "sim_device", "train_devices", "finetune_from.checkpoint"}, diff
    assert plain["finetune_from.checkpoint"] == "${logdir}/init/exact_eff_ft_latest.pt"


# =====================================================================================================================
# 6. The launcher (DRY_RUN: copy logic + trainer command, no GPU)
# =====================================================================================================================


def _python():
    py = Path("/workspace/isaaclab/_isaac_sim/python.sh")
    return str(py) if py.is_file() else sys.executable


def _launch(tmp_path, src, **env):
    run_env = {**os.environ, "GPU": "3", "C": str(ROOT), "T": str(tmp_path / "runs"), "SRC": str(src), "DRY_RUN": "1",
               "PY": _python(), **{k: str(v) for k, v in env.items()}}
    return subprocess.run(["bash", str(LAUNCHER)], env=run_env, capture_output=True, text=True, timeout=600)


def test_launcher_copies_once_and_passes_the_schedule(tmp_path):
    src = tmp_path / "source_run/latest.pt"
    src.parent.mkdir()
    torch.save({"step": START, "agent_state_dict": {}, "optims_state_dict": {}}, src)
    src_stat = (src.stat().st_size, src.stat().st_mtime_ns)
    out = _launch(tmp_path, src)
    assert out.returncode == 0, out.stderr
    d = tmp_path / "runs/eff_ft_v2hw_ramp"
    init = d / "init/exact_eff_ft_latest.pt"
    assert init.read_bytes() == src.read_bytes() and (d / "init/COPY_DONE").is_file()
    assert (src.stat().st_size, src.stat().st_mtime_ns) == src_stat  # the source is only read
    assert not (d / "init/exact_eff_ft_latest.pt.part").exists()
    cmd = out.stdout.split("\n")
    print("\n".join(cmd))
    for arg in (
        "--config-name=config-sprite-r2-pfsp-eff-v2hw-ft", "wandb_name=r2_pfsp_eff_ft_v2hw_ramp", f"logdir={d}",
        "device=cuda:3", "train_devices=[cuda:3]", "micro_batch_size=10", f"env.steps={END}",
        "opponent_separation.enabled=true", "model.imag_opponent_recompute=true", f"finetune_from.checkpoint={init}",
        f"finetune_from.expect_step={START}", f"env.hardware_schedule.steps=[{START},{RAMP_END}]", "WANDB_MODE=online",
    ):
        assert arg in cmd, arg
    status = (tmp_path / "runs/STATUS").read_text()
    assert "start checkpoint copied (step 10404928, byte-identical to the source)" in status
    # Later launches (crash-restart, or after the first save) never copy again; the trainer resumes from latest.pt.
    mtime = init.stat().st_mtime_ns
    (d / "latest.pt").write_bytes(b"own")
    torch.save({"step": START + 1}, src)  # even if the source changed meanwhile
    out2 = _launch(tmp_path, src)
    assert out2.returncode == 0 and out2.stdout == out.stdout and init.stat().st_mtime_ns == mtime
    assert (tmp_path / "runs/STATUS").read_text().count("copied") == 1


def test_launcher_refuses_a_source_at_another_step(tmp_path):
    src = tmp_path / "latest.pt"
    torch.save({"step": START + 16}, src)
    out = _launch(tmp_path, src, NAME="wrong_step")
    assert out.returncode == 1
    assert "expected START=10404928; not started" in (tmp_path / "runs/STATUS").read_text()
    assert not (tmp_path / "runs/wrong_step/init/COPY_DONE").exists()


def test_launcher_smoke_overrides(tmp_path):
    src = tmp_path / "latest.pt"
    torch.save({"step": START}, src)
    out = _launch(tmp_path, src, NAME="smoke", RAMP=8000, RUN=4000, WANDB_MODE="disabled",
                  EXTRA="trainer.eval_every=2000 self_play.pfsp.snapshot_every=4000")
    assert out.returncode == 0, out.stderr
    cmd = out.stdout.split("\n")
    for arg in (f"env.steps={START + 4000}", f"env.hardware_schedule.steps=[{START},{START + 8000}]",
                "WANDB_MODE=disabled", "trainer.eval_every=2000", "self_play.pfsp.snapshot_every=4000",
                "wandb_name=r2_pfsp_smoke"):
        assert arg in cmd, arg
