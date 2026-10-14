# =============================================================================
# Phase 1: AppLauncher must run before any IsaacLab / USD / warp imports.
# =============================================================================

import argparse
import cProfile
import os
import pathlib
import pstats
import sys

# Persist torch.compile caches across runs
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(pathlib.Path(__file__).resolve().parents[2], ".torch_cache"))

# Auto-detect vision env from CLI and enable cameras before AppLauncher
# parses argv, so the user doesn't have to pass --enable_cameras manually.
_vision = False
for _arg in sys.argv[1:]:
    if _arg.startswith("env=") and "vision" in _arg.split("=", 1)[1]:
        _vision = True
        break

# Detect sprite-based task — these render images on CPU from state, so they
# do NOT need GPU cameras enabled (saves VRAM and avoids rendering overhead).
_sprite_task = any(
    "Sprite" in _arg.split("=", 1)[1]
    for _arg in sys.argv[1:]
    if _arg.startswith("env.task=") or _arg.startswith("env=")
)

# Enable GPU cameras for all tasks except sprite-rendered ones.
if not _sprite_task and "--enable_cameras" not in sys.argv:
    sys.argv.insert(1, "--enable_cameras")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train r2dreamer with IsaacLab.")
AppLauncher.add_app_launcher_args(parser)
# Capture only the args AppLauncher understands; pass the rest to Hydra.
args_cli, hydra_args = parser.parse_known_args()

# Prevent the Omniverse renderer from initialising GPU contexts on devices
# other than the simulation GPU.  Without this, the MGPU renderer allocates
# ~2 GB of VRAM on every visible GPU — wasting memory on GPUs reserved for
# training.  Safe even on single-GPU (we never use multi-GPU rendering).
sys.argv += [
    "--/renderer/multiGpu/enabled=false",
    "--/renderer/multiGpu/maxGpuCount=1",
]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Phase 2: Everything else — safe to import now that the sim is running.
# =============================================================================

import atexit
import faulthandler
import math
import shutil
import signal
import time
import warnings

import hydra
import torch
from omegaconf import OmegaConf, open_dict

# Isaac Sim may override SIGINT; restore Python's default so Ctrl-C works.
signal.signal(signal.SIGINT, signal.default_int_handler)
# `kill -USR1 <pid>` dumps all Python thread stacks to the original stderr
# (the launch log) without stopping the run, e.g. to inspect a hang.
faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, all_threads=True)

# Absolute path to this script's directory — used by Hydra searchpath via
# the ${script_dir:} OmegaConf resolver so configs resolve regardless of cwd.
_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent)

# Register a resolver so config.yaml can write ${script_dir:} to get an
# absolute path that is independent of the current working directory.
OmegaConf.register_new_resolver("script_dir", lambda: _SCRIPT_DIR, use_cache=True)

sys.path.append(str(pathlib.Path(_SCRIPT_DIR) / "r2dreamer"))
sys.path.append(_SCRIPT_DIR)
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")
if os.environ.get("KLASK_CONV_FAST", "0") == "1":
    # Speed (2026-10-08, compile-speed): cuDNN picks its fastest algorithm per conv shape (fixed shapes here).
    torch.backends.cudnn.benchmark = True
    print("[speed] KLASK_CONV_FAST=1: channels-last convs, cudnn.benchmark", flush=True)

# Register IsaacLab task environments (triggers gymnasium gym.register calls).
import isaaclab_tasks  # noqa: F401
import klask_rl.tasks  # noqa: F401
import tools
from buffer import Buffer, PrioritizedBuffer
from dreamer import Dreamer

# Self-play wrapper (lives outside the r2dreamer submodule)
from dreamer_self_play import DreamerSelfPlayWrapper
from opponent_schedule import eval_anchor, last_dreamer_phase, opponent_fingerprint, parse_schedule, phase_at
from pfsp_pool import LATEST, PFSPPool, outcome_counts
from run_init import (
    buffer_dir_to_restore,
    check_finetune_checkpoint,
    load_trainer_state,
    restored_entry,
    start_pfsp_pool,
    start_plan,
)
from env_cfg_utils import apply_camera_size_to_env_cfg
from envs import make_envs
from envs.isaaclab import IsaacLabVecEnv
from gymnasium import Wrapper
from isaaclab.sim import RenderCfg
from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    CurriculumWrapper,
    HardwareSchedule,
    HardwareScheduleWrapper,
    InitializationWrapper,
    KlaskRlCollisionAvoidanceWrapper,
    KlaskRlRandomOpponentWrapper,
    OpponentActionWrapper,
    VelocityScaleWrapper,
    configure_domain_randomization,
)
from trainer import OnlineTrainer


class EpisodeMetricsWrapper(Wrapper):
    """Logs per-term episode rewards and termination rates.

    Reads directly from the reward and termination managers so that
    logged values are in the **same scale** as ``episode/score`` (the raw
    cumulative return the optimizer trains on).  A sliding window of
    recent episodes (default 100) smooths the logged values.

    Call :meth:`set_logger` after construction to enable logging.
    """

    def __init__(self, env, window_size: int = 100):
        super().__init__(env)
        self._logger = None
        self._window_size = window_size
        from collections import deque

        self._deque_factory = lambda: deque(maxlen=self._window_size)
        self._reward_history: dict[str, deque] = {}
        self._term_history: dict[str, deque] = {}
        self._total_episodes: int = 0

    def set_logger(self, logger):
        """Attach a :class:`tools.Logger` for autonomous metric logging."""
        self._logger = logger

    def step(self, actions):
        if self._logger is None:
            return self.env.step(actions)

        # Snapshot episode sums BEFORE step — _reset_idx (called inside
        # step) zeros them for done envs, so they'd be lost afterwards.
        rm = self.env.unwrapped.reward_manager
        pre_sums = {name: tensor.clone() for name, tensor in rm._episode_sums.items()}

        obs, rew, terminated, truncated, info = self.env.step(actions)

        done = terminated | truncated
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)

        if done_ids.numel() > 0:
            self._total_episodes += done_ids.numel()
            self._logger.scalar("buffer/episodes_total", self._total_episodes)
            dt = self.env.unwrapped.step_dt

            # Per-term cumulative reward for each done env.
            # pre_sums has the total through step N-1; _step_reward has
            # step N's func*weight (without dt), so multiply by dt.
            for env_id in done_ids.tolist():
                for term_idx, term_name in enumerate(rm._term_names):
                    full_sum = pre_sums[term_name][env_id].item() + rm._step_reward[env_id, term_idx].item() * dt
                    if term_name not in self._reward_history:
                        self._reward_history[term_name] = self._deque_factory()
                    self._reward_history[term_name].append(full_sum)

            # Only terms that carry weight (the others are identically 0).
            weights = dict(zip(rm.active_terms, rm._term_cfgs))
            for term_name, window in self._reward_history.items():
                cfg = weights.get(term_name)
                w = 0.0 if cfg is None else (cfg.weight.item() if isinstance(cfg.weight, torch.Tensor) else float(cfg.weight))
                if w != 0.0:
                    self._logger.scalar(f"episode/reward/{term_name}", sum(window) / len(window))

            # Termination metrics: which termination fired for each done env.
            tm = self.env.unwrapped.termination_manager
            for env_id in done_ids.tolist():
                for term_idx, term_name in enumerate(tm._term_names):
                    fired = tm._term_dones[env_id, term_idx].item()
                    if term_name not in self._term_history:
                        self._term_history[term_name] = self._deque_factory()
                    self._term_history[term_name].append(float(fired))

            for term_name, window in self._term_history.items():
                self._logger.scalar(f"episode/termination/{term_name}", sum(window) / len(window))

        return obs, rew, terminated, truncated, info


class RewardWeightLogWrapper(Wrapper):
    """Logs reward term weights from the env's ``reward_manager`` on each step.

    Call :meth:`set_logger` after construction to enable logging.
    """

    def __init__(self, env):
        super().__init__(env)
        self._logger = None
        self._last = {}

    def set_logger(self, logger):
        """Attach a :class:`tools.Logger` for autonomous metric logging."""
        self._logger = logger

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        if self._logger is not None:
            rm = self.env.unwrapped.reward_manager
            for term, cfg in zip(rm.active_terms, rm._term_cfgs):
                w = cfg.weight
                val = w.item() if isinstance(w, torch.Tensor) else float(w)
                # Log a weight only when it is non-zero or changes (curricula); constant
                # zero weights were 20+ flat curves per run.
                if self._last.get(term) != val and (val != 0.0 or term in self._last):
                    self._logger.scalar(f"rewards/weights/{term}", val)
                self._last[term] = val
        return obs, rew, terminated, truncated, info


# =============================================================================
# Episode tag tracking — weight-independent per-step signal accumulation
# =============================================================================


def _make_reward_term_bool_checker(term_name: str):
    """Return a callable that evaluates a reward term's raw function.

    The function is called with its configured parameters directly via the
    reward manager, bypassing the weight.  This means contact is detected
    even when the ``collision_player_ball`` reward weight has decayed to 0.
    """

    def check(env):
        rm = env.reward_manager
        if term_name not in rm._term_names:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        tidx = rm._term_names.index(term_name)
        cfg = rm._term_cfgs[tidx]
        return cfg.func(env, **cfg.params).bool()

    return check


class EventFlagWrapper(Wrapper):
    """Adds ``obs["event"]`` (B, 1): 1 when either peg touches the ball in this step (peg-ball centre distance
    < 17 mm, the rule of ``collision_player_ball_simple``), for the event-anchored replay
    (``buffer.prioritized.mode: events``, which also anchors on terminal steps). Evaluated inside
    ``env.step()`` together with the rewards, i.e. before Isaac Lab resets the envs that just ended."""

    def __init__(self, env):
        super().__init__(env)
        import copy

        from isaaclab.managers import SceneEntityCfg
        from klask_rl.tasks.manager_based.klask_rl.env_cfg.klask_rl_rewards_cfg import (
            _opponent_reward_terms,
            _player_reward_terms,
        )

        u = self.env.unwrapped
        self._terms = []
        for terms in (_player_reward_terms, _opponent_reward_terms):
            term = terms["collision_player_ball_simple"]
            params = copy.deepcopy(term.params)
            for v in params.values():
                if isinstance(v, SceneEntityCfg):
                    v.resolve(u.scene)
            self._terms.append((term.func, params))
        self._event = torch.zeros(u.num_envs, device=u.device)
        rm = u.reward_manager
        compute = rm.compute

        def compute_with_events(dt):
            reward = compute(dt)
            hit = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
            for func, params in self._terms:
                hit |= func(u, **params) > 0
            self._event = hit.float()
            return reward

        rm.compute = compute_with_events

    def step(self, action, *args, **kwargs):
        obs, rew, terminated, truncated, info = self.env.step(action, *args, **kwargs)
        obs["event"] = self._event.unsqueeze(-1).clone()
        return obs, rew, terminated, truncated, info


class EpisodeTagTracker(Wrapper):
    """OR-accumulates per-step boolean signals across each episode.

    *step_checkers* maps a tag name to a callable
    ``fn(env_unwrapped) -> (B,) bool tensor``.  At each ``step()`` call the
    checker result is OR-ed into a per-env flag.  When an episode ends the
    flag is snapshotted before the env's auto-reset zeroes related state.
    """

    def __init__(self, env, step_checkers: dict):
        super().__init__(env)
        n = env.unwrapped.num_envs
        dev = env.unwrapped.device
        self._step_checkers = step_checkers
        self._episode_flags = {t: torch.zeros(n, dtype=torch.bool, device=dev) for t in step_checkers}
        self._episode_snapshot = {t: torch.zeros(n, dtype=torch.bool, device=dev) for t in step_checkers}

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        done = terminated | truncated
        unwrapped = self.env.unwrapped
        for tag, checker in self._step_checkers.items():
            self._episode_flags[tag] |= checker(unwrapped)
        # Snapshot current episode flags for done envs, then reset them.
        for tag in self._step_checkers:
            self._episode_snapshot[tag][done] = self._episode_flags[tag][done]
            self._episode_flags[tag][done] = False
        return obs, rew, terminated, truncated, info

    def get_episode_flag(self, env_idx: int, tag: str) -> bool:
        """Return whether *tag* fired at any point during the last episode of *env_idx*."""
        return self._episode_snapshot[tag][env_idx].item()


# =============================================================================
# Task registry — all task-specific knowledge lives here, not in envs/__init__.py
# =============================================================================


# Global reference to the self-play wrapper (set during env construction,
# used later to initialise / update the opponent from the training agent).
_self_play_wrapper = None

# Global reference to the EpisodeTagTracker wrapper (set during env construction,
# used in KlaskTrainer.on_episode_end to query per-step episode flags).
_tag_tracker: EpisodeTagTracker | None = None

# Global reference to the HardwareScheduleWrapper (env.hardware_schedule; None = constant hardware), bound to the
# trainer's step in main().
_hw_schedule_wrapper = None


def _make_env(
    config, gym_id, render_mode=None, trainer_steps=None, self_play=False, self_play_config=None, prioritized_cfg=None,
    trajectory_mirroring=False,
):
    """Construct a GPU-resident IsaacLab env with KLASK-specific wrappers.

    Applies (in order, innermost → outermost):
      0. KlaskRlCollisionAvoidanceWrapper (if config.collision_avoidance.enable —
         innermost; clips world-frame m/s actions near board boundaries)
      1. OpponentActionWrapper (ego → world frame; negates opponent dims)
      2. ActuatorModelWrapper (if config.actuator_model.enable — expects m/s)
      3. VelocityScaleWrapper (scales [-1, 1] → m/s; action manager _scale = 1.0)
      3a. HardwareScheduleWrapper (if config.hardware_schedule.enable — sets the velocity scale, the rate cap and
          the collision avoidance's speed from the step; hardware_schedule_wrapper.py)
      4. InitializationWrapper
      5. CurriculumWrapper (if config.rewards)
      6. Opponent wrapper (emits opponent action in [-1, 1]):
         - DreamerSelfPlayWrapper if self_play=True
         - KlaskRlRandomOpponentWrapper otherwise
      7. EpisodeTagTracker (if buffer.prioritized configured)
      8. EpisodeMetricsWrapper (episode termination logging)
      9. RewardWeightLogWrapper (reward weight logging)
    """
    global _self_play_wrapper, _tag_tracker, _hw_schedule_wrapper
    import importlib

    import gymnasium as gym

    env_cfg_entry = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    if isinstance(env_cfg_entry, str):
        module_name, class_name = env_cfg_entry.rsplit(":", 1)
        env_cfg_class = getattr(importlib.import_module(module_name), class_name)
    else:
        env_cfg_class = env_cfg_entry

    env_cfg = env_cfg_class()
    # Simulate on the configured device (env.device defaults to ${device} =
    # cuda:0). Pair with the AppLauncher flag --device cuda:N so the renderer
    # and PhysX use the same GPU; CUDA_VISIBLE_DEVICES does not work for Isaac
    # Sim (its Vulkan device enumeration ignores it).
    if getattr(config, "device", None):
        env_cfg.sim.device = str(config.device)

    # Override ball reset position from yaml config if provided.
    # Sets named fields on env_cfg, then re-propagates to event params
    # (post_init already ran with defaults).
    ball_reset_x = getattr(config, "ball_reset_position_x", None)
    ball_reset_y = getattr(config, "ball_reset_position_y", None)
    if ball_reset_x is not None and hasattr(env_cfg, "ball_reset_position_x"):
        env_cfg.ball_reset_position_x = tuple(ball_reset_x)
        env_cfg.ball_reset_position_y = tuple(ball_reset_y)
        env_cfg.events.reset_ball_position.params["pose_range"]["x"] = env_cfg.ball_reset_position_x
        env_cfg.events.reset_ball_position.params["pose_range"]["y"] = env_cfg.ball_reset_position_y

    # Reachable ball starts (2026-10-07): never inside the band around the midline that neither peg can reach.
    ball_min_abs_y = getattr(config, "ball_reset_min_abs_y", None)
    if ball_min_abs_y is not None and hasattr(env_cfg.events, "reset_ball_position"):
        from klask_rl.tasks.manager_based.klask_rl.utils_manager_based import reset_ball_outside_midline_band

        env_cfg.events.reset_ball_position.func = reset_ball_outside_midline_band
        env_cfg.events.reset_ball_position.params["min_abs_y"] = float(ball_min_abs_y)

    # State-based runs (2026-10-09): the model reads the 12 raw state values of each seat
    # (own peg, other peg, ball: x/y position and velocity, in that seat's frame) instead of
    # the image. Only env 0's board is still rendered, as "image", for the occasional game
    # video; the trainer leaves it out of the replay buffer (trainer.buffer_exclude_keys).
    if bool(getattr(config, "state_only", False)):
        from klask_rl.tasks.manager_based.klask_rl.env_cfg.klask_rl_observations_cfg import (
            _OpponentPolicyCfg,
            _PlayerPolicyCfg,
            apply_state_units,
        )
        from klask_rl.tasks.manager_based.klask_rl.utils_manager_based import sprite_rendered_image_env0

        env_cfg.observations.policy = _PlayerPolicyCfg()
        env_cfg.observations.opponent = _OpponentPolicyCfg()
        # env.state_units: every value divided by its unit (board half-size, command limit, 0.5 m/s ball),
        # so it is of order 1 for the encoder and the reconstruction target (state-v3 s80 failed in metres).
        if bool(getattr(config, "state_units", False)):
            apply_state_units(env_cfg.observations.policy, config.max_velocity)
            apply_state_units(env_cfg.observations.opponent, config.max_velocity)
        env_cfg.observations.image.image.func = sprite_rendered_image_env0
        env_cfg.observations.opponent_image = None

    sim_dt = getattr(config, "sim_dt", None)
    if sim_dt is not None:
        env_cfg.sim.dt = float(sim_dt)

    env_cfg.scene.num_envs = int(config.env_num)
    env_cfg.decimation = int(config.decimation)
    env_cfg.seed = int(config.seed)
    env_cfg.episode_length_s = config.episode_length_s

    aug_raw = getattr(config, "augmentation", None)
    if aug_raw is not None and hasattr(env_cfg, "augmentation_cfg"):
        from klask_rl.tasks.manager_based.klask_rl.klask_rl_env_cfg import AugmentationAxisCfg, AugmentationCfg

        aug_dict = OmegaConf.to_container(aug_raw, resolve=True) if OmegaConf.is_config(aug_raw) else dict(aug_raw)
        aug = AugmentationCfg(hold_per_episode=bool(aug_dict.get("hold_per_episode", True)))
        for axis_name in ("brightness", "contrast", "gamma", "color_temp", "saturation", "hue"):
            block = aug_dict.get(axis_name)
            if not isinstance(block, dict):
                continue
            setattr(
                aug,
                axis_name,
                AugmentationAxisCfg(
                    enabled=bool(block.get("enabled", False)),
                    range=tuple(block.get("range", (0.0, 0.0))),
                    levels=int(block.get("levels", 1)),
                ),
            )
        env_cfg.augmentation_cfg = aug

    # IsaacLab defaults to DLSS which smooths the image significantly
    # so we disable the antialiasing for a more pixelated (and hence more realistic) image.
    env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")

    # --- Camera resolution & padding derived from env.size ---
    apply_camera_size_to_env_cfg(env_cfg, getattr(config, "size", None))

    # Null out disabled termination terms on env_cfg BEFORE construction so
    # IsaacLab's TerminationManager never registers them (_prepare_terms skips None).
    terminations_cfg = getattr(config, "terminations", None)
    if terminations_cfg is not None and hasattr(env_cfg, "terminations"):
        term_dict = (
            OmegaConf.to_container(terminations_cfg, resolve=True)
            if OmegaConf.is_config(terminations_cfg)
            else dict(terminations_cfg)
        )
        for term, active in term_dict.items():
            if not active and hasattr(env_cfg.terminations, term):
                setattr(env_cfg.terminations, term, None)

    # Dead-ball truncation (env.ball_stall, off by default; 2026-10-11: in the pilot 23% of training steps were a
    # ball lying still until the 10 s timeout). A time-out-like term: no reward, not terminal; a new kick-off follows.
    _stall_cfg = getattr(config, "ball_stall", None)
    if _stall_cfg is not None and bool(getattr(_stall_cfg, "enable", False)):
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from klask_rl.tasks.manager_based.klask_rl.utils_manager_based import ball_stalled

        env_cfg.terminations.ball_stalled = DoneTerm(
            func=ball_stalled,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "min_speed": float(getattr(_stall_cfg, "min_speed", 0.01)),
                "duration_s": float(getattr(_stall_cfg, "duration_s", 2.0)),
            },
            time_out=True,
        )

    # Configure domain randomization events from YAML BEFORE env construction.
    _dr_cfg_raw = getattr(config, "domain_randomization", None)
    _dr_dict = (
        OmegaConf.to_container(_dr_cfg_raw, resolve=True)
        if _dr_cfg_raw is not None and OmegaConf.is_config(_dr_cfg_raw)
        else _dr_cfg_raw
    )
    configure_domain_randomization(env_cfg, _dr_dict)

    # --- Create the base gymnasium env ---
    isaac_env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode)

    # --- 1. Action manager scale = 1.0 (pass-through, m/s in → m/s out) ---
    # Velocity scaling is done by VelocityScaleWrapper below so that the
    # actuator model sees commands in m/s (matching its training units).
    # max_velocity is also reused by KlaskRlCollisionAvoidanceWrapper.
    max_velocity = getattr(config, "max_velocity", None)
    if max_velocity is None:
        raise ValueError("config.env.max_velocity is required; VelocityScaleWrapper needs it to scale [-1, 1] → m/s.")
    action_mgr = isaac_env.unwrapped.action_manager
    for term in action_mgr._terms.values():
        term._scale = 1.0

    # --- 0. Collision avoidance (world-frame m/s clipping; innermost gym wrapper) ---
    # Sits inside OpponentActionWrapper so it sees actions already converted
    # from ego to world frame, matching the world-frame peg positions and
    # edge constants (X_EDGE, Y_EDGE_1, Y_EDGE_2) that the wrapper uses.
    # With env.state_units the observation values are in units, but these wrappers read peg positions and
    # velocities from the observations in metres (the actuator model was fitted on m/s): pass the units back.
    _obs_units = {}
    if bool(getattr(config, "state_only", False)) and bool(getattr(config, "state_units", False)):
        from klask_rl.tasks.manager_based.klask_rl.env_cfg.klask_rl_observations_cfg import state_unit_divisors

        _div = state_unit_divisors(float(max_velocity))
        assert _div["own_x_vel"] == _div["own_y_vel"]
        _obs_units = {"pos": tuple(_div["own_pos"]), "vel": float(_div["own_x_vel"])}
    ca_cfg = getattr(config, "collision_avoidance", None)
    ca_wrapper = None
    if ca_cfg is not None and getattr(ca_cfg, "enable", False):
        isaac_env = KlaskRlCollisionAvoidanceWrapper(
            isaac_env, max_vel=float(max_velocity), obs_pos_unit=_obs_units.get("pos", (1.0, 1.0))
        )
        ca_wrapper = isaac_env

    # --- 1a. Opponent action frame transform ---
    isaac_env = OpponentActionWrapper(isaac_env)

    # --- 2. Actuator model wrapper (expects m/s commands) ---
    actuator_cfg = getattr(config, "actuator_model", None)
    lag_cfg = getattr(config, "actuator_lag", None)
    if actuator_cfg is not None and actuator_cfg.enable:
        if lag_cfg is not None and lag_cfg.enable:
            raise ValueError("actuator_model and actuator_lag are alternatives; enable only one")
        isaac_env = ActuatorModelWrapper(
            isaac_env,
            model_file=actuator_cfg.checkpoint,
            obs_vel_unit=_obs_units.get("vel", 1.0),
            obs_pos_unit=_obs_units.get("pos", (1.0, 1.0)),
        )
    elif lag_cfg is not None and lag_cfg.enable:
        # Parametric surrogate (dead time + first-order lag) for hardware without a fitted model.
        from klask_rl.tasks.manager_based.klask_rl.actuator_model.actuator_lag import ActuatorLagWrapper

        isaac_env = ActuatorLagWrapper(isaac_env, dead_time_s=lag_cfg.dead_time_s, tau_s=lag_cfg.tau_s)

    # --- 2a. Velocity scaling: [-1, 1] → [-max_velocity, max_velocity] m/s ---
    # Sits outside the actuator model (so the model sees m/s) and inside
    # the opponent wrappers (so player and opponent actions are scaled
    # together in one pass — both are produced in [-1, 1]).
    max_acceleration = getattr(config, "max_acceleration", None)
    _act_space = isaac_env.unwrapped.single_action_space
    isaac_env = VelocityScaleWrapper(
        isaac_env,
        max_velocity=float(max_velocity),
        max_acceleration=None if max_acceleration is None else float(max_acceleration),
        num_envs=int(isaac_env.unwrapped.num_envs),
        action_dim=int(_act_space.shape[0]),
        device=isaac_env.unwrapped.device,
    )

    # --- 2a'. Hardware-speed curriculum (env.hardware_schedule, off when absent) ---
    # The velocity scale, the rate cap and the collision avoidance's speed follow the step (both seats share them).
    _hw_schedule_wrapper = None
    _hw_cfg = getattr(config, "hardware_schedule", None)
    if OmegaConf.is_config(_hw_cfg):
        _hw_cfg = OmegaConf.to_container(_hw_cfg, resolve=True)
    hw_schedule = HardwareSchedule.from_config(
        _hw_cfg, float(max_velocity), None if max_acceleration is None else float(max_acceleration)
    )
    if hw_schedule is not None:
        if actuator_cfg is not None and actuator_cfg.enable:
            print("WARNING: hardware_schedule with the learned v1 actuator model (fitted on the v1 table only)")
        velocity_wrapper = isaac_env
        isaac_env = HardwareScheduleWrapper(
            velocity_wrapper, hw_schedule, velocity_wrapper=velocity_wrapper, collision_wrapper=ca_wrapper
        )
        _hw_schedule_wrapper = isaac_env

    # --- 2b. Initialization wrapper (player init velocity, etc.) ---
    init_cfg = getattr(config, "initialization", None)
    if init_cfg is not None:
        if OmegaConf.is_config(init_cfg):
            init_dict = OmegaConf.to_container(init_cfg, resolve=True)
        else:
            init_dict = dict(init_cfg)
        isaac_env = InitializationWrapper(isaac_env, init_dict)

    # --- 3. Reward curriculum wrapper ---
    rewards_cfg = getattr(config, "rewards", None)
    if rewards_cfg is not None:
        if OmegaConf.is_config(rewards_cfg):
            rewards_dict = OmegaConf.to_container(rewards_cfg, resolve=True)
        else:
            rewards_dict = dict(rewards_cfg)
        isaac_env = CurriculumWrapper(isaac_env, rewards_dict)

    # --- 3b. Opponent reward computation (needed for trajectory mirroring) ---
    if trajectory_mirroring:
        from klask_rl.tasks.manager_based.klask_rl.env_cfg.klask_rl_rewards_cfg import _opponent_reward_terms
        from klask_rl.tasks.manager_based.klask_rl.wrappers.klask_rl_training_wrappers import OpponentRewardWrapper

        term_mapping = {name: (term.func, term.params) for name, term in _opponent_reward_terms.items()}
        isaac_env = OpponentRewardWrapper(isaac_env, term_mapping)

    # --- 5. Opponent wrapper ---
    if self_play:
        sp_cfg = self_play_config or {}
        _self_play_wrapper = DreamerSelfPlayWrapper(
            isaac_env,
            update_score=float(sp_cfg.get("update_score", 0.7)),
            games_to_track=int(sp_cfg.get("games_to_track", 4096)),
            compile=bool(sp_cfg.get("compile", False)),
            emit_latents=bool(sp_cfg.get("emit_latents", True)),
            explore_rho=float(sp_cfg.get("explore_rho", 0.0)),
        )
        isaac_env = _self_play_wrapper
    else:
        isaac_env = KlaskRlRandomOpponentWrapper(isaac_env)

    # --- 5b. Episode tag tracker (only when prioritized buffer is configured) ---
    _tag_tracker = None
    if prioritized_cfg is not None:
        _tags_raw = getattr(prioritized_cfg, "tags", [])
        _tags_list = (
            list(OmegaConf.to_container(_tags_raw, resolve=True)) if OmegaConf.is_config(_tags_raw) else list(_tags_raw)
        )
        step_checkers = {
            s["name"]: _make_reward_term_bool_checker(s["reward_term"]) for s in _tags_list if "reward_term" in s
        }
        if step_checkers:
            isaac_env = EpisodeTagTracker(isaac_env, step_checkers)
            _tag_tracker = isaac_env
        # --- 5c. Per-step contact flag for the event-anchored replay ---
        if str(getattr(prioritized_cfg, "mode", "tags")) == "events":
            isaac_env = EventFlagWrapper(isaac_env)

    # --- 6. Episode metrics capture ---
    isaac_env = EpisodeMetricsWrapper(isaac_env)

    # --- 7. Reward weight logging (outermost gymnasium wrapper) ---
    isaac_env = RewardWeightLogWrapper(isaac_env)

    # Wrap in the r2dreamer IsaacLabVecEnv adapter
    return IsaacLabVecEnv(isaac_env, simulation_app=simulation_app)


# =============================================================================
# Main
# =============================================================================


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    # env.task follows the codebase convention: "isaaclab_<task_name>"
    # e.g. "isaaclab_cartpole_balance"
    full_task = config.env.task  # e.g. "isaaclab_cartpole_balance"
    _, task_name = full_task.split("_", 1)  # e.g. "cartpole_balance"

    render_mode = "rgb_array" if _vision else None
    trainer_steps = getattr(config.trainer, "steps", None)

    # --- Parse hierarchical self_play config ---
    _sp_cfg_raw = getattr(config, "self_play", None)
    if OmegaConf.is_config(_sp_cfg_raw):
        _sp_dict = OmegaConf.to_container(_sp_cfg_raw, resolve=True)
        self_play = bool(_sp_dict.get("enabled", False))
        sp_cfg = {k: v for k, v in _sp_dict.items() if k != "enabled"}
    elif isinstance(_sp_cfg_raw, bool):
        # Legacy flat format: self_play: true + self_play_config: {...}
        self_play = _sp_cfg_raw
        _legacy_sp = getattr(config, "self_play_config", None)
        sp_cfg = OmegaConf.to_container(_legacy_sp, resolve=True) if OmegaConf.is_config(_legacy_sp) else {}
    else:
        self_play = False
        sp_cfg = {}
    # Mirror the model compile flag so the opponent is also compiled when the
    # training agent is.
    if self_play:
        sp_cfg.setdefault("compile", bool(getattr(config.model, "compile", False)))
        # The self-play opponent explores with the same AR(1) noise as the learner (model.explore_noise).
        _en = getattr(config.model, "explore_noise", None)
        _tau = float(getattr(_en, "tau", 0.0) or 0.0) if _en is not None else 0.0
        sp_cfg.setdefault("explore_rho", math.exp(-float(getattr(_en, "dt", 0.02)) / _tau) if _tau > 0 else 0.0)

    # --- Parse hierarchical opponent_separation config ---
    _action_timing = "legacy"
    _opp_sep_raw = getattr(config, "opponent_separation", None)
    if OmegaConf.is_config(_opp_sep_raw):
        _opp_sep_dict = OmegaConf.to_container(_opp_sep_raw, resolve=True)
        _opp_sep = bool(_opp_sep_dict.get("enabled", False))
        _imag_opponent = str(_opp_sep_dict.get("imag_opponent", "random"))
        _action_timing = str(_opp_sep_dict.get("action_timing", "legacy"))
        _traj_mirror_raw = _opp_sep_dict.get("trajectory_mirroring", {})
        if isinstance(_traj_mirror_raw, dict):
            _traj_mirror = bool(_traj_mirror_raw.get("enabled", False))
            _mirror_opp_ws = bool(_traj_mirror_raw.get("warm_start", False))
        else:
            # Simple bool: opponent_separation.trajectory_mirroring: true
            _traj_mirror = bool(_traj_mirror_raw)
            _mirror_opp_ws = False
    elif isinstance(_opp_sep_raw, bool):
        # Legacy flat format
        _opp_sep = _opp_sep_raw
        _legacy_opp_cfg = getattr(config, "opponent_separation_config", None)
        if _legacy_opp_cfg is not None:
            _legacy_dict = OmegaConf.to_container(_legacy_opp_cfg, resolve=True) if OmegaConf.is_config(_legacy_opp_cfg) else dict(_legacy_opp_cfg)
            _imag_opponent = str(_legacy_dict.get("imag_opponent", "random"))
        else:
            _imag_opponent = "random"
        _traj_mirror = bool(getattr(config, "trajectory_mirroring", False))
        _mirror_opp_ws = bool(getattr(config, "mirror_opp_warm_start", False))
    else:
        _opp_sep = False
        _imag_opponent = "random"
        _traj_mirror = False
        _mirror_opp_ws = False
    # Opponent-action timing in the replay buffer and act():
    #   legacy  - historical behaviour: obs_t paired with (a_{t-1}, o_{t-2}) and,
    #             in mirrored samples, the own action one step late.
    #   aligned - obs_t paired with (a_{t-1}, o_{t-1}), the commands executed together.
    if _action_timing not in ("legacy", "aligned"):
        raise SystemExit(f"opponent_separation.action_timing must be 'legacy' or 'aligned', got {_action_timing!r}")
    _aligned_opp_action = _action_timing == "aligned"
    # Shared-world imagination (model.imag_opponent_world shared, 2026-10-11): the imagined self-play opponent
    # observes our decoded state from its seat (opponent_seat_map, installed below). It needs a self-play Dreamer
    # opponent in imagination, its encoder (option A) and the 12-value state observations.
    _imag_world = str(OmegaConf.select(config, "model.imag_opponent_world", default="separate"))
    if _imag_world == "shared":
        _need = {
            "self_play.enabled": self_play,
            "opponent_separation.enabled": _opp_sep,
            "opponent_separation.imag_opponent == selfplay": _imag_opponent == "selfplay",
            "model.imag_opponent_recompute": bool(OmegaConf.select(config, "model.imag_opponent_recompute", default=False)),
            "env.state_only": bool(OmegaConf.select(config, "env.state_only", default=False)),
        }
        _missing = [k for k, ok in _need.items() if not ok]
        if _missing:
            raise SystemExit(f"model.imag_opponent_world 'shared' needs {', '.join(_missing)}")

    # --- Fixed opponent schedule (controlled opponent-switch experiments) ---
    _opp_schedule = None
    if "schedule" in sp_cfg:
        _schedule_entries = sp_cfg.pop("schedule")
        if _schedule_entries:
            if not self_play:
                raise SystemExit("self_play.schedule requires self_play.enabled: true")
            if int(getattr(config.trainer, "eval_episode_num", 0)) <= 0:
                raise SystemExit("self_play.schedule requires trainer.eval_episode_num > 0 (switches run in the eval hook)")
            try:
                _opp_schedule = parse_schedule(
                    _schedule_entries, config.trainer.eval_every, root=pathlib.Path(_SCRIPT_DIR).parents[1]
                )
            except ValueError as exc:
                raise SystemExit(f"Invalid self_play.schedule: {exc}") from exc
            print("Opponent schedule:")
            for _phase in _opp_schedule:
                print(f"  phase {_phase.index}: from step {_phase.start:>9,} {_phase.kind:7s} {_phase.checkpoint}")

    # Self-play opponent mode: "score_gate" (copy the learner when it wins enough)
    # or "pfsp" (prioritized fictitious self-play over a pool of own snapshots).
    _sp_mode = str(sp_cfg.get("mode", "score_gate"))
    if _sp_mode not in ("score_gate", "pfsp"):
        raise SystemExit(f"self_play.mode must be 'score_gate' or 'pfsp', got {_sp_mode!r}")
    if _sp_mode == "pfsp":
        if _opp_schedule is not None:
            raise SystemExit("self_play.mode 'pfsp' and self_play.schedule are mutually exclusive")
        if int(getattr(config.trainer, "eval_episode_num", 0)) <= 0:
            # Rotations then run in KlaskTrainer.rotation_hook at the same period (trainer.eval_every), without
            # evaluation games.
            print("PFSP: trainer.eval_episode_num 0: opponent rotations every trainer.eval_every steps without evaluation games")
    _pfsp_cfg = dict(sp_cfg.get("pfsp", {}) or {})

    _prioritized_cfg = getattr(config.buffer, "prioritized", None)
    _use_prioritized = _prioritized_cfg is not None and bool(getattr(_prioritized_cfg, "enable", False))
    _learned_prio = _use_prioritized and str(getattr(_prioritized_cfg, "mode", "tags")) == "learned"
    if _use_prioritized:
        _tags_raw = getattr(_prioritized_cfg, "tags", [])
        _tags_cfg_list = (
            list(OmegaConf.to_container(_tags_raw, resolve=True)) if OmegaConf.is_config(_tags_raw) else list(_tags_raw)
        )
    else:
        _tags_cfg_list = []

    # --- Resolve sim / train devices ---
    sim_device = getattr(config, "sim_device", None) or config.device

    # train_devices: list of training GPUs.  First element is the primary
    # (holds optimizer).  Falls back to [device] when null.
    train_devices_cfg = getattr(config, "train_devices", None)
    if train_devices_cfg:
        train_devices = list(train_devices_cfg)
    else:
        train_devices = [config.device]
    train_device = train_devices[0]  # primary training GPU

    _mbs = int(getattr(config, "micro_batch_size", config.batch_size))
    _bs = int(config.batch_size)
    num_train_gpus = len(train_devices)
    # Trajectory mirroring appends a mirrored copy of every sampled row, so the learner
    # splits 2 x batch_size rows into micro-batches.
    _mirror = bool(OmegaConf.select(config, "opponent_separation.trajectory_mirroring.enabled", default=False))
    _rows = _bs * (2 if _mirror else 1)
    if _rows % _mbs != 0:
        raise SystemExit(
            f"batch rows ({_rows} = batch_size {_bs}{' x 2 mirrored' if _mirror else ''}) "
            f"must be divisible by micro_batch_size ({_mbs})."
        )
    if num_train_gpus > 1:
        if _mbs >= _bs:
            raise SystemExit(
                f"With {num_train_gpus} training GPUs, micro_batch_size ({_mbs}) "
                f"must be smaller than batch_size ({_bs})."
            )
        if (_bs // _mbs) % num_train_gpus != 0:
            raise SystemExit(
                f"num_micro_batches ({_bs // _mbs} = batch_size {_bs} / micro_batch_size {_mbs}) "
                f"must be divisible by num_train_gpus ({num_train_gpus})."
            )
    if _learned_prio and num_train_gpus > 1:
        raise SystemExit("buffer.prioritized.mode 'learned' supports one training GPU only")
    with open_dict(config):
        config.micro_batch_size = _mbs

    with open_dict(config.model):
        config.model.sim_device = sim_device
        config.model.train_device = train_device
        config.model.train_devices = train_devices
        config.model.device = train_device
        config.model.micro_batch_size = _mbs
        if _learned_prio:
            config.model.prio_burn_in = int(getattr(_prioritized_cfg, "burn_in", 8))

    with open_dict(config.buffer):
        config.buffer.device = train_device  # sampled batches go to training GPU

    # --- How this run starts (run_init.py), decided before the simulator starts: resume from the run's own
    # <logdir>/latest.pt (always, when it exists), else a fine-tune from the full trainer state of a checkpoint copy
    # inside the logdir (finetune_from: fresh buffer, fresh PFSP pool), else weights only (init_checkpoint), else scratch.
    _ft_cfg = getattr(config, "finetune_from", None)
    _ft_ckpt = None if _ft_cfg is None else getattr(_ft_cfg, "checkpoint", None)
    _ft_expect = None if _ft_cfg is None else getattr(_ft_cfg, "expect_step", None)
    _ft_source = None if _ft_cfg is None else getattr(_ft_cfg, "source", None)
    try:
        _plan = start_plan(config.logdir, _ft_ckpt, config.init_checkpoint)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Cannot start: {exc}") from exc
    print(f"Run start: {_plan.mode}" + (f" from {_plan.checkpoint}" if _plan.checkpoint is not None else ""))

    vec_env = _make_env(
        config.env,
        task_name,
        render_mode,
        trainer_steps=trainer_steps,
        self_play=self_play,
        self_play_config=sp_cfg,
        prioritized_cfg=_prioritized_cfg,
        trajectory_mirroring=_traj_mirror,
    )

    # Auto-add unconfigured termination terms with baseline_priority so they
    # are logged in buffer/tagged_* metrics without affecting sampling.
    if _use_prioritized:
        _baseline = float(getattr(_prioritized_cfg, "baseline_priority", 1.0))
        _configured_terms = {s["termination_term"] for s in _tags_cfg_list if "termination_term" in s}
        for _tname in vec_env._env.unwrapped.termination_manager._term_names:
            if _tname not in _configured_terms:
                _tags_cfg_list.append({"name": _tname, "priority": _baseline, "termination_term": _tname})

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    wandb_cfg = {
        "project": getattr(config, "wandb_project", "r2dreamer-isaaclab"),
        "name": getattr(config, "wandb_name", f"dreamer_{task_name}"),
        "dir": str(logdir),
    }
    # Resume: continue the same W&B run (id from the previous online run in this logdir)
    # instead of opening a second run with the same name.
    if (logdir / "latest.pt").exists():
        _previous = sorted((logdir / "wandb").glob("run-*-*"))
        if _previous:
            wandb_cfg.update(id=_previous[-1].name.rsplit("-", 1)[-1], resume="allow")
            print(f"Resuming W&B run {wandb_cfg['id']}")
    # Compute agent control frequency for real-time video playback.
    _sim_dt = float(getattr(config.env, "sim_dt", 0.001))
    _decimation = int(config.env.decimation)
    _video_fps = int(round(1.0 / (_decimation * _sim_dt)))
    logger = tools.Logger(
        logdir,
        backends=[
            tools.JSONLBackend(logdir),
            # tools.TensorBoardBackend(logdir, video_fps=_video_fps),
            tools.WandbBackend(wandb_cfg, video_fps=_video_fps),
        ],
    )
    logger.log_hydra_config(config)

    # Attach the logger to logging wrappers so they log autonomously.
    env = vec_env._env
    while isinstance(env, Wrapper):
        if hasattr(env, "set_logger"):
            env.set_logger(logger)
        env = env.env

    # Derive buffer flags from the parsed opponent_separation config.
    OmegaConf.update(config, "buffer.mirror", _traj_mirror, force_add=True)
    OmegaConf.update(config, "buffer.mirror_opp_warm_start", _mirror_opp_ws, force_add=True)
    OmegaConf.update(config, "buffer.aligned_opponent_action", _aligned_opp_action, force_add=True)

    if _use_prioritized:
        replay_buffer = PrioritizedBuffer(config.buffer)
    else:
        replay_buffer = Buffer(config.buffer)

    print("Create env.")
    # OmegaConf configs are read-only; use a plain object to pass the env through.
    env_config = OmegaConf.to_container(config.env, resolve=True)
    env_config = type("EnvConfig", (), env_config)()
    env_config.isaac_vec_env = vec_env
    env_config.sim_device = sim_device
    train_envs, eval_envs, obs_space, act_space = make_envs(env_config)

    # Pass parsed flags to model config so Dreamer can read them.
    config.model.opponent_separation = _opp_sep
    config.model.trajectory_mirroring = _traj_mirror
    config.model.imag_opponent = _imag_opponent
    OmegaConf.update(config, "model.aligned_opponent_action", _aligned_opp_action, force_add=True)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(train_device)

    # Validate init_checkpoint path early.
    _init_ckpt = config.init_checkpoint

    # Register self-play opponent submodules on the agent BEFORE loading any
    # checkpoint.  This ensures _imag_opp_rssm / _imag_opp_actor (and, in
    # multi-GPU mode, _imag_opp_rssm_src / _imag_opp_actor_src) are known to
    # PyTorch so load_state_dict can restore their weights instead of silently
    # dropping them as "unexpected" keys.  The deep-copied modules start with
    # random weights but will be overwritten by the checkpoint or synced from
    # the agent below.
    # Scheduled frozen opponents: phase index currently installed (None = ordinary self-play).
    _opp_phase_state = {"current": None}

    def _load_snapshot_agent(phase):
        """Load a frozen Dreamer snapshot (its own run config) on the sim device."""
        from checkpoint_loaders import _load_dreamer_agent

        snapshot_cfg = OmegaConf.load(phase.config)
        with open_dict(snapshot_cfg):
            snapshot_cfg.model.compile = False
        return _load_dreamer_agent(snapshot_cfg, obs_space, act_space, phase.checkpoint, sim_device)

    def _load_ppo_player(phase):
        """Load a frozen rl_games PPO player without perturbing the training RNG streams."""
        import random as _random

        import numpy as _np
        from checkpoint_loaders import _load_ppo_opponent

        rng_state = (
            _random.getstate(),
            _np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )
        try:
            ppo_obs_space = vec_env._env.unwrapped.single_observation_space["opponent"]
            return _load_ppo_opponent(
                phase.config, phase.checkpoint, vec_env._env.unwrapped.num_envs, sim_device, ppo_obs_space
            )
        finally:
            # rl_games Runner.load() reseeds the global generators.
            _random.setstate(rng_state[0])
            _np.random.set_state(rng_state[1])
            torch.set_rng_state(rng_state[2])
            if rng_state[3] is not None:
                torch.cuda.set_rng_state_all(rng_state[3])

    def _install_scheduled_opponent(index):
        """Make schedule phase ``index`` the active opponent.

        Dreamer phase: the snapshot drives the sim opponent and the imagination
        opponent. PPO phase: PPO drives the sim opponent; imagination keeps the
        most recent Dreamer snapshot (it cannot run a state-based policy).
        """
        phase = _opp_schedule[index]
        if phase.kind == "dreamer":
            snapshot = _load_snapshot_agent(phase)
            _self_play_wrapper.load_opponent_agent(snapshot, device=sim_device)
            del snapshot
            imag_phase = index
        else:
            dreamer_index = last_dreamer_phase(_opp_schedule, index)
            if _opp_phase_state["current"] is None or agent._imag_opp_phase != dreamer_index:
                # Resume directly into a PPO phase: restore the imagination opponent first.
                snapshot = _load_snapshot_agent(_opp_schedule[dreamer_index])
                _self_play_wrapper.load_opponent_agent(snapshot, device=sim_device)
                del snapshot
            _self_play_wrapper.use_ppo_opponent(_load_ppo_player(phase))
            imag_phase = dreamer_index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        agent._imag_opp_phase = imag_phase
        if agent._imag_opp_rssm is not None:
            agent.sync_imag_opponent_networks()
        _self_play_wrapper.set_schedule_phase(index)
        _self_play_wrapper.reset_scores()
        _opp_phase_state["current"] = index
        # Fingerprint of the Dreamer opponent weights now loaded (sim and
        # imagination): distinct snapshots must give distinct values.
        _opp_phase_state["fingerprint"] = opponent_fingerprint(
            (
                _self_play_wrapper._opponent_encoder_orig,
                _self_play_wrapper._opponent_rssm_orig,
                _self_play_wrapper._opponent_actor_orig,
            )
        )
        print(
            f"Opponent schedule: phase {index} active ({phase.kind}: {phase.checkpoint}); "
            f"imagination opponent = phase {imag_phase}; "
            f"Dreamer opponent weight fingerprint {_opp_phase_state['fingerprint']:.6f}"
        )

    if _self_play_wrapper is not None:
        if _opp_schedule is not None:
            # The phase-0 snapshot defines the opponent network architecture.
            _snapshot0 = _load_snapshot_agent(_opp_schedule[0])
            _self_play_wrapper.set_opponent(_snapshot0, device=sim_device)
            del _snapshot0
        else:
            _self_play_wrapper.set_opponent(agent, device=sim_device)
        opp_rssm, opp_actor = _self_play_wrapper.opponent_networks
        if opp_rssm is not None and agent.opponent_separation:
            agent.set_imag_opponent_networks(opp_rssm, opp_actor, _self_play_wrapper.opponent_encoder)
    if _imag_world == "shared":
        # Our decoded "policy" state seen from the opponent's seat: the convention of obs["opponent"]
        # (swap the pegs, rotate by 180 degrees; the same in units, see opponent_seat_map).
        from klask_rl.tasks.manager_based.klask_rl.env_cfg.klask_rl_observations_cfg import opponent_seat_map

        agent.set_imag_opponent_view({"policy": opponent_seat_map(float(config.env.max_velocity))})
        print("Shared-world imagination: the imagined opponent observes our decoded state (seat map installed)")

    # Resume from the run's own <logdir>/latest.pt, or start a fine-tune from the full trainer state of a checkpoint
    # copy (finetune_from), or load weights only (init_checkpoint): the plan was made before the simulator started.
    _resume_step = 0
    checkpoint_path = logdir / "latest.pt"
    checkpoint = None
    _ft_provenance = None
    if _plan.mode in ("resume", "finetune"):
        if _plan.mode == "resume":
            print(f"Resuming from checkpoint: {checkpoint_path}")
        else:
            print(f"Fine-tune start from the full trainer state of {_plan.checkpoint} (no latest.pt in the logdir)")
        checkpoint = torch.load(_plan.checkpoint, map_location=train_device)
        if _plan.mode == "finetune":
            try:
                _source_step = check_finetune_checkpoint(checkpoint, _ft_expect)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            _ft_provenance = {"checkpoint": str(_plan.checkpoint), "source": _ft_source, "source_step": _source_step}
        else:
            _ft_provenance = checkpoint.get("finetune_from")
        # Weights, optimizer states, LR scheduler (warmup does not restart), GradScaler, slow target / EMA counters.
        _restored = load_trainer_state(agent, checkpoint, tools.recursively_load_optim_state_dict)
        _resume_step = _restored["step"]
        # Restore curriculum step so reward weight schedules continue correctly.
        _curriculum_step = _restored["curriculum_step"]
        if _curriculum_step > 0:
            env = vec_env._env
            while isinstance(env, Wrapper):
                if isinstance(env, CurriculumWrapper):
                    env._step = _curriculum_step
                    break
                env = env.env
        # Restore initialization step so velocity annealing continues correctly.
        _init_step = _restored["init_step"]
        if _init_step > 0:
            env = vec_env._env
            while isinstance(env, Wrapper):
                if isinstance(env, InitializationWrapper):
                    env._step = _init_step
                    break
                env = env.env
        print(f"  Restored agent weights, optimizer states, step={_resume_step}, curriculum_step={_curriculum_step}")
        # Sync inference copies so they pick up restored weights (old checkpoints
        # won't have _inference_* keys, so strict=False leaves them stale).
        agent._sync_inference_copies()
        if _plan.mode == "resume":
            # Restore the opponent encoder.  It is not a submodule of the agent,
            # so load_state_dict above doesn't cover it.  (The opponent rssm/actor
            # ARE submodules and were already restored above.)  Fall back to the
            # agent's encoder for older checkpoints without this key.
            if _self_play_wrapper is not None and _self_play_wrapper._opponent_encoder_orig is not None:
                opp_enc_sd = checkpoint.get("selfplay_opp_encoder_state_dict")
                if opp_enc_sd is not None:
                    _self_play_wrapper._opponent_encoder_orig.load_state_dict(opp_enc_sd)
                else:
                    _self_play_wrapper._opponent_encoder_orig.load_state_dict(agent.encoder.state_dict())
                # Propagate restored opponent weights to multi-GPU train-device
                # copies and data-parallel replicas.  No-op in single-GPU.
                agent.sync_imag_opponent_networks()
        else:
            # Fine-tune: the opponent (simulator and imagination) starts as these weights, i.e. member 0 of the fresh
            # PFSP pool, not as the source run's last opponent (whose rssm/actor the agent state dict carried).
            if (
                _opp_schedule is None
                and _self_play_wrapper is not None
                and _self_play_wrapper._opponent_encoder_orig is not None
            ):
                _self_play_wrapper._opponent_encoder_orig.load_state_dict(agent.encoder.state_dict())
                _self_play_wrapper._opponent_rssm_orig.load_state_dict(agent.rssm.state_dict())
                _self_play_wrapper._opponent_actor_orig.load_state_dict(agent.actor.state_dict())
                agent.sync_imag_opponent_networks()
            print(
                "  Fine-tune: step counter continues; fresh (empty) replay buffer, fresh PFSP pool (member 0 = these "
                "weights), empty self-play score window, opponent = these weights"
            )
    elif _plan.mode == "init":
        _wm_only = getattr(config, "load_world_model_only", False)
        tools.load_init_checkpoint(agent, _init_ckpt, train_device, world_model_only=_wm_only)
        agent._sync_inference_copies()
        # Sync ALL opponent networks from the newly initialised agent.  Unlike
        # resume (where opponent weights come from the checkpoint), init
        # checkpoint means a fresh training run so the opponent should match
        # the agent.
        if (
            _opp_schedule is None
            and _self_play_wrapper is not None
            and _self_play_wrapper._opponent_encoder_orig is not None
        ):
            _self_play_wrapper._opponent_encoder_orig.load_state_dict(agent.encoder.state_dict())
            _self_play_wrapper._opponent_rssm_orig.load_state_dict(agent.rssm.state_dict())
            _self_play_wrapper._opponent_actor_orig.load_state_dict(agent.actor.state_dict())
            agent.sync_imag_opponent_networks()

    if _hw_schedule_wrapper is not None:
        # Hardware at the start step (bound to the trainer's step once the trainer exists, below).
        _hw_schedule_wrapper.set_step(_resume_step)

    if _opp_schedule is not None:
        # Install the phase active at the (resumed) step. This also empties the
        # score window; a resumed run restores its saved window just below.
        _start_phase = phase_at(_opp_schedule, _resume_step)
        if _plan.mode == "resume" and checkpoint.get("opp_schedule_phase", _start_phase) != _start_phase:
            print(
                f"  Warning: checkpoint recorded opponent phase {checkpoint.get('opp_schedule_phase')}, "
                f"schedule gives phase {_start_phase} at step {_resume_step}; using the schedule."
            )
        _install_scheduled_opponent(_start_phase)

    if _self_play_wrapper is not None:
        # Restore score buffer from checkpoint so mean_score continues
        # correctly instead of being diluted by 4096 pre-filled zeros.
        if _plan.restore_scores:
            _saved_scores = restored_entry(_plan, checkpoint, "selfplay_score_buffer")
            if _saved_scores is not None:
                _self_play_wrapper._score_buffer.clear()
                _self_play_wrapper._score_buffer.extend(_saved_scores)
                print(
                    f"  Restored self-play score buffer ({len(_saved_scores)} entries,"
                    f" mean={_self_play_wrapper.mean_score:.3f})"
                )
        if _opp_schedule is None:
            print("Self-play enabled: opponent initialised from current agent.")
        else:
            print("Self-play enabled: fixed opponent schedule (score-gated updates disabled).")

    # --- PFSP pool of the learner's own snapshots ---
    _pfsp = None
    _pfsp_state = {"active": LATEST, "last_snapshot": 0}

    def _learner_snapshot():
        return {
            name: {k: v.detach().to("cpu", copy=True) for k, v in getattr(agent, name).state_dict().items()}
            for name in ("encoder", "rssm", "actor")
        }

    if _sp_mode == "pfsp" and _self_play_wrapper is not None:
        _pfsp = PFSPPool(
            logdir / "pfsp",
            p_latest=float(_pfsp_cfg.get("p_latest", 0.5)),
            power=float(_pfsp_cfg.get("power", 2.0)),
            floor=float(_pfsp_cfg.get("floor", 0.05)),
            prior_games=float(_pfsp_cfg.get("prior_games", 2.0)),
            seed=int(config.seed),
            score_mode=str(_pfsp_cfg.get("score_mode", "wdl")),
            selection=str(_pfsp_cfg.get("selection", "weights")),
        )
        _pfsp_snapshot_every = int(float(_pfsp_cfg.get("snapshot_every", 250_000)))
        # Optional denser snapshots early on: [[until_step, every], ...], then snapshot_every.
        _pfsp_snapshot_schedule = [(int(float(a)), int(float(b))) for a, b in (_pfsp_cfg.get("snapshot_schedule") or [])]
        _pfsp_include_initial = bool(_pfsp_cfg.get("include_initial", True))
        # Resume: the run's own pool. Otherwise (scratch, init, fine-tune) a fresh pool: member 0 = the learner's
        # initial weights (which the opponent already has), or empty with include_initial false.
        _saved_pfsp = restored_entry(_plan, checkpoint, "pfsp_state")
        _pfsp_start, _pfsp_restored = start_pfsp_pool(
            _pfsp, _saved_pfsp, _pfsp_include_initial, _resume_step, _learner_snapshot
        )
        _pfsp_state.update(_pfsp_start)
        if _pfsp_restored:
            if _pfsp_state["active"] != LATEST:
                # The checkpoint restored the opponent weights already; reinstall from
                # the pool file so the active member is exact regardless.
                _self_play_wrapper.load_opponent_state(_pfsp.load(_pfsp_state["active"]))
                agent.sync_imag_opponent_networks()
            print(f"PFSP: restored pool with {len(_pfsp.members)} members, active {_pfsp_state['active']}")
        elif _pfsp_include_initial:
            print(
                f"PFSP: pool started with the initial weights (member 0, step {_resume_step}) under {logdir / 'pfsp'}"
            )
        else:
            # No snapshot of the initial (untrained) weights: pure self-play until the first snapshot is due.
            print(f"PFSP: empty pool (include_initial false) under {logdir / 'pfsp'}; latest weights until the first snapshot")
        _self_play_wrapper.set_schedule_phase(_pfsp_state["active"])
        if _saved_pfsp is None or "selfplay_score_buffer" not in checkpoint:
            _self_play_wrapper.reset_scores()  # no zero prefill: scores are per rotation

    def _pfsp_record_window(train_step):
        """Attribute the training games since the last rotation to the active opponent."""
        wins, draws, losses = outcome_counts(list(_self_play_wrapper._score_buffer))
        active = _pfsp_state["active"]
        _pfsp.record(active, wins, draws, losses)
        games = wins + draws + losses
        log = logger.scalar
        log("pfsp/active", float(active))
        log("pfsp/window_games", float(games))
        if games:
            log("pfsp/window_score", (wins + 0.5 * draws) / games)
            log("pfsp/window_win", wins / games)
            log("pfsp/window_loss", losses / games)
        log("pfsp/pool_size", float(len(_pfsp.members)))
        log("pfsp/score_vs_latest", _pfsp.score(LATEST))
        scores = [_pfsp.score(m["id"]) for m in _pfsp.members]
        if scores:
            log("pfsp/score_vs_pool_mean", sum(scores) / len(scores))
            log("pfsp/score_vs_pool_min", min(scores))
            log("pfsp/score_vs_member0", scores[0])
            probs = [p for mid, p in _pfsp.probabilities().items() if mid != LATEST]
            log("pfsp/p_member_min", min(probs))
            log("pfsp/p_member_max", max(probs))
            log("pfsp/members_unplayed", float(sum(1 for m in _pfsp.members if _pfsp.games(m) == 0)))
        _pfsp.save_meta()

    def _pfsp_snapshot_interval(train_step):
        for until, every in _pfsp_snapshot_schedule:
            if train_step < until:
                return every
        return _pfsp_snapshot_every

    def _pfsp_rotate(train_step):
        """Snapshot the learner if due, then pick and install the next opponent."""
        if train_step - _pfsp_state["last_snapshot"] >= _pfsp_snapshot_interval(train_step):
            mid = _pfsp.add(train_step, _learner_snapshot())
            _pfsp_state["last_snapshot"] = train_step
            print(f"[step {train_step}] PFSP: snapshot member {mid}")
        choice = _pfsp.choose()
        if choice == LATEST:
            _self_play_wrapper.update_opponent(agent)
            _self_play_wrapper._restart_opponent_rssm()
        else:
            _self_play_wrapper.load_opponent_state(_pfsp.load(choice))
        agent.sync_imag_opponent_networks()
        _self_play_wrapper.set_schedule_phase(choice)
        _self_play_wrapper.reset_scores()
        _pfsp_state["active"] = choice
        probs = _pfsp.probabilities()
        print(
            f"[step {train_step}] PFSP: opponent {'latest' if choice == LATEST else f'member {choice}'} "
            f"(p={probs.get(choice, 0):.3f}; pool {len(_pfsp.members)})"
        )

    # Subclass OnlineTrainer to hook score-gated opponent updates.
    class KlaskTrainer(OnlineTrainer):
        """OnlineTrainer with self-play opponent updates.

        Reward weights and episode metrics are logged autonomously by
        :class:`RewardWeightLogWrapper` and :class:`EpisodeMetricsWrapper`.
        """

        def eval(self, agent, train_step):
            # --- Scheduled opponent switch (replaces the score-gated update) ---
            if _opp_schedule is not None:
                target = phase_at(_opp_schedule, train_step)
                switched = target != _opp_phase_state["current"]
                # Log the finished window against the outgoing opponent before switching.
                self.logger.scalar("selfplay/mean_score", _self_play_wrapper.mean_score)
                self.logger.scalar("selfplay/games_vs_current", _self_play_wrapper.games_tracked)
                if switched:
                    print(
                        f"[step {train_step}] Opponent schedule: switching phase "
                        f"{_opp_phase_state['current']} -> {target} "
                        f"(final mean_score={_self_play_wrapper.mean_score:.3f} over "
                        f"{_self_play_wrapper.games_tracked} games)"
                    )
                    _install_scheduled_opponent(target)
                self.logger.scalar("selfplay/opponent_updated", float(switched))
                self.logger.scalar("selfplay/phase", float(_opp_phase_state["current"]))
                self.logger.scalar("selfplay/imag_opp_phase", float(agent._imag_opp_phase))
                self.logger.scalar("selfplay/opp_weight_fingerprint", _opp_phase_state["fingerprint"])
            # --- PFSP: score the outgoing opponent on training games only ---
            elif _pfsp is not None:
                _pfsp_record_window(train_step)
            # --- Self-play opponent update ---
            elif _self_play_wrapper is not None:
                updated = _self_play_wrapper.maybe_update_opponent(
                    agent,
                    logger=self.logger,
                    train_step=train_step,
                )
                if updated:
                    agent.sync_imag_opponent_networks()
            # --- Debug: scan buffer for accurate per-tag episode counts ---
            if (isinstance(self.replay_buffer, PrioritizedBuffer)
                    and self.replay_buffer._debug_metrics):
                tag_counts, ep_count = self.replay_buffer.compute_current_tag_counts()
                for tag, count in tag_counts.items():
                    self.logger.scalar(f"buffer/tagged_current/{tag}", count)
                self.logger.scalar("buffer/episodes_current", ep_count)
            if hasattr(self.replay_buffer, "stored_term_rate") and _learned_prio:
                self.logger.scalar("replay/stored_term_rate", self.replay_buffer.stored_term_rate())
            if getattr(self.replay_buffer, "_mode", None) == "events":
                self.logger.scalar("replay/stored_term_rate", self.replay_buffer.stored_term_rate())
                for name, value in self.replay_buffer.event_stats().items():
                    self.logger.scalar(name, value)
            result = super().eval(agent, train_step)
            if _pfsp is not None:
                # After the evaluation games (which must not count for PFSP scores).
                _pfsp_rotate(train_step)
            return result

        def rotation_hook(self, agent, train_step):
            """With trainer.eval_episode_num 0 (no evaluation games, which cost ~20% extra env steps and cut the
            running training games): the same window bookkeeping and opponent rotation, without the games."""
            if _pfsp is not None:
                _pfsp_record_window(train_step)
                self.logger.write(train_step)
                _pfsp_rotate(train_step)

        def on_episode_end(self, episode_id: int, env_index: int) -> None:
            if not isinstance(self.replay_buffer, PrioritizedBuffer):
                return
            tm = vec_env._env.unwrapped.termination_manager
            for s in _tags_cfg_list:
                tag = s["name"]
                priority = float(s["priority"])
                matched = False
                if "termination_term" in s:
                    term = s["termination_term"]
                    if term in tm._term_names:
                        tidx = tm._term_names.index(term)
                        matched = bool(tm._term_dones[env_index, tidx].item())
                if not matched and _tag_tracker is not None and tag in _tag_tracker._step_checkers:
                    matched = _tag_tracker.get_episode_flag(env_index, tag)
                if matched:
                    self.replay_buffer.tag_episode(episode_id, priority, tag=tag)
            self.replay_buffer.flush_episode(episode_id)

        def on_log(self) -> None:
            self.logger.scalar("buffer/fill_ratio", self.replay_buffer.count() / int(config.buffer.max_size))
            if not isinstance(self.replay_buffer, PrioritizedBuffer):
                return
            if self.replay_buffer._debug_metrics:
                for tag, frac in self.replay_buffer.compute_sampled_tag_fractions().items():
                    self.logger.scalar(f"buffer/sampled_frac/{tag}", frac)

    def _save_checkpoint(step, final=False):
        """Save a full checkpoint at the given step.

        Writes both a numbered ``checkpoint_{step}.pt`` and overwrites
        ``latest.pt`` so that resume always picks up the newest one.
        """
        # Find curriculum step from wrapper chain.
        _curr_step = 0
        _env = vec_env._env
        while isinstance(_env, Wrapper):
            if isinstance(_env, CurriculumWrapper):
                _curr_step = _env._step
                break
            _env = _env.env
        # Find initialization step from wrapper chain.
        _init_step = 0
        _env = vec_env._env
        while isinstance(_env, Wrapper):
            if isinstance(_env, InitializationWrapper):
                _init_step = _env._step
                break
            _env = _env.env
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "step": step,
            "curriculum_step": _curr_step,
            "init_step": _init_step,
            "scheduler_state_dict": agent._scheduler.state_dict(),
            "scaler_state_dict": agent._scaler.state_dict(),
            "slow_value_updates": agent._slow_value_updates,
            **({"ema_updates": agent._ema_updates} if hasattr(agent, "_ema_updates") else {}),
            # Metadata only (never read back): hardware at this save, and the fine-tune source of the run.
            **({"hardware_schedule": dict(_hw_schedule_wrapper.current)} if _hw_schedule_wrapper is not None else {}),
            **({"finetune_from": _ft_provenance} if _ft_provenance is not None else {}),
            **(
                {
                    "selfplay_score_buffer": list(_self_play_wrapper._score_buffer),
                    "selfplay_opp_encoder_state_dict": _self_play_wrapper._opponent_encoder_orig.state_dict(),
                    **(
                        {"opp_schedule_phase": _opp_phase_state["current"]}
                        if _opp_phase_state["current"] is not None
                        else {}
                    ),
                    **(
                        {"pfsp_state": {"pool": _pfsp.state_dict(), **_pfsp_state}}
                        if _pfsp is not None
                        else {}
                    ),
                }
                if _self_play_wrapper is not None
                else {}
            ),
        }
        torch.save(items_to_save, logdir / f"checkpoint_{step}.pt")
        torch.save(items_to_save, logdir / "latest.pt")
        print(f"Checkpoint saved: {str(logdir.absolute())}/checkpoint_{step}.pt + latest.pt")
        # Replay buffer (~60 GB at 5e5 x 128 px x 2 views): every buffer.checkpoint_every
        # steps and at exit, so a resume continues with the same data and priorities.
        if final and not _buffer_save_when_finished and step >= _buffer_total_steps:
            # buffer.save_when_finished false (state runs): a finished run is never resumed, so its
            # replay buffer is not written at the end (a crash or stop before env.steps still saves it for the resume).
            print(f"Replay buffer not saved: run finished at {step} (buffer.save_when_finished false)")
        elif _buffer_every > 0 and hasattr(replay_buffer, "save") and (final or step - _buffer_state["last_save"] >= _buffer_every):
            _t0 = time.time()
            keep = _buffer_keep > 0 and step > 0 and step % _buffer_keep == 0
            target = logdir / (f"replay_buffer_{step}" if keep else "replay_buffer")
            if replay_buffer.save(target):
                _buffer_state["last_save"] = step
                _buffer_pointer.write_text(target.name)
                rolling = logdir / "replay_buffer"
                if keep and rolling.exists():
                    shutil.rmtree(rolling)  # superseded by the permanent snapshot
                print(f"Replay buffer saved: {target} ({time.time() - _t0:.0f} s)")

    # --- Replay buffer: restore the saved copy on resume (saved by _save_checkpoint) ---
    # Rolling copy in <logdir>/replay_buffer (every checkpoint_every steps and at exit) plus
    # permanent snapshots <logdir>/replay_buffer_<step> every keep_every steps (rewind points).
    # <logdir>/replay_buffer.latest names the newest save; resume loads that one.
    _buffer_every = int(float(getattr(config.buffer, "checkpoint_every", 1_000_000)))
    _buffer_keep = int(float(getattr(config.buffer, "keep_every", 0) or 0))
    _buffer_pointer = logdir / "replay_buffer.latest"
    _buffer_save_when_finished = bool(OmegaConf.select(config, "buffer.save_when_finished", default=True))
    _buffer_total_steps = int(float(config.env.steps))
    # Only a resume restores a buffer (its own); a fine-tune starts with an empty one (run_init.buffer_dir_to_restore).
    _buffer_dir = buffer_dir_to_restore(_plan, logdir)
    _buffer_state = {"last_save": _resume_step}
    if _buffer_dir is not None and hasattr(replay_buffer, "load"):
        _t0 = time.time()
        replay_buffer.load(_buffer_dir)
        print(
            f"  Restored replay buffer from {_buffer_dir} ({replay_buffer.count()} transitions, "
            f"{time.time() - _t0:.0f} s; new episode ids from {replay_buffer.episode_id_offset})"
        )
    elif _plan.restore_buffer:
        print("  No saved replay buffer: resuming with an empty buffer")
    elif _plan.mode == "finetune":
        print(f"  Fine-tune: empty replay buffer ({replay_buffer.count()} transitions)")

    policy_trainer = KlaskTrainer(
        config.trainer,
        replay_buffer,
        logger,
        logdir,
        train_stepper=train_envs,
        eval_stepper=eval_envs,
        initial_step=_resume_step,
        save_fn=_save_checkpoint,
    )
    if _hw_schedule_wrapper is not None:
        # The schedule reads the trainer's step (absolute env steps, as in checkpoint names; restored on resume).
        _hw_schedule_wrapper.bind_step(lambda: policy_trainer._step)
        print(f"Hardware schedule bound to the trainer step ({policy_trainer._step}): {_hw_schedule_wrapper.current}")
    if _opp_schedule is not None and _resume_step > 0:
        # Keep opponent switches on the schedule grid after a resume (the
        # trainer would otherwise evaluate eval_every steps after the resume step).
        policy_trainer._should_eval._last = eval_anchor(_resume_step, config.trainer.eval_every)

    exit_code = 0
    try:
        policy_trainer.begin(agent)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
        exit_code = 1
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"TRAINING CRASHED: {type(e).__name__}: {e}")
        print(f"{'='*60}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        _save_checkpoint(policy_trainer._step, final=True)

        # Dump cProfile data before simulation_app.close() kills the process.
        if _PROFILER is not None:
            _PROFILER.disable()
            prof_path = os.environ.get("PROFILE_OUTPUT", "train_profile.prof")
            _PROFILER.dump_stats(prof_path)
            stats = pstats.Stats(_PROFILER)
            stats.sort_stats("cumulative")
            stats.print_stats(40)
            print(f"\nFull profile saved to: {prof_path}")

        logger.close(exit_code=exit_code)
        vec_env._env.close()
        simulation_app.close()


_PROFILER = None

if __name__ == "__main__":
    # Forward only the Hydra-style args (everything after AppLauncher args).
    sys.argv = [sys.argv[0]] + hydra_args

    if os.environ.get("PROFILE", ""):
        _PROFILER = cProfile.Profile()
        _PROFILER.enable()

    main()
