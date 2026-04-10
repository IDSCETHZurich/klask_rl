# =============================================================================
# Phase 1: AppLauncher must run before any IsaacLab / USD / warp imports.
# =============================================================================

import argparse
import cProfile
import os
import pathlib
import pstats
import sys

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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Phase 2: Everything else — safe to import now that the sim is running.
# =============================================================================

import atexit
import signal
import warnings

import hydra
import torch
from omegaconf import OmegaConf

# Isaac Sim may override SIGINT; restore Python's default so Ctrl-C works.
signal.signal(signal.SIGINT, signal.default_int_handler)

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

# Register IsaacLab task environments (triggers gymnasium gym.register calls).
import isaaclab_tasks  # noqa: F401
import klask_rl.tasks  # noqa: F401
import tools
from buffer import Buffer, PrioritizedBuffer
from dreamer import Dreamer

# Self-play wrapper (lives outside the r2dreamer submodule)
from dreamer_self_play import DreamerSelfPlayWrapper
from env_cfg_utils import apply_camera_size_to_env_cfg
from envs import make_envs
from envs.isaaclab import IsaacLabVecEnv
from gymnasium import Wrapper
from isaaclab.sim import RenderCfg
from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    CurriculumWrapper,
    InitializationWrapper,
    KlaskRlRandomOpponentWrapper,
    OpponentActionWrapper,
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

            for term_name, window in self._reward_history.items():
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
                self._logger.scalar(f"rewards/weights/{term}", val)
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


def _make_env(
    config, gym_id, render_mode=None, trainer_steps=None, self_play=False, self_play_config=None, prioritized_cfg=None,
    trajectory_mirroring=False,
):
    """Construct a GPU-resident IsaacLab env with KLASK-specific wrappers.

    Applies (in order):
      1. Bounded action space (max_velocity)
      2. ActuatorModelWrapper (if config.actuator_model)
      3. CurriculumWrapper (if config.rewards)
      4. Termination filtering (if config.terminations)
      5. Opponent wrapper:
         - DreamerSelfPlayWrapper if self_play=True
         - KlaskRlRandomOpponentWrapper otherwise
      5b. EpisodeTagTracker (if buffer.prioritized configured)
      6. EpisodeMetricsWrapper (episode termination logging)
      7. RewardWeightLogWrapper (reward weight logging)
    """
    global _self_play_wrapper, _tag_tracker
    import importlib

    import gymnasium as gym

    env_cfg_entry = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    if isinstance(env_cfg_entry, str):
        module_name, class_name = env_cfg_entry.rsplit(":", 1)
        env_cfg_class = getattr(importlib.import_module(module_name), class_name)
    else:
        env_cfg_class = env_cfg_entry

    env_cfg = env_cfg_class()

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

    sim_dt = getattr(config, "sim_dt", None)
    if sim_dt is not None:
        env_cfg.sim.dt = float(sim_dt)

    env_cfg.scene.num_envs = int(config.env_num)
    env_cfg.decimation = int(config.decimation)
    env_cfg.seed = int(config.seed)
    env_cfg.episode_length_s = config.episode_length_s

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

    # --- 1a. Opponent action frame transform (innermost) ---
    isaac_env = OpponentActionWrapper(isaac_env)

    # --- 1. Set bounded action space ---
    # Scale the IsaacLab action terms so that the agent's [-1, 1] output
    # maps to [-max_velocity, +max_velocity] m/s.  Simply overriding the
    # gym action-space metadata is NOT enough — Dreamer's actor always
    # outputs in [-1, 1] regardless of the reported space bounds.
    max_velocity = getattr(config, "max_velocity", None)
    if max_velocity is not None:
        vel = float(max_velocity)
        # Set the scale on every JointVelocityAction term in the action manager
        # so that raw_action * scale produces the desired velocity in m/s.
        action_mgr = isaac_env.unwrapped.action_manager
        for term in action_mgr._terms.values():
            term._scale = vel

    # --- 2. Actuator model wrapper ---
    if getattr(config, "actuator_model", False):
        isaac_env = ActuatorModelWrapper(isaac_env)

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
    self_play = getattr(config, "self_play", False)

    # Attach self_play_config to env config so _make_env can read it.
    sp_cfg = (
        OmegaConf.to_container(config.get("self_play_config", OmegaConf.create({})), resolve=True) if self_play else {}
    )
    # Mirror the model compile flag so the opponent is also compiled when the
    # training agent is.
    if self_play:
        sp_cfg.setdefault("compile", bool(getattr(config.model, "compile", False)))

    _prioritized_cfg = getattr(config.buffer, "prioritized", None)
    _use_prioritized = _prioritized_cfg is not None and bool(getattr(_prioritized_cfg, "enable", False))
    if _use_prioritized:
        _tags_raw = getattr(_prioritized_cfg, "tags", [])
        _tags_cfg_list = (
            list(OmegaConf.to_container(_tags_raw, resolve=True)) if OmegaConf.is_config(_tags_raw) else list(_tags_raw)
        )
    else:
        _tags_cfg_list = []

    vec_env = _make_env(
        config.env,
        task_name,
        render_mode,
        trainer_steps=trainer_steps,
        self_play=self_play,
        self_play_config=sp_cfg,
        prioritized_cfg=_prioritized_cfg,
        trajectory_mirroring=getattr(config, "trajectory_mirroring", False),
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

    # Derive buffer.mirror from the single trajectory_mirroring flag.
    _traj_mirror = getattr(config, "trajectory_mirroring", False)
    OmegaConf.update(config, "buffer.mirror", _traj_mirror, force_add=True)

    if _use_prioritized:
        replay_buffer = PrioritizedBuffer(config.buffer)
    else:
        replay_buffer = Buffer(config.buffer)

    print("Create env.")
    # OmegaConf configs are read-only; use a plain object to pass the env through.
    env_config = OmegaConf.to_container(config.env, resolve=True)
    env_config = type("EnvConfig", (), env_config)()
    env_config.isaac_vec_env = vec_env
    train_envs, eval_envs, obs_space, act_space = make_envs(env_config)

    # Pass top-level flags to model config so Dreamer / _make_env can read them.
    _opp_sep = getattr(config, "opponent_separation", False)
    config.model.opponent_separation = _opp_sep
    config.model.trajectory_mirroring = getattr(config, "trajectory_mirroring", False)
    _opp_sep_cfg = getattr(config, "opponent_separation_config", None)
    if _opp_sep_cfg is not None:
        if OmegaConf.is_config(_opp_sep_cfg):
            _opp_sep_dict = OmegaConf.to_container(_opp_sep_cfg, resolve=True)
        else:
            _opp_sep_dict = dict(_opp_sep_cfg)
        config.model.imag_opponent = _opp_sep_dict.get("imag_opponent", "random")
    else:
        config.model.imag_opponent = "random"

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    # Validate init_checkpoint path early.
    _init_ckpt = config.init_checkpoint

    # Resume from checkpoint if one exists in the logdir.
    _resume_step = 0
    checkpoint_path = logdir / "latest.pt"
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        _missing, _unexpected = agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
        if _missing or _unexpected:
            print(f"  Checkpoint key mismatch: {len(_missing)} missing, {len(_unexpected)} unexpected")
            if _missing:
                print(f"    Missing: {_missing}")
            if _unexpected:
                print(f"    Unexpected: {_unexpected}")
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        _resume_step = checkpoint.get("step", 0)
        # Restore curriculum step so reward weight schedules continue correctly.
        _curriculum_step = checkpoint.get("curriculum_step", 0)
        if _curriculum_step > 0:
            env = vec_env._env
            while isinstance(env, Wrapper):
                if isinstance(env, CurriculumWrapper):
                    env._step = _curriculum_step
                    break
                env = env.env
        # Restore initialization step so velocity annealing continues correctly.
        _init_step = checkpoint.get("init_step", 0)
        if _init_step > 0:
            env = vec_env._env
            while isinstance(env, Wrapper):
                if isinstance(env, InitializationWrapper):
                    env._step = _init_step
                    break
                env = env.env
        # Restore LR scheduler state so warmup doesn't restart.
        if "scheduler_state_dict" in checkpoint:
            agent._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Restore GradScaler state so AMP scale factor continues correctly.
        if "scaler_state_dict" in checkpoint:
            agent._scaler.load_state_dict(checkpoint["scaler_state_dict"])
        # Restore slow target update counter.
        if "slow_value_updates" in checkpoint:
            agent._slow_value_updates = checkpoint["slow_value_updates"]
        # Restore DreamerPro EMA update counter (only present with dreamerpro config).
        if "ema_updates" in checkpoint and hasattr(agent, "_ema_updates"):
            agent._ema_updates = checkpoint["ema_updates"]
        print(f"  Restored agent weights, optimizer states, step={_resume_step}, curriculum_step={_curriculum_step}")
    elif _init_ckpt is not None:
        _wm_only = getattr(config, "load_world_model_only", False)
        tools.load_init_checkpoint(agent, _init_ckpt, config.device, world_model_only=_wm_only)

    # Initialise self-play opponent from the (randomly initialised) agent.
    if _self_play_wrapper is not None:
        _self_play_wrapper.set_opponent(agent)
        # Restore score buffer from checkpoint so mean_score continues
        # correctly instead of being diluted by 4096 pre-filled zeros.
        if checkpoint_path.exists():
            _saved_scores = checkpoint.get("selfplay_score_buffer")
            if _saved_scores is not None:
                _self_play_wrapper._score_buffer.clear()
                _self_play_wrapper._score_buffer.extend(_saved_scores)
                print(
                    f"  Restored self-play score buffer ({len(_saved_scores)} entries,"
                    f" mean={_self_play_wrapper.mean_score:.3f})"
                )
        print("Self-play enabled: opponent initialised from current agent.")

    # Subclass OnlineTrainer to hook score-gated opponent updates.
    from collections import deque as _deque

    _current_episodes: int = 0
    _episode_count_queue: _deque[int] = _deque()
    _buffer_capacity: int = int(config.buffer.max_size)

    # TODO: Remove later
    # ==============================================
    # --- Periodic profiling state ---
    _profile_state = {
        "every": 100_000,      # Profile every N steps
        "window": 10_000,      # Each profile captures this many steps
        "profiler": None,      # Active cProfile.Profile or None
        "start_step": 0,       # Step when current profile window started
    }
    # ==============================================

    class KlaskTrainer(OnlineTrainer):
        """OnlineTrainer with self-play opponent updates.

        Reward weights and episode metrics are logged autonomously by
        :class:`RewardWeightLogWrapper` and :class:`EpisodeMetricsWrapper`.
        """

        def eval(self, agent, train_step):
            # --- Self-play opponent update ---
            if _self_play_wrapper is not None:
                _self_play_wrapper.maybe_update_opponent(
                    agent,
                    logger=self.logger,
                    train_step=train_step,
                )
            # --- Debug: scan buffer for accurate per-tag episode counts ---
            if (isinstance(self.replay_buffer, PrioritizedBuffer)
                    and self.replay_buffer._debug_metrics):
                tag_counts, ep_count = self.replay_buffer.compute_current_tag_counts()
                for tag, count in tag_counts.items():
                    self.logger.scalar(f"buffer/tagged_current/{tag}", count)
                self.logger.scalar("buffer/episodes_current_scan", ep_count)
            return super().eval(agent, train_step)

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
            # Track approximate current episode count in buffer.
            nonlocal _current_episodes
            _current_episodes += 1
            _episode_count_queue.append(self.replay_buffer._transitions_added)
            while _episode_count_queue:
                if self.replay_buffer._transitions_added - _episode_count_queue[0] >= _buffer_capacity:
                    _episode_count_queue.popleft()
                    _current_episodes = max(0, _current_episodes - 1)
                else:
                    break

        def on_log(self) -> None:
            self.logger.scalar("buffer/fill_ratio", self.replay_buffer.count() / int(config.buffer.max_size))
            self.logger.scalar("buffer/episodes_current", _current_episodes)

            # TODO: Remove later
            # ==============================================
            # --- Periodic cProfile snapshots ---
            ps = _profile_state
            step = self._step
            if ps["profiler"] is not None:
                # Active profile window — check if we've captured enough steps.
                if step - ps["start_step"] >= ps["window"]:
                    ps["profiler"].disable()
                    prof_dir = logdir / "profiles"
                    prof_dir.mkdir(exist_ok=True)
                    prof_path = prof_dir / f"profile_{ps['start_step']}.prof"
                    ps["profiler"].dump_stats(str(prof_path))
                    print(f"\n=== Profile snapshot @ step {ps['start_step']} "
                          f"({ps['window']} steps) saved to {prof_path} ===")
                    st = pstats.Stats(ps["profiler"])
                    st.sort_stats("cumulative")
                    st.print_stats(30)
                    ps["profiler"] = None
            elif step > 0 and step % ps["every"] < ps["window"]:
                # Time to start a new profiling window.
                ps["profiler"] = cProfile.Profile()
                ps["start_step"] = step
                ps["profiler"].enable()
                print(f"\n=== Starting profile snapshot @ step {step} ===")
            # ==============================================

            if not isinstance(self.replay_buffer, PrioritizedBuffer):
                return
            if self.replay_buffer._debug_metrics:
                for tag, frac in self.replay_buffer.compute_sampled_tag_fractions().items():
                    self.logger.scalar(f"buffer/sampled_frac/{tag}", frac)

    def _save_checkpoint(step):
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
            **(
                {"selfplay_score_buffer": list(_self_play_wrapper._score_buffer)}
                if _self_play_wrapper is not None
                else {}
            ),
        }
        torch.save(items_to_save, logdir / f"checkpoint_{step}.pt")
        torch.save(items_to_save, logdir / "latest.pt")
        print(f"Checkpoint saved: {str(logdir.absolute())}/checkpoint_{step}.pt + latest.pt")

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
        _save_checkpoint(policy_trainer._step)

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
