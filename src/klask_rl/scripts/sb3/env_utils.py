"""Shared environment setup utilities for SAC training and evaluation.

This module contains all the common logic for building the Isaac Lab environment
with the correct wrappers (HER, Two-Stage HER, VecNormalize, etc.) so that
train_sac.py and play_sac.py stay in sync without code duplication.

The lightweight helpers (resolve_config_path, get_log_root_path, find_latest_checkpoint)
can be imported BEFORE AppLauncher.  The heavy helpers (make_env, wrap_env_for_sb3, …)
import isaaclab lazily and must only be called AFTER AppLauncher has been initialised.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # These imports are only used for type-checking / IDE support.
    # At runtime they are imported lazily inside the functions that need them.
    import gymnasium as gym

    from experiment_config import ExperimentConfig
    from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg


# ---------------------------------------------------------------------------
# Config path resolution
# ---------------------------------------------------------------------------


def resolve_config_path(raw_path: str, script_file: str) -> Path:
    """Resolve an experiment config path relative to scripts/sb3/config/.

    Args:
        raw_path: The path string from --config (e.g. "experiments/klask_sac_her.yaml").
        script_file: ``__file__`` of the calling script, used to locate the config dir.

    Returns:
        Resolved absolute Path to the YAML config.
    """
    config_dir = Path(script_file).parent / "config"
    if Path(raw_path).is_absolute():
        return Path(raw_path)
    if raw_path.startswith("experiments/") or "/" in raw_path or "\\" in raw_path:
        return config_dir / raw_path
    # Check experiments folder first, then root config folder
    if (config_dir / "experiments" / raw_path).exists():
        return config_dir / "experiments" / raw_path
    return config_dir / raw_path


# ---------------------------------------------------------------------------
# Log root path
# ---------------------------------------------------------------------------


def get_log_root_path(cfg: ExperimentConfig, script_file: str) -> str:
    """Return the root log directory for a task (e.g. ``logs/sb3_sac/<task>``)."""
    script_dir = Path(
        script_file
    ).parent.parent.parent  # Go up from scripts/sb3/ to klask_rl/
    return str(script_dir / "logs" / "sb3_sac" / cfg.task)


# ---------------------------------------------------------------------------
# Reward weight overrides
# ---------------------------------------------------------------------------


def apply_reward_weights(
    cfg: ExperimentConfig,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
) -> None:
    """Apply reward weights from the experiment config to the env config.

    Modifies *env_cfg* in-place.
    """
    if cfg.use_her and cfg.her_env_reward_scale is not None:
        if hasattr(env_cfg, "rewards"):
            if hasattr(env_cfg.rewards, "collision_player_ball_reward"):
                print(
                    f"[INFO] Setting env collision_player_ball_reward weight to {cfg.her_env_reward_scale}"
                )
                env_cfg.rewards.collision_player_ball_reward.weight = (
                    cfg.her_env_reward_scale
                )

    if cfg.use_two_stage_her:
        if cfg.two_stage_ball_hit_env_reward is not None:
            if hasattr(env_cfg, "rewards") and hasattr(
                env_cfg.rewards, "collision_player_ball"
            ):
                print(
                    f"[INFO] Setting env collision_player_ball weight to {cfg.two_stage_ball_hit_env_reward}"
                )
                env_cfg.rewards.collision_player_ball.weight = (
                    cfg.two_stage_ball_hit_env_reward
                )
        if cfg.two_stage_goal_score_env_reward is not None:
            if hasattr(env_cfg, "rewards") and hasattr(env_cfg.rewards, "goal_scored"):
                print(
                    f"[INFO] Setting env goal_scored weight to {cfg.two_stage_goal_score_env_reward}"
                )
                env_cfg.rewards.goal_scored.weight = cfg.two_stage_goal_score_env_reward


# ---------------------------------------------------------------------------
# Full environment creation + wrapping pipeline
# ---------------------------------------------------------------------------


def make_env(
    cfg: ExperimentConfig,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    *,
    render_mode: str | None = None,
    video_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    """Create the Isaac environment, apply MARL conversion and action space bounds.

    Does **not** apply SB3 wrappers (VecEnv / HER / VecNormalize) — see
    :func:`wrap_env_for_sb3` for that.

    Args:
        cfg: Experiment configuration.
        env_cfg: Environment configuration (from Hydra).
        render_mode: Gym render mode (``"rgb_array"`` for video, else ``None``).
        video_kwargs: If provided, wraps with ``gym.wrappers.RecordVideo``.

    Returns:
        A Gymnasium environment ready to be wrapped for SB3.
    """
    import gymnasium as gym
    import numpy as np

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict

    env = gym.make(cfg.task, cfg=env_cfg, render_mode=render_mode)

    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Set bounded action space
    action_dim = env.unwrapped.single_action_space.shape[-1]
    max_vel = cfg.max_velocity
    print(f"[INFO] Setting action space bounds to [-{max_vel}, {max_vel}] m/s")
    env.unwrapped.single_action_space = gym.spaces.Box(
        low=-max_vel, high=max_vel, shape=(action_dim,), dtype=np.float32
    )
    env.unwrapped.action_space = gym.vector.utils.batch_space(
        env.unwrapped.single_action_space, env.unwrapped.num_envs
    )

    # Video recording
    if video_kwargs is not None:
        print("[INFO] Recording videos.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    return env


def wrap_env_for_sb3(
    env: gym.Env,
    cfg: ExperimentConfig,
) -> gym.Env:
    """Apply the SB3-specific wrappers: VecEnv → HER / Two-Stage HER.

    Args:
        env: A Gymnasium environment (typically from :func:`make_env`).
        cfg: Experiment configuration.

    Returns:
        A wrapped SB3-compatible vectorized environment.
    """
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    from klask_rl.tasks.manager_based.klask_rl.wrappers import (
        Sb3TwoStageHerWrapper,
        Sb3VecHerWrapper,
    )

    # Stable-Baselines3 vectorized wrapper
    env = Sb3VecEnvWrapper(env, fast_variant=True)

    # HER wrappers
    if cfg.use_two_stage_her:
        print("[INFO] Wrapping environment with Two-Stage HER wrapper...")
        player_pos_indices = tuple(cfg.two_stage_player_pos_indices or [0, 2])
        ball_pos_indices = tuple(cfg.two_stage_ball_pos_indices or [8, 10])
        goal_pos_indices = tuple(cfg.two_stage_goal_pos_indices or [12, 14])

        print(f"[INFO] Two-Stage HER player_pos indices: {player_pos_indices}")
        print(f"[INFO] Two-Stage HER ball_pos indices: {ball_pos_indices}")
        print(f"[INFO] Two-Stage HER goal_pos indices: {goal_pos_indices}")
        print(
            f"[INFO] Two-Stage HER ball_hit_threshold: {cfg.two_stage_ball_hit_threshold}"
        )
        print(
            f"[INFO] Two-Stage HER goal_score_threshold: {cfg.two_stage_goal_score_threshold}"
        )
        print(
            f"[INFO] Two-Stage HER ball_hit_wrapper_reward: {cfg.two_stage_ball_hit_wrapper_reward}"
        )
        print(
            f"[INFO] Two-Stage HER goal_score_wrapper_reward: {cfg.two_stage_goal_score_wrapper_reward}"
        )

        env = Sb3TwoStageHerWrapper(
            env,
            player_pos_indices=player_pos_indices,
            ball_pos_indices=ball_pos_indices,
            goal_pos_indices=goal_pos_indices,
            ball_hit_threshold=cfg.two_stage_ball_hit_threshold,
            goal_score_threshold=cfg.two_stage_goal_score_threshold,
            ball_hit_reward=cfg.two_stage_ball_hit_wrapper_reward,
            goal_score_reward=cfg.two_stage_goal_score_wrapper_reward,
        )
    elif cfg.use_her:
        print(
            "[INFO] Wrapping environment with HER (Hindsight Experience Replay) wrapper..."
        )
        achieved_indices = tuple(cfg.her_achieved_goal_indices or [0, 2])
        desired_indices = tuple(cfg.her_desired_goal_indices or [8, 10])

        print(f"[INFO] HER achieved_goal indices: {achieved_indices}")
        print(f"[INFO] HER desired_goal indices: {desired_indices}")
        print(f"[INFO] HER distance threshold: {cfg.her_distance_threshold}")
        print(f"[INFO] HER wrapper reward scale: {cfg.her_wrapper_reward_scale}")

        env = Sb3VecHerWrapper(
            env,
            achieved_goal_indices=achieved_indices,
            desired_goal_indices=desired_indices,
            distance_threshold=cfg.her_distance_threshold,
            reward_scale=cfg.her_wrapper_reward_scale,
        )

    return env


# ---------------------------------------------------------------------------
# VecNormalize helpers
# ---------------------------------------------------------------------------


def pop_norm_keys(agent_cfg: dict) -> dict[str, Any]:
    """Pop normalization-related keys from *agent_cfg* and return them."""
    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    return {k: agent_cfg.pop(k) for k in norm_keys if k in agent_cfg}


def apply_training_normalization(
    env,
    cfg: ExperimentConfig,
    norm_args: dict[str, Any],
    gamma: float = 0.99,
):
    """Wrap *env* with ``VecNormalize`` for training if normalization is enabled.

    Returns the (possibly wrapped) environment.
    """
    import numpy as np

    from stable_baselines3.common.vec_env import VecNormalize

    if not (norm_args and norm_args.get("normalize_input")):
        return env

    if cfg.use_her or cfg.use_two_stage_her:
        print(
            "[WARNING] VecNormalize is not fully compatible with HER. "
            "Disabling observation normalization."
        )
        return env

    print(f"[INFO] Normalizing input, {norm_args=}")
    return VecNormalize(
        env,
        training=True,
        norm_obs=norm_args["normalize_input"],
        norm_reward=norm_args.get("normalize_value", False),
        clip_obs=norm_args.get("clip_obs", 100.0),
        gamma=gamma,
        clip_reward=np.inf,
    )


def apply_inference_normalization(
    env,
    cfg: ExperimentConfig,
    norm_args: dict[str, Any],
    checkpoint_path: str,
    gamma: float = 0.99,
):
    """Load or create ``VecNormalize`` for inference.

    Tries to load saved normalization stats from the checkpoint directory.
    Falls back to creating a non-updating wrapper if normalization was
    configured but no saved stats exist.

    Returns the (possibly wrapped) environment.
    """
    import numpy as np

    from stable_baselines3.common.vec_env import VecNormalize

    vec_norm_path = Path(checkpoint_path).parent / "sac_model_vecnormalize.pkl"
    if vec_norm_path.exists():
        print(f"[INFO] Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        env.training = False
        env.norm_reward = False
        return env

    if norm_args and norm_args.get("normalize_input"):
        if cfg.use_her or cfg.use_two_stage_her:
            return env  # HER + VecNormalize not supported
        print("[INFO] Applying observation normalization (no saved stats found).")
        return VecNormalize(
            env,
            training=False,
            norm_obs=norm_args["normalize_input"],
            norm_reward=False,
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=gamma,
            clip_reward=np.inf,
        )

    return env


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def find_latest_checkpoint(log_root_path: str) -> str:
    """Find the latest checkpoint in the log directory for this task.

    Searches through all run directories under *log_root_path* (sorted by
    timestamp name, most recent first) and returns the first checkpoint found.

    Priority per run directory:
        1. ``sac_model_final.zip``
        2. The ``sac_model_*_steps.zip`` with the highest step count.

    Args:
        log_root_path: Root log directory, e.g. ``logs/sb3_sac/Klask-Rl-HER-SAC-v0``.

    Returns:
        Absolute path to the latest ``.zip`` checkpoint.

    Raises:
        FileNotFoundError: If no checkpoints are found.
    """
    log_root = Path(log_root_path)
    if not log_root.exists():
        raise FileNotFoundError(f"Log directory not found: {log_root}")

    run_dirs = sorted(
        [d for d in log_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )

    for run_dir in run_dirs:
        final_model = run_dir / "sac_model_final.zip"
        if final_model.exists():
            return str(final_model)

        step_checkpoints = sorted(
            run_dir.glob("sac_model_*_steps.zip"),
            key=_extract_step_number,
            reverse=True,
        )
        if step_checkpoints:
            return str(step_checkpoints[0])

    raise FileNotFoundError(
        f"No checkpoints found in any run directory under: {log_root}"
    )


def _extract_step_number(path: Path) -> int:
    """Extract the step number from e.g. ``sac_model_40960000_steps.zip``."""
    try:
        parts = path.stem.split("_")
        for i, part in enumerate(parts):
            if part == "steps" and i > 0:
                return int(parts[i - 1])
    except (ValueError, IndexError):
        pass
    return 0
