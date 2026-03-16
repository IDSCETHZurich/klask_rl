"""Script to train RL agent with Stable Baselines3 SAC.

This script uses a unified configuration system where all hyperparameters
are loaded from a single YAML file. Uses Hydra for parameter overrides,
enabling the same mechanism for both standalone training and Ray tuning.

Usage:
    python train_sac.py --config experiments/klask_sac_base.yaml
    python train_sac.py --config experiments/klask_sac_her.yaml
    python train_sac.py -c experiments/klask_sac_two_stage_her.yaml
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# Parse config file argument BEFORE any isaaclab imports
parser = argparse.ArgumentParser(description="Train SAC agent", add_help=False)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="experiments/klask_sac_base.yaml",
    help="Config file path relative to scripts/sb3/config/ (default: experiments/klask_sac_base.yaml)",
)
args, remaining_argv = parser.parse_known_args()

# Resolve config path
from env_utils import resolve_config_path, get_log_root_path

CONFIG_PATH = resolve_config_path(args.config, __file__)

# Load config and set CUDA_VISIBLE_DEVICES BEFORE importing isaaclab
from experiment_config import ExperimentConfig

CONFIG = ExperimentConfig.from_file(CONFIG_PATH)

# Setup CUDA visibility (handles both normal training and Ray Tune cases)
CONFIG.setup_cuda_visibility()

# Print experiment info EARLY for Ray Tune to detect
# (must be before Isaac Sim startup which produces lots of output)
run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Use absolute path relative to script location (not cwd) for Ray Tune compatibility
log_root_path = get_log_root_path(CONFIG, __file__)
print(f"[INFO] Logging experiment in directory: {log_root_path}")
print(f"Exact experiment name requested from command line: {run_info}")

# IMPORTANT: Set Hydra CLI overrides BEFORE importing isaaclab
# This is the key for unified config handling - both standalone and Ray
# use the same Hydra override mechanism

# Apply her.* and two_stage.* overrides directly to CONFIG (they bypass Hydra)
# This lets Ray Tune vary HER parameters like her.n_sampled_goal
remaining_argv = CONFIG.apply_cli_overrides(remaining_argv)

# Extract Ray Tune parameter keys from remaining_argv to avoid duplicates
# and apply overrides to CONFIG so to_dict() reflects actual values
# (needed for accurate wandb logging)
ray_tune_keys = set()

# Mapping from Hydra override keys to CONFIG attributes.
# get_hydra_overrides() generates Hydra keys like "env.scene.num_envs" from
# CONFIG attributes like "num_envs". When Ray Tune provides these as CLI
# overrides, we need to map them back to update CONFIG for accurate logging.
_HYDRA_KEY_TO_CONFIG_ATTR = {
    # env overrides (Hydra key → CONFIG attribute name)
    "env.scene.num_envs": "num_envs",
    "env.episode_length_s": "episode_length_s",
    "env.seed": "seed",
    # env fields not in get_hydra_overrides but in the YAML
    "env.max_velocity": "max_velocity",
}


def _apply_cli_override_to_config(key: str, raw_value: str) -> None:
    """Apply a single CLI override to CONFIG so to_dict() is accurate.

    Handles:
    - env.* keys via _HYDRA_KEY_TO_CONFIG_ATTR (mapped to top-level CONFIG attrs)
    - agent.* keys → CONFIG.agent_cfg (nested dict, e.g. policy_kwargs.net_arch)
    - her.* and two_stage.* are already handled by apply_cli_overrides() upstream
    """
    parsed_value = ExperimentConfig._parse_cli_value(raw_value)

    # Direct attribute mappings (Hydra key → CONFIG attribute)
    if key in _HYDRA_KEY_TO_CONFIG_ATTR:
        setattr(CONFIG, _HYDRA_KEY_TO_CONFIG_ATTR[key], parsed_value)
        return

    # agent.* → CONFIG.agent_cfg (nested dict)
    if key.startswith("agent."):
        agent_subkey = key[len("agent.") :]
        parts = agent_subkey.split(".")
        target = CONFIG.agent_cfg
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = parsed_value


for arg in remaining_argv:
    # Parse Hydra overrides like "agent.policy_kwargs.net_arch=[256,128,64]"
    if "=" in arg:
        key, raw_value = arg.split("=", 1)
        key = key.strip("'\"")
        ray_tune_keys.add(key)
        _apply_cli_override_to_config(key, raw_value)

hydra_overrides = CONFIG.get_hydra_overrides(exclude_keys=ray_tune_keys)
sys.argv = [sys.argv[0]] + hydra_overrides + remaining_argv
if remaining_argv:
    print(f"[INFO] Extra Hydra overrides: {remaining_argv}")
print(f"[INFO] Hydra overrides: {hydra_overrides}")

# NOW import isaaclab after CUDA_VISIBLE_DEVICES is set and sys.argv is configured
from isaaclab.app import AppLauncher
from utils import cleanup_pbar

# launch omniverse app
app_launcher = AppLauncher(CONFIG.app_launcher_args())
simulation_app = app_launcher.app

# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import random
from datetime import datetime

import omni
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

# Optional wandb import
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback

    WANDB_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    # TypeError can occur with pydantic/inspect issues when wandb tries to inspect isaaclab module
    # Catch all exceptions to ensure wandb issues don't crash training
    print(f"[WARNING] wandb import failed: {e}")
    WANDB_AVAILABLE = False

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import klask_rl.tasks  # noqa: F401

from env_utils import (
    apply_reward_weights,
    apply_training_normalization,
    make_env,
    pop_norm_keys,
    wrap_env_for_sb3,
)


def build_wandb_config(
    cfg: ExperimentConfig,
    remaining_argv: list[str],
    hydra_overrides: list[str],
) -> dict:
    """Build complete config dictionary for wandb logging.

    Captures the three sources of truth for full reproducibility:
    1. Effective config (YAML base + CLI overrides already merged into CONFIG)
    2. CLI overrides (e.g., from Ray Tune or command line) for reference
    3. Hydra overrides (generated from experiment config) for reference

    Note: agent.* CLI overrides (e.g. from Ray Tune) are pre-merged into
    CONFIG.agent_cfg before this function is called, so the top-level
    config reflects the actual values used for training.

    Returns a nested dictionary suitable for wandb.init(config=...).
    """
    # Start with the base experiment config from YAML (includes ALL fields)
    wandb_config = cfg.to_dict()

    # Add metadata for easy filtering and identification
    wandb_config["algorithm"] = "SAC"
    wandb_config["config_file"] = str(CONFIG_PATH)
    wandb_config["command"] = " ".join(sys.orig_argv)

    # Add overrides for full reproducibility
    wandb_config["overrides"] = {
        "cli": remaining_argv if remaining_argv else [],
        "hydra": hydra_overrides if hydra_overrides else [],
    }

    return wandb_config


class TwoStageHerMetricsCallback(BaseCallback):
    """Callback to track and log two-stage HER goal achievements.

    Tracks:
    - Number of envs (out of total) that hit the ball (goal 1) per rollout
    - Number of envs (out of total) that scored a goal (goal 2) per rollout
    """

    def __init__(self, num_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self.num_envs = num_envs
        self.envs_hit_ball = set()
        self.envs_scored_goal = set()

    def _on_step(self) -> bool:
        if "infos" in self.locals:
            infos = self.locals["infos"]

            for i, info in enumerate(infos):
                if info.get("ball_hit", False) or info.get("terminal_ball_hit", False):
                    self.envs_hit_ball.add(i)

                if info.get("goal_scored", False) or info.get(
                    "terminal_goal_scored", False
                ):
                    self.envs_scored_goal.add(i)

        return True

    def _on_rollout_end(self) -> None:
        ball_hits_count = len(self.envs_hit_ball)
        goal_scores_count = len(self.envs_scored_goal)

        ball_hit_rate = ball_hits_count / self.num_envs if self.num_envs > 0 else 0
        goal_score_rate = goal_scores_count / self.num_envs if self.num_envs > 0 else 0

        self.logger.record("two_stage/ball_hits_count", ball_hits_count)
        self.logger.record("two_stage/goal_scores_count", goal_scores_count)
        self.logger.record("two_stage/total_envs", self.num_envs)
        self.logger.record("two_stage/ball_hit_rate", ball_hit_rate)
        self.logger.record("two_stage/goal_score_rate", goal_score_rate)

        if self.verbose > 0:
            print(
                f"[TwoStageMetrics] Envs that hit ball: {ball_hits_count}/{self.num_envs} ({ball_hit_rate:.2%}), "
                f"Envs that scored: {goal_scores_count}/{self.num_envs} ({goal_score_rate:.2%})"
            )

        self.envs_hit_ball.clear()
        self.envs_scored_goal.clear()


@hydra_task_config(CONFIG.task, "sb3_sac_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
):
    """Train with stable-baselines SAC agent.

    Uses Hydra to load configs from registry and apply overrides from sys.argv.
    The experiment config generates Hydra overrides that are already in sys.argv,
    so env_cfg and agent_cfg arrive with values from your experiment YAML.
    """
    cfg = CONFIG

    # Set seed on env_cfg and agent_cfg (cannot be done via Hydra due to type constraints)
    if cfg.seed is not None:
        seed_value = cfg.seed if cfg.seed != -1 else random.randint(0, 10000)
        env_cfg.seed = seed_value
        agent_cfg["seed"] = seed_value
        if cfg.seed == -1:
            cfg.seed = seed_value  # Update for logging

    # Merge experiment config with Hydra-configured agent_cfg
    # (Hydra overrides already applied most values, this catches any extras)
    # Only merge fields that don't exist in agent_cfg to preserve Hydra overrides
    def merge_preserving_overrides(target: dict, source: dict):
        """Recursively merge source into target, preserving existing target values."""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                merge_preserving_overrides(target[key], value)

    merge_preserving_overrides(agent_cfg, cfg.agent_cfg)

    print(f"[INFO] Loaded experiment config: {CONFIG_PATH}")
    print_dict(cfg.to_dict(), nesting=4)

    # Directory for logging (run_info and log_root_path already printed early for Ray Tune)
    log_dir = os.path.join(log_root_path, run_info)

    # Dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "experiment.yaml"), cfg.to_dict())

    # Save command used to run the script
    command = " ".join(sys.orig_argv)
    command += f"\nconfig: {CONFIG_PATH}"
    (Path(log_dir) / "command.txt").write_text(command)

    # Post-process agent configuration for SB3
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)

    # Read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # Set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = cfg.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments."
        )

    # Set the log directory for the environment
    env_cfg.log_dir = log_dir

    # Apply reward weights from config to environment
    apply_reward_weights(cfg, env_cfg)

    # Build video kwargs (training uses periodic recording)
    video_kwargs = None
    if cfg.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % cfg.video_interval == 0,
            "video_length": cfg.video_length,
            "disable_logger": True,
        }

    # Create isaac environment with action space bounds and optional video
    env = make_env(
        cfg,
        env_cfg,
        render_mode="rgb_array" if cfg.video else None,
        video_kwargs=video_kwargs,
    )

    # Wrap for SB3 (VecEnv + HER wrappers)
    env = wrap_env_for_sb3(env, cfg)

    # Handle normalization
    norm_args = pop_norm_keys(agent_cfg)
    env = apply_training_normalization(
        env, cfg, norm_args, gamma=agent_cfg.get("gamma", 0.99)
    )

    # Configure replay buffer (standard or HER)
    replay_buffer_class = None
    replay_buffer_kwargs = None

    if cfg.use_her or cfg.use_two_stage_her:
        print("[INFO] Configuring HER replay buffer...")
        print(f"[INFO] HER goal selection strategy: {cfg.her_goal_selection_strategy}")
        print(f"[INFO] HER n_sampled_goal: {cfg.her_n_sampled_goal}")
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = {
            "n_sampled_goal": cfg.her_n_sampled_goal,
            "goal_selection_strategy": cfg.her_goal_selection_strategy,
        }
        # HER requires MultiInputPolicy for dict observation space
        if policy_arch == "MlpPolicy":
            print(
                "[INFO] Switching to MultiInputPolicy for HER (dict observation space)"
            )
            policy_arch = "MultiInputPolicy"

    # Create SAC agent
    print("[INFO] Creating SAC agent...")
    print_dict(agent_cfg, nesting=4)

    # For Ray Tune compatibility: configure custom logger to write directly to log_dir
    # For wandb compatibility: let agent create its logger first, then reconfigure it
    agent = SAC(
        policy_arch,
        env,
        verbose=1,
        tensorboard_log=log_dir,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        **agent_cfg,
    )

    # Load checkpoint if provided
    if cfg.checkpoint is not None:
        print(f"[INFO] Loading checkpoint from: {cfg.checkpoint}")
        agent = agent.load(cfg.checkpoint, env, print_system_info=True)

    # Initialize wandb BEFORE reconfiguring the logger.
    # wandb.init(sync_tensorboard=True) monkey-patches SummaryWriter.__init__,
    # so it must be called BEFORE configure() creates the TensorBoard writer.
    wandb_run = None
    if cfg.wandb_project is not None:
        if not WANDB_AVAILABLE:
            print(
                "[WARNING] wandb not installed. Skipping wandb logging. Install with: pip install wandb"
            )
        else:
            print(f"[INFO] Initializing wandb project: {cfg.wandb_project}")
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if wandb_api_key:
                wandb.login(key=wandb_api_key, relogin=False)
            else:
                print(
                    "[WARNING] WANDB_API_KEY not set. If you are not already logged in, wandb may fail to init."
                )

            # Build complete config for wandb logging (automatically captures all fields)
            wandb_config = build_wandb_config(
                cfg=cfg,
                remaining_argv=remaining_argv,
                hydra_overrides=hydra_overrides,
            )

            # Initialize wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_name or run_info,
                sync_tensorboard=True,
                config=wandb_config,
                monitor_gym=True,
                save_code=True,
            )

    # Reconfigure logger to write directly to log_dir without SAC_1 subdirectory.
    # IMPORTANT: This MUST happen AFTER wandb.init(sync_tensorboard=True) so the
    # TensorBoard SummaryWriter created here is intercepted by wandb's monkey-patch.
    new_logger = configure(log_dir, ["tensorboard", "stdout"])
    agent.set_logger(new_logger)

    # Callbacks for agent
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="sac_model",
        verbose=2,
    )
    callbacks = [checkpoint_callback]

    # Add two-stage HER metrics callback if using two-stage HER
    if cfg.use_two_stage_her:
        two_stage_callback = TwoStageHerMetricsCallback(
            num_envs=env_cfg.scene.num_envs, verbose=1
        )
        callbacks.append(two_stage_callback)
        print("[INFO] Added TwoStageHerMetricsCallback to track goal achievements")

    # Add wandb callback if enabled
    if wandb_run is not None:
        wandb_callback = WandbCallback(
            model_save_path=os.path.join(log_dir, "wandb_models"),
            model_save_freq=50000,
            verbose=2,
        )
        callbacks.append(wandb_callback)

    # Train the agent
    print(f"[INFO] Starting SAC training for {n_timesteps} timesteps...")
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=cfg.log_interval,
        )

    # Save the final model
    final_model_path = os.path.join(log_dir, "sac_model_final")
    agent.save(final_model_path)
    print(f"[INFO] Final model saved to: {final_model_path}.zip")

    # Save normalization stats if used
    if isinstance(env, VecNormalize):
        norm_path = os.path.join(log_dir, "sac_model_vecnormalize.pkl")
        print(f"[INFO] Saving normalization stats to: {norm_path}")
        env.save(norm_path)

    # Finish wandb run
    if wandb_run is not None:
        artifact = wandb.Artifact(
            name=f"sac-{cfg.task}-model",
            type="model",
            description=f"SAC model trained on {cfg.task}",
        )
        artifact.add_file(f"{final_model_path}.zip")
        wandb_run.log_artifact(artifact)
        wandb.finish()
        print("[INFO] Wandb run finished and model artifact logged.")

    # Close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
