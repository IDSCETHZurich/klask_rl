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
config_dir = Path(__file__).parent / "config"
if Path(args.config).is_absolute():
    CONFIG_PATH = Path(args.config)
elif (
    args.config.startswith("experiments/") or "/" in args.config or "\\" in args.config
):
    CONFIG_PATH = config_dir / args.config
else:
    # Check experiments folder first, then root config folder
    if (config_dir / "experiments" / args.config).exists():
        CONFIG_PATH = config_dir / "experiments" / args.config
    else:
        CONFIG_PATH = config_dir / args.config

# Load config and set CUDA_VISIBLE_DEVICES BEFORE importing isaaclab
from experiment_config import ExperimentConfig

CONFIG = ExperimentConfig.from_file(CONFIG_PATH)

# Setup CUDA visibility (handles both normal training and Ray Tune cases)
CONFIG.setup_cuda_visibility()

# Print experiment info EARLY for Ray Tune to detect
# (must be before Isaac Sim startup which produces lots of output)
run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Use absolute path relative to script location (not cwd) for Ray Tune compatibility
script_dir = Path(__file__).parent.parent  # Go up to klask_rl/
log_root_path = str(script_dir / "logs" / "sb3_sac" / CONFIG.task)
print(f"[INFO] Logging experiment in directory: {log_root_path}")
print(f"Exact experiment name requested from command line: {run_info}")

# IMPORTANT: Set Hydra CLI overrides BEFORE importing isaaclab
# This is the key for unified config handling - both standalone and Ray
# use the same Hydra override mechanism

# Extract Ray Tune parameter keys from remaining_argv to avoid duplicates
ray_tune_keys = set()
for arg in remaining_argv:
    # Parse Hydra overrides like "agent.policy_kwargs.net_arch=[256,128,64]"
    if "=" in arg:
        key = arg.split("=")[0].strip("'\"")
        ray_tune_keys.add(key)

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

import gymnasium as gym
import numpy as np
import os
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
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import klask_rl.tasks  # noqa: F401
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    Sb3VecHerWrapper,
    Sb3TwoStageHerWrapper,
)


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
            dones = self.locals.get("dones", [])

            for i, info in enumerate(infos):
                if info.get("ball_hit", False) or info.get("terminal_ball_hit", False):
                    self.envs_hit_ball.add(i)

                if dones[i]:
                    if info.get("ball_hit", False) and not info.get(
                        "TimeLimit.truncated", False
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

    # Create isaac environment
    env = gym.make(
        cfg.task,
        cfg=env_cfg,
        render_mode="rgb_array" if cfg.video else None,
    )

    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Set bounded action space for SAC
    action_dim = env.unwrapped.single_action_space.shape[-1]
    max_vel = cfg.max_velocity
    print(f"[INFO] Setting action space bounds to [-{max_vel}, {max_vel}] m/s")
    env.unwrapped.single_action_space = gym.spaces.Box(
        low=-max_vel, high=max_vel, shape=(action_dim,), dtype=np.float32
    )
    env.unwrapped.action_space = gym.vector.utils.batch_space(
        env.unwrapped.single_action_space, env.unwrapped.num_envs
    )

    # Wrap for video recording
    if cfg.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % cfg.video_interval == 0,
            "video_length": cfg.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=True)

    # Wrap with HER wrapper if enabled
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

    # Handle normalization settings if present
    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
        if cfg.use_her or cfg.use_two_stage_her:
            print(
                "[WARNING] VecNormalize is not fully compatible with HER. Disabling observation normalization."
            )
        else:
            print(f"Normalizing input, {norm_args=}")
            env = VecNormalize(
                env,
                training=True,
                norm_obs=norm_args["normalize_input"],
                norm_reward=norm_args.get("normalize_value", False),
                clip_obs=norm_args.get("clip_obs", 100.0),
                gamma=agent_cfg.get("gamma", 0.99),
                clip_reward=np.inf,
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
    
    # Reconfigure logger to write directly to log_dir without SAC_1 subdirectory
    # This works for both Ray Tune (finds metrics) and wandb (we'll point it to log_dir)
    new_logger = configure(log_dir, ["tensorboard", "stdout"])
    agent.set_logger(new_logger)

    # Load checkpoint if provided
    if cfg.checkpoint is not None:
        print(f"[INFO] Loading checkpoint from: {cfg.checkpoint}")
        agent = agent.load(cfg.checkpoint, env, print_system_info=True)

    # Initialize wandb if requested
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
            # Set wandb to look for tensorboard logs in log_dir (where we reconfigured the logger to write)
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_name or run_info,
                dir=log_dir,  # Set wandb working directory to log_dir
                sync_tensorboard=True,  # Will sync tensorboard events from log_dir
                config={
                    "algorithm": "SAC",
                    "task": cfg.task,
                    "num_envs": env_cfg.scene.num_envs,
                    "policy": policy_arch,
                    "n_timesteps": n_timesteps,
                    **agent_cfg,
                },
                monitor_gym=True,
                save_code=True,
            )

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
