"""Script to play a checkpoint of an RL agent trained with Stable-Baselines3 SAC.

This script uses the unified configuration system with Hydra overrides.

Usage:
    python play_sac.py --config experiments/klask_sac_base.yaml
    python play_sac.py --config experiments/klask_sac_her.yaml --checkpoint /path/to/model.zip
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

# Parse config file argument BEFORE any isaaclab imports
parser = argparse.ArgumentParser(description="Play SAC agent", add_help=False)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="experiments/klask_sac_base.yaml",
    help="Experiment config file (default: experiments/klask_sac_base.yaml)",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint (default: latest from logs)",
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=1,
    help="Number of environments (default: 1)",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Record video during playback",
)
args, remaining_argv = parser.parse_known_args()

# Resolve config path
config_dir = Path(__file__).parent / "config"
if Path(args.config).is_absolute():
    CONFIG_PATH = Path(args.config)
elif args.config.startswith("experiments/") or "/" in args.config:
    CONFIG_PATH = config_dir / args.config
else:
    if (config_dir / "experiments" / args.config).exists():
        CONFIG_PATH = config_dir / "experiments" / args.config
    else:
        CONFIG_PATH = config_dir / args.config

from experiment_config import ExperimentConfig

CONFIG = ExperimentConfig.from_file(CONFIG_PATH)
CONFIG.setup_cuda_visibility()

# Set Hydra CLI overrides before importing isaaclab
# Override num_envs for playback if specified
hydra_overrides = CONFIG.get_hydra_overrides()
if args.num_envs is not None:
    # Filter out existing num_envs override and add the playback one
    hydra_overrides = [
        o for o in hydra_overrides if not o.startswith("env.scene.num_envs=")
    ]
    hydra_overrides.append(f"env.scene.num_envs={args.num_envs}")
sys.argv = [sys.argv[0]] + hydra_overrides
print(f"[INFO] Hydra overrides: {hydra_overrides}")

from isaaclab.app import AppLauncher

# Override for video if requested
if args.video:
    CONFIG.app_launcher["enable_cameras"] = True

# launch omniverse app
app_launcher = AppLauncher(CONFIG.app_launcher_args())
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import klask_rl.tasks  # noqa: F401


@hydra_task_config(CONFIG.task, "sb3_sac_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
):
    """Play with stable-baselines SAC agent.

    Uses Hydra for configuration, same mechanism as training.
    """
    cfg = CONFIG

    # Override agent_cfg with our experiment config values
    for key, value in cfg.agent_cfg.items():
        agent_cfg[key] = value

    print(f"[INFO] Loaded experiment config: {CONFIG_PATH}")
    print_dict(cfg.to_dict(), nesting=4)

    # Grab task name for checkpoint path
    task_name = cfg.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # Directory for logging
    log_root_path = os.path.join("logs", "sb3_sac", train_task_name)
    log_root_path = os.path.abspath(log_root_path)

    # Checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(
            log_root_path, ".*", "sac_model_.*.zip", sort_alpha=False
        )
    else:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir

    # Create isaac environment
    env = gym.make(
        cfg.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None
    )

    # Post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Set bounded action space for SAC (must match training configuration)
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
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step % 2000 == 0,
            "video_length": 200,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=True)

    # Check for normalization file
    vec_norm_path = checkpoint_path.replace(
        "/sac_model_final", "/sac_model_vecnormalize"
    ).replace(".zip", ".pkl")
    vec_norm_path = Path(vec_norm_path)

    # Normalize environment (if needed)
    if vec_norm_path.exists():
        print(f"[INFO] Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    elif "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=False,
            norm_obs="normalize_input" in agent_cfg
            and agent_cfg.pop("normalize_input"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        )

    # Load SAC agent
    print(f"[INFO] Loading SAC checkpoint from: {checkpoint_path}")
    agent = SAC.load(checkpoint_path, env, print_system_info=True)

    dt = env.unwrapped.step_dt

    # Reset environment
    obs = env.reset()
    timestep = 0

    deterministic = True
    real_time = False
    print(f"[INFO] Starting playback (deterministic={deterministic})...")

    # Simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=deterministic)
            obs, _, _, _ = env.step(actions)

        if args.video:
            timestep += 1
            if timestep == 200:
                break

        sleep_time = dt - (time.time() - start_time)
        if real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
