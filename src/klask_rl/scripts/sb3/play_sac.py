"""Script to play/evaluate a trained SAC checkpoint.

Loads the same experiment config used for training to reconstruct the exact
environment, then runs inference with a saved checkpoint.

Usage:
    python play_sac.py --config experiments/klask_sac_her.yaml --checkpoint path/to/model.zip
    python play_sac.py -c experiments/klask_sac_base.yaml  # uses latest checkpoint
    python play_sac.py -c experiments/klask_sac_two_stage_her.yaml --num_envs 1 --video
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

# Parse arguments BEFORE any isaaclab imports
parser = argparse.ArgumentParser(description="Play a trained SAC checkpoint")
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="experiments/klask_sac_base.yaml",
    help="Config file path relative to scripts/sb3/config/ (default: experiments/klask_sac_base.yaml)",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint (.zip). If not provided, uses the latest checkpoint.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Override number of environments (default: from config).",
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record video during playback.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length of the recorded video in steps (default: 500).",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=True,
    help="Use deterministic actions (default: True).",
)
parser.add_argument(
    "--no-deterministic",
    action="store_true",
    default=False,
    help="Use stochastic actions instead of deterministic.",
)
args, remaining_argv = parser.parse_known_args()

deterministic = not args.no_deterministic

# Resolve config path
from env_utils import resolve_config_path, get_log_root_path

CONFIG_PATH = resolve_config_path(args.config, __file__)

# Load config and set CUDA_VISIBLE_DEVICES BEFORE importing isaaclab
from experiment_config import ExperimentConfig

CONFIG = ExperimentConfig.from_file(CONFIG_PATH)

# Override num_envs if specified
if args.num_envs is not None:
    CONFIG.num_envs = args.num_envs

# Setup CUDA visibility
CONFIG.setup_cuda_visibility()

# Override video settings for playback
if args.video:
    CONFIG.video_cfg = CONFIG.video_cfg or {}
    CONFIG.video_cfg["length"] = args.video_length
    CONFIG.video_cfg["interval"] = 1  # Record from the start

# Generate Hydra overrides from config
hydra_overrides = CONFIG.get_hydra_overrides()
sys.argv = [sys.argv[0]] + hydra_overrides + remaining_argv

# NOW import isaaclab
from isaaclab.app import AppLauncher

# Always enable cameras for video
app_launcher_args = CONFIG.app_launcher_args()
if args.video:
    app_launcher_args["enable_cameras"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import torch

from stable_baselines3 import SAC

from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.sb3 import process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import klask_rl.tasks  # noqa: F401

from env_utils import (
    apply_inference_normalization,
    apply_reward_weights,
    find_latest_checkpoint,
    make_env,
    pop_norm_keys,
    wrap_env_for_sb3,
)


@hydra_task_config(CONFIG.task, "sb3_sac_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
):
    """Play with a trained SAC agent."""
    cfg = CONFIG

    print(f"[INFO] Loaded experiment config: {CONFIG_PATH}")
    print_dict(cfg.to_dict(), nesting=4)

    # Resolve checkpoint path
    log_root_path = get_log_root_path(cfg, __file__)

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        print(
            f"[INFO] No checkpoint specified, searching for latest in: {log_root_path}"
        )
        checkpoint_path = find_latest_checkpoint(log_root_path)

    print(f"[INFO] Using checkpoint: {checkpoint_path}")
    log_dir = str(Path(checkpoint_path).parent)

    # Set the log directory for the environment
    env_cfg.log_dir = log_dir

    # Post-process agent configuration for SB3
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)

    # Pop fields not needed for agent constructor
    agent_cfg.pop("policy", None)
    agent_cfg.pop("n_timesteps", None)

    # Apply reward weights from config to environment (same as training)
    apply_reward_weights(cfg, env_cfg)

    # Build video kwargs (playback records from the start)
    video_kwargs = None
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }

    # Create isaac environment with action space bounds and optional video
    env = make_env(
        cfg,
        env_cfg,
        render_mode="rgb_array" if args.video else None,
        video_kwargs=video_kwargs,
    )

    # Wrap for SB3 (VecEnv + HER wrappers)
    env = wrap_env_for_sb3(env, cfg)

    # Handle normalization (must match training)
    norm_args = pop_norm_keys(agent_cfg)
    env = apply_inference_normalization(
        env, cfg, norm_args, checkpoint_path, gamma=agent_cfg.get("gamma", 0.99)
    )

    # Load the trained SAC agent
    print(f"[INFO] Loading SAC model from: {checkpoint_path}")
    agent = SAC.load(checkpoint_path, env=env, print_system_info=True)

    dt = env.unwrapped.step_dt

    # Reset environment
    obs = env.reset()
    timestep = 0
    print(f"[INFO] Running inference (deterministic={deterministic})...")

    # Simulate
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(actions)

        timestep += 1

        if args.video and timestep >= args.video_length:
            print(f"[INFO] Recorded {args.video_length} steps of video. Exiting.")
            break

        # Real-time pacing
        elapsed = time.time() - start_time
        sleep_time = dt - elapsed
        if args.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Close
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
