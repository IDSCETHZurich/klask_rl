"""Script to train RL agent with Stable Baselines3 SAC."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3 SAC.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Klask-Rl-SAC-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_sac_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=1000, help="Log data every n episodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_timesteps", type=int, default=None, help="Maximum number of timesteps to train.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--wandb_project", type=str, default=None, help="Wandb project name. If provided, enables wandb logging."
)
parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (team/username).")
parser.add_argument(
    "--wandb_name", type=str, default=None, help="Wandb run name. Defaults to timestamp if not provided."
)
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


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
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

# Optional wandb import
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback

    WANDB_AVAILABLE = True
except ImportError:
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines SAC agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    # max timesteps for training
    if args_cli.max_timesteps is not None:
        agent_cfg["n_timesteps"] = args_cli.max_timesteps

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3_sac", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)

    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    # handle normalization settings if present
    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
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

    # create SAC agent from stable baselines
    print("[INFO] Creating SAC agent...")
    print_dict(agent_cfg, nesting=4)

    agent = SAC(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)

    # load checkpoint if provided
    if args_cli.checkpoint is not None:
        print(f"[INFO] Loading checkpoint from: {args_cli.checkpoint}")
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    # Initialize wandb if requested
    wandb_run = None
    if args_cli.wandb_project is not None:
        if not WANDB_AVAILABLE:
            print("[WARNING] wandb not installed. Skipping wandb logging. Install with: pip install wandb")
        else:
            print(f"[INFO] Initializing wandb project: {args_cli.wandb_project}")
            wandb_run = wandb.init(
                project=args_cli.wandb_project,
                entity=args_cli.wandb_entity,
                name=args_cli.wandb_name or run_info,
                config={
                    "algorithm": "SAC",
                    "task": args_cli.task,
                    "num_envs": env_cfg.scene.num_envs,
                    "policy": policy_arch,
                    "n_timesteps": n_timesteps,
                    **agent_cfg,
                },
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )

    # callbacks for agent
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=log_dir, name_prefix="sac_model", verbose=2  # Save every 10k steps
    )
    callbacks = [checkpoint_callback]

    # Add wandb callback if enabled
    if wandb_run is not None:
        wandb_callback = WandbCallback(
            model_save_path=os.path.join(log_dir, "wandb_models"),
            model_save_freq=50000,
            verbose=2,
        )
        callbacks.append(wandb_callback)

    # train the agent
    print(f"[INFO] Starting SAC training for {n_timesteps} timesteps...")
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=args_cli.log_interval,
        )

    # save the final model
    final_model_path = os.path.join(log_dir, "sac_model_final")
    agent.save(final_model_path)
    print(f"[INFO] Final model saved to: {final_model_path}.zip")

    # save normalization stats if used
    if isinstance(env, VecNormalize):
        norm_path = os.path.join(log_dir, "sac_model_vecnormalize.pkl")
        print(f"[INFO] Saving normalization stats to: {norm_path}")
        env.save(norm_path)

    # Finish wandb run
    if wandb_run is not None:
        # Log final model as artifact
        artifact = wandb.Artifact(
            name=f"sac-{args_cli.task}-model",
            type="model",
            description=f"SAC model trained on {args_cli.task}",
        )
        artifact.add_file(f"{final_model_path}.zip")
        wandb_run.log_artifact(artifact)
        wandb.finish()
        print("[INFO] Wandb run finished and model artifact logged.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
