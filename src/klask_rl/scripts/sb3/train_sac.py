"""Script to train RL agent with Stable Baselines3 SAC."""

"""Launch Isaac Sim Simulator first."""

import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
from train_config import TrainConfig
from utils import cleanup_pbar

TRAIN_CFG_PATH = Path(__file__).parent / "config" / "klask_rl_sac.yaml"
TRAIN_CFG = TrainConfig.from_file(TRAIN_CFG_PATH)

# Ignore any CLI overrides; training is fully config-driven.
sys.argv = [sys.argv[0]]

# launch omniverse app
app_launcher = AppLauncher(TRAIN_CFG.app_launcher_args())
simulation_app = app_launcher.app

# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
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


@hydra_task_config(TRAIN_CFG.task, TRAIN_CFG.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines SAC agent."""
    train_cfg = TRAIN_CFG
    train_cfg.apply(env_cfg, agent_cfg)
    print(f"[INFO] Loaded training config: {TRAIN_CFG_PATH}")
    print_dict(train_cfg.to_dict(), nesting=4)

    # directory for logging into
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3_sac", train_cfg.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"[INFO] Experiment name: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "train.yaml"), train_cfg.to_dict())

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    command += f"\nconfig: {TRAIN_CFG_PATH}"
    (Path(log_dir) / "command.txt").write_text(command)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)

    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = train_cfg.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(train_cfg.task, cfg=env_cfg, render_mode="rgb_array" if train_cfg.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Set bounded action space for SAC.
    action_dim = env.unwrapped.single_action_space.shape[-1]
    max_vel = train_cfg.max_velocity
    print(f"[INFO] Setting action space bounds to [-{max_vel}, {max_vel}] m/s")
    env.unwrapped.single_action_space = gym.spaces.Box(
        low=-max_vel, high=max_vel, shape=(action_dim,), dtype=np.float32
    )
    env.unwrapped.action_space = gym.vector.utils.batch_space(env.unwrapped.single_action_space, env.unwrapped.num_envs)

    # wrap for video recording
    if train_cfg.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % train_cfg.video_interval == 0,
            "video_length": train_cfg.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not train_cfg.keep_all_info)

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
    if train_cfg.checkpoint is not None:
        print(f"[INFO] Loading checkpoint from: {train_cfg.checkpoint}")
        agent = agent.load(train_cfg.checkpoint, env, print_system_info=True)

    # Initialize wandb if requested
    wandb_run = None
    if train_cfg.wandb_project is not None:
        if not WANDB_AVAILABLE:
            print("[WARNING] wandb not installed. Skipping wandb logging. Install with: pip install wandb")
        else:
            print(f"[INFO] Initializing wandb project: {train_cfg.wandb_project}")
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if wandb_api_key:
                # Use API key from environment to avoid interactive login.
                wandb.login(key=wandb_api_key, relogin=False)
            else:
                print("[WARNING] WANDB_API_KEY not set. If you are not already logged in, wandb may fail to init.")
            wandb_run = wandb.init(
                project=train_cfg.wandb_project,
                entity=train_cfg.wandb_entity,
                name=train_cfg.wandb_name or run_info,
                config={
                    "algorithm": "SAC",
                    "task": train_cfg.task,
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
            log_interval=train_cfg.log_interval,
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
            name=f"sac-{train_cfg.task}-model",
            type="model",
            description=f"SAC model trained on {train_cfg.task}",
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
