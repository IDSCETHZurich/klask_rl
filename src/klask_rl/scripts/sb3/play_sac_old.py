"""Script to play a checkpoint of an RL agent trained with Stable-Baselines3 SAC."""

"""Launch Isaac Sim Simulator first."""

import sys
from pathlib import Path

from isaaclab.app import AppLauncher
from train_config import TrainConfig
from play_config import PlayConfig


TRAIN_CFG_PATH = Path(__file__).parent / "config" / "klask_rl_sac.yaml"
TRAIN_CFG = TrainConfig.from_file(TRAIN_CFG_PATH)
PLAY_CFG_PATH = Path(__file__).parent / "config" / "klask_rl_sac_play.yaml"
PLAY_CFG = PlayConfig.from_file(PLAY_CFG_PATH)

# Ignore any CLI overrides; playback is config-driven.
sys.argv = [sys.argv[0]]

# launch omniverse app
app_launcher = AppLauncher(TRAIN_CFG.app_launcher_args())
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
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

import klask_rl.tasks  # noqa: F401


@hydra_task_config(TRAIN_CFG.task, TRAIN_CFG.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Play with stable-baselines SAC agent."""
    train_cfg = TRAIN_CFG
    train_cfg.apply(env_cfg, agent_cfg)
    play_cfg = PLAY_CFG
    play_cfg.apply(env_cfg)
    print(f"[INFO] Loaded training config: {TRAIN_CFG_PATH}")
    print_dict(train_cfg.to_dict(), nesting=4)
    print(f"[INFO] Loaded play config: {PLAY_CFG_PATH}")
    print_dict(play_cfg.to_dict(), nesting=4)

    # grab task name for checkpoint path
    task_name = train_cfg.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3_sac", train_task_name)
    log_root_path = os.path.abspath(log_root_path)

    # checkpoint and log_dir stuff
    checkpoint_path = play_cfg.play_checkpoint
    if isinstance(checkpoint_path, str):
        checkpoint_path = checkpoint_path.strip() or None
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(
            log_root_path, ".*", "sac_model_.*.zip", sort_alpha=False
        )
    else:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    log_dir = os.path.dirname(checkpoint_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        train_cfg.task, cfg=env_cfg, render_mode="rgb_array" if play_cfg.video else None
    )

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Set bounded action space for SAC (must match training configuration).
    # SAC outputs will be in [-max_velocity, max_velocity] m/s directly.
    # Must set on unwrapped env because Sb3VecEnvWrapper reads from there.
    action_dim = env.unwrapped.single_action_space.shape[-1]
    max_vel = train_cfg.max_velocity
    print(f"[INFO] Setting action space bounds to [-{max_vel}, {max_vel}] m/s")
    env.unwrapped.single_action_space = gym.spaces.Box(
        low=-max_vel, high=max_vel, shape=(action_dim,), dtype=np.float32
    )
    env.unwrapped.action_space = gym.vector.utils.batch_space(
        env.unwrapped.single_action_space, env.unwrapped.num_envs
    )

    # wrap for video recording
    if play_cfg.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step % play_cfg.video_interval == 0,
            "video_length": play_cfg.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not train_cfg.keep_all_info)

    # check for normalization file
    vec_norm_path = checkpoint_path.replace(
        "/sac_model_final", "/sac_model_vecnormalize"
    ).replace(".zip", ".pkl")
    vec_norm_path = Path(vec_norm_path)

    # normalize environment (if needed)
    if vec_norm_path.exists():
        print(f"[INFO] Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        # do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
    elif "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=False,
            norm_obs="normalize_input" in agent_cfg
            and agent_cfg.pop("normalize_input"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        )

    # create SAC agent from stable baselines
    print(f"[INFO] Loading SAC checkpoint from: {checkpoint_path}")
    agent = SAC.load(checkpoint_path, env, print_system_info=True)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    timestep = 0

    deterministic = True
    real_time = False
    print(f"[INFO] Starting playback (deterministic={deterministic})...")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=deterministic)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if play_cfg.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == play_cfg.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
