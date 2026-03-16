# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

# Note: it is likly that the following command produced teh best agent (Tobias)
# python scripts/rl_games/train_klask.py --config /workspace/klask_rl/scripts/rl_games/config/klask_config_3.yaml --device cuda:0 --headless --num_envs 4096 --wandb-project-name KLASK --training_curriculum --mode 0 --checkpoint /workspace/klask_rl/logs/rl_games/klask/pretrained_agent_action_1.0/nn/last_klask_ep_35_rew_3.9405801.pth --project_folder /workspace/klask_rl/logs/rl_games/klask/pool_of_players/

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Klask-Rl-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

parser.add_argument(
    "--config", type=str, default=None, help="config.yaml file, rl_games_cfg_entry_point used when not provided."
)
parser.add_argument("--full_experiment_name", type=str, default=None, help="Experiment name used for logs.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--training_curriculum", action="store_true", default=False)
parser.add_argument("--mode", type=int, default=None, help="mode for training curriculum")
parser.add_argument("--project_folder", type=str, default=None, help="mode for training curriculum")

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

"""Rest everything follows."""

import math
import os
import pickle
import random
import signal
import time
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import yaml
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from klask_rl.assets.robots.klask import KLASK_PARAMS
from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.tasks.manager_based.klask_rl.utils_manager_based import set_terminations
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    ActionHistoryWrapper,
    CurriculumWrapper,
    KlaskRlCollisionAvoidanceWrapper,
    KlaskRlRandomOpponentWrapper,
    ObservationNoiseWrapper,
    OpponentActionWrapper,
    OpponentObservationWrapper,
    RlGamesGpuEnvSelfPlay,
)
from klask_rl_games import KlaskRlAlgoObserver, KlaskRlRunner
from rl_games.common import env_configurations, vecenv


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    if args_cli.config is not None:
        with open(args_cli.config, "r") as file:
            config = yaml.safe_load(file)
        agent_cfg.update(config)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.full_experiment_name is not None:
        agent_cfg["params"]["config"]["full_experiment_name"] = args_cli.full_experiment_name

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    elif agent_cfg["params"].get("load_checkpoint", False) and agent_cfg["params"].get("load_path"):
        resume_path = retrieve_file_path(agent_cfg["params"]["load_path"])
        print(f"[INFO]: Loading model checkpoint from config: {resume_path}")
    else:
        resume_path = None
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

    # Save pickle files using standard pickle module
    with open(os.path.join(log_root_path, log_dir, "params", "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)
    with open(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)

    # read configurations about the agent-training
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Negate opponent actions to convert from player frame back to world frame.
    # Must be innermost wrapper so the actuator model sees player-frame data.
    env = OpponentActionWrapper(env)

    if agent_cfg["env"].get("actuator_model", False):
        env = ActuatorModelWrapper(env)

    if agent_cfg["env"].get("collision_avoidance", False):
        env = KlaskRlCollisionAvoidanceWrapper(env, max_vel=clip_actions)

    if KLASK_PARAMS["action_history"] > 0:
        env = ActionHistoryWrapper(env, history_length=KLASK_PARAMS["action_history"])

    obs_noise = agent_cfg["env"].get("obs_noise", 0.0)
    if obs_noise > 0.0:
        env = ObservationNoiseWrapper(env, obs_noise, list(range(12)))

    # configure active reward terms and curricula as specified in agent_cfg:
    if "rewards" in agent_cfg.keys():
        env = CurriculumWrapper(env, agent_cfg["rewards"])

    # if self-play, use opponent observation wrapper to get access to opponent player's observations:
    if agent_cfg["params"]["config"].get("self_play", False):
        env = OpponentObservationWrapper(env)
    # if no self-play, pick random actions for the opponent:
    else:
        env = KlaskRlRandomOpponentWrapper(env)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    if agent_cfg["params"]["config"].get("self_play", False):
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnvSelfPlay(
                config_name,
                num_actors,
                agent_cfg.copy(),
                training_curriculum=args_cli.training_curriculum,
                mode=args_cli.mode,
                folder=args_cli.project_folder,
                **kwargs,
            ),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    else:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set active termination terms specified in agent_cfg:
    if "terminations" in agent_cfg.keys():
        set_terminations(env, agent_cfg["terminations"])

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = KlaskRlRunner(KlaskRlAlgoObserver())
    runner.load(agent_cfg)

    # create complete config and log to wandb:
    if "env" in agent_cfg.keys():
        agent_cfg["env"].update(KLASK_PARAMS)
    else:
        agent_cfg["env"] = KLASK_PARAMS

    use_wandb = args_cli.wandb_project_name is not None
    if use_wandb:
        import wandb

        config = {"agent": agent_cfg, "env": env_cfg.to_dict()}
        wandb.init(
            project=args_cli.wandb_project_name,
            entity=args_cli.wandb_entity,
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )

    # reset the agent and env
    runner.reset()
    start_time = time.time()
    interrupted = False
    try:
        # train the agent
        run_args = {"train": True, "play": False, "sigma": train_sigma}
        if resume_path is not None:
            run_args["checkpoint"] = resume_path
        runner.run(run_args)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[INFO] Training interrupted by user (Ctrl+C).")
    finally:
        print(f"Total training time: {time.time() - start_time}")

        # log model checkpoint to wandb and finish the run:
        if use_wandb:
            if interrupted:
                wandb.finish(exit_code=1)
            else:
                model = wandb.Artifact("model", type="model")
                model.add_file(
                    os.path.join(log_root_path, log_dir, "nn", f"{agent_cfg['params']['config']['name']}.pth")
                )
                wandb.log_artifact(model)
                wandb.finish()

        # close the simulator
        env.close()

    # re-raise so the process exits cleanly after cleanup
    if interrupted:
        signal.raise_signal(signal.SIGINT)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
