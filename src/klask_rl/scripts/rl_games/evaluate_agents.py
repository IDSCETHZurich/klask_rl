# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a tournament between RL agents from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Klask-Rl-v0", help="Name of the task.")
parser.add_argument("--dir", type=str, default=None, help="Path to tournament directory containing config directories.")
parser.add_argument("--checkpoints", action="append", help="Paths to agent checkpoints", default=[])
parser.add_argument(
    "--config", type=str, default=None, help="config.yaml file, rl_games_cfg_entry_point used when not provided"
)
parser.add_argument("--num_rounds", type=int, default=1, help="Number of rounds in the tournament.")
parser.add_argument("--num_games_per_round", type=int, default=5, help="Number of games per round.")
parser.add_argument("--tournament_name", type=str, default="tournament", help="Name used for logging videos.")
parser.add_argument("--num_instances", type=int, default=4)
parser.add_argument("--run_number", type=int, default=None)
parser.add_argument("--player_configs", action="append")
parser.add_argument("--elo_thresehold", type=int, default=None)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import itertools
import math
import os
import shutil
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
import yaml
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from klask_rl.assets.robots.klask import KLASK_PARAMS
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    ObservationNoiseWrapper,
    OpponentObservationWrapper,
    RewardWeightWrapper,
    RlGamesGpuEnvSelfPlay,
    find_wrapper,
)
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner


def update_elo(p1, p2, score_1, score_2, k=10.0):
    elo_1 = p1.elo + k * (score_1 - 1 / (1 + 10 ** ((p2.elo - p1.elo) / 400)))
    elo_2 = p2.elo + k * (score_2 - 1 / (1 + 10 ** ((p1.elo - p2.elo) / 400)))
    p1.elo = elo_1.item()
    p2.elo = elo_2.item()


def compute_scores(info):
    average_score_1 = (
        info["episode"]["Episode_Termination/goal_scored"]
        + info["episode"]["Episode_Termination/opponent_in_goal"]
        + 0.5 * info["episode"]["Episode_Termination/time_out"]
    )
    average_score_2 = (
        info["episode"]["Episode_Termination/goal_conceded"]
        + info["episode"]["Episode_Termination/player_in_goal"]
        + 0.5 * info["episode"]["Episode_Termination/time_out"]
    )
    return average_score_1, average_score_2


def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    if args_cli.config is not None:
        with open(args_cli.config, "r") as file:
            config = yaml.safe_load(file)
        agent_cfg.update(config)

    # specify directory for logging experiments
    # log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    # log_root_path = os.path.abspath(log_root_path)
    # log_dir = os.path.dirname(args_cli.tournament_name)

    # wrap around environment for rl-games
    agent_cfg["terminations"]["opponent_in_goal"] = True
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # Null out disabled termination terms on env_cfg BEFORE construction.
    if "terminations" in agent_cfg and hasattr(env_cfg, "terminations"):
        for term, active in agent_cfg["terminations"].items():
            if not active and hasattr(env_cfg.terminations, term):
                setattr(env_cfg.terminations, term, None)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    # if args_cli.video:
    #    video_kwargs = {
    #        "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
    #        "step_trigger": lambda step: step == 0,
    #        "video_length": args_cli.video_length,
    #        "disable_logger": True,
    #    }
    #    print("[INFO] Recording videos during training.")
    #    print_dict(video_kwargs, nesting=4)
    #    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    # if isinstance(env.unwrapped, DirectMARLEnv):
    #    env = multi_agent_to_single_agent(env)

    obs_noise = agent_cfg["env"].get("obs_noise", 0.0)
    if obs_noise > 0.0:
        env = ObservationNoiseWrapper(
            env, obs_noise,
            own_goal=KLASK_PARAMS["player_goal"],
            other_goal=KLASK_PARAMS["opponent_goal"],
        )
    env = OpponentObservationWrapper(env)
    if "rewards" in agent_cfg.keys():
        env = RewardWeightWrapper(env, agent_cfg["rewards"])

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs=clip_obs, clip_actions=clip_actions, evaluation_mode=True)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnvSelfPlay(
            config_name, num_actors, agent_cfg.copy(), **kwargs
        ),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # reset environment
    obs = env.reset()
    timestep = 0

    # create runner from rl-games
    runner = Runner()
    players = []

    for i in range(args_cli.num_instances):
        model_file = args_cli.checkpoints[i]
        player_name = f"agent_{i+1}"
        with open(args_cli.player_configs[i], "r") as file:
            config = yaml.safe_load(file)

        runner.load(config)
        player = runner.create_player()
        player.restore(model_file)
        player.reset()
        # initialize RNN states if used
        if player.is_rnn:
            player.init_rnn()
        _ = player.get_batch_size(obs, 1)
        player.elo = 1200.0
        player.config = args_cli.player_configs[i]
        player.checkpoint = model_file
        player.name = player_name
        player.device = config["params"]["config"]["device"]
        players.append(player)

    player_elos = []

    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.

    for i, _ in enumerate(range(args_cli.num_rounds)):
        for p1, p2 in itertools.combinations(players, 2):

            num_games = 0
            average_score_1 = 0.0
            average_score_2 = 0.0
            while num_games < args_cli.num_games_per_round:
                # run everything in inference mode
                with torch.inference_mode():
                    # convert obs to agent format
                    obs_1 = p1.obs_to_torch(obs.to(p1.device))
                    opponent_obs = find_wrapper(env, OpponentObservationWrapper).opponent_obs
                    obs_2 = p2.obs_to_torch(opponent_obs.to(p2.device))
                    # agent stepping

                    actions_p1 = p1.get_action(obs_1, is_deterministic=True)
                    actions_p2 = -p2.get_action(obs_2, is_deterministic=True)
                    # env stepping

                    obs, rew, dones, info = env.step(
                        torch.cat([actions_p1.to(rl_device), actions_p2.to(rl_device)], dim=1)
                    )
                    # perform operations for terminated episodes
                    if len(dones) > 0:
                        num_games += dones.sum()
                        score_1, score_2 = compute_scores(info)
                        average_score_1 += score_1
                        average_score_2 += score_2
                        # reset rnn state for terminated episodes
                        if p1.is_rnn and p1.states is not None:
                            for s in p1.states:
                                s[:, dones, :] = 0.0
                        if p2.is_rnn and p2.states is not None:
                            for s in p2.states:
                                s[:, dones, :] = 0.0
                if args_cli.video:
                    timestep += 1
                    # Exit the play loop after recording one video
                    if timestep == args_cli.video_length:
                        break

            update_elo(p1, p2, average_score_1 / num_games, average_score_2 / num_games)
        player_elos.append([p.elo for p in players])

        print(f"Round {i} completed. ELO scores:")
        for p in players:
            print(f"{p.name}: {p.elo}")
            if args_cli.elo_thresehold is not None and p.elo > args_cli.elo_thresehold:
                if p.name != "agent_1":  # this is the benchmark agent
                    agent_dir = Path(args_cli.dir) / f"agent_{args_cli.run_number}"
                    agent_dir.mkdir(parents=True, exist_ok=True)

                    # Save checkpoint and config (replace with actual saving logic if needed)
                    shutil.copy(p.checkpoint, agent_dir / "player_checkpoint.pth")
                    shutil.copy(p.config, agent_dir / "player_config.yaml")
                    break

    # Plot ELO scores:
    player_elos = np.array(player_elos)
    max_elo = player_elos.max()
    max_index = player_elos.argmax()
    print("Players elo scores: ", player_elos)
    print(f"Agent {max_index+1} has achieved the best elo score of: {max_elo}")

    if args_cli.elo_thresehold is None:
        source_path = Path(args_cli.checkpoints[i])
        dest_path = source_path.parent.parent.parent / "best_agent"  # go two levels up (to agent_X folder)
        for filename in os.listdir(dest_path):
            file_path = os.path.join(dest_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        shutil.copyfile(source_path, dest_path / f"best_agent({max_index+1})_number_{args_cli.run_number}.pth")

    # kwargs = {p.name: player_elos[:, i] for i, p in enumerate(players)}
    # np.savez(os.path.join(log_root_path, log_dir, "elos.npz"), **kwargs)
    # for i, p in enumerate(players):
    #    plt.plot(player_elos[:, i], label=p.name)
    # plt.xlabel("game")
    # plt.ylabel("ELO")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


# DEGUB CONSOLE:

# 'Episode_Termination/time_out': 0, 'Episode_Termination/goal_scored': 0, 'Episode_Termination/goal_conceded': 0, 'Episode_Termination/player_in_goal': 0, 'Episode_Termination/opponent_in_goal': 2}}
# BUT dones.sum() = tensor(0, device='cuda:0')
