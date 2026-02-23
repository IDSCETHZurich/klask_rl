# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Klask-Rl-v0", help="Name of the task.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/rl_games/klask/demo_agents/best_one/klask.pth",
    help="Path to model checkpoint.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument(
    "--config",
    type=str,
    default="logs/rl_games/klask/demo_agents/best_one/agent.yaml",
    help="config.yaml file, rl_games_cfg_entry_point used when not provided",
)
parser.add_argument(
    "--opponent_checkpoint",
    type=str,
    default=None,
    help="Path to opponent model checkpoint. When set, runs a head-to-head evaluation.",
)
parser.add_argument(
    "--opponent_config",
    type=str,
    default=None,
    help="config.yaml for the opponent agent. Uses --config when not provided.",
)
parser.add_argument(
    "--num_games",
    type=int,
    default=1000,
    help="Number of games to play in head-to-head evaluation mode.",
)


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

import gymnasium as gym
import math
import os
import torch
import yaml
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    KlaskRlRandomOpponentWrapper,
    CurriculumWrapper,
    RlGamesGpuEnvSelfPlay,
    KlaskRlAgentOpponentWrapper,
    ObservationNoiseWrapper,
    KlaskRlCollisionAvoidanceWrapper,
    ActionHistoryWrapper,
    find_wrapper,
)
from klask_rl.tasks.manager_based.klask_rl.utils_manager_based import set_terminations
from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.assets.robots.klask import KLASK_PARAMS


def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    if args_cli.config is not None:
        with open(args_cli.config, "r") as file:
            config = yaml.safe_load(file)
        agent_cfg.update(config)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games

    rl_device = agent_cfg["params"]["config"]["device"]
    rl_device = args_cli.device

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg["env"].get("actuator_model", False):
        env = ActuatorModelWrapper(env, device=args_cli.device)

    if agent_cfg["env"].get("collision_avoidance", False):
        env = KlaskRlCollisionAvoidanceWrapper(env)

    if KLASK_PARAMS["action_history"] > 0:
        env = ActionHistoryWrapper(env, history_length=KLASK_PARAMS["action_history"])

    obs_noise = agent_cfg["env"].get("obs_noise", 0.0)
    if obs_noise > 0.0:
        env = ObservationNoiseWrapper(env, obs_noise)

    head_to_head = args_cli.opponent_checkpoint is not None
    if agent_cfg["params"]["config"].get("self_play", False) or head_to_head:
        env = KlaskRlAgentOpponentWrapper(env)
    else:
        env = KlaskRlRandomOpponentWrapper(env)
    if "rewards" in agent_cfg.keys():
        env = CurriculumWrapper(env, agent_cfg["rewards"], mode="test")

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs=clip_obs, clip_actions=clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    if agent_cfg["params"]["config"].get("self_play", False) or head_to_head:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnvSelfPlay(
                config_name, num_actors, agent_cfg.copy(), **kwargs
            ),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
        )

    else:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
        )

    # set active termination terms specified in agent_cfg:
    if "terminations" in agent_cfg.keys():
        set_terminations(env, agent_cfg["terminations"])

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    agent.device = torch.device(args_cli.device)
    agent.model.to(args_cli.device)
    agent.actions_low = agent.actions_low.to(args_cli.device)
    agent.actions_high = agent.actions_high.to(args_cli.device)
    # reset environment
    obs = env.reset()
    rewards = []
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()

    if head_to_head:
        # Load opponent from a separate checkpoint and (optionally) separate config
        opponent_config = agent_cfg.copy()
        opponent_config_path = args_cli.opponent_config or args_cli.config
        if opponent_config_path is not None:
            with open(opponent_config_path, "r") as file:
                opp_yaml = yaml.safe_load(file)
            opponent_config.update(opp_yaml)
        opponent_config["params"]["load_checkpoint"] = True
        opponent_config["params"]["load_path"] = args_cli.opponent_checkpoint
        opponent_config["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        opp_runner = Runner()
        opp_runner.load(opponent_config)
        opponent = opp_runner.create_player()
        opp_checkpoint = torch.load(args_cli.opponent_checkpoint, map_location=args_cli.device, weights_only=False)
        opponent.set_weights(opp_checkpoint)
        opponent.reset()
        opponent.device = torch.device(args_cli.device)
        opponent.model.to(args_cli.device)
        opponent.actions_low = opponent.actions_low.to(args_cli.device)
        opponent.actions_high = opponent.actions_high.to(args_cli.device)
        if opponent.is_rnn:
            opponent.init_rnn()
        _ = opponent.get_batch_size(obs, 1)
        find_wrapper(env, KlaskRlAgentOpponentWrapper).add_opponent(opponent)
        print(f"[INFO]: Opponent checkpoint: {args_cli.opponent_checkpoint}")
    elif agent_cfg["params"]["config"].get("self_play", False):
        opponent = runner.create_player()
        opponent.device = torch.device(args_cli.device)
        opponent.model.to(args_cli.device)
        opponent.actions_low = agent.actions_low.to(args_cli.device)
        opponent.actions_high = agent.actions_high.to(args_cli.device)
        opponent.set_weights(agent.get_weights())
        find_wrapper(env, KlaskRlAgentOpponentWrapper).add_opponent(opponent)

    # -- Termination tracking for head-to-head mode --
    term_counts = {
        "player_scored": 0,  # Episode_Termination/goal_scored
        "opponent_scored": 0,  # Episode_Termination/goal_conceded
        "player_in_goal": 0,  # Episode_Termination/player_in_goal
        "opponent_in_goal": 0,  # Episode_Termination/opponent_in_goal
        "time_expired": 0,  # Episode_Termination/time_out
    }
    total_games = 0

    TERM_NAME_MAP = {
        "goal_scored": "player_scored",
        "goal_conceded": "opponent_scored",
        "player_in_goal": "player_in_goal",
        "opponent_in_goal": "opponent_in_goal",
        "time_out": "time_expired",
    }
    term_manager = env.unwrapped.termination_manager

    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    pbar = tqdm(total=args_cli.num_games, desc="Games", disable=not head_to_head) if head_to_head else None

    start_time = time.time()
    while simulation_app.is_running() and time.time() - start_time < 1000.0:
        # In head-to-head mode, stop after the requested number of games
        if head_to_head and total_games >= args_cli.num_games:
            break

        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=True)
            # env stepping
            obs, rew, dones, info = env.step(actions)
            rewards.append(rew.detach().cpu())
            # perform operations for terminated episodes
            if len(dones) > 0:
                # --- accumulate termination stats ---
                num_done = int(dones.sum().item()) if isinstance(dones, torch.Tensor) else int(sum(dones))
                if num_done > 0:
                    total_games += num_done
                    # Use per-env termination data, counting only envs that terminated this step
                    done_mask = dones.bool().to(term_manager._term_dones.device)
                    for i, term_name in enumerate(term_manager._term_names):
                        if term_name in TERM_NAME_MAP:
                            count_key = TERM_NAME_MAP[term_name]
                            term_counts[count_key] += int(term_manager._term_dones[done_mask, i].sum().item())
                    # update progress bar with live stats
                    if pbar is not None:
                        pbar.update(min(num_done, args_cli.num_games - pbar.n))
                        p_wins = term_counts["player_scored"] + term_counts["opponent_in_goal"]
                        o_wins = term_counts["opponent_scored"] + term_counts["player_in_goal"]
                        pbar.set_postfix(
                            P_wins=p_wins,
                            O_wins=o_wins,
                            Draws=term_counts["time_expired"],
                            P_wr=f"{p_wins / total_games * 100:.1f}%",
                        )
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
                if head_to_head and opponent.is_rnn and opponent.states is not None:
                    for s in opponent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    if pbar is not None:
        pbar.close()

    # close the simulator
    env.close()

    # -- Print summary --
    if head_to_head:
        print("\n" + "=" * 50)
        print("  HEAD-TO-HEAD EVALUATION RESULTS")
        print("=" * 50)
        print(f"  Player checkpoint : {resume_path}")
        print(f"  Opponent checkpoint: {args_cli.opponent_checkpoint}")
        print(f"  Total games played : {total_games}")
        print("-" * 50)
        print(f"  Player scored (goal_scored)      : {term_counts['player_scored']}")
        print(f"  Opponent scored (goal_conceded)   : {term_counts['opponent_scored']}")
        print(f"  Player fell in goal (player_in)   : {term_counts['player_in_goal']}")
        print(f"  Opponent fell in goal (opp_in)    : {term_counts['opponent_in_goal']}")
        print(f"  Time expired (time_out)           : {term_counts['time_expired']}")
        print("-" * 50)
        player_wins = term_counts["player_scored"] + term_counts["opponent_in_goal"]
        opponent_wins = term_counts["opponent_scored"] + term_counts["player_in_goal"]
        draws = term_counts["time_expired"]
        print(f"  Player wins  : {player_wins}")
        print(f"  Opponent wins: {opponent_wins}")
        print(f"  Draws        : {draws}")
        if total_games > 0:
            print(f"  Player win rate: {player_wins / total_games * 100:.1f}%")
        print("=" * 50 + "\n")
    else:
        plt.plot(rewards, label="Reward")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
