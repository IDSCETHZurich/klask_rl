import random
import re
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium import Wrapper
from isaaclab_rl.rl_games import RlGamesGpuEnv
from rl_games.torch_runner import Runner

from .klask_rl_ovservation_wrappers import OpponentObservationWrapper
from .utils import find_wrapper


class KlaskRlRandomOpponentWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Override action space to reflect only player actions (2 instead of 4)
        # We need to modify the unwrapped environment's action space because
        # RlGamesVecEnvWrapper accesses it directly via self.unwrapped.single_action_space
        if hasattr(self.env.unwrapped, "single_action_space"):
            original_space = self.env.unwrapped.single_action_space
            if hasattr(original_space, "shape") and original_space.shape[0] == 4:
                # Store the original action space for restoration if needed
                self.env.unwrapped._klask_original_single_action_space = original_space
                # Replace with a 2-action version (only player actions, opponent is random)
                self.env.unwrapped.single_action_space = gym.spaces.Box(
                    low=original_space.low[:2],
                    high=original_space.high[:2],
                    dtype=original_space.dtype,
                )

    def step(self, actions, *args, **kwargs):
        # CAREFUL: Opponent actions
        if actions.size() == torch.Size([4096, 4]):
            actions[:, 2:] = 2 * torch.rand_like(actions)[:, :2] - 1
        else:
            opponent_actions = 2 * torch.rand_like(actions) - 1  # shape: [4096, 2]
            actions = torch.cat([actions, opponent_actions], dim=1)
        return self.env.step(actions, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)


class RlGamesGpuEnvSelfPlay(RlGamesGpuEnv):
    def __init__(
        self,
        config_name,
        num_actors,
        config,
        training_curriculum=False,
        mode=None,
        folder=None,
        is_deterministic=False,
        **kwargs,
    ):
        self.agent = None
        self.config = config
        self.instance_device = config["params"]["config"]["device"]
        self.is_deterministic = is_deterministic
        self.sum_rewards = 0
        self.training_curriculum = training_curriculum

        if training_curriculum:
            self.mode = mode
            self.base_folder = Path(folder)
            self.current_checkpoint = None
            self.config_path = None
            self.counter = 0
            # self.folder_path_checkpoint = Path("/home/student/klask_rl/IsaacLab/logs/rl_games/klask/training_curriculum/best_agent")
            # self.folder_path_config = Path("/home/student/klask_rl/IsaacLab/planned_runs/training_curriculum")
            # self.current_checkpoint = str(list(self.folder_path_checkpoint.glob("*"))[-1])
        # elif random_pool_curriculum:
        #    self.folder_path_checkpoint = Path("/home/student/klask_rl/IsaacLab/logs/rl_games/klask/pool_of_players")
        self.current_config = self.config
        super().__init__(config_name, num_actors, **kwargs)

    def reset(self):
        if self.training_curriculum:
            self.should_update_agents()
        if self.agent is None:
            self.create_agent()
        # if self.training_curriculum and new_file:

        obs = self.env.reset()
        # self.opponent_obs = self.get_opponent_obs(obs)
        self.opponent_obs = find_wrapper(self.env, OpponentObservationWrapper).opponent_obs
        self.sum_rewards = 0
        return obs

    def create_agent(self):
        runner = Runner()
        from rl_games.common.env_configurations import get_env_info

        self.config["params"]["config"]["env_info"] = get_env_info(self.env)
        runner.load(self.current_config)

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        restore_checkpoint = self.training_curriculum and self.current_checkpoint is not None
        self.agent = runner.create_player()
        if restore_checkpoint:
            self.agent.restore(self.current_checkpoint)

        self.agent.has_batch_dimension = True

    def should_update_agents(self, weights=None):
        if self.mode == 2 and self.counter > 15:
            use_old_opp = random.random() < 0.3
            if not use_old_opp and weights is not None:
                self.set_weights(indices=None, weigths=weights)
                self.counter = 1
                return

            base_folder = Path(self.base_folder)
            checkpoints = list(base_folder.glob("last*.pth"))

            if len(checkpoints) == 0:
                return
            checkpoints_sorted = sorted(checkpoints, key=lambda f: f.stat().st_ctime)
            decay = 0.9  # Closer to 1 → slower decay, closer to 0 → steeper bias to latest
            n = len(checkpoints_sorted)
            weights = np.array([decay ** (n - i - 1) for i in range(n)])  # newest gets highest weight
            weights /= weights.sum()  # normalize to sum to 1

            # Randomly choose using the computed weights
            self.current_checkpoint = np.random.choice(checkpoints_sorted, p=weights)

            self.create_agent()
            self.counter = 0

        if self.mode == 1:
            matching_file = str(list(self.base_folder.glob("*"))[-1])
            if matching_file == self.current_checkpoint:
                return
            self.current_checkpoint = matching_file

            match = re.search(r"best_agent\((\d+)\)", self.current_checkpoint)
            if match:
                agent_number = match.group(1)
            self.config_path = Path(self.base_folder) / f"klask_config_{agent_number}.yaml"

        if self.mode == 0 and self.counter > 8:
            agent_folders = [f for f in self.base_folder.iterdir() if f.is_dir()]
            agent_folders_sorted = sorted(agent_folders, key=lambda f: f.stat().st_ctime)

            if len(agent_folders) > 4:  # if only the benchmark folder is in there
                folders_to_delete = agent_folders_sorted[1:-4]
                for folder in folders_to_delete:
                    shutil.rmtree(folder)

                use_self_play = random.random() < 0.5
                if use_self_play and weights is not None:
                    self.set_weights(indices=None, weigths=weights)
                    self.counter = 1
                    return
                # Choose one at random
                chosen_folder = random.choice(agent_folders)

                # Find the checkpoint and config file in the chosen folder
                checkpoint_path = next(chosen_folder.glob("**/player_checkpoint.pth"), None)
                self.config_path = next(chosen_folder.glob("**/player_config.yaml"), None)
                self.current_checkpoint = checkpoint_path

        if self.mode == 0 and self.counter > 8 and self.current_checkpoint is not None:
            with open(self.config_path, "r") as f:
                self.current_config = yaml.safe_load(f)
                self.current_config["params"]["config"]["device"] = self.instance_device
                self.current_config["params"]["config"]["device_name"] = self.instance_device

            self.create_agent()
            self.counter = 0
        self.counter += 1

    def step(self, action, *args, **kwargs):
        opponent_obs = self.agent.obs_to_torch(self.opponent_obs)
        opponent_action = self.agent.get_action(opponent_obs, self.is_deterministic)
        full_action = torch.cat([action, opponent_action], dim=1)
        obs, reward, dones, info = self.env.step(full_action, *args, **kwargs)
        # self.opponent_obs = self.get_opponent_obs(obs)
        self.opponent_obs = find_wrapper(self.env, OpponentObservationWrapper).opponent_obs
        return obs, reward, dones, info

    def set_weights(self, indices, weigths):
        print("SETTING WEIGHTS")

        self.agent.set_weights(weigths)
        self.is_deterministic = True


class KlaskRlAgentOpponentWrapper(Wrapper):
    def __init__(self, env, is_deterministic=False):
        super().__init__(env)
        self.opponent = None
        self.is_deterministic = is_deterministic

        # Override action space to reflect only player actions (2 instead of 4)
        # This allows loading checkpoints trained with 2-action output
        if hasattr(self.env.unwrapped, "single_action_space"):
            original_space = self.env.unwrapped.single_action_space
            if hasattr(original_space, "shape") and original_space.shape[0] == 4:
                # Store the original action space for restoration if needed
                self.env.unwrapped._klask_original_single_action_space = original_space
                # Replace with a 2-action version (only player actions, opponent uses separate agent)
                self.env.unwrapped.single_action_space = gym.spaces.Box(
                    low=original_space.low[:2],
                    high=original_space.high[:2],
                    dtype=original_space.dtype,
                )

    def add_opponent(self, opponent):
        self.opponent = opponent
        self.opponent.has_batch_dimension = True

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.opponent_obs = obs["opponent"]
        return obs, info

    def step(self, action, *args, **kwargs):
        opponent_obs = self.opponent.obs_to_torch(self.opponent_obs)
        opponent_action = self.opponent.get_action(opponent_obs, self.is_deterministic)
        full_action = torch.cat([action, opponent_action], dim=1)
        obs, reward, terminated, truncated, info = self.env.step(full_action, *args, **kwargs)

        self.opponent_obs = obs["opponent"]
        return obs, reward, terminated, truncated, info
