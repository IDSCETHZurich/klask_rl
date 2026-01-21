import numpy as np
import torch
import gymnasium as gym
from gymnasium import Wrapper, ObservationWrapper
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv
from pathlib import Path
import re
import yaml
import random
import shutil

from klask_rl.assets.robots.klask import KLASK_PARAMS


def find_wrapper(env, wrapper_type):
    """Recursively searches for a wrapper of a given type."""
    while not isinstance(env, wrapper_type):
        env = env.env  # Move to the next layer
    if isinstance(env, wrapper_type):
        return env  # Found the wrapper
    return None  # Wrapper not found


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
                    low=original_space.low[:2], high=original_space.high[:2], dtype=original_space.dtype
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


class ObservationNoiseWrapper(ObservationWrapper):
    def __init__(self, env, noise_std, indices=None):
        super().__init__(env)
        self.noise_std = noise_std
        self.indices = indices
        if self.indices is None:
            self.indices = env.unwrapped.single_action_space.shape[-1]

    def observation(self, observation):
        if type(observation) is dict:
            for k, v in observation.items():
                noise = self.noise_std * torch.randn_like(v)
                observation[k][:, self.indices] += noise[:, self.indices]
        else:
            noise = self.noise_std * torch.randn_like(observation)
            observation[:, self.indices] += noise[:, self.indices]
        return observation


class CurriculumWrapper(Wrapper):
    def __init__(self, env, cfg, num_steps=None, mode="train", dynamic=False):
        super().__init__(env)
        self.dynamic = dynamic
        self.cfg = cfg
        self.num_steps = num_steps
        self.mode = mode
        self._step = 0
        for term, weight in cfg.items():
            term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
            if type(weight) is dict:
                if type(weight["weight"]) is list:
                    _weight = weight["weight"][0]
                else:
                    _weight = weight["weight"]
                if not weight.get("per_second", False):
                    _weight /= KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]
            else:
                _weight = weight / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
            self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = _weight

    def step(self, actions):
        if self.mode == "train":
            self._step += self.env.unwrapped.num_envs
            for term, weight in self.cfg.items():
                if type(weight) is dict and type(weight["weight"]) is list:
                    term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
                    weight_step = (weight["weight"][1] - weight["weight"][0]) / self.num_steps
                    if not weight.get("per_second", False):
                        weight_step /= KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]
                    self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight += weight_step

                if self.dynamic and not (
                    term == "ball_stationary"
                    or term == "time_out_punishment"
                    or term == "time_punishment"
                    or term == "goal_scored"
                    or term == "goal_conceded"
                    or term == "opponent_in_goal"
                    or term == "player_in_goal"
                ):
                    term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
                    self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = weight * (
                        torch.exp(
                            -torch.tensor(
                                self._step / 10000000,
                                device=self.env.unwrapped.device,
                                dtype=torch.float32,
                            )
                        )
                    )  # coeff chosen sucht that half the max reward at 20 mio steps
                    if self._step > 20_000_000:
                        self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = torch.tensor(
                            0.0, device=self.env.unwrapped.device, dtype=torch.float32
                        )

        return self.env.step(actions)


class OpponentObservationWrapper(Wrapper):
    def __init__(self, env, mode="train"):
        super().__init__(env)
        self.mode = mode
        # Override action space to reflect only player actions (2 instead of 4)
        # Same as KlaskRlRandomOpponentWrapper - the opponent actions come from a separate agent
        if hasattr(self.env.unwrapped, "single_action_space"):
            original_space = self.env.unwrapped.single_action_space
            if hasattr(original_space, "shape") and original_space.shape[0] == 4:
                # Store the original action space for restoration if needed
                self.env.unwrapped._klask_original_single_action_space = original_space
                # Replace with a 2-action version (only player actions, opponent from separate agent)
                self.env.unwrapped.single_action_space = gym.spaces.Box(
                    low=original_space.low[:2], high=original_space.high[:2], dtype=original_space.dtype
                )

    def get_opponent_obs(self, obs):
        opponent_obs = obs.detach().clone()
        opponent_obs[:, :12] = -obs[:, :12]
        return opponent_obs

    def reset(self, *args, **kwargs):
        obs_dict, extras = self.env.reset(*args, **kwargs)
        if self.mode == "train":
            self.opponent_obs = self.get_opponent_obs(obs_dict["opponent"])
        else:
            obs_dict["opponent"] = self.get_opponent_obs(obs_dict["opponent"])
        return obs_dict, extras

    def step(self, actions, *args, **kwargs):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions, *args, **kwargs)
        if self.mode == "train":
            self.opponent_obs = self.get_opponent_obs(obs_dict["opponent"])
        else:
            obs_dict["opponent"] = self.get_opponent_obs(obs_dict["opponent"])
        return obs_dict, rew, terminated, truncated, extras


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
        full_action = torch.cat([action, -opponent_action], dim=1)
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

    def add_opponent(self, opponent):
        self.opponent = opponent
        self.opponent.has_batch_dimension = True

    def get_opponent_obs(self, obs):
        opponent_obs = obs.detach().clone()
        opponent_obs[:, :12] = -obs[:, :12]
        return opponent_obs

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.opponent_obs = self.get_opponent_obs(obs["opponent"])
        return obs, info

    def step(self, action, *args, **kwargs):
        opponent_obs = self.opponent.obs_to_torch(self.opponent_obs)
        opponent_action = self.opponent.get_action(opponent_obs, self.is_deterministic)
        full_action = torch.cat([action, -opponent_action], dim=1)
        obs, reward, terminated, truncated, info = self.env.step(full_action, *args, **kwargs)

        self.opponent_obs = self.get_opponent_obs(obs["opponent"])
        return obs, reward, terminated, truncated, info


class KlaskRlCollisionAvoidanceWrapper(Wrapper):
    real_to_sim_factor_long_side = 0.0008285
    real_to_sim_factor_short_side = 1 / 1150
    DEACCELERATION_DISTANCE = 0.09
    PEG_RADIUS = 0.0075
    X_EDGE = (-0.16, 0.16)
    Y_EDGE_PLAYER = (-0.03, -0.22)
    Y_EDGE_OPPONENT = (-0.03, -0.22)
    board_dimensions = (0.32, 0.44)
    speed_limit_weight = 70.0

    def __init__(self, env, max_vel=0.2):
        super().__init__(env)

        self.x_min, self.x_max = 15.0, 360.0
        self.y_min_1, self.y_max_1 = 15.0, 235.0
        self.y_min_2, self.y_max_2 = 340.0, 530.0
        self.MAX_VEL = max_vel

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.state_1 = obs["policy"].clone()[:, :2]
        self.state_2 = obs["opponent"].clone()[:, :2]

        return obs, info

    def step(self, actions, *args, **kwargs):
        left_zone = self.state_1[:, 0] <= self.X_EDGE[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        soft_dist_left = self.state_1[left_zone, 0] - self.X_EDGE[0] - self.PEG_RADIUS
        actions[left_zone, 0] = torch.maximum(actions[left_zone, 0], -self.interpolate_vel(soft_dist_left))

        left_zone_opp = self.state_2[:, 0] <= self.X_EDGE[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        soft_dist_left_opp = self.state_2[left_zone_opp, 0] - self.X_EDGE[0] - self.PEG_RADIUS
        actions[left_zone_opp, 2] = torch.maximum(actions[left_zone_opp, 2], -self.interpolate_vel(soft_dist_left_opp))

        right_zone = self.state_1[:, 0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS >= self.X_EDGE[1]
        actions[right_zone, 0] = torch.minimum(
            actions[right_zone, 0],
            self.interpolate_vel(self.X_EDGE[1] - self.state_1[right_zone, 0] - self.PEG_RADIUS),
        )

        right_zone_opp = self.state_2[:, 0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS >= self.X_EDGE[1]
        actions[right_zone_opp, 1] = torch.minimum(
            actions[right_zone_opp, 1],
            self.interpolate_vel(self.X_EDGE[1] - self.state_2[right_zone_opp, 0] - self.PEG_RADIUS),
        )

        bottom_zone = self.state_1[:, 1] <= self.Y_EDGE_PLAYER[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        actions[bottom_zone, 1] = torch.maximum(
            actions[bottom_zone, 1],
            -self.interpolate_vel(self.state_1[:, 1][bottom_zone] - self.Y_EDGE_PLAYER[0] - self.PEG_RADIUS),
        )

        bottom_zone_opp = self.state_2[:, 1] <= self.Y_EDGE_OPPONENT[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        actions[bottom_zone_opp, 1] = torch.maximum(
            actions[bottom_zone_opp, 1],
            -self.interpolate_vel(self.state_2[:, 1][bottom_zone_opp] - self.Y_EDGE_OPPONENT[0] - self.PEG_RADIUS),
        )

        # Y - TOP
        top_zone = self.state_1[:, 1] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS >= self.Y_EDGE_PLAYER[1]
        actions[top_zone, 1] = torch.minimum(
            actions[top_zone, 1],
            self.interpolate_vel(self.Y_EDGE_PLAYER[1] - self.state_1[:, 1][top_zone] + self.PEG_RADIUS),
        )

        top_zone_opp = self.state_2[:, 1] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS >= self.Y_EDGE_OPPONENT[1]
        actions[top_zone_opp, 1] = torch.minimum(
            actions[top_zone_opp, 1],
            self.interpolate_vel(self.Y_EDGE_OPPONENT[1] - self.state_2[:, 1][top_zone_opp] + self.PEG_RADIUS),
        )

        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        self.state_1 = obs["policy"].clone()[:, :2]
        self.state_2 = obs["opponent"].clone()[:, :2]

        return obs, rew, terminated, truncated, info

    def interpolate_vel(self, distance):
        return (self.MAX_VEL / self.DEACCELERATION_DISTANCE) * distance


class ActionHistoryWrapper(Wrapper):
    def __init__(self, env, history_length):
        super().__init__(env)
        self.history_length = history_length
        self.history_x_player = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)
        self.history_y_player = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)
        self.history_x_opponent = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)
        self.history_y_opponent = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions, *args, **kwargs):
        self.history_x_player[:, :-1] = self.history_x_player.clone()[:, 1:]
        self.history_x_player[:, -1] = actions[:, 0]
        self.history_y_player[:, :-1] = self.history_y_player.clone()[:, 1:]
        self.history_y_player[:, -1] = actions[:, 1]
        self.history_x_opponent[:, :-1] = self.history_x_opponent.clone()[:, 1:]
        self.history_x_opponent[:, -1] = actions[:, 2]
        self.history_y_opponent[:, :-1] = self.history_y_opponent.clone()[:, 1:]
        self.history_y_opponent[:, -1] = actions[:, 3]

        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        obs["policy"][:, 12:] = torch.cat([self.history_x_player, self.history_y_player], dim=1)
        obs["opponent"][:, 12:] = torch.cat([self.history_x_opponent, self.history_y_opponent], dim=1)
        return obs, rew, terminated, truncated, info
