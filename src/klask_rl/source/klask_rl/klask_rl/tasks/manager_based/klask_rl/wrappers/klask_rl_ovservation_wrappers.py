import torch
import gymnasium as gym
from gymnasium import Wrapper, ObservationWrapper


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


class OpponentObservationWrapper(Wrapper):  # TODO: why does this inherite from Wrapper and not ObservationWrapper?
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
