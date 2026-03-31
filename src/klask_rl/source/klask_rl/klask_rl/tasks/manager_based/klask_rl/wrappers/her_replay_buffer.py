"""HER replay buffer that recomputes done flags after goal relabeling.

The standard SB3 HerReplayBuffer relabels goals and recomputes rewards for
virtual transitions, but keeps the original done/terminated flags unchanged.
This means the agent never learns that achieving a relabeled goal should
terminate the episode.

This subclass adds a ``compute_terminated`` call (via ``env_method``) after
goal relabeling so that virtual transitions correctly mark ``done=True``
when the relabeled goal is achieved.

The wrapped environment (or its HER wrapper) must implement
``compute_terminated(achieved_goal, desired_goal, info) -> np.ndarray``
with the same signature as ``compute_reward``.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


class HerReplayBufferWithDone(HerReplayBuffer):
    """HER replay buffer that recomputes ``done`` for virtual transitions.

    After goal relabeling, calls ``compute_terminated`` on the environment
    to determine whether the relabeled goal was achieved at each timestep.
    The final ``done`` flag is the logical OR of the original (non-timeout)
    done and the newly computed terminated flag.
    """

    def _get_virtual_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """Get virtual samples with recomputed done flags.

        Mirrors the parent implementation but adds a ``compute_terminated``
        call after reward computation to update the done flags.
        """
        # Get infos and obs
        obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
        if self.copy_info_dict:
            infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]

        # Sample and set new goals
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        next_obs["desired_goal"] = new_goals

        assert (
            self.env is not None
        ), "You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions"

        # Compute new reward
        rewards = self.env.env_method(
            "compute_reward",
            next_obs["achieved_goal"],
            obs["desired_goal"],
            infos,
            indices=[0],
        )
        rewards = rewards[0].astype(np.float32)

        # Compute new terminated flag for the relabeled goal
        terminated = self.env.env_method(
            "compute_terminated",
            next_obs["achieved_goal"],
            obs["desired_goal"],
            infos,
            indices=[0],
        )
        terminated = terminated[0].astype(np.float32)

        # Combine: done if originally terminated (non-timeout) OR relabeled
        # goal is achieved
        original_dones = (
            self.dones[batch_indices, env_indices]
            * (1 - self.timeouts[batch_indices, env_indices])
        )
        dones = np.maximum(original_dones, terminated)

        obs = self._normalize_obs(obs, env)  # type: ignore[assignment]
        next_obs = self._normalize_obs(next_obs, env)  # type: ignore[assignment]

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(dones).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),  # type: ignore[attr-defined]
        )
