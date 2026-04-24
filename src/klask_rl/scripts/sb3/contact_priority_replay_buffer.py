"""Replay buffer with contact-priority stratified sampling for SB3 SAC.

Transitions from episodes where ball contact occurred are sampled with
higher probability.  When no contact transitions exist yet, falls back
to uniform sampling.

Usage with SB3::

    agent = SAC(
        ...,
        replay_buffer_class=ContactPriorityReplayBuffer,
        replay_buffer_kwargs={"contact_ratio": 0.8},
    )
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class ContactPriorityReplayBuffer(ReplayBuffer):
    """ReplayBuffer that over-samples transitions from episodes with ball contact.

    Extends SB3's ReplayBuffer with a ``had_contact`` flag per transition.
    On :meth:`sample`, *contact_ratio* of the batch is drawn from
    contact-flagged transitions and the rest uniformly.

    The contact flag is read from ``infos[i]["ball_hit"]`` (set by
    :class:`Sb3ContactTrackingWrapper`).  When an episode ends (``done=True``)
    all transitions in that episode are back-filled with the episode's
    contact status.

    Args:
        buffer_size: Max elements in the buffer.
        observation_space: Observation space.
        action_space: Action space.
        device: PyTorch device.
        n_envs: Number of parallel environments.
        contact_ratio: Fraction of each batch drawn from contact episodes.
        optimize_memory_usage: Memory-efficient variant (see SB3 docs).
        handle_timeout_termination: Handle timeout separately.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        contact_ratio: float = 0.8,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.contact_ratio = contact_ratio

        # Per-transition contact flag: [buffer_size, n_envs]
        self.had_contact = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool_)

        # Episode tracking per env: start position and running contact flag
        self._ep_start = np.zeros(self.n_envs, dtype=np.int64)
        self._ep_contact = np.zeros(self.n_envs, dtype=np.bool_)

        # Stats for logging
        self.contact_sample_frac = 0.0
        self.contact_buffer_frac = 0.0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Read contact flags from info dicts before parent modifies self.pos
        current_pos = self.pos
        for i, info in enumerate(infos):
            if info.get("ball_hit", False) or info.get("terminal_ball_hit", False):
                self._ep_contact[i] = True

        # Clear contact flag at current write position (may be overwriting old data)
        self.had_contact[current_pos] = False

        # Call parent add (advances self.pos, may wrap)
        super().add(obs, next_obs, action, reward, done, infos)

        # Back-fill contact flag for completed episodes
        done_arr = np.asarray(done).flatten()
        for i in range(self.n_envs):
            if done_arr[i]:
                start = self._ep_start[i]
                end = current_pos
                contact = self._ep_contact[i]

                if end >= start:
                    self.had_contact[start : end + 1, i] = contact
                else:
                    # Circular wrap
                    self.had_contact[start :, i] = contact
                    self.had_contact[: end + 1, i] = contact

                # Reset episode tracking
                self._ep_start[i] = self.pos  # next write position
                self._ep_contact[i] = False

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos

        if self.contact_ratio > 0:
            # Build contact pool: all (pos, env) pairs with had_contact=True
            contact_mask = self.had_contact[:upper_bound]
            contact_indices = np.argwhere(contact_mask)  # [N, 2] with (pos, env)

            if len(contact_indices) > 0:
                n_contact = int(batch_size * self.contact_ratio)
                n_uniform = batch_size - n_contact

                # Sample from contact pool
                contact_sample_idx = np.random.randint(0, len(contact_indices), size=n_contact)
                contact_samples = contact_indices[contact_sample_idx]
                contact_batch_inds = contact_samples[:, 0]
                contact_env_inds = contact_samples[:, 1]

                # Uniform samples
                uniform_batch_inds = np.random.randint(0, upper_bound, size=n_uniform)
                uniform_env_inds = np.random.randint(0, self.n_envs, size=n_uniform)

                batch_inds = np.concatenate([contact_batch_inds, uniform_batch_inds])
                env_indices = np.concatenate([contact_env_inds, uniform_env_inds])

                # Shuffle to avoid any ordering bias
                perm = np.random.permutation(batch_size)
                batch_inds = batch_inds[perm]
                env_indices = env_indices[perm]

                self.contact_sample_frac = n_contact / batch_size
            else:
                # No contact transitions yet — fall back to uniform
                batch_inds = np.random.randint(0, upper_bound, size=batch_size)
                env_indices = np.random.randint(0, self.n_envs, size=batch_size)
                self.contact_sample_frac = 0.0
        else:
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            env_indices = np.random.randint(0, self.n_envs, size=batch_size)
            self.contact_sample_frac = 0.0

        # Update buffer stats
        total_transitions = upper_bound * self.n_envs
        if total_transitions > 0:
            self.contact_buffer_frac = self.had_contact[:upper_bound].sum() / total_transitions
        else:
            self.contact_buffer_frac = 0.0

        return self._get_samples_with_env_indices(batch_inds, env_indices, env=env)

    def _get_samples_with_env_indices(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        """Like _get_samples but with pre-determined env_indices."""
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
