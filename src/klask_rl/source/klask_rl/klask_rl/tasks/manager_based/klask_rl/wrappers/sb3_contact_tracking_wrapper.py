"""Lightweight VecEnv wrapper that tracks ball contact for priority replay.

Detects ball contact from observation distances and injects ``ball_hit``
/ ``goal_scored`` flags into the info dicts so that
:class:`ContactPriorityReplayBuffer` can mark episodes accordingly.

Does NOT modify observations, rewards, actions, or termination signals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class Sb3ContactTrackingWrapper(VecEnvWrapper):
    """Tracks per-env ball contact and injects info flags.

    Ball contact is detected when ``dist(player_pos, ball_pos) < threshold``.
    Once contact is detected in an episode, ``ball_hit=True`` is set in the
    info dict for all subsequent steps until the episode resets.

    For terminal observations (``done=True``), ``terminal_ball_hit`` is also
    set so the replay buffer can read it after environment auto-reset.

    Args:
        venv: The wrapped SB3 VecEnv.
        player_pos_indices: ``(start, end)`` slice indices for player XY in obs.
        ball_pos_indices: ``(start, end)`` slice indices for ball XY in obs.
        ball_hit_threshold: Distance threshold for contact detection.
        goal_pos_indices: ``(start, end)`` slice indices for opponent goal XY in obs.
        goal_score_threshold: Distance threshold for goal scoring detection.
    """

    def __init__(
        self,
        venv: VecEnv,
        player_pos_indices: tuple[int, int] = (0, 2),
        ball_pos_indices: tuple[int, int] = (8, 10),
        ball_hit_threshold: float = 0.017,
        goal_pos_indices: tuple[int, int] | None = None,
        goal_score_threshold: float = 0.025,
    ):
        super().__init__(venv)
        self.player_pos_indices = player_pos_indices
        self.ball_pos_indices = ball_pos_indices
        self.ball_hit_threshold = ball_hit_threshold
        self.goal_pos_indices = goal_pos_indices
        self.goal_score_threshold = goal_score_threshold

        self._ball_hit = np.zeros(self.num_envs, dtype=bool)

        print(f"[Sb3ContactTrackingWrapper] Initialized:")
        print(f"  - player_pos_indices: {self.player_pos_indices}")
        print(f"  - ball_pos_indices: {self.ball_pos_indices}")
        print(f"  - ball_hit_threshold: {self.ball_hit_threshold}")
        if self.goal_pos_indices is not None:
            print(f"  - goal_pos_indices: {self.goal_pos_indices}")
            print(f"  - goal_score_threshold: {self.goal_score_threshold}")

    def _check_contact(self, obs: np.ndarray) -> np.ndarray:
        """Return bool array [num_envs] of contact detections."""
        player_pos = obs[:, self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[:, self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        dist = np.linalg.norm(player_pos - ball_pos, axis=-1)
        return dist < self.ball_hit_threshold

    def _check_goal(self, obs: np.ndarray) -> np.ndarray:
        """Return bool array [num_envs] of goal scoring detections."""
        if self.goal_pos_indices is None:
            return np.zeros(self.num_envs, dtype=bool)
        ball_pos = obs[:, self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        goal_pos = obs[:, self.goal_pos_indices[0] : self.goal_pos_indices[1]]
        dist = np.linalg.norm(ball_pos - goal_pos, axis=-1)
        return dist < self.goal_score_threshold

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs, rewards, dones, infos = self.venv.step_wait()

        # Detect new contacts this step
        contact_now = self._check_contact(obs)
        self._ball_hit |= contact_now

        # Detect goal scoring
        goal_now = self._check_goal(obs)

        # Inject flags into info dicts
        for i in range(self.num_envs):
            infos[i]["ball_hit"] = bool(self._ball_hit[i])
            infos[i]["goal_scored"] = bool(goal_now[i])

            # For terminal observations, also set terminal flags
            if dones[i]:
                infos[i]["terminal_ball_hit"] = bool(self._ball_hit[i])
                infos[i]["terminal_goal_scored"] = bool(goal_now[i])

        # Reset tracking for done environments
        self._ball_hit[dones.astype(bool)] = False

        return obs, rewards, dones, infos

    def reset(self) -> np.ndarray:
        self._ball_hit[:] = False
        return self.venv.reset()
