"""Two-Stage HER (Hindsight Experience Replay) wrapper for KLASK environment.

This wrapper implements a two-stage goal-conditioned learning approach:

Stage 1 (Pre-hit): Agent learns to hit the ball
- achieved_goal: player XY position
- desired_goal: ball XY position
- HER relabels: "pretend you wanted to go where you ended up"

Stage 2 (Post-hit): Agent learns to score a goal
- achieved_goal: ball XY position (where it ended up)
- desired_goal: opponent goal center XY
- HER relabels: "pretend the goal was where the ball ended up"

The wrapper tracks whether the ball has been hit in each environment and
switches the goal structure accordingly.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class Sb3TwoStageHerWrapper(VecEnvWrapper):
    """Vectorized two-stage HER wrapper for KLASK goal-scoring task.

    This wrapper implements hierarchical goal-conditioned learning:
    1. First, learn to hit the ball (player → ball)
    2. Then, learn to score (ball → opponent goal)

    The observation space is extended with opponent_goal_center which can be
    relabeled by HER when the ball is hit but doesn't score.

    Observation structure:
    - observation: [player_pos(2), player_vel(2), opponent_pos(2), opponent_vel(2),
                    ball_pos(2), ball_vel(2), opponent_goal(2)] = 14 dims
    - achieved_goal: depends on phase (player_pos pre-hit, ball_pos post-hit)
    - desired_goal: depends on phase (ball_pos pre-hit, opponent_goal post-hit)
    """

    def __init__(
        self,
        venv: VecEnv,
        # Observation indices for extracting goals from the base observation
        player_pos_indices: tuple[int, int] = (0, 2),  # player XY position
        ball_pos_indices: tuple[int, int] = (8, 10),  # ball XY position
        # Goal configuration
        opponent_goal_center: tuple[float, float] = (0.0, 0.176215),  # opponent goal XY
        # Thresholds
        ball_hit_threshold: float = 0.02,  # distance for ball hit detection
        goal_score_threshold: float = 0.025,  # distance for goal scoring
        # Reward configuration
        ball_hit_reward: float = 1.0,  # reward for hitting ball (non-terminating)
        goal_score_reward: float = 10.0,  # reward for scoring (terminating)
    ):
        """Initialize the two-stage HER wrapper.

        Args:
            venv: The Sb3VecEnvWrapper environment (a VecEnv)
            player_pos_indices: Start and end indices for player position in obs
            ball_pos_indices: Start and end indices for ball position in obs
            opponent_goal_center: XY position of opponent goal center
            ball_hit_threshold: Distance threshold for ball hit detection
            goal_score_threshold: Distance threshold for goal scoring
            ball_hit_reward: Sparse reward given when ball is hit
            goal_score_reward: Sparse reward given when goal is scored
        """
        self.player_pos_indices = player_pos_indices
        self.ball_pos_indices = ball_pos_indices
        self.opponent_goal_center = np.array(opponent_goal_center, dtype=np.float32)
        self.ball_hit_threshold = ball_hit_threshold
        self.goal_score_threshold = goal_score_threshold
        self.ball_hit_reward = ball_hit_reward
        self.goal_score_reward = goal_score_reward

        # Get observation dimension from wrapped env
        orig_obs_space = venv.observation_space
        if isinstance(orig_obs_space, spaces.Box):
            self.base_obs_dim = orig_obs_space.shape[-1]
        else:
            raise ValueError(
                f"Expected Box observation space, got {type(orig_obs_space)}"
            )

        # Extended observation includes opponent_goal_center (2 dims)
        extended_obs_dim = self.base_obs_dim + 2
        goal_dim = 2  # XY position

        # Create GoalEnv observation space
        observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(extended_obs_dim,),
                    dtype=np.float32,
                ),
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
            }
        )

        # Initialize parent VecEnvWrapper
        super().__init__(venv, observation_space=observation_space)

        # Track ball hit state per environment
        self.num_envs = venv.num_envs
        self.ball_hit = np.zeros(self.num_envs, dtype=bool)

        # Store previous ball position for hit detection
        self._prev_ball_pos = None

    def _extend_obs_with_goal(self, obs: np.ndarray) -> np.ndarray:
        """Extend observation with opponent goal center.

        Args:
            obs: Base observation of shape (num_envs, base_obs_dim)

        Returns:
            Extended observation of shape (num_envs, base_obs_dim + 2)
        """
        # Broadcast opponent_goal_center to all envs
        goal_broadcast = np.broadcast_to(
            self.opponent_goal_center, (obs.shape[0], 2)
        ).astype(np.float32)
        return np.concatenate([obs, goal_broadcast], axis=-1)

    def _extract_goals(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Extract achieved and desired goals based on current phase.

        The goal extraction depends on whether the ball has been hit:
        - Pre-hit: achieved=player_pos, desired=ball_pos
        - Post-hit: achieved=ball_pos, desired=opponent_goal

        Args:
            obs: Base observation (before extension)

        Returns:
            Dict with 'observation', 'achieved_goal', 'desired_goal'
        """
        player_pos = obs[..., self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]

        # Initialize goals arrays
        achieved_goal = np.zeros((obs.shape[0], 2), dtype=np.float32)
        desired_goal = np.zeros((obs.shape[0], 2), dtype=np.float32)

        # Pre-hit envs: achieved=player, desired=ball
        pre_hit_mask = ~self.ball_hit
        achieved_goal[pre_hit_mask] = player_pos[pre_hit_mask]
        desired_goal[pre_hit_mask] = ball_pos[pre_hit_mask]

        # Post-hit envs: achieved=ball, desired=opponent_goal
        post_hit_mask = self.ball_hit
        achieved_goal[post_hit_mask] = ball_pos[post_hit_mask]
        desired_goal[post_hit_mask] = self.opponent_goal_center

        # Extend observation with opponent goal
        extended_obs = self._extend_obs_with_goal(obs)

        return {
            "observation": extended_obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _extract_goals_single(
        self, obs: np.ndarray, ball_hit: bool
    ) -> dict[str, np.ndarray]:
        """Extract goals from a single observation (not batched).

        Args:
            obs: Single observation (1D array)
            ball_hit: Whether ball was hit in this env

        Returns:
            Dict with goal-env formatted observation
        """
        player_pos = obs[self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[self.ball_pos_indices[0] : self.ball_pos_indices[1]]

        if ball_hit:
            achieved_goal = ball_pos
            desired_goal = self.opponent_goal_center
        else:
            achieved_goal = player_pos
            desired_goal = ball_pos

        # Extend observation with opponent goal
        extended_obs = np.concatenate([obs, self.opponent_goal_center])

        return {
            "observation": extended_obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _check_ball_hit(self, obs: np.ndarray) -> np.ndarray:
        """Check if player hit the ball in each environment.

        Args:
            obs: Current observation

        Returns:
            Boolean array indicating new hits this step
        """
        player_pos = obs[..., self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]

        distance = np.linalg.norm(player_pos - ball_pos, axis=-1)
        return distance < self.ball_hit_threshold

    def _check_goal_scored(self, obs: np.ndarray) -> np.ndarray:
        """Check if ball is in opponent goal.

        Args:
            obs: Current observation

        Returns:
            Boolean array indicating goal scored
        """
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        distance = np.linalg.norm(ball_pos - self.opponent_goal_center, axis=-1)
        return distance < self.goal_score_threshold

    def reset(self) -> dict[str, np.ndarray]:
        """Reset and return GoalEnv observation."""
        obs = self.venv.reset()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # Reset ball hit tracking
        self.ball_hit[:] = False
        self._prev_ball_pos = obs[
            ..., self.ball_pos_indices[0] : self.ball_pos_indices[1]
        ].copy()

        return self._extract_goals(obs)

    def step_wait(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        """Wait for step and return GoalEnv observation with two-stage rewards.

        Returns:
            obs: Dict with observation, achieved_goal, desired_goal
            rewards: Array of rewards (sparse: ball_hit_reward or goal_score_reward)
            dones: Array of done flags
            infos: List of info dicts
        """
        obs, base_rewards, dones, infos = self.venv.step_wait()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # Detect events
        just_hit_ball = self._check_ball_hit(obs) & ~self.ball_hit  # New hits only
        goal_scored = (
            self._check_goal_scored(obs) & self.ball_hit
        )  # Only count if ball was hit

        # Update ball hit state (sticky until reset)
        self.ball_hit = self.ball_hit | just_hit_ball

        # Compute two-stage sparse rewards
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        rewards[just_hit_ball] = self.ball_hit_reward
        rewards[goal_scored] = self.goal_score_reward

        # Override dones: only terminate on goal scored (not ball hit)
        # Note: timeout (truncation) is still handled by base env
        new_dones = dones.copy()
        # Ball hit should NOT terminate, so we don't modify dones for that
        # Goal scored SHOULD terminate
        new_dones = new_dones | goal_scored

        # Extract goal-based observations
        goal_obs = self._extract_goals(obs)

        # Add goals and phase info to infos for HER
        for i, info in enumerate(infos):
            info["achieved_goal"] = goal_obs["achieved_goal"][i]
            info["desired_goal"] = goal_obs["desired_goal"][i]
            info["ball_hit"] = self.ball_hit[i]
            info["goal_scored"] = goal_scored[i]
            info["phase"] = "post_hit" if self.ball_hit[i] else "pre_hit"

            # Convert terminal_observation to dict format for HER
            if "terminal_observation" in info:
                terminal_obs = info["terminal_observation"]
                if isinstance(terminal_obs, torch.Tensor):
                    terminal_obs = terminal_obs.cpu().numpy()
                # Use the ball_hit state at termination for goal extraction
                info["terminal_observation"] = self._extract_goals_single(
                    terminal_obs, self.ball_hit[i]
                )
                info["terminal_ball_hit"] = self.ball_hit[i]

        # Reset ball_hit for environments that are done
        self.ball_hit[dones] = False

        return goal_obs, rewards, new_dones, infos

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Compute sparse reward for HER goal relabeling.

        This method is called by HER to compute rewards for relabeled goals.
        The reward structure is simple: 1.0 if achieved_goal is close to desired_goal.

        Note: For two-stage HER, the relabeling handles the phase logic:
        - If pre-hit and relabeling to where player ended → success
        - If post-hit and relabeling to where ball ended → success

        Args:
            achieved_goal: The goal that was actually achieved
            desired_goal: The goal that was desired (possibly relabeled)
            info: Additional info (may contain 'ball_hit' flag)

        Returns:
            Sparse reward: 1.0 if distance < threshold, 0.0 otherwise
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        # Use appropriate threshold based on phase if available
        # For relabeled experiences, we use a unified threshold
        threshold = self.ball_hit_threshold

        return (distance < threshold).astype(np.float32)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: list[int] | None = None,
        **method_kwargs,
    ) -> list[Any]:
        """Call a method on the wrapped environment(s).

        Override to handle compute_reward calls for HER.
        """
        if method_name == "compute_reward":
            return [self.compute_reward(*method_args, **method_kwargs)]
        else:
            return self.venv.env_method(
                method_name, *method_args, indices=indices, **method_kwargs
            )


class Sb3TwoStageHerWrapperV2(VecEnvWrapper):
    """Alternative two-stage HER wrapper with phase-aware goal relabeling.

    This version provides more explicit control over the goal relabeling strategy
    by tracking which phase each transition belongs to and using appropriate
    thresholds for each phase.

    Key differences from V1:
    - Stores phase information in the replay buffer via info dict
    - Provides separate compute_reward methods for each phase
    - Supports custom reward shaping per phase
    """

    def __init__(
        self,
        venv: VecEnv,
        # Observation indices
        player_pos_indices: tuple[int, int] = (0, 2),
        ball_pos_indices: tuple[int, int] = (8, 10),
        # Goal configuration
        opponent_goal_center: tuple[float, float] = (0.0, 0.176215),
        # Thresholds (can be different per phase)
        phase1_threshold: float = 0.02,  # player → ball
        phase2_threshold: float = 0.025,  # ball → goal
        # Reward configuration
        phase1_reward: float = 1.0,  # ball hit
        phase2_reward: float = 10.0,  # goal scored
    ):
        """Initialize the two-stage HER wrapper V2."""
        self.player_pos_indices = player_pos_indices
        self.ball_pos_indices = ball_pos_indices
        self.opponent_goal_center = np.array(opponent_goal_center, dtype=np.float32)
        self.phase1_threshold = phase1_threshold
        self.phase2_threshold = phase2_threshold
        self.phase1_reward = phase1_reward
        self.phase2_reward = phase2_reward

        # Get observation dimension
        orig_obs_space = venv.observation_space
        if isinstance(orig_obs_space, spaces.Box):
            self.base_obs_dim = orig_obs_space.shape[-1]
        else:
            raise ValueError(
                f"Expected Box observation space, got {type(orig_obs_space)}"
            )

        # Extended observation: base + opponent_goal(2) + phase_indicator(1)
        extended_obs_dim = self.base_obs_dim + 3
        goal_dim = 2

        observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(extended_obs_dim,),
                    dtype=np.float32,
                ),
                "achieved_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
                "desired_goal": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32
                ),
            }
        )

        super().__init__(venv, observation_space=observation_space)

        self.num_envs = venv.num_envs
        self.ball_hit = np.zeros(self.num_envs, dtype=bool)

    def _extend_obs(self, obs: np.ndarray) -> np.ndarray:
        """Extend observation with opponent goal and phase indicator."""
        batch_size = obs.shape[0]
        goal_broadcast = np.broadcast_to(
            self.opponent_goal_center, (batch_size, 2)
        ).astype(np.float32)
        phase_indicator = self.ball_hit.astype(np.float32).reshape(-1, 1)
        return np.concatenate([obs, goal_broadcast, phase_indicator], axis=-1)

    def _extract_goals(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Extract goals based on current phase."""
        player_pos = obs[..., self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]

        achieved_goal = np.zeros((obs.shape[0], 2), dtype=np.float32)
        desired_goal = np.zeros((obs.shape[0], 2), dtype=np.float32)

        pre_hit = ~self.ball_hit
        post_hit = self.ball_hit

        achieved_goal[pre_hit] = player_pos[pre_hit]
        desired_goal[pre_hit] = ball_pos[pre_hit]

        achieved_goal[post_hit] = ball_pos[post_hit]
        desired_goal[post_hit] = self.opponent_goal_center

        extended_obs = self._extend_obs(obs)

        return {
            "observation": extended_obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _extract_goals_single(
        self, obs: np.ndarray, ball_hit: bool
    ) -> dict[str, np.ndarray]:
        """Extract goals from single observation."""
        player_pos = obs[self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[self.ball_pos_indices[0] : self.ball_pos_indices[1]]

        if ball_hit:
            achieved_goal = ball_pos
            desired_goal = self.opponent_goal_center
        else:
            achieved_goal = player_pos
            desired_goal = ball_pos

        phase_indicator = np.array([float(ball_hit)], dtype=np.float32)
        extended_obs = np.concatenate([obs, self.opponent_goal_center, phase_indicator])

        return {
            "observation": extended_obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _check_ball_hit(self, obs: np.ndarray) -> np.ndarray:
        """Check for ball hit."""
        player_pos = obs[..., self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        distance = np.linalg.norm(player_pos - ball_pos, axis=-1)
        return distance < self.phase1_threshold

    def _check_goal_scored(self, obs: np.ndarray) -> np.ndarray:
        """Check for goal scored."""
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        distance = np.linalg.norm(ball_pos - self.opponent_goal_center, axis=-1)
        return distance < self.phase2_threshold

    def reset(self) -> dict[str, np.ndarray]:
        """Reset environments."""
        obs = self.venv.reset()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        self.ball_hit[:] = False
        return self._extract_goals(obs)

    def step_wait(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        """Step with two-stage reward logic."""
        obs, _, dones, infos = self.venv.step_wait()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # Detect events
        just_hit = self._check_ball_hit(obs) & ~self.ball_hit
        goal_scored = self._check_goal_scored(obs) & self.ball_hit

        # Update state
        self.ball_hit = self.ball_hit | just_hit

        # Compute rewards
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        rewards[just_hit] = self.phase1_reward
        rewards[goal_scored] = self.phase2_reward

        # Terminate on goal only
        new_dones = dones | goal_scored

        # Extract goals
        goal_obs = self._extract_goals(obs)

        # Update infos
        for i, info in enumerate(infos):
            info["achieved_goal"] = goal_obs["achieved_goal"][i]
            info["desired_goal"] = goal_obs["desired_goal"][i]
            info["ball_hit"] = self.ball_hit[i]
            info["goal_scored"] = goal_scored[i]
            info["phase"] = int(self.ball_hit[i])  # 0 = pre-hit, 1 = post-hit

            if "terminal_observation" in info:
                terminal_obs = info["terminal_observation"]
                if isinstance(terminal_obs, torch.Tensor):
                    terminal_obs = terminal_obs.cpu().numpy()
                info["terminal_observation"] = self._extract_goals_single(
                    terminal_obs, self.ball_hit[i]
                )
                info["terminal_phase"] = int(self.ball_hit[i])

        # Reset state for done envs
        self.ball_hit[dones] = False

        return goal_obs, rewards, new_dones, infos

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Compute reward for HER relabeling.

        Uses a single threshold for simplicity, but the phase info in the
        observation allows the policy to behave differently per phase.
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        # Use the smaller threshold to be conservative
        threshold = min(self.phase1_threshold, self.phase2_threshold)
        return (distance < threshold).astype(np.float32)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: list[int] | None = None,
        **method_kwargs,
    ) -> list[Any]:
        """Handle compute_reward calls for HER."""
        if method_name == "compute_reward":
            return [self.compute_reward(*method_args, **method_kwargs)]
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )
