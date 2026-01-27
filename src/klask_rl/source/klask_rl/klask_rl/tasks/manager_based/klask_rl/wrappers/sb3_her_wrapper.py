"""HER (Hindsight Experience Replay) wrapper for KLASK environment.

This wrapper converts the standard IsaacLab environment into a GoalEnv
compatible format required by Stable Baselines3's HerReplayBuffer.

The key insight for HER in "hit the ball" task:
- achieved_goal: player's current XY position
- desired_goal: ball's current XY position
- When the agent fails to hit the ball, HER relabels: "pretend you wanted to
  go where you actually ended up" - creating successful experiences from failures.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class Sb3HerWrapper(gym.Wrapper):
    """Wrapper that converts IsaacLab env to GoalEnv format for HER.

    This wrapper:
    1. Extracts achieved_goal (player position) and desired_goal (ball position)
       from the existing observations
    2. Provides compute_reward() method for HER goal relabeling
    3. Keeps the original observation structure intact

    The observation indices are based on the PolicyCfg observation order:
    - peg_1_pos: indices 0-1 (player XY position) -> achieved_goal
    - ball_pos_rel: indices 8-9 (ball XY position) -> desired_goal
    """

    def __init__(
        self,
        env: gym.Env,
        achieved_goal_indices: tuple[int, int] = (0, 2),  # peg_1_pos XY
        desired_goal_indices: tuple[int, int] = (8, 10),  # ball_pos_rel XY
        distance_threshold: float = 0.02,  # Same as collision detection eps
    ):
        """Initialize the HER wrapper.

        Args:
            env: The environment to wrap
            achieved_goal_indices: Start and end indices for achieved_goal in obs
            desired_goal_indices: Start and end indices for desired_goal in obs
            distance_threshold: Distance threshold for successful goal achievement
        """
        super().__init__(env)

        self.achieved_goal_indices = achieved_goal_indices
        self.desired_goal_indices = desired_goal_indices
        self.distance_threshold = distance_threshold

        # Get the original observation space
        if isinstance(self.env.observation_space, spaces.Dict):
            # Already a dict space, get the policy observations
            if "policy" in self.env.observation_space.spaces:
                self._orig_obs_space = self.env.observation_space["policy"]
            else:
                self._orig_obs_space = list(self.env.observation_space.spaces.values())[0]
        else:
            self._orig_obs_space = self.env.observation_space

        # Determine observation dimension
        if isinstance(self._orig_obs_space, spaces.Box):
            obs_dim = self._orig_obs_space.shape[-1]
        else:
            raise ValueError(f"Unsupported observation space type: {type(self._orig_obs_space)}")

        # Goal dimension (XY position)
        goal_dim = achieved_goal_indices[1] - achieved_goal_indices[0]

        # Create GoalEnv-compatible observation space
        # For vectorized envs, we need to handle the batch dimension
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32),
                "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32),
            }
        )

    def _extract_goals(self, obs: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        """Extract achieved and desired goals from observation.

        Args:
            obs: Raw observation array of shape (obs_dim,) or (num_envs, obs_dim)

        Returns:
            Dict with 'observation', 'achieved_goal', 'desired_goal'
        """
        # Convert torch tensor to numpy if needed
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # Handle both single and batched observations
        achieved_goal = obs[..., self.achieved_goal_indices[0] : self.achieved_goal_indices[1]]
        desired_goal = obs[..., self.desired_goal_indices[0] : self.desired_goal_indices[1]]

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _process_obs(self, obs: dict | np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        """Process observation from environment into GoalEnv format.

        Args:
            obs: Observation from environment (dict with 'policy' key or raw array)

        Returns:
            GoalEnv-compatible observation dict
        """
        # Extract raw observation array
        if isinstance(obs, dict):
            if "policy" in obs:
                raw_obs = obs["policy"]
            else:
                # Take first value from dict
                raw_obs = list(obs.values())[0]
        else:
            raw_obs = obs

        return self._extract_goals(raw_obs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment and return GoalEnv-compatible observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self._process_obs(obs), info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment and return GoalEnv-compatible observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        goal_obs = self._process_obs(obs)

        # Store achieved and desired goals in info for HER
        info["achieved_goal"] = goal_obs["achieved_goal"]
        info["desired_goal"] = goal_obs["desired_goal"]

        return goal_obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Compute sparse reward for HER goal relabeling.

        This is the key method for HER - it computes the reward for any
        achieved_goal/desired_goal pair, allowing HER to relabel failed
        experiences as successful ones.

        Args:
            achieved_goal: The goal that was actually achieved (player position)
            desired_goal: The goal that was desired (ball position)
            info: Additional info (unused)

        Returns:
            Sparse reward: 1.0 if distance < threshold, 0.0 otherwise
        """
        # Compute L2 distance between achieved and desired goals
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        # Sparse reward: success if within threshold
        return (distance < self.distance_threshold).astype(np.float32)


class Sb3VecHerWrapper(VecEnvWrapper):
    """Vectorized HER wrapper for IsaacLab environments with Sb3VecEnvWrapper.

    This wrapper sits on top of the Sb3VecEnvWrapper and converts observations
    to GoalEnv format while maintaining vectorized environment compatibility.

    Inherits from VecEnvWrapper (not gym.Wrapper) because Sb3VecEnvWrapper
    is a VecEnv, not a standard Gymnasium environment.
    """

    def __init__(
        self,
        venv: VecEnv,
        achieved_goal_indices: tuple[int, int] = (0, 2),  # peg_1_pos XY
        desired_goal_indices: tuple[int, int] = (8, 10),  # ball_pos_rel XY
        distance_threshold: float = 0.02,
    ):
        """Initialize the vectorized HER wrapper.

        Args:
            venv: The Sb3VecEnvWrapper environment (a VecEnv)
            achieved_goal_indices: Start and end indices for achieved_goal
            desired_goal_indices: Start and end indices for desired_goal
            distance_threshold: Distance threshold for goal achievement
        """
        self.achieved_goal_indices = achieved_goal_indices
        self.desired_goal_indices = desired_goal_indices
        self.distance_threshold = distance_threshold

        # Get observation dimension from wrapped env
        orig_obs_space = venv.observation_space
        if isinstance(orig_obs_space, spaces.Box):
            obs_dim = orig_obs_space.shape[-1]
        else:
            raise ValueError(f"Expected Box observation space, got {type(orig_obs_space)}")

        goal_dim = achieved_goal_indices[1] - achieved_goal_indices[0]

        # Create GoalEnv observation space (single env version for SB3)
        observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32),
                "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32),
            }
        )

        # Initialize parent VecEnvWrapper with the new observation space
        super().__init__(venv, observation_space=observation_space)

    def _extract_goals(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Extract goals from batched observations."""
        achieved_goal = obs[..., self.achieved_goal_indices[0] : self.achieved_goal_indices[1]]
        desired_goal = obs[..., self.desired_goal_indices[0] : self.desired_goal_indices[1]]

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _extract_goals_single(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Extract goals from a single observation (not batched)."""
        achieved_goal = obs[self.achieved_goal_indices[0] : self.achieved_goal_indices[1]]
        desired_goal = obs[self.desired_goal_indices[0] : self.desired_goal_indices[1]]

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def reset(self) -> dict[str, np.ndarray]:
        """Reset and return GoalEnv observation."""
        obs = self.venv.reset()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        return self._extract_goals(obs)

    def step_wait(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        """Wait for step and return GoalEnv observation.

        Returns:
            obs: Dict with observation, achieved_goal, desired_goal
            rewards: Array of rewards
            dones: Array of done flags
            infos: List of info dicts
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        goal_obs = self._extract_goals(obs)

        # Add goals to infos for HER (required for goal relabeling)
        # Also convert terminal_observation to dict format if present
        for i, info in enumerate(infos):
            info["achieved_goal"] = goal_obs["achieved_goal"][i]
            info["desired_goal"] = goal_obs["desired_goal"][i]

            # CRITICAL: Convert terminal_observation to dict format for HER
            # When episode terminates, SB3 stores the final observation in info
            # HER needs this in dict format too
            if "terminal_observation" in info:
                terminal_obs = info["terminal_observation"]
                if isinstance(terminal_obs, torch.Tensor):
                    terminal_obs = terminal_obs.cpu().numpy()
                # Convert to goal dict format
                info["terminal_observation"] = self._extract_goals_single(terminal_obs)

        return goal_obs, rewards, dones, infos

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Compute sparse reward for HER goal relabeling.

        This is the key method for HER - it computes the reward for any
        achieved_goal/desired_goal pair, allowing HER to relabel failed
        experiences as successful ones.

        Args:
            achieved_goal: The goal that was actually achieved (player position)
            desired_goal: The goal that was desired (ball position)
            info: Additional info (unused)

        Returns:
            Sparse reward: 1.0 if distance < threshold, 0.0 otherwise
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (distance < self.distance_threshold).astype(np.float32)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: list[int] | None = None,
        **method_kwargs,
    ) -> list[Any]:
        """Call a method on the wrapped environment(s).

        Override to handle compute_reward calls for HER, which needs to be
        handled by this wrapper rather than the underlying environment.

        Args:
            method_name: Name of the method to call
            indices: Indices of envs to call method on (None = all)
            *method_args: Positional arguments for the method
            **method_kwargs: Keyword arguments for the method

        Returns:
            List of return values from each environment
        """
        if method_name == "compute_reward":
            # Handle compute_reward directly - HER needs this for goal relabeling
            # The args are (achieved_goal, desired_goal, info)
            return [self.compute_reward(*method_args, **method_kwargs)]
        else:
            # Pass through to underlying environment
            return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)
