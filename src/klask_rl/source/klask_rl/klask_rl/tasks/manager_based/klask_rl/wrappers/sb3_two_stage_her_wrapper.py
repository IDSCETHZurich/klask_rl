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

IMPORTANT: This wrapper does NOT modify observations or rewards!
- Observations (including goal position) are provided by the env's observation manager
- Rewards (ball hit, goal scored) are provided by the env's reward manager
- Terminations are provided by the env's termination manager

This wrapper ONLY:
1. Converts flat observations to GoalEnv dict format (observation, achieved_goal, desired_goal)
2. Tracks ball_hit phase for goal extraction
3. Provides compute_reward for HER goal relabeling
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class Sb3TwoStageHerWrapper(VecEnvWrapper):
    """Vectorized two-stage HER wrapper for KLASK goal-scoring task.

    This wrapper converts the flat observation from the environment into
    the GoalEnv dict format required by HER, implementing two-stage goal logic:

    1. Pre-hit phase: achieved=player_pos, desired=ball_pos
    2. Post-hit phase: achieved=ball_pos, desired=opponent_goal

    The observation space from the wrapped env should already include:
    - player_pos (2): indices 0-1
    - player_vel (2): indices 2-3
    - opponent_pos (2): indices 4-5
    - opponent_vel (2): indices 6-7
    - ball_pos (2): indices 8-9
    - ball_vel (2): indices 10-11
    - opponent_goal (2): indices 12-13
    Total: 14 dimensions

    This wrapper does NOT modify the observations or rewards - it only
    restructures them for HER compatibility.
    """

    def __init__(
        self,
        venv: VecEnv,
        # Observation indices for extracting goals
        player_pos_indices: tuple[int, int] = (0, 2),  # player XY position
        ball_pos_indices: tuple[int, int] = (8, 10),  # ball XY position
        goal_pos_indices: tuple[int, int] = (12, 14),  # opponent goal XY in obs
        # Thresholds for phase detection and HER reward computation
        ball_hit_threshold: float = 0.02,  # distance for ball hit detection
        goal_score_threshold: float = 0.025,  # distance for goal scoring
    ):
        """Initialize the two-stage HER wrapper.

        Args:
            venv: The Sb3VecEnvWrapper environment (a VecEnv)
            player_pos_indices: Start and end indices for player position in obs
            ball_pos_indices: Start and end indices for ball position in obs
            goal_pos_indices: Start and end indices for opponent goal position in obs
            ball_hit_threshold: Distance threshold for ball hit detection
            goal_score_threshold: Distance threshold for goal scoring
        """
        self.player_pos_indices = player_pos_indices
        self.ball_pos_indices = ball_pos_indices
        self.goal_pos_indices = goal_pos_indices
        self.ball_hit_threshold = ball_hit_threshold
        self.goal_score_threshold = goal_score_threshold

        # Get observation dimension from wrapped env
        orig_obs_space = venv.observation_space
        if isinstance(orig_obs_space, spaces.Box):
            self.obs_dim = orig_obs_space.shape[-1]
        else:
            raise ValueError(
                f"Expected Box observation space, got {type(orig_obs_space)}"
            )

        # Verify expected observation dimension (14 for two-stage HER env)
        expected_dim = 14
        if self.obs_dim != expected_dim:
            print(
                f"[Sb3TwoStageHerWrapper] Warning: Expected obs_dim={expected_dim}, got {self.obs_dim}"
            )
            print(
                f"  Make sure to use TwoStageHerObservationsCfg which includes opponent_goal"
            )

        goal_dim = 2  # XY position

        # Create GoalEnv observation space (same obs, just restructured)
        observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
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

        # Print configuration for debugging
        print(f"[Sb3TwoStageHerWrapper] Initialized with:")
        print(f"  - obs_dim: {self.obs_dim}")
        print(f"  - player_pos_indices: {self.player_pos_indices}")
        print(f"  - ball_pos_indices: {self.ball_pos_indices}")
        print(f"  - goal_pos_indices: {self.goal_pos_indices}")
        print(f"  - ball_hit_threshold: {self.ball_hit_threshold}")
        print(f"  - goal_score_threshold: {self.goal_score_threshold}")

    def _extract_goals(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Extract achieved and desired goals based on current phase.

        The goal extraction depends on whether the ball has been hit:
        - Pre-hit: achieved=player_pos, desired=ball_pos
        - Post-hit: achieved=ball_pos, desired=opponent_goal (from obs)

        Args:
            obs: Observation array of shape (num_envs, obs_dim)

        Returns:
            Dict with 'observation', 'achieved_goal', 'desired_goal'
        """
        player_pos = obs[..., self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        goal_pos = obs[..., self.goal_pos_indices[0] : self.goal_pos_indices[1]]

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
        desired_goal[post_hit_mask] = goal_pos[post_hit_mask]

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _extract_goals_single(
        self, obs: np.ndarray, ball_hit: bool
    ) -> dict[str, np.ndarray]:
        """Extract goals from a single observation (for terminal_observation).

        Args:
            obs: Single observation (1D array)
            ball_hit: Whether ball was hit in this env

        Returns:
            Dict with goal-env formatted observation
        """
        player_pos = obs[self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        goal_pos = obs[self.goal_pos_indices[0] : self.goal_pos_indices[1]]

        if ball_hit:
            achieved_goal = ball_pos
            desired_goal = goal_pos
        else:
            achieved_goal = player_pos
            desired_goal = ball_pos

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _check_ball_hit(self, obs: np.ndarray) -> np.ndarray:
        """Check if player hit the ball in each environment.

        Args:
            obs: Current observation

        Returns:
            Boolean array indicating ball hit this step
        """
        player_pos = obs[..., self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]

        distance = np.linalg.norm(player_pos - ball_pos, axis=-1)
        return distance < self.ball_hit_threshold

    def reset(self) -> dict[str, np.ndarray]:
        """Reset and return GoalEnv observation."""
        obs = self.venv.reset()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # Reset ball hit tracking
        self.ball_hit[:] = False

        return self._extract_goals(obs)

    def step_wait(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        """Wait for step and return GoalEnv observation.

        IMPORTANT: This wrapper does NOT modify rewards or dones!
        The env's reward manager and termination manager handle those.
        This wrapper only restructures observations for HER.

        Returns:
            obs: Dict with observation, achieved_goal, desired_goal
            rewards: Array of rewards (unchanged from env)
            dones: Array of done flags (unchanged from env)
            infos: List of info dicts (with goal info added)
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()

        # Detect ball hit (new hits only, not already hit)
        just_hit_ball = self._check_ball_hit(obs) & ~self.ball_hit

        # Update ball hit state (sticky until reset)
        self.ball_hit = self.ball_hit | just_hit_ball

        # Extract goal-based observations
        goal_obs = self._extract_goals(obs)

        # Add goals and phase info to infos for HER
        for i, info in enumerate(infos):
            info["achieved_goal"] = goal_obs["achieved_goal"][i]
            info["desired_goal"] = goal_obs["desired_goal"][i]
            info["ball_hit"] = self.ball_hit[i]
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
        self.ball_hit[dones.astype(bool)] = False

        return goal_obs, rewards, dones, infos

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Compute sparse reward for HER goal relabeling.

        This method is called by HER to compute rewards for relabeled goals.
        The reward is 1.0 if achieved_goal is close to desired_goal.

        For two-stage HER:
        - Pre-hit phase: player reached ball position -> success
        - Post-hit phase: ball reached goal position -> success

        Args:
            achieved_goal: The goal that was actually achieved
            desired_goal: The goal that was desired (possibly relabeled by HER)
            info: Additional info (unused)

        Returns:
            Sparse reward: 1.0 if distance < threshold, 0.0 otherwise
        """
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)

        # Use the smaller threshold to be conservative
        threshold = min(self.ball_hit_threshold, self.goal_score_threshold)

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
