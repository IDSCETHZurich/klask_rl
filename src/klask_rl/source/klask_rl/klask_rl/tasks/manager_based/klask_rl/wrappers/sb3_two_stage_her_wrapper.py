"""Two-Stage Compound-Goal HER wrapper for KLASK environment.

This wrapper implements a compound 4-dim goal approach for two-stage
goal-conditioned learning with Hindsight Experience Replay (HER).

Instead of switching goal semantics mid-episode (which breaks HER relabeling),
both stages are encoded simultaneously in every transition:

    achieved_goal = [player_pos_xy, ball_pos_xy]   (4-dim)
    desired_goal  = [ball_pos_xy, opponent_goal_xy] (4-dim)

Stage 1 (hit the ball):  achieved[0:2] vs desired[0:2]  →  player near ball?
Stage 2 (score a goal):  achieved[2:4] vs desired[2:4]  →  ball near goal?

compute_reward checks both halves independently:
- Stage 1 success (player reached ball): ball_hit_reward
- Stage 1 + Stage 2 success (ball also reached goal): ball_hit_reward + goal_score_reward

With HER ``final`` strategy, relabeled desired_goal = final achieved_goal
= [final_player_pos, final_ball_pos]. Both halves compare same-type
quantities (player↔player, ball↔ball), so relabeling is always consistent.

IMPORTANT: This wrapper does NOT modify observations or rewards from the env.
- Observations are provided by the env's observation manager
- Rewards (ball hit, goal scored) are provided by the env's reward manager
- Terminations are provided by the env's termination manager

This wrapper ONLY:
1. Converts flat observations to GoalEnv dict format with 4-dim compound goals
2. Tracks ball_hit / goal_scored state for the info dict (metrics callback)
3. Provides compute_reward for HER goal relabeling
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class Sb3TwoStageHerWrapper(VecEnvWrapper):
    """Vectorized compound-goal HER wrapper for KLASK two-stage task.

    Uses a 4-dim compound goal that encodes both stages simultaneously,
    avoiding the semantic inconsistency of switching goal meanings mid-episode.

    Compound goal layout:
        achieved_goal = [player_pos_x, player_pos_y, ball_pos_x, ball_pos_y]
        desired_goal  = [ball_pos_x,   ball_pos_y,   goal_x,     goal_y    ]

    The observation space from the wrapped env should include:
    - player_pos (2): indices 0-1
    - player_vel (2): indices 2-3
    - opponent_pos (2): indices 4-5
    - opponent_vel (2): indices 6-7
    - ball_pos (2): indices 8-9
    - ball_vel (2): indices 10-11
    - opponent_goal (2): indices 12-13
    Total: 14 dimensions
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
        # Reward scales for HER virtual transitions
        ball_hit_reward: float = 0.0,  # reward for hitting ball
        goal_score_reward: float = 0.0,  # reward for scoring goal
    ):
        """Initialize the compound-goal two-stage HER wrapper.

        Args:
            venv: The Sb3VecEnvWrapper environment (a VecEnv)
            player_pos_indices: Start and end indices for player position in obs
            ball_pos_indices: Start and end indices for ball position in obs
            goal_pos_indices: Start and end indices for opponent goal position in obs
            ball_hit_threshold: Distance threshold for stage 1 (player hit ball)
            goal_score_threshold: Distance threshold for stage 2 (ball in goal)
            ball_hit_reward: HER reward for stage 1 success
            goal_score_reward: HER reward for stage 1 + stage 2 success (added on top)
        """
        self.player_pos_indices = player_pos_indices
        self.ball_pos_indices = ball_pos_indices
        self.goal_pos_indices = goal_pos_indices
        self.ball_hit_threshold = ball_hit_threshold
        self.goal_score_threshold = goal_score_threshold
        self.ball_hit_reward = ball_hit_reward
        self.goal_score_reward = goal_score_reward

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
                "Make sure to use TwoStageHerObservationsCfg which includes opponent_goal"
            )

        # Compound goal: [player_pos(2), ball_pos(2)] and [ball_pos(2), goal_pos(2)]
        goal_dim = 4

        # Create GoalEnv observation space with 4-dim compound goals
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

        # Track ball hit state per environment (for info dict / metrics only)
        self.num_envs = venv.num_envs
        self.ball_hit = np.zeros(self.num_envs, dtype=bool)

        # Print configuration for debugging
        print("[Sb3TwoStageHerWrapper] Initialized with COMPOUND 4-dim goals:")
        print(f"  - obs_dim: {self.obs_dim}")
        print(f"  - goal_dim: {goal_dim}")
        print(f"  - player_pos_indices: {self.player_pos_indices}")
        print(f"  - ball_pos_indices: {self.ball_pos_indices}")
        print(f"  - goal_pos_indices: {self.goal_pos_indices}")
        print(f"  - ball_hit_threshold: {self.ball_hit_threshold}")
        print(f"  - goal_score_threshold: {self.goal_score_threshold}")
        print(f"  - ball_hit_reward: {self.ball_hit_reward}")
        print(f"  - goal_score_reward: {self.goal_score_reward}")

    # ------------------------------------------------------------------
    # Goal extraction (no phase branching — always the same layout)
    # ------------------------------------------------------------------

    def _extract_goals(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Extract compound 4-dim achieved/desired goals from observations.

        achieved_goal = [player_pos_xy, ball_pos_xy]
        desired_goal  = [ball_pos_xy,   opponent_goal_xy]

        Args:
            obs: Observation array of shape (num_envs, obs_dim)

        Returns:
            Dict with 'observation', 'achieved_goal', 'desired_goal'
        """
        player_pos = obs[..., self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        goal_pos = obs[..., self.goal_pos_indices[0] : self.goal_pos_indices[1]]

        achieved_goal = np.concatenate([player_pos, ball_pos], axis=-1)
        desired_goal = np.concatenate([ball_pos, goal_pos], axis=-1)

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    def _extract_goals_single(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Extract compound goals from a single observation (for terminal_observation).

        Args:
            obs: Single observation (1D array)

        Returns:
            Dict with goal-env formatted observation
        """
        player_pos = obs[self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        goal_pos = obs[self.goal_pos_indices[0] : self.goal_pos_indices[1]]

        achieved_goal = np.concatenate([player_pos, ball_pos])
        desired_goal = np.concatenate([ball_pos, goal_pos])

        return {
            "observation": obs.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Ball hit / goal scored detection (for info dict & metrics only)
    # ------------------------------------------------------------------

    def _check_ball_hit(self, obs: np.ndarray) -> np.ndarray:
        """Check if player hit the ball in each environment."""
        player_pos = obs[..., self.player_pos_indices[0] : self.player_pos_indices[1]]
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        distance = np.linalg.norm(player_pos - ball_pos, axis=-1)
        return distance < self.ball_hit_threshold

    def _check_goal_scored(self, obs: np.ndarray) -> np.ndarray:
        """Check if ball reached the opponent goal in each environment."""
        ball_pos = obs[..., self.ball_pos_indices[0] : self.ball_pos_indices[1]]
        goal_pos = obs[..., self.goal_pos_indices[0] : self.goal_pos_indices[1]]
        distance = np.linalg.norm(ball_pos - goal_pos, axis=-1)
        return distance < self.goal_score_threshold

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

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

        Returns:
            obs: Dict with observation, achieved_goal (4-dim), desired_goal (4-dim)
            rewards: Array of rewards (unchanged from env)
            dones: Array of done flags (unchanged from env)
            infos: List of info dicts (with goal info and ball_hit/goal_scored flags)
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()

        # Detect ball hit (new hits only, not already hit) — for info/metrics
        just_hit_ball = self._check_ball_hit(obs) & ~self.ball_hit
        self.ball_hit = self.ball_hit | just_hit_ball

        # Check goal scored — for info/metrics
        goal_scored = self._check_goal_scored(obs)

        # Extract compound goal observations (no phase branching)
        goal_obs = self._extract_goals(obs)

        # Add goals and tracking info for HER and metrics callback
        for i, info in enumerate(infos):
            info["achieved_goal"] = goal_obs["achieved_goal"][i]
            info["desired_goal"] = goal_obs["desired_goal"][i]
            info["ball_hit"] = bool(self.ball_hit[i])
            info["goal_scored"] = bool(goal_scored[i])
            info["phase"] = "post_hit" if self.ball_hit[i] else "pre_hit"

            # Convert terminal_observation to dict format for HER
            if "terminal_observation" in info:
                terminal_obs = info["terminal_observation"]
                if isinstance(terminal_obs, torch.Tensor):
                    terminal_obs = terminal_obs.cpu().numpy()
                # Compute terminal flags from terminal_observation (pre-reset),
                # NOT from obs (which is post-reset for done envs)
                info["terminal_ball_hit"] = bool(
                    self._check_ball_hit(terminal_obs.reshape(1, -1))[0]
                ) or bool(self.ball_hit[i])
                info["terminal_goal_scored"] = bool(
                    self._check_goal_scored(terminal_obs.reshape(1, -1))[0]
                )
                info["terminal_observation"] = self._extract_goals_single(terminal_obs)

        # Reset ball_hit for environments that are done
        self.ball_hit[dones.astype(bool)] = False

        return goal_obs, rewards, dones, infos

    # ------------------------------------------------------------------
    # HER compute_reward (compound 4-dim goals)
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Compute sparse reward for HER goal relabeling with compound goals.

        Both stages are checked independently from the 4-dim goals:
            Stage 1: distance(achieved[0:2], desired[0:2]) — player near target ball pos
            Stage 2: distance(achieved[2:4], desired[2:4]) — ball near target goal pos

        Reward structure (hierarchical):
            - Neither stage succeeded:  0
            - Stage 1 only:             ball_hit_reward
            - Stage 1 + Stage 2:        ball_hit_reward + goal_score_reward

        Stage 2 reward requires stage 1 success (must hit ball before scoring counts).

        Args:
            achieved_goal: shape (..., 4) = [player_pos_xy, ball_pos_xy]
            desired_goal:  shape (..., 4) = [target_ball_xy, target_goal_xy]
            info: Additional info (unused)

        Returns:
            Sparse reward array
        """
        # Stage 1: player reached target ball position
        stage1_dist = np.linalg.norm(
            achieved_goal[..., :2] - desired_goal[..., :2], axis=-1
        )
        stage1_success = stage1_dist < self.ball_hit_threshold

        # Stage 2: ball reached target goal position
        stage2_dist = np.linalg.norm(
            achieved_goal[..., 2:] - desired_goal[..., 2:], axis=-1
        )
        stage2_success = stage2_dist < self.goal_score_threshold

        # Hierarchical reward: stage 2 only counts if stage 1 also succeeded
        reward = np.where(
            stage1_success & stage2_success,
            self.ball_hit_reward + self.goal_score_reward,
            np.where(
                stage1_success,
                self.ball_hit_reward,
                0.0,
            ),
        )

        return reward.astype(np.float32)

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
