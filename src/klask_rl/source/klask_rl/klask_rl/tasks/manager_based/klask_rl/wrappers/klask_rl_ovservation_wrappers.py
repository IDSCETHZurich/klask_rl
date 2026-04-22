import gymnasium as gym
import torch
from gymnasium import Wrapper


class OpponentActionWrapper(Wrapper):
    """Negates opponent actions (indices 2:) to convert from player frame back to world frame.

    Going down: opponent actions in player frame are rotated 180° to world frame.
    Going up: observations are passed through unchanged (rotation is handled by ObservationsCfg).

    Placement: this wrapper sits close to the env, but may have world-frame
    action wrappers (e.g. ``KlaskRlCollisionAvoidanceWrapper``) below it. Any
    wrapper placed below must NOT re-rotate opponent dims — only magnitude
    modifications (clipping, scaling) are safe there.
    """

    def step(self, actions, *args, **kwargs):
        actions = actions.clone()
        actions[:, 2:] = -actions[:, 2:]
        return self.env.step(actions, *args, **kwargs)


class ObservationNoiseWrapper(Wrapper):
    """Adds Gaussian noise to base observations and recomputes derived observations.

    When extended observations (angles, distances) are present (obs dim >= 20),
    they are recomputed from the noisy base observations to ensure consistency.
    This means noise in position measurements properly propagates to angles and
    distances, rather than those being computed from clean simulation state.

    Base observation layout (first 12 dims):
        [own_pos_x, own_pos_y, own_vel_x, own_vel_y,
         other_pos_x, other_pos_y, other_vel_x, other_vel_y,
         ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y]

    Extended observations (dims 12-19, recomputed when present):
        [angle_own_ball_goal, angle_other_ball_goal,
         angle_own_ball_other, angle_other_ball_own,
         distance_ball_own_goal, distance_ball_other_goal,
         distance_ball_own, distance_ball_other]

    The board is symmetric: opponent observations are 180°-rotated (positions negated).
    Since angles and distances are invariant under simultaneous negation of all points,
    the same goal positions work for both player and opponent observation keys.
    """

    BASE_DIM = 12
    EXTENDED_DIM = 8
    OWN_POS = slice(0, 2)
    OTHER_POS = slice(4, 6)
    BALL_POS = slice(8, 10)

    def __init__(self, env, noise_std, own_goal, other_goal):
        super().__init__(env)
        self.noise_std = noise_std
        # Store goal xy tuples; tensors created lazily on correct device
        self._own_goal_xy = (own_goal[0], own_goal[1])
        self._other_goal_xy = (other_goal[0], other_goal[1])
        self._own_goal = None
        self._other_goal = None

        # Detect whether extended observations are present
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            sample_dim = next(iter(obs_space.spaces.values())).shape[-1]
        else:
            sample_dim = obs_space.shape[-1]
        self.has_extended = sample_dim >= self.BASE_DIM + self.EXTENDED_DIM

    def _ensure_goals(self, device):
        """Lazily create goal tensors on the correct device."""
        if self._own_goal is None or self._own_goal.device != device:
            self._own_goal = torch.tensor(self._own_goal_xy, dtype=torch.float32, device=device)
            self._other_goal = torch.tensor(self._other_goal_xy, dtype=torch.float32, device=device)

    @staticmethod
    def _angle(A, B, C):
        """Angle at point A between vectors A->B and A->C. Returns shape (N, 1)."""
        vec_ab = B - A
        vec_ac = C - A
        dot = (vec_ab * vec_ac).sum(dim=1)
        cos_theta = dot / (vec_ab.norm(dim=1) * vec_ac.norm(dim=1) + 1e-8)
        return torch.acos(cos_theta.clamp(-1.0, 1.0)).unsqueeze(-1)

    @staticmethod
    def _distance(A, B):
        """Euclidean distance between A and B. Returns shape (N, 1)."""
        return (A - B).norm(dim=1).unsqueeze(-1)

    def _recompute_extended(self, obs):
        """Recompute extended observations (dims 12:20) from (noisy) base positions."""
        own = obs[:, self.OWN_POS]
        other = obs[:, self.OTHER_POS]
        ball = obs[:, self.BALL_POS]
        og = self._own_goal
        tg = self._other_goal

        extended = torch.cat([
            self._angle(own, ball, tg),      # angle_own_ball_goal
            self._angle(other, ball, og),     # angle_other_ball_goal
            self._angle(own, ball, other),    # angle_own_ball_other
            self._angle(other, ball, own),    # angle_other_ball_own
            self._distance(ball, og),         # distance_ball_own_goal
            self._distance(ball, tg),         # distance_ball_other_goal
            self._distance(ball, own),        # distance_ball_own
            self._distance(ball, other),      # distance_ball_other
        ], dim=-1)

        # Preserve any dims beyond 20 (action history, goal position, etc.)
        parts = [obs[:, :self.BASE_DIM], extended]
        tail_start = self.BASE_DIM + self.EXTENDED_DIM
        if obs.shape[-1] > tail_start:
            parts.append(obs[:, tail_start:])
        return torch.cat(parts, dim=-1)

    def _apply_noise(self, obs):
        """Add noise to base dims and recompute extended if present."""
        obs = obs.clone()
        noise = self.noise_std * torch.randn(obs.shape[0], self.BASE_DIM, device=obs.device)
        obs[:, :self.BASE_DIM] += noise
        if self.has_extended:
            obs = self._recompute_extended(obs)
        return obs

    def _process_obs(self, observation):
        """Apply noise to observation (dict or flat tensor)."""
        device = next(iter(observation.values())).device if isinstance(observation, dict) else observation.device
        self._ensure_goals(device)
        if isinstance(observation, dict):
            return {k: self._apply_noise(v) for k, v in observation.items()}
        return self._apply_noise(observation)

    def step(self, actions, *args, **kwargs):
        obs, rew, terminated, truncated, extras = self.env.step(actions, *args, **kwargs)
        return self._process_obs(obs), rew, terminated, truncated, extras

    def reset(self, *args, **kwargs):
        obs, extras = self.env.reset(*args, **kwargs)
        return self._process_obs(obs), extras


class OpponentObservationWrapper(Wrapper):
    """Stores opponent observations for access by the self-play env wrapper.

    The opponent observations are already rotated to player frame by the ObservationsCfg,
    so this wrapper only stores/passes them without additional transformation.
    """

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

    def reset(self, *args, **kwargs):
        obs_dict, extras = self.env.reset(*args, **kwargs)
        if self.mode == "train":
            self.opponent_obs = obs_dict["opponent"]
        return obs_dict, extras

    def step(self, actions, *args, **kwargs):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions, *args, **kwargs)
        if self.mode == "train":
            self.opponent_obs = obs_dict["opponent"]
        return obs_dict, rew, terminated, truncated, extras
