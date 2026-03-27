import math

import torch
from gymnasium import Wrapper


class RewardWeightWrapper(Wrapper):
    """Sets initial reward term weights from a config dict.

    Each reward term in ``cfg`` is a dict with a ``type`` field that controls
    how the weight behaves.  This wrapper only applies the **initial** weight;
    see :class:`CurriculumWrapper` for schedules that update weights during
    training.

    Supported types
    ---------------
    ``static``
        Constant weight, never changed after initialisation.
    ``linear``
        ``weight`` is ``[start, end]``; the initial weight is set to ``start``.
        Requires ``num_steps``.
    ``sigmoid``
        ``weight`` is ``[start, end]``; the initial weight is set to ``start``.
        Smoothly transitions from *start* to *end* using a sigmoid (logistic)
        curve.  Requires ``num_steps``.  Optional ``steepness`` controls how
        sharp the S-curve is (default 6.0; higher = sharper).
    ``exponential``
        ``weight`` is the initial value; it will be decayed by
        :class:`CurriculumWrapper`.  Requires ``decay_rate``.
    ``schedule``
        A list of phases, each with its own ``type``, ``steps: [start, end]``
        range, and parameters.  The initial weight is taken from the first
        phase.

    Example config::

        rewards:
          goal_scored:
            type: static
            weight: 5.0
          distance_ball_opponent_goal:
            type: linear
            weight: [0.0, 1.0]
            num_steps: 10000000
          ball_proximity:
            type: sigmoid
            weight: [0.0, 1.0]
            num_steps: 10000000
            steepness: 8.0
          collision_player_ball:
            type: exponential
            weight: 0.5
            decay_rate: 1.0e-7
          ball_speed:
            type: schedule
            phases:
              - type: static
                weight: 0.0
                steps: [0, 1000000]
              - type: linear
                weight: [0.0, 2.0]
                steps: [1000000, 5000000]
              - type: exponential
                weight: 2.0
                decay_rate: 1.0e-7
                steps: [5000000, -1]
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg

        for term, spec in cfg.items():
            term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)

            # Determine initial weight.
            if spec["type"] == "schedule":
                # Use the weight from the first phase.
                first = spec["phases"][0]
                raw_weight = first["weight"]
                if isinstance(raw_weight, list):
                    _weight = raw_weight[0]
                else:
                    _weight = raw_weight
            else:
                raw_weight = spec["weight"]
                if isinstance(raw_weight, list):
                    _weight = raw_weight[0]  # linear: start value
                else:
                    _weight = raw_weight

            self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = _weight


class CurriculumWrapper(RewardWeightWrapper):
    """Sets initial reward weights and adjusts them over training.

    Each reward term declares its own schedule via the ``type`` field:

    ``static``
        Weight is set once at initialisation and never changed.
    ``linear``
        ``weight: [start, end]`` — linearly interpolated from *start* to
        *end* over ``num_steps`` environment steps.
    ``sigmoid``
        ``weight: [start, end]`` — smoothly transitions from *start* to
        *end* using a sigmoid (S-curve) over ``num_steps`` environment steps.
        Optional ``steepness`` (default 6.0) controls the sharpness of the
        transition.
    ``exponential``
        ``weight`` decays as ``weight * exp(-decay_rate * step)``.
    ``schedule``
        A list of phases, each with its own ``type`` (``static``,
        ``linear``, ``sigmoid``, or ``exponential``), a ``steps: [start, end]`` range,
        and the corresponding parameters.  The ``steps`` range controls
        when each phase is active and is used as the reference for linear
        interpolation and exponential elapsed time.  The first phase whose
        range contains the current step is applied.  Use ``-1`` as
        ``end`` to make a phase extend indefinitely.

    Args:
        env: The wrapped environment.
        cfg: Reward config dict mapping term names to their spec dicts.

    Example config::

        rewards:
          # Constant reward — never changes.
          goal_scored:
            type: static
            weight: 5.0

          # Linearly ramp from 0 to 1 over num_steps (starting at step 0).
          distance_ball_opponent_goal:
            type: linear
            weight: [0.0, 1.0]
            num_steps: 10000000

          # Sigmoid ramp from 0 to 1 (S-curve, default steepness k=6).
          ball_proximity:
            type: sigmoid
            weight: [0.0, 1.0]
            num_steps: 10000000

          # Exponential decay (starting at step 0).
          # w(t) = weight * exp(-decay_rate * t)
          collision_player_ball:
            type: exponential
            weight: 0.5
            decay_rate: 1.0e-7

          # Multi-phase schedule: compose any sequence of types.
          # Each phase has a steps: [start, end] range.
          # Use -1 as end to extend a phase indefinitely.
          ball_speed:
            type: schedule
            phases:
              - type: static
                weight: 0.0
                steps: [0, 1000000]
              - type: sigmoid
                weight: [0.0, 2.0]
                steps: [1000000, 5000000]
                steepness: 10.0
              - type: static
                weight: 2.0
                steps: [5000000, -1]
    """

    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self._step = 0

    def _get_active_phase(self, spec):
        """Return the active phase config for the current step.

        For ``schedule`` types, finds the first phase whose step range
        contains ``self._step`` (or the last phase if past all ranges).
        An ``end_step`` of ``-1`` means the phase extends indefinitely.
        For all other types the spec itself is the phase.
        """
        if spec["type"] == "schedule":
            for phase in spec["phases"]:
                start_step, end_step = phase["steps"]
                if start_step <= self._step:
                    if end_step == -1 or self._step <= end_step:
                        return phase
            # Past all phases — fall back to the last one.
            return spec["phases"][-1]
        return spec

    def _apply_phase(self, term_idx, phase):
        """Apply a single phase to the reward term at the current step.

        For standalone types (no ``steps`` key), defaults to
        ``[0, num_steps]`` for linear and ``[0, 0]`` for exponential.
        """
        phase_type = phase["type"]

        if phase_type == "static":
            self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = phase["weight"]
            return

        start_step = phase.get("steps", [0])[0]

        if phase_type == "linear":
            end_step = phase["steps"][1] if "steps" in phase else phase["num_steps"]
            progress = (self._step - start_step) / (end_step - start_step)
            progress = min(max(progress, 0.0), 1.0)
            w = phase["weight"][0] + (phase["weight"][1] - phase["weight"][0]) * progress
            self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = w

        elif phase_type == "sigmoid":
            end_step = phase["steps"][1] if "steps" in phase else phase["num_steps"]
            progress = (self._step - start_step) / (end_step - start_step)
            progress = min(max(progress, 0.0), 1.0)
            k = phase.get("steepness", 6.0)
            x = torch.tensor(
                k * (2.0 * progress - 1.0),
                device=self.env.unwrapped.device,
                dtype=torch.float32,
            )
            raw = torch.sigmoid(x)
            sig_0 = torch.sigmoid(torch.tensor(-k, device=self.env.unwrapped.device, dtype=torch.float32))
            sig_1 = torch.sigmoid(torch.tensor(k, device=self.env.unwrapped.device, dtype=torch.float32))
            normalized = (raw - sig_0) / (sig_1 - sig_0)
            w = phase["weight"][0] + (phase["weight"][1] - phase["weight"][0]) * normalized
            self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = w

        elif phase_type == "exponential":
            elapsed = max(self._step - start_step, 0)
            w_0 = phase["weight"]
            self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = w_0 * torch.exp(
                -torch.tensor(
                    elapsed * phase["decay_rate"],
                    device=self.env.unwrapped.device,
                    dtype=torch.float32,
                )
            )

    def step(self, actions):
        self._step += self.env.unwrapped.num_envs

        for term, spec in self.cfg.items():
            term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
            phase = self._get_active_phase(spec)
            self._apply_phase(term_idx, phase)

        return self.env.step(actions)


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


class InitializationWrapper(Wrapper):
    """Manages reset-time initialization helpers via a configurable schedule.

    Sets ``env.unwrapped._init_velocity_speed`` before every ``reset()`` and
    ``step()`` call.  The event function ``reset_player_velocity_toward_ball``
    registered in ``EventCfgDreamer`` reads this attribute and writes the
    corresponding velocity into the physics simulation *inside* the event
    manager — ensuring that observations (and ``ActuatorModelWrapper``'s state
    history) see the correct non-zero velocity from the very first step.

    The ``init_velocity`` config key accepts the same schedule types as
    :class:`CurriculumWrapper` (``static``, ``linear``, ``sigmoid``,
    ``exponential``, ``schedule``), using ``speed`` instead of ``weight``.

    Example config::

        initialization:
          init_velocity:
            type: schedule
            phases:
              - type: static
                speed: 1.0
                steps: [0, 1_000_000]
              - type: sigmoid
                speed: [1.0, 0.0]
                steps: [1_000_000, 3_000_000]
                steepness: 6.0
              - type: static
                speed: 0.0
                steps: [3_000_000, -1]
    """

    def __init__(self, env, cfg: dict):
        super().__init__(env)
        self.cfg = cfg
        self._step = 0

    def _compute_speed(self, spec: dict, step: int) -> float:
        """Return the current speed scalar from a schedule spec."""
        if spec["type"] == "schedule":
            active = spec["phases"][-1]
            for phase in spec["phases"]:
                start, end = phase["steps"]
                if start <= step and (end == -1 or step <= end):
                    active = phase
                    break
        else:
            active = spec

        phase_type = active["type"]
        raw = active["speed"]

        if phase_type == "static":
            return float(raw)

        start_step = active["steps"][0] if "steps" in active else 0

        if phase_type == "linear":
            end_step = active["steps"][1] if "steps" in active else active["num_steps"]
            progress = min(max((step - start_step) / (end_step - start_step), 0.0), 1.0)
            return float(raw[0] + (raw[1] - raw[0]) * progress)

        if phase_type == "sigmoid":
            end_step = active["steps"][1] if "steps" in active else active["num_steps"]
            progress = min(max((step - start_step) / (end_step - start_step), 0.0), 1.0)
            k = active.get("steepness", 6.0)

            def _sig(x):
                return 1.0 / (1.0 + math.exp(-x))

            normalized = (_sig(k * (2.0 * progress - 1.0)) - _sig(-k)) / (_sig(k) - _sig(-k))
            return float(raw[0] + (raw[1] - raw[0]) * normalized)

        if phase_type == "exponential":
            elapsed = max(step - start_step, 0)
            return float(raw) * math.exp(-elapsed * active["decay_rate"])

        return 0.0

    def _set_env_speed(self):
        init_vel_spec = self.cfg.get("init_velocity")
        speed = self._compute_speed(init_vel_spec, self._step) if init_vel_spec else 0.0
        self.env.unwrapped._init_velocity_speed = speed

    def reset(self, *args, **kwargs):
        # Set speed BEFORE env.reset() so the event manager reads the correct value.
        self._set_env_speed()
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        self._step += self.env.unwrapped.num_envs
        # Update speed BEFORE env.step() so partial resets triggered inside
        # env.step() also see the current scheduled speed.
        self._set_env_speed()
        return self.env.step(actions)


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
