import math

import torch
from gymnasium import Wrapper
from isaaclab.managers import SceneEntityCfg
from klask_rl.assets.robots.klask_params import KLASK_PARAMS


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


class OpponentRewardWrapper(Wrapper):
    """Computes opponent rewards using the same functions/weights as player rewards.

    Calls opponent reward functions directly (not through the reward manager,
    since it skips 0-weight terms — see ``RewardManager.compute``).  Reads
    current player weights from the reward manager so the opponent reward
    automatically follows the same curriculum schedule.

    Adds ``obs["opponent_reward"]`` (shape ``(B,)``) to the step output.

    Parameters
    ----------
    env : gymnasium.Env
        The wrapped environment (must be below ``CurriculumWrapper``).
    term_mapping : dict[str, tuple[callable, dict]]
        Maps player reward term name to ``(opponent_func, opponent_params)``.
        Built from ``_opponent_reward_terms`` in ``klask_rl_rewards_cfg.py``.
    """

    def __init__(self, env, term_mapping: dict):
        super().__init__(env)
        self._term_mapping = term_mapping
        rm = self.env.unwrapped.reward_manager
        self._player_term_indices = {}
        for player_name in term_mapping:
            if player_name in rm.active_terms:
                self._player_term_indices[player_name] = rm.active_terms.index(player_name)

        # Resolve SceneEntityCfg objects (body_names → body_ids) that would
        # normally be resolved by the reward manager during _prepare_terms().
        scene = self.env.unwrapped.scene
        for _func, params in term_mapping.values():
            for v in params.values():
                if isinstance(v, SceneEntityCfg):
                    v.resolve(scene)

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)

        rm = self.env.unwrapped.reward_manager
        dt = self.env.unwrapped.step_dt
        opp_reward = torch.zeros(self.env.unwrapped.num_envs, device=self.env.unwrapped.device)

        for player_name, (opp_func, opp_params) in self._term_mapping.items():
            if player_name not in self._player_term_indices:
                continue
            player_idx = self._player_term_indices[player_name]
            weight = rm._term_cfgs[player_idx].weight
            if isinstance(weight, torch.Tensor):
                if weight.item() == 0.0:
                    continue
            elif weight == 0.0:
                continue
            raw = opp_func(rm._env, **opp_params)
            opp_reward += raw * weight * dt

        obs["opponent_reward"] = opp_reward
        return obs, rew, terminated, truncated, info


class KlaskRlCollisionAvoidanceWrapper(Wrapper):

    DEACCELERATION_DISTANCE = KLASK_PARAMS["collision_avoidance_decel_distance"]
    PEG_RADIUS = KLASK_PARAMS["peg_radius"]
    MIN_CLEARANCE = KLASK_PARAMS["peg_radius"] * KLASK_PARAMS["collision_avoidance_min_clearance_factor"]
    X_EDGE = KLASK_PARAMS["joint_x_pos_limit"]
    Y_EDGE_1 = KLASK_PARAMS["joint_y1_pos_limit"]  # [0]=outer, [1]=inner
    Y_EDGE_2 = KLASK_PARAMS["joint_y2_pos_limit"]  # [0]=inner, [1]=outer

    def __init__(self, env, max_vel=0.2, peg1_idx=slice(0, 2), peg2_idx=slice(4, 6)):
        super().__init__(env)
        self.MAX_VEL = max_vel
        self._peg1_idx = peg1_idx
        self._peg2_idx = peg2_idx

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.state_1 = obs["policy"].clone()[:, self._peg1_idx]  # Peg_1 world xy
        self.state_2 = obs["policy"].clone()[:, self._peg2_idx]  # Peg_2 world xy (unrotated)
        return obs, info

    def _apply_decel_axis(self, vel, pos, edge_min, edge_max):
        """Clamp vel in-place near boundaries using linear deceleration + hard stop."""
        safe_zone = self.DEACCELERATION_DISTANCE + self.PEG_RADIUS

        dist_min = pos - edge_min
        near_min = dist_min <= safe_zone
        if near_min.any():
            hard_mask = near_min & (dist_min <= self.MIN_CLEARANCE)
            soft_mask = near_min & ~hard_mask
            vel[hard_mask] = torch.maximum(vel[hard_mask], torch.zeros_like(vel[hard_mask]))
            vel[soft_mask] = torch.maximum(
                vel[soft_mask], -self.interpolate_vel(dist_min[soft_mask] - self.MIN_CLEARANCE)
            )

        dist_max = edge_max - pos
        near_max = dist_max <= safe_zone
        if near_max.any():
            hard_mask = near_max & (dist_max <= self.MIN_CLEARANCE)
            soft_mask = near_max & ~hard_mask
            vel[hard_mask] = torch.minimum(vel[hard_mask], torch.zeros_like(vel[hard_mask]))
            vel[soft_mask] = torch.minimum(
                vel[soft_mask], self.interpolate_vel(dist_max[soft_mask] - self.MIN_CLEARANCE)
            )

    def step(self, actions, *args, **kwargs):
        self._apply_decel_axis(actions[:, 0], self.state_1[:, 0], self.X_EDGE[0], self.X_EDGE[1])
        self._apply_decel_axis(actions[:, 1], self.state_1[:, 1], self.Y_EDGE_1[0], self.Y_EDGE_1[1])
        self._apply_decel_axis(actions[:, 2], self.state_2[:, 0], self.X_EDGE[0], self.X_EDGE[1])
        self._apply_decel_axis(actions[:, 3], self.state_2[:, 1], self.Y_EDGE_2[0], self.Y_EDGE_2[1])

        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        self.state_1 = obs["policy"].clone()[:, self._peg1_idx]
        self.state_2 = obs["policy"].clone()[:, self._peg2_idx]
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
