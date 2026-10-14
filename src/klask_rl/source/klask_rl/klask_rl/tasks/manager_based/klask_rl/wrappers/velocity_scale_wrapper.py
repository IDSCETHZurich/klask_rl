import math
from collections import deque

import torch
from gymnasium import Wrapper


class VelocityScaleWrapper(Wrapper):
    """Scale actions from [-1, 1] to [-max_velocity, max_velocity] m/s, with optional rate limiting.

    Pipeline per step:
      1. clamp action ∈ [-1, 1]                       (defensive)
      2. v_target = action x max_velocity              [m/s]
      3. if rate limit active: clamp(v_target - v_prev, ±max_acceleration·step_dt) + v_prev
      4. forward to inner env

    The per-step velocity-delta cap is derived from the env's actual control period,
    `step_dt = decimation x sim.dt`, so `max_acceleration` keeps consistent physical
    meaning when timing parameters change.

    Also exposes per-step action-smoothness signals on ``env.unwrapped`` for use by
    reward terms (action_rate_cap_saturation, action_smooth_hinge, action_delta_l2)
    and logs three player-side smoothness metrics in events/sec via :meth:`set_logger`.

    Per-column limits (tournament seats on different hardware, 2026-10-14): ``max_velocity`` may be a
    sequence with one value per action column, and ``max_acceleration`` then a scalar, None or a sequence
    (None entries = no rate cap for that column). Each column is then processed exactly as a scalar
    wrapper with that column's limits would process it; ``_action_max_dv_norm`` becomes a per-column
    tensor (inf where uncapped). With scalar arguments the wrapper is unchanged.
    """

    METRIC_WINDOW = 100  # FIFO window for player-side metric averaging (steps)

    def __init__(
        self,
        env,
        max_velocity: float,
        max_acceleration: float | None = None,
        num_envs: int | None = None,
        action_dim: int | None = None,
        device=None,
    ):
        super().__init__(env)
        self._capped = None  # per-column mode only: columns with a rate cap (None = all / scalar mode)
        if isinstance(max_velocity, (list, tuple)):
            self._init_per_column(env, max_velocity, max_acceleration, num_envs, action_dim, device)
            return
        self.max_velocity = float(max_velocity)
        self._scale = self.max_velocity
        self._rate_limit_active = max_acceleration is not None

        if self._rate_limit_active:
            if num_envs is None or action_dim is None or device is None:
                raise ValueError(
                    "VelocityScaleWrapper: num_envs, action_dim, and device are required when max_acceleration is set."
                )
            self.max_acceleration = float(max_acceleration)
            self._step_dt = float(self.env.unwrapped.step_dt)
            self._max_dv = self.max_acceleration * self._step_dt
            self._sat_threshold = 0.99 * self._max_dv
            self._max_dv_norm = self._max_dv / self.max_velocity
            self._prev_v = torch.zeros(int(num_envs), int(action_dim), device=device)
            self._prev_a = torch.zeros(int(num_envs), int(action_dim), device=device)
            self._fresh = torch.ones(int(num_envs), dtype=torch.bool, device=device)
            print(
                f"VelocityScaleWrapper: max_velocity={self.max_velocity:.3f} m/s, "
                f"max_acceleration={self.max_acceleration:.3f} m/s^2, "
                f"step_dt={self._step_dt:.4f} s, "
                f"max_dv_per_step={self._max_dv:.4f} m/s, "
                f"action_dim={int(action_dim)}, num_envs={int(num_envs)}"
            )
        else:
            self._step_dt = float(self.env.unwrapped.step_dt)
            self._max_dv_norm = None
            if num_envs is not None and action_dim is not None and device is not None:
                self._prev_a = torch.zeros(int(num_envs), int(action_dim), device=device)
                self._fresh = torch.ones(int(num_envs), dtype=torch.bool, device=device)
            else:
                self._prev_a = None
                self._fresh = None
            print(
                f"VelocityScaleWrapper: max_velocity={self.max_velocity:.3f} m/s, "
                f"rate limiting disabled (max_acceleration=None)"
            )

        self._init_metrics()

    def _init_metrics(self):
        self._logger = None
        self._q_l1 = deque(maxlen=self.METRIC_WINDOW)
        self._q_flip = deque(maxlen=self.METRIC_WINDOW)
        self._q_sat = deque(maxlen=self.METRIC_WINDOW)

    def _init_per_column(self, env, max_velocity, max_acceleration, num_envs, action_dim, device):
        """One (max_velocity, max_acceleration) pair per action column; same arithmetic per column."""
        if num_envs is None or action_dim is None or device is None:
            raise ValueError("VelocityScaleWrapper: per-column limits need num_envs, action_dim and device.")
        columns = int(action_dim)
        if len(max_velocity) != columns:
            raise ValueError(f"VelocityScaleWrapper: {len(max_velocity)} max_velocity values for {columns} columns")
        if not isinstance(max_acceleration, (list, tuple)):
            max_acceleration = [max_acceleration] * columns
        if len(max_acceleration) != columns:
            raise ValueError(f"VelocityScaleWrapper: {len(max_acceleration)} max_acceleration values for {columns}")
        self.max_velocity = tuple(float(v) for v in max_velocity)
        self.max_acceleration = tuple(None if a is None else float(a) for a in max_acceleration)
        self._step_dt = float(self.env.unwrapped.step_dt)
        # Python-float arithmetic per column, then one cast to float32: the same values a scalar wrapper uses.
        max_dv = [math.inf if a is None else a * self._step_dt for a in self.max_acceleration]
        self._scale = torch.tensor(self.max_velocity, dtype=torch.float32, device=device)
        self._max_dv = torch.tensor(max_dv, dtype=torch.float32, device=device)
        self._sat_threshold = torch.tensor([0.99 * dv for dv in max_dv], dtype=torch.float32, device=device)
        self._max_dv_norm = torch.tensor(
            [dv / v for dv, v in zip(max_dv, self.max_velocity)], dtype=torch.float32, device=device
        )
        capped = [a is not None for a in self.max_acceleration]
        self._rate_limit_active = any(capped)
        if self._rate_limit_active and not all(capped):
            self._capped = torch.tensor(capped, dtype=torch.bool, device=device)
        if self._rate_limit_active:
            self._prev_v = torch.zeros(int(num_envs), columns, device=device)
        else:
            self._max_dv_norm = None  # as the scalar wrapper without a rate cap
        self._prev_a = torch.zeros(int(num_envs), columns, device=device)
        self._fresh = torch.ones(int(num_envs), dtype=torch.bool, device=device)
        print(
            f"VelocityScaleWrapper (per column): max_velocity={self.max_velocity} m/s, "
            f"max_acceleration={self.max_acceleration} m/s^2, step_dt={self._step_dt:.4f} s, num_envs={int(num_envs)}"
        )
        self._init_metrics()

    def set_logger(self, logger):
        """Attach a :class:`tools.Logger` for autonomous metric logging."""
        self._logger = logger

    def set_limits(self, max_velocity: float, max_acceleration: float | None = None):
        """Change the command scale and the rate cap in place (hardware_schedule_wrapper.py). The rate limiter's
        previous command (m/s) is kept, so a change between two steps acts like a change of the physical limits.
        Scalar limits only (not a per-column wrapper); every value derived from the limits is updated."""
        if isinstance(self.max_velocity, tuple):
            raise ValueError("VelocityScaleWrapper.set_limits: not supported with per-column limits")
        self.max_velocity = float(max_velocity)
        self._scale = self.max_velocity
        if self._rate_limit_active:
            if max_acceleration is None:
                raise ValueError("VelocityScaleWrapper.set_limits: rate limiting is active, max_acceleration is required")
            self.max_acceleration = float(max_acceleration)
            self._max_dv = self.max_acceleration * self._step_dt
            self._sat_threshold = 0.99 * self._max_dv
            self._max_dv_norm = self._max_dv / self.max_velocity
        elif max_acceleration is not None:
            raise ValueError("VelocityScaleWrapper.set_limits: built without max_acceleration (no rate limiting)")

    def step(self, actions, *args, **kwargs):
        actions = torch.clamp(actions, -1.0, 1.0)
        v_target = actions * self._scale

        if self._rate_limit_active:
            dv = torch.clamp(v_target - self._prev_v, -self._max_dv, self._max_dv)
            v_cmd = self._prev_v + dv
            saturation = (v_target - self._prev_v).abs() >= self._sat_threshold
            if self._capped is not None:  # per-column mode, some columns uncapped: their command is v_target
                v_cmd = torch.where(self._capped, v_cmd, v_target)
                saturation = saturation & self._capped
        else:
            v_cmd = v_target
            saturation = torch.zeros_like(v_target, dtype=torch.bool)

        if self._prev_a is not None:
            da = actions - self._prev_a
        else:
            da = torch.zeros_like(actions)

        # Mask the first step of each episode: _prev_a / _prev_v are zero
        # placeholders, not real previous actions/velocities, so the delta
        # and saturation flag would be spurious. Zeroing them out gives
        # consistent semantics across the metric and all three penalties.
        if self._fresh is not None and self._fresh.any():
            da[self._fresh] = 0.0
            saturation[self._fresh] = False

        unwrapped = self.env.unwrapped
        unwrapped._action_rate_cap_saturation = saturation
        unwrapped._action_delta = da
        unwrapped._action_max_dv_norm = self._max_dv_norm

        if self._prev_a is not None:
            self._q_l1.append(da[:, 0:2].abs().mean().item())
            self._q_flip.append(((actions[:, 0:2] * self._prev_a[:, 0:2]) < 0).float().mean().item())
            self._q_sat.append(saturation[:, 0:2].float().mean().item())

        obs, rew, terminated, truncated, info = self.env.step(v_cmd, *args, **kwargs)

        if self._rate_limit_active:
            self._prev_v.copy_(v_cmd.detach())
        if self._prev_a is not None:
            self._prev_a.copy_(actions.detach())
        if self._fresh is not None:
            self._fresh.fill_(False)

        done = terminated | truncated
        if done.any():
            if self._rate_limit_active:
                self._prev_v[done] = 0.0
            if self._prev_a is not None:
                self._prev_a[done] = 0.0
            if self._fresh is not None:
                self._fresh[done] = True

        if self._logger is not None and len(self._q_l1) > 0:
            inv_dt = 1.0 / self._step_dt
            self._logger.scalar("actions/abs_delta_l1_per_sec", sum(self._q_l1) / len(self._q_l1) * inv_dt)
            self._logger.scalar("actions/sign_flip_per_sec", sum(self._q_flip) / len(self._q_flip) * inv_dt)
            self._logger.scalar("actions/rate_cap_saturation_per_sec", sum(self._q_sat) / len(self._q_sat) * inv_dt)

        return obs, rew, terminated, truncated, info

    def reset(self, *args, **kwargs):
        if self._rate_limit_active:
            self._prev_v.zero_()
        if self._prev_a is not None:
            self._prev_a.zero_()
        if self._fresh is not None:
            self._fresh.fill_(True)
        return self.env.reset(*args, **kwargs)
