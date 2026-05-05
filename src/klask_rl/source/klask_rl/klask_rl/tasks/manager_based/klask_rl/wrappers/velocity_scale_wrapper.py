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
        self.max_velocity = float(max_velocity)
        self._rate_limit_active = max_acceleration is not None

        if self._rate_limit_active:
            if num_envs is None or action_dim is None or device is None:
                raise ValueError(
                    "VelocityScaleWrapper: num_envs, action_dim, and device are required when max_acceleration is set."
                )
            self.max_acceleration = float(max_acceleration)
            self._step_dt = float(self.env.unwrapped.step_dt)
            self._max_dv = self.max_acceleration * self._step_dt
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

        self._logger = None
        self._q_l1 = deque(maxlen=self.METRIC_WINDOW)
        self._q_flip = deque(maxlen=self.METRIC_WINDOW)
        self._q_sat = deque(maxlen=self.METRIC_WINDOW)

    def set_logger(self, logger):
        """Attach a :class:`tools.Logger` for autonomous metric logging."""
        self._logger = logger

    def step(self, actions, *args, **kwargs):
        actions = torch.clamp(actions, -1.0, 1.0)
        v_target = actions * self.max_velocity

        if self._rate_limit_active:
            dv = torch.clamp(v_target - self._prev_v, -self._max_dv, self._max_dv)
            v_cmd = self._prev_v + dv
            saturation = (v_target - self._prev_v).abs() >= 0.99 * self._max_dv
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
