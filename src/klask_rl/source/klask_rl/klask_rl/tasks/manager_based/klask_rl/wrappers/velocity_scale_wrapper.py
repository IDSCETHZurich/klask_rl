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
    """

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
            self._prev_v = torch.zeros(int(num_envs), int(action_dim), device=device)
            print(
                f"VelocityScaleWrapper: max_velocity={self.max_velocity:.3f} m/s, "
                f"max_acceleration={self.max_acceleration:.3f} m/s^2, "
                f"step_dt={self._step_dt:.4f} s, "
                f"max_dv_per_step={self._max_dv:.4f} m/s, "
                f"action_dim={int(action_dim)}, num_envs={int(num_envs)}"
            )
        else:
            print(
                f"VelocityScaleWrapper: max_velocity={self.max_velocity:.3f} m/s, "
                f"rate limiting disabled (max_acceleration=None)"
            )

    def step(self, actions, *args, **kwargs):
        actions = torch.clamp(actions, -1.0, 1.0)
        v_target = actions * self.max_velocity

        if self._rate_limit_active:
            dv = torch.clamp(v_target - self._prev_v, -self._max_dv, self._max_dv)
            v_cmd = self._prev_v + dv
        else:
            v_cmd = v_target

        obs, rew, terminated, truncated, info = self.env.step(v_cmd, *args, **kwargs)

        if self._rate_limit_active:
            self._prev_v.copy_(v_cmd.detach())
            done = terminated | truncated
            if done.any():
                self._prev_v[done] = 0.0

        return obs, rew, terminated, truncated, info

    def reset(self, *args, **kwargs):
        if self._rate_limit_active:
            self._prev_v.zero_()
        return self.env.reset(*args, **kwargs)
