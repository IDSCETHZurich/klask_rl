"""Parametric actuator surrogate: dead time + first-order lag on the velocity command.

Stand-in for a learned actuator model when no data exists yet (KLASK v2 hardware, 2026-10):
for each control step of length dt, the velocity sent to the simulator is

    target_t = (1 - f) * u_t + f * u_{t-1}       f = dead_time / dt  (fractional delay, 0 <= f <= 1)
    v_t      = v_{t-1} + alpha * (target_t - v_{t-1}),   alpha = 1 - exp(-dt / tau)

applied per action dimension (both pegs, m/s). Dead times longer than one control step are not
supported (use the learned model for those). The acceleration cap is applied before this wrapper
by VelocityScaleWrapper (max_acceleration), the speed cap by its max_velocity.
State is reset to zero for terminated/truncated envs, like ActuatorModelWrapper's buffers.
"""

import math

import torch
from gymnasium import Wrapper


def lag_coefficients(dt, dead_time, tau):
    if not 0.0 <= dead_time <= dt:
        raise ValueError(f"dead_time must be in [0, dt={dt}], got {dead_time}")
    f = dead_time / dt
    alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
    return f, alpha


class ActuatorLagStep:
    """The filter itself (tensor in, tensor out), separate from the env for testing."""

    def __init__(self, num_envs, action_dim, dt, dead_time, tau, device="cpu"):
        self.f, self.alpha = lag_coefficients(dt, dead_time, tau)
        self.prev_u = torch.zeros(num_envs, action_dim, device=device)
        self.v = torch.zeros(num_envs, action_dim, device=device)

    def __call__(self, u):
        target = (1.0 - self.f) * u + self.f * self.prev_u
        self.v = self.v + self.alpha * (target - self.v)
        self.prev_u = u.clone()
        return self.v.clone()

    def reset(self, done):
        self.prev_u[done] = 0.0
        self.v[done] = 0.0


class ActuatorLagWrapper(Wrapper):
    """Applies the lag to the action columns ``columns`` (a slice; default: every column).

    ``columns=slice(2, 4)`` lags only seat 1's peg (tournament seats on different hardware, 2026-10-14); the
    filter is elementwise, so each lagged column equals what the all-column wrapper gives for it.
    """

    def __init__(self, env, dead_time_s, tau_s, device=None, columns=None):
        super().__init__(env)
        unwrapped = env.unwrapped
        dt = float(unwrapped.step_dt)
        num_envs = unwrapped.num_envs
        action_dim = unwrapped.single_action_space.shape[0] if hasattr(unwrapped, "single_action_space") else 4
        self.columns = slice(0, action_dim) if columns is None else columns
        width = len(range(*self.columns.indices(action_dim)))
        self.lag = ActuatorLagStep(num_envs, width, dt, float(dead_time_s), float(tau_s), device or unwrapped.device)
        print(
            f"[ActuatorLagWrapper] dt={dt:.3f}s dead_time={dead_time_s}s tau={tau_s}s -> f={self.lag.f:.2f} "
            f"alpha={self.lag.alpha:.3f} columns={self.columns}"
        )

    def reset(self, *args, **kwargs):
        self.lag.reset(torch.ones(self.lag.v.shape[0], dtype=torch.bool, device=self.lag.v.device))
        return self.env.reset(*args, **kwargs)

    def step(self, actions, *args, **kwargs):
        actions = actions.clone()
        actions[:, self.columns] = self.lag(actions[:, self.columns])
        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        self.lag.reset(terminated | truncated)
        return obs, rew, terminated, truncated, info
