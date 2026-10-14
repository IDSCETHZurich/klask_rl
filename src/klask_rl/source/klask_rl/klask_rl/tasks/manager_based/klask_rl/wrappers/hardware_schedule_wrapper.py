"""Hardware-speed curriculum (branch r2-ft-v2hw, 2026-10-14): the peg command scale and the command rate cap follow a
step schedule, e.g. a fine-tune that moves a v1-trained agent onto the faster v2 table step by step.

Config (``env.hardware_schedule``; absent or ``enable: false`` = constant hardware, exactly as before):

    hardware_schedule:
      enable: true
      steps: [10_404_928, 11_404_928]   # absolute env steps (the trainer's step counter, as in checkpoint names)
      max_velocity: [0.4, 1.4]          # [m/s] per axis: start value (until steps[0]) -> end value (from steps[1])
      max_acceleration: [10.0, 35.0]    # [m/s^2]

Linear in between, held at the end values afterwards and at the start values before ``steps[0]``. The end values must
equal ``env.max_velocity`` / ``env.max_acceleration`` (checked by :meth:`HardwareSchedule.from_config`), so the saved
config's top-level hardware block is the hardware the run ends on (what evaluations and tournaments read).

What follows the schedule (set before every env step and reset):

  - ``VelocityScaleWrapper.set_limits(v, a)``: the [-1, 1] -> m/s command scale ``v`` and the per-step rate cap
    ``a * step_dt``, for both pegs (self-play: both seats always share the hardware);
  - ``KlaskRlCollisionAvoidanceWrapper.set_max_vel(v)``: the boundary deceleration ramp ``v / decel_distance * d``.

Not scheduled: the actuator stage (learned v1 ``actuator_model`` or the parametric ``actuator_lag``) is whatever the
env block enables, from the first step.

The current step comes from ``step_fn`` when bound (train_dreamer binds the trainer's step, so a resumed run continues
the schedule at its checkpoint step), otherwise from the wrapper's own counter (``num_envs`` per step, the convention
of CurriculumWrapper and of the trainer's step; :meth:`set_step` positions it).
"""

from __future__ import annotations

from gymnasium import Wrapper


def _pair(value, name):
    try:
        a, b = (float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"hardware_schedule.{name} must be a pair [start, end], got {value!r}") from exc
    if a <= 0 or b <= 0:
        raise ValueError(f"hardware_schedule.{name} values must be positive, got {value!r}")
    return a, b


class HardwareSchedule:
    """Step -> (max_velocity, max_acceleration): linear between ``steps[0]`` and ``steps[1]``, constant outside."""

    def __init__(self, steps, max_velocity, max_acceleration):
        try:
            start, end = (int(float(s)) for s in steps)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"hardware_schedule.steps must be [start_step, end_step], got {steps!r}") from exc
        if end < start:
            raise ValueError(f"hardware_schedule.steps must have end >= start, got {steps!r}")
        self.start_step, self.end_step = start, end
        self.max_velocity = _pair(max_velocity, "max_velocity")
        self.max_acceleration = _pair(max_acceleration, "max_acceleration")

    @classmethod
    def from_config(cls, cfg, final_max_velocity, final_max_acceleration):
        """Build from ``env.hardware_schedule`` (dict-like); ``None`` when absent or disabled."""
        if cfg is None:
            return None
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
        if not bool(get("enable", False)):
            return None
        if final_max_acceleration is None:
            raise ValueError("hardware_schedule needs env.max_acceleration (the rate cap it schedules)")
        schedule = cls(get("steps"), get("max_velocity"), get("max_acceleration"))
        for name, (_, end), final in (
            ("max_velocity", schedule.max_velocity, final_max_velocity),
            ("max_acceleration", schedule.max_acceleration, final_max_acceleration),
        ):
            if abs(end - float(final)) > 1e-9:
                raise ValueError(
                    f"hardware_schedule.{name} ends at {end}, but env.{name} is {final}: the top-level value must be "
                    "the hardware the schedule ends on (evaluations and tournaments read it)"
                )
        return schedule

    def progress(self, step) -> float:
        if self.end_step == self.start_step:  # a single switch at start_step
            return 0.0 if step < self.start_step else 1.0
        return min(max((step - self.start_step) / (self.end_step - self.start_step), 0.0), 1.0)

    def values(self, step) -> tuple[float, float]:
        p = self.progress(step)
        (v0, v1), (a0, a1) = self.max_velocity, self.max_acceleration
        if p <= 0.0:
            return v0, a0
        if p >= 1.0:
            return v1, a1  # exactly the end values (= env.max_velocity / env.max_acceleration)
        return v0 + (v1 - v0) * p, a0 + (a1 - a0) * p

    def describe(self) -> str:
        (v0, v1), (a0, a1) = self.max_velocity, self.max_acceleration
        return (
            f"max_velocity {v0:g} -> {v1:g} m/s, max_acceleration {a0:g} -> {a1:g} m/s^2, linear over env steps "
            f"[{self.start_step:,}, {self.end_step:,}], constant outside"
        )


class HardwareScheduleWrapper(Wrapper):
    """Applies a :class:`HardwareSchedule` to the velocity-scale and collision-avoidance wrappers (see module doc)."""

    PRINT_EVERY = 0.05  # console line at every 5% of the ramp

    def __init__(self, env, schedule: HardwareSchedule, velocity_wrapper, collision_wrapper=None, step=0):
        super().__init__(env)
        if not getattr(velocity_wrapper, "_rate_limit_active", False):
            raise ValueError("HardwareScheduleWrapper schedules the rate cap: VelocityScaleWrapper needs max_acceleration")
        self.schedule = schedule
        self._velocity = velocity_wrapper
        self._collision = collision_wrapper
        self._num_envs = int(env.unwrapped.num_envs)
        self._step = int(step)
        self.step_fn = None
        self._logger = None
        self._printed = None
        self.current = {}
        print(f"HardwareScheduleWrapper: {schedule.describe()}; collision avoidance follows max_velocity: "
              f"{collision_wrapper is not None}")
        self._apply()

    # -- step source -------------------------------------------------------------------------------------------
    def bind_step(self, step_fn):
        """Read the step from ``step_fn()`` from now on (train_dreamer: the trainer's step)."""
        self.step_fn = step_fn
        self._apply()

    def set_step(self, step):
        self._step = int(step)
        self._apply()

    @property
    def step_count(self) -> int:
        return int(self.step_fn()) if self.step_fn is not None else self._step

    def set_logger(self, logger):
        self._logger = logger

    # -- application -------------------------------------------------------------------------------------------
    def _apply(self):
        step = self.step_count
        v, a = self.schedule.values(step)
        self._velocity.set_limits(v, a)
        if self._collision is not None:
            self._collision.set_max_vel(v)
        p = self.schedule.progress(step)
        self.current = {
            "step": step,
            "progress": p,
            "max_velocity": v,
            "max_acceleration": a,
            "collision_avoidance_max_vel": None if self._collision is None else float(self._collision.MAX_VEL),
        }
        mark = int(p / self.PRINT_EVERY + 1e-9)
        if mark != self._printed:
            self._printed = mark
            print(f"[hardware step {step}] ramp {100 * p:.0f}%: max_velocity {v:.4f} m/s, max_acceleration {a:.3f} m/s^2"
                  f"{'' if self._collision is None else f', collision avoidance {self._collision.MAX_VEL:.4f} m/s'}",
                  flush=True)
        return self.current

    def reset(self, *args, **kwargs):
        self._apply()
        return self.env.reset(*args, **kwargs)

    def step(self, actions, *args, **kwargs):
        self._step += self._num_envs
        cur = self._apply()
        if self._logger is not None:
            self._logger.scalar("hardware/max_velocity", cur["max_velocity"])
            self._logger.scalar("hardware/max_acceleration", cur["max_acceleration"])
            self._logger.scalar("hardware/ramp_progress", cur["progress"])
            if cur["collision_avoidance_max_vel"] is not None:
                self._logger.scalar("hardware/collision_avoidance_max_vel", cur["collision_avoidance_max_vel"])
        return self.env.step(actions, *args, **kwargs)
