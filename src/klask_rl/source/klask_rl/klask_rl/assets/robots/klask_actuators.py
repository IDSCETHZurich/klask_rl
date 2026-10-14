"""Actuator models for the KLASK peg joints."""

from __future__ import annotations

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, IdealPDActuator
from isaaclab.utils import configclass


class ZeroDelayAwarePDActuator(DelayedPDActuator):
    """:class:`DelayedPDActuator` that bypasses its delay buffers when ``max_delay == 0``.

    With a zero time lag each of the three delay buffers just returns a copy of its input,
    but every ``compute`` still issues two host-device synchronizations per buffer (the
    ``torch.any`` checks inside ``CircularBuffer.append`` / ``__getitem__``), i.e. 480 syncs per
    env step for the four KLASK actuators at decimation 20.  Skipping the buffers yields
    bit-identical efforts.  ``reset`` is inherited unchanged, so the (zero) time lags are still
    sampled and the random number stream is the same as with :class:`DelayedPDActuator`.
    """

    def compute(self, control_action, joint_pos, joint_vel):
        if self.cfg.max_delay == 0:
            return IdealPDActuator.compute(self, control_action, joint_pos, joint_vel)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class ZeroDelayAwarePDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`ZeroDelayAwarePDActuator`."""

    class_type: type = ZeroDelayAwarePDActuator
