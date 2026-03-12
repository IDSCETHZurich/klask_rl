"""ManagerBasedRLEnv subclass that captures terminal observations before auto-reset.

IsaacLab's ``ManagerBasedRLEnv.step()`` resets terminated environments *before*
computing observations, so the returned obs for done envs is the post-reset
obs of the **new** episode — the true terminal obs is lost.

This subclass inserts an ``observation_manager.compute()`` call **before** the
reset, storing the result in ``self.extras["terminal_obs"]``.  A downstream
wrapper (e.g. ``IsaacLabVecEnv``) can then use it to return the correct
terminal observation to the agent/buffer.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


class KlaskRlDreamerEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with pre-reset terminal observation capture."""

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # ---- identical to ManagerBasedRLEnv.step() up to the reset ----

        # Clear terminal obs from the previous step so the wrapper doesn't
        # see stale data when no envs terminate this step.
        self.extras.pop("terminal_obs", None)
        self.extras.pop("terminal_env_ids", None)

        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # post-step: counters, terminations, rewards
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # ============================================================
            # ADDED: capture observations BEFORE the reset so the wrapper
            # can return the true terminal obs to the agent.
            # ============================================================
            terminal_obs = self.observation_manager.compute()
            self.extras["terminal_obs"] = {key: val[reset_env_ids].clone() for key, val in terminal_obs.items()}
            self.extras["terminal_env_ids"] = reset_env_ids

            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations (post-reset obs for done envs)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
