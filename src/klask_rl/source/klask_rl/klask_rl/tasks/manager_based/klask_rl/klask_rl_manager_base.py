"""Extended ManagerBasedRLEnv that captures terminal observations before auto-reset."""

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv


class ExtendedManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with pre-reset terminal observation capture.

    Before auto-resetting terminated environments, captures the true
    terminal observation so the wrapper can return it as ``final_obs``.
    """

    def step(self, action: torch.Tensor):
        self.extras.pop("terminal_obs", None)
        self.extras.pop("terminal_env_ids", None)
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            terminal_obs = self.observation_manager.compute()
            self.extras["terminal_obs"] = {key: val[env_ids].clone() for key, val in terminal_obs.items()}
            self.extras["terminal_env_ids"] = env_ids
        super()._reset_idx(env_ids)
        super()._reset_idx(env_ids)
