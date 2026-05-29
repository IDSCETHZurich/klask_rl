"""IsaacLab adapter for FastSAC training.

Wraps an IsaacLab ManagerBasedRLEnv to expose the same API as ``KlaskWarpEnv``
(``reset_all``, ``step``, ``obs_dim``, ``act_dim``, ``num_envs``), so the
FastSAC training loop can use it as a drop-in replacement.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Adapter wrapper
# ---------------------------------------------------------------------------

# Contact detection threshold: ball_radius(0.007) + peg_radius(0.0075) + margin(0.002)
CONTACT_DIST_THRESHOLD = 0.0165


class FastSACEnvWrapper:
    """Wraps an IsaacLab env to match the ``KlaskWarpEnv`` API for FastSAC.

    The env must use ``FastSACManagerBasedRLEnv`` as entry_point so terminal
    observations are captured before auto-reset.

    Args:
        env: An IsaacLab ManagerBasedRLEnv (should be FastSACManagerBasedRLEnv).
    """

    def __init__(self, env: ManagerBasedRLEnv):
        self._env = env
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        self.num_envs = unwrapped.num_envs
        self.device = unwrapped.device

        # Obs/act dims from the env's spaces
        policy_space = unwrapped.single_observation_space["policy"]
        self.obs_dim = policy_space.shape[0]
        self.act_dim = unwrapped.single_action_space.shape[0]

    def reset_all(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Reset all environments.

        Returns ``policy`` obs [num_envs, obs_dim], or ``(policy, opponent)``
        when the env exposes a separate opponent observation group.
        """
        obs_dict, _ = self._env.reset()
        if "opponent" in obs_dict:
            return obs_dict["policy"], obs_dict["opponent"]
        return obs_dict["policy"]

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step all environments.

        Args:
            actions: [num_envs, act_dim] in action space range.

        Returns:
            obs: [num_envs, obs_dim] post-reset obs for terminated envs.
            rewards: [num_envs].
            dones: [num_envs] bool.
            info: dict with time_outs, ball_peg_contact, goal_scored,
                  own_goal, peg1_in_goal, peg2_in_goal, final_obs.
        """
        # Pad to full action dim if only player actions are provided
        # (e.g. self_play=False sends 2-dim, env expects 4-dim).
        if actions.shape[-1] < self.act_dim:
            pad = torch.zeros(
                actions.shape[0],
                self.act_dim - actions.shape[-1],
                device=actions.device,
                dtype=actions.dtype,
            )
            actions = torch.cat([actions, pad], dim=-1)

        obs_dict, reward, terminated, truncated, extras = self._env.step(actions)

        dones = terminated | truncated
        unwrapped = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env

        # --- Build info dict ---
        tm = unwrapped.termination_manager
        info = {
            "time_outs": unwrapped.reset_time_outs.long(),
            "goal_scored": tm.get_term("goal_scored"),
            "own_goal": tm.get_term("goal_conceded"),
            "peg1_in_goal": tm.get_term("player_in_goal"),
            "peg2_in_goal": tm.get_term("opponent_in_goal"),
        }

        # Contact detection: distance between ball and peg1
        ball_pos = unwrapped.scene["ball"].data.root_pos_w[:, :2]
        peg1_pos = unwrapped.scene["klask"].data.body_pos_w[:, unwrapped.scene["klask"].find_bodies("Peg_1")[0][0], :2]
        dist_ball_peg = torch.norm(ball_pos - peg1_pos, dim=-1)
        info["ball_peg_contact"] = dist_ball_peg < CONTACT_DIST_THRESHOLD

        # --- Terminal observations (final_obs) ---
        # Default: current obs for all envs
        obs_policy = obs_dict["policy"]
        final_obs = obs_policy.clone()

        # Swap in true terminal obs for envs that just ended
        terminal_obs = extras.get("terminal_obs")
        terminal_env_ids = extras.get("terminal_env_ids")
        if terminal_obs is not None and terminal_env_ids is not None and len(terminal_env_ids) > 0:
            final_obs[terminal_env_ids] = terminal_obs["policy"]

        info["final_obs"] = final_obs

        # --- Opponent observations (for self-play) ---
        if "opponent" in obs_dict:
            info["opponent_obs"] = obs_dict["opponent"]
            # Also provide terminal opponent obs
            if terminal_obs is not None and terminal_env_ids is not None and len(terminal_env_ids) > 0:
                opp_final = obs_dict["opponent"].clone()
                opp_final[terminal_env_ids] = terminal_obs["opponent"]
                info["opponent_final_obs"] = opp_final

        return obs_policy, reward, dones, info
