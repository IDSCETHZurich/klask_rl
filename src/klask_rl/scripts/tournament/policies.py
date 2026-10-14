"""Seat-independent policy adapters; no simulator imports, so timing is testable."""

import torch


class DreamerPolicy:
    def __init__(
        self, agent, num_envs, seat, opponent_input, seed=0, action_timing="aligned", state_dim=None, state_units=None
    ):
        if opponent_input not in ("exact", "random", "zero", "mirror"):
            raise ValueError(f"Unknown opponent input: {opponent_input}")
        if action_timing not in ("aligned", "legacy"):
            raise ValueError(f"Unknown action timing: {action_timing}")
        if opponent_input == "mirror" and action_timing != "aligned":
            raise ValueError("The mirror opponent input is implemented for aligned timing only")
        self.agent = agent
        self.seat = seat
        # State-based agents (env.state_only) read the first state_dim values of the policy
        # observation: the raw positions/velocities, which come first in the extended group.
        self.state_dim = state_dim
        # ... divided by these units when the agent was trained with env.state_units (one per value).
        self.state_units = state_units
        self.opponent_input = opponent_input
        self.state = agent.get_initial_state(num_envs)
        self.action_timing = action_timing
        self.delayed_other = None
        self.random_generator = torch.Generator(device=self.state["prev_action"].device).manual_seed(seed)
        # "mirror" (2026-10-10): the same model plays the opponent's seat on the opponent's view with its
        # own recurrent state; the command it chose at the previous step stands in for the opponent's
        # command that produced this observation (P1 of the opponent-adaptation study, aligned timing).
        self.mirror_state = agent.get_initial_state(num_envs) if opponent_input == "mirror" else None
        self.mirror_last = None

    def action(self, observations, is_first, previous_actions):
        own = previous_actions[:, 2 * self.seat : 2 * self.seat + 2]
        other_seat = 1 - self.seat
        other = previous_actions[:, 2 * other_seat : 2 * other_seat + 2]
        if self.action_timing == "legacy":
            delayed = torch.zeros_like(other) if self.delayed_other is None else self.delayed_other
            self.delayed_other = other.masked_fill(is_first[:, None], 0).clone()
            other = delayed
        if self.opponent_input == "zero":
            other = torch.zeros_like(other)
        elif self.opponent_input == "random":
            other = torch.rand(other.shape, device=other.device, generator=self.random_generator) * 2 - 1
        elif self.opponent_input == "mirror":
            other = self._mirror_slot(observations, is_first, own)
        # D-RSSM-drop: an availability bit, 0 in the deployable "zero" condition (command unknown).
        available = torch.full_like(other[:, :1], 0.0 if self.opponent_input == "zero" else 1.0)
        if self.agent.opponent_separation and getattr(self.agent, "opponent_flag", False):
            wm_action = torch.cat((own, other * available, available), -1)
        else:
            wm_action = torch.cat((own, other), -1) if self.agent.opponent_separation else own
        # act() consumes state.prev_action BEFORE it reads obs.opponent_action.
        # Both slots must describe the transition that produced this observation.
        # The simulator's auto-reset observation starts a fresh recurrent state.
        self.state["prev_action"] = wm_action.masked_fill(is_first[:, None], 0)
        policy_obs = observations["policy" if self.seat == 0 else "opponent"]
        obs = {
            "policy": self._state(policy_obs),
            "image": observations["image" if self.seat == 0 else "opponent_image"],
            "is_first": is_first[:, None],
            "opponent_action": other,
            "opponent_action_available": available,
        }
        action, self.state = self.agent.act(obs, self.state, eval=True)
        # Clone: compiled inference can reuse its output buffer on the next call.
        return action.clone()


    def _mirror_slot(self, observations, is_first, own):
        """The mirror's previous command (its estimate of the opponent's last command); then its next one."""
        reset = is_first[:, None]
        last = torch.zeros_like(own) if self.mirror_last is None else self.mirror_last
        last = last.masked_fill(reset, 0)
        width = self.mirror_state["prev_action"].shape[-1]
        known = torch.ones_like(own[:, :1])
        prev = torch.cat((last, own, known), -1)[:, :width] if width > 2 else last
        # act() keeps the first slot (the mirror's own last command) and takes its opponent part,
        # our executed command, from obs["opponent_action"] (aligned timing).
        self.mirror_state["prev_action"] = prev.masked_fill(reset, 0)
        opponent_view = observations["opponent" if self.seat == 0 else "policy"]
        obs = {
            "policy": self._state(opponent_view),
            "image": observations["opponent_image" if self.seat == 0 else "image"],
            "is_first": reset,
            "opponent_action": own,
            "opponent_action_available": known,
        }
        command, self.mirror_state = self.agent.act(obs, self.mirror_state, eval=True)
        self.mirror_last = command.clone()
        return last

    def _state(self, policy_obs):
        if self.state_dim is None:
            return policy_obs
        values = policy_obs[:, : self.state_dim]
        if self.state_units is not None:
            values = values / torch.as_tensor(self.state_units, dtype=values.dtype, device=values.device)
        return values


class FeedForwardPolicy:
    def __init__(self, agent, kind, seat, initial_observations):
        self.agent = agent
        self.kind = kind
        self.key = (("policy", "opponent") if kind == "ppo" else ("fast_sac_player", "fast_sac_opponent"))[seat]
        if kind == "ppo":
            agent.get_batch_size(initial_observations[self.key], 1)
            if agent.is_rnn:
                raise ValueError("This tournament supports the supplied feed-forward PPO-B checkpoint only.")
        agent.has_batch_dimension = True

    def action(self, observations, is_first, previous_actions):
        obs = self.agent.obs_to_torch(observations[self.key])
        return self.agent.get_action(obs, True).clone()


def episode_quotas(games, num_envs, device="cpu"):
    """Predetermine per-env sample counts, avoiding a fastest-finishers cutoff."""
    if games <= 0 or num_envs <= 0:
        raise ValueError("games and num_envs must be positive")
    indices = torch.arange(num_envs, device=device)
    return games // num_envs + (indices < games % num_envs).long()
