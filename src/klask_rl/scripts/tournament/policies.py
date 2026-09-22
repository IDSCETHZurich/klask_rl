"""Seat-independent policy adapters; no simulator imports, so timing is testable."""

import torch


class DreamerPolicy:
    def __init__(self, agent, num_envs, seat, opponent_input, seed=0, action_timing="aligned"):
        if opponent_input not in ("exact", "random", "zero"):
            raise ValueError(f"Unknown opponent input: {opponent_input}")
        if action_timing not in ("aligned", "legacy"):
            raise ValueError(f"Unknown action timing: {action_timing}")
        self.agent = agent
        self.seat = seat
        self.opponent_input = opponent_input
        self.state = agent.get_initial_state(num_envs)
        self.action_timing = action_timing
        self.delayed_other = None
        self.random_generator = torch.Generator(device=self.state["prev_action"].device).manual_seed(seed)

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
        wm_action = torch.cat((own, other), -1) if self.agent.opponent_separation else own
        # act() consumes state.prev_action BEFORE it reads obs.opponent_action.
        # Both slots must describe the transition that produced this observation.
        # The simulator's auto-reset observation starts a fresh recurrent state.
        self.state["prev_action"] = wm_action.masked_fill(is_first[:, None], 0)
        obs = {
            "policy": observations["policy" if self.seat == 0 else "opponent"],
            "image": observations["image" if self.seat == 0 else "opponent_image"],
            "is_first": is_first[:, None],
            "opponent_action": other,
        }
        action, self.state = self.agent.act(obs, self.state, eval=True)
        # Clone: compiled inference can reuse its output buffer on the next call.
        return action.clone()


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
