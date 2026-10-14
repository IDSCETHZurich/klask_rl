"""The Dreamer loader must play with the trainable weights, not stale frozen copies."""

import sys
from pathlib import Path

import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS / "dreamer"))

from checkpoint_loaders import sync_alias_modules


class Agent(nn.Module):
    """Mimics Dreamer.clone_and_freeze: frozen copy shares parameters, not buffers."""

    def __init__(self):
        super().__init__()
        self.actor = nn.BatchNorm1d(2)
        self._frozen_actor = nn.BatchNorm1d(2)
        for mine, frozen in zip(self.actor.parameters(), self._frozen_actor.parameters()):
            frozen.data = mine.data
        self._inference_actor = self._frozen_actor
        self._slow_value = nn.Linear(2, 1)
        self._frozen_slow_value = nn.Linear(2, 1)


def test_stale_copies_are_overwritten_by_trainable_weights():
    agent = Agent()
    state = {k: torch.full_like(v, 7.0) if k.startswith("actor.") or k.startswith("_slow_value.") else v for k, v in agent.state_dict().items()}
    # A snapshot with stale copies: keys of the copies load after the trainable
    # ones and overwrite the shared parameters with the old values.
    stale = {**state, **{k: torch.full_like(v, -1.0) for k, v in state.items() if k.startswith(("_frozen_", "_inference_"))}}
    agent.load_state_dict(stale)
    assert agent.actor.weight[0] == -1  # the bug: shared storage took the stale value
    agent.load_state_dict({k: v for k, v in state.items() if not k.startswith(("_frozen_", "_inference_"))}, strict=False)
    sync_alias_modules(agent)
    for module in (agent.actor, agent._frozen_actor, agent._inference_actor):
        assert torch.all(module.weight == 7) and torch.all(module.running_mean == 7)
    assert torch.all(agent._frozen_slow_value.weight == 7)
