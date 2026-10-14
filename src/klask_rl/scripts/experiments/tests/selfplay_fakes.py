"""Minimal fakes of the self-play environment and a Dreamer snapshot for the CPU tests of DreamerSelfPlayWrapper
(test_pfsp.py)."""

from types import SimpleNamespace

import gymnasium as gym
import torch
from torch import nn

class _FakeTerminations:
    active_terms = ["goal_scored", "goal_conceded", "time_out"]

    def __init__(self, n):
        self.flags = {name: torch.zeros(n, dtype=torch.bool) for name in self.active_terms}

    def get_term(self, name):
        return self.flags[name]


class _FakeEnv(gym.Env):
    def __init__(self, n=3):
        self.num_envs = n
        self.single_action_space = gym.spaces.Box(-1, 1, (4,))
        self.termination_manager = _FakeTerminations(n)
        self.actions = []
        self.done = torch.zeros(n, dtype=torch.bool)

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        n = self.num_envs
        return {"opponent": torch.full((n, 20), 0.5), "opponent_image": torch.zeros(n, 4, 4, 3, dtype=torch.uint8)}

    def reset(self, *args, **kwargs):
        return self._obs(), {}

    def step(self, action):
        self.actions.append(action.clone())
        return self._obs(), torch.zeros(self.num_envs), self.done.clone(), torch.zeros_like(self.done), {}


class _FakeRSSM(nn.Module):
    def __init__(self, act_dim=4):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self._act_dim = act_dim
        self._device = torch.device("cpu")

    def initial(self, n):
        return torch.zeros(n, 2, 2), torch.zeros(n, 3)

    def get_feat(self, stoch, deter):
        return deter

    def obs_step(self, stoch, deter, prev_action, embed, is_first):
        return stoch + 1, deter + 1, None


class _FakeActor(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, feat):
        out = self.value.detach().expand(feat.shape[0], 2).clone()
        return SimpleNamespace(mode=out, rsample=lambda: out)


class _FakeEncoder(nn.Module):
    def forward(self, obs):
        return torch.zeros(obs["image"].shape[0], 5)


def _snapshot(value):
    return SimpleNamespace(encoder=_FakeEncoder(), rssm=_FakeRSSM(), actor=_FakeActor(value), device="cpu")


