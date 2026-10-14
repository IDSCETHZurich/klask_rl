"""Regression tests (2026-10-10): with env.state_units the observations are in units, but the learned actuator model
(fitted on m/s) and the collision avoidance (edges in metres) read peg velocities and positions from the observations.
Both must convert back. Run in the Isaac Lab container:

    python.sh -m pytest -q scripts/experiments/tests/test_state_units_wrappers.py
"""

import importlib.util
import sys
import types
from pathlib import Path

import gymnasium as gym
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "source/klask_rl/klask_rl"


def _load(name, rel):
    """Load one module from its file with light stubs (the klask_rl package import needs the Isaac Sim app)."""
    params = importlib.util.spec_from_file_location("klask_params", PKG / "assets/robots/klask_params.py")
    kp = importlib.util.module_from_spec(params)
    params.loader.exec_module(kp)
    robots = types.ModuleType("klask_rl.assets.robots")
    robots.klask_params = kp
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = type("SceneEntityCfg", (), {})
    stubs = {"klask_rl.assets.robots": robots, "klask_rl.assets.robots.klask_params": kp, "isaaclab.managers": managers}
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(name, PKG / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod

CKPT = PKG / "tasks/manager_based/klask_rl/actuator_model" / (
    "model_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states.pt"
)
V, HALF = 0.4, (0.16, 0.21)


class FakeEnv(gym.Env):
    """Returns fixed own/other peg states (metres, m/s); ``units`` scales them as env.state_units does."""

    def __init__(self, n=3, units=False):
        self.num_envs, self.device, self.units, self.last_action = n, "cpu", units, None
        pos, vel = torch.tensor([0.05, -0.10]), torch.tensor([0.30, -0.20])
        other = torch.tensor([-0.04, 0.12])
        row = torch.cat([pos, vel, other, torch.zeros(6)])
        if units:
            div = torch.tensor([HALF[0], HALF[1], V, V, HALF[0], HALF[1], 1, 1, 1, 1, 1, 1])
            row = row / div
        self.row = row.repeat(n, 1)

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        return {"policy": self.row.clone(), "opponent": self.row.clone()}

    def reset(self, *a, **k):
        return self._obs(), {}

    def step(self, action):
        self.last_action = action.clone()
        z = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), torch.zeros(self.num_envs), z, z, {}


def _actuator(units):
    ActuatorModelWrapper = _load("actuator_model", "tasks/manager_based/klask_rl/actuator_model/actuator_model.py").ActuatorModelWrapper

    kw = {"obs_vel_unit": V, "obs_pos_unit": HALF} if units else {}
    return ActuatorModelWrapper(FakeEnv(units=units), device="cpu", model_file=str(CKPT), **kw)


@pytest.mark.skipif(not CKPT.is_file(), reason="actuator checkpoint missing")
def test_actuator_model_sees_m_per_s_with_state_units():
    plain, units = _actuator(False), _actuator(True)
    for w in (plain, units):
        w.reset()
        for _ in range(3):
            w.step(torch.full((3, 4), 0.2))
    assert torch.allclose(units.state_buffer_1, plain.state_buffer_1, atol=1e-6)
    assert torch.allclose(units.state_buffer_1[0, :2], torch.tensor([0.30, -0.20]), atol=1e-6)
    assert torch.allclose(units.position_player, plain.position_player, atol=1e-6)
    assert torch.allclose(units.env.last_action, plain.env.last_action, atol=1e-6)  # same realised velocity command


def test_collision_avoidance_sees_metres_with_state_units():
    KlaskRlCollisionAvoidanceWrapper = _load(
        "klask_rl_training_wrappers", "tasks/manager_based/klask_rl/wrappers/klask_rl_training_wrappers.py"
    ).KlaskRlCollisionAvoidanceWrapper

    plain = KlaskRlCollisionAvoidanceWrapper(FakeEnv(units=False), max_vel=V)
    units = KlaskRlCollisionAvoidanceWrapper(FakeEnv(units=True), max_vel=V, obs_pos_unit=HALF)
    for w in (plain, units):
        w.reset()
    assert torch.allclose(units.state_1, plain.state_1, atol=1e-6)
    assert torch.allclose(units.state_2, plain.state_2, atol=1e-6)
    a = torch.tensor([[0.4, -0.4, 0.4, 0.4]] * 3)
    units.step(a.clone())
    plain.step(a.clone())
    assert torch.allclose(units.env.last_action, plain.env.last_action, atol=1e-6)
