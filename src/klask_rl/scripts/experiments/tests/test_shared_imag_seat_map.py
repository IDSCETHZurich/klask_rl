"""Shared-world imagination (branch shared-imag, 2026-10-11): the seat map that turns our decoded state into the
opponent's observation (opponent_seat_map) must reproduce exactly what the env gives the opponent seat.

- synthetic: the real observation groups (_PlayerPolicyCfg / _OpponentPolicyCfg, with and without
  env.state_units) evaluated on random world states (bodies, joints, ball), with stand-ins for the Isaac Lab
  term functions: the opponent group equals the seat map of the player group, bit for bit;
- recorded (optional): KLASK_SEAT_MAP_RUNS=<run dir>[:<run dir>...] checks stored replay buffers
  (policy / opponent memmaps) of finished state runs, read-only;
- the agent side: Dreamer._opponent_view with this map gives the same values.

CPU; run in the Isaac Lab container:
    python.sh -m pytest -q scripts/experiments/tests/test_shared_imag_seat_map.py
"""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]  # .../klask_rl (scripts/, source/)
PKG = ROOT / "source/klask_rl/klask_rl"
OBS_CFG = PKG / "tasks/manager_based/klask_rl/env_cfg/klask_rl_observations_cfg.py"
R2 = ROOT / "scripts/dreamer/r2dreamer"


# ---------------------------------------------------------------------------
# The observation config with stand-ins for Isaac Lab (its import needs the Isaac Sim app)
# ---------------------------------------------------------------------------


class _Term:
    def __init__(self, func=None, params=None, scale=None, **kw):
        self.func, self.params, self.scale = func, params or {}, scale


class _Entity:
    def __init__(self, name=None, body_names=None, joint_names=None, **kw):
        self.name, self.body_names, self.joint_names = name, body_names, joint_names


class _Group:
    pass


# Term functions: read a fake world (N envs) the way the Isaac Lab terms read the simulation.
def _body_xy_pos_w(env, asset_cfg):
    return env.world["body"][asset_cfg.body_names[0]]


def _joint_vel_rel(env, asset_cfg):
    return env.world["joint"][asset_cfg.joint_names[0]]


def _root_xy_pos_w(env, asset_cfg):
    return env.world["root_pos"][asset_cfg.name]


def _root_lin_xy_vel_w(env, asset_cfg):
    return env.world["root_vel"][asset_cfg.name]


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _load_obs_cfg():
    spec = importlib.util.spec_from_file_location("klask_params", PKG / "assets/robots/klask_params.py")
    kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)
    _unused = lambda *a, **k: None  # noqa: E731  (image / angle terms: not evaluated here)
    utils = _module("klask_rl.tasks.manager_based.klask_rl.utils_manager_based", body_xy_pos_w=_body_xy_pos_w,
                    root_xy_pos_w=_root_xy_pos_w, root_lin_xy_vel_w=_root_lin_xy_vel_w)
    utils.__getattr__ = lambda name: _unused
    mdp = _module("isaaclab.envs.mdp", joint_vel_rel=_joint_vel_rel)
    mdp.__getattr__ = lambda name: _unused
    stubs = {
        "isaaclab": _module("isaaclab"),
        "isaaclab.envs": _module("isaaclab.envs", mdp=mdp),
        "isaaclab.envs.mdp": mdp,
        "isaaclab.managers": _module("isaaclab.managers", ObservationGroupCfg=_Group, ObservationTermCfg=_Term,
                                     SceneEntityCfg=_Entity),
        "isaaclab.utils": _module("isaaclab.utils", configclass=lambda cls: cls),
        "klask_rl.assets.robots.klask_params": kp,
        "klask_rl.tasks.manager_based.klask_rl.utils_manager_based": utils,
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        name = "klask_rl.tasks.manager_based.klask_rl.env_cfg.klask_rl_observations_cfg"
        spec = importlib.util.spec_from_file_location(name, OBS_CFG)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "klask_rl.tasks.manager_based.klask_rl.env_cfg"
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


CFG = _load_obs_cfg()


def _terms(group):
    """(name, term) in observation order: the class definition order, bases first (dataclass field order)."""
    names = []
    for cls in reversed(type(group).__mro__):
        names += [k for k, v in vars(cls).items() if isinstance(v, _Term) and k not in names]
    return [(k, getattr(group, k)) for k in names]


def _observe(group, env):
    """Concatenated terms, each times its scale (float32), as the ObservationManager computes a group."""
    out = []
    for _, term in _terms(group):
        value = term.func(env, **term.params).float()
        if term.scale is not None:
            value = value * torch.as_tensor(term.scale, dtype=torch.float32)
        out.append(value)
    return torch.cat(out, dim=-1)


def _world(n=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return types.SimpleNamespace(world={
        "body": {"Peg_1": r(n, 2) * 0.1, "Peg_2": r(n, 2) * 0.1},
        "joint": {j: r(n, 1) * 0.3 for j in ("slider_to_peg_1", "ground_to_slider_1", "slider_to_peg_2", "ground_to_slider_2")},
        "root_pos": {"ball": r(n, 2) * 0.1},
        "root_vel": {"ball": r(n, 2) * 0.5},
    })


def _groups(units, max_velocity):
    player, opponent = CFG._PlayerPolicyCfg(), CFG._OpponentPolicyCfg()
    if units:
        CFG.apply_state_units(player, max_velocity)
        CFG.apply_state_units(opponent, max_velocity)
    return player, opponent


def _apply(index, sign, x):
    """The agent's map (Dreamer._opponent_view): x[..., index] * sign."""
    return x[..., torch.as_tensor(index)] * torch.as_tensor(sign, dtype=x.dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_seat_map_layout():
    index, sign = CFG.opponent_seat_map()
    assert index == [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11]  # own peg <-> other peg, ball in place
    assert sign == [-1.0] * 12  # 180-degree rotation
    assert CFG.opponent_seat_map(0.4) == CFG.opponent_seat_map(1.4)  # the same in every unit system
    x = torch.randn(10, 12)
    assert torch.equal(_apply(index, sign, _apply(index, sign, x)), x)  # an involution: seat 1 -> 0 as well


def test_the_term_order_is_the_state_layout():
    player, opponent = _groups(False, 0.4)
    assert [k for k, _ in _terms(player)] == list(CFG.STATE_TERMS)
    assert [k for k, _ in _terms(opponent)] == list(CFG.STATE_TERMS)


@pytest.mark.parametrize("units,max_velocity", [(False, 0.4), (True, 0.4), (True, 1.4)])
def test_opponent_group_is_the_seat_map_of_the_player_group(units, max_velocity):
    """Random bodies, joints and ball: what the env gives the opponent seat equals the map of our observation."""
    player, opponent = _groups(units, max_velocity)
    env = _world()
    own, opp = _observe(player, env), _observe(opponent, env)
    index, sign = CFG.opponent_seat_map(max_velocity)
    assert own.shape == opp.shape == (4096, 12)
    assert torch.equal(_apply(index, sign, own), opp)
    assert torch.equal(_apply(index, sign, opp), own)
    if units:  # in units the values are of order 1 (positions / board half-size, velocities / limits)
        plain = _observe(_groups(False, max_velocity)[0], env)
        assert torch.allclose(own * torch.tensor(CFG.state_unit_vector(max_velocity)), plain, rtol=1e-6, atol=1e-7)


def test_seat_map_refuses_seat_dependent_units(monkeypatch):
    divisors = CFG.state_unit_divisors(0.4)
    divisors["other_x_vel"] = 1.4
    monkeypatch.setattr(CFG, "state_unit_divisors", lambda v: divisors)
    with pytest.raises(ValueError, match="different units"):
        CFG.opponent_seat_map(0.4)


def _recorded_runs():
    env = os.environ.get("KLASK_SEAT_MAP_RUNS", "")
    return [p for p in env.split(":") if p]


@pytest.mark.skipif(not _recorded_runs(), reason="set KLASK_SEAT_MAP_RUNS to state-run dirs with a saved replay buffer")
@pytest.mark.parametrize("run", _recorded_runs())
def test_recorded_opponent_observation_is_the_seat_map(run):
    storage = Path(run) / "replay_buffer" / "storage"
    meta = json.loads((storage / "meta.json").read_text())
    stored = json.loads((storage / "storage_metadata.json").read_text())["len"]
    arr = {}
    for key in ("policy", "opponent"):
        shape = tuple(meta[key]["shape"])
        arr[key] = np.memmap(storage / f"{key}.memmap", dtype=np.float32, mode="r", shape=shape)
    rows = stored // arr["policy"].shape[1]
    own = torch.from_numpy(np.array(arr["policy"][:rows])).reshape(-1, 12)
    opp = torch.from_numpy(np.array(arr["opponent"][:rows])).reshape(-1, 12)
    index, sign = CFG.opponent_seat_map()
    assert own.shape[0] == stored and torch.equal(_apply(index, sign, own), opp)


def test_agent_opponent_view_with_the_seat_map():
    """The agent applies the same map to its decoded state (r2dreamer Dreamer.set_imag_opponent_view)."""
    if not (R2 / "dreamer.py").is_file():
        pytest.skip("r2dreamer not at scripts/dreamer/r2dreamer")
    pytest.importorskip("torchrl")
    import gymnasium as gym
    from omegaconf import OmegaConf

    sys.path.insert(0, str(R2))
    from dreamer import Dreamer

    m = OmegaConf.load(R2 / "configs/model/_base_.yaml")
    root = OmegaConf.create({"device": "cpu", "env": {"encoder": {"mlp_keys": "^policy$", "cnn_keys": "$^"},
                                                       "decoder": {"mlp_keys": "^policy$", "cnn_keys": "$^"}}})
    root.model = m
    m = root.model
    m.deter, m.hidden, m.discrete, m.depth, m.units, m.act, m.norm = 64, 32, 4, 4, 32, "SiLU", True
    m.rssm.stoch, m.compile, m.rep_loss, m.viz_decoder = 4, False, "dreamer", False
    m.opponent_separation, m.imag_opponent, m.imag_opponent_recompute, m.imag_opponent_world = True, "selfplay", True, "shared"
    m.train_device = m.sim_device = "cpu"
    m.train_devices, m.micro_batch_size = ["cpu"], 4
    box = gym.spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)
    agent = Dreamer(m, gym.spaces.Dict({"policy": box, "opponent": box}), gym.spaces.Box(-1, 1, (2,), dtype=np.float32))
    agent.set_imag_opponent_view({"policy": CFG.opponent_seat_map(0.4)})
    player, opponent = _groups(True, 0.4)
    env = _world(n=256, seed=1)
    own, opp = _observe(player, env), _observe(opponent, env)
    view = agent._opponent_view({"policy": types.SimpleNamespace(mode=lambda: own)})
    assert torch.equal(view["policy"], opp)
