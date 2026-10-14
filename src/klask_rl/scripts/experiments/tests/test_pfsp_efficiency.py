"""Efficient self-play (branch selfplay-eff, 2026-10-11): decisive-game PFSP scores, new members first, and the
dead-ball truncation counter. CPU; run in the Isaac Lab container (needs torch)."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS / "dreamer"))

from pfsp_pool import LATEST, PFSPPool  # noqa: E402


def _state(v):
    return {k: {"w": torch.tensor([float(v)])} for k in ("encoder", "rssm", "actor")}


def _pilot_like_pool(tmp_path, **kw):
    """Three members with the pilot's pattern (pool.json at 8.5M): mostly draws, decisive games decide."""
    pool = PFSPPool(tmp_path, p_latest=0.5, power=2.0, floor=0.02, prior_games=2.0, seed=3, **kw)
    for step, (w, d, l) in zip((0, 280_000, 5_550_000), ((294, 961, 45), (180, 1173, 59), (31, 352, 24))):
        mid = pool.add(step, _state(step))
        pool.record(mid, w, d, l)
    return pool


def test_wdl_scores_are_compressed_by_draws(tmp_path):
    pool = _pilot_like_pool(tmp_path)  # default score_mode "wdl"
    scores = [pool.score(m["id"]) for m in pool.members]
    assert all(0.5 < x < 0.6 for x in scores)
    p = [v for k, v in pool.probabilities().items() if k != LATEST]
    assert max(p) / min(p) < 1.6  # nearly uniform (decisive scoring: > 5x, next test)


def test_decisive_scores_prioritise_the_hard_member(tmp_path):
    pool = _pilot_like_pool(tmp_path, score_mode="decisive")
    assert pool.score(0) == pytest.approx((294 + 1) / (294 + 45 + 2))
    assert pool.score(2) == pytest.approx((31 + 1) / (31 + 24 + 2))
    p = pool.probabilities()
    assert p[2] > 5 * p[0]  # the recent, balanced member is played far more than the random initial policy
    assert p[LATEST] == 0.5 and sum(p.values()) == pytest.approx(1.0)


def test_thompson_explores_new_members_and_rarely_picks_beaten_ones(tmp_path):
    pool = _pilot_like_pool(tmp_path, score_mode="decisive", selection="thompson")
    new = pool.add(8_350_000, _state(9))  # unplayed: Beta(1, 1)
    picks = [pool.choose() for _ in range(6000)]
    pool_picks = [c for c in picks if c != LATEST]
    share = {m: pool_picks.count(m) / len(pool_picks) for m in range(4)}
    assert picks.count(LATEST) / len(picks) == pytest.approx(0.5, abs=0.03)
    assert share[new] > 0.2  # explored without any threshold
    assert share[2] > 3 * share[0]  # the balanced member over the clearly beaten random policy
    assert share[0] < 0.1
    probs = pool.probabilities()  # Monte Carlo estimate, own RNG
    assert sum(probs.values()) == pytest.approx(1.0) and probs[new] > probs[0]
    # More games against the new member narrow its posterior: a clearly beaten one fades.
    pool.record(new, 60, 30, 2)
    later = [c for c in (pool.choose() for _ in range(6000)) if c != LATEST]
    assert later.count(new) / len(later) < share[new] / 2


def test_thompson_state_round_trip(tmp_path):
    a = _pilot_like_pool(tmp_path / "a", score_mode="decisive", selection="thompson")
    state = a.state_dict()
    b = PFSPPool(tmp_path / "a", p_latest=0.5, power=2.0, floor=0.02, prior_games=2.0, seed=99, score_mode="decisive",
                 selection="thompson")
    b.load_state_dict(state)
    assert [a.choose() for _ in range(50)] == [b.choose() for _ in range(50)]


def test_thompson_needs_decisive_scores(tmp_path):
    with pytest.raises(ValueError):
        PFSPPool(tmp_path, selection="thompson")
    with pytest.raises(ValueError):
        PFSPPool(tmp_path, score_mode="decisive", selection="ucb")


def test_bad_score_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        PFSPPool(tmp_path, score_mode="elo")


def _ball_stalled():
    """Load ball_stalled from utils_manager_based.py without the Isaac Sim app (only its two speed helpers)."""
    path = SCRIPTS.parent / "source/klask_rl/klask_rl/tasks/manager_based/klask_rl/utils_manager_based.py"
    src = path.read_text()
    start = src.index("def ball_stalled(")
    end = src.index("\ndef ", start + 10)
    ns = {"torch": torch, "ManagerBasedRLEnv": object, "SceneEntityCfg": object}
    exec(
        "def ball_speed(env, ball_cfg):\n    v = env.scene[ball_cfg].data.root_lin_vel_w[:, :2]\n"
        "    return torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)\n" + src[start:end],
        ns,
    )
    return ns["ball_stalled"]


class _Env:
    def __init__(self, n):
        self.num_envs, self.device, self.step_dt = n, "cpu", 0.02
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.vel = torch.zeros(n, 3)
        self.scene = {"ball": types.SimpleNamespace(data=types.SimpleNamespace(root_lin_vel_w=self.vel))}

    def step(self, vel_xy):
        self.vel[:, :2] = vel_xy
        self.episode_length_buf += 1


def test_ball_stall_counter():
    f = _ball_stalled()
    env = _Env(3)
    moving = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.0, 0.0]])
    out = []
    for t in range(101):
        v = moving.clone()
        if t == 50:
            v[2] = torch.tensor([0.1, 0.0])  # env 2: one moving step resets its counter
        env.step(v)
        out.append(f(env, "ball", 0.01, 2.0))
    # Step 1 of an episode never counts; then 100 still steps (2 s) are needed.
    assert not out[99][0] and out[100][0]  # env 0: still from step 2 on -> stalled at step 101
    assert not any(o[1] for o in out)  # env 1: moving ball
    assert not out[100][2]  # env 2: counter restarted at step 51
    # A new episode restarts the counter.
    env.episode_length_buf[0] = 0
    env.step(moving)
    assert not f(env, "ball", 0.01, 2.0)[0]
