"""PFSP pool (scripts/dreamer/pfsp_pool.py) and the self-play wrapper's pool loading.
CPU; run in the Isaac Lab container (needs torch)."""

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS / "dreamer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfsp_pool import LATEST, PFSPPool, outcome_counts  # noqa: E402


def _state(value):
    return {"encoder": {"w": torch.tensor([value])}, "rssm": {"w": torch.tensor([value])}, "actor": {"w": torch.tensor([value])}}


def test_scores_weights_and_choice(tmp_path):
    pool = PFSPPool(tmp_path, p_latest=0.5, power=2.0, floor=0.05, prior_games=2.0, seed=1)
    assert pool.choose() == LATEST  # empty pool: always the current weights
    a = pool.add(0, _state(0.0))
    b = pool.add(250_000, _state(1.0))
    assert (a, b) == (0, 1) and (tmp_path / "pool.json").is_file()
    assert pool.score(a) == 0.5  # unplayed: prior score
    pool.record(a, wins=18, draws=0, losses=0)  # easy opponent
    pool.record(b, wins=2, draws=4, losses=12)  # hard opponent
    assert pool.score(a) == pytest.approx((18 + 1) / 20)
    assert pool.score(b) == pytest.approx((2 + 2 + 1) / 20)
    w = pool.weights()
    assert w[1] > 10 * w[0]  # hard opponents dominate
    probs = pool.probabilities()
    assert probs[LATEST] == 0.5 and sum(probs.values()) == pytest.approx(1.0)
    draws = [pool.choose() for _ in range(4000)]
    for option, p in probs.items():
        assert draws.count(option) / len(draws) == pytest.approx(p, abs=0.03)
    assert torch.equal(pool.load(b)["actor"]["w"], torch.tensor([1.0]))


def test_state_round_trip_continues_the_rng(tmp_path):
    pool = PFSPPool(tmp_path, seed=3)
    pool.add(0, _state(0.0))
    pool.add(1, _state(1.0))
    pool.record(LATEST, 1, 2, 3)
    state = torch.load(_save(tmp_path, pool.state_dict()), weights_only=False)
    expected = [pool.choose() for _ in range(20)]
    restored = PFSPPool(tmp_path, seed=99)
    restored.load_state_dict(state)
    assert [restored.choose() for _ in range(20)] == expected
    assert restored.latest == {"wins": 1, "draws": 2, "losses": 3}


def _save(tmp_path, obj):
    path = tmp_path / "ckpt.pt"
    torch.save(obj, path)
    return path


def test_missing_member_file_is_an_error(tmp_path):
    pool = PFSPPool(tmp_path)
    pool.add(0, _state(0.0))
    state = pool.state_dict()
    (tmp_path / pool.members[0]["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        PFSPPool(tmp_path).load_state_dict(state)


def test_outcome_counts():
    assert outcome_counts([1.0, 0.0, -1.0, 1.0, 0.0]) == (2, 2, 1)
    assert outcome_counts([]) == (0, 0, 0)


def test_wrapper_loads_pool_members_in_place_and_hides_latents():
    pytest.importorskip("tensordict")
    from dreamer_self_play import DreamerSelfPlayWrapper
    from selfplay_fakes import _FakeEnv, _snapshot

    wrapper = DreamerSelfPlayWrapper(_FakeEnv(), games_to_track=8, emit_latents=False)
    wrapper.set_opponent(_snapshot(0.1), device="cpu")
    actor = wrapper._opponent_actor_orig
    wrapper.reset()
    obs, *_ = wrapper.step(torch.zeros(3, 2))
    assert torch.allclose(obs["opponent_action"], torch.full((3, 2), 0.1))
    assert "opp_stoch" not in obs and "opp_deter" not in obs
    member = {
        "encoder": wrapper._opponent_encoder_orig.state_dict(),
        "rssm": wrapper._opponent_rssm_orig.state_dict(),
        "actor": {"value": torch.tensor(0.7)},
    }
    wrapper.load_opponent_state(member)
    assert wrapper._opponent_actor_orig is actor  # in place (keeps compiled wrappers valid)
    obs, *_ = wrapper.step(torch.zeros(3, 2))
    assert torch.allclose(obs["opponent_action"], torch.full((3, 2), 0.7))
    assert wrapper.opponent_encoder is wrapper._opponent_encoder_orig
