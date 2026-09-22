"""CPU regressions for statistical accounting, recurrent timing and experiment planning."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS / "tournament"))

from policies import DreamerPolicy, episode_quotas
from report import generate_report, load_payoffs, statistics
from run_tournament import build_plan, resolve_path


def test_bootstrap_matches_individual_resampling():
    values = np.array([1, 1, 0.5, 0, 0])
    result = statistics(values, samples=1000, seed=42)
    expected = np.percentile(
        np.random.default_rng(42).choice(values, (1000, 5), replace=True).mean(1),
        [2.5, 97.5],
    )
    assert result["N"] == result["W"] + result["L"] + result["D"] == 5
    assert result["score"] == 0.5
    assert result["decisive_ratio"] == 1
    np.testing.assert_array_equal([result["score_ci_low"], result["score_ci_high"]], expected)


@pytest.mark.parametrize(
    "values,score,ratio",
    [([1] * 10, 1, "infinity"), ([0.5] * 10, 0.5, "undefined"), ([0] * 10, 0, 0)],
)
def test_degenerate_scores(values, score, ratio):
    result = statistics(values, samples=100)
    assert result["score_ci_low"] == result["score_ci_high"] == result["score"] == score
    assert result["decisive_ratio"] == ratio


@pytest.mark.parametrize("values", [[], [0.3], [np.nan], [[1]]])
def test_invalid_payoffs_rejected(values):
    with pytest.raises(ValueError):
        statistics(values)


class RecordingDreamer:
    def __init__(self, separated=True):
        self.opponent_separation = separated
        self.calls = []

    def get_initial_state(self, count):
        return {"prev_action": torch.zeros(count, 4 if self.opponent_separation else 2)}

    def act(self, obs, state, eval):
        self.calls.append((state["prev_action"].clone(), obs))
        return torch.ones(len(obs["is_first"]), 2), {"prev_action": torch.full_like(state["prev_action"], -999)}


@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize("condition", ["exact", "zero"])
def test_opponent_action_is_aligned_before_rssm_and_masked_at_reset(seat, condition):
    model = RecordingDreamer()
    policy = DreamerPolicy(model, 2, seat, condition)
    observations = {
        name: torch.full((2, 1), index) for index, name in enumerate(("policy", "opponent", "image", "opponent_image"))
    }
    actions = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]])
    original = actions.clone()
    for multiplier in (1, 2):
        policy.action(observations, torch.tensor([False, True]), multiplier * actions)
        wm, obs = model.calls[-1]
        own = multiplier * actions[0, seat * 2 : seat * 2 + 2]
        other = multiplier * actions[0, (1 - seat) * 2 : (1 - seat) * 2 + 2]
        if condition == "zero":
            other = torch.zeros_like(other)
        torch.testing.assert_close(wm[0], torch.cat((own, other)))
        torch.testing.assert_close(wm[1], torch.zeros(4))
        assert obs["image"] is observations["image" if seat == 0 else "opponent_image"]
    torch.testing.assert_close(actions, original)


def test_stock_policy_keeps_two_dimensional_action():
    model = RecordingDreamer(separated=False)
    policy = DreamerPolicy(model, 1, 0, "zero")
    policy.action(
        {"policy": torch.zeros(1, 20), "image": torch.zeros(1, 4, 4, 3)},
        torch.tensor([False]),
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
    )
    torch.testing.assert_close(model.calls[0][0], torch.tensor([[0.1, 0.2]]))


def test_quotas_do_not_select_fastest_finishers():
    quotas = episode_quotas(7, 3)
    completed = torch.zeros(3, dtype=torch.long)
    accepted_total = 0
    for done in ([True, False, False],) * 5 + ([True, True, True],) * 3:
        accepted = torch.tensor(done) & (completed < quotas)
        accepted_total += int(accepted.sum())
        completed += accepted.long()
    assert quotas.tolist() == completed.tolist() == [3, 2, 2]
    assert accepted_total == 7


@pytest.fixture
def metrics_module(monkeypatch):
    managers = ModuleType("isaaclab.managers")

    class Entity:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self, scene):
            pass

    managers.SceneEntityCfg = Entity
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    utils_name = "klask_rl.tasks.manager_based.klask_rl.utils_manager_based"
    utils = ModuleType(utils_name)
    for name in ("ball_in_own_half", "ball_speed", "collision_player_ball_simple"):
        setattr(utils, name, lambda env, *args, **kwargs: torch.zeros(env.num_envs))
    monkeypatch.setitem(sys.modules, utils_name, utils)
    path = SCRIPTS.parent / "source/klask_rl/klask_rl/tasks/manager_based/klask_rl/eval_metrics.py"
    spec = importlib.util.spec_from_file_location("test_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tracker_for(module, names, flags):
    env = SimpleNamespace(
        num_envs=len(flags),
        device="cpu",
        step_dt=0.02,
        scene={},
        termination_manager=SimpleNamespace(_term_names=names, _term_dones=torch.tensor(flags)),
    )
    env.unwrapped = env
    tracker = module.EvalMetricsTracker(env)
    tracker.update()
    return tracker


def test_simultaneous_flags_count_one_game(metrics_module, tmp_path):
    tracker = tracker_for(
        metrics_module,
        list(metrics_module.OUTCOME_NAMES),
        [
            [True, False, False, False, True],
            [False, True, True, False, False],
            [False, False, False, False, True],
        ],
    )
    tracker.finalize(torch.ones(3, dtype=torch.bool))
    assert (
        tracker.total_games,
        tracker.player_wins,
        tracker.opponent_wins,
        tracker.draws,
    ) == (3, 1, 1, 1)
    assert sum(tracker.term_counts.values()) == 5
    target = tmp_path / "outcomes.npz"
    tracker.save_npz(target)
    np.testing.assert_array_equal(load_payoffs(target), [1, 0, 0.5])


def test_disabled_terms_preserve_canonical_codes(metrics_module, tmp_path):
    tracker = tracker_for(metrics_module, ["opponent_in_goal", "time_out"], [[True, False], [False, True]])
    tracker.finalize(torch.ones(2, dtype=torch.bool))
    assert tracker.outcome == [3, 4]
    path = tmp_path / "games.npz"
    tracker.save_npz(path, extra_arrays={"env_id": [0, 1]})
    np.testing.assert_array_equal(load_payoffs(path, reverse=True), [0, 0.5])


def test_unrecognized_termination_is_not_silently_a_draw(metrics_module):
    tracker = tracker_for(metrics_module, ["unexpected"], [[True]])
    with pytest.raises(RuntimeError, match="no recognized termination"):
        tracker.finalize(torch.ones(1, dtype=torch.bool))
    assert tracker.total_games == 0 and tracker.outcome == []


def write_games(path, codes, run_id="run", job_id="leg0", complete=True):
    np.savez(
        path,
        outcome=np.asarray(codes, dtype=np.int8),
        outcome_schema_version=2,
        outcome_legend=np.array(
            [
                "goal_scored",
                "goal_conceded",
                "player_in_goal",
                "opponent_in_goal",
                "time_out",
            ]
        ),
        num_games=len(codes),
        meta_run_id=run_id,
        meta_job_id=job_id,
        meta_complete=str(complete),
    )


def test_report_reverses_seats_and_marks_partial_runs(tmp_path):
    plan = {
        "run_id": "run",
        "bootstrap_samples": 100,
        "bootstrap_seed": 5,
        "matches": [
            {
                "id": "pair",
                "a": "D-RSSM",
                "b": "PPO-B",
                "condition": "zero",
                "primary": True,
            }
        ],
        "jobs": [
            {"id": "leg0", "match": "pair", "seat0": "D-RSSM", "games": 2},
            {"id": "leg1", "match": "pair", "seat0": "PPO-B", "games": 2},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(plan))
    (tmp_path / "leg0").mkdir()
    write_games(tmp_path / "leg0/games.npz", [0, 4])
    result = generate_report(tmp_path)
    assert result["complete"] is False
    (tmp_path / "leg1").mkdir()
    write_games(tmp_path / "leg1/games.npz", [1, 2], job_id="leg1")
    result = generate_report(tmp_path)
    row = result["results"][0]
    assert result["complete"] is True
    assert (row["N"], row["W"], row["L"], row["D"], row["score"]) == (4, 3, 0, 1, 0.875)
    write_games(tmp_path / "leg1/games.npz", [1, 2], run_id="wrong", job_id="leg1")
    with pytest.raises(ValueError, match="Provenance"):
        generate_report(tmp_path)


def test_legacy_outcomes_require_explicit_validation(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(
        path,
        outcome=np.array([0]),
        outcome_legend=np.array(["goal_scored"]),
        num_games=1,
    )
    with pytest.raises(ValueError, match="legacy"):
        load_payoffs(path)


def test_paths_are_portable_without_checkpoint_search(tmp_path):
    assert resolve_path("/workspace/klask_rl/logs/model.pt", tmp_path) == tmp_path / "logs/model.pt"
    assert resolve_path("logs/model.pt", tmp_path) == tmp_path / "logs/model.pt"
    assert resolve_path("/elsewhere/model.pt", tmp_path) == Path("/elsewhere/model.pt")


def test_plan_uses_same_checkpoint_both_conditions_and_all_pairs(tmp_path, monkeypatch):
    # Missing assets are reported, but planning itself remains inspectable on CPU.
    config = SCRIPTS / "tournament/config.yaml"
    args = SimpleNamespace(
        project_root=tmp_path,
        config=config,
        games=11,
        num_envs=4,
        seed=7,
        episode_length_s=None,
        device="cuda:0",
        actuator_checkpoint=None,
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    plan, blockers = build_plan(args)
    assert blockers
    assert len(plan["matches"]) == 4 and len(plan["jobs"]) == 8
    assert sum(job["games"] for job in plan["jobs"]) == 44
    assert len([match for match in plan["matches"] if match["primary"]]) == 3
    assert plan["jobs"][0]["seed"] == plan["jobs"][2]["seed"] == 7
    assert plan["agents"]["PPO-B"]["checkpoint"].endswith("new_AM/export/klask_ppo_nn_v1.1.pth")
    assert list(plan["agents"]) == ["D-RSSM", "PPO-B", "FastSAC"]


@pytest.mark.parametrize("interrupt", [False, True])
def test_worker_saves_games_and_resets_state_with_autoreset(tmp_path, monkeypatch, metrics_module, interrupt):
    gym = pytest.importorskip("gymnasium")
    from run_tournament import sha256
    from worker import run

    class Environment:
        num_envs = 2
        device = "cpu"
        step_dt = 0.02
        scene = ()
        closed = False
        unwrapped = property(lambda self: self)
        single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(-1, 1, (20,)),
                "opponent": gym.spaces.Box(-1, 1, (20,)),
                "image": gym.spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8),
            }
        )
        termination_manager = SimpleNamespace(
            _term_names=list(metrics_module.OUTCOME_NAMES), _term_dones=torch.zeros(2, 5, dtype=torch.bool)
        )

        def obs(self):
            return {
                "policy": torch.zeros(2, 20),
                "opponent": torch.zeros(2, 20),
                "image": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
            }

        def reset(self, seed):
            self.steps = 0
            return self.obs(), {}

        def step(self, actions):
            self.steps += 1
            if interrupt and self.steps == 5:
                raise KeyboardInterrupt
            dones = torch.full((2,), self.steps % 2 == 0)
            self.termination_manager._term_dones.zero_()
            self.termination_manager._term_dones[:, 0] = dones
            # Exercise in-place mutation by the real action wrapper chain.
            actions.mul_(10)
            return self.obs(), torch.zeros(2), dones, torch.zeros(2, dtype=torch.bool), {}

        def close(self):
            self.closed = True

    class PPO:
        is_rnn = False

        def get_batch_size(self, obs, count):
            pass

        def obs_to_torch(self, obs):
            return obs

        def get_action(self, obs, deterministic):
            return torch.full((2, 2), 0.2)

    env = Environment()
    dreamer = RecordingDreamer()
    evaluation = ModuleType("evaluation")
    evaluation._make_eval_env = lambda *args, **kwargs: (env, None)
    evaluation._load_dreamer_agent = lambda *args: dreamer
    evaluation._load_ppo_opponent = lambda *args: PPO()
    evaluation._load_fast_sac_opponent = lambda *args: None
    monkeypatch.setitem(sys.modules, "evaluation", evaluation)
    monkeypatch.setitem(sys.modules, "klask_rl.tasks.manager_based.klask_rl.eval_metrics", metrics_module)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test CPU")
    artifact = tmp_path / "artifact"
    artifact.write_text("fake checkpoint/config")
    spec = {
        "checkpoint": str(artifact),
        "config": str(artifact),
        "checkpoint_sha256": sha256(artifact),
        "config_sha256": sha256(artifact),
    }
    plan = {
        "run_id": "test",
        "project_root": str(tmp_path),
        "num_envs": 2,
        "device": "cpu",
        "source_sha256": {},
        "asset_sha256": {},
        "dependency_source_sha256": {},
        "evaluation_config": {"seed": 0, "env": {"seed": 0}},
        "agents": {"D-RSSM": {**spec, "kind": "dreamer"}, "PPO-B": {**spec, "kind": "ppo"}},
        "jobs": [{"id": "leg", "seat0": "D-RSSM", "seat1": "PPO-B", "condition": "exact", "seed": 0, "games": 5}],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(plan))
    status = run(SimpleNamespace(manifest=manifest, job="leg", device="cpu"), SimpleNamespace(is_running=lambda: True))
    assert env.closed and status == (130 if interrupt else 0)
    with np.load(tmp_path / "leg/games.npz", allow_pickle=False) as data:
        assert int(data["num_games"]) == (4 if interrupt else 5)
        assert str(data["meta_complete"]) == str(not interrupt)
        np.testing.assert_array_equal(data["env_id"], [0, 1, 0, 1] if interrupt else [0, 1, 0, 1, 0])
    # Next observation uses the unmodified normalized commands, and an
    # auto-reset at the second step clears both action slots on the third call.
    torch.testing.assert_close(dreamer.calls[1][0], torch.tensor([[1, 1, 0.2, 0.2]] * 2))
    torch.testing.assert_close(dreamer.calls[2][0], torch.zeros(2, 4))
