"""CPU checks for counters, paired diagnostics, data export and experiment isolation."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS / "experiments"))
sys.path.insert(0, str(SCRIPTS / "tournament"))
from analyze import (
    conditioning_summary,
    games_summary,
    learning_curve_summary,
    read_rows,
)
from baselines import devices_for_run, scratch_ppo_config
from conditioning import conditioning_actions, open_loop, raw_kl
from policies import DreamerPolicy
from puzzle import experiment_config
from raw_data import TrajectoryRecorder, export_games, write_csv
from training import PPOExperimentObserver, TrainingRecorder


@pytest.mark.parametrize("count,expected", [(2, ["cuda:0", "cuda:1"]), (4, ["cuda:2", "cuda:3"])])
def test_last_two_visible_gpus(monkeypatch, count, expected):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    assert devices_for_run() == expected
    with pytest.raises(ValueError):
        devices_for_run(["cuda:0", "cuda:0"])


def test_scratch_ppo_cannot_inherit_old_initialization():
    cfg = scratch_ppo_config(
        SCRIPTS / "rl_games/config/klask_ppo_config_with_pretraining.yaml",
        SCRIPTS.parent,
        7,
        "cuda:3",
        512,
        9_000_000,
    )
    assert cfg["params"]["load_checkpoint"] is False
    assert cfg["params"]["load_path"] == ""
    assert cfg["params"]["seed"] == 7
    assert cfg["params"]["config"]["device"] == "cuda:3"
    assert cfg["params"]["config"]["max_frames"] == 9_000_000
    assert cfg["params"]["network"]["mlp"]["units"] == [256, 128, 64]


def test_checkpoint_threshold_records_actual_steps_and_atomic_path(tmp_path):
    recorder = TrainingRecorder(tmp_path, "ppo", 2, {}, targets=[9_000_000], hours=36)
    save = lambda path: path.write_text("model")
    recorder.tick(8_978_432, 137, save)
    assert not (tmp_path / "checkpoints.csv").exists()
    recorder.tick(9_043_968, 138, save)
    recorder.tick(9_043_968, 138, save)  # rl_games can call observer twice per epoch
    recorder.finish(save, True)
    rows = read_rows(tmp_path / "checkpoints.csv")
    assert len(rows) == 1 and rows[0]["requested_steps"] == "9000000"
    assert rows[0]["environment_steps"] == "9043968"
    assert Path(rows[0]["checkpoint"]).read_text() == "model"
    assert len(read_rows(tmp_path / "training.csv")) == 2
    assert not list((tmp_path / "checkpoints").glob("*.tmp.*"))
    assert json.loads((tmp_path / "metadata.json").read_text())["complete"]


def test_failed_update_does_not_publish_mislabelled_final_checkpoint(tmp_path):
    recorder = TrainingRecorder(tmp_path, "fast_sac", 0, {}, targets=[10])
    recorder.tick(8, 1, lambda path: path.write_text("valid"))
    recorder.finish(lambda path: pytest.fail("Partial update must not be saved"), False)
    assert not (tmp_path / "checkpoints.csv").exists()
    assert not json.loads((tmp_path / "metadata.json").read_text())["complete"]


def test_ppo_observer_stops_after_wall_budget(tmp_path, monkeypatch):
    recorder = TrainingRecorder(tmp_path, "ppo", 0, {}, targets=[], hours=1)
    observer = SimpleNamespace(after_init=lambda algo: None, after_print_stats=lambda *args: None)
    adapter = PPOExperimentObserver(observer, recorder)
    algo = SimpleNamespace(
        max_epochs=-1,
        game_rewards=SimpleNamespace(current_size=0),
        save=lambda stem: Path(stem + ".pth").write_text("ppo"),
    )
    adapter.after_init(algo)
    monkeypatch.setattr("training.time.monotonic", lambda: recorder.started + 3601)
    adapter.after_print_stats(64, 1, 3601)
    assert algo.max_epochs == 1
    adapter.finish(True)
    assert Path(read_rows(tmp_path / "checkpoints.csv")[0]["checkpoint"]).exists()


def test_conditioning_timing_random_seed_and_own_commands():
    actions = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 24
    original = actions.clone()
    legacy = conditioning_actions(actions, "exact", "legacy", 1)
    torch.testing.assert_close(legacy[1:, 2:], actions[:-1, 2:])
    assert torch.count_nonzero(legacy[0, 2:]) == 0
    zero = conditioning_actions(actions, "zero", "aligned", 1)
    assert torch.count_nonzero(zero[:, 2:]) == 0
    rng = torch.random.get_rng_state().clone()
    random = conditioning_actions(actions, "random", "aligned", 1)
    torch.testing.assert_close(random, conditioning_actions(actions, "random", "aligned", 1))
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(random[:, :2], actions[:, :2])
    torch.testing.assert_close(actions, original)
    assert torch.all(random[:, 2:].abs() <= 1)


def test_random_gameplay_uses_dedicated_rng_and_resets():
    class Agent:
        opponent_separation = True

        def get_initial_state(self, count):
            return {"prev_action": torch.zeros(count, 4)}

        def act(self, obs, state, eval):
            self.input = state["prev_action"].clone()
            return torch.zeros(2, 2), state

    model = Agent()
    policy = DreamerPolicy(model, 2, 0, "random", seed=4)
    rng = torch.random.get_rng_state().clone()
    obs = {"policy": torch.zeros(2, 20), "image": torch.zeros(2, 2, 2, 3)}
    policy.action(obs, torch.tensor([True, False]), torch.full((2, 4), 0.3))
    torch.testing.assert_close(model.input[0], torch.zeros(4))
    torch.testing.assert_close(model.input[1, :2], torch.full((2,), 0.3))
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


def test_trajectory_shards_keep_seat_frame_and_exclude_reset_image(tmp_path):
    recorder = TrajectoryRecorder(tmp_path, 1, 1, "id")
    actions = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    for pixel in (1, 2):
        obs = {"opponent_image": torch.full((1, 2, 2, 3), pixel, dtype=torch.uint8)}
        recorder.capture(obs, actions, torch.tensor([True]))
    recorder.finish(0, 0)
    rows = read_rows(tmp_path / "observations.csv")
    with np.load(tmp_path / rows[0]["shard"], allow_pickle=False) as data:
        assert data["image"][:, 0, 0, 0].tolist() == [1, 2]
        np.testing.assert_allclose(data["action"][0], [0.3, 0.4, 0.1, 0.2])
    assert len(rows) == 2
    assert recorder.buffers == [[]]


def test_open_loop_does_not_feed_target_embeddings_into_next_latent():
    class RSSM:
        def __init__(self):
            self.history = []

        def _deter_net(self, stoch, deter, action):
            self.history.append(stoch.clone())
            return deter + action[:, :1]

        def _img_net(self, deter):
            return torch.stack((deter, -deter), -1)

        def _obs_net(self, value):
            return torch.stack((value[:, 1:], -value[:, 1:]), -1)

        def get_dist(self, logits):
            return SimpleNamespace(rsample=lambda: logits.softmax(-1))

    class ImageDistribution:
        def log_prob(self, image):
            return -image.square().sum((2, 3, 4))

    rssm = RSSM()
    model = SimpleNamespace(rssm=rssm, decoder=lambda *args: {"image": ImageDistribution()})
    initial = (torch.zeros(1, 1, 2), torch.zeros(1, 1))
    actions = torch.ones(2, 4)
    images = torch.zeros(2, 2, 2, 3, dtype=torch.uint8)
    results = list(open_loop(model, initial, actions, torch.full((2, 1), 100.0), images))
    first = [value.clone() for value in rssm.history]
    rssm.history = []
    list(open_loop(model, initial, actions, torch.full((2, 1), -100.0), images))
    for before, after in zip(first, rssm.history):
        torch.testing.assert_close(before, after)
    assert [row["horizon"] for row in results] == [1, 2]
    assert all(row["kl_nats"] >= 0 for row in results)
    torch.testing.assert_close(raw_kl(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)), torch.zeros(1))


def test_diagnostic_bootstrap_uses_games_not_correlated_origins(tmp_path):
    rows = []
    for game, count, gap in (("long", 10, 1), ("short", 1, 3)):
        for origin in range(count):
            for condition in ("exact", "random", "zero"):
                value = 0 if condition == "exact" else gap
                rows.append(
                    {
                        "dataset_run_id": "r",
                        "job_id": "j",
                        "trajectory_id": game,
                        "origin": origin,
                        "timing": "aligned",
                        "condition": condition,
                        "draw": 0,
                        "horizon": 1,
                        "checkpoint": "c",
                        "domain": "id",
                        "opponent": "lineage",
                        "image_nll": value,
                        "image_nll_per_pixel": value,
                        "posterior_image_nll": value,
                        "kl_nats": value,
                    }
                )
    path = tmp_path / "conditioning.csv"
    write_csv(path, rows, list(rows[0]))
    _, gaps = conditioning_summary([path], 100, 42)
    assert all(row["independent_samples"] == 2 and row["mean"] == 2 for row in gaps)
    write_csv(path, rows[:-1], list(rows[0]))
    with pytest.raises(ValueError, match="Unpaired"):
        conditioning_summary([path], 100, 42)


def test_game_csv_reverses_seat_perspective(tmp_path):
    job = {
        "id": "leg1",
        "match": "m",
        "seat0": "PPO-B",
        "seat1": "D-RSSM",
        "condition": "zero",
        "seed": 1,
        "games": 3,
    }
    plan = {
        "run_id": "r",
        "jobs": [job],
        "matches": [{"id": "m", "a": "D-RSSM", "b": "PPO-B", "condition": "zero"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(plan))
    directory = tmp_path / "leg1"
    directory.mkdir()
    path = directory / "games.npz"
    np.savez(
        path,
        num_games=3,
        outcome_schema_version=2,
        outcome=[0, 1, 4],
        env_id=[0, 1, 2],
        episode_id=[0, 0, 0],
        termination_names=["goal_scored", "goal_conceded", "time_out"],
        termination_flags=np.eye(3, dtype=bool),
    )
    export_games(path, job, "r")
    rows = read_rows(path.with_suffix(".csv"))
    assert [float(row["payoff_seat0"]) for row in rows] == [1, 0, 0.5]
    stats = games_summary(tmp_path, 100, 42)[0]
    assert (stats["W"], stats["L"], stats["D"], stats["score"]) == (1, 1, 1, 0.5)
    assert stats["complete"]


def test_learning_curve_resamples_seed_scores():
    rows = [
        {
            "algorithm": "ppo",
            "complete": True,
            "requested_steps": "9000000",
            "opponent": "fixed",
            "training_seed": seed,
            "score": score,
            "environment_steps": 9043968,
        }
        for seed, score in enumerate([0.2, 0.5, 0.8])
    ]
    curve = learning_curve_summary(rows, 100, 42)[0]
    assert curve["seeds"] == 3 and curve["complete_three_seeds"]
    assert curve["mean"] == 0.5 and curve["score_std"] == pytest.approx(0.3)


def test_gameplay_changes_only_focal_condition():
    result = experiment_config(
        {"agents": {}},
        "gameplay",
        "chosen.pt",
        "chosen.yaml",
        "unverified proxy",
        "aligned",
    )
    assert [match["condition"] for match in result["matches"]] == [
        "exact",
        "random",
        "zero",
    ]
    assert all(match["agent_conditions"]["Dreamer-lineage"] == "zero" for match in result["matches"])
    assert all("trajectory_domain" not in match for match in result["matches"])
    assert result["experiment"]["id_provenance"] == "unverified proxy"


def test_ppo_raw_score_counts_only_actual_games_and_resolves_overlap():
    sys.path.insert(0, str(SCRIPTS / "rl_games"))
    from klask_rl_games import KlaskRlAlgoObserver

    observer = KlaskRlAlgoObserver(raw_outcomes=True)
    manager = SimpleNamespace(
        _term_names=["goal_scored", "goal_conceded", "time_out"],
        _term_dones=torch.tensor(
            [
                [False, False, False],
                [True, False, True],
                [False, True, False],
                [False, False, True],
            ]
        ),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(termination_manager=manager))
    observer.algo = SimpleNamespace(vec_env=SimpleNamespace(env=env), ppo_device="cpu", num_actors=4)
    collected = []
    observer.mean_scores = SimpleNamespace(update=lambda scores: collected.extend(scores.tolist()))
    observer.ep_infos = []
    observer.process_infos(
        {"episode": {"Episode_Termination/goal_scored": 0.0}},
        torch.empty(0, dtype=torch.long),
    )
    assert collected == []
    observer.process_infos({"episode": {"Episode_Termination/goal_scored": 0.0}}, torch.tensor([1, 2, 3]))
    assert collected == [1, -1, 0]
