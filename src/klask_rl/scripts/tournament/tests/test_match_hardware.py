"""CPU tests of the match hardware setting (match_hardware.py): v1 unchanged, v2 as in training."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS / "tournament"))

from match_hardware import (  # noqa: E402
    HARDWARE_KEYS,
    PRESETS,
    agent_hardware_notes,
    apply_hardware,
    select_hardware,
    training_hardware,
)
from report import hardware_line  # noqa: E402
from run_tournament import build_plan  # noqa: E402

DREAMER = SCRIPTS / "dreamer"


def _training_block(name):
    """Hardware block of a training config in scripts/dreamer (hardware keys are literal values there)."""
    return training_hardware(OmegaConf.load(DREAMER / name))[0]


@pytest.mark.parametrize(
    "preset, training_configs",
    [
        ("v1", ["config-sprite-v3-state-1m-eff.yaml", "config-sprite-r2-pfsp.yaml"]),
        (
            "v2",
            ["config-sprite-v3-state-1m-v2sim-eff.yaml"],
        ),
    ],
)
def test_presets_equal_the_training_configs(preset, training_configs):
    for name in training_configs:
        assert _training_block(name) == PRESETS[preset], name


def test_v2_preset_values():
    v2 = PRESETS["v2"]
    assert (v2["max_velocity"], v2["max_acceleration"]) == (1.4, 35.0)
    assert v2["actuator_model"]["enable"] is False
    assert v2["actuator_lag"] == {"enable": True, "dead_time_s": 0.005, "tau_s": 0.01}
    assert v2["collision_avoidance"] == {"enable": True}


def test_apply_replaces_and_removes_keys():
    v1_env = {"task": "t", "sim_dt": 0.001, "decimation": 20, **copy.deepcopy(PRESETS["v1"])}
    env = apply_hardware(copy.deepcopy(v1_env), PRESETS["v2"])
    assert {k: env[k] for k in HARDWARE_KEYS} == PRESETS["v2"]
    assert (env["task"], env["sim_dt"], env["decimation"]) == ("t", 0.001, 20)
    back = apply_hardware(copy.deepcopy(env), PRESETS["v1"])
    assert back == v1_env and "actuator_lag" not in back  # the v1 table has no lag entry at all


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="Unknown hardware preset"):
        select_hardware("v3", None)
    with pytest.raises(ValueError, match="preset name"):
        select_hardware({"agent": "x"}, None)
    both = {"env": {"max_velocity": 1.0, "actuator_model": {"enable": True, "checkpoint": "m.pt"}}}
    both["env"]["actuator_lag"] = {"enable": True, "dead_time_s": 0.0, "tau_s": 0.0}
    with pytest.raises(ValueError, match="alternatives"):
        select_hardware({"from_agent": "A"}, lambda name: both)


def test_notes_flag_cross_hardware_agents_only():
    env = {"sim_dt": 0.001, "decimation": 20, **copy.deepcopy(PRESETS["v2"])}
    v2_agent = {"env": {"sim_dt": 0.001, "decimation": 20, **copy.deepcopy(PRESETS["v2"])}}
    v1_agent = {"env": {"sim_dt": 0.001, "decimation": 20, **copy.deepcopy(PRESETS["v1"])}}
    # A disabled wrapper with other parameters equals an absent one.
    v2_variant = copy.deepcopy(v2_agent)
    v2_variant["env"]["actuator_model"] = {"enable": False, "checkpoint": "elsewhere.pt"}
    notes = agent_hardware_notes(env, {"v2": v2_agent, "v1": v1_agent, "v2b": v2_variant})
    assert notes["v2"]["differs_from_match"] == [] and notes["v2b"]["differs_from_match"] == []
    assert notes["v1"]["differs_from_match"] == ["max_velocity", "max_acceleration", "actuator_model", "actuator_lag"]


# ---- build_plan on a synthetic project (no checkpoints: blockers, but the plan is inspectable) ----


def _provider_config(hardware):
    """A composed D-RSSM sprite training config (only what build_plan reads)."""
    env = {
        "task": "isaaclab_Klask-Rl-Dreamer-Sprite-v0",
        "device": "${device}",
        "seed": "${seed}",
        "sim_dt": 0.001,
        "decimation": 20,
        "episode_length_s": 10.0,
        "ball_reset_position_x": [-0.15, 0.15],
        "ball_reset_position_y": [-0.1, 0.1],
        "size": [128, 128],
        **copy.deepcopy(hardware),
    }
    return {
        "device": "cuda:0",
        "seed": 0,
        "model": {"compile": True, "device": "${device}"},
        "env": env,
        "opponent_separation": {"enabled": True, "action_timing": "aligned"},
    }


def _state_agent_config(hardware):
    cfg = _provider_config(hardware)
    cfg["env"].update(state_only=True, state_units=True, ball_reset_min_abs_y=0.03)
    return cfg


def _project(tmp_path, tag, hardware=None, extra=None):
    """Provider (v1), a v2 and a v1 state agent and a PPO entry under one project root; return build_plan args."""
    root = tmp_path / "project"
    for name, cfg in (
        ("drssm", _provider_config(PRESETS["v1"])),
        ("state", _state_agent_config(PRESETS["v2"])),
        ("state_v1", _state_agent_config(PRESETS["v1"])),
    ):
        (root / "logs" / name / ".hydra").mkdir(parents=True, exist_ok=True)
        (root / "logs" / name / ".hydra/config.yaml").write_text(yaml.safe_dump(cfg))
    tournament = {
        "games": 4,
        "num_envs": 2,
        "episode_length_s": 30.0,
        "seed": 0,
        "bootstrap_samples": 10,
        "bootstrap_seed": 1,
        "agents": {
            "D-RSSM": {
                "kind": "dreamer",
                "checkpoint": "logs/drssm/ckpt.pt",
                "config": "logs/drssm/.hydra/config.yaml",
            },
            "State": {"kind": "dreamer", "checkpoint": "logs/state/ckpt.pt", "config": "logs/state/.hydra/config.yaml"},
            "State-v1": {
                "kind": "dreamer",
                "checkpoint": "logs/state_v1/ckpt.pt",
                "config": "logs/state_v1/.hydra/config.yaml",
            },
            "PPO-B": {"kind": "ppo", "checkpoint": "logs/ppo/model.pth", "config": "logs/ppo/agent.yaml"},
        },
        "matches": [{"id": "state_vs_ppob", "a": "State", "b": "PPO-B", "condition": "exact", "primary": False}],
        **(extra or {}),
    }
    if hardware is not None:
        tournament["hardware"] = hardware
    config = tmp_path / f"tournament_{tag}.yaml"
    config.write_text(yaml.safe_dump(tournament))
    return SimpleNamespace(
        project_root=root,
        config=config,
        games=None,
        num_envs=None,
        seed=None,
        episode_length_s=None,
        device="cuda:0",
        actuator_checkpoint=None,
    )


@pytest.fixture
def plan_of(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    counter = iter(range(100))

    def plan(**kwargs):
        return build_plan(_project(tmp_path, next(counter), **kwargs))

    return plan


def _env(plan):
    return plan["evaluation_config"]["env"]


def test_no_setting_keeps_the_provider_hardware_and_manifest_layout(plan_of):
    plan, blockers = plan_of()
    assert any("Missing" in b for b in blockers)  # no checkpoints in the synthetic project
    env = _env(plan)
    assert {k: env[k] for k in ("max_velocity", "max_acceleration", "collision_avoidance")} == {
        "max_velocity": 0.4,
        "max_acceleration": 10.0,
        "collision_avoidance": {"enable": True},
    }
    assert env["actuator_model"]["enable"] is True and "actuator_lag" not in env
    assert "hardware" not in plan
    # hardware: v1 on a v1 provider resolves to the same environment (only the manifest records the setting).
    explicit, _ = plan_of(hardware="v1")
    assert _env(explicit) == env
    assert explicit["hardware"]["setting"] == "v1" and explicit["hardware"]["mismatched_agents"] == ["State"]


@pytest.mark.parametrize("setting", ["v2", {"from_agent": "State"}])
def test_v2_match_env_equals_the_v2_training_block(plan_of, setting):
    plan, blockers = plan_of(hardware=setting)
    env = _env(plan)
    trained = training_hardware(OmegaConf.load(DREAMER / "config-sprite-v3-state-1m-v2sim-eff.yaml"))[0]
    assert {k: env[k] for k in HARDWARE_KEYS} == trained
    # The v1 actuator checkpoint is neither resolved nor required on v2.
    assert not any("actuator checkpoint" in b for b in blockers)
    assert "sha256" not in env["actuator_model"]
    # Everything else still comes from the provider.
    assert (env["sim_dt"], env["decimation"], env["task"]) == (0.001, 20, "isaaclab_Klask-Rl-Dreamer-Sprite-v0")
    assert env["episode_length_s"] == 30.0 and env["terminations"]["goal_scored"] is True
    hw = plan["hardware"]
    assert hw["block"] == trained and hw["timing"] == {"sim_dt": 0.001, "decimation": 20}
    assert hw["mismatched_agents"] == [] and set(hw["agents"]) == {"State"}  # PPO-B has no saved hardware
    line = hardware_line(hw)
    assert "max_velocity 1.4 m/s" in line and "max_acceleration 35.0" in line and "tau 0.01 s" in line


def test_worker_env_config_carries_the_v2_block_on_every_job(plan_of):
    """configuration_for_job builds the cfg.env that the worker hands to _make_eval_env (both seats, both legs)."""
    from worker import configuration_for_job

    plan, _ = plan_of(hardware="v2")
    assert len(plan["jobs"]) == 2
    for job in plan["jobs"]:
        env = OmegaConf.to_container(configuration_for_job(plan, job).env)
        assert {k: env[k] for k in HARDWARE_KEYS} == PRESETS["v2"]
        assert env["device"] == job["device"] and env["seed"] == job["seed"]


def test_v2_and_from_agent_give_the_same_plan_env(plan_of):
    preset, _ = plan_of(hardware="v2")
    copied, _ = plan_of(hardware={"from_agent": "State"})
    assert _env(preset) == _env(copied)


def test_cross_hardware_is_recorded_not_refused(plan_of):
    extra = {"matches": [{"id": "x", "a": "State", "b": "State-v1", "condition": "exact", "primary": False}]}
    plan, _ = plan_of(hardware="v2", extra=extra)
    assert plan["hardware"]["mismatched_agents"] == ["State-v1"]
    assert "Trained on other hardware: State-v1" in hardware_line(plan["hardware"])


def test_from_agent_must_be_a_dreamer_entry(plan_of):
    with pytest.raises(ValueError, match="from_agent"):
        plan_of(hardware={"from_agent": "PPO-B"})
    with pytest.raises(ValueError, match="from_agent"):
        plan_of(hardware={"from_agent": "Nobody"})


def test_from_agent_outside_the_matches_is_hashed_into_the_plan(plan_of):
    plan, _ = plan_of(hardware={"from_agent": "State-v1"})
    assert "State-v1" in plan["agents"] and plan["agents"]["State-v1"]["config_sha256"]
    assert _env(plan)["max_velocity"] == 0.4 and _env(plan)["actuator_model"]["enable"] is True


def test_manifest_is_json_serialisable(plan_of):
    plan, _ = plan_of(hardware="v2")
    json.dumps(plan)
