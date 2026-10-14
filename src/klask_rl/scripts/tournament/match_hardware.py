"""Match hardware: the actuation physics shared by both board seats (no simulator imports; CPU-testable).

The match environment is built from the saved training config of the ``D-RSSM`` entry (the environment-config
provider). Its hardware block is used unchanged unless the tournament config sets a top-level ``hardware`` key:

    (absent)                      the provider's own block, unchanged (v1 for every config written before 2026-10-13)
    hardware: v1                  the v1 table (preset V1 below)
    hardware: v2                  the v2-like table (preset V2 below)
    hardware: {from_agent: NAME}  copied from that Dreamer entry's saved training config (.hydra/config.yaml)

The block replaces these ``env`` keys for both seats; a key the source lacks is removed, exactly as training
treats a missing key (``getattr(config, key, None)``):

    max_velocity          [m/s] per axis: the [-1, 1] command scale (VelocityScaleWrapper) and the collision
                          avoidance's deceleration ramp (KlaskRlCollisionAvoidanceWrapper)
    max_acceleration      [m/s^2] command rate cap (VelocityScaleWrapper)
    actuator_model        learned v1 actuator model (ActuatorModelWrapper); only on the v1 table
    actuator_lag          parametric dead time + first-order lag (ActuatorLagWrapper); the v2 stand-in
    collision_avoidance   boundary clipping of the commands (on for both tables)

Everything else (sim_dt, decimation, observations, rewards, ...) still comes from the provider. sim_dt and
decimation set the control period that the acceleration cap and the lag coefficients use, so they are checked
against each Dreamer agent's training config together with the block (``agent_hardware_notes``).

Each seat on its own agent's hardware (2026-10-14):

    hardware:
      per_agent:
        NAME_A: v2                    a preset, or
        PPO-B: {from_agent: PPO-B}    that Dreamer/PPO entry's saved training config (PPO: agent.yaml's env block)

Every agent that plays a selected match needs an entry; names that are not agents of the config are refused.
When all players resolve to the same block the match env is exactly that of ``hardware: <block>`` (both seats
share it). Otherwise the plan env carries no shared hardware keys but ``per_agent_hardware`` {name: block}, and
each job resolves its seats (``seat_environment``): a job whose two seats carry the same block gets the shared
keys again, any other job gets ``seat_hardware`` [seat-0 block, seat-1 block] (each with its ``agent``), which
evaluation._seat_hardware_chain builds with every stage acting on its seat's action columns only (seat 0 =
columns 0:2 = Peg_1, seat 1 = columns 2:4 = Peg_2). The block follows the agent when the seats swap between legs.

The observations do not depend on the hardware: the env's observation groups are raw simulator state (metres,
m/s, no max_velocity terms); a state agent trained with env.state_units divides by units derived from its OWN
training max_velocity (worker.py). The opponent command fed to a Dreamer in the exact condition is the opponent's
raw [-1, 1] action, i.e. in the opponent's command scale (see README: cross-hardware caveat).
"""

import copy

HARDWARE_KEYS = ("max_velocity", "max_acceleration", "actuator_model", "actuator_lag", "collision_avoidance")
TIMING_KEYS = ("sim_dt", "decimation")

# The v1 table's learned actuator model (the same file every v1 training config names).
V1_ACTUATOR_CHECKPOINT = (
    "/workspace/klask_rl/logs/actuator_model/data/new/checkpoints/"
    "model_data_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states_train_seed0.pt"
)

PRESETS = {
    # scripts/dreamer/config-sprite-v3-state-1m-eff.yaml (and every v1 run: the pilot, D-RSSM, 1m_s80_eff, ...).
    "v1": {
        "max_velocity": 0.4,
        "max_acceleration": 10.0,
        "actuator_model": {"enable": True, "checkpoint": V1_ACTUATOR_CHECKPOINT},
        "collision_avoidance": {"enable": True},
    },
    # scripts/dreamer/config-sprite-v3-state-1m-v2sim-eff.yaml (runs v2_1m_s80_eff, v2_1m_s80_eff_shared), as
    # config-sprite-r2-v2sim.yaml and config-sprite-v3-state-v2sim(-eff).yaml: 1.4 m/s per axis, 35 m/s^2 (the v2
    # magnet disconnects at 40), the learned v1 model off, actuator_lag 5 ms dead time + 10 ms first-order lag.
    "v2": {
        "max_velocity": 1.4,
        "max_acceleration": 35.0,
        "actuator_model": {"enable": False, "checkpoint": V1_ACTUATOR_CHECKPOINT},
        "actuator_lag": {"enable": True, "dead_time_s": 0.005, "tau_s": 0.01},
        "collision_avoidance": {"enable": True},
    },
}


def _plain(value):
    """OmegaConf node -> resolved plain Python value (dicts/lists/scalars pass through)."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else copy.deepcopy(value)


def training_hardware(cfg):
    """The hardware block (and control timing) of a saved, composed training config (OmegaConf or dict)."""
    env = cfg["env"] if "env" in cfg else {}
    block = {key: _plain(env[key]) for key in HARDWARE_KEYS if key in env and env[key] is not None}
    timing = {key: _plain(env[key]) for key in TIMING_KEYS if key in env and env[key] is not None}
    return block, timing


def validate_block(block, label):
    if block.get("max_velocity") is None:
        raise ValueError(f"Hardware {label}: max_velocity is required")
    model = bool((block.get("actuator_model") or {}).get("enable", False))
    lag = bool((block.get("actuator_lag") or {}).get("enable", False))
    if model and lag:
        # train_dreamer.py refuses this combination; _make_eval_env would silently prefer the model.
        raise ValueError(f"Hardware {label}: actuator_model and actuator_lag are alternatives; enable only one")
    if model and not (block["actuator_model"] or {}).get("checkpoint"):
        raise ValueError(f"Hardware {label}: actuator_model.enable needs a checkpoint")
    if lag and not {"dead_time_s", "tau_s"} <= set(block["actuator_lag"]):
        raise ValueError(f"Hardware {label}: actuator_lag needs dead_time_s and tau_s")


def is_per_agent(setting):
    return isinstance(setting, dict) and "per_agent" in setting


def check_per_agent(setting, agent_specs, players):
    """Validate ``{per_agent: {NAME: preset | {from_agent: NAME}}}`` against the config's agents (``agent_specs``)
    and the playing agents; return the ``from_agent`` sources of the players' entries (hashed like the players')."""
    entries = setting.get("per_agent")
    if set(setting) != {"per_agent"} or not isinstance(entries, dict) or not entries:
        raise ValueError("hardware.per_agent must map agent names to a preset (v1, v2) or {from_agent: NAME}")
    unknown = sorted(set(entries) - set(agent_specs))
    if unknown:
        raise ValueError(f"hardware.per_agent names agents that are not in the config: {unknown}")
    missing = sorted(set(players) - set(entries))
    if missing:
        raise ValueError(f"hardware.per_agent has no entry for the playing agent(s) {missing}")
    sources = set()
    for name, value in entries.items():
        if isinstance(value, dict) and set(value) == {"from_agent"}:
            source = value["from_agent"]
            if source not in agent_specs or agent_specs[source].get("kind") not in ("dreamer", "ppo"):
                raise ValueError(
                    f"hardware.per_agent.{name}.from_agent must name a Dreamer or PPO agent of the config, "
                    f"got {source!r}"
                )
            if name in players:
                sources.add(source)
        elif not isinstance(value, str) or value not in PRESETS:
            raise ValueError(
                f"hardware.per_agent.{name} must be a preset {sorted(PRESETS)} or {{from_agent: NAME}}, got {value!r}"
            )
    return sources


def select_hardware(setting, load_agent_config):
    """Resolve the tournament's ``hardware`` setting to (label, block); None when it is absent.

    ``load_agent_config(name)`` returns that agent's saved training config (only called for ``from_agent``).
    """
    if setting is None:
        return None
    if isinstance(setting, str):
        if setting not in PRESETS:
            raise ValueError(f"Unknown hardware preset {setting!r}; use {sorted(PRESETS)} or {{from_agent: NAME}}")
        label, block = setting, copy.deepcopy(PRESETS[setting])
    elif isinstance(setting, dict) and set(setting) == {"from_agent"}:
        name = setting["from_agent"]
        block, _ = training_hardware(load_agent_config(name))
        label = f"from_agent {name}"
    else:
        raise ValueError("hardware must be a preset name (v1, v2) or {from_agent: NAME}")
    validate_block(block, label)
    return label, block


def apply_hardware(env, block):
    """Replace the hardware keys of a resolved ``env`` dict in place (both seats share one environment)."""
    for key in HARDWARE_KEYS:
        if key in block:
            env[key] = copy.deepcopy(block[key])
        else:
            env.pop(key, None)
    return env


def comparable(block):
    """The block as the environment builders read it: disabled actuator entries carry no parameters."""
    out = {}
    for key, value in block.items():
        if isinstance(value, dict) and "enable" in value and not value["enable"]:
            continue  # a disabled wrapper (any parameters) == an absent one
        out[key] = value
    return out


def plan_hardware(setting, env, agents, blockers, players):
    """Apply the tournament's ``hardware`` setting to the resolved match ``env`` (in place) for build_plan.

    ``agents`` are build_plan's resolved agent entries (config paths already resolved and checked); ``players``
    the names that play a selected match (the training hardware of their Dreamer entries is recorded). Returns
    the manifest record, or None when a missing source config blocks the plan (the blocker is appended).
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    def load(name):
        return OmegaConf.load(agents[name]["config"])

    if is_per_agent(setting):
        return plan_per_agent_hardware(setting, env, agents, blockers, players)
    if isinstance(setting, dict) and "from_agent" in setting:
        source = setting["from_agent"]
        if not Path(agents[source]["config"]).is_file():
            blockers.append(f"Missing hardware source config of {source}: {agents[source]['config']}")
            return None
    label, block = select_hardware(setting, load)
    apply_hardware(env, block)
    configs = {
        name: load(name)
        for name, agent in agents.items()
        if name in players and agent["kind"] == "dreamer" and Path(agent["config"]).is_file()
    }
    notes = agent_hardware_notes(env, configs)
    return {
        "setting": label,
        "block": block,
        "timing": training_hardware({"env": env})[1],
        "agents": notes,
        "mismatched_agents": sorted(name for name, note in notes.items() if note["differs_from_match"]),
    }


def agent_hardware_notes(env, agent_configs):
    """Per Dreamer agent: its training hardware/timing and whether the match environment differs from it.

    ``env`` is the resolved match env dict; ``agent_configs`` maps name -> saved training config. Mismatches are
    reported, not refused: cross-hardware matches (a v1 agent on the v2 table) are legitimate experiments.
    """
    match_block, match_timing = training_hardware({"env": env})
    return {name: hardware_note(match_block, match_timing, cfg) for name, cfg in agent_configs.items()}


def hardware_note(match_block, match_timing, cfg):
    """One agent's training hardware/timing (saved config ``cfg``) and the keys in which ``match_block`` differs."""
    block, timing = training_hardware(cfg)
    differs = [key for key in HARDWARE_KEYS if comparable(block).get(key) != comparable(match_block).get(key)]
    differs += [key for key in TIMING_KEYS if timing.get(key) != match_timing.get(key)]
    return {"training_hardware": block, "training_timing": timing, "differs_from_match": differs}


def plan_per_agent_hardware(setting, env, agents, blockers, players):
    """``hardware: {per_agent: ...}`` for build_plan (see the module docstring); returns the manifest record.

    The record names each player's setting and block; ``agents`` holds, for every player with a saved training
    config (Dreamer .hydra/config.yaml, PPO agent.yaml env block), its training hardware and the keys in which ITS
    OWN seat block differs from it.
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    def load(name):
        return OmegaConf.load(agents[name]["config"])

    entries = setting["per_agent"]
    resolved = {}
    for name in sorted(players):
        value = entries[name]
        if isinstance(value, dict):
            source = value["from_agent"]
            if not Path(agents[source]["config"]).is_file():
                blockers.append(f"Missing hardware source config of {source} (for {name}): {agents[source]['config']}")
                return None
        resolved[name] = select_hardware(value, load)
    blocks = [block for _, block in resolved.values()]
    shared = all(block == blocks[0] for block in blocks)
    if shared:
        apply_hardware(env, blocks[0])  # exactly the env of `hardware: <that block>`
    else:
        apply_hardware(env, {})  # no shared hardware keys: every job resolves its own seats
        env["per_agent_hardware"] = {name: copy.deepcopy(block) for name, (_, block) in resolved.items()}
    timing = training_hardware({"env": env})[1]
    notes = {}
    for name, (_, block) in resolved.items():
        agent = agents[name]
        if agent["kind"] in ("dreamer", "ppo") and Path(agent.get("config", "")).is_file():
            cfg = load(name)
            if "env" in cfg and cfg["env"] is not None and "max_velocity" in cfg["env"]:
                notes[name] = hardware_note(block, timing, cfg)
    return {
        "setting": "per_agent",
        "per_agent": {name: {"setting": label, "block": block} for name, (label, block) in resolved.items()},
        "seats_share_block": shared,
        "block": copy.deepcopy(blocks[0]) if shared else None,
        "timing": timing,
        "agents": notes,
        "mismatched_agents": sorted(name for name, note in notes.items() if note["differs_from_match"]),
    }


def seat_environment(env, seat_agents):
    """Resolve a per-agent plan env (dict, in place) for one job whose seats 0 and 1 are ``seat_agents``.

    No-op for any other plan. Same block on both seats: the shared keys (the env of ``hardware: <block>``);
    otherwise ``seat_hardware`` = [seat-0 block, seat-1 block], each with its ``agent`` name.
    """
    per_agent = env.pop("per_agent_hardware", None)
    if per_agent is None:
        return env
    blocks = [per_agent[name] for name in seat_agents]
    if blocks[0] == blocks[1]:
        apply_hardware(env, blocks[0])
    else:
        env["seat_hardware"] = [{"agent": name, **copy.deepcopy(block)} for name, block in zip(seat_agents, blocks)]
    return env
