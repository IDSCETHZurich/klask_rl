"""Shared evaluation builders. Import only after Isaac Lab AppLauncher has started."""

import importlib
import pathlib
import sys

import gymnasium as gym
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent)
_PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
for _candidate in (
    pathlib.Path(_SCRIPT_DIR) / "r2dreamer",
    _PROJECT_DIR.parent.parent / "third_party" / "r2dreamer",
):
    if _candidate.is_dir():
        sys.path.insert(0, str(_candidate))
        break
OmegaConf.register_new_resolver("script_dir", lambda: _SCRIPT_DIR, use_cache=True, replace=True)

import isaaclab_tasks  # noqa: F401
import klask_rl.tasks  # noqa: F401
from checkpoint_loaders import (  # noqa: F401
    _init_ppo_opponent_batch,
    _load_dreamer_agent,
    _load_fast_sac_opponent,
    _load_ppo_opponent,
)
from dreamer_self_play import DreamerSelfPlayWrapper
from env_cfg_utils import apply_camera_size_to_env_cfg
from envs.isaaclab import IsaacLabVecEnv
from isaaclab.sim import RenderCfg
from klask_rl.assets.robots.klask_params import KLASK_PARAMS
from klask_rl.tasks.manager_based.klask_rl.actuator_model import (
    ActuatorModelWrapper,
)
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    KlaskRlAgentOpponentWrapper,
    KlaskRlCollisionAvoidanceWrapper,
    KlaskRlFastSACOpponentWrapper,
    ObservationNoiseWrapper,
    OpponentActionWrapper,
    VelocityScaleWrapper,
    configure_domain_randomization,
)


def _load_config(config_path, device):
    """Load a Hydra training config YAML using Hydra's compose API.

    This mirrors train_dreamer.py's @hydra.main approach: it composes defaults
    (e.g. model: size50M), resolves ${now:...} and other Hydra resolvers, and
    applies the device override — all exactly as training does.
    """
    config_path = pathlib.Path(config_path).resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        cfg = compose(config_name=config_path.stem, overrides=[f"device={device}"])
        OmegaConf.resolve(cfg)
    return cfg


def _make_eval_env(
    env_config,
    num_envs,
    episode_length_s=None,
    opponent_type="dreamer",
    opponent_action_method=None,
    obs_noise_stds=None,
    *,
    simulation_app=None,
    tournament=False,
):
    """Create an evaluation env with the appropriate opponent wrapper.

    Simplified wrapper chain (no curriculum/logging, innermost → outermost):
      0. KlaskRlCollisionAvoidanceWrapper (if config.collision_avoidance.enable —
         innermost; clips world-frame m/s actions near board boundaries)
      1. OpponentActionWrapper — negate opponent actions for coordinate frame
      2. ActuatorModelWrapper (if config.actuator_model.enable — expects m/s)
      3. VelocityScaleWrapper — scales [-1, 1] → m/s; action manager _scale = 1.0
      4. ObservationNoiseWrapper (when any std > 0; noises both 'policy' and
         'opponent' obs keys with independent samples, recomputes extended dims)
      5. Opponent wrapper (emits opponent action in [-1, 1]):
         DreamerSelfPlayWrapper or KlaskRlAgentOpponentWrapper
      6. IsaacLabVecEnv — r2dreamer adapter

    ``env_config.seat_hardware`` (tournament, one hardware block per seat; match_hardware.py) replaces stages
    0-3 by _seat_hardware_chain, where each stage acts on its seat's action columns only.

    ``obs_noise_stds`` is an optional dict with keys ``peg_position_std``,
    ``peg_velocity_std``, ``ball_position_std``, ``ball_velocity_std`` (all in
    m and m/s). Wrapper is skipped when ``None`` or when all stds are zero.

    Returns (vec_env, opponent_wrapper).
    """
    _, task_name = env_config.task.split("_", 1)

    env_cfg_entry = gym.spec(task_name).kwargs["env_cfg_entry_point"]
    if isinstance(env_cfg_entry, str):
        module_name, class_name = env_cfg_entry.rsplit(":", 1)
        env_cfg_class = getattr(importlib.import_module(module_name), class_name)
    else:
        env_cfg_class = env_cfg_entry

    env_cfg = env_cfg_class()
    env_cfg.sim.device = str(env_config.device)

    # When a FastSAC opponent is requested, attach its 18-dim observation
    # group to the env so the env emits obs["fast_sac_opponent"] in the exact
    # layout the SAC actor was trained on. Uses the same ObsTerm functions as
    # FastSACObservationsCfg, so scaling / 180° rotation / peg role-swap match
    # training exactly.
    if opponent_type == "fast_sac":
        from klask_rl.tasks.manager_based.klask_rl.env_cfg.klask_rl_observations_cfg import (
            _FastSACOpponentPolicyCfg,
        )

        env_cfg.observations.fast_sac_opponent = _FastSACOpponentPolicyCfg()

    if tournament:
        from klask_rl.tasks.manager_based.klask_rl.env_cfg.klask_rl_observations_cfg import (
            _FastSACOpponentPolicyCfg,
            _FastSACPlayerPolicyCfg,
        )

        env_cfg.observations.fast_sac_player = _FastSACPlayerPolicyCfg()
        env_cfg.observations.fast_sac_opponent = _FastSACOpponentPolicyCfg()

    # Mirror train_dreamer.py: apply ball reset position override from YAML.
    # Without this, the env uses KLASK_PARAMS defaults which put ~90% of resets
    # in the player's half, causing a strong structural win-rate asymmetry.
    ball_reset_x = getattr(env_config, "ball_reset_position_x", None)
    ball_reset_y = getattr(env_config, "ball_reset_position_y", None)
    if ball_reset_x is not None and hasattr(env_cfg, "ball_reset_position_x"):
        env_cfg.ball_reset_position_x = tuple(ball_reset_x)
        env_cfg.ball_reset_position_y = tuple(ball_reset_y)
        env_cfg.events.reset_ball_position.params["pose_range"]["x"] = env_cfg.ball_reset_position_x
        env_cfg.events.reset_ball_position.params["pose_range"]["y"] = env_cfg.ball_reset_position_y

    sim_dt = getattr(env_config, "sim_dt", None)
    if sim_dt is not None:
        env_cfg.sim.dt = float(sim_dt)

    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.decimation = int(env_config.decimation)
    env_cfg.seed = int(env_config.seed)
    env_cfg.episode_length_s = float(episode_length_s or env_config.episode_length_s)
    env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")

    # --- Camera resolution & padding derived from env.size ---
    apply_camera_size_to_env_cfg(env_cfg, getattr(env_config, "size", None))

    # Reachable ball starts (2026-10-07): never inside the band around the midline that neither peg can reach.
    ball_min_abs_y = getattr(env_config, "ball_reset_min_abs_y", None)
    if ball_min_abs_y is not None and hasattr(env_cfg.events, "reset_ball_position"):
        from klask_rl.tasks.manager_based.klask_rl.utils_manager_based import reset_ball_outside_midline_band

        env_cfg.events.reset_ball_position.func = reset_ball_outside_midline_band
        env_cfg.events.reset_ball_position.params["min_abs_y"] = float(ball_min_abs_y)

    # Null out disabled termination terms.
    terminations_cfg = getattr(env_config, "terminations", None)
    if terminations_cfg is not None and hasattr(env_cfg, "terminations"):
        term_dict = (
            OmegaConf.to_container(terminations_cfg, resolve=True)
            if OmegaConf.is_config(terminations_cfg)
            else dict(terminations_cfg)
        )
        for term, active in term_dict.items():
            if not active and hasattr(env_cfg.terminations, term):
                setattr(env_cfg.terminations, term, None)

    # Configure domain randomization events from YAML BEFORE env construction.
    _dr_cfg_raw = getattr(env_config, "domain_randomization", None)
    _dr_dict = (
        OmegaConf.to_container(_dr_cfg_raw, resolve=True)
        if _dr_cfg_raw is not None and OmegaConf.is_config(_dr_cfg_raw)
        else _dr_cfg_raw
    )
    configure_domain_randomization(env_cfg, _dr_dict)

    # --- Create base env ---
    isaac_env = gym.make(task_name, cfg=env_cfg)

    # Tournament matches with each seat on its own agent's hardware (match_hardware.py, per_agent) carry one
    # block per seat in env.seat_hardware instead of the shared keys; _seat_hardware_chain builds that chain.
    seat_hardware = getattr(env_config, "seat_hardware", None)

    # max_velocity is needed by both KlaskRlCollisionAvoidanceWrapper (below)
    # and VelocityScaleWrapper (further down), so look it up once up front.
    max_velocity = getattr(env_config, "max_velocity", None)
    if max_velocity is None and seat_hardware is None:
        raise ValueError("env_config.max_velocity is required; VelocityScaleWrapper needs it to scale [-1, 1] → m/s.")

    # --- Action manager scale = 1.0 (pass-through, m/s in → m/s out) ---
    # Velocity scaling is done by VelocityScaleWrapper below so that the
    # actuator model sees commands in m/s (matching its training units).
    action_mgr = isaac_env.unwrapped.action_manager
    for term in action_mgr._terms.values():
        term._scale = 1.0

    if seat_hardware is not None:
        isaac_env = _seat_hardware_chain(isaac_env, env_config, seat_hardware, env_cfg.sim.device)
    else:
        # --- 0. Collision avoidance (world-frame m/s clipping; innermost gym wrapper) ---
        # Mirrors train_dreamer.py: sits inside OpponentActionWrapper so it sees
        # world-frame actions (after opponent dim negation) for both player and opponent.
        ca_cfg = getattr(env_config, "collision_avoidance", None)
        if ca_cfg is not None and getattr(ca_cfg, "enable", False):
            isaac_env = KlaskRlCollisionAvoidanceWrapper(isaac_env, max_vel=float(max_velocity))

        # InitializationWrapper is intentionally skipped at inference. Its _step
        # counter would start at 0 and select the EARLY curriculum phase, while a
        # converged agent saw the FINAL phase at end of training. Leaving
        # _init_velocity_speed unset makes reset_player_velocity_toward_ball a
        # no-op (utils_manager_based.py:363-365), matching converged-training env.

        # --- 1. Opponent action frame transform ---
        isaac_env = OpponentActionWrapper(isaac_env)

        # --- 2. Actuator model wrapper (expects m/s commands) ---
        actuator_cfg = getattr(env_config, "actuator_model", None)
        lag_cfg = getattr(env_config, "actuator_lag", None)
        if actuator_cfg is not None and actuator_cfg.enable:
            if lag_cfg is not None and lag_cfg.enable:  # as train_dreamer.py: never both
                raise ValueError("actuator_model and actuator_lag are alternatives; enable only one")
            isaac_env = ActuatorModelWrapper(isaac_env, device=env_cfg.sim.device, model_file=actuator_cfg.checkpoint)
        elif lag_cfg is not None and lag_cfg.enable:
            # Parametric surrogate (v2-like hardware), same arguments and wrapper position as train_dreamer.py.
            from klask_rl.tasks.manager_based.klask_rl.actuator_model.actuator_lag import ActuatorLagWrapper

            isaac_env = ActuatorLagWrapper(isaac_env, dead_time_s=lag_cfg.dead_time_s, tau_s=lag_cfg.tau_s)

        # --- 3. Velocity scaling: [-1, 1] → [-max_velocity, max_velocity] m/s ---
        max_acceleration = getattr(env_config, "max_acceleration", None)
        _act_space = isaac_env.unwrapped.single_action_space
        isaac_env = VelocityScaleWrapper(
            isaac_env,
            max_velocity=float(max_velocity),
            max_acceleration=None if max_acceleration is None else float(max_acceleration),
            num_envs=int(isaac_env.unwrapped.num_envs),
            action_dim=int(_act_space.shape[0]),
            device=isaac_env.unwrapped.device,
        )

    # --- 4. Observation noise (any opponent type, when any std > 0) ---
    # Mirrors train_klask.py's wrapper position: noise is applied to both the
    # 'policy' and 'opponent' obs keys before either model reads them. The
    # extended obs dims are recomputed from the noisy base positions inside
    # the wrapper. Skipped entirely when all stds are zero.
    if obs_noise_stds is not None and any(v > 0.0 for v in obs_noise_stds.values()):
        isaac_env = ObservationNoiseWrapper(
            isaac_env,
            peg_position_std=float(obs_noise_stds["peg_position_std"]),
            peg_velocity_std=float(obs_noise_stds["peg_velocity_std"]),
            ball_position_std=float(obs_noise_stds["ball_position_std"]),
            ball_velocity_std=float(obs_noise_stds["ball_velocity_std"]),
            own_goal=KLASK_PARAMS["player_goal"],
            other_goal=KLASK_PARAMS["opponent_goal"],
        )

    if tournament:
        # Both seats are driven by the tournament runner. Keep the 4D action
        # space and the ordinary post-reset observations from gym.step().
        return isaac_env, None

    # --- 5. Opponent wrapper ---
    if opponent_type == "dreamer":
        opponent_wrapper = DreamerSelfPlayWrapper(
            isaac_env,
            eval_mode=True,
            opponent_action_method=opponent_action_method,
        )
    elif opponent_type == "ppo":
        opponent_wrapper = KlaskRlAgentOpponentWrapper(isaac_env, is_deterministic=True)
    elif opponent_type == "fast_sac":
        opponent_wrapper = KlaskRlFastSACOpponentWrapper(isaac_env, is_deterministic=True)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    isaac_env = opponent_wrapper

    # --- 6. IsaacLabVecEnv adapter ---
    vec_env = IsaacLabVecEnv(isaac_env, simulation_app=simulation_app)

    return vec_env, opponent_wrapper


def _seat_hardware_chain(isaac_env, env_config, seat_hardware, device):
    """The actuation chain with each seat on its own hardware block (tournament ``env.seat_hardware``).

    Action layout (tournament worker): columns 0:2 = seat 0 (Peg_1, observation key "policy"), columns 2:4 = seat 1
    (Peg_2, key "opponent"; in its own 180-degree-rotated frame until OpponentActionWrapper). Same wrappers in the
    same order as the shared chain in _make_eval_env, but every stage acts only on the columns of the seats whose
    block enables it, with that seat's parameters. Each seat's columns therefore equal what the shared chain gives
    when both seats carry that seat's block (tests: scripts/tournament/tests/test_seat_hardware.py):

      0. KlaskRlCollisionAvoidanceWrapper  pegs of the seats with collision_avoidance, each ramp at its seat's
                                           max_velocity
      1. OpponentActionWrapper
      2. ActuatorLagWrapper                columns of the seats with actuator_lag (one wrapper per distinct setting)
      2. ActuatorModelWrapper              pegs of the seats with actuator_model (one wrapper per checkpoint),
                                           directly inside VelocityScaleWrapper as in the v1 chain: it writes the
                                           model outputs into VelocityScaleWrapper's command tensor in place, which
                                           then keeps them as that seat's previous velocity, exactly as on v1
      3. VelocityScaleWrapper              per-column max_velocity and max_acceleration
    """
    hardware_keys = ("max_velocity", "max_acceleration", "actuator_model", "actuator_lag", "collision_avoidance")
    if any(getattr(env_config, key, None) is not None for key in hardware_keys):
        raise ValueError("env.seat_hardware replaces the shared hardware keys; the env config must not set both")
    if len(seat_hardware) != 2:
        raise ValueError(f"env.seat_hardware needs one block per seat (2), got {len(seat_hardware)}")

    def enabled(block, key):
        entry = getattr(block, key, None)
        return entry if entry is not None and bool(getattr(entry, "enable", False)) else None

    seats = []
    for index, block in enumerate(seat_hardware):
        if getattr(block, "max_velocity", None) is None:
            raise ValueError(f"env.seat_hardware[{index}].max_velocity is required")
        model, lag = enabled(block, "actuator_model"), enabled(block, "actuator_lag")
        if model is not None and lag is not None:  # as train_dreamer.py: never both
            raise ValueError(f"Seat {index}: actuator_model and actuator_lag are alternatives; enable only one")
        acceleration = getattr(block, "max_acceleration", None)
        seats.append(
            {
                "max_velocity": float(block.max_velocity),
                "max_acceleration": None if acceleration is None else float(acceleration),
                "model": model,
                "lag": lag,
                "collision": enabled(block, "collision_avoidance") is not None,
            }
        )
    columns = (slice(0, 2), slice(2, 4))

    # --- 0. Collision avoidance (world frame), each peg's ramp at its seat's speed limit ---
    clipped = tuple(index for index, seat in enumerate(seats) if seat["collision"])
    if clipped:
        isaac_env = KlaskRlCollisionAvoidanceWrapper(
            isaac_env, max_vel=tuple(seat["max_velocity"] for seat in seats), pegs=clipped
        )

    # --- 1. Opponent action frame transform ---
    isaac_env = OpponentActionWrapper(isaac_env)

    # --- 2. Actuator lag (the v2 stand-in) on the lag seats' columns ---
    lagged = {}
    for index, seat in enumerate(seats):
        if seat["lag"] is not None:
            lagged.setdefault((float(seat["lag"].dead_time_s), float(seat["lag"].tau_s)), []).append(index)
    if lagged:
        from klask_rl.tasks.manager_based.klask_rl.actuator_model.actuator_lag import ActuatorLagWrapper

        for (dead_time, tau), indices in lagged.items():
            span = slice(0, 4) if len(indices) == 2 else columns[indices[0]]
            isaac_env = ActuatorLagWrapper(isaac_env, dead_time_s=dead_time, tau_s=tau, columns=span)

    # --- 2. Learned actuator model (v1) on the model seats' pegs ---
    modelled = {}
    for index, seat in enumerate(seats):
        if seat["model"] is not None:
            modelled.setdefault(str(seat["model"].checkpoint), []).append(index)
    for checkpoint, pegs in modelled.items():
        isaac_env = ActuatorModelWrapper(isaac_env, device=device, model_file=checkpoint, pegs=tuple(pegs))

    # --- 3. Velocity scaling, per column ---
    action_dim = int(isaac_env.unwrapped.single_action_space.shape[0])
    if action_dim != 4:
        raise ValueError(f"Per-seat hardware expects the 4-column two-seat action, got {action_dim}")
    return VelocityScaleWrapper(
        isaac_env,
        max_velocity=[seat["max_velocity"] for seat in seats for _ in range(2)],
        max_acceleration=[seat["max_acceleration"] for seat in seats for _ in range(2)],
        num_envs=int(isaac_env.unwrapped.num_envs),
        action_dim=action_dim,
        device=isaac_env.unwrapped.device,
    )
