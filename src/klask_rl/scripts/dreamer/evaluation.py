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

    # max_velocity is needed by both KlaskRlCollisionAvoidanceWrapper (below)
    # and VelocityScaleWrapper (further down), so look it up once up front.
    max_velocity = getattr(env_config, "max_velocity", None)
    if max_velocity is None:
        raise ValueError("env_config.max_velocity is required; VelocityScaleWrapper needs it to scale [-1, 1] → m/s.")

    # --- Action manager scale = 1.0 (pass-through, m/s in → m/s out) ---
    # Velocity scaling is done by VelocityScaleWrapper below so that the
    # actuator model sees commands in m/s (matching its training units).
    action_mgr = isaac_env.unwrapped.action_manager
    for term in action_mgr._terms.values():
        term._scale = 1.0

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
    if actuator_cfg is not None and actuator_cfg.enable:
        isaac_env = ActuatorModelWrapper(isaac_env, device=env_cfg.sim.device, model_file=actuator_cfg.checkpoint)

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
