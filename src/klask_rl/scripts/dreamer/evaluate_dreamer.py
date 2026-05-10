# =============================================================================
# Phase 1: AppLauncher must run before any IsaacLab / USD / warp imports.
# =============================================================================

import argparse
import pathlib
import sys

# The KLASK Dreamer env always uses cameras (cnn_keys: "image"), so we
# unconditionally enable them.
if "--enable_cameras" not in sys.argv:
    sys.argv.insert(1, "--enable_cameras")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Head-to-head evaluation of Dreamer agents. Supports glob patterns in "
        "--checkpoint to evaluate multiple player checkpoints against the same opponent."
    )
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help=(
        "Path (or glob pattern) to the player agent checkpoint(s) (.pt). "
        "Supports wildcards such as 'runs/*/checkpoint_*.pt' to evaluate "
        "multiple checkpoints sequentially against the same opponent."
    ),
)
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the player's Hydra training config YAML.",
)
parser.add_argument(
    "--opponent_checkpoint",
    type=str,
    required=True,
    help="Path to the opponent agent checkpoint (.pt).",
)
parser.add_argument(
    "--opponent_config",
    type=str,
    default=None,
    help="Path to the opponent's config YAML. Uses --config when not provided.",
)
parser.add_argument(
    "--num_games",
    type=int,
    default=10000,
    help="Number of complete games to play.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1024,
    help="Number of parallel environments.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Override episode length in seconds.",
)
parser.add_argument(
    "--opponent_type",
    type=str,
    default="dreamer",
    choices=["dreamer", "ppo"],
    help="Type of opponent agent: 'dreamer' or 'ppo' (rl_games).",
)
parser.add_argument(
    "--opponent_action_method",
    type=str,
    default=None,
    choices=["zero", "random"],
    help=(
        "Override what each agent's RSSM ingests in the OTHER agent's action "
        "slot of prev_action. 'zero' feeds zeros; 'random' feeds samples in "
        "[-1, 1]. The simulator still receives both agents' real actions; "
        "only the RSSM inputs are overridden. Requires opponent_separation=True "
        "in the player config and --opponent_type=dreamer. Omit for normal behavior."
    ),
)
parser.add_argument(
    "--boxplot",
    action="store_true",
    help="After saving the per-checkpoint eval-metrics .npz, also generate a box-plot PNG next to it.",
)
parser.add_argument(
    "--peg_position_std",
    type=float,
    default=0.0,
    help=(
        "Std [m] of Gaussian noise added to peg positions in the observation. "
        "Applied to both 'policy' and 'opponent' obs keys (independent samples) "
        "for any opponent type. Extended obs dims are recomputed from the noisy "
        "base positions. Default 0.0 (no noise)."
    ),
)
parser.add_argument(
    "--peg_velocity_std",
    type=float,
    default=0.0,
    help="Std [m/s] of Gaussian noise added to peg velocities (any opponent type).",
)
parser.add_argument(
    "--ball_position_std",
    type=float,
    default=0.0,
    help="Std [m] of Gaussian noise added to the ball position (any opponent type).",
)
parser.add_argument(
    "--ball_velocity_std",
    type=float,
    default=0.0,
    help="Std [m/s] of Gaussian noise added to the ball velocity (any opponent type).",
)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Phase 2: Everything else — safe to import now that the sim is running.
# =============================================================================

import glob as glob_module
import importlib
import os
import shlex
import signal
import warnings
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

# Isaac Sim may override SIGINT; restore Python's default so Ctrl-C works.
signal.signal(signal.SIGINT, signal.default_int_handler)

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent)
OmegaConf.register_new_resolver("script_dir", lambda: _SCRIPT_DIR, use_cache=True)

sys.path.append(str(pathlib.Path(_SCRIPT_DIR) / "r2dreamer"))
sys.path.append(_SCRIPT_DIR)
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# Register IsaacLab task environments.
import isaaclab_tasks  # noqa: F401
import klask_rl.tasks  # noqa: F401
from dreamer import Dreamer
from dreamer_self_play import DreamerSelfPlayWrapper
from env_cfg_utils import apply_camera_size_to_env_cfg
from envs.isaaclab import IsaacLabVecEnv
from isaaclab.sim import RenderCfg
from klask_rl.assets.robots.klask_params import KLASK_PARAMS
from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.tasks.manager_based.klask_rl.eval_metrics import EvalMetricsTracker
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    KlaskRlAgentOpponentWrapper,
    KlaskRlCollisionAvoidanceWrapper,
    ObservationNoiseWrapper,
    OpponentActionWrapper,
    VelocityScaleWrapper,
    configure_domain_randomization,
)

# =============================================================================
# Helpers
# =============================================================================


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
        isaac_env = ActuatorModelWrapper(isaac_env, model_file=actuator_cfg.checkpoint)

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

    # --- 5. Opponent wrapper ---
    if opponent_type == "dreamer":
        opponent_wrapper = DreamerSelfPlayWrapper(
            isaac_env,
            eval_mode=True,
            opponent_action_method=opponent_action_method,
        )
    elif opponent_type == "ppo":
        opponent_wrapper = KlaskRlAgentOpponentWrapper(isaac_env, is_deterministic=True)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    isaac_env = opponent_wrapper

    # --- 6. IsaacLabVecEnv adapter ---
    vec_env = IsaacLabVecEnv(isaac_env, simulation_app=simulation_app)

    return vec_env, opponent_wrapper


def _load_ppo_opponent(config_path, checkpoint_path, num_envs, device, ppo_obs_space):
    """Load a PPO opponent agent from an rl_games checkpoint.

    Mirrors the loading pattern from play_klask.py.
    """
    import yaml
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    # Load base config from registry, then override with user YAML.
    agent_cfg = load_cfg_from_registry("Klask-Rl-v0", "rl_games_cfg_entry_point")
    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f)
        agent_cfg.update(user_cfg)

    # PPO outputs in [-1, 1]; VelocityScaleWrapper handles the m/s scaling
    # uniformly with the player and random opponents. Declaring [-1, 1] makes
    # rl_games' rescale_actions a no-op, so the raw post-clamp network output
    # is returned unchanged (which is what the model was trained to produce
    # pre-rescale).
    ppo_act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # Minimal stub env so rl_games can query observation/action spaces
    # without creating a full Isaac environment.
    class _StubEnv:
        def __init__(self):
            self.observation_space = ppo_obs_space
            self.action_space = ppo_act_space
            self.num_envs = num_envs
            self.num_agents = 1

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint_path
    agent_cfg["params"]["config"]["num_actors"] = num_envs

    # Register a stub rl_games env so Runner.create_player() can query spaces.
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: _StubEnv(),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: _StubEnv()},
    )

    runner = Runner()
    runner.load(agent_cfg)
    opponent: BasePlayer = runner.create_player()

    # Use set_weights (not restore) to properly load running_mean_std state.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    opponent.set_weights(ckpt)

    opponent.reset()
    opponent.device = torch.device(device)
    opponent.model.to(device)
    opponent.actions_low = opponent.actions_low.to(device)
    opponent.actions_high = opponent.actions_high.to(device)

    # init_rnn() is deferred to _init_ppo_opponent_batch() — we need a real
    # post-reset opponent obs to call get_batch_size() first so rl-games sets
    # has_batch_dimension/batch_size correctly before allocating RNN state.
    return opponent


def _init_ppo_opponent_batch(opponent, vec_env):
    """Run rl-games' batch-size handshake against a freshly reset env.

    Mirrors play_klask.py:354,357,381: opponent.get_batch_size(obs, 1) sets
    has_batch_dimension/batch_size from the actual obs shape, and only then
    is init_rnn() safe to call (it sizes hidden state from batch_size).

    Must be called AFTER vec_env.reset() and BEFORE the first vec_env.step().
    """
    opp_obs = vec_env._env.unwrapped.observation_manager.compute()["opponent"]
    _ = opponent.get_batch_size(opp_obs, 1)
    if opponent.is_rnn:
        opponent.init_rnn()


def _load_dreamer_agent(full_cfg, obs_space, act_space, checkpoint_path, device):
    """Create a Dreamer agent and load checkpoint weights."""
    import copy

    cfg = copy.deepcopy(full_cfg.model)

    # Mirror the opponent_separation parsing that train_dreamer.py does at runtime.
    _opp_sep_raw = getattr(full_cfg, "opponent_separation", None)
    if OmegaConf.is_config(_opp_sep_raw):
        _opp_sep_dict = OmegaConf.to_container(_opp_sep_raw, resolve=True)
        _opp_sep = bool(_opp_sep_dict.get("enabled", False))
        _imag_opponent = str(_opp_sep_dict.get("imag_opponent", "random"))
        _traj_mirror = bool(
            _opp_sep_dict.get("trajectory_mirroring", {}).get("enabled", False)
            if isinstance(_opp_sep_dict.get("trajectory_mirroring"), dict)
            else _opp_sep_dict.get("trajectory_mirroring", False)
        )
    elif isinstance(_opp_sep_raw, bool):
        _opp_sep = _opp_sep_raw
        _imag_opponent = str(
            getattr(getattr(full_cfg, "opponent_separation_config", None) or {}, "imag_opponent", "random")
        )
        _traj_mirror = bool(getattr(full_cfg, "trajectory_mirroring", False))
    else:
        _opp_sep = False
        _imag_opponent = "random"
        _traj_mirror = False

    with open_dict(cfg):
        cfg.sim_device = str(device)
        cfg.train_device = str(device)
        cfg.train_devices = [str(device)]
        cfg.device = str(device)
        cfg.opponent_separation = _opp_sep
        cfg.imag_opponent = _imag_opponent
        cfg.trajectory_mirroring = _traj_mirror

    agent = Dreamer(cfg, obs_space, act_space).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    # _inference_* keys are aliases to _frozen_* in single-GPU mode — their
    # parameters are shared objects already loaded via _frozen_* keys.
    # Unexpected keys are from multi-GPU checkpoints (_inference_*_orig,
    # _imag_opp_*, compiled ._orig_mod copies) not needed for evaluation.
    real_missing = [k for k in missing if not k.startswith("_inference_")]
    if real_missing:
        raise RuntimeError(f"Missing keys in checkpoint: {real_missing}")
    agent.eval()
    return agent


# =============================================================================
# Main
# =============================================================================


def main():
    """Run head-to-head evaluation of one or more player checkpoints vs. a fixed opponent.

    When ``--checkpoint`` contains glob characters (``*``, ``?``, ``[``), all
    matching paths are resolved and each checkpoint is evaluated sequentially
    against the same opponent. The environment and opponent are created once and
    reused across all player checkpoints. Per-checkpoint results are printed and
    saved individually; when multiple checkpoints are evaluated an aggregate
    summary table is printed at the end.
    """
    device = args_cli.device
    opponent_type = args_cli.opponent_type

    # Reconstruct the exact invocation for posterity in the saved results files.
    invocation_cmd = shlex.join([sys.executable, *sys.argv])

    # --- Resolve checkpoint glob pattern ---
    checkpoint_pattern = args_cli.checkpoint
    if any(c in checkpoint_pattern for c in ("*", "?", "[")):
        checkpoint_paths = sorted(glob_module.glob(checkpoint_pattern, recursive=True))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints matched pattern: {checkpoint_pattern}")
        print(f"[INFO] Found {len(checkpoint_paths)} checkpoints matching '{checkpoint_pattern}':")
        for i, cp in enumerate(checkpoint_paths, 1):
            print(f"  {i}. {os.path.basename(cp)}")
    else:
        checkpoint_paths = [checkpoint_pattern]

    # --- Load player config ---
    player_cfg = _load_config(args_cli.config, device)

    # --- Validate --opponent_action_method preconditions before building the env ---
    # The flag overrides what each agent's RSSM sees in the OTHER agent's slot of
    # prev_action. The player-side override always fires when the flag is set
    # (works against any opponent type). The opponent-side override only fires
    # when the opponent has its own RSSM (i.e. --opponent_type=dreamer); for a
    # PPO opponent there is no RSSM to override, which is fine.
    if args_cli.opponent_action_method is not None:
        _opp_sep_raw = getattr(player_cfg, "opponent_separation", None)
        if OmegaConf.is_config(_opp_sep_raw):
            _opp_sep_enabled = bool(
                OmegaConf.to_container(_opp_sep_raw, resolve=True).get("enabled", False)
            )
        elif isinstance(_opp_sep_raw, bool):
            _opp_sep_enabled = _opp_sep_raw
        else:
            _opp_sep_enabled = False
        if not _opp_sep_enabled:
            raise SystemExit(
                "--opponent_action_method requires the player checkpoint to be "
                "trained with opponent_separation=True; this checkpoint has it disabled."
            )

    # --- Create evaluation environment (shared across all player checkpoints) ---
    num_envs = args_cli.num_envs
    obs_noise_stds = {
        "peg_position_std": args_cli.peg_position_std,
        "peg_velocity_std": args_cli.peg_velocity_std,
        "ball_position_std": args_cli.ball_position_std,
        "ball_velocity_std": args_cli.ball_velocity_std,
    }
    vec_env, opponent_wrapper = _make_eval_env(
        player_cfg.env,
        num_envs,
        args_cli.episode_length_s,
        opponent_type=opponent_type,
        opponent_action_method=args_cli.opponent_action_method,
        obs_noise_stds=obs_noise_stds,
    )
    obs_space = vec_env.observation_space
    act_space = vec_env.action_space

    # --- Load opponent agent (once) ---
    print(f"[INFO] Loading {opponent_type} opponent checkpoint: {args_cli.opponent_checkpoint}")
    if opponent_type == "dreamer":
        opp_config_path = args_cli.opponent_config or args_cli.config
        opp_cfg = _load_config(opp_config_path, device)
        opp_agent = _load_dreamer_agent(opp_cfg, obs_space, act_space, args_cli.opponent_checkpoint, device)
        opponent_wrapper.set_opponent(opp_agent)
        del opp_agent  # Wrapper deep-copied the needed modules.
    elif opponent_type == "ppo":
        base_env = opponent_wrapper.env.unwrapped
        ppo_obs_space = base_env.single_observation_space["opponent"]
        ppo_opponent = _load_ppo_opponent(
            args_cli.opponent_config,
            args_cli.opponent_checkpoint,
            num_envs,
            device,
            ppo_obs_space,
        )
        opponent_wrapper.add_opponent(ppo_opponent)

    # Collect per-checkpoint results for aggregate summary.
    all_results = []
    interrupted = False

    # --- Evaluate each player checkpoint ---
    for ckpt_idx, ckpt_path in enumerate(checkpoint_paths):
        if len(checkpoint_paths) > 1:
            print(f"\n{'#' * 60}\n  Checkpoint {ckpt_idx + 1}/{len(checkpoint_paths)}: {ckpt_path}\n{'#' * 60}")

        # --- Load player agent ---
        print(f"[INFO] Loading player checkpoint: {ckpt_path}")
        player = _load_dreamer_agent(player_cfg, obs_space, act_space, ckpt_path, device)

        # --- Reset tracking state ---
        tracker = EvalMetricsTracker(vec_env._env)

        # --- Game loop ---
        pbar = tqdm(total=args_cli.num_games, desc="Games")

        try:
            with torch.inference_mode():
                vec_env.reset()
                # PPO opponent needs to see a real post-reset obs for rl-games to
                # set has_batch_dimension/batch_size before init_rnn() runs. Must
                # happen after every reset(), since init_rnn() reallocates state.
                if opponent_type == "ppo":
                    _init_ppo_opponent_batch(opponent_wrapper.opponent, vec_env)
                done = torch.ones(num_envs, dtype=torch.bool, device=device)
                agent_state = player.get_initial_state(num_envs)
                act = agent_state["action"].clone()

                while tracker.total_games < args_cli.num_games and simulation_app.is_running():
                    # Step env (opponent actions generated inside the opponent wrapper).
                    trans, done = vec_env.step(act.detach(), done.detach())

                    # Override the opponent slot in the player's prev_action.
                    # Works for any opponent type (PPO opponents don't add
                    # opponent_action to obs, so this also fills it in for them).
                    if args_cli.opponent_action_method == "zero":
                        trans["opponent_action"] = torch.zeros_like(act)
                    elif args_cli.opponent_action_method == "random":
                        trans["opponent_action"] = 2.0 * torch.rand_like(act) - 1.0

                    # Player agent inference (deterministic).
                    act, agent_state = player.act(trans, agent_state, eval=True)

                    # Per-step metric accumulation (must run before finalize).
                    tracker.update()

                    # Track terminations.
                    if done.any():
                        num_done = int(done.sum().item())
                        tracker.finalize(done)
                        pbar.update(min(num_done, args_cli.num_games - pbar.n))
                        pbar.set_postfix(**tracker.live_postfix())
        except KeyboardInterrupt:
            interrupted = True
            print(
                f"\n[INFO] Evaluation interrupted by user after {tracker.total_games} games. "
                "Printing summary for completed games..."
            )

        pbar.close()

        # --- Print and save results ---
        all_results.append({
            "checkpoint": ckpt_path,
            "total_games": tracker.total_games,
            "player_wins": tracker.player_wins,
            "opponent_wins": tracker.opponent_wins,
            "draws": tracker.draws,
            "term_counts": dict(tracker.term_counts),
        })

        summary_lines = [
            "\n" + "=" * 50,
            "  HEAD-TO-HEAD EVALUATION RESULTS",
            "=" * 50,
            f"  Command            : {invocation_cmd}",
            f"  Player checkpoint  : {ckpt_path}",
            f"  Opponent type      : {opponent_type}",
            f"  Opponent checkpoint: {args_cli.opponent_checkpoint}",
            f"  Total games played : {tracker.total_games}",
            "-" * 50,
        ]
        summary_lines += tracker.summary_body_lines()
        summary_lines.append("=" * 50 + "\n")

        summary_text = "\n".join(summary_lines)
        print(summary_text)

        # Save to timestamped file.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("logs", "dreamer", "head_to_head_results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"h2h_results_{timestamp}.txt")
        with open(results_file, "w") as f:
            f.write(summary_text)
        print(f"[INFO] Head-to-head results saved to: {results_file}")

        npz_file = os.path.join(results_dir, f"h2h_results_{timestamp}.npz")
        tracker.save_npz(
            npz_file,
            metadata={
                "player_checkpoint": ckpt_path,
                "opponent_type": opponent_type,
                "opponent_checkpoint": args_cli.opponent_checkpoint or "",
            },
        )
        print(f"[INFO] Per-game metrics saved to:    {npz_file}")

        if args_cli.boxplot:
            import importlib.util

            _ep_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval_plot.py"))
            _ep_spec = importlib.util.spec_from_file_location("eval_plot", _ep_path)
            _ep_mod = importlib.util.module_from_spec(_ep_spec)
            _ep_spec.loader.exec_module(_ep_mod)
            png_file = _ep_mod.plot_boxplots(npz_file, title_suffix=os.path.basename(ckpt_path))
            print(f"[INFO] Box-plot saved to:            {png_file}")

        # Free player model before loading the next one.
        del player

        # Stop evaluating further checkpoints if user interrupted.
        if interrupted:
            remaining = len(checkpoint_paths) - (ckpt_idx + 1)
            if remaining > 0:
                print(f"[INFO] Skipping {remaining} remaining checkpoint(s) due to interrupt.")
            break

    # --- Aggregate summary (when multiple checkpoints) ---
    if len(all_results) > 1:
        agg_lines = [
            "\n" + "=" * 70,
            "  AGGREGATE RESULTS ACROSS ALL CHECKPOINTS",
            "=" * 70,
            f"  Command: {invocation_cmd}",
            "-" * 70,
            f"  {'Checkpoint':<45} {'Win%':>6}  {'W':>5}  {'L':>5}  {'D':>5}",
            "-" * 70,
        ]
        for r in all_results:
            ckpt_name = os.path.basename(r["checkpoint"])
            wr = f"{r['player_wins'] / r['total_games'] * 100:.1f}%" if r["total_games"] > 0 else "N/A"
            agg_lines.append(
                f"  {ckpt_name:<45} {wr:>6}  {r['player_wins']:>5}  {r['opponent_wins']:>5}  {r['draws']:>5}"
            )
        agg_lines.append("=" * 70 + "\n")
        agg_text = "\n".join(agg_lines)
        print(agg_text)

        # Save aggregate results.
        agg_file = os.path.join(results_dir, f"h2h_aggregate_{timestamp}.txt")
        with open(agg_file, "w") as f:
            f.write(agg_text)
        print(f"[INFO] Aggregate results saved to: {agg_file}")

    # Cleanup.
    vec_env._env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
