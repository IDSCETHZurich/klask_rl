#!/usr/bin/env python3
"""Play / head-to-head evaluation for FastSAC+HER (IsaacLab).

Loads one or two FastSAC checkpoints and runs them against each other in the
same IsaacLab environment used by ``train_fast_sac_isaaclab.py``. The wrapper
chain mirrors training byte-for-byte so the policy sees identical action
scaling, action frame, and dynamics during inference.

Single-agent mode (no ``--opponent_checkpoint``):
    Opponent is a deep copy of the player actor — mirrors the training-time
    self-play setup (the env's opponent observation group is already mirrored).

Head-to-head mode (``--opponent_checkpoint`` set):
    Two separate checkpoints control the two pegs. Per-game metrics are
    accumulated via ``EvalMetricsTracker``; on exit a text summary and an
    ``.npz`` of per-game metrics are written next to the player checkpoint.
"""

# =============================================================================
# Phase 1: AppLauncher must run before any IsaacLab / USD / warp imports.
# =============================================================================

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play / head-to-head eval for FastSAC+IsaacLab.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to player .pt checkpoint.")
parser.add_argument(
    "--opponent_checkpoint",
    type=str,
    default=None,
    help="Path to opponent .pt checkpoint. When set, runs head-to-head; when absent, opponent = copy of player.",
)
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel envs (default: 1024).")
parser.add_argument("--num_games", type=int, default=10000, help="Stop after this many completed games.")
parser.add_argument("--task", type=str, default="Klask-Rl-FastSAC-v0", help="Gym task id (must match training).")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Override episode length in seconds. Defaults to the value stored in the player checkpoint.",
)
parser.add_argument(
    "--deterministic",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use mean action (deterministic) vs. sampling from the policy distribution.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in env steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
)
parser.add_argument(
    "--boxplot",
    action="store_true",
    help="After saving the eval-metrics .npz, also generate a box-plot PNG next to it.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Sim may override SIGINT; restore Python's default so Ctrl-C works.
import signal

signal.signal(signal.SIGINT, signal.default_int_handler)

# =============================================================================
# Phase 2: Everything else.
# =============================================================================

import copy
import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.utils.dict import print_dict
from tqdm import tqdm

# Ensure klask_her (Actor / EmpiricalNormalization) is importable.
_local_klask_her_root = Path(__file__).resolve().parent / "klask_her"
if _local_klask_her_root.exists():
    sys.path.insert(0, str(_local_klask_her_root))

import klask_rl.tasks  # noqa: F401  — registers Klask-Rl-FastSAC-v0
from klask_her.agents.fast_sac import Actor
from klask_her.agents.fast_sac_utils import EmpiricalNormalization
from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.tasks.manager_based.klask_rl.eval_metrics import EvalMetricsTracker
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    FastSACEnvWrapper,
    InitializationWrapper,
    KlaskRlCollisionAvoidanceWrapper,
    VelocityScaleWrapper,
    configure_domain_randomization,
)
from klask_rl.tasks.manager_based.klask_rl.wrappers.klask_rl_ovservation_wrappers import OpponentActionWrapper


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def load_checkpoint(path: str, device: str) -> dict:
    """Load a FastSAC checkpoint. The dict embeds actor weights, obs_normalizer
    state, and the training config (see ``save_params`` in fast_sac_utils.py)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not isinstance(cfg, dict):
        # Defensive: older checkpoints might have stringified the config.
        cfg = {}
    ckpt["config"] = cfg
    return ckpt


def build_actor_from_ckpt(ckpt: dict, obs_dim: int, act_dim: int, device: torch.device) -> Actor:
    cfg = ckpt["config"]
    hidden_dim = int(cfg.get("actor_hidden_dim", 512))
    actor = Actor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        action_scale=1.0,
        device=device,
    ).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor


def build_normalizer_from_ckpt(ckpt: dict, obs_dim: int, device: torch.device) -> EmpiricalNormalization:
    normalizer = EmpiricalNormalization(obs_dim, device=device)
    state = ckpt.get("obs_normalizer_state")
    if state is not None:
        normalizer.load_state_dict(state)
    else:
        print("[WARN] Checkpoint has no 'obs_normalizer_state'; using identity normalization.")
    normalizer.eval()
    return normalizer


def build_env(player_cfg: dict, num_envs: int, device: str, gym_id: str, episode_length_s_override: float | None,
              use_fabric: bool, record_video: bool):
    """Build the IsaacLab env with the same wrapper chain as training.

    Returns ``(fast_sac_env, isaac_env)`` so callers can pass the inner
    ``isaac_env`` to ``EvalMetricsTracker`` (which needs ``env.unwrapped``).
    """
    # Resolve the env config class from the gym registration.
    env_cfg_entry = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    module_name, class_name = env_cfg_entry.rsplit(":", 1)
    env_cfg_class = getattr(importlib.import_module(module_name), class_name)
    env_cfg = env_cfg_class()

    # --- Pre-construction overrides (mirrors make_isaaclab_env in training) ---
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.sim.dt = float(player_cfg.get("sim_dt", 0.001))
    env_cfg.decimation = int(player_cfg.get("decimation", 20))
    env_cfg.episode_length_s = (
        float(episode_length_s_override)
        if episode_length_s_override is not None
        else float(player_cfg.get("episode_length_s", 10.0))
    )
    ball_reset_x = tuple(player_cfg.get("ball_reset_x", (-0.15, 0.15)))
    ball_reset_y = tuple(player_cfg.get("ball_reset_y", (-0.1, -0.04)))
    env_cfg.ball_reset_position_x = ball_reset_x
    env_cfg.ball_reset_position_y = ball_reset_y
    env_cfg.events.reset_ball_position.params["pose_range"]["x"] = ball_reset_x
    env_cfg.events.reset_ball_position.params["pose_range"]["y"] = ball_reset_y

    # All-disabled DR matches training default.
    configure_domain_randomization(
        env_cfg,
        {
            "ball_mass": {"enable": False},
            "material_ball": {"enable": False},
            "material_board": {"enable": False},
            "material_peg": {"enable": False},
            "actuator": {"enable": False},
        },
    )

    goal_reward = float(player_cfg.get("goal_reward", 1000.0))
    env_cfg.rewards.goal_scored.weight = goal_reward
    env_cfg.rewards.goal_conceded.weight = -goal_reward
    env_cfg.rewards.player_in_goal.weight = -goal_reward
    env_cfg.rewards.opponent_in_goal.weight = goal_reward

    isaac_env = gym.make(
        gym_id,
        cfg=env_cfg,
        render_mode="rgb_array" if record_video else None,
    )

    # Optional video recording — wraps the raw gym env so it captures before
    # the custom (non-gym-wrapper) FastSAC adapter is added.
    if record_video:
        ckpt_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
        video_kwargs = {
            "video_folder": os.path.join(ckpt_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording playback video.")
        print_dict(video_kwargs, nesting=4)
        isaac_env = gym.wrappers.RecordVideo(isaac_env, **video_kwargs)

    # Action-manager pass-through: VelocityScaleWrapper owns [-1, 1] -> m/s.
    for term in isaac_env.unwrapped.action_manager._terms.values():
        term._scale = 1.0

    # --- Wrapper chain (innermost -> outermost), exactly matches training ---

    # 0. Collision avoidance (innermost; world-frame m/s clipping).
    max_velocity = float(player_cfg.get("max_velocity", 0.4))
    if bool(player_cfg.get("enable_collision_avoidance", True)):
        isaac_env = KlaskRlCollisionAvoidanceWrapper(
            isaac_env,
            max_vel=max_velocity,
            peg1_idx=slice(4, 6),
            peg2_idx=slice(6, 8),
        )

    # 1. Opponent action frame transform (ego -> world; negates opponent dims).
    isaac_env = OpponentActionWrapper(isaac_env)

    # 2. Actuator model (m/s in, m/s out). Must sit inside VelocityScaleWrapper.
    if bool(player_cfg.get("enable_actuator_model", True)):
        actuator_ckpt = player_cfg.get("actuator_model_checkpoint")
        isaac_env = ActuatorModelWrapper(
            isaac_env,
            model_file=actuator_ckpt,
            pos_idx=slice(4, 6),
            vel_idx=slice(8, 10),
        )

    # 3. Velocity scaling: [-1, 1] -> [-max_velocity, max_velocity] m/s.
    max_acceleration = player_cfg.get("max_acceleration", 10.0)
    act_space = isaac_env.unwrapped.single_action_space
    isaac_env = VelocityScaleWrapper(
        isaac_env,
        max_velocity=max_velocity,
        max_acceleration=None if max_acceleration is None else float(max_acceleration),
        num_envs=int(isaac_env.unwrapped.num_envs),
        action_dim=int(act_space.shape[0]),
        device=isaac_env.unwrapped.device,
    )

    # 4. Initialization (sets env.unwrapped._init_velocity_speed for the event manager).
    if bool(player_cfg.get("enable_initialization", False)):
        init_speed = float(player_cfg.get("init_velocity_speed", 1.0))
        isaac_env = InitializationWrapper(
            isaac_env,
            {"init_velocity": {"type": "static", "speed": init_speed}},
        )

    # 5. FastSAC adapter (outermost).
    fast_sac_env = FastSACEnvWrapper(isaac_env)
    return fast_sac_env, isaac_env


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    device = torch.device(args_cli.device)

    # --- Load player checkpoint ---
    print(f"[INFO] Loading player checkpoint: {args_cli.checkpoint}")
    player_ckpt = load_checkpoint(args_cli.checkpoint, device=str(device))
    player_cfg = player_ckpt["config"]

    # --- Build env (player's config drives the env setup) ---
    env, isaac_env = build_env(
        player_cfg=player_cfg,
        num_envs=args_cli.num_envs,
        device=str(device),
        gym_id=args_cli.task,
        episode_length_s_override=args_cli.episode_length_s,
        use_fabric=not args_cli.disable_fabric,
        record_video=args_cli.video,
    )

    obs_dim = env.obs_dim
    act_dim = 2  # Player controls 2 dims; full env action is 4-dim concat.

    # --- Build player actor + normalizer ---
    player_actor = build_actor_from_ckpt(player_ckpt, obs_dim=obs_dim, act_dim=act_dim, device=device)
    player_normalizer = build_normalizer_from_ckpt(player_ckpt, obs_dim=obs_dim, device=device)

    # --- Build opponent actor + normalizer ---
    head_to_head = args_cli.opponent_checkpoint is not None
    if head_to_head:
        print(f"[INFO] Loading opponent checkpoint: {args_cli.opponent_checkpoint}")
        opponent_ckpt = load_checkpoint(args_cli.opponent_checkpoint, device=str(device))
        opponent_actor = build_actor_from_ckpt(opponent_ckpt, obs_dim=obs_dim, act_dim=act_dim, device=device)
        opponent_normalizer = build_normalizer_from_ckpt(opponent_ckpt, obs_dim=obs_dim, device=device)
    else:
        # Self vs. self: deep-copy the player actor + normalizer.
        opponent_actor = copy.deepcopy(player_actor)
        opponent_actor.eval()
        opponent_normalizer = copy.deepcopy(player_normalizer)
        opponent_normalizer.eval()

    deterministic = bool(args_cli.deterministic)
    print(
        f"[INFO] num_envs={env.num_envs}  obs_dim={obs_dim}  act_dim={act_dim}  "
        f"deterministic={deterministic}  head_to_head={head_to_head}"
    )

    # --- Per-game metrics tracker (head-to-head only; same gating as play_klask.py) ---
    tracker = EvalMetricsTracker(isaac_env) if head_to_head else None

    # --- Reset + first-step opp_obs seed (matches training) ---
    obs = env.reset_all()
    opp_obs = obs.clone()
    infos: dict = {}

    pbar = tqdm(total=args_cli.num_games, desc="Games") if head_to_head else None
    timestep = 0
    start_time = time.time()

    try:
        while simulation_app.is_running() and (head_to_head or time.time() - start_time < 1000.0):
            if head_to_head and tracker.total_games >= args_cli.num_games:
                break

            with torch.inference_mode():
                norm_obs = player_normalizer(obs, update=False)
                actions = player_actor.explore(norm_obs, deterministic=deterministic)

                opp_norm = opponent_normalizer(opp_obs, update=False)
                a2 = opponent_actor.explore(opp_norm, deterministic=deterministic)

                # OpponentActionWrapper will negate the opponent dims back to world frame.
                full_actions = torch.cat([actions, a2], dim=-1)
                obs, _rewards, dones, infos = env.step(full_actions)

                if "opponent_obs" in infos:
                    opp_obs = infos["opponent_obs"]

                if head_to_head:
                    tracker.update()
                    if dones.any():
                        tracker.finalize(dones)
                        num_done = int(dones.sum().item())
                        if pbar is not None and num_done > 0:
                            pbar.update(min(num_done, args_cli.num_games - pbar.n))
                            pbar.set_postfix(**tracker.live_postfix())

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
    except KeyboardInterrupt:
        if head_to_head:
            print(
                f"\n[INFO] Evaluation interrupted by user after {tracker.total_games} games. "
                "Printing summary for completed games..."
            )
        else:
            print("\n[INFO] Playback interrupted by user.")

    if pbar is not None:
        pbar.close()

    isaac_env.close()

    # --- Summary + result files (head-to-head only) ---
    if head_to_head:
        summary_lines = [
            "\n" + "=" * 50,
            "  HEAD-TO-HEAD EVALUATION RESULTS",
            "=" * 50,
            f"  Player checkpoint  : {args_cli.checkpoint}",
            f"  Opponent checkpoint: {args_cli.opponent_checkpoint}",
            f"  Total games played : {tracker.total_games}",
            "-" * 50,
        ]
        summary_lines += tracker.summary_body_lines()
        summary_lines.append("=" * 50 + "\n")

        summary_text = "\n".join(summary_lines)
        print(summary_text)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)), "head_to_head_results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"h2h_results_{timestamp}.txt")
        with open(results_file, "w") as f:
            f.write(summary_text)
        print(f"[INFO] Head-to-head results saved to: {results_file}")

        npz_file = os.path.join(results_dir, f"h2h_results_{timestamp}.npz")
        tracker.save_npz(
            npz_file,
            metadata={
                "player_checkpoint": args_cli.checkpoint,
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
            png_file = _ep_mod.plot_boxplots(npz_file)
            print(f"[INFO] Box-plot saved to:            {png_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
