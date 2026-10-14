# =============================================================================
# Phase 1: AppLauncher must run before any IsaacLab / USD / warp imports.
# =============================================================================

import os

# Reduce caching-allocator fragmentation across the per-checkpoint reload
# cycle. Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    nargs="+",
    required=True,
    help=(
        "One or more paths or glob patterns to the player agent checkpoint(s) (.pt). "
        "Each entry is resolved independently — literal paths are kept as-is, glob "
        "patterns (containing '*', '?' or '[') are expanded. Pass multiple values to "
        "evaluate specific files: --checkpoint a.pt b.pt, or mix patterns: "
        "--checkpoint 'runs/A/*.pt' 'runs/B/last.pt'."
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
    choices=["dreamer", "ppo", "fast_sac"],
    help=(
        "Type of opponent agent: 'dreamer', 'ppo' (rl_games), or 'fast_sac' "
        "(distributional SAC trained by train_fast_sac_isaaclab.py; distinct "
        "from any SB3 SAC). For 'fast_sac', --opponent_config is ignored — the "
        "actor hyperparameters are read from the checkpoint's embedded config."
    ),
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
        "in the player config. Supports any opponent type. Omit for normal behavior."
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

import gc
import glob as glob_module
import shlex
import signal
import warnings
from datetime import datetime

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Isaac Sim may override SIGINT; restore Python's default so Ctrl-C works.
signal.signal(signal.SIGINT, signal.default_int_handler)

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent)
OmegaConf.register_new_resolver("script_dir", lambda: _SCRIPT_DIR, use_cache=True)

sys.path.append(str(pathlib.Path(_SCRIPT_DIR) / "r2dreamer"))
sys.path.append(_SCRIPT_DIR)
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

from evaluation import (
    _init_ppo_opponent_batch,
    _load_config,
    _load_dreamer_agent,
    _load_fast_sac_opponent,
    _load_ppo_opponent,
    _make_eval_env,
)
from klask_rl.tasks.manager_based.klask_rl.eval_metrics import EvalMetricsTracker

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

    # --- Resolve checkpoint patterns ---
    # Each --checkpoint entry is resolved independently: literal paths kept as-is,
    # glob patterns expanded. Order is preserved across entries; within an
    # expanded pattern matches are sorted; duplicates are dropped.
    checkpoint_paths: list[str] = []
    seen: set[str] = set()
    for entry in args_cli.checkpoint:
        if any(c in entry for c in ("*", "?", "[")):
            matches = sorted(glob_module.glob(entry, recursive=True))
            if not matches:
                raise FileNotFoundError(f"No checkpoints matched pattern: {entry}")
            resolved = matches
        else:
            resolved = [entry]
        for p in resolved:
            if p not in seen:
                seen.add(p)
                checkpoint_paths.append(p)
    if len(checkpoint_paths) > 1 or any(c in e for e in args_cli.checkpoint for c in ("*", "?", "[")):
        print(f"[INFO] Resolved {len(checkpoint_paths)} checkpoint(s):")
        for i, cp in enumerate(checkpoint_paths, 1):
            print(f"  {i}. {os.path.basename(cp)}")

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
        simulation_app=simulation_app,
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
    elif opponent_type == "fast_sac":
        fast_sac_opponent = _load_fast_sac_opponent(args_cli.opponent_checkpoint, device)
        opponent_wrapper.add_opponent(fast_sac_opponent)

    # Collect per-checkpoint results for aggregate summary.
    all_results = []
    npz_paths_written: list[str] = []
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
        npz_paths_written.append(npz_file)
        print(f"[INFO] Per-game metrics saved to:    {npz_file}")

        if args_cli.boxplot:
            import importlib.util

            _ep_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval_plot.py"))
            _ep_spec = importlib.util.spec_from_file_location("eval_plot", _ep_path)
            _ep_mod = importlib.util.module_from_spec(_ep_spec)
            _ep_spec.loader.exec_module(_ep_mod)
            png_file = _ep_mod.plot_boxplots(npz_file, title_suffix=os.path.basename(ckpt_path))
            print(f"[INFO] Box-plot saved to:            {png_file}")

        # Free per-checkpoint GPU state before loading the next one.
        del player, tracker, pbar
        try:
            del act, agent_state, done
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        # Stop evaluating further checkpoints if user interrupted.
        if interrupted:
            remaining = len(checkpoint_paths) - (ckpt_idx + 1)
            if remaining > 0:
                print(f"[INFO] Skipping {remaining} remaining checkpoint(s) due to interrupt.")
            break

    # --- Aggregate summary (when multiple checkpoints) ---
    if len(all_results) > 1:
        import importlib.util

        _ea_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval_aggregate.py"))
        _ea_spec = importlib.util.spec_from_file_location("eval_aggregate", _ea_path)
        _ea_mod = importlib.util.module_from_spec(_ea_spec)
        _ea_spec.loader.exec_module(_ea_mod)
        agg_file = os.path.join(results_dir, f"h2h_aggregate_{timestamp}.txt")
        agg_plot = os.path.join(results_dir, f"h2h_aggregate_{timestamp}.png")
        agg_term_plot = os.path.join(results_dir, f"h2h_aggregate_{timestamp}_terminations.png")
        agg_text, _, _, _ = _ea_mod.aggregate_from_npz_files(
            npz_paths_written,
            out_path=agg_file,
            invocation_cmd=invocation_cmd,
            plot_path=agg_plot,
            term_plot_path=agg_term_plot,
        )
        print(agg_text)
        print(f"[INFO] Aggregate results saved to:  {agg_file}")
        print(f"[INFO] Win-rate plot saved to:      {agg_plot}")
        print(f"[INFO] Termination plot saved to:   {agg_term_plot}")

    # Cleanup.
    vec_env._env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
