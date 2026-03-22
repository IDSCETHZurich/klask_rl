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

parser = argparse.ArgumentParser(description="Head-to-head evaluation of two Dreamer agents.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the player agent checkpoint (.pt).",
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
    default=2048,
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

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Phase 2: Everything else — safe to import now that the sim is running.
# =============================================================================

import importlib
import os
import signal
import warnings
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
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

# Register IsaacLab task environments.
import isaaclab_tasks  # noqa: F401
import klask_rl.tasks  # noqa: F401
from dreamer import Dreamer
from dreamer_self_play import DreamerSelfPlayWrapper
from envs.isaaclab import IsaacLabVecEnv
from isaaclab.sim import RenderCfg
from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    KlaskRlAgentOpponentWrapper,
    OpponentActionWrapper,
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


def _make_eval_env(env_config, num_envs, episode_length_s=None, opponent_type="dreamer"):
    """Create an evaluation env with the appropriate opponent wrapper.

    Simplified wrapper chain (no curriculum/logging):
      1. OpponentActionWrapper — negate opponent actions for coordinate frame
      2. ActuatorModelWrapper (if config.actuator_model)
      3. max_velocity scaling
      4. Opponent wrapper (DreamerSelfPlayWrapper or KlaskRlAgentOpponentWrapper)
      5. IsaacLabVecEnv — r2dreamer adapter

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

    sim_dt = getattr(env_config, "sim_dt", None)
    if sim_dt is not None:
        env_cfg.sim.dt = float(sim_dt)

    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.decimation = int(env_config.decimation)
    env_cfg.seed = int(env_config.seed)
    env_cfg.episode_length_s = float(episode_length_s or env_config.episode_length_s)
    env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")

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

    # --- Create base env ---
    isaac_env = gym.make(task_name, cfg=env_cfg)

    # --- 1. Opponent action frame transform (innermost) ---
    isaac_env = OpponentActionWrapper(isaac_env)

    # --- 2. Bounded action space (max_velocity) ---
    max_velocity = getattr(env_config, "max_velocity", None)
    if max_velocity is not None:
        vel = float(max_velocity)
        action_mgr = isaac_env.unwrapped.action_manager
        for term in action_mgr._terms.values():
            term._scale = vel

    # --- 3. Actuator model wrapper ---
    if getattr(env_config, "actuator_model", False):
        isaac_env = ActuatorModelWrapper(isaac_env)

    # --- 4. Opponent wrapper ---
    if opponent_type == "dreamer":
        opponent_wrapper = DreamerSelfPlayWrapper(isaac_env, eval_mode=True)
    elif opponent_type == "ppo":
        opponent_wrapper = KlaskRlAgentOpponentWrapper(isaac_env, is_deterministic=True)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    isaac_env = opponent_wrapper

    # --- 5. IsaacLabVecEnv adapter ---
    vec_env = IsaacLabVecEnv(isaac_env, simulation_app=simulation_app)

    return vec_env, opponent_wrapper


def _load_ppo_opponent(config_path, checkpoint_path, num_envs, device):
    """Load a PPO opponent agent from an rl_games checkpoint.

    Mirrors the loading pattern from play_klask.py.
    """
    import yaml
    from isaaclab_rl.rl_games import RlGamesGpuEnv
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rl_games.algos_torch import torch_ext
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    # Load base config from registry, then override with user YAML.
    agent_cfg = load_cfg_from_registry("Klask-Rl-v0", "rl_games_cfg_entry_point")
    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f)
        agent_cfg.update(user_cfg)

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint_path
    agent_cfg["params"]["config"]["num_actors"] = num_envs

    # Register a dummy rl_games env so Runner.create_player() works.
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: None},
    )

    runner = Runner()
    runner.load(agent_cfg)
    opponent: BasePlayer = runner.create_player()

    # Monkey-patch safe_load for device mapping (same as play_klask.py).
    _original_safe_load = torch_ext.safe_load

    def _safe_load_mapped(filename):
        return torch_ext.safe_filesystem_op(
            torch.load, filename, map_location=device, weights_only=False
        )

    torch_ext.safe_load = _safe_load_mapped
    opponent.restore(checkpoint_path)
    torch_ext.safe_load = _original_safe_load

    opponent.reset()
    opponent.device = torch.device(device)
    opponent.model.to(device)
    opponent.actions_low = opponent.actions_low.to(device)
    opponent.actions_high = opponent.actions_high.to(device)

    if opponent.is_rnn:
        opponent.init_rnn()

    return opponent


def _load_dreamer_agent(model_config, obs_space, act_space, checkpoint_path, device):
    """Create a Dreamer agent and load checkpoint weights."""
    agent = Dreamer(model_config, obs_space, act_space).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    return agent


# =============================================================================
# Main
# =============================================================================


def main():
    device = args_cli.device
    opponent_type = args_cli.opponent_type

    # --- Load player config ---
    player_cfg = _load_config(args_cli.config, device)

    # --- Create evaluation environment ---
    num_envs = args_cli.num_envs
    vec_env, opponent_wrapper = _make_eval_env(
        player_cfg.env, num_envs, args_cli.episode_length_s, opponent_type=opponent_type
    )
    obs_space = vec_env.observation_space
    act_space = vec_env.action_space

    # --- Load player agent ---
    print(f"[INFO] Loading player checkpoint: {args_cli.checkpoint}")
    player = _load_dreamer_agent(
        player_cfg.model, obs_space, act_space, args_cli.checkpoint, device
    )

    # --- Load opponent agent ---
    print(f"[INFO] Loading {opponent_type} opponent checkpoint: {args_cli.opponent_checkpoint}")
    if opponent_type == "dreamer":
        opp_config_path = args_cli.opponent_config or args_cli.config
        opp_cfg = _load_config(opp_config_path, device)
        opp_agent = _load_dreamer_agent(
            opp_cfg.model, obs_space, act_space, args_cli.opponent_checkpoint, device
        )
        opponent_wrapper.set_opponent(opp_agent)
        del opp_agent  # Wrapper deep-copied the needed modules.
    elif opponent_type == "ppo":
        ppo_opponent = _load_ppo_opponent(
            args_cli.opponent_config, args_cli.opponent_checkpoint, num_envs, device
        )
        opponent_wrapper.add_opponent(ppo_opponent)

    # --- Termination tracking (same as play_klask.py) ---
    term_counts = {
        "player_scored": 0,
        "opponent_scored": 0,
        "player_in_goal": 0,
        "opponent_in_goal": 0,
        "time_expired": 0,
    }
    TERM_NAME_MAP = {
        "goal_scored": "player_scored",
        "goal_conceded": "opponent_scored",
        "player_in_goal": "player_in_goal",
        "opponent_in_goal": "opponent_in_goal",
        "time_out": "time_expired",
    }
    term_manager = vec_env._env.unwrapped.termination_manager
    total_games = 0

    # --- Game loop ---
    pbar = tqdm(total=args_cli.num_games, desc="Games")

    vec_env.reset()
    done = torch.ones(num_envs, dtype=torch.bool, device=device)
    agent_state = player.get_initial_state(num_envs)
    act = agent_state["prev_action"].clone()

    while total_games < args_cli.num_games and simulation_app.is_running():
        with torch.inference_mode():
            # Step env (opponent actions generated inside the opponent wrapper).
            trans, done = vec_env.step(act.detach(), done.detach())

            # Player agent inference (deterministic).
            act, agent_state = player.act(trans, agent_state, eval=True)

            # Track terminations.
            if done.any():
                num_done = int(done.sum().item())
                total_games += num_done

                done_mask = done.bool().to(term_manager._term_dones.device)
                for i, term_name in enumerate(term_manager._term_names):
                    if term_name in TERM_NAME_MAP:
                        count_key = TERM_NAME_MAP[term_name]
                        term_counts[count_key] += int(
                            term_manager._term_dones[done_mask, i].sum().item()
                        )

                # Update tqdm with live stats.
                pbar.update(min(num_done, args_cli.num_games - pbar.n))
                p_wins = term_counts["player_scored"] + term_counts["opponent_in_goal"]
                o_wins = term_counts["opponent_scored"] + term_counts["player_in_goal"]
                pbar.set_postfix(
                    P_wins=p_wins,
                    O_wins=o_wins,
                    Draws=term_counts["time_expired"],
                    P_wr=f"{p_wins / total_games * 100:.1f}%",
                )

    pbar.close()

    # --- Print and save results ---
    player_wins = term_counts["player_scored"] + term_counts["opponent_in_goal"]
    opponent_wins = term_counts["opponent_scored"] + term_counts["player_in_goal"]
    draws = term_counts["time_expired"]

    summary_lines = [
        "\n" + "=" * 50,
        "  HEAD-TO-HEAD EVALUATION RESULTS",
        "=" * 50,
        f"  Player checkpoint  : {args_cli.checkpoint}",
        f"  Opponent type      : {opponent_type}",
        f"  Opponent checkpoint: {args_cli.opponent_checkpoint}",
        f"  Total games played : {total_games}",
        "-" * 50,
        f"  Player scored (goal_scored)      : {term_counts['player_scored']}",
        f"  Opponent scored (goal_conceded)   : {term_counts['opponent_scored']}",
        f"  Player fell in goal (player_in)   : {term_counts['player_in_goal']}",
        f"  Opponent fell in goal (opp_in)    : {term_counts['opponent_in_goal']}",
        f"  Time expired (time_out)           : {term_counts['time_expired']}",
        "-" * 50,
        f"  Player wins  : {player_wins}",
        f"  Opponent wins: {opponent_wins}",
        f"  Draws        : {draws}",
    ]
    if total_games > 0:
        summary_lines.append(f"  Player win rate: {player_wins / total_games * 100:.1f}%")
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

    # Cleanup.
    vec_env._env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
