# =============================================================================
# Phase 1: AppLauncher must run before any IsaacLab / USD / warp imports.
# =============================================================================

import argparse
import pathlib
import sys

# Auto-detect vision env from CLI and enable cameras before AppLauncher
# parses argv, so the user doesn't have to pass --enable_cameras manually.
_vision = False
for _arg in sys.argv[1:]:
    if _arg.startswith("env=") and "vision" in _arg.split("=", 1)[1]:
        _vision = True
        break

# The KLASK Dreamer env always uses cameras (cnn_keys: "image"), so we
# unconditionally enable them.  This avoids having to pass --enable_cameras
# on every invocation.
if "--enable_cameras" not in sys.argv:
    sys.argv.insert(1, "--enable_cameras")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train r2dreamer with IsaacLab.")
AppLauncher.add_app_launcher_args(parser)
# Capture only the args AppLauncher understands; pass the rest to Hydra.
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# Phase 2: Everything else — safe to import now that the sim is running.
# =============================================================================

import atexit
import signal
import warnings

import hydra
import torch
from omegaconf import OmegaConf

# Isaac Sim may override SIGINT; restore Python's default so Ctrl-C works.
signal.signal(signal.SIGINT, signal.default_int_handler)

# Absolute path to this script's directory — used by Hydra searchpath via
# the ${script_dir:} OmegaConf resolver so configs resolve regardless of cwd.
_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent)

# Register a resolver so config.yaml can write ${script_dir:} to get an
# absolute path that is independent of the current working directory.
OmegaConf.register_new_resolver("script_dir", lambda: _SCRIPT_DIR, use_cache=True)

sys.path.append(str(pathlib.Path(_SCRIPT_DIR) / "r2dreamer"))
sys.path.append(_SCRIPT_DIR)
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

# Register IsaacLab task environments (triggers gymnasium gym.register calls).
import isaaclab_tasks  # noqa: F401
import klask_rl.tasks  # noqa: F401
import tools
from buffer import Buffer
from dreamer import Dreamer

# Self-play wrapper (lives outside the r2dreamer submodule)
from dreamer_self_play import DreamerSelfPlayWrapper
from envs import make_envs
from envs.isaaclab import IsaacLabVecEnv
from gymnasium import Wrapper
from isaaclab.sim import RenderCfg
from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.tasks.manager_based.klask_rl.utils_manager_based import set_terminations
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    CurriculumWrapper,
    KlaskRlRandomOpponentWrapper,
    OpponentActionWrapper,
)
from trainer import OnlineTrainer


class EpisodeMetricsWrapper(Wrapper):
    """Captures episode-level metrics from the IsaacLab info dict and logs them.

    IsaacLab populates ``info["log"]`` with per-step scalars such as
    ``Episode_Termination/goal_scored`` (fraction of envs where that
    termination fired).  The generic ``IsaacLabVecEnv`` adapter discards
    the info dict, so this wrapper sits just below it and forwards
    the metrics to a logger.

    Metrics are accumulated over a sliding window of recent episodes
    (default 100) so that logged values represent a meaningful rate
    rather than a single episode's binary indicator.

    Call :meth:`set_logger` after construction to enable logging.
    """

    def __init__(self, env, window_size: int = 100):
        super().__init__(env)
        self._logger = None
        self._window_size = window_size
        # Per-episode accumulators: key -> running sum for the current episode
        self._ep_sums: dict[str, float] = {}
        self._ep_counts: dict[str, int] = {}
        # Sliding window of completed episode means: key -> deque of floats
        from collections import deque

        self._history: dict[str, deque] = {}
        self._deque_factory = lambda: deque(maxlen=self._window_size)

    def set_logger(self, logger):
        """Attach a :class:`tools.Logger` for autonomous metric logging."""
        self._logger = logger

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        if self._logger is not None:
            # Accumulate per-step values for the current episode
            for key, val in info.get("log", {}).items():
                if isinstance(val, torch.Tensor):
                    val = val.item() if val.ndim == 0 else val.mean().item()
                val = float(val)
                self._ep_sums[key] = self._ep_sums.get(key, 0.0) + val
                self._ep_counts[key] = self._ep_counts.get(key, 0) + 1

            # On episode end, push the per-episode mean into the sliding
            # window and log the windowed average.
            done = terminated or truncated
            # Handle both scalar bool and tensor
            if isinstance(done, torch.Tensor):
                done = done.any().item()
            if done:
                for key in list(self._ep_sums.keys()):
                    ep_mean = self._ep_sums[key] / max(self._ep_counts[key], 1)
                    if key not in self._history:
                        self._history[key] = self._deque_factory()
                    self._history[key].append(ep_mean)
                    # Log the windowed average across recent episodes
                    window = self._history[key]
                    windowed_avg = sum(window) / len(window)
                    key = key.lower()
                    if key.startswith("episode_"):
                        key = key.replace("_", "/", 1)
                    else:
                        key = f"episode/{key}"
                    self._logger.scalar(key, windowed_avg)
                # Reset per-episode accumulators
                self._ep_sums.clear()
                self._ep_counts.clear()

        return obs, rew, terminated, truncated, info


class RewardWeightLogWrapper(Wrapper):
    """Logs reward term weights from the env's ``reward_manager`` on each step.

    Call :meth:`set_logger` after construction to enable logging.
    """

    def __init__(self, env):
        super().__init__(env)
        self._logger = None

    def set_logger(self, logger):
        """Attach a :class:`tools.Logger` for autonomous metric logging."""
        self._logger = logger

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        if self._logger is not None:
            rm = self.env.unwrapped.reward_manager
            for term, cfg in zip(rm.active_terms, rm._term_cfgs):
                w = cfg.weight
                val = w.item() if isinstance(w, torch.Tensor) else float(w)
                self._logger.scalar(f"rewards/weights/{term}", val)
        return obs, rew, terminated, truncated, info


# =============================================================================
# Task registry — all task-specific knowledge lives here, not in envs/__init__.py
# =============================================================================


# Global reference to the self-play wrapper (set during env construction,
# used later to initialise / update the opponent from the training agent).
_self_play_wrapper = None


def _make_env(config, gym_id, render_mode=None, trainer_steps=None, self_play=False, self_play_config=None):
    """Construct a GPU-resident IsaacLab env with KLASK-specific wrappers.

    Applies (in order):
      1. Bounded action space (max_velocity)
      2. ActuatorModelWrapper (if config.actuator_model)
      3. CurriculumWrapper (if config.rewards)
      4. Termination filtering (if config.terminations)
      5. Opponent wrapper:
         - DreamerSelfPlayWrapper if self_play=True
         - KlaskRlRandomOpponentWrapper otherwise
      6. EpisodeMetricsWrapper (episode termination logging)
      7. RewardWeightLogWrapper (reward weight logging)
    """
    global _self_play_wrapper
    import importlib

    import gymnasium as gym
    import numpy as np

    env_cfg_entry = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    if isinstance(env_cfg_entry, str):
        module_name, class_name = env_cfg_entry.rsplit(":", 1)
        env_cfg_class = getattr(importlib.import_module(module_name), class_name)
    else:
        env_cfg_class = env_cfg_entry

    env_cfg = env_cfg_class()

    sim_dt = getattr(config, "sim_dt", None)
    if sim_dt is not None:
        env_cfg.sim.dt = float(sim_dt)

    env_cfg.scene.num_envs = int(config.env_num)
    env_cfg.decimation = int(config.action_repeat)
    env_cfg.seed = int(config.seed)
    env_cfg.episode_length_s = config.episode_length_s

    # IsaacLab defaults to DLSS which smooths the image significantly
    # so we disable the antialiasing for a more pixelated (and hence more realistic) image.
    env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")

    # --- Create the base gymnasium env ---
    isaac_env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode)

    # --- 1a. Opponent action frame transform (innermost) ---
    isaac_env = OpponentActionWrapper(isaac_env)

    # --- 1. Set bounded action space ---
    max_velocity = getattr(config, "max_velocity", None)
    if max_velocity is not None:
        action_dim = isaac_env.unwrapped.single_action_space.shape[-1]
        isaac_env.unwrapped.single_action_space = gym.spaces.Box(
            low=-float(max_velocity),
            high=float(max_velocity),
            shape=(action_dim,),
            dtype=np.float32,
        )
        isaac_env.unwrapped.action_space = gym.vector.utils.batch_space(
            isaac_env.unwrapped.single_action_space, isaac_env.unwrapped.num_envs
        )

    # --- 2. Actuator model wrapper ---
    if getattr(config, "actuator_model", False):
        isaac_env = ActuatorModelWrapper(isaac_env)

    # --- 3. Reward curriculum wrapper ---
    rewards_cfg = getattr(config, "rewards", None)
    if rewards_cfg is not None:
        if OmegaConf.is_config(rewards_cfg):
            rewards_dict = OmegaConf.to_container(rewards_cfg, resolve=True)
        else:
            rewards_dict = dict(rewards_cfg)
        isaac_env = CurriculumWrapper(isaac_env, rewards_dict)

    # --- 4. Termination filtering ---
    terminations_cfg = getattr(config, "terminations", None)
    if terminations_cfg is not None:
        if OmegaConf.is_config(terminations_cfg):
            term_dict = OmegaConf.to_container(terminations_cfg, resolve=True)
        else:
            term_dict = dict(terminations_cfg)
        set_terminations(isaac_env, term_dict)

    # --- 5. Opponent wrapper ---
    if self_play:
        sp_cfg = self_play_config or {}
        _self_play_wrapper = DreamerSelfPlayWrapper(
            isaac_env,
            update_score=float(sp_cfg.get("update_score", 0.7)),
            games_to_track=int(sp_cfg.get("games_to_track", 4096)),
            compile=bool(sp_cfg.get("compile", False)),
        )
        isaac_env = _self_play_wrapper
    else:
        isaac_env = KlaskRlRandomOpponentWrapper(isaac_env)

    # --- 6. Episode metrics capture ---
    isaac_env = EpisodeMetricsWrapper(isaac_env)

    # --- 7. Reward weight logging (outermost gymnasium wrapper) ---
    isaac_env = RewardWeightLogWrapper(isaac_env)

    # Wrap in the r2dreamer IsaacLabVecEnv adapter
    return IsaacLabVecEnv(isaac_env, simulation_app=simulation_app)


# =============================================================================
# Main
# =============================================================================


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    # env.task follows the codebase convention: "isaaclab_<task_name>"
    # e.g. "isaaclab_cartpole_balance"
    full_task = config.env.task  # e.g. "isaaclab_cartpole_balance"
    _, task_name = full_task.split("_", 1)  # e.g. "cartpole_balance"

    render_mode = "rgb_array" if _vision else None
    trainer_steps = getattr(config.trainer, "steps", None)
    self_play = getattr(config, "self_play", False)

    # Attach self_play_config to env config so _make_env can read it.
    sp_cfg = (
        OmegaConf.to_container(config.get("self_play_config", OmegaConf.create({})), resolve=True) if self_play else {}
    )
    # Mirror the model compile flag so the opponent is also compiled when the
    # training agent is.
    if self_play:
        sp_cfg.setdefault("compile", bool(getattr(config.model, "compile", False)))

    vec_env = _make_env(
        config.env,
        task_name,
        render_mode,
        trainer_steps=trainer_steps,
        self_play=self_play,
        self_play_config=sp_cfg,
    )

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    wandb_cfg = {
        "project": getattr(config, "wandb_project", "r2dreamer-isaaclab"),
        "name": getattr(config, "wandb_name", f"dreamer_{task_name}"),
        "dir": str(logdir),
    }
    logger = tools.Logger(
        logdir,
        backends=[
            tools.JSONLBackend(logdir),
            # tools.TensorBoardBackend(logdir),
            tools.WandbBackend(wandb_cfg),
        ],
    )
    logger.log_hydra_config(config)

    # Attach the logger to logging wrappers so they log autonomously.
    env = vec_env._env
    while isinstance(env, Wrapper):
        if hasattr(env, "set_logger"):
            env.set_logger(logger)
        env = env.env

    replay_buffer = Buffer(config.buffer)

    print("Create env.")
    # OmegaConf configs are read-only; use a plain object to pass the env through.
    env_config = OmegaConf.to_container(config.env, resolve=True)
    env_config = type("EnvConfig", (), env_config)()
    env_config.isaac_vec_env = vec_env
    train_envs, eval_envs, obs_space, act_space = make_envs(env_config)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    # Resume from checkpoint if one exists in the logdir.
    _resume_step = 0
    checkpoint_path = logdir / "latest.pt"
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        _resume_step = checkpoint.get("step", 0)
        # Restore curriculum step so reward weight schedules continue correctly.
        _curriculum_step = checkpoint.get("curriculum_step", 0)
        if _curriculum_step > 0:
            env = vec_env._env
            while isinstance(env, Wrapper):
                if isinstance(env, CurriculumWrapper):
                    env._step = _curriculum_step
                    break
                env = env.env
        # Restore LR scheduler state so warmup doesn't restart.
        if "scheduler_state_dict" in checkpoint:
            agent._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Restore GradScaler state so AMP scale factor continues correctly.
        if "scaler_state_dict" in checkpoint:
            agent._scaler.load_state_dict(checkpoint["scaler_state_dict"])
        # Restore slow target update counter.
        if "slow_value_updates" in checkpoint:
            agent._slow_value_updates = checkpoint["slow_value_updates"]
        # Restore DreamerPro EMA update counter (only present with dreamerpro config).
        if "ema_updates" in checkpoint and hasattr(agent, "_ema_updates"):
            agent._ema_updates = checkpoint["ema_updates"]
        print(f"  Restored agent weights, optimizer states, step={_resume_step}, curriculum_step={_curriculum_step}")

    # Initialise self-play opponent from the (randomly initialised) agent.
    if _self_play_wrapper is not None:
        _self_play_wrapper.set_opponent(agent)
        # Restore score buffer from checkpoint so mean_score continues
        # correctly instead of being diluted by 4096 pre-filled zeros.
        if checkpoint_path.exists():
            _saved_scores = checkpoint.get("selfplay_score_buffer")
            if _saved_scores is not None:
                _self_play_wrapper._score_buffer.clear()
                _self_play_wrapper._score_buffer.extend(_saved_scores)
                print(f"  Restored self-play score buffer ({len(_saved_scores)} entries, mean={_self_play_wrapper.mean_score:.3f})")
        print("Self-play enabled: opponent initialised from current agent.")

    # Subclass OnlineTrainer to hook score-gated opponent updates.
    class KlaskTrainer(OnlineTrainer):
        """OnlineTrainer with self-play opponent updates.

        Reward weights and episode metrics are logged autonomously by
        :class:`RewardWeightLogWrapper` and :class:`EpisodeMetricsWrapper`.
        """

        def eval(self, agent, train_step):
            # --- Self-play opponent update ---
            if _self_play_wrapper is not None:
                _self_play_wrapper.maybe_update_opponent(
                    agent,
                    logger=self.logger,
                    train_step=train_step,
                )
            return super().eval(agent, train_step)

    def _save_checkpoint(step):
        """Save a full checkpoint at the given step.

        Writes both a numbered ``checkpoint_{step}.pt`` and overwrites
        ``latest.pt`` so that resume always picks up the newest one.
        """
        # Find curriculum step from wrapper chain.
        _curr_step = 0
        _env = vec_env._env
        while isinstance(_env, Wrapper):
            if isinstance(_env, CurriculumWrapper):
                _curr_step = _env._step
                break
            _env = _env.env
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "step": step,
            "curriculum_step": _curr_step,
            "scheduler_state_dict": agent._scheduler.state_dict(),
            "scaler_state_dict": agent._scaler.state_dict(),
            "slow_value_updates": agent._slow_value_updates,
            **({"ema_updates": agent._ema_updates} if hasattr(agent, "_ema_updates") else {}),
            **({"selfplay_score_buffer": list(_self_play_wrapper._score_buffer)} if _self_play_wrapper is not None else {}),
        }
        torch.save(items_to_save, logdir / f"checkpoint_{step}.pt")
        torch.save(items_to_save, logdir / "latest.pt")
        print(f"Checkpoint saved: {str(logdir.absolute())}/checkpoint_{step}.pt + latest.pt")

    policy_trainer = KlaskTrainer(
        config.trainer,
        replay_buffer,
        logger,
        logdir,
        train_stepper=train_envs,
        eval_stepper=eval_envs,
        initial_step=_resume_step,
        save_fn=_save_checkpoint,
    )

    exit_code = 0
    try:
        policy_trainer.begin(agent)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
        exit_code = 1
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"TRAINING CRASHED: {type(e).__name__}: {e}")
        print(f"{'='*60}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        _save_checkpoint(policy_trainer._step)

        logger.close(exit_code=exit_code)
        vec_env._env.close()
        simulation_app.close()


if __name__ == "__main__":
    # Forward only the Hydra-style args (everything after AppLauncher args).
    sys.argv = [sys.argv[0]] + hydra_args
    main()
