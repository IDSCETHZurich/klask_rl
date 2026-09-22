"""Dreamer/FastSAC checkpoint loading, usable independently of Isaac Lab."""

import pathlib
import sys

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent)
_PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
for _candidate in (
    pathlib.Path(_SCRIPT_DIR) / "r2dreamer",
    _PROJECT_DIR.parent.parent / "third_party" / "r2dreamer",
):
    if (_candidate / "dreamer.py").is_file():
        sys.path.insert(0, str(_candidate))
        break


def _load_fast_sac_opponent(checkpoint_path, device):
    """Load a FastSAC opponent agent from a train_fast_sac_isaaclab.py checkpoint.

    Returns an object exposing the small interface expected by
    KlaskRlFastSACOpponentWrapper: ``has_batch_dimension``, ``obs_to_torch``,
    and ``get_action(obs, is_deterministic)``. Actor hyperparameters come from
    the checkpoint's embedded ``config`` dict (saved by fast_sac_utils.save_params).
    Observations are normalized with the saved EmpiricalNormalization state.
    Inference is deterministic when ``is_deterministic=True`` (mode of the
    tanh-squashed Gaussian), matching the PPO opponent's behavior.
    """
    # The fast_sac package lives under scripts/fast_sac/klask_her — make sure
    # it's importable when this script is run from elsewhere.
    fast_sac_pkg_dir = pathlib.Path(_SCRIPT_DIR).parent / "fast_sac" / "klask_her"
    if not fast_sac_pkg_dir.exists():
        fast_sac_pkg_dir = _PROJECT_DIR.parent.parent / "third_party" / "fast_sac"
    if fast_sac_pkg_dir.exists() and str(fast_sac_pkg_dir) not in sys.path:
        sys.path.append(str(fast_sac_pkg_dir))

    from klask_her.agents.fast_sac import Actor
    from klask_her.agents.fast_sac_utils import EmpiricalNormalization

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}

    obs_dim = 18  # _FastSACOpponentPolicyCfg always emits 18
    act_dim = 2
    hidden_dim = int(cfg.get("actor_hidden_dim", 512))
    # train_fast_sac_isaaclab.py uses action_scale=1.0; Actor's default is 0.5.
    action_scale = float(cfg.get("action_scale", 1.0))

    actor = Actor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        action_scale=action_scale,
        device=device,
    )
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    obs_normalizer = EmpiricalNormalization(shape=obs_dim, device=device)
    obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])
    obs_normalizer.eval()
    if int(obs_normalizer.count.item()) <= 0:
        raise RuntimeError(
            f"FastSAC obs_normalizer in {checkpoint_path} has count=0 — "
            "running stats look uninitialized; cannot evaluate."
        )

    class _FastSACOpponent:
        has_batch_dimension = True

        def obs_to_torch(self, obs):
            if not torch.is_tensor(obs):
                obs = torch.as_tensor(obs, device=device)
            return obs.to(device=device, dtype=torch.float32)

        def get_action(self, obs, is_deterministic):
            norm = obs_normalizer(obs, update=False)
            return actor.explore(norm, deterministic=bool(is_deterministic))

    return _FastSACOpponent()


def _load_dreamer_agent(full_cfg, obs_space, act_space, checkpoint_path, device):
    """Create a Dreamer agent and load checkpoint weights."""
    import copy

    from dreamer import Dreamer

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
            getattr(
                getattr(full_cfg, "opponent_separation_config", None) or {},
                "imag_opponent",
                "random",
            )
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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, _unexpected = agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
    # _inference_* keys are aliases to _frozen_* in single-GPU mode — their
    # parameters are shared objects already loaded via _frozen_* keys.
    # Unexpected keys are from multi-GPU checkpoints (_inference_*_orig,
    # _imag_opp_*, compiled ._orig_mod copies) not needed for evaluation.
    real_missing = [k for k in missing if not k.startswith("_inference_")]
    if real_missing:
        raise RuntimeError(f"Missing keys in checkpoint: {real_missing}")
    agent.eval()
    return agent


def _load_ppo_opponent(config_path, checkpoint_path, num_envs, device, ppo_obs_space):
    """Load a PPO opponent agent from an rl_games checkpoint.

    Mirrors the loading pattern from play_klask.py.
    """
    import yaml
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    # Saved agent YAMLs define the full params tree, including normalization.
    if config_path is not None:
        with open(config_path, "r") as f:
            agent_cfg = yaml.safe_load(f)
    else:
        from isaaclab_tasks.utils import load_cfg_from_registry

        agent_cfg = load_cfg_from_registry("Klask-Rl-v0", "rl_games_cfg_entry_point")

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
    agent_cfg["params"]["config"]["device"] = str(device)
    agent_cfg["params"]["config"]["device_name"] = str(device)
    agent_cfg["params"]["config"]["multi_gpu"] = False

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
