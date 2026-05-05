# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug a trained Klask RL-Games checkpoint under a fixed scenario.

Pins the opponent peg and ball at fixed positions, sets the player's initial
position and velocity, then rolls the policy out for ``N_STEPS`` steps with no
episode resets and plots player position, velocity, and policy actions.

Mirrors the wrapper stack from train_klask.py (incl. VelocityScaleWrapper +
KlaskRlCollisionAvoidanceWrapper + action-manager scale reset) so the policy
sees the same input/output scaling it was trained with. Strips stochastic /
training-only wrappers (domain randomization, observation noise, initialization,
curriculum) so the rollout is deterministic.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Debug an RL-Games Klask checkpoint under a fixed scenario.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--task", type=str, default="Klask-Rl-v0", help="Name of the task.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/rl_games/klask/demo_agents/best_one/klask.pth",
    help="Path to model checkpoint.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument(
    "--config",
    type=str,
    default="logs/rl_games/klask/demo_agents/best_one/agent.yaml",
    help="config.yaml file, rl_games_cfg_entry_point used when not provided",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force a single environment for the debug rollout.
args_cli.num_envs = 1

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
import yaml
from gymnasium import Wrapper
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)
from klask_rl.assets.robots.klask_params import KLASK_PARAMS
from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    ActionHistoryWrapper,
    KlaskRlCollisionAvoidanceWrapper,
    OpponentActionWrapper,
    VelocityScaleWrapper,
)
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

# ----------------------------------------------------------------------------
# Hardcoded debug scenario.
#
# Coordinate system (env-local, world frame):
#   x in [-0.16, 0.16]            (slider_to_peg_*)
#   player y (Peg_1) in [-0.21, -0.02]
#   opponent y (Peg_2) in [+0.02, +0.21]
#   ball z = 0.032 (board surface; must NOT be changed)
# Edit these to debug different scenarios.
# ----------------------------------------------------------------------------
PLAYER_INIT_POS = (0.00, -0.1)  # x, y  (joint pos of slider_to_peg_1, ground_to_slider_1)
PLAYER_INIT_VEL = (0.0, 0.0)  # x, y  joint velocities (m/s)
OPPONENT_FIXED_POS = (0.079, 0.081)  # x, y  (joint pos of slider_to_peg_2, ground_to_slider_2)
BALL_FIXED_POS = (-0.084, 0.042)  # x, y  ball world XY; z is fixed below
BALL_Z = 0.032  # ball radius — matches event_cfg.reset_ball_position
N_STEPS = 500


class ZeroOpponentWrapper(Wrapper):
    """Inject zero opponent actions, mirroring KlaskRlRandomOpponentWrapper's shape contract.

    Replaces KlaskRlRandomOpponentWrapper used at train/play time. We pin the
    opponent peg by writing its joint state directly each step, but we also
    feed zero velocity commands so the simulator's PD actuator doesn't fight
    the pin.
    """

    def __init__(self, env):
        super().__init__(env)
        if hasattr(self.env.unwrapped, "single_action_space"):
            original_space = self.env.unwrapped.single_action_space
            if hasattr(original_space, "shape") and original_space.shape[0] == 4:
                self.env.unwrapped._klask_original_single_action_space = original_space
                self.env.unwrapped.single_action_space = gym.spaces.Box(
                    low=original_space.low[:2],
                    high=original_space.high[:2],
                    dtype=original_space.dtype,
                )

    def step(self, actions, *args, **kwargs):
        if actions.size() == torch.Size([self.env.unwrapped.num_envs, 4]):
            actions = actions.clone()
            actions[:, 2:] = 0.0
        else:
            zero_opp = torch.zeros_like(actions)
            actions = torch.cat([actions, zero_opp], dim=1)
        return self.env.step(actions, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)


def _pin_opponent_and_ball(env_unwrapped, klask: Articulation, ball: RigidObject, opp_joint_ids, env_ids, device):
    """Write opponent joint state and ball root state to their fixed targets."""
    opp_pos = torch.tensor([[OPPONENT_FIXED_POS[0], OPPONENT_FIXED_POS[1]]], device=device)
    opp_vel = torch.zeros((1, 2), device=device)
    klask.write_joint_state_to_sim(opp_pos, opp_vel, env_ids=env_ids, joint_ids=opp_joint_ids)

    # Match reset_root_state_uniform's frame convention: position is env-local + env_origins.
    origin = env_unwrapped.scene.env_origins[env_ids]
    default_quat = ball.data.default_root_state[env_ids, 3:7]
    pose = torch.zeros((1, 7), device=device)
    pose[0, 0] = BALL_FIXED_POS[0]
    pose[0, 1] = BALL_FIXED_POS[1]
    pose[0, 2] = BALL_Z
    pose[:, 0:3] = pose[:, 0:3] + origin
    pose[:, 3:7] = default_quat
    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(torch.zeros((1, 6), device=device), env_ids=env_ids)


def main():
    """Debug rollout entry point."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    if args_cli.config is not None:
        with open(args_cli.config, "r") as file:
            config = yaml.safe_load(file)
        agent_cfg.update(config)

    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    rl_device = args_cli.device
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # Disable any termination terms that the agent_cfg flags as off (mirrors play/train).
    if "terminations" in agent_cfg and hasattr(env_cfg, "terminations"):
        for term, active in agent_cfg["terminations"].items():
            if not active and hasattr(env_cfg.terminations, term):
                setattr(env_cfg.terminations, term, None)

    # NOTE: deliberately skipping configure_domain_randomization here so the
    # rollout is deterministic.

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ---- Wrapper chain mirrors train_klask.py:222-290 (sans noise/init/curriculum) ----
    env_block = agent_cfg.get("env", {})

    # If VelocityScaleWrapper will own scaling, neutralise the action manager's scale.
    max_velocity = env_block.get("max_velocity")
    if max_velocity is not None:
        for term in env.unwrapped.action_manager._terms.values():
            term._scale = 1.0

    # 0. Collision avoidance (innermost; world-frame m/s clipping).
    collision_cfg = env_block.get("collision_avoidance")
    enable_collision = (isinstance(collision_cfg, dict) and collision_cfg.get("enable", False)) or (
        collision_cfg is True
    )
    if enable_collision:
        env = KlaskRlCollisionAvoidanceWrapper(env, max_vel=float(max_velocity) if max_velocity is not None else 0.2)

    # 1. Opponent action frame transform (ego -> world; negates opponent dims).
    env = OpponentActionWrapper(env)

    # 2. Actuator model — supports both old (bool) and new (dict) yaml formats.
    actuator_cfg = env_block.get("actuator_model")
    if isinstance(actuator_cfg, dict) and actuator_cfg.get("enable", False):
        env = ActuatorModelWrapper(env, device=args_cli.device, model_file=actuator_cfg.get("checkpoint"))
    elif actuator_cfg is True:
        env = ActuatorModelWrapper(env, device=args_cli.device)

    # 3. Velocity scaling (only if the agent was trained with it).
    if max_velocity is not None:
        max_acceleration = env_block.get("max_acceleration")
        _act_space = env.unwrapped.single_action_space
        env = VelocityScaleWrapper(
            env,
            max_velocity=float(max_velocity),
            max_acceleration=None if max_acceleration is None else float(max_acceleration),
            num_envs=int(env.unwrapped.num_envs),
            action_dim=int(_act_space.shape[0]),
            device=env.unwrapped.device,
        )

    # 4. Action history (rare; default disabled via KLASK_PARAMS).
    if KLASK_PARAMS["action_history"] > 0:
        env = ActionHistoryWrapper(env, history_length=KLASK_PARAMS["action_history"])

    # 5. Opponent: zero actions instead of random or learned policy.
    env = ZeroOpponentWrapper(env)

    # 6. rl-games adapter (outermost).
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs=clip_obs, clip_actions=clip_actions)

    # Register the env with rl-games (single-agent path; no self-play even if
    # the agent was trained with self_play because we only roll out the player).
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
    )

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # Force off self_play for player creation (we don't load an opponent).
    agent_cfg["params"]["config"]["self_play"] = False
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()

    # Map checkpoints saved on multi-GPU machines onto our device.
    def _safe_load_mapped(filename):
        return torch_ext.safe_filesystem_op(torch.load, filename, map_location=args_cli.device, weights_only=False)

    torch_ext.safe_load = _safe_load_mapped

    agent.restore(resume_path)
    agent.reset()
    agent.device = torch.device(args_cli.device)
    agent.model.to(args_cli.device)
    agent.actions_low = agent.actions_low.to(args_cli.device)
    agent.actions_high = agent.actions_high.to(args_cli.device)

    # Reset and obtain handles to scene assets for state pinning.
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    klask: Articulation = env.unwrapped.scene["klask"]
    ball: RigidObject = env.unwrapped.scene["ball"]
    device = env.unwrapped.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    # Resolve joint indices once. find_joints returns (ids, names) in match order.
    player_joint_ids, _ = klask.find_joints(["slider_to_peg_1", "ground_to_slider_1"])
    opp_joint_ids, _ = klask.find_joints(["slider_to_peg_2", "ground_to_slider_2"])

    # Apply initial conditions: player set once; opponent + ball pinned each step below.
    # Wrap in inference_mode because the simulator's tensors may already be
    # inference tensors (set during env.reset).
    player_pos = torch.tensor([[PLAYER_INIT_POS[0], PLAYER_INIT_POS[1]]], device=device)
    player_vel = torch.tensor([[PLAYER_INIT_VEL[0], PLAYER_INIT_VEL[1]]], device=device)
    with torch.inference_mode():
        klask.write_joint_state_to_sim(player_pos, player_vel, env_ids=env_ids, joint_ids=player_joint_ids)
        _pin_opponent_and_ball(env.unwrapped, klask, ball, opp_joint_ids, env_ids, device)
        obs_buf = env.unwrapped.observation_manager.compute()
        obs = obs_buf["policy"] if isinstance(obs_buf, dict) else obs_buf

    # ---- Capture loop ----
    captured_obs_full = np.zeros((N_STEPS, obs.shape[-1]), dtype=np.float32)
    captured_act = None  # allocated on first iteration once action_dim is known

    for t in range(N_STEPS):
        if not simulation_app.is_running():
            break

        with torch.inference_mode():
            obs_torch = agent.obs_to_torch(obs)
            action = agent.get_action(obs_torch, is_deterministic=True)
            obs, _rew, _dones, _info = env.step(action)
            if isinstance(obs, dict):
                obs = obs["obs"]

            # Capture the obs that produced this step's action (pre-step state).
            captured_obs_full[t] = obs_torch.detach().cpu().numpy()[0]
            act_np = action.detach().cpu().numpy()[0]
            if captured_act is None:
                captured_act = np.zeros((N_STEPS, act_np.shape[0]), dtype=np.float32)
            captured_act[t] = act_np

            # Re-pin opponent and ball, then refresh obs so the next iteration
            # feeds the agent a state with both at their fixed targets.
            # Must stay inside inference_mode: env.step touched simulator
            # tensors as inference tensors, so out-of-mode in-place writes fail.
            _pin_opponent_and_ball(env.unwrapped, klask, ball, opp_joint_ids, env_ids, device)
            obs_buf = env.unwrapped.observation_manager.compute()
            obs = obs_buf["policy"] if isinstance(obs_buf, dict) else obs_buf

    env.close()

    # ---- Sanity prints ----
    init_pos_err = np.linalg.norm(captured_obs_full[0, 0:2] - np.array(PLAYER_INIT_POS))
    opp_pos_drift = float(np.max(np.linalg.norm(captured_obs_full[:, 4:6] - np.array(OPPONENT_FIXED_POS), axis=1)))
    ball_pos_drift = float(np.max(np.linalg.norm(captured_obs_full[:, 8:10] - np.array(BALL_FIXED_POS), axis=1)))
    print(f"[INFO] step 0 player pos error vs init: {init_pos_err:.4f} m")
    print(f"[INFO] max opponent pos drift over rollout: {opp_pos_drift:.4f} m")
    print(f"[INFO] max ball pos drift over rollout: {ball_pos_drift:.4f} m")

    # ---- Save raw arrays (plotting lives in debug_sim2real_plot.py) ----
    out_dir = os.path.join(log_root_path, "debug_sim2real")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_npz = os.path.join(out_dir, f"debug_{timestamp}.npz")
    np.savez(
        out_npz,
        obs_full=captured_obs_full,
        action=captured_act,
        player_init_pos=np.array(PLAYER_INIT_POS),
        player_init_vel=np.array(PLAYER_INIT_VEL),
        opponent_fixed_pos=np.array(OPPONENT_FIXED_POS),
        ball_fixed_pos=np.array(BALL_FIXED_POS),
        n_steps=np.array(N_STEPS),
    )
    print(f"[INFO] arrays saved to: {out_npz}")


if __name__ == "__main__":
    main()
    simulation_app.close()
