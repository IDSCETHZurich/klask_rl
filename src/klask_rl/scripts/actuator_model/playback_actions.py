"""Playback recorded action trajectories in an Isaac Lab environment.

Loads all .npz trajectory files from a directory, replays them in simulation
with the environment reset to each trajectory's initial state, and saves the
resulting sim observations alongside the real ones for comparison.

Usage
-----
Basic playback (actuator model enabled by default):

    python playback_actions.py --trajectory_dir scripts/actuator_model/data/real_traj

Disable the actuator model wrapper:

    python playback_actions.py --trajectory_dir scripts/actuator_model/data/real_traj --no_actuator_model

Disable collision avoidance wrapper:
    python playback_actions.py --trajectory_dir scripts/actuator_model/data/real_traj --no_collision_avoidance

Record a video of the first trajectory:

    python playback_actions.py --trajectory_dir scripts/actuator_model/data/real_traj --video

Output
------
For each input file ``<name>.npz`` a ``sim2<name>.npz`` is written to
the same directory as the input files, containing:
  - ``actions``           - the original actions
  - ``observations_real`` - the original real observations
  - ``observations_sim``  - the observations produced by the simulator
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

TASK = "Klask-Rl-v0"

parser = argparse.ArgumentParser(description="Playback recorded action trajectories in an Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--trajectory_dir", type=str, required=True, help="Directory containing .npz trajectory files.")
parser.add_argument(
    "--no_actuator_model",
    action="store_false",
    dest="actuator_model",
    default=True,
    help="Disable the actuator model wrapper.",
)
parser.add_argument(
    "--no_collision_avoidance", action="store_false", dest="collision_avoidance", default=True, help="Disable the collision avoidance wrapper."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg
from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    KlaskRlCollisionAvoidanceWrapper,
    configure_domain_randomization,
)
from tqdm import tqdm


def _set_state_range(term_cfg, pos, vel):
    term_cfg.params["position_range"] = (pos, pos)
    term_cfg.params["velocity_range"] = (vel, vel)


def initialize_sim_state(event_manager, ball_state, peg_1_state, peg_2_state):
    """Update the live event manager's reset ranges to match the initial state of a trajectory.

    Must be called after env creation (uses event_manager.get_term_cfg to reach the
    deep-copied term configs that are actually used at reset time).
    """
    em = event_manager
    _set_state_range(em.get_term_cfg("reset_x_position_peg_1"), peg_1_state[0, 0], peg_1_state[0, 2])
    _set_state_range(em.get_term_cfg("reset_y_position_peg_1"), peg_1_state[0, 1], peg_1_state[0, 3])
    _set_state_range(em.get_term_cfg("reset_x_position_peg_2"), peg_2_state[0, 0], peg_2_state[0, 2])
    _set_state_range(em.get_term_cfg("reset_y_position_peg_2"), peg_2_state[0, 1], peg_2_state[0, 3])

    ball = em.get_term_cfg("reset_ball_position")
    ball.params["pose_range"]["x"] = (ball_state[0, 0], ball_state[0, 0])
    ball.params["pose_range"]["y"] = (ball_state[0, 1], ball_state[0, 1])
    ball.params["velocity_range"]["x"] = (ball_state[0, 2], ball_state[0, 2])
    ball.params["velocity_range"]["y"] = (ball_state[0, 3], ball_state[0, 3])


def main():
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )

    # Disable domain randomization events (they have zero-value placeholders that
    # would zero out actuator damping if left active). Must be called before gym.make().
    configure_domain_randomization(env_cfg, None)

    # Disable all terminations — playback always runs for the full trajectory length.
    for term in vars(env_cfg.terminations):
        if not term.startswith("_"):
            setattr(env_cfg.terminations, term, None)

    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = args_cli.trajectory_dir
    os.makedirs(log_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args_cli.trajectory_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {args_cli.trajectory_dir}")
    print(f"[INFO]: Found {len(files)} trajectory file(s) in {args_cli.trajectory_dir}")

    if args_cli.video:
        first_traj = np.load(files[0])
        video_kwargs = {
            "video_folder": log_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": len(first_traj["actions"]),
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if args_cli.actuator_model:
        env = ActuatorModelWrapper(env)
    if args_cli.collision_avoidance:
        env = KlaskRlCollisionAvoidanceWrapper(env)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    event_manager = env.unwrapped.event_manager

    with torch.inference_mode():
        for filepath in tqdm(files, desc="trajectories"):
            trajectories = np.load(filepath)
            peg_1_state = trajectories["obs"][:, :4]
            peg_2_state = trajectories["obs"][:, 4:8]
            ball_state = trajectories["obs"][:, 8:12]

            initialize_sim_state(event_manager, ball_state, peg_1_state, peg_2_state)
            obs, _ = env.reset()

            obs_buffer = []
            for step_actions in tqdm(trajectories["actions"], desc="steps", leave=False):
                step_actions = torch.from_numpy(step_actions).to(env.unwrapped.device)
                obs_buffer.append(obs["policy"][0].cpu().numpy())
                actions_input = step_actions.unsqueeze(0)
                if actions_input.shape[1] == 2:
                    actions_input = torch.cat([actions_input, torch.zeros(1, 2, device=actions_input.device)], dim=1)
                obs, _, _, _, _ = env.step(actions_input)

            np.savez(
                os.path.join(log_dir, f"sim2{os.path.basename(filepath)}"),
                actions=trajectories["actions"],
                observations_real=trajectories["obs"],
                observations_sim=np.array(obs_buffer),
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
