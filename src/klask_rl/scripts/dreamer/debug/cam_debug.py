"""Minimal debug script to visualize the Dreamer camera output.

Launches the Klask-Rl-Dreamer-v0 environment, steps the simulation,
and streams the camera image of one env to a browser via HTTP.

Usage:
    cd /workspace/klask_rl
    python scripts/dreamer/debug/cam_debug.py
    # Then open http://localhost:8089 in your browser
"""

"""Launch Isaac Sim first."""
import os

from isaaclab.app import AppLauncher

livestream = int(os.environ.get("LIVESTREAM", 0))
app_launcher = AppLauncher({"headless": True, "enable_cameras": True, "livestream": livestream})
simulation_app = app_launcher.app

"""Rest follows after sim is ready."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import klask_rl.tasks  # noqa: F401

from klask_rl.tasks.manager_based.klask_rl.klask_rl_env_cfg import KlaskRlDreamerEnvCfg
from mjpeg_server import MJPEGServer
from isaaclab.sim import RenderCfg

# ---- Configuration ----
ENV_INDEX = 0  # which env's camera to display
NUM_ENVS = 2
STREAM_PORT = 8089
# ------------------------


def main():
    server = MJPEGServer(port=STREAM_PORT, title="Klask Camera Debug")
    server.start()

    env_cfg = KlaskRlDreamerEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")

    env = gym.make("Klask-Rl-Dreamer-v0", cfg=env_cfg)
    env.reset()

    camera = env.unwrapped.scene["camera"]
    print(f"[INFO] Streaming env {ENV_INDEX} camera to http://localhost:{STREAM_PORT}")
    server.print_url()

    count = 0
    try:
        while simulation_app.is_running():
            action = torch.zeros(
                NUM_ENVS,
                env.unwrapped.action_space.shape[-1],
                device=env.unwrapped.device,
            )
            env.step(action)

            count += 1
            if count % 10 == 0:
                rgb = camera.data.output["rgb"]  # (N, H, W, 4) RGBA uint8
                frame = rgb[ENV_INDEX, :, :, :3].cpu().numpy()
                server.update_frame(frame)
    except KeyboardInterrupt:
        print("Exiting.")

    server.stop()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
