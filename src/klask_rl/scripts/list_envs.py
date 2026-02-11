# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the `klask_rl` extension. They start
with `Isaac` in their name.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import klask_rl.tasks  # noqa: F401
import benchmark.tasks  # noqa: F401


def main():
    """Print all environments registered in `klask_rl` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # environment name prefixes to search for
    env_keywords = ["Klask-", "Benchmark-", "CartPole-Direct-"]
    # acquire all Isaac environments names
    for task_spec in gym.registry.values():
        if any(keyword in task_spec.id for keyword in env_keywords):
            # add details to table
            try:
                table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
                # increment count
                index += 1
            except (KeyError, AttributeError):
                print(f"Warning: Skipping {task_spec.id} - missing env_cfg_entry_point")

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
