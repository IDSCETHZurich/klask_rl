import subprocess
import time
from pathlib import Path
import copy
import re
import argparse

parser = argparse.ArgumentParser(description="Pool training")

parser.add_argument("--pool_name", type=str, default="same_config", help="Name of the user")

args = parser.parse_args()
# List of config files to train with
configs = [
    # "scripts/rl_games/config/klask_config_1.yaml",
    # "scripts/rl_games/config/klask_config_2.yaml",
    "scripts/rl_games/config/klask_config_3.yaml",
    # "scripts/rl_games/config/klask_config_4.yaml",
]
checkpoint = [
    # "logs/rl_games/klask/pretrained_agent_action_1.0/nn/last_klask_ep_70_rew_2.950554.pth",
    # "logs/rl_games/klask/pretrained_agent_action_1.0/nn/last_klask_ep_70_rew_2.950554.pth",
    "logs/rl_games/klask/pretrained_agent_action_1.0/nn/last_klask_ep_35_rew_3.9405801.pth",
    # "logs/rl_games/klask/pretrained_agent_action_1.0/nn/last_klask_ep_35_rew_3.9405801.pth",
]

# Python executable and training script
PYTHON = "/workspace/isaaclab/_isaac_sim/python.sh"
TRAIN_SCRIPT = "scripts/rl_games/train_klask.py"  # your main train script
EVALUATE_SCRIPT = "scripts/rl_games/evaluate_agents.py"
# Store process handles
processes = []
project_folder = Path("logs/rl_games/klask/pool_of_players") / Path(args.pool_name)


# Start training with different config files
for i, cfg in enumerate(configs):
    print(f"Launching training #{i+1} with config: {cfg}")
    cmd = [
        PYTHON,
        TRAIN_SCRIPT,
        "--config",
        cfg,
        "--device",
        f"cuda:{i}",
        "--headless",
        "--num_envs",
        "4096",
        "--wandb-project-name",
        f"Training_curriculum_agent_{i+1}",
        "--training_curriculum",
        "--mode",
        "0",
        "--checkpoint",
        checkpoint[i],
        "--project_folder",
        project_folder,
    ]

    # Start process without redirecting output
    proc = subprocess.Popen(cmd)
    processes.append(proc)

# Periodic progress check
base_folder = Path("logs/rl_games/klask/training_curriculum")
agent_names = [f"agent_{i+1}" for i in range(len(configs))]

# Track previously seen files per agent
files_at_beginnig = {name: set((base_folder / name).glob("**/nn/last*")).copy() for name in agent_names}

last_seen_files = copy.deepcopy(files_at_beginnig)
run_number = 0
new_files = {}

try:
    while True:
        new_agents = True
        for name in agent_names:
            folder = base_folder / name
            current_files = set(folder.glob("**/nn/last*"))
            added_files = current_files - last_seen_files[name]

            if current_files != last_seen_files[name]:
                new_files[name] = list(added_files)
                last_seen_files[name] = current_files.copy()

        if new_files != {}:

            print("New agents for all scripts, starting to evaluate:")

            for name, checkpoint in new_files.items():

                run_number += 1
                match = re.search(r"agent_(\d+)", name)
                if not match:
                    raise ValueError(f"Could not extract agent number from name: {name}")
                agent_number = int(match.group(1))

                checkpoint_benchmark = project_folder / Path("benchmark/benchmark.pth")
                checkpoints_paths = [checkpoint_benchmark]
                checkpoints_paths.append(checkpoint[0])
                config_benchmark = project_folder / Path("benchmark/benchmark.yaml")
                configs_paths = [config_benchmark]
                configs_paths.append(Path(f"scripts/rl_games/config/klask_config_{agent_number}.yaml"))

                cmd = [
                    PYTHON,
                    EVALUATE_SCRIPT,
                    "--config",
                    "scripts/rl_games/config/klask_config_1.yaml",  # Dummy, actual configs passed later
                    "--tournament_name",
                    "training_curriculum",
                    "--num_envs",
                    "1000",
                    "--headless",
                    "--num_instances",
                    "2",
                    "--run_number",
                    str(run_number),
                    "--num_games_per_round",
                    "150",
                    "--elo_thresehold",
                    "1200",
                    "--dir",
                    project_folder,
                ]
                for ckpt, cfpt in zip(checkpoints_paths, configs_paths):
                    cmd.extend(["--checkpoints", ckpt])
                    cmd.extend(["--player_configs", cfpt])

                proc = subprocess.Popen(cmd)
            new_files = {}

        time.sleep(20)

except KeyboardInterrupt:
    print("Interrupted. Terminating all processes...")
    for proc, _ in processes:
        proc.terminate()

# Ensure all complete
for proc, _ in processes:
    proc.wait()
