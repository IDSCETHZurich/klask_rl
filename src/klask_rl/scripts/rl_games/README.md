# PPO Training (rl_games)

This folder trains the KLASK playing policy with Proximal Policy Optimization (PPO) using the [`rl_games`](https://github.com/Denys88/rl_games) framework against the `Klask-Rl-v0` Isaac Lab environment.

All commands below are meant to be run **inside the training container** (see the [root README](../../../../README.md) for how to build and enter it). Throughout, `python` refers to the Isaac Lab Python interpreter at `/workspace/isaaclab/_isaac_sim/python.sh`.

## Folder Contents

| File | Description |
| --- | --- |
| `train_klask.py` | Training entry point. |
| `play_klask.py` | Load a checkpoint to watch or evaluate a trained policy. |
| `klask_rl_games.py` | Glue code connecting the Isaac Lab environment to `rl_games`. |
| `debug_sim2real.py`, `debug_sim2real_plot.py` | Sim-to-real debugging utilities. |
| `config/` | The PPO training configurations (see below). |

## Configurations

The agent configuration controls the network, the PPO hyperparameters, and the training schedule. By default the task's registered `rl_games_cfg_entry_point` is used, but you can override it with the `--config` flag to point at one of the YAML files in `config/`:

| Config | Use case |
| --- | --- |
| `klask_ppo_config_with_pretraining.yaml` | PPO training that starts from a pretraining stage. |
| `pretrain_agent_config.yaml` | Configuration for the pretraining phase itself. |

When `--config` is provided, its contents are merged on top of the default agent configuration, so you only need to specify the values you want to change.

## Training

A typical training run looks like this:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/rl_games/train_klask.py \
    --config /workspace/klask_rl/scripts/rl_games/config/klask_ppo_config_with_pretraining.yaml \
    --headless \
    --num_envs 4096 \
    --wandb-project-name KLASK_PPO
```

The most important flags are:

| Flag | Description |
| --- | --- |
| `--config` | Path to the agent configuration YAML (see [Configurations](#configurations)). |
| `--headless` | Run without a GUI window. Use this for training; see the [WebRTC livestream](../../../../README.md#viewing-the-simulation-webrtc-livestream) section of the root README to watch the simulation instead. |
| `--num_envs` | Number of parallel environments to simulate (e.g. `4096`). Higher values speed up data collection but use more GPU memory. |
| `--wandb-project-name` | [Weights & Biases](https://wandb.ai/) project to log to (`--wandb-entity` sets the team). |
| `--checkpoint` | Path to a `.pth` checkpoint to resume training from. |
| `--max_iterations` | Override the number of training epochs from the config. |
| `--seed` | Seed for the environment and agent. |
| `--sigma` | Override the policy's initial action standard deviation. |

A full list of options is available with:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/rl_games/train_klask.py --help
```

## Watching / Evaluating a Policy

To load a trained checkpoint and watch it play, use `play_klask.py`:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/rl_games/play_klask.py \
    --checkpoint /path/to/checkpoint.pth \
    --num_envs 1
```

You can pit two policies against each other with `--opponent_checkpoint` (and `--opponent_config`), run a fixed number of matches with `--num_games`, and collect evaluation statistics with `--boxplot`. As with training, run it `--headless` and connect through the WebRTC livestream client to view the simulation remotely.

## Logging & Checkpoints

Logs and checkpoints are written to `logs/rl_games/klask/<experiment_name>/`, with model weights stored under the `nn/` subfolder. The full environment and agent configurations used for a run are also dumped there (`params/env.yaml`, `params/agent.yaml`) for reproducibility. When a `--wandb-project-name` is given, training metrics are additionally streamed to Weights & Biases.

## Sim-to-Real Debugging

`debug_sim2real.py` replays recorded real-world trajectories in simulation so you can compare the policy's actions against the real system and debug discrepancies between simulation and hardware.
