# SAC Training (Stable-Baselines3)

This folder trains the KLASK playing policy with Soft Actor-Critic (SAC) from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (SB3) against the state-based KLASK environments. It was primarily used to experiment with different **Hindsight Experience Replay (HER)** buffer setups, but it is structured generically around SB3, so it can be extended to other SB3 algorithms in the future.

It also serves as a worked example of **hyperparameter tuning with [Ray](https://www.ray.io/) Tune and [Optuna](https://optuna.org/)**, which are already integrated into the training container — see [Hyperparameter Tuning](#hyperparameter-tuning-with-ray--optuna) below.

All commands below are meant to be run **inside the training container** (see the [root README](../../../../README.md) for how to build and enter it). Throughout, `python` refers to the Isaac Lab Python interpreter at `/workspace/isaaclab/_isaac_sim/python.sh`.

## Folder Contents

| File | Description |
| --- | --- |
| `train_sac.py` | Training entry point. |
| `play_sac.py` | Load a checkpoint to watch or evaluate a trained policy. |
| `experiment_config.py` | The `ExperimentConfig` system that loads an experiment YAML and turns it into Hydra overrides. |
| `env_utils.py` | Helpers for building the environment and resolving config/log paths. |
| `contact_priority_replay_buffer.py` | Custom replay buffer that prioritizes ball-contact transitions. |
| `utils.py` | Small shared utilities. |
| `config/experiments/` | The experiment configurations (see below). |
| `ray/` | Ray Tune + Optuna hyperparameter tuning setup (see [`ray/README.md`](ray/README.md)). |

## Experiment Configurations

Each experiment is described by a YAML file under `config/experiments/`, loaded through the `ExperimentConfig` system (`experiment_config.py`). A config bundles the SB3 agent hyperparameters, the environment, and the buffer/HER settings; `ExperimentConfig` converts it into the Hydra overrides that configure the registered Isaac Lab task. Each experiment targets a different registered environment:

| Config | Environment | Focus |
| --- | --- | --- |
| `klask_sac_base.yaml` | `Klask-Rl-SAC-v0` | Plain SAC baseline. |
| `klask_sac_her.yaml` | `Klask-Rl-HER-SAC-v0` | SAC with a HER replay buffer. |
| `klask_sac_contact_priority.yaml` | `Klask-Rl-ContactPriority-v0` | SAC with the contact-priority replay buffer. |
| `klask_sac_two_stage_her.yaml` | `Klask-Rl-TwoStage-HER-v0` | Two-stage HER (hit the ball, then score). |

## Training

Run a training by pointing `train_sac.py` at one of the experiment configs:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/sb3/train_sac.py \
    --config experiments/klask_sac_her.yaml
```

The environments are state-based, so no cameras are needed for training. Logs and checkpoints are written under `logs/sb3/`. Training metrics are logged to TensorBoard via SB3's logger, and to [Weights & Biases](https://wandb.ai/) when `wandb` is available (configured through the experiment YAML).

## Hyperparameter Tuning with Ray + Optuna

In addition to running a single config, this folder can sweep hyperparameters using Ray Tune with the Optuna search algorithm. Ray is already installed and configured in the container, which makes this a good reference for how to set up Ray-based tuning in this project.

The sweep reuses Isaac Lab's built-in Ray tuner together with the job-config classes in `ray/hyperparameter_tuning/klask_sac_base_cfg.py`, which define the search spaces on top of the experiment YAMLs. A local sweep looks like this:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/ray/tuner.py --run_mode local \
    --cfg_file /workspace/klask_rl/scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py \
    --cfg_class KlaskSacHerTuneJobCfg \
    --workflow /workspace/klask_rl/scripts/sb3/train_sac.py \
    --metric rollout_ep_rew_mean \
    --mode max \
    --num_samples 8
```

Ray Tune uses Optuna for the search, reads the TensorBoard metrics back from each trial, and stores results under `/tmp/ray/IsaacRay-<cfg_class>-tune/`. The full set of available job-config classes, tunable parameters, metrics, and troubleshooting tips is documented in [`ray/README.md`](ray/README.md).

## Watching / Evaluating a Policy

To load a trained checkpoint and watch it play, use `play_sac.py` with the same experiment config that was used for training:

```bash
# Use a specific checkpoint
/workspace/isaaclab/_isaac_sim/python.sh scripts/sb3/play_sac.py \
    --config experiments/klask_sac_her.yaml \
    --checkpoint /path/to/model.zip

# Or omit --checkpoint to use the latest checkpoint for that experiment
/workspace/isaaclab/_isaac_sim/python.sh scripts/sb3/play_sac.py \
    --config experiments/klask_sac_her.yaml
```
