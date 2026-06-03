# Fast SAC Training

This folder trains the KLASK playing policy with Fast SAC — a GPU-parallel, distributional Soft Actor-Critic — against the state-based `Klask-Rl-FastSAC-v0` Isaac Lab environment. Its main component is a prioritized experience replay (PER) buffer; a hindsight experience replay (HER) buffer is additionally available on top to speed up training (it is off by default, with `her_ratio = 0.0`).

The Fast SAC algorithm itself (actor/critic networks, the distributional SAC update, the GPU replay buffer, and observation normalization) was implemented by [Thomas Bi](https://github.com/thomasbi1) and is included here only as the [`klask_her`](https://github.com/thomasbi1/klask_her) Git submodule. What lives in **this** repository is the Isaac Lab integration: an environment wrapper and the training loop in `train_fast_sac_isaaclab.py` that drive `klask_her` against the KLASK simulation.

All commands below are meant to be run **inside the training container** (see the [root README](../../../../README.md) for how to build and enter it). Throughout, `python` refers to the Isaac Lab Python interpreter at `/workspace/isaaclab/_isaac_sim/python.sh`.

## Folder Contents

| File | Description |
| --- | --- |
| `train_fast_sac_isaaclab.py` | Training entry point: wraps the KLASK env and runs the Fast SAC + HER training loop. |
| `play_fast_sac_isaaclab.py` | Load one or two checkpoints to watch or evaluate a trained policy. |
| `klask_her/` | Thomas Bi's Fast SAC + HER implementation (Git submodule). |

## Configuration

Training is configured through the `KlaskTrainingConfig` dataclass at the top of `train_fast_sac_isaaclab.py`, parsed with [tyro](https://github.com/brentyi/tyro). Every field can be overridden on the command line (e.g. `--num_envs 8192 --gamma 0.98`), and `--help` lists them all with their defaults:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/fast_sac/train_fast_sac_isaaclab.py --help
```

The most relevant options, grouped by area:

| Area | Options |
| --- | --- |
| Environment / simulation | `num_envs`, `episode_length_s`, `goal_reward`, `sim_dt`, `decimation`, `max_velocity`, `max_acceleration`. |
| Replay buffer | `buffer_size`, `contact_priority` (PER weighting of contact transitions), `mirror_y`; HER add-on: `her_ratio` (`0.0` disables HER), `goal_radius`, `max_future_steps`. |
| SAC | `num_learning_iterations`, `batch_size`, `num_updates`, `policy_frequency`, `gamma`, `tau`, `actor_lr` / `critic_lr` / `alpha_lr`, `actor_hidden_dim` / `critic_hidden_dim`, distributional critic settings (`num_atoms`, `v_min`, `v_max`), `use_autotune`, `obs_normalization`. |
| Domain matching | `enable_actuator_model` / `actuator_model_checkpoint` (aligns the simulated motors with the real hardware), `enable_collision_avoidance`, `enable_initialization`. |
| Self-play | `self_play`, `opponent_pool_size`, `opponent_save_interval`, `opponent_update_interval`, `pfsp_epsilon`, `pfsp_p`. |
| Performance | `do_compile`, `amp` / `amp_dtype`. |
| Run / logging | `seed`, `device`, `output_dir`, `logging_interval`, `save_interval`. |

## Training

A minimal training run uses all defaults:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/fast_sac/train_fast_sac_isaaclab.py
```

Add any configuration overrides as flags, for example:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/fast_sac/train_fast_sac_isaaclab.py \
    --num_envs 8192 \
    --seed 1
```

The environment is state-based, so no cameras (and no `--enable_cameras`) are needed for training. Checkpoints and logs are written to `output_dir` (by default `logs/fast_sac_isaaclab/training/<date>/<time>/`).

## Logging with TensorBoard

Unlike the other frameworks in this repo, Fast SAC logs directly to [TensorBoard](https://www.tensorflow.org/tensorboard) rather than to Weights & Biases — the training loop writes event files to the `tb/` subfolder of the run's `output_dir`. To view them, start TensorBoard and point it at the log directory:

```bash
tensorboard --logdir logs/fast_sac_isaaclab/training
```

Then open the printed URL (default `http://localhost:6006`) in your browser. Alternatively, if you prefer to track runs in W&B, you can sync the TensorBoard event files to your account:

```bash
wandb sync logs/fast_sac_isaaclab/training/<date>/<time>/tb
```

## Watching / Evaluating a Policy

`play_fast_sac_isaaclab.py` loads a trained checkpoint and either plays it against a mirrored copy of itself or against a second checkpoint:

```bash
# Single-agent playback (opponent is a copy of the player)
/workspace/isaaclab/_isaac_sim/python.sh scripts/fast_sac/play_fast_sac_isaaclab.py \
    --checkpoint /path/to/checkpoint.pt \
    --num_envs 1
```

Pass `--opponent_checkpoint` to run a head-to-head between two policies and collect statistics over `--num_games`:

```bash
# Head-to-head evaluation between two checkpoints
/workspace/isaaclab/_isaac_sim/python.sh scripts/fast_sac/play_fast_sac_isaaclab.py \
    --checkpoint /path/to/player.pt \
    --opponent_checkpoint /path/to/opponent.pt \
    --num_games 10000 \
    --boxplot
```

In head-to-head mode a `.npz` of per-game metrics is written next to the player checkpoint; with `--boxplot` a box-plot PNG is generated next to it as well. Other useful flags include `--episode_length_s` (override the episode length, otherwise read from the checkpoint), `--deterministic/--no-deterministic` (use the mean action vs. sampling), and `--video` / `--video_length` to record playback.
