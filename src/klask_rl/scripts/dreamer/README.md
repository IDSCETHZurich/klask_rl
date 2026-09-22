# Dreamer Training

This folder trains the KLASK playing policy with a model-based world model on the image-based KLASK Dreamer environment (see [Environments](#environments) below). It is built on [R2-Dreamer](https://github.com/MeierTobias/r2dreamer), which was chosen as a basis because it provides a clean PyTorch DreamerV3 implementation and exposes the representation loss as a configurable option. We experimented with R2-Dreamer's redundancy-reduced (R2) representation loss, but it did not improve training on KLASK, so the default configuration uses the standard Dreamer representation loss (`model.rep_loss: "dreamer"`). The R2-Dreamer implementation itself lives in the `r2dreamer` submodule, which is mounted into this folder inside the container.

All commands below are meant to be run **inside the training container** (see the [root README](../../../../README.md) for how to build and enter it). Throughout, `python` refers to the Isaac Lab Python interpreter at `/workspace/isaaclab/_isaac_sim/python.sh`.

## Folder Contents

| File | Description |
| --- | --- |
| `train_dreamer.py` | Training entry point (configured with [Hydra](https://hydra.cc/)). |
| `evaluate_dreamer.py` | Head-to-head evaluation of one or more player checkpoints against an opponent. |
| `dreamer_self_play.py` | Self-play utilities. |
| `export_dreamer_checkpoint.py` | Export a trained checkpoint for deployment in `klask_software`. |
| `env_cfg_utils.py` | Helpers for building the environment configuration. |
| `config-sprite-per-oppo-sep*.yaml` | Training configurations (see below). |
| `sprite_renderer/` | Renderer that composites pre-rendered sprites for the image observations (see [Sprite Renderer](#sprite-renderer)). |
| `debug/` | Debugging utilities for the camera output and reward schedules (see [Debug Utilities](#debug-utilities)). |
| `r2dreamer/` | The R2-Dreamer world-model implementation (Git submodule). |

## Environments

The Dreamer agent learns from image observations of the board. Two image-based KLASK Dreamer environments are registered, differing only in how those images are produced:

| Task id | Observation source |
| --- | --- |
| `Klask-Rl-Dreamer-v0` | Images are rendered by a `TiledCamera` inside Isaac Sim, i.e. the actual 3D scene is rendered on the GPU. |
| `Klask-Rl-Dreamer-Sprite-v0` | Images are produced by the [sprite renderer](#sprite-renderer), which composites pre-rendered ball/peg sprites onto a background image on the CPU from the board state. |

In the YAML the environment is selected through `env.task`, which follows the codebase convention `isaaclab_<task_id>` — e.g. `isaaclab_Klask-Rl-Dreamer-Sprite-v0`.

## Configurations

Training is configured with [Hydra](https://hydra.cc/). The configs live next to `train_dreamer.py` and are selected by name (without the `.yaml` extension) via `--config-name`:

| Config | Use case |
| --- | --- |
| `config-sprite-per-oppo-sep.yaml` | Default configuration: sprite-based observations, prioritized experience replay, and separated opponent handling. |
| `config-sprite-per-oppo-sep_1gpu.yaml` | Single-GPU variant of the above. |

The individual parameters are documented in [Configuration Reference](#configuration-reference) below.

While Hydra also allows individual values to be overridden on the command line (e.g. `seed=1`), this is **not** the intended workflow here. Instead, adapt the config file itself and commit it to git for every run. This way each run's exact configuration is captured in version control, so an old config can always be reconstructed from its commit and retrained at a later stage.

## Training

A typical training run looks like this:

```bash
/workspace/isaaclab/_isaac_sim/python.sh /workspace/klask_rl/scripts/dreamer/train_dreamer.py \
    --config-name=config-sprite-per-oppo-sep
```

Before starting a run, adapt the config file to the experiment you want and commit it to git, so the run is reproducible from version control (see [Configurations](#configurations)).

Training logs and checkpoints are written to the `logdir` set in the config (by default under `logs/dreamer/training/<date>/<time>/`). Each run directory contains the periodic `checkpoint_<step>.pt` files and the fully resolved Hydra config under `.hydra/config.yaml` — both of which are needed again at evaluation time. If `logdir` points at a non-empty directory, training resumes from the latest checkpoint found there.

## Evaluation

For the reconstructed PPO-B / D-RSSM / FastSAC round robin and the exact-versus-zero
opponent-action ablation, use the [tournament runner](../tournament/README.md).
It saves individual games, balances board seats, and reports game-level bootstrap
intervals with zeroed opponent input as the primary result.

`evaluate_dreamer.py` plays a trained Dreamer policy head-to-head against an opponent and records the outcome over many games:

```bash
/workspace/isaaclab/_isaac_sim/python.sh ./scripts/dreamer/evaluate_dreamer.py \
    --enable_cameras \
    --checkpoint /workspace/klask_rl/logdir/2026-05-09/09-16-19/checkpoint_5700000.pt \
    --config /workspace/klask_rl/logdir/2026-05-09/09-16-19/.hydra/config.yaml \
    --opponent_config /workspace/klask_rl/scripts/rl_games/config/klask_ppo_config_with_pretraining.yaml \
    --opponent_checkpoint /workspace/klask_rl/logs/rl_games/klask/new_agent_from_pretrained/new_AM/export/klask_ppo_nn_v1.1.pth \
    --opponent_type ppo \
    --num_games 10000 \
    --episode_length_s 30.0
```

The most important flags are:

| Flag | Description |
| --- | --- |
| `--checkpoint` | One or more paths or glob patterns to the player checkpoint(s) `.pt` (see [below](#evaluating-multiple-checkpoints)). |
| `--config` | The player's Hydra training config, i.e. the `.hydra/config.yaml` from its run directory. |
| `--opponent_checkpoint` | Path to the opponent's checkpoint. |
| `--opponent_config` | The opponent's config YAML (defaults to `--config` when omitted). |
| `--opponent_type` | Opponent agent type: `dreamer`, `ppo` (rl_games), or `fast_sac`. |
| `--num_games` | Number of complete games to play (default `10000`). |
| `--num_envs` | Number of parallel environments (default `1024`). |
| `--episode_length_s` | Override the episode length in seconds. |
| `--boxplot` | Also generate a box-plot PNG next to the saved metrics (see [below](#evaluation-output--box-plots)). |

`evaluate_dreamer.py` always operates on image observations, so it enables `--enable_cameras` automatically — passing it explicitly (as above) is harmless but not required.

### Evaluating Multiple Checkpoints

The `--checkpoint` argument accepts **glob patterns**. If a pattern matches several checkpoint files, all of them are evaluated sequentially against the same opponent. This is the convenient way to track how a policy improves over the course of training, for example:

```bash
--checkpoint '/workspace/klask_rl/logdir/2026-05-09/09-16-19/checkpoint_*.pt'
```

You can also mix literal paths and patterns, e.g. `--checkpoint 'runs/A/*.pt' 'runs/B/last.pt'`.

## Evaluation Output & Box Plots

Results are written to `logs/dreamer/head_to_head_results/`. For each evaluated checkpoint you get a timestamped set of files:

- `h2h_results_<timestamp>.txt` — a human-readable summary of the head-to-head outcome.
- `h2h_results_<timestamp>.npz` — the per-game metrics, used for plotting and aggregation.

When you pass `--boxplot`, a box-plot PNG (`h2h_results_<timestamp>.png`) is generated next to the `.npz` summarizing the per-game metrics. If you evaluate **multiple checkpoints** (via a glob pattern), an aggregate summary is additionally produced across all of them:

- `h2h_aggregate_<timestamp>.txt` — combined summary.
- `h2h_aggregate_<timestamp>.png` — win-rate across checkpoints.
- `h2h_aggregate_<timestamp>_terminations.png` — termination-reason breakdown.

You don't have to decide on the box plot up front. The plot is generated from the saved `.npz`, so you can create it after the fact by running the plotting script on any results file:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/plot_eval_boxplots.py \
    logs/dreamer/head_to_head_results/h2h_results_<timestamp>.npz
```

By default the PNG is written next to the `.npz`; use `--out` to choose a different path.

## Configuration Reference

The following describes the parameters in `config-sprite-per-oppo-sep.yaml`. The intended way to change any of them is to edit the config file and commit it to git (see [Configurations](#configurations)), keeping each run reproducible — even though Hydra would also accept command-line overrides.

### Top-level

| Key | Description |
| --- | --- |
| `defaults.model` | World-model size preset pulled from `r2dreamer/configs` (e.g. `size50M`). |
| `model.rep_loss` | Representation loss variant. `dreamer` (the default) uses the standard Dreamer representation loss; alternatives such as `r2dreamer`, `infonce`, and `dreamerpro` are available, but R2-Dreamer's redundancy-reduced loss did not improve training on KLASK. |
| `model.compile` / `model.compile_mode` / `model.inference_compile_mode` | Enable and configure `torch.compile` for the model (training and inference). |

### Logging

| Key | Description |
| --- | --- |
| `wandb_project` / `wandb_entity` / `wandb_name` | [Weights & Biases](https://wandb.ai/) project, team, and run name for logging. |

### Initialization

| Key | Description |
| --- | --- |
| `logdir` | Run directory for checkpoints and logs. Pointing it at a non-empty directory resumes training from the latest checkpoint there. |
| `seed` | Global random seed. |
| `deterministic_run` | Force deterministic execution (slower, reproducible). |
| `device` / `sim_device` | Devices for the model/inference and for the simulator. |
| `train_devices` | List of GPUs used for the (data-parallel) training updates. |
| `init_checkpoint` | Optional checkpoint to initialize weights from (pretrain → finetune). |
| `load_world_model_only` | When initializing from `init_checkpoint`, load only the world-model weights. |
| `hydra.run.dir` / `hydra.searchpath` | Hydra output directory and the search path for the bundled R2-Dreamer configs. |

### Batching

| Key | Description |
| --- | --- |
| `batch_size` | Number of sequences per training batch. |
| `batch_length` | Number of timesteps per sequence. |
| `micro_batch_size` | Micro-batch size used to split a batch for gradient accumulation. |

### Environment (`env`)

| Key | Description |
| --- | --- |
| `task` | Environment to train on, in the `isaaclab_<task_id>` convention. The default `isaaclab_Klask-Rl-Dreamer-Sprite-v0` selects the sprite renderer; `isaaclab_Klask-Rl-Dreamer-v0` uses the TiledCamera instead (see [Environments](#environments)). |
| `steps` | Total number of environment steps to train for. |
| `env_num` | Number of parallel simulation environments. |
| `eval_episode_num` | Episodes per periodic evaluation. |
| `action_repeat` | How many sim steps each agent action is held for. |
| `decimation` | Control decimation (sim steps per policy step). |
| `train_ratio` | Ratio of gradient updates to collected environment steps. |
| `sim_dt` | Physics timestep in seconds. |
| `ball_reset_position_x` / `ball_reset_position_y` | Ranges `[min, max]` for randomizing the ball's reset position. |
| `max_velocity` | Maximum player (peg) velocity in m/s. |
| `max_acceleration` | Maximum rate-of-change of player velocity in m/s². |
| `episode_length_s` | Episode length in seconds. |
| `actuator_model.enable` / `actuator_model.checkpoint` | Enable the learned actuator model and the checkpoint to load (matches the simulated motors to the real hardware). |
| `collision_avoidance.enable` | Enable peg/wall collision avoidance. |
| `size` | Rendered sprite image resolution `[H, W]`. |
| `encoder` / `decoder` (`mlp_keys`, `cnn_keys`) | Which observation keys feed the MLP and CNN encoder/decoder branches (`image` for the rendered board). |

#### Observation Augmentation (`env.augmentation`)

Randomized image augmentations applied by the [sprite renderer](#sprite-renderer) to improve sim-to-real transfer (the TiledCamera environment does not use this block).

| Key | Description |
| --- | --- |
| `hold_per_episode` | `true` samples an augmentation once per environment at reset; `false` resamples on every render. |
| `brightness` | Multiplicative BGR scale (`1.0` = identity, `<1` darker, `>1` brighter). |
| `contrast` | Stretch/compress around mid-gray (`1.0` = identity). |
| `gamma` | Power curve on midtones (`1.0` = identity, `>1` brightens midtones). |
| `color_temp` | Warm/cool shift (`+` boosts red / reduces blue; `0` = identity). |
| `saturation` | HSV chroma scale (`1.0` = identity; too strong can swap peg colors). |
| `hue` | HSV hue shift in degrees (`0` = identity; strong shifts swap red/black peg identity). |

Each augmentation takes `enabled`, a `range` it is sampled from, and the number of discrete `levels`.

#### Rewards (`env.rewards`)

Each reward term has a `type` (`static` for a constant weight, or `exponential` with a `decay_rate`) and a `weight`. Setting a weight to `0.0` disables the term. The terms cover goal events (`goal_scored`, `goal_conceded`, `player_in_goal`, `opponent_in_goal`), shaping signals (distances between player/ball/goals, ball and player speed, positioning, ball-in-own-half), and action-smoothness penalties (`action_rate_cap_saturation`, `action_smooth_hinge`, `action_delta_l2`). In the default config the sparse goal-related terms dominate (e.g. `goal_scored: 1000`, `goal_conceded: -1000`, `player_in_goal: -1000`) and most shaping terms are set to `0.0`.

#### Terminations (`env.terminations`)

Boolean flags selecting which events end an episode: `time_out`, `goal_scored`, `goal_conceded`, `player_in_goal`, `opponent_in_goal`.

#### Initialization Curriculum (`env.initialization.init_velocity`)

A `schedule` of `phases` that ramps the ball's initial velocity over training. Each phase has a `type` (`static` for a fixed `speed`, or `sigmoid` for a smooth transition between two speeds with a given `steepness`) and the `steps` range it applies to (`-1` means "until the end"). The default starts the ball at speed `1.0`, sigmoid-decays it to `0.0` between 1M and 3M steps, then holds it at `0.0`.

### Replay Buffer (`buffer`)

| Key | Description |
| --- | --- |
| `batch_size` / `batch_length` | Sampling shape (inherited from the top-level values). |
| `max_size` | Maximum number of transitions stored. |
| `device` / `storage_device` | Sampling device and where the buffer is stored (`cpu` to save GPU memory). |
| `prioritized.enable` | Enable prioritized experience replay (PER). |
| `prioritized.alpha` | Prioritization exponent (`0` = uniform, `1` = fully priority-proportional). |
| `prioritized.beta` | Importance-sampling correction (`0` = off). |
| `prioritized.baseline_priority` | Base priority assigned to every transition. |
| `prioritized.debug_metrics` | Log additional PER diagnostics. |
| `prioritized.tags` | Boost the sampling priority of transitions matching a reward or termination term (each tag has a `name`, a `priority`, and a `reward_term`/`termination_term`). The default up-weights ball contacts and goal events. |

### Trainer (`trainer`)

| Key | Description |
| --- | --- |
| `steps` | Total training steps (mirrors `env.steps`). |
| `pretrain` | Number of pretraining updates before normal training. |
| `eval_every` | Run an evaluation every N steps. |
| `eval_episode_num` | Episodes per evaluation. |
| `batch_size` / `batch_length` / `train_ratio` | Training batch shape and update-to-step ratio. |
| `video_pred_log` | Log world-model video predictions. |
| `params_hist_log` | Log parameter histograms. |
| `update_log_every` | Log training metrics every N updates. |
| `action_repeat` | Action repeat (mirrors `env.action_repeat`). |
| `save_checkpoint_every` | Save a checkpoint every N steps (`0` = only at the end). |
| `profiling.*` | Optional Torch/cProfile profiling controls (Perfetto traces and periodic cProfile dumps). |

### Self-Play (`self_play`)

| Key | Description |
| --- | --- |
| `enabled` | Enable self-play training. |
| `update_score` | The opponent snapshot is updated only when the rolling-mean score exceeds this threshold. |
| `games_to_track` | Rolling window size used to average the score. |

### Opponent Separation (`opponent_separation`)

| Key | Description |
| --- | --- |
| `enabled` | Feed explicit opponent actions to the world model (4D action = player + opponent). |
| `imag_opponent` | Opponent policy used during imagination: `random`, `zero`, or `selfplay`. |
| `trajectory_mirroring.enabled` | Compute opponent rewards and mirror trajectories at sampling time. |
| `trajectory_mirroring.warm_start` | Use the opponent's RSSM states to warm-start the mirrored trajectories. |

## Sprite Renderer

The `sprite_renderer/` folder provides the image renderer used by the `Klask-Rl-Dreamer-Sprite-v0` environment. Instead of rendering the 3D scene with a camera, `board_renderer.py`'s `BoardRenderer` composites pre-extracted ball and peg sprites onto a background image purely from the board state. All heavy work is precomputed at initialization for maximum runtime speed.

```
sprite_renderer/
├── board_renderer.py   # BoardRenderer: composites sprites onto the background
├── assets/
│   ├── background/      # median_background.png — the board background
│   └── sprites/         # pre-extracted ball/ and peg/ sprite PNGs (+ JSON metadata)
└── debug/               # tools to test and visualize the renderer (see below)
```

The sprite assets here are consumed automatically by the environment; the paths are configurable via `env` keys (`sprite_dir`, `background_path`) but default to this folder. Color/lighting **augmentations** are applied by the renderer and configured through the `env.augmentation` block (see the [Configuration Reference](#observation-augmentation-envaugmentation)).

The `sprite_renderer/debug/` subfolder contains tools for inspecting the renderer in isolation, without launching Isaac Sim:

| File | Description |
| --- | --- |
| `test_renderer.py` | Renders sample board states and benchmarks the renderer's throughput. |
| `visualize_augmentations.py` | Renders an (axis × level) grid of augmentation variants from an augmentation YAML so you can see each `aug_id`. |
| `generate_demo_gif.py` | Produces a GIF of the pegs sweeping across all board positions. |
| `augmentation_example.yaml` | Example augmentation config (same schema as the training config's `augmentation` block). |
| `test_renders/` | Output directory for the generated images and GIFs. |

For example, to visualize the augmentation variants:

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
    scripts/dreamer/sprite_renderer/debug/visualize_augmentations.py
```

## Debug Utilities

The top-level `debug/` folder holds small helpers used while developing the Dreamer environment:

| File | Description |
| --- | --- |
| `cam_debug.py` | Launches `Klask-Rl-Dreamer-v0`, steps the simulation, and streams the camera image of one environment to the browser for visual inspection. |
| `mjpeg_server.py` | A small reusable HTTP image server (used by `cam_debug.py`) that streams a single RGB frame to the browser with auto-refresh. |
| `rewards.py` | Plots the reward weight-decay and sigmoid schedules so you can tune `decay_rate`/midpoint values before training. |

To inspect the TiledCamera output live:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/dreamer/debug/cam_debug.py
# then open http://localhost:8089 in your browser
```
