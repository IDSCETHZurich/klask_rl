# Ray Tune Integration for Klask RL

This directory contains the hyperparameter tuning configuration for Klask RL, using Isaac Lab's built-in Ray Tune integration.

## Structure

```
ray/
└── hyperparameter_tuning/
    └── klask_sac_base_cfg.py  # Job config classes defining search spaces
```

**Note**: This uses Isaac Lab's tuner at `/workspace/isaaclab/scripts/reinforcement_learning/ray/tuner.py` - no need to duplicate it!

## How It Works

Your training script (`train_sac.py`) uses `ExperimentConfig` to:
1. Load base config from YAML (e.g., `experiments/klask_sac_her.yaml`)
2. Convert it to Hydra overrides via `get_hydra_overrides()`
3. Ray Tune adds **additional** Hydra overrides on top for hyperparameter tuning

### Command Flow Example

For a trial with `learning_rate=0.0001`, Isaac Lab's tuner generates:

```bash
cd /workspace/klask_rl
python scripts/sb3/train_sac.py \
  --config experiments/klask_sac_her.yaml \
  agent.learning_rate=0.0001
```

What happens:
1. `argparse` reads `--config experiments/klask_sac_her.yaml`
2. `ExperimentConfig.from_file()` loads base YAML
3. `get_hydra_overrides()` converts YAML → Hydra CLI args
4. Remaining `sys.argv` includes `agent.learning_rate=0.0001`
5. `@hydra_task_config` applies ALL overrides (base + tune) to agent_cfg
6. Tuner reads tensorboard logs and reports metrics back to Ray/Optuna

## Usage

### Local Hyperparameter Tuning

Run a hyperparameter sweep locally using Isaac Lab's tuner:

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

**Important flags**:
- `--workflow`: Must point to your `train_sac.py` (Isaac Lab defaults to rl_games)
- `--metric`: Tensorboard scalar key to optimize (e.g., `rollout_ep_rew_mean`, `rollout_success_rate`)
- `--mode`: `max` (for rewards/success) or `min` (for losses)

### Quick Test Run

To verify the setup works:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/ray/tuner.py --run_mode local \
    --cfg_file /workspace/klask_rl/scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py \
    --cfg_class KlaskSacHerJobCfg \
    --workflow /workspace/klask_rl/scripts/sb3/train_sac.py \
    --num_samples 1
```

### Available Config Classes

- `KlaskSacBaseJobCfg`: Base SAC - loads `experiments/klask_sac_base.yaml`
- `KlaskSacHerJobCfg`: HER SAC - loads `experiments/klask_sac_her.yaml`
- `KlaskSacHerTuneJobCfg`: HER SAC with hyperparameter tuning - tunes on top of `experiments/klask_sac_her.yaml`
- `KlaskSacTwoStageHerJobCfg`: Two-stage HER SAC - loads `experiments/klask_sac_two_stage_her.yaml`

All configs use `tune.choice([value])` to satisfy Ray Tune requirements. The non-Tune configs use single values for deterministic runs.

### Command Line Arguments

Isaac Lab's tuner supports:

- `--run_mode`: Run mode (`local` or `remote`, default: `remote`)
- `--cfg_file`: Absolute path to config file (required)
- `--cfg_class`: Config class name (required)
- `--workflow`: Absolute path to training script (required for non-rl_games workflows)
- `--num_samples`: Number of hyperparameter samples to try (default: `100`)
- `--metric`: Metric to optimize from tensorboard logs (default: `rewards/time`)
- `--mode`: Optimization mode `min` or `max` (default: `max`)
- `--repeat_run_count`: How many times to repeat each hyperparameter config (default: `3`)
- `--num_workers_per_node`: Parallel workers per GPU node (default: `1`)

## Creating Custom Search Spaces

Create a new config class in `hyperparameter_tuning/` following Isaac Lab's `JobCfg` pattern:

```python
from ray import tune

class MyCustomTuneJobCfg:
    """Custom hyperparameter tuning config."""
    
    def __init__(self):
        self.cfg = {
            "runner_args": {
                # Base YAML config to load
                "--config": "experiments/my_custom_config.yaml",
            },
            "hydra_args": {
                # Hyperparameters to tune (these override values from YAML)
                "agent.learning_rate": tune.loguniform(1e-5, 3e-4),
                "agent.batch_size": tune.choice([256, 512, 1024]),
                "agent.gamma": tune.uniform(0.95, 0.995),
                # Can also tune environment parameters
                "env.scene.num_envs": tune.choice([2048, 4096, 8192]),
            },
        }
```

## Notes

- **Workflow Path**: You must specify `--workflow /workspace/klask_rl/scripts/sb3/train_sac.py` because Isaac Lab defaults to `rl_games/train.py`
- **Config Path**: Your YAML config is specified in `runner_args["--config"]` (relative to `scripts/sb3/config/`)
- **Hydra Overrides**: Only override fields that exist in the base agent config (Hydra struct mode)
  - ✅ `agent.learning_rate` (exists in sb3_sac_cfg.yaml)
  - ❌ `agent.new_field` (doesn't exist → error)
- **Search Algorithm**: Ray Tune uses Optuna for intelligent Bayesian optimization
- **Results**: Saved to `/tmp/ray/IsaacRay-<cfg_class>-tune/` (local mode)
- **Metrics**: Must match tensorboard scalar keys exactly (e.g., `rollout/ep_rew_mean` in SB3)
- **Multiple Runs**: Use `--repeat_run_count` to run each hyperparameter config multiple times for statistical significance

### Common Metrics to Optimize

For SB3 SAC training, useful tensorboard metrics include:
- `rollout_ep_rew_mean` - Average episode reward (maximize)
- `rollout_success_rate` - Success rate if using goal-based env (maximize)
- `train_actor_loss` - Actor loss (minimize, but not primary metric)
- `time_fps` - Training speed (maximize for efficiency)

### Debugging

**Check Command Generation**: The tuner prints the generated command before executing:
```
[INFO]: Recovered invocation with <full command>
```

**Verify Hydra Overrides**: Your training script logs:
```
[INFO] Hydra overrides: ['agent.learning_rate=0.0001', ...]
```

**Monitor Ray Progress**:
- Local mode results: `/tmp/ray/IsaacRay-<cfg_class>-tune/`
- Tensorboard: `tensorboard --logdir /tmp/ray/IsaacRay-<cfg_class>-tune/`
- Ray dashboard: http://localhost:8265

**Common Issues**:
1. **"Field not found" error**: Overriding a field that doesn't exist in base config
2. **Type mismatch error**: Passing wrong type (e.g., string instead of float)
3. **Process timeout**: Training is too slow or crashed - check GPU usage
4. **LogExtractionError**: Tuner couldn't find experiment name in logs - check train_sac.py output format
