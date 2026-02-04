# Ray Tune Integration for Klask RL

This directory contains the Ray Tune hyperparameter tuning setup for Klask RL, following the Isaac Lab Ray integration pattern.

## Structure

```
ray/
├── hyperparameter_tuning/
│   └── klask_sac_base_cfg.py  # Job config classes defining search spaces
├── tuner.py                    # Main Ray Tune runner
└── util.py                     # Utility functions (command generation, log parsing)
```

## Usage

### Local Hyperparameter Tuning

Run a hyperparameter sweep locally:

```bash
python scripts/sb3/ray/tuner.py --run_mode local \
    --cfg_file scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py \
    --cfg_class KlaskSacHerTuneJobCfg \
    --num_samples 8
```

### Available Config Classes

- `KlaskSacBaseJobCfg`: Base SAC (no tuning)
- `KlaskSacHerJobCfg`: HER SAC (no tuning)
- `KlaskSacHerTuneJobCfg`: HER SAC with hyperparameter tuning
- `KlaskSacTwoStageHerJobCfg`: Two-stage HER SAC (no tuning)

### Command Line Arguments

- `--run_mode`: Run mode (default: `local`)
- `--cfg_file`: Path to config file (required)
- `--cfg_class`: Config class name (required)
- `--num_samples`: Number of hyperparameter samples to try (default: `1`)
- `--metric`: Metric to optimize from tensorboard logs (default: `rollout_ep_rew_mean`)
- `--mode`: Optimization mode `min` or `max` (default: `max`)

## How It Works

1. **Config Class**: Defines `runner_args` (CLI flags like `--config`) and `hydra_args` (Hydra overrides for hyperparameters)
2. **Tuner**: Samples hyperparameters using Ray Tune and Optuna search
3. **Training**: Each trial runs `scripts/sb3/train_sac.py` as a subprocess with sampled Hydra overrides
4. **Logging**: Reads tensorboard logs from training runs to report metrics back to Ray

## Creating Custom Search Spaces

Create a new config class in `hyperparameter_tuning/`:

```python
from ray import tune

class MyCustomJobCfg:
    def __init__(self, cfg: dict = {}):
        self.runner_args = {
            "--config": "experiments/klask_sac_her.yaml",
        }
        
        self.hydra_args = {
            "agent.learning_rate": tune.loguniform(1e-5, 3e-4),
            "agent.batch_size": tune.choice([256, 512, 1024]),
            "agent.gamma": tune.uniform(0.95, 0.995),
        }
    
    def to_dict(self) -> dict:
        return {
            "runner_args": self.runner_args,
            "hydra_args": self.hydra_args,
        }
```

## Notes

- Only Hydra overrides for fields that exist in the base agent config can be tuned (due to Hydra struct mode)
- Ray Tune uses Optuna for intelligent search (Bayesian optimization)
- Results are saved to `~/ray_results/klask_sac_tune/`
