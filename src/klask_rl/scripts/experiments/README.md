# Experiment tools

| Path | Purpose |
| --- | --- |
| [`run_scripts/`](run_scripts/README.md) | Launchers of the self-play training runs (pixel R2 pilot v2, state runs, fine-tunes). |
| `learner_memory_probe.py` | Measures the GPU memory of one learner update for a config and micro-batch before a launch (does the run fit next to others on one GPU?). |
| `render_recorded_games.py` | Renders tournament games recorded with `record_states` to MP4 (one file per game and one combined). |
| `tests/` | CPU tests of the launchers, PFSP pool, efficient self-play, actuator lag, state-unit wrappers, shared-imagination seat map and the fine-tune starts. |

Run the tests inside the Isaac Lab container, without a GPU:

```bash
cd /workspace/klask_rl && CUDA_VISIBLE_DEVICES= /workspace/isaaclab/_isaac_sim/python.sh -m pytest -q scripts/experiments/tests
```

The model-side tests live in the `r2dreamer` submodule (`scripts/dreamer/r2dreamer/tests`).
