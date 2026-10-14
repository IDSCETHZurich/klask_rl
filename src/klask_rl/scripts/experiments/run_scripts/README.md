# Training launchers

All launchers run inside the Isaac Lab container, wait until their GPU has room, train with W&B online, restart from
the run's own `latest.pt` (+ saved replay buffer) after a crash (up to 3 times) and append progress lines to a `STATUS`
file next to the run directories. The code directory `C` defaults to this checkout's `src/klask_rl`; log roots default
to `/workspace/klask_rl/logs/dreamer/training/...` and can be changed with `T`.

| Launcher | Runs | Example |
| --- | --- | --- |
| `r2_pfsp_v2_run.sh` | Pixel R2 pilot v2 (`config-sprite-r2-pfsp-eff-v2`), 10M steps, arms `exact`, `vanilla`, `exact_dynbp`, `vanilla_dynbp`, `exact_events` | `ARM=exact GPU=0 bash r2_pfsp_v2_run.sh` |
| `state_v3_run.sh` | State-based runs (`config-sprite-v3-state-1m-eff`, `...-1m-v2sim-eff`), arms listed in the script | `GPU=0 ARMS="1m_s80_eff_shared" bash state_v3_run.sh` |
| `r2_eff_ft_dynbp_run.sh` | Continues `r2_pfsp/exact_eff_ft` from copies of its checkpoint, pool and buffer with `model.actor_grad=dynamics` | `GPU=0 bash r2_eff_ft_dynbp_run.sh` |
| `r2_eff_ft_v2hw_run.sh` | Fine-tune of `r2_pfsp/exact_eff_ft` onto the v2 hardware with a speed curriculum (fresh buffer and pool); `r2_eff_ft_v2hw_smoke.sh` is its short GPU smoke test | `GPU=1 bash r2_eff_ft_v2hw_run.sh` |

Start them detached, e.g. `GPU=0 ARMS="1m_s80_eff_shared" setsid nohup bash scripts/experiments/run_scripts/state_v3_run.sh > /dev/null 2>&1 &`.
The fine-tune launchers copy their source files once into the new run directory and never modify the source run.
