#!/bin/bash
# GPU smoke test of r2_eff_ft_v2hw_run.sh (W&B disabled; scratch logdir logs/experiments/r2_ft_v2hw_smoke/<NAME>), two
# launches of the real launcher with a short ramp (8k steps from 10,404,928) so the hardware values visibly change:
#   A) fine-tune start from a COPY of exact_eff_ft/latest.pt, fresh buffer + 1-member pool; stops after 4k steps (ramp ~50%)
#   B) the same launcher to 10k steps: resumes from the smoke run's OWN latest.pt + saved buffer, the ramp continues
#      from ~50% and completes (full v2 for the last ~2k steps)
# Smoke-only overrides: rotations every 2k steps, PFSP snapshot every 4k, logs every 1k. ~30-40 min, ~7 GB of files.
# Inside the allgpus container:
#   GPU=0 bash scripts/experiments/run_scripts/r2_eff_ft_v2hw_smoke.sh
set -u
GPU=${GPU:?GPU index}
C=${C:-$(cd "$(dirname "$0")/../../.." && pwd)}
NAME=${NAME:-smoke}
S=/workspace/klask_rl/logs/experiments/r2_ft_v2hw_smoke
D=$S/$NAME
[ -e "$D" ] && { echo "$D exists; pick another NAME=..." >&2; exit 1; }
L=$C/scripts/experiments/run_scripts/r2_eff_ft_v2hw_run.sh
EXTRA="trainer.eval_every=2000 trainer.update_log_every=1000 self_play.pfsp.snapshot_every=4000"
run() {  # $1 = fine-tune length
  GPU=$GPU C=$C T=$S NAME=$NAME WANDB_MODE=disabled RAMP=8000 RUN=$1 MAX_RESTARTS=0 FREE_MIB=${FREE_MIB:-4000} \
    EXTRA="$EXTRA" bash "$L"
}
run 4000
run 10000
echo "== STATUS"; grep -a " $NAME: " "$S/STATUS"
echo "== env chain (launch log: printed before console.log opens)"
grep -a -E "Run start|HardwareScheduleWrapper|ActuatorLag|ActuatorModel|VelocityScaleWrapper|Traceback|Error" "$S/$NAME.launch.log"
echo "== start / restore lines"
grep -a -E "Fine-tune|Resuming from|Restored agent|Restored replay buffer|empty replay buffer|PFSP: (pool|restored)|Hardware schedule bound|TRAINING CRASHED" "$D/console.log"
echo "== hardware values (console)"; grep -a "\[hardware step" "$D/console.log"
echo "== hardware values (metrics.jsonl, logged steps)"
CUDA_VISIBLE_DEVICES="" /workspace/isaaclab/_isaac_sim/python.sh -c '
import json, sys
for line in open(sys.argv[1]):
    d = json.loads(line)
    if "hardware/max_velocity" in d:
        print(d["step"], {k[9:]: round(v, 4) for k, v in d.items() if k.startswith("hardware/")})
' "$D/metrics.jsonl"
echo "== PFSP"; grep -a "PFSP" "$D/console.log" | tail -8
echo "== files"; ls -la "$D" "$D/init" "$D/pfsp"
