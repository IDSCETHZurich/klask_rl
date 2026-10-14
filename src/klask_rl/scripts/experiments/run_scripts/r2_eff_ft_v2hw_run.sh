#!/bin/bash
# Fine-tune of r2_pfsp/exact_eff_ft onto the v2 hardware with a speed curriculum
# (config-sprite-r2-pfsp-eff-v2hw-ft.yaml): exact arm, efficient self-play, everything as exact_eff_ft except the start
# (full trainer state of a COPY of exact_eff_ft/latest.pt, fresh buffer, fresh PFSP pool) and the hardware (v2 actuator
# from the first step; max_velocity 0.4 -> 1.4 m/s and max_acceleration 10 -> 35 m/s^2 over [START, START + RAMP], then
# full v2), for RUN env steps. Inside the allgpus container:
#   GPU=2 setsid nohup bash scripts/experiments/run_scripts/r2_eff_ft_v2hw_run.sh > /dev/null 2>&1 &
# 1. First launch only (no <logdir>/latest.pt and no init/COPY_DONE): copies SRC to <logdir>/init/exact_eff_ft_latest.pt
#    (cp to .part, byte-compare, rename) and checks that its step is START. SRC and its run are only read.
# 2. Waits until the GPU's used memory is below FREE_MIB, then trains. The trainer resumes from <logdir>/latest.pt (+ its
#    own saved replay buffer) whenever that exists, so a restart after a crash (up to MAX_RESTARTS) continues the run and
#    the schedule at the checkpoint step; the init copy is only used while no latest.pt exists.
# STATUS lines: $T/STATUS. Console: $T/$NAME.launch.log and <logdir>/console.log.
# Env: GPU (required), C (code copy), FREE_MIB, START, RAMP, RUN, MAX_RESTARTS. For the smoke test only: T, NAME, SRC,
# WANDB_MODE, EXTRA (more Hydra overrides), RESTART_SLEEP; DRY_RUN=1 prints the trainer command instead of running it.
set -u
GPU=${GPU:?GPU index}
C=${C:-$(cd "$(dirname "$0")/../../.." && pwd)}
FREE_MIB=${FREE_MIB:-4000}
T=${T:-/workspace/klask_rl/logs/dreamer/training/r2_pfsp}
NAME=${NAME:-eff_ft_v2hw_ramp}
SRC=${SRC:-/workspace/klask_rl/logs/dreamer/training/r2_pfsp/exact_eff_ft/latest.pt}
START=${START:-10404928}            # step of the source checkpoint (verified); the step counter continues from it
RAMP=${RAMP:-1000000}               # ramp [START, START + RAMP] in absolute env steps
RUN=${RUN:-3000000}                 # fine-tune length: env.steps = START + RUN
MAX_RESTARTS=${MAX_RESTARTS:-3}
RESTART_SLEEP=${RESTART_SLEEP:-180}
WANDB_MODE=${WANDB_MODE:-online}
EXTRA=${EXTRA:-}
DRY_RUN=${DRY_RUN:-0}
PY=${PY:-/workspace/isaaclab/_isaac_sim/python.sh}
D=$T/$NAME
INIT=$D/init/exact_eff_ft_latest.pt
RAMP_END=$((START + RAMP)); END=$((START + RUN))
mkdir -p "$T"
status() { echo "$(date -u) $NAME: $*" >> "$T/STATUS"; }
cd "$C" || { echo "no code copy $C" >&2; exit 1; }
if [ "$DRY_RUN" != 1 ] && pgrep -f "train_dreamer.py.*logdir=$D( |$)" > /dev/null; then
  status "a trainer with logdir=$D is already running; not started"; exit 1
fi

# --- 1. The start checkpoint copy (first launch only) ---
if [ ! -f "$D/latest.pt" ] && [ ! -f "$D/init/COPY_DONE" ]; then
  case "$(readlink -f "$SRC")" in "$(readlink -f "$D")"/*) status "SRC lies inside the run's logdir; not started"; exit 1 ;; esac
  [ -f "$SRC" ] || { status "source $SRC missing; not started"; exit 1; }
  mkdir -p "$D/init"
  status "copying the start checkpoint $SRC -> $INIT (source only read)"
  cp "$SRC" "$INIT.part" && cmp -s "$SRC" "$INIT.part" || { status "copy of $SRC failed or differs; not started"; exit 1; }
  mv "$INIT.part" "$INIT"
  step=$(CUDA_VISIBLE_DEVICES="" PYTHONPATH=$C/source/klask_rl "$PY" scripts/dreamer/run_init.py --print-step "$INIT" 2>/dev/null | tail -1)
  if [ "$step" != "$START" ]; then status "the copy is at step '${step}', expected START=$START; not started"; exit 1; fi
  echo "$SRC -> $INIT: step $step, $(stat -c %s "$INIT") bytes, copied $(date -u)" > "$D/init/SOURCE"
  touch "$D/init/COPY_DONE"
  status "start checkpoint copied (step $step, byte-identical to the source)"
fi

# --- 2. Train (restart after a crash from the run's own latest.pt) ---
CMD=(env PYTHONPATH=$C/source/klask_rl WANDB_MODE=$WANDB_MODE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "$PY" scripts/dreamer/train_dreamer.py --device cuda:$GPU --config-name=config-sprite-r2-pfsp-eff-v2hw-ft
  wandb_name=r2_pfsp_$NAME logdir=$D device=cuda:$GPU sim_device=cuda:$GPU "train_devices=[cuda:$GPU]"
  micro_batch_size=10 env.steps=$END opponent_separation.enabled=true model.imag_opponent_recompute=true
  finetune_from.checkpoint=$INIT finetune_from.expect_step=$START finetune_from.source=$SRC
  "env.hardware_schedule.steps=[$START,$RAMP_END]")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && CMD+=($EXTRA)
if [ "$DRY_RUN" = 1 ]; then printf '%s\n' "${CMD[@]}"; exit 0; fi
tries=0
while true; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -lt "$FREE_MIB" ]; do sleep 60; done
  if [ -f "$D/latest.pt" ]; then how="resume from its own latest.pt + saved buffer"; else how="fine-tune start from $INIT (fresh buffer, fresh pool)"; fi
  status "starting on cuda:$GPU (try $tries; $how; code $(cat "$C/CODE_VERSION" 2>/dev/null); ramp [$START, $RAMP_END], end $END)"
  "${CMD[@]}" >> "$T/$NAME.launch.log" 2>&1
  code=$?
  step=$(ls "$D"/checkpoint_*.pt 2>/dev/null | sed 's/.*checkpoint_\([0-9]*\).pt/\1/' | sort -n | tail -1)
  status "exited (code $code, last checkpoint ${step:-none})"
  if [ "${step:-0}" -ge "$END" ]; then status "finished"; break; fi
  tries=$((tries + 1)); [ $tries -gt "$MAX_RESTARTS" ] && { status "giving up after $MAX_RESTARTS restarts"; break; }
  sleep "$RESTART_SLEEP"
done
