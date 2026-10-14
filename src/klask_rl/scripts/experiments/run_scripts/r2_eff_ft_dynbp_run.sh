#!/bin/bash
# Fine-tune of r2_pfsp/exact_eff_ft with dynamics backpropagation for the actor (model.actor_grad=dynamics; the default
# reinforce is bit-identical to the code exact_eff_ft ran with). The run is exact_eff_ft CONTINUED, as exact_eff_ft itself was
# started from the pilot: COPIES of its latest.pt (full trainer state, step START), its PFSP pool and its saved replay buffer
# in a new logdir, the same trainer overrides as exact_eff_ft (gpu2_pilot_eff_ft_v2.sh), and ONE change:
# model.actor_grad=dynamics (pure dynamics backpropagation, as r2_pfsp_v2/exact_dynbp_s0 and the state dynbp runs).
# Inside the allgpus container:
#   GPU=0 setsid nohup bash scripts/experiments/run_scripts/r2_eff_ft_dynbp_run.sh > /dev/null 2>&1 &
# 1. First launch only (no <logdir>/COPY_DONE): refuses if the logdir or <logdir>.copying already exists (a partial copy), if the
#    source run is training, if its latest.pt is not at step START, or if the disk has less than MIN_FREE_GB free. Copies into <logdir>.copying
#    (the source is only read), checks sizes (latest.pt byte-compared), then renames it to <logdir>; COPY_DONE is the last file.
# 2. Waits until the GPU's used memory is below FREE_MIB, then trains. The trainer resumes from <logdir>/latest.pt + the buffer
#    named in replay_buffer.latest (the copy at first, its own exit save after a pause or crash), up to MAX_RESTARTS restarts.
#    The logdir has no W&B folder at the first start, so W&B opens a new run (wandb_name=r2_pfsp_$NAME).
# STATUS lines: $T/STATUS. Console: $T/$NAME.launch.log and <logdir>/console.log.
# Env: GPU (required), C (code copy), FREE_MIB, START, RUN, MIN_FREE_GB, MAX_RESTARTS. For tests: T, NAME, SRC, PY, WANDB_MODE,
# EXTRA (more Hydra overrides), RESTART_SLEEP, COPY_ONLY=1 (stop after the copy); DRY_RUN=1 prints the trainer command.
set -u
GPU=${GPU:?GPU index}
C=${C:-$(cd "$(dirname "$0")/../../.." && pwd)}
FREE_MIB=${FREE_MIB:-4000}
T=${T:-/workspace/klask_rl/logs/dreamer/training/r2_pfsp}
NAME=${NAME:-eff_ft_dynbp}
SRC=${SRC:-/workspace/klask_rl/logs/dreamer/training/r2_pfsp/exact_eff_ft}   # source run directory (only read)
START=${START:-10404928}            # step of the source's latest.pt (exact_eff_ft paused there 2026-10-12)
RUN=${RUN:-2000000}                 # fine-tune length: env.steps = START + RUN (exact_eff_ft was planned for 2M too)
MIN_FREE_GB=${MIN_FREE_GB:-120}     # the copy needs ~62 GB (buffer 56, pool 4.4, checkpoint 1.3)
MAX_RESTARTS=${MAX_RESTARTS:-3}
RESTART_SLEEP=${RESTART_SLEEP:-180}
WANDB_MODE=${WANDB_MODE:-online}
EXTRA=${EXTRA:-}
DRY_RUN=${DRY_RUN:-0}
COPY_ONLY=${COPY_ONLY:-0}
PY=${PY:-/workspace/isaaclab/_isaac_sim/python.sh}
D=$T/$NAME
END=$((START + RUN))
mkdir -p "$T"
status() { echo "$(date -u) $NAME: $*" >> "$T/STATUS"; }

CMD=(env PYTHONPATH=$C/source/klask_rl WANDB_MODE=$WANDB_MODE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "$PY" scripts/dreamer/train_dreamer.py --device cuda:$GPU --config-name=config-sprite-r2-pfsp-eff
  wandb_name=r2_pfsp_$NAME logdir=$D device=cuda:$GPU sim_device=cuda:$GPU "train_devices=[cuda:$GPU]"
  micro_batch_size=10 env.steps=$END opponent_separation.enabled=true model.imag_opponent_recompute=true
  model.actor_grad=dynamics)
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && CMD+=($EXTRA)
if [ "$DRY_RUN" = 1 ]; then printf '%s\n' "${CMD[@]}"; exit 0; fi
cd "$C" || { echo "no code copy $C" >&2; exit 1; }
if pgrep -f "train_dreamer.py.*logdir=$D( |$)" > /dev/null; then
  status "a trainer with logdir=$D is already running; not started"; exit 1
fi

# --- 1. The start copy (first launch only) ---
if [ ! -f "$D/COPY_DONE" ]; then
  if [ -e "$D" ]; then status "logdir $D exists without COPY_DONE (partial copy?); not started"; exit 1; fi
  case "$(readlink -f "$D")/" in "$(readlink -f "$SRC")"/*) status "logdir lies inside the source run; not started"; exit 1 ;; esac
  if pgrep -f "train_dreamer.py.*logdir=$SRC( |$)" > /dev/null; then status "the source run $SRC is training; not started"; exit 1; fi
  BUF=$(cat "$SRC/replay_buffer.latest" 2>/dev/null)
  for f in "$SRC/latest.pt" "$SRC/pfsp" "$SRC/${BUF:-none}"; do
    [ -e "$f" ] || { status "source file $f missing; not started"; exit 1; }
  done
  step=$(CUDA_VISIBLE_DEVICES="" "$PY" -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)['step'])" "$SRC/latest.pt" 2>/dev/null | tail -1)
  if [ "$step" != "$START" ]; then status "the source latest.pt is at step '${step}', expected START=$START; not started"; exit 1; fi
  free=$(df -P -BG "$T" | awk 'NR == 2 {print $4 + 0}')
  if [ "$free" -lt "$MIN_FREE_GB" ]; then status "only ${free} GB free on the disk (need $MIN_FREE_GB); not started"; exit 1; fi
  P=$D.copying
  if [ -e "$P" ]; then status "a partial copy $P exists (interrupted earlier copy; remove it by hand); not started"; exit 1; fi
  status "copying $SRC (latest.pt step $step, pfsp, $BUF) -> $D (source only read; ${free} GB free)"
  mkdir -p "$P" && cp -r "$SRC/pfsp" "$P/pfsp" && cp -r "$SRC/$BUF" "$P/replay_buffer" && cp "$SRC/latest.pt" "$P/latest.pt" \
    || { status "copy failed; not started (partial copy left in $P)"; exit 1; }
  if ! cmp -s "$SRC/latest.pt" "$P/latest.pt" || [ "$(du -sb "$SRC/$BUF" | cut -f1)" != "$(du -sb "$P/replay_buffer" | cut -f1)" ] \
     || [ "$(du -sb "$SRC/pfsp" | cut -f1)" != "$(du -sb "$P/pfsp" | cut -f1)" ]; then
    status "the copy differs from the source; not started (left in $P)"; exit 1
  fi
  echo replay_buffer > "$P/replay_buffer.latest"
  echo "$SRC: latest.pt step $step ($(stat -c %s "$SRC/latest.pt") bytes), pfsp, $BUF -> replay_buffer; copied $(date -u)" > "$P/SOURCE"
  touch "$P/COPY_DONE"
  mv "$P" "$D"
  status "start copy done (latest.pt byte-identical, buffer and pool same size as the source)"
fi
[ "$COPY_ONLY" = 1 ] && exit 0

# --- 2. Train (resume from the run's own latest.pt + buffer) ---
tries=0
while true; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")" -lt "$FREE_MIB" ]; do sleep 60; done
  status "starting on cuda:$GPU (try $tries; resume from $D/latest.pt + $(cat "$D/replay_buffer.latest"); code $(cat "$C/CODE_VERSION" 2>/dev/null); model.actor_grad=dynamics; end $END)"
  "${CMD[@]}" >> "$T/$NAME.launch.log" 2>&1
  code=$?
  step=$(ls "$D"/checkpoint_*.pt 2>/dev/null | sed 's/.*checkpoint_\([0-9]*\).pt/\1/' | sort -n | tail -1)
  status "exited (code $code, last checkpoint ${step:-none})"
  if [ "${step:-0}" -ge "$END" ]; then status "finished"; break; fi
  tries=$((tries + 1)); [ $tries -gt "$MAX_RESTARTS" ] && { status "giving up after $MAX_RESTARTS restarts"; break; }
  sleep "$RESTART_SLEEP"
done
