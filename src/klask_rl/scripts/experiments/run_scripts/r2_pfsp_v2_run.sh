#!/bin/bash
# R2 pilot v2 from scratch (config-sprite-r2-pfsp-eff-v2.yaml), 10M env steps, seed SEED, one GPU per arm, W&B online.
# The arms differ ONLY in the opponent input or ONE training change; all keep trajectory mirroring (set in the config):
#   exact          - D-RSSM: opponent_separation.enabled=true model.imag_opponent_recompute=true
#   vanilla        - no opponent input: opponent_separation.enabled=false
#   exact_dynbp    - exact + model.actor_grad=dynamics (actor gradient through the frozen world model's imagination,
#                    DreamerV1/V2; the imagined opponent under no_grad, its command a constant input; checkpointing on)
#   vanilla_dynbp  - vanilla + model.actor_grad=dynamics
#   exact_events   - exact with event-anchored replay instead of learned sequence priorities (25% of each batch's windows
#                    contain a stored goal, 65% a ball-contact onset at least 8 steps in, 10% uniform)
# Waits until its GPU is below FREE_MIB, then trains; restarts from latest.pt (+ saved buffer) after a crash, up to 3
# times. Logs: $T/<arm>_s<seed>, STATUS there. Optional: MICRO (micro_batch_size, default 10; 640 samples per Barlow
# cross-correlation), EXTRA_OVR (Hydra overrides appended last), C (code directory, default this checkout's src/klask_rl).
# Inside the Isaac Lab container:
#   ARM=exact GPU=1 setsid nohup bash scripts/experiments/run_scripts/r2_pfsp_v2_run.sh &
set -u
ARM=${ARM:?exact|vanilla|exact_dynbp|vanilla_dynbp|exact_events}; GPU=${GPU:?}; SEED=${SEED:-0}; C=${C:-$(cd "$(dirname "$0")/../../.." && pwd)}
STEPS=${STEPS:-10000000}; FREE_MIB=${FREE_MIB:-2000}; MICRO=${MICRO:-10}; EXTRA_OVR=${EXTRA_OVR:-}
case $ARM in
  exact) OVR="opponent_separation.enabled=true model.imag_opponent_recompute=true" ;;
  vanilla) OVR="opponent_separation.enabled=false" ;;
  exact_dynbp) OVR="opponent_separation.enabled=true model.imag_opponent_recompute=true model.actor_grad=dynamics" ;;
  vanilla_dynbp) OVR="opponent_separation.enabled=false model.actor_grad=dynamics" ;;
  exact_events) OVR="opponent_separation.enabled=true model.imag_opponent_recompute=true buffer.prioritized.mode=events buffer.prioritized.alpha=1.0 ++buffer.prioritized.events.terminal_frac=0.25 ++buffer.prioritized.events.contact_frac=0.65 ++buffer.prioritized.events.key=event ++buffer.prioritized.events.min_offset=8" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
T=${T:-/workspace/klask_rl/logs/dreamer/training/r2_pfsp_v2}; NAME=${ARM}_s${SEED}
mkdir -p $T
status() { echo "$(date -u) $NAME: $1" >> $T/STATUS; }
cd $C
status "queued on GPU $GPU (code $(cat $C/CODE_VERSION 2>/dev/null)); waiting until it is below $FREE_MIB MiB"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)" -lt $FREE_MIB ]; do sleep 60; done
tries=0
while true; do
  status "starting on cuda:$GPU (try $tries; micro_batch_size=$MICRO $OVR${EXTRA_OVR:+ $EXTRA_OVR})"
  PYTHONPATH=$C/source/klask_rl WANDB_MODE=online PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /workspace/isaaclab/_isaac_sim/python.sh scripts/dreamer/train_dreamer.py --device cuda:$GPU --config-name=config-sprite-r2-pfsp-eff-v2 \
    wandb_name=r2_pfsp_v2_$NAME logdir=$T/$NAME seed=$SEED device=cuda:$GPU sim_device=cuda:$GPU "train_devices=[cuda:$GPU]" \
    micro_batch_size=$MICRO env.steps=$STEPS $OVR $EXTRA_OVR >> $T/$NAME.launch.log 2>&1
  code=$?
  step=$(ls $T/$NAME/checkpoint_*.pt 2>/dev/null | sed 's/.*checkpoint_\([0-9]*\).pt/\1/' | sort -n | tail -1)
  status "exited (code $code, last checkpoint ${step:-none})"
  if [ "${step:-0}" -ge $STEPS ]; then status "finished"; break; fi
  tries=$((tries + 1)); [ $tries -gt 3 ] && { status "giving up after 3 restarts"; break; }
  sleep 180
done
