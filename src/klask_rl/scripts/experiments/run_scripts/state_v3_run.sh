#!/bin/bash
# State-based DreamerV3 self-play runs (12-value state input instead of pixels), the arms one after the other on one GPU,
# W&B online. Inside the Isaac Lab container:
#   GPU=0 ARMS="1m_s80_eff_shared" setsid nohup bash scripts/experiments/run_scripts/state_v3_run.sh &
# Base recipe (config-sprite-v3-state-1m-eff.yaml): state in units of order 1, AR(1) exploration noise, event-anchored
# replay, the initial push, efficient self-play (PFSP on decisive games, Thompson selection, 2 s dead-ball truncation),
# 1M-step replay buffer, 1M env steps. Hardware v1 unless the arm says v2 (config-sprite-v3-state-1m-v2sim-eff.yaml).
# Waits until the GPU's used memory is below MAX_USED (default 3000 MiB; set it higher to start next to another run),
# resumes from latest.pt after a crash (up to 3 times per arm). Logs: $T/<arm>, STATUS there.
# Env: GPU (required), ARMS, C (code directory, default this checkout's src/klask_rl), T (log root), P (python), MAX_USED,
# EXTRA (Hydra overrides appended to every arm).
set -u
GPU=${GPU:?}; ARMS=${ARMS:-"1m_s80_eff_shared"}
C=${C:-$(cd "$(dirname "$0")/../../.." && pwd)}
P=${P:-/workspace/isaaclab/_isaac_sim/python.sh}
T=${T:-/workspace/klask_rl/logs/dreamer/training/v3_state}
STEPS=1000000
EXTRA=${EXTRA:-}
mkdir -p $T
status() { echo "$(date -u) $1: $2" >> $T/STATUS; }
cd $C
V1=config-sprite-v3-state-1m-eff; V2=config-sprite-v3-state-1m-v2sim-eff
SHARED="model.imag_opponent_world=shared"   # shared-world imagination: the imagined opponent observes our decoded state
DYNBP="model.actor_grad=dynamics"           # actor gradient through the frozen world model's imagination (dynamics backprop)
NONOISE="model.explore_noise.tau=0.0"       # white instead of AR(1) exploration noise
for ARM in $ARMS; do
  RATIO=80; CONFIG=$V1; ARM_EXTRA=""; ARM_STEPS=$STEPS; ARM_SEED=0
  case $ARM in
    # v1 hardware
    1m_s80_eff) ;;                                                             # base recipe, train ratio 80
    1m_s512_eff) RATIO=512 ;;                                                  # base recipe, train ratio 512
    1m_s80_eff_nonoise) ARM_EXTRA="$NONOISE" ;;
    1m_s80_eff_dynbp) ARM_EXTRA="$DYNBP" ;;
    1m_s80_eff_nonoise_dynbp) ARM_EXTRA="$NONOISE $DYNBP" ;;
    1m_s80_eff_shared) ARM_EXTRA="$SHARED" ;;                                  # default for new v1 state runs
    1m_s80_eff_shared_imag32) ARM_EXTRA="$SHARED model.imag_horizon=32" ;;     # imagination 15 -> 32 steps (0.3 -> 0.64 s)
    1m_s80_eff_shared_ent) ARM_EXTRA="$SHARED model.act_entropy=1e-3" ;;       # entropy bonus 3e-4 -> 1e-3
    1m_s80_eff_shared_buf500k) ARM_EXTRA="$SHARED buffer.max_size=500000" ;;   # replay buffer size tests
    1m_s80_eff_shared_buf250k) ARM_EXTRA="$SHARED buffer.max_size=250000" ;;
    1m_s80_eff_shared_buf100k) ARM_EXTRA="$SHARED buffer.max_size=100000" ;;
    1m_s80_eff_shared_dynbp) ARM_EXTRA="$SHARED $DYNBP" ;;
    1m_s80_eff_shared_dynbp_imag32) ARM_EXTRA="$SHARED $DYNBP model.imag_horizon=32" ;;
    1m_s512_eff_shared_dynbp_imag32) RATIO=512; ARM_EXTRA="$SHARED $DYNBP model.imag_horizon=32" ;;
    2m_s80_eff_shared_dynbp_imag32) ARM_EXTRA="$SHARED $DYNBP model.imag_horizon=32"; ARM_STEPS=2000000 ;;
    2m_s256_eff_shared) RATIO=256; ARM_EXTRA="$SHARED"; ARM_STEPS=2000000 ;;
    # v2 (deploy) hardware: 1.4 m/s, 35 m/s^2, parametric actuator lag
    v2_1m_s80_eff) CONFIG=$V2 ;;
    v2_1m_s80_eff_shared) CONFIG=$V2; ARM_EXTRA="$SHARED" ;;
    v2_1m_s80_eff_nonoise) CONFIG=$V2; ARM_EXTRA="$NONOISE" ;;
    v2_1m_s80_eff_nonoise_dynbp) CONFIG=$V2; ARM_EXTRA="$NONOISE $DYNBP" ;;
    *) status $ARM "unknown arm"; continue ;;
  esac
  tries=0
  while true; do
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)" -lt ${MAX_USED:-3000} ]; do sleep 120; done
    status $ARM "starting on cuda:$GPU (try $tries, code $(cat $C/CODE_VERSION 2>/dev/null); ratio $RATIO; seed $ARM_SEED; $CONFIG; $ARM_EXTRA $EXTRA)"
    PYTHONPATH=$C/source/klask_rl WANDB_MODE=online PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $P scripts/dreamer/train_dreamer.py --device cuda:$GPU --config-name=$CONFIG \
      wandb_name=v3_state_$ARM logdir=$T/$ARM seed=$ARM_SEED device=cuda:$GPU sim_device=cuda:$GPU "train_devices=[cuda:$GPU]" \
      opponent_separation.enabled=true model.imag_opponent_recompute=true env.train_ratio=$RATIO env.state_units=true env.steps=$ARM_STEPS ++buffer.save_when_finished=false $ARM_EXTRA $EXTRA \
      >> $T/$ARM.launch.log 2>&1
    code=$?
    step=$(ls $T/$ARM/checkpoint_*.pt 2>/dev/null | sed 's/.*checkpoint_\([0-9]*\).pt/\1/' | sort -n | tail -1)
    status $ARM "exited (code $code, last checkpoint ${step:-none})"
    if [ "${step:-0}" -ge $ARM_STEPS ]; then status $ARM "finished"; break; fi
    tries=$((tries + 1)); [ $tries -gt 3 ] && { status $ARM "giving up"; break; }
    sleep 180
  done
done
