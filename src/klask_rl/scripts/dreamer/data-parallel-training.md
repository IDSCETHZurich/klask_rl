# Plan: Data-Parallel Training Across Multiple GPUs

## Context

With the multi-gpu-split plan, GPU 0 handles simulation+inference and GPU 1 handles training. This leaves GPUs 2 and 3 idle. The goal here is to use all three remaining GPUs (1, 2, 3) for training, enabling larger batch sizes and faster updates without exceeding any single GPU's 24GB VRAM.

Depends on: multi-gpu-split.md being implemented first (sim/train device separation).

### Why not standard DDP (torchrun)?

Standard `DistributedDataParallel` requires multi-process launching (one process per GPU). IsaacLab's `AppLauncher` creates a single simulation instance tied to one process — it cannot be split across processes. Running sim on only rank 0 and broadcasting transitions adds complexity (process coordination, data serialization, AppLauncher conflicts with torchrun). A single-process approach avoids all of this.

### Why not `torch.nn.DataParallel`?

`DataParallel` is deprecated, funnels all gradients through one GPU (memory bottleneck), and suffers from the Python GIL. Manual parallelism is cleaner for this use case.

---

## Architecture

```
GPU 0 (sim_device):              GPUs 1-3 (train_devices):
┌──────────────────────┐         ┌────────────────────────────────────────┐
│ IsaacLab (16 envs)   │         │ GPU 1 (primary)  GPU 2    GPU 3       │
│ Inference copies     │         │ ┌────────────┐ ┌────────┐ ┌────────┐  │
│ Self-play opponent   │         │ │ Model      │ │Replica │ │Replica │  │
│                      │         │ │ Optimizer  │ │        │ │        │  │
│ act() runs here      │         │ │ GradScaler │ │        │ │        │  │
│                      │         │ └────────────┘ └────────┘ └────────┘  │
└──────────────────────┘         │       ↑             ↑          ↑      │
         ↑                       │       └─── all-reduce grads ───┘      │
         │ weight sync           │       ↓                               │
         └────────────────────── │  optimizer.step() on primary          │
                                 │       ↓                               │
                                 │  broadcast params → replicas          │
                                 └────────────────────────────────────────┘
```

Single-GPU fallback: when only one training GPU is configured, replicas are not created and the code reduces to the existing micro-batch loop with zero overhead.

---

## Core idea: parallel micro-batches

The existing micro-batch gradient accumulation loop processes chunks sequentially on one GPU:

```python
# Current (dreamer.py lines 385-398)
for i in range(num_acc):
    micro_data = p_data[i*mbs : (i+1)*mbs]
    with autocast(...):
        (stoch, deter), mets = self._cal_grad(micro_data, ...)
```

Instead of looping sequentially, distribute micro-batches across GPUs and run them in parallel:

```python
# Proposed
# 3 training GPUs, batch_size=24, micro_batch_size=8 → 3 micro-batches
# Each GPU processes one micro-batch simultaneously
```

This converts a sequential loop into parallel execution. If you have N training GPUs and N micro-batches, the forward+backward pass runs ~N times faster (minus communication overhead).

---

## Config changes

**Top-level** in `config.yaml` and `vision-pre-training-config.yaml`:
```yaml
device: "cuda:0"
sim_device: null          # from multi-gpu-split plan
train_device: null        # from multi-gpu-split plan
train_devices: null       # NEW: list of training GPUs, e.g. ["cuda:1", "cuda:2", "cuda:3"]
                          # if null → falls back to [train_device]
```

These propagate into sub-configs in `train_dreamer.py`:
- `model.train_devices` → list of training GPU devices
- `model.train_device` → `train_devices[0]` (primary, holds optimizer)
- Batch size auto-adjusted: `effective_batch_size = batch_size * len(train_devices)`

---

## Implementation in `dreamer.py`

### `__init__` additions

```python
# After multi-gpu-split device setup:
self.train_device = torch.device(config.train_device or config.device)
self.sim_device = torch.device(config.sim_device or config.device)

# Data-parallel setup
train_devices_cfg = config.get("train_devices", None)
if train_devices_cfg and len(train_devices_cfg) > 1:
    self.train_devices = [torch.device(d) for d in train_devices_cfg]
else:
    self.train_devices = [self.train_device]
self.data_parallel = len(self.train_devices) > 1
```

### `_create_training_replicas()` — new method

Called after model is created and moved to `train_device` (the primary training GPU). Creates copies on GPUs 2, 3, etc.

```python
def _create_training_replicas(self):
    """Create model replicas on secondary training GPUs for data-parallel training."""
    if not self.data_parallel:
        self._replicas = []
        return

    self._replicas = []
    secondary_devices = self.train_devices[1:]  # skip primary

    for dev in secondary_devices:
        replica = {}
        # Deep-copy all trainable modules to this GPU
        replica["encoder"] = copy.deepcopy(self.encoder).to(dev)
        replica["rssm"] = copy.deepcopy(self.rssm).to(dev)
        replica["rssm"]._device = dev
        replica["actor"] = copy.deepcopy(self.actor).to(dev)
        replica["value"] = copy.deepcopy(self.value).to(dev)
        replica["reward"] = copy.deepcopy(self.reward).to(dev)
        replica["cont"] = copy.deepcopy(self.cont).to(dev)
        if hasattr(self, "decoder"):
            replica["decoder"] = copy.deepcopy(self.decoder).to(dev)
        if hasattr(self, "prj"):
            replica["prj"] = copy.deepcopy(self.prj).to(dev)
        replica["device"] = dev

        # Create frozen copies for this replica (for imagination rollouts)
        replica["frozen"] = {}
        for name in ("encoder", "rssm", "reward", "cont", "actor", "value", "slow_value"):
            if name == "slow_value":
                src = copy.deepcopy(self.slow_value).to(dev)
            else:
                src = replica[name]
            frozen = copy.deepcopy(src)
            for p_orig, p_frozen in zip(src.parameters(), frozen.parameters()):
                p_frozen.data = p_orig.data  # share .data (same as clone_and_freeze)
                p_frozen.requires_grad_(False)
            frozen.eval()
            replica["frozen"][name] = frozen

        # Optionally compile replica modules
        if self._compile:
            for name in ("encoder", "rssm", "actor", "value", "reward", "cont"):
                replica[name] = torch.compile(replica[name], mode="reduce-overhead")
            for name in replica["frozen"]:
                replica["frozen"][name] = torch.compile(
                    replica["frozen"][name], mode="reduce-overhead"
                )

        self._replicas.append(replica)
```

### `_broadcast_params()` — new method

After optimizer step on primary, push updated weights to all replicas.

```python
def _broadcast_params(self):
    """Copy primary model parameters to all replicas."""
    if not self.data_parallel:
        return

    for replica in self._replicas:
        for name in ("encoder", "rssm", "actor", "value", "reward", "cont"):
            primary_module = getattr(self, name)
            replica_module = replica[name]
            # load_state_dict handles cross-device copy
            replica_module.load_state_dict(primary_module.state_dict())
        # Also sync slow_value
        replica["frozen"]["slow_value"].load_state_dict(
            self.slow_value.state_dict()
        )
```

### `_reduce_gradients()` — new method

After all GPUs finish their backward passes, sum gradients from replicas into primary.

```python
def _reduce_gradients(self):
    """Sum gradients from replicas into primary model parameters."""
    if not self.data_parallel:
        return

    for replica in self._replicas:
        for name in ("encoder", "rssm", "actor", "value", "reward", "cont"):
            primary_module = getattr(self, name)
            replica_module = replica[name]
            for p_primary, p_replica in zip(
                primary_module.parameters(), replica_module.parameters()
            ):
                if p_primary.grad is not None and p_replica.grad is not None:
                    # Move replica grad to primary device and accumulate
                    p_primary.grad.add_(p_replica.grad.to(self.train_device))
                elif p_replica.grad is not None:
                    p_primary.grad = p_replica.grad.to(self.train_device)
        # Zero replica gradients for next step
        for name in ("encoder", "rssm", "actor", "value", "reward", "cont"):
            replica[name].zero_grad(set_to_none=True)
```

### `_cal_grad_on_replica()` — new method

Runs `_cal_grad()` logic but using a replica's modules instead of self.

```python
def _cal_grad_on_replica(self, replica, data, initial, loss_scale, opp_initial=None):
    """Run _cal_grad logic on a replica's modules and device."""
    dev = replica["device"]
    # Move data to replica device
    data_dev = data.to(dev)
    initial_dev = (initial[0].to(dev), initial[1].to(dev))
    opp_initial_dev = None
    if opp_initial is not None:
        opp_initial_dev = (opp_initial[0].to(dev), opp_initial[1].to(dev))

    # Temporarily swap self's modules with replica's for _cal_grad
    # (see implementation note below for cleaner approach)
    return self._cal_grad_with_modules(
        modules=replica,
        frozen=replica["frozen"],
        data=data_dev,
        initial=initial_dev,
        loss_scale=loss_scale,
        opp_initial=opp_initial_dev,
    )
```

### Refactoring `_cal_grad()` — key change

Currently `_cal_grad()` (lines 434-629) references `self.encoder`, `self.rssm`, etc. directly. To support replicas, refactor it to accept modules as parameters:

```python
def _cal_grad(self, data, initial, loss_scale, opp_initial=None):
    """Backward pass on primary GPU using self's modules."""
    return self._cal_grad_with_modules(
        modules={
            "encoder": self.encoder,
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "decoder": getattr(self, "decoder", None),
            "prj": getattr(self, "prj", None),
        },
        frozen={
            "reward": self._frozen_reward,
            "cont": self._frozen_cont,
            "value": self._frozen_value,
            "slow_value": self._frozen_slow_value,
        },
        data=data,
        initial=initial,
        loss_scale=loss_scale,
        opp_initial=opp_initial,
    )

def _cal_grad_with_modules(self, modules, frozen, data, initial, loss_scale, opp_initial=None):
    """Core training computation, parameterized by module set."""
    encoder = modules["encoder"]
    rssm = modules["rssm"]
    actor = modules["actor"]
    value = modules["value"]
    reward = modules["reward"]
    cont = modules["cont"]
    frozen_reward = frozen["reward"]
    frozen_cont = frozen["cont"]
    frozen_value = frozen["value"]
    frozen_slow_value = frozen["slow_value"]

    # ... rest of _cal_grad logic, replacing self.encoder with encoder, etc.
    # Loss scales, hyperparameters, and config values still read from self.
    # Only the nn.Module references change.
```

This refactor is the largest code change. It replaces ~20 `self.encoder`/`self.rssm`/etc. references in `_cal_grad` with local variables. The logic itself is unchanged.

### Modified `update()` method

```python
def update(self, replay_buffer, step):
    # ... existing setup, sampling, preprocessing ...

    B = p_data.shape[0]
    mbs = self.micro_batch_size
    num_acc = B // mbs
    loss_scale = 1.0 / num_acc  # each micro-batch contributes equally

    all_stoch, all_deter = [], []

    if not self.data_parallel:
        # ─── Single training GPU: existing sequential micro-batch loop ───
        for i in range(num_acc):
            s, e = i * mbs, (i + 1) * mbs
            micro_data = p_data[s:e]
            micro_initial = (initial[0][s:e], initial[1][s:e])
            micro_opp = (opp_initial[0][s:e], opp_initial[1][s:e]) if opp_initial else None

            with autocast(device_type=self.train_device.type, dtype=torch.float16):
                (stoch, deter), mets = self._cal_grad(
                    micro_data, micro_initial, loss_scale, opp_initial=micro_opp
                )
            all_stoch.append(stoch)
            all_deter.append(deter)
    else:
        # ─── Multi-GPU: distribute micro-batches across GPUs ───
        num_gpus = len(self.train_devices)
        # Assign micro-batches to GPUs round-robin
        gpu_assignments = [[] for _ in range(num_gpus)]
        for i in range(num_acc):
            gpu_assignments[i % num_gpus].append(i)

        # Process primary GPU's micro-batches
        primary_results = []
        for i in gpu_assignments[0]:
            s, e = i * mbs, (i + 1) * mbs
            micro_data = p_data[s:e]
            micro_initial = (initial[0][s:e], initial[1][s:e])
            micro_opp = (opp_initial[0][s:e], opp_initial[1][s:e]) if opp_initial else None

            with autocast(device_type=self.train_device.type, dtype=torch.float16):
                (stoch, deter), mets = self._cal_grad(
                    micro_data, micro_initial, loss_scale, opp_initial=micro_opp
                )
            primary_results.append((i, stoch, deter))

        # Process replica GPUs' micro-batches
        # Note: CUDA kernels on different devices run concurrently in a single
        # process (different CUDA streams per device). We launch all replica
        # work, then synchronize.
        replica_results = []
        for gpu_idx, replica in enumerate(self._replicas, start=1):
            dev = replica["device"]
            for i in gpu_assignments[gpu_idx]:
                s, e = i * mbs, (i + 1) * mbs
                micro_data = p_data[s:e]
                micro_initial = (initial[0][s:e], initial[1][s:e])
                micro_opp = (opp_initial[0][s:e], opp_initial[1][s:e]) if opp_initial else None

                with autocast(device_type=dev.type, dtype=torch.float16):
                    (stoch, deter), mets = self._cal_grad_on_replica(
                        replica, micro_data, micro_initial, loss_scale,
                        opp_initial=micro_opp,
                    )
                # stoch/deter are on replica device — move back for buffer update
                replica_results.append((i, stoch.to(self.train_device), deter.to(self.train_device)))

        # Synchronize all CUDA devices
        for dev in self.train_devices:
            torch.cuda.synchronize(dev)

        # Reduce gradients from replicas into primary
        self._reduce_gradients()

        # Collect results in original micro-batch order for buffer update
        all_results = sorted(primary_results + replica_results, key=lambda x: x[0])
        for _, stoch, deter in all_results:
            all_stoch.append(stoch)
            all_deter.append(deter)

    # ─── Optimizer step (unchanged, runs on primary training GPU) ───
    self._scaler.unscale_(self._optimizer)
    self._agc(self._named_params.values())
    self._scaler.step(self._optimizer)
    self._scaler.update()
    self._scheduler.step()
    self._optimizer.zero_grad(set_to_none=True)

    # ─── Sync weights ───
    if self.data_parallel:
        self._broadcast_params()       # primary → replicas (GPUs 2, 3)
    self._sync_inference_copies()      # train → sim GPU (from multi-gpu-split)

    # ... rest of update (buffer latent update, metrics) ...
```

### `to()` override — updated

```python
def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    self.clone_and_freeze()
    self._create_inference_copies()    # from multi-gpu-split
    self._create_training_replicas()   # new
    return self
```

---

## Implementation in `train_dreamer.py`

### Device resolution

```python
sim_device = getattr(config, "sim_device", None) or config.device
train_device = getattr(config, "train_device", None) or config.device

# Data-parallel: list of training GPUs
train_devices_cfg = getattr(config, "train_devices", None)
if train_devices_cfg:
    train_devices = list(train_devices_cfg)
    train_device = train_devices[0]  # primary
else:
    train_devices = [train_device]
```

### Propagate into sub-configs

```python
with open_dict(config.model):
    config.model.sim_device = sim_device
    config.model.train_device = train_device
    config.model.train_devices = train_devices

# Buffer delivers samples to primary training GPU
with open_dict(config.buffer):
    config.buffer.device = train_device
```

### Batch size scaling

With more training GPUs, you can proportionally increase the batch size since each GPU only holds `batch_size / num_gpus` worth of activations:

```python
# Auto-scale batch size if user hasn't manually adjusted
num_train_gpus = len(train_devices)
if num_train_gpus > 1:
    # Ensure batch_size is divisible by micro_batch_size * num_train_gpus
    # so micro-batches distribute evenly
    assert (config.batch_size % config.model.micro_batch_size == 0), \
        "batch_size must be divisible by micro_batch_size"
    num_micro = config.batch_size // config.model.micro_batch_size
    if num_micro % num_train_gpus != 0:
        # Pad up to nearest multiple
        num_micro = ((num_micro + num_train_gpus - 1) // num_train_gpus) * num_train_gpus
        config.batch_size = num_micro * config.model.micro_batch_size
        logger.info(f"Adjusted batch_size to {config.batch_size} for even GPU distribution")
```

---

## GradScaler considerations

`torch.cuda.amp.GradScaler` is device-local. Replicas running on different GPUs need their own scalers, OR we handle it manually:

**Option A (simpler)**: Disable AMP scaling on replicas, use full fp16 backward without scaling. Risk: gradient underflow on replicas.

**Option B (recommended)**: One GradScaler per device.

```python
# In __init__:
self._scaler = GradScaler()  # primary
if self.data_parallel:
    self._replica_scalers = [GradScaler() for _ in self._replicas]

# In _cal_grad_on_replica:
scaler = self._replica_scalers[replica_idx]
scaler.scale(total_loss * loss_scale).backward()

# In update(), before _reduce_gradients:
for scaler in self._replica_scalers:
    scaler.unscale_(...)  # unscale replica grads before reducing

# After optimizer step:
self._scaler.update()
for scaler in self._replica_scalers:
    scaler.update()
```

**Option C (simplest, may be sufficient)**: Run replicas in fp32 for backward. Slightly more memory per replica but avoids scaler complexity. The primary GPU still uses AMP.

Given the complexity, **Option C is recommended for the initial implementation**. AMP on the primary GPU is enough to keep memory low where it matters (the optimizer states live on the primary). Replicas only hold model params + activations — fp32 backward is fine on 24GB with micro-batches.

```python
# Primary GPU: autocast fp16 + GradScaler (existing behavior)
# Replica GPUs: no autocast, no scaling, fp32 backward
# _cal_grad_on_replica runs WITHOUT autocast and WITHOUT scaler.scale()
# Gradients are naturally fp32, compatible with primary's unscaled grads
```

---

## CUDA stream parallelism

In a single process, CUDA kernels on **different devices** execute concurrently by default (each device has its own default stream). No explicit stream management is needed.

However, multiple operations on the **same device** are sequential on the default stream. If a GPU is assigned multiple micro-batches (e.g., 6 micro-batches across 3 GPUs = 2 per GPU), those two micro-batches on the same GPU run sequentially. This is fine — it matches the current behavior but across more GPUs.

For maximum overlap, ensure each GPU gets roughly equal work:
```python
# Round-robin assignment ensures balance
# 9 micro-batches across 3 GPUs → 3 per GPU
# 7 micro-batches across 3 GPUs → 3, 2, 2
```

---

## Memory budget per GPU

Assuming the 50M-parameter model:

| Component | Size (fp32) |
|-----------|-------------|
| Model parameters | ~200 MB |
| Frozen copies (.data shared) | ~0 MB |
| Optimizer states (LaProp: 2 moments) | ~400 MB |
| Activations (micro_batch=8, seq_len=128) | ~2-4 GB |
| AMP fp16 copies | ~100 MB |

**Primary GPU (GPU 1)**: params + optimizer + activations + scaler = ~3-5 GB
**Replica GPUs (GPU 2, 3)**: params + frozen copies + activations = ~2.5-4.5 GB
**Sim GPU (GPU 0)**: IsaacLab + inference copies = varies by env

With 24 GB per GPU, there is ample headroom. The main benefit is that the **batch can be 3x larger** (since activations are the dominant cost, and they're split across 3 GPUs).

---

## Interaction with multi-gpu-split plan

This plan layers on top of multi-gpu-split. The two are independent:

| Feature | multi-gpu-split | data-parallel (this plan) |
|---------|----------------|---------------------------|
| Sim/train separation | Yes | Assumes it |
| Training GPU count | 1 | 1-N |
| Weight sync | train → sim | train → replicas + train → sim |
| Inference copies | On sim_device | Unchanged |
| Backward compat | sim_device=null → single GPU | train_devices=null → single training GPU |

Implementation order:
1. Implement multi-gpu-split first (GPU 0 sim, GPU 1 train)
2. Implement data-parallel on top (GPU 1 primary + GPUs 2, 3 replicas)

---

## Files to modify

| File | Changes |
|------|---------|
| `config.yaml` | Add `train_devices: null` |
| `vision-pre-training-config.yaml` | Add `train_devices: null` |
| `r2dreamer/dreamer.py` | `_create_training_replicas`, `_broadcast_params`, `_reduce_gradients`, `_cal_grad_on_replica`, refactor `_cal_grad` → `_cal_grad_with_modules`, update `update()` with parallel dispatch, update `to()` |
| `train_dreamer.py` | Resolve `train_devices`, propagate to model config, batch size validation |

---

## Verification

1. **Backward compat**: `train_devices: null` → `train_devices = [train_device]` → `data_parallel = False` → existing sequential micro-batch loop. Zero overhead, identical behavior.

2. **Single training GPU** (multi-gpu-split only): `train_devices: ["cuda:1"]` → same as above.

3. **3 training GPUs**: `sim_device: "cuda:0"`, `train_devices: ["cuda:1", "cuda:2", "cuda:3"]`:
   - `nvidia-smi` shows model replicas on GPUs 1, 2, 3; sim on GPU 0
   - Micro-batches distributed round-robin across GPUs
   - Gradients accumulated on GPU 1 after all GPUs finish
   - Single optimizer step on GPU 1
   - Params broadcast to GPUs 2, 3 and inference copies on GPU 0
   - Training metrics match single-GPU (same effective batch, same loss scaling)
   - Checkpointing saves primary model only (replicas are transient)

4. **Numerical equivalence**: With the same batch, seed, and loss_scale, the data-parallel path should produce identical gradients to the sequential path (up to fp32 rounding from cross-device transfers). Verify by comparing gradient norms and loss values for the first 100 steps.

5. **torch.compile**: Replicas are compiled independently on their respective GPUs. `load_state_dict` from `_broadcast_params` updates underlying params without recompilation (same pattern as self-play opponent sync).

---

## Example launch configurations

```bash
# Single GPU (default, backward compat)
python train_dreamer.py device="cuda:0"

# 2 GPUs: sim + train split only
python train_dreamer.py sim_device="cuda:0" train_device="cuda:1"

# 4 GPUs: sim on 0, data-parallel training on 1-3
python train_dreamer.py sim_device="cuda:0" \
    train_devices='["cuda:1","cuda:2","cuda:3"]' \
    batch_size=24 model.micro_batch_size=8

# 4 GPUs: larger batch now fits
python train_dreamer.py sim_device="cuda:0" \
    train_devices='["cuda:1","cuda:2","cuda:3"]' \
    batch_size=48 model.micro_batch_size=8
```
