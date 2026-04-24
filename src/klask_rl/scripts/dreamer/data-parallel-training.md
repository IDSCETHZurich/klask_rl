# Plan: Data-Parallel Training Across Multiple GPUs

## Context

With the multi-gpu-split plan implemented, GPU 0 handles simulation+inference and GPU 1 handles training. This leaves GPUs 2 and 3 idle. The goal here is to use all three remaining GPUs (1, 2, 3) for training, enabling larger batch sizes and faster updates without exceeding any single GPU's 24GB VRAM.

Depends on: multi-gpu-split.md (fully implemented).

### Why not standard DDP (torchrun)?

Standard `DistributedDataParallel` requires multi-process launching (one process per GPU). IsaacLab's `AppLauncher` creates a single simulation instance tied to one process — it cannot be split across processes. Running sim on only rank 0 and broadcasting transitions adds complexity (process coordination, data serialization, AppLauncher conflicts with torchrun). A single-process approach avoids all of this.

### Why not `torch.nn.DataParallel`?

`DataParallel` is deprecated, funnels all gradients through one GPU (memory bottleneck), and suffers from the Python GIL. Manual parallelism is cleaner for this use case.

---

## Architecture

```
GPU 0 (sim_device):              GPUs 1-3 (train_devices):
+-----------------------+         +----------------------------------------+
| IsaacLab (16 envs)   |         | GPU 1 (primary)  GPU 2    GPU 3       |
| Inference copies      |         | +------------+ +--------+ +--------+  |
| Self-play opponent    |         | | Model      | |Replica | |Replica |  |
|                       |         | | Optimizer  | |Scaler  | |Scaler  |  |
| act() runs here      |         | | GradScaler | |        | |        |  |
|                       |         | +------------+ +--------+ +--------+  |
+-----------------------+         |       ^             ^          ^      |
         ^                        |       +--- all-reduce grads ---+      |
         | weight sync            |       v                               |
         +----------------------- |  optimizer.step() on primary          |
                                  |       v                               |
                                  |  broadcast params -> replicas         |
                                  +----------------------------------------+
```

Single-GPU fallback: when only one training GPU is configured, replicas are not created and the code reduces to the existing micro-batch loop with zero overhead.

---

## Design decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | backward/GradScaler | One `GradScaler` per device, all fp16 | All GPUs run identical fp16 code paths — numerically equivalent gradients |
| 2 | `_imagine` / frozen modules | Include all frozen + opponent modules in replica dict | Replicas are fully self-contained, no cross-device transfers mid-computation |
| 3 | `torch.compile` | Compile `_cal_grad_with_modules` per-device | All GPUs get CUDA graphs + kernel fusion for maximum throughput |
| 4 | `load_state_dict` on compiled replicas | Store `replica_orig[name]` refs (like `_inference_*_orig`) | Proven pattern in the codebase, `load_state_dict` is the fastest bulk copy |
| 5 | `return_ema` | Compute on primary only, pass `ret_offset`/`ret_scale` into replicas | Single source of truth — independent EMAs would drift and break equivalence |
| 6 | DreamerPro | Exclude with assertion | Sinkhorn assignment is a global operation; splitting it across GPUs without sync gives wrong results. Not currently used |

---

## Core idea: parallel micro-batches

The existing micro-batch gradient accumulation loop processes chunks sequentially on one GPU:

```python
# Current (dreamer.py lines 498-511)
for i in range(num_acc):
    micro_data = p_data[i*mbs : (i+1)*mbs]
    with autocast(...):
        (stoch, deter), mets = self._cal_grad(micro_data, ...)
```

Instead of looping sequentially, distribute micro-batches across GPUs and run them in parallel:

```python
# Proposed
# 3 training GPUs, batch_size=24, micro_batch_size=8 -> 3 micro-batches
# Each GPU processes one micro-batch simultaneously
```

This converts a sequential loop into parallel execution. If you have N training GPUs and N micro-batches, the forward+backward pass runs ~N times faster (minus communication overhead).

---

## Config changes

**Top-level** in `config.yaml` and `vision-pre-training-config.yaml`:
```yaml
device: "cuda:0"
sim_device: null          # from multi-gpu-split
train_devices: null       # list of training GPUs, e.g. ["cuda:1", "cuda:2", "cuda:3"]
                          # if null -> falls back to [device]
```

`train_device` is no longer a user-facing config key.  `train_dreamer.py`
derives it as `train_devices[0]` and propagates both into model/buffer configs.

These propagate into sub-configs in `train_dreamer.py`:
- `model.train_devices` -> list of training GPU devices
- `model.train_device` -> `train_devices[0]` (primary, holds optimizer)
- Batch size auto-adjusted: `effective_batch_size = batch_size * len(train_devices)`

---

## Implementation in `dreamer.py`

### `__init__` additions

```python
# After multi-gpu-split device setup (existing lines 22-25):
self.train_device = torch.device(config.train_device or config.device)
self.sim_device = torch.device(config.sim_device or config.device)
self.device = self.sim_device
self.multi_gpu = (self.sim_device != self.train_device)

# Data-parallel setup (NEW)
train_devices_cfg = getattr(config, "train_devices", None)
if train_devices_cfg and len(train_devices_cfg) > 1:
    self.train_devices = [torch.device(d) for d in train_devices_cfg]
else:
    self.train_devices = [self.train_device]
self.data_parallel = len(self.train_devices) > 1

# DreamerPro guard
if self.data_parallel:
    assert config.rep_loss != "dreamerpro", (
        "DreamerPro is not supported with data-parallel training. "
        "Sinkhorn assignment requires the full batch and cannot be split across GPUs."
    )
```

### `_create_training_replicas()` -- new method

Called after model is created and moved to `train_device`. Creates full self-contained copies on GPUs 2, 3, etc., including all frozen and opponent modules.

```python
def _create_training_replicas(self):
    """Create model replicas on secondary training GPUs for data-parallel training."""
    if not self.data_parallel:
        self._replicas = []
        self._replica_scalers = []
        return

    self._replicas = []
    self._replica_scalers = []
    secondary_devices = self.train_devices[1:]  # skip primary

    for dev in secondary_devices:
        replica = {}

        # --- Trainable modules ---
        trainable_names = ["encoder", "rssm", "actor", "value", "reward", "cont"]
        if hasattr(self, "decoder"):
            trainable_names.append("decoder")
        if hasattr(self, "prj"):
            trainable_names.append("prj")

        for name in trainable_names:
            module = copy.deepcopy(getattr(self, name)).to(dev)
            replica[name] = module
        replica["rssm"]._device = dev

        # --- Frozen copies (share .data with replica's trainable modules) ---
        frozen_names = ["encoder", "rssm", "reward", "cont", "actor", "value"]
        replica["frozen"] = {}
        for name in frozen_names:
            src = replica[name]
            frozen = copy.deepcopy(src)
            for p_orig, p_frozen in zip(src.parameters(), frozen.parameters()):
                p_frozen.data = p_orig.data  # share .data
                p_frozen.requires_grad_(False)
            frozen.eval()
            replica["frozen"][name] = frozen
        replica["frozen"]["rssm"]._device = dev

        # slow_value: independent frozen copy from primary
        slow = copy.deepcopy(self._slow_value).to(dev)
        for p in slow.parameters():
            p.requires_grad_(False)
        slow.eval()
        replica["frozen"]["slow_value"] = slow

        # return_ema: NOT replicated (Decision 5 -- primary only)

        # --- Opponent imagination modules (if selfplay) ---
        if hasattr(self, "_imag_opp_rssm") and self._imag_opp_rssm is not None:
            opp_rssm = copy.deepcopy(self._imag_opp_rssm).to(dev).eval()
            opp_rssm._device = dev
            opp_actor = copy.deepcopy(self._imag_opp_actor).to(dev).eval()
            for p in opp_rssm.parameters():
                p.requires_grad_(False)
            for p in opp_actor.parameters():
                p.requires_grad_(False)
            replica["imag_opp_rssm"] = opp_rssm
            replica["imag_opp_actor"] = opp_actor

        replica["device"] = dev

        # --- Keep uncompiled originals for load_state_dict (Decision 4) ---
        replica["orig"] = {}
        for name in trainable_names:
            replica["orig"][name] = replica[name]
        for name in replica["frozen"]:
            replica["orig"]["frozen_" + name] = replica["frozen"][name]
        if "imag_opp_rssm" in replica:
            replica["orig"]["imag_opp_rssm"] = replica["imag_opp_rssm"]
            replica["orig"]["imag_opp_actor"] = replica["imag_opp_actor"]

        # --- Compile replica modules (Decision 3) ---
        if self._compile:
            for name in trainable_names:
                replica[name] = torch.compile(replica[name], mode="reduce-overhead")
            for name in list(replica["frozen"].keys()):
                replica["frozen"][name] = torch.compile(
                    replica["frozen"][name], mode="reduce-overhead"
                )
            if "imag_opp_rssm" in replica:
                replica["imag_opp_rssm"] = torch.compile(
                    replica["imag_opp_rssm"], mode="reduce-overhead"
                )
                replica["imag_opp_actor"] = torch.compile(
                    replica["imag_opp_actor"], mode="reduce-overhead"
                )

        self._replicas.append(replica)

        # Per-device GradScaler (Decision 1)
        self._replica_scalers.append(GradScaler())
```

### `_broadcast_params()` -- new method

After optimizer step on primary, push updated weights to all replicas. Uses `replica["orig"]` references to avoid compiled-module key mismatches.

```python
def _broadcast_params(self):
    """Copy primary model parameters to all replicas."""
    if not self.data_parallel:
        return

    trainable_names = ["encoder", "rssm", "actor", "value", "reward", "cont"]
    if hasattr(self, "decoder"):
        trainable_names.append("decoder")
    if hasattr(self, "prj"):
        trainable_names.append("prj")

    for replica in self._replicas:
        # Trainable modules: primary -> replica
        for name in trainable_names:
            primary_module = getattr(self, name)
            replica["orig"][name].load_state_dict(primary_module.state_dict())
        # Frozen copies share .data with trainable, so they update automatically.
        # But slow_value is independent -- sync it.
        replica["orig"]["frozen_slow_value"].load_state_dict(
            self._slow_value.state_dict()
        )
```

### `_broadcast_opponent_to_replicas()` -- new method

Called after `sync_imag_opponent_networks()` to push opponent weights to replicas.

```python
def _broadcast_opponent_to_replicas(self):
    """Copy imagination opponent weights to replicas."""
    if not self.data_parallel:
        return
    if not hasattr(self, "_imag_opp_rssm") or self._imag_opp_rssm is None:
        return
    for replica in self._replicas:
        if "imag_opp_rssm" not in replica:
            continue
        replica["orig"]["imag_opp_rssm"].load_state_dict(
            self._imag_opp_rssm.state_dict()
        )
        replica["orig"]["imag_opp_actor"].load_state_dict(
            self._imag_opp_actor.state_dict()
        )
```

### `_reduce_gradients()` -- new method

After all GPUs finish their backward passes, sum gradients from replicas into primary. Unscale replica gradients first so they're in the same space as primary's unscaled gradients.

```python
def _reduce_gradients(self):
    """Unscale replica gradients and sum into primary model parameters."""
    if not self.data_parallel:
        return

    trainable_names = ["encoder", "rssm", "actor", "value", "reward", "cont"]
    if hasattr(self, "decoder"):
        trainable_names.append("decoder")
    if hasattr(self, "prj"):
        trainable_names.append("prj")

    # Unscale replica gradients (each has its own scaler)
    all_replica_params = []
    for replica, scaler in zip(self._replicas, self._replica_scalers):
        params = []
        for name in trainable_names:
            params.extend(replica["orig"][name].parameters())
        scaler.unscale_(
            # We need to pass the optimizer that "owns" these params.
            # Since replicas don't have their own optimizer, we use a
            # lightweight wrapper -- see note below.
            _FakeOptimizer(params)
        )
        all_replica_params.append((replica, params))

    # Sum replica gradients into primary
    for replica, _ in all_replica_params:
        for name in trainable_names:
            primary_module = getattr(self, name)
            replica_module = replica["orig"][name]
            for p_primary, p_replica in zip(
                primary_module.parameters(), replica_module.parameters()
            ):
                if p_primary.grad is not None and p_replica.grad is not None:
                    p_primary.grad.add_(p_replica.grad.to(self.train_device))
                elif p_replica.grad is not None:
                    p_primary.grad = p_replica.grad.to(self.train_device)
        # Zero replica gradients for next step
        for name in trainable_names:
            replica["orig"][name].zero_grad(set_to_none=True)
```

Note: `GradScaler.unscale_` expects an optimizer. A minimal shim:

```python
class _FakeOptimizer:
    """Minimal shim so GradScaler.unscale_ can iterate parameter groups."""
    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]
```

### Refactoring `_cal_grad()` -> `_cal_grad_with_modules()` -- key change

Currently `_cal_grad()` (lines 548-743) references `self.encoder`, `self.rssm`, `self._frozen_*`, `self._scaler`, `self.return_ema`, etc. directly. Refactor to accept modules as parameters:

```python
def _cal_grad(self, data, initial, loss_scale=1.0, opp_initial=None):
    """Backward pass on primary GPU using self's modules."""
    # Compute return_ema on primary (Decision 5)
    # ret_norm_state is passed to _cal_grad_with_modules so it can
    # normalise advantages without calling return_ema itself.
    # On the first call we don't have ret_norm_state yet -- pass None
    # and let _cal_grad_with_modules call self.return_ema directly.
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
            "encoder": self._frozen_encoder,
            "rssm": self._frozen_rssm,
            "reward": self._frozen_reward,
            "cont": self._frozen_cont,
            "actor": self._frozen_actor,
            "value": self._frozen_value,
            "slow_value": self._frozen_slow_value,
        },
        opp_modules={
            "rssm": getattr(self, "_imag_opp_rssm", None),
            "actor": getattr(self, "_imag_opp_actor", None),
        },
        scaler=self._scaler,
        data=data,
        initial=initial,
        loss_scale=loss_scale,
        opp_initial=opp_initial,
        ret_norm_state=None,  # primary computes its own
    )
```

```python
def _cal_grad_with_modules(
    self, modules, frozen, opp_modules, scaler,
    data, initial, loss_scale, opp_initial=None,
    ret_norm_state=None,
):
    """Core training computation, parameterised by module set.

    Parameters
    ----------
    modules : dict
        Trainable modules: encoder, rssm, actor, value, reward, cont,
        and optionally decoder, prj.
    frozen : dict
        Frozen copies: encoder, rssm, reward, cont, actor, value, slow_value.
    opp_modules : dict
        Opponent imagination modules: rssm, actor (may be None).
    scaler : GradScaler
        Per-device gradient scaler.
    ret_norm_state : tuple(float, float) | None
        Pre-computed (ret_offset, ret_scale) from primary's return_ema.
        If None, calls self.return_ema directly (primary GPU path).
    """
    encoder = modules["encoder"]
    rssm = modules["rssm"]
    actor = modules["actor"]
    value = modules["value"]
    reward = modules["reward"]
    cont = modules["cont"]
    frozen_rssm = frozen["rssm"]
    frozen_actor = frozen["actor"]
    frozen_reward = frozen["reward"]
    frozen_cont = frozen["cont"]
    frozen_value = frozen["value"]
    frozen_slow_value = frozen["slow_value"]
    imag_opp_rssm = opp_modules.get("rssm")
    imag_opp_actor = opp_modules.get("actor")

    # ... rest of _cal_grad logic, replacing:
    #   self.encoder      -> encoder
    #   self.rssm         -> rssm
    #   self.actor        -> actor
    #   self.value        -> value
    #   self.reward       -> reward
    #   self.cont         -> cont
    #   self._frozen_*    -> frozen_*
    #   self._scaler      -> scaler
    #   self._imag_opp_*  -> imag_opp_*
    #
    # Config values (self.rep_loss, self._loss_scales, self.kl_free,
    # self.horizon, self.lamb, self.act_entropy, self.barlow_lambd,
    # self.opponent_separation, self._imag_opponent, etc.) still read
    # from self -- they are shared config, not per-device state.

    # === _imagine call (Decision 2: pass frozen + opp from args) ===
    imag_feat, imag_action = self._imagine_with_modules(
        start, self.imag_horizon + 1,
        frozen_rssm=frozen_rssm,
        frozen_actor=frozen_actor,
        imag_opp_rssm=imag_opp_rssm,
        imag_opp_actor=imag_opp_actor,
        opp_start=opp_start,
    )

    # === return_ema (Decision 5: use pre-computed or compute on primary) ===
    if ret_norm_state is not None:
        ret_offset, ret_scale = ret_norm_state
    else:
        ret_offset, ret_scale = self.return_ema(ret)

    # ... compute advantages, losses using ret_offset/ret_scale ...

    # === backward (Decision 1: per-device scaler) ===
    scaler.scale(total_loss * loss_scale).backward()

    return (post_stoch, post_deter), metrics
```

This refactor replaces ~30 `self.encoder`/`self.rssm`/`self._frozen_*`/`self._scaler` references with local variables. The logic itself is unchanged.

### `_imagine_with_modules()` -- new method

Parameterised version of `_imagine()` (lines 746-805). Accepts frozen and opponent modules instead of reading from `self`.

```python
@torch.no_grad()
def _imagine_with_modules(
    self, start, imag_horizon,
    frozen_rssm, frozen_actor,
    imag_opp_rssm=None, imag_opp_actor=None,
    opp_start=None,
):
    """Roll out the policy in latent space using provided modules.

    Same logic as _imagine() but parameterised so replicas can pass
    their own frozen copies and opponent modules.
    """
    stoch, deter = start
    B = stoch.shape[0]
    feats, actions = [], []

    selfplay = self.opponent_separation and self._imag_opponent == "selfplay"
    if selfplay:
        if opp_start is not None:
            opp_stoch, opp_deter = opp_start
        else:
            opp_stoch, opp_deter = imag_opp_rssm.initial(B)
        opp_prev_action = torch.zeros(
            B, imag_opp_rssm._act_dim, device=stoch.device
        )

    for _ in range(imag_horizon):
        feat = frozen_rssm.get_feat(stoch, deter)
        player_action = frozen_actor(feat).rsample()
        feats.append(feat)
        actions.append(player_action)

        if self.opponent_separation:
            if selfplay:
                opp_feat = imag_opp_rssm.get_feat(opp_stoch, opp_deter)
                opp_action = imag_opp_actor(opp_feat).rsample()
                wm_action = torch.cat([player_action, opp_action], dim=-1)
                opp_prev_action = torch.cat([opp_action, player_action], dim=-1)
                opp_stoch, opp_deter = imag_opp_rssm.img_step(
                    opp_stoch, opp_deter, opp_prev_action
                )
            else:
                opp_action = self._get_imag_opponent_action(feat, player_action)
                wm_action = torch.cat([player_action, opp_action], dim=-1)
        else:
            wm_action = player_action
        stoch, deter = frozen_rssm.img_step(stoch, deter, wm_action)

    return torch.stack(feats, dim=1), torch.stack(actions, dim=1)
```

The existing `_imagine()` becomes a thin wrapper:

```python
@torch.no_grad()
def _imagine(self, start, imag_horizon, opp_start=None):
    return self._imagine_with_modules(
        start, imag_horizon,
        frozen_rssm=self._frozen_rssm,
        frozen_actor=self._frozen_actor,
        imag_opp_rssm=getattr(self, "_imag_opp_rssm", None),
        imag_opp_actor=getattr(self, "_imag_opp_actor", None),
        opp_start=opp_start,
    )
```

### `_cal_grad_on_replica()` -- new method

Runs `_cal_grad_with_modules` using a replica's modules and device.

```python
def _cal_grad_on_replica(self, replica, replica_scaler, data, initial,
                         loss_scale, ret_norm_state, opp_initial=None):
    """Run _cal_grad logic on a replica's modules and device."""
    dev = replica["device"]
    # Move data to replica device
    data_dev = data.to(dev)
    initial_dev = (initial[0].to(dev), initial[1].to(dev))
    opp_initial_dev = None
    if opp_initial is not None:
        opp_initial_dev = (opp_initial[0].to(dev), opp_initial[1].to(dev))

    return self._cal_grad_with_modules(
        modules={
            "encoder": replica["encoder"],
            "rssm": replica["rssm"],
            "actor": replica["actor"],
            "value": replica["value"],
            "reward": replica["reward"],
            "cont": replica["cont"],
            "decoder": replica.get("decoder"),
            "prj": replica.get("prj"),
        },
        frozen=replica["frozen"],
        opp_modules={
            "rssm": replica.get("imag_opp_rssm"),
            "actor": replica.get("imag_opp_actor"),
        },
        scaler=replica_scaler,
        data=data_dev,
        initial=initial_dev,
        loss_scale=loss_scale,
        opp_initial=opp_initial_dev,
        ret_norm_state=ret_norm_state,
    )
```

### `torch.compile` of `_cal_grad_with_modules` (Decision 3)

Currently (line 474-476) `_cal_grad` is compiled as a whole function:

```python
if self._compile and not self._compiled:
    self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")
    self._compiled = True
```

With the refactor, compile `_cal_grad_with_modules` instead. For replicas, create per-device compiled versions:

```python
# In __init__ or first update():
if self._compile and not self._compiled:
    self._cal_grad_with_modules = torch.compile(
        self._cal_grad_with_modules, mode="reduce-overhead"
    )
    # Replicas use the same compiled function -- torch.compile dispatches
    # to different CUDA graphs based on which device the tensors are on.
    # No separate compilation needed per replica.
    self._compiled = True
```

Note: `torch.compile` with `mode="reduce-overhead"` records separate CUDA graphs per device automatically when it detects tensors on different GPUs. The compiled function itself is shared.

### Modified `update()` method

```python
def update(self, replay_buffer):
    if self._compile and not self._compiled:
        self._cal_grad_with_modules = torch.compile(
            self._cal_grad_with_modules, mode="reduce-overhead"
        )
        self._compiled = True

    sample = replay_buffer.sample()
    if sample is None:
        return {}
    data, index, initial, opp_initial = sample
    torch.compiler.cudagraph_mark_step_begin()
    p_data = self.preprocess(data)
    self._update_slow_target()
    if self.rep_loss == "dreamerpro":
        self.ema_update()

    B = p_data.shape[0]
    mbs = self.micro_batch_size
    num_acc = B // mbs
    loss_scale = 1.0 / num_acc

    all_stoch, all_deter = [], []
    mets = {}

    if not self.data_parallel:
        # --- Single training GPU: existing sequential micro-batch loop ---
        for i in range(num_acc):
            s, e = i * mbs, (i + 1) * mbs
            micro_data = p_data[s:e]
            micro_initial = (initial[0][s:e], initial[1][s:e])
            micro_opp = None
            if opp_initial is not None:
                micro_opp = (opp_initial[0][s:e], opp_initial[1][s:e])
            with autocast(device_type=self.train_device.type, dtype=torch.float16):
                (stoch, deter), mets = self._cal_grad(
                    micro_data, micro_initial, loss_scale, opp_initial=micro_opp
                )
            all_stoch.append(stoch)
            all_deter.append(deter)
    else:
        # --- Multi-GPU: distribute micro-batches across GPUs ---
        num_gpus = len(self.train_devices)
        # Round-robin assignment ensures balance
        gpu_assignments = [[] for _ in range(num_gpus)]
        for i in range(num_acc):
            gpu_assignments[i % num_gpus].append(i)

        # == Phase 1: run primary's first micro-batch to get return_ema state ==
        # Primary processes its first micro-batch to compute return_ema,
        # then shares the (ret_offset, ret_scale) with all subsequent batches.
        first_idx = gpu_assignments[0][0]
        s, e = first_idx * mbs, (first_idx + 1) * mbs
        micro_data = p_data[s:e]
        micro_initial = (initial[0][s:e], initial[1][s:e])
        micro_opp = None
        if opp_initial is not None:
            micro_opp = (opp_initial[0][s:e], opp_initial[1][s:e])
        with autocast(device_type=self.train_device.type, dtype=torch.float16):
            (stoch, deter), mets = self._cal_grad(
                micro_data, micro_initial, loss_scale, opp_initial=micro_opp
            )
        primary_results = [(first_idx, stoch, deter)]
        # Grab return_ema state for replicas
        ret_norm_state = (self.return_ema.ema_vals[0], self.return_ema.ema_vals[1])

        # == Phase 2: remaining primary micro-batches + all replica micro-batches ==
        # These all use the frozen ret_norm_state from phase 1.

        # Primary's remaining micro-batches
        for i in gpu_assignments[0][1:]:
            s, e = i * mbs, (i + 1) * mbs
            micro_data = p_data[s:e]
            micro_initial = (initial[0][s:e], initial[1][s:e])
            micro_opp = None
            if opp_initial is not None:
                micro_opp = (opp_initial[0][s:e], opp_initial[1][s:e])
            with autocast(device_type=self.train_device.type, dtype=torch.float16):
                (stoch, deter), mets = self._cal_grad_with_modules(
                    modules={
                        "encoder": self.encoder, "rssm": self.rssm,
                        "actor": self.actor, "value": self.value,
                        "reward": self.reward, "cont": self.cont,
                        "decoder": getattr(self, "decoder", None),
                        "prj": getattr(self, "prj", None),
                    },
                    frozen={
                        "encoder": self._frozen_encoder,
                        "rssm": self._frozen_rssm,
                        "reward": self._frozen_reward,
                        "cont": self._frozen_cont,
                        "actor": self._frozen_actor,
                        "value": self._frozen_value,
                        "slow_value": self._frozen_slow_value,
                    },
                    opp_modules={
                        "rssm": getattr(self, "_imag_opp_rssm", None),
                        "actor": getattr(self, "_imag_opp_actor", None),
                    },
                    scaler=self._scaler,
                    data=micro_data, initial=micro_initial,
                    loss_scale=loss_scale, opp_initial=micro_opp,
                    ret_norm_state=ret_norm_state,
                )
            primary_results.append((i, stoch, deter))

        # Replica GPUs' micro-batches (concurrent via separate CUDA devices)
        replica_results = []
        for gpu_idx, (replica, rep_scaler) in enumerate(
            zip(self._replicas, self._replica_scalers), start=1
        ):
            dev = replica["device"]
            for i in gpu_assignments[gpu_idx]:
                s, e = i * mbs, (i + 1) * mbs
                micro_data = p_data[s:e]
                micro_initial = (initial[0][s:e], initial[1][s:e])
                micro_opp = None
                if opp_initial is not None:
                    micro_opp = (opp_initial[0][s:e], opp_initial[1][s:e])
                with autocast(device_type=dev.type, dtype=torch.float16):
                    (stoch, deter), mets = self._cal_grad_on_replica(
                        replica, rep_scaler,
                        micro_data, micro_initial, loss_scale,
                        ret_norm_state=ret_norm_state,
                        opp_initial=micro_opp,
                    )
                # Move posteriors back for buffer update
                replica_results.append((
                    i,
                    stoch.to(self.train_device),
                    deter.to(self.train_device),
                ))

        # Synchronize all CUDA devices
        for dev in self.train_devices:
            torch.cuda.synchronize(dev)

        # Reduce gradients from replicas into primary
        self._reduce_gradients()

        # Collect results in original micro-batch order
        all_results = sorted(
            primary_results + replica_results, key=lambda x: x[0]
        )
        for _, stoch, deter in all_results:
            all_stoch.append(stoch)
            all_deter.append(deter)

    # --- Optimizer step (unchanged, runs on primary training GPU) ---
    self._scaler.unscale_(self._optimizer)
    if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
        self._prototypes.grad.zero_()
    if self._log_grads:
        old_params = [p.data.clone().detach() for p in self._named_params.values()]
        grads = [p.grad for p in self._named_params.values() if p.grad is not None]
        grad_norm = tools.compute_global_norm(grads)
        grad_rms = tools.compute_rms(grads)
        mets["opt/grad_norm"] = grad_norm
        mets["opt/grad_rms"] = grad_rms
    self._agc(self._named_params.values())
    self._scaler.step(self._optimizer)
    self._scaler.update()
    self._scheduler.step()
    self._optimizer.zero_grad(set_to_none=True)
    mets["opt/lr"] = self._scheduler.get_lr()[0]
    mets["opt/grad_scale"] = self._scaler.get_scale()
    if self._log_grads:
        updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
        update_rms = tools.compute_rms(updates)
        params_rms = tools.compute_rms(self._named_params.values())
        mets["opt/param_rms"] = params_rms
        mets["opt/update_rms"] = update_rms

    # Update replica scalers (Decision 1)
    for scaler in self._replica_scalers:
        scaler.update()

    # --- Sync weights ---
    if self.data_parallel:
        self._broadcast_params()           # primary -> replicas (GPUs 2, 3)
    self._sync_inference_copies()          # train -> sim GPU (from multi-gpu-split)

    # Update latent vectors in replay buffer
    all_stoch = torch.cat(all_stoch, dim=0)
    all_deter = torch.cat(all_deter, dim=0)
    orig_B = getattr(replay_buffer, "_original_batch_size", all_stoch.shape[0])
    replay_buffer.update(index, all_stoch[:orig_B].detach(), all_deter[:orig_B].detach())
    return mets
```

### `to()` override -- updated

```python
def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    self.clone_and_freeze()
    self._create_inference_copies()    # from multi-gpu-split
    self._create_training_replicas()   # new
    return self
```

### `sync_imag_opponent_networks()` -- updated

Extend the existing method to also sync replicas:

```python
def sync_imag_opponent_networks(self):
    if not self.multi_gpu:
        return
    if self._imag_opp_rssm is None:
        return
    self._imag_opp_rssm.load_state_dict(self._imag_opp_rssm_src.state_dict())
    self._imag_opp_actor.load_state_dict(self._imag_opp_actor_src.state_dict())
    # Propagate to data-parallel replicas
    self._broadcast_opponent_to_replicas()
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

### Batch size validation

```python
num_train_gpus = len(train_devices)
if num_train_gpus > 1:
    assert (config.batch_size % config.model.micro_batch_size == 0), \
        "batch_size must be divisible by micro_batch_size"
    num_micro = config.batch_size // config.model.micro_batch_size
    if num_micro % num_train_gpus != 0:
        num_micro = ((num_micro + num_train_gpus - 1) // num_train_gpus) * num_train_gpus
        config.batch_size = num_micro * config.model.micro_batch_size
        logger.info(f"Adjusted batch_size to {config.batch_size} for even GPU distribution")
```

---

## GradScaler handling (Decision 1)

All GPUs run identical fp16 forward+backward paths using `autocast` + per-device `GradScaler`:

- **Primary GPU**: `self._scaler` (existing)
- **Replica GPUs**: `self._replica_scalers[i]` (one per replica)

Flow per update step:

1. All GPUs run `scaler.scale(loss).backward()` inside `autocast`
2. Primary: `self._scaler.unscale_(self._optimizer)` (existing)
3. Replicas: `replica_scaler.unscale_(_FakeOptimizer(params))` inside `_reduce_gradients`
4. Replica gradients (now unscaled fp32) are summed into primary gradients
5. Primary: `self._scaler.step(self._optimizer)` + `self._scaler.update()`
6. Replicas: `replica_scaler.update()` (adjusts scale based on inf/nan detection)

If a replica scaler detects inf/nan, its gradients will contain inf after unscaling. The primary scaler will then also detect inf in the summed gradients and skip the optimizer step. This is correct behaviour -- the entire update is skipped.

---

## CUDA stream parallelism

In a single process, CUDA kernels on **different devices** execute concurrently by default (each device has its own default stream). No explicit stream management is needed.

Multiple operations on the **same device** are sequential on the default stream. If a GPU is assigned multiple micro-batches, those run sequentially on that GPU. This matches current behaviour but spreads work across more GPUs.

Round-robin assignment ensures balance:
```
9 micro-batches across 3 GPUs -> 3, 3, 3
7 micro-batches across 3 GPUs -> 3, 2, 2
```

---

## Memory budget per GPU

Assuming the 50M-parameter model:

| Component | Size (fp32) |
|-----------|-------------|
| Model parameters | ~200 MB |
| Frozen copies (.data shared) | ~0 MB extra |
| Opponent imagination copies | ~60 MB |
| Optimizer states (LaProp: 2 moments) | ~400 MB |
| Activations (micro_batch=8, seq_len=128) | ~2-4 GB |
| AMP fp16 copies | ~100 MB |
| GradScaler state | negligible |

**Primary GPU (GPU 1)**: params + optimizer + activations + scaler = ~3-5 GB
**Replica GPUs (GPU 2, 3)**: params + frozen + opp + activations + scaler = ~2.5-4.5 GB
**Sim GPU (GPU 0)**: IsaacLab + inference copies = varies by env

With 24 GB per GPU, there is ample headroom. The main benefit is that the **batch can be 3x larger** (activations are the dominant cost, split across 3 GPUs).

---

## Interaction with multi-gpu-split

This plan layers on top of multi-gpu-split. The two are independent:

| Feature | multi-gpu-split | data-parallel (this plan) |
|---------|----------------|---------------------------|
| Sim/train separation | Yes | Assumes it |
| Training GPU count | 1 | 1-N |
| Weight sync | train -> sim | train -> replicas + train -> sim |
| Inference copies | On sim_device | Unchanged |
| Opponent sync | sim -> train (imag copies) | + train -> replicas |
| Backward compat | sim_device=null -> single GPU | train_devices=null -> single training GPU |

---

## Files to modify

| File | Changes |
|------|---------|
| `config.yaml` | Add `train_devices: null` |
| `vision-pre-training-config.yaml` | Add `train_devices: null` |
| `r2dreamer/dreamer.py` | `_FakeOptimizer`, `_create_training_replicas`, `_broadcast_params`, `_broadcast_opponent_to_replicas`, `_reduce_gradients`, `_cal_grad_on_replica`, refactor `_cal_grad` -> `_cal_grad_with_modules`, `_imagine_with_modules`, update `update()` with parallel dispatch, update `to()`, update `sync_imag_opponent_networks()`, DreamerPro assertion |
| `train_dreamer.py` | Resolve `train_devices`, propagate to model config, batch size validation |

---

## Verification

1. **Backward compat**: `train_devices: null` -> `train_devices = [train_device]` -> `data_parallel = False` -> existing sequential micro-batch loop. Zero overhead, identical behaviour.

2. **Single training GPU** (multi-gpu-split only): `train_devices: ["cuda:1"]` -> same as above.

3. **3 training GPUs**: `sim_device: "cuda:0"`, `train_devices: ["cuda:1", "cuda:2", "cuda:3"]`:
   - `nvidia-smi` shows model replicas on GPUs 1, 2, 3; sim on GPU 0
   - Micro-batches distributed round-robin across GPUs
   - All GPUs run fp16 forward+backward with per-device GradScaler
   - Replica gradients unscaled then summed into primary
   - Single optimizer step on GPU 1
   - Params broadcast to GPUs 2, 3 and inference copies on GPU 0
   - Training metrics match single-GPU (same effective batch, same loss scaling)
   - Checkpointing saves primary model only (replicas are transient)

4. **Numerical equivalence**: With the same batch, seed, and loss_scale, the data-parallel path should produce identical gradients to the sequential path (up to fp16 rounding). Verify by comparing gradient norms and loss values for the first 100 steps. The return_ema is computed on primary's first micro-batch and shared, so advantage normalisation is consistent.

5. **torch.compile**: `_cal_grad_with_modules` compiled once, records separate CUDA graphs per device automatically. `load_state_dict` from `_broadcast_params` updates underlying params via `replica["orig"]` references without recompilation (same pattern as `_inference_*_orig`).

6. **DreamerPro guard**: Assertion in `__init__` prevents data-parallel with `rep_loss="dreamerpro"`.

7. **Opponent sync**: `sync_imag_opponent_networks()` now also calls `_broadcast_opponent_to_replicas()`, keeping replica opponent modules in sync.

---

## Example launch configurations

```bash
# Single GPU (default, backward compat)
python train_dreamer.py device="cuda:0"

# 2 GPUs: sim + train split only
python train_dreamer.py sim_device="cuda:0" \
    train_devices='["cuda:1"]'

# 4 GPUs: sim on 0, data-parallel training on 1-3
python train_dreamer.py sim_device="cuda:0" \
    train_devices='["cuda:1","cuda:2","cuda:3"]' \
    batch_size=24 model.micro_batch_size=8

# 4 GPUs: larger batch now fits
python train_dreamer.py sim_device="cuda:0" \
    train_devices='["cuda:1","cuda:2","cuda:3"]' \
    batch_size=48 model.micro_batch_size=8
```
