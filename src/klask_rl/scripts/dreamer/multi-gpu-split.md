# Plan: Multi-GPU Sim/Train Device Split

## Context

With IsaacLab simulation, policy inference, and model training all on one GPU, memory is the bottleneck — especially at 128×128 resolution. This plan separates simulation+inference onto one GPU and training onto another, giving each its own memory pool. The self-play opponent shares the sim GPU.

Depends on: micro-batch gradient accumulation — already implemented in `update()` (lines 375-398).

---

## Config changes

**Top-level** in `config.yaml` and `vision-pre-training-config.yaml`:
```yaml
device: "cuda:0"          # backward-compat default (used if sim/train not set)
sim_device: null           # if null → falls back to device
train_device: null         # if null → falls back to device
```

These propagate into sub-configs in `train_dreamer.py`:
- `model.device` → `train_device` (training modules)
- `model.sim_device` → `sim_device` (inference copies)
- `model.train_device` → `train_device`
- `buffer.device` → `train_device` (samples go to training GPU)
- `env.device` stays as-is (IsaacLab picks GPU via AppLauncher `--device`)
- GpuEnvStepper `agent_device` → `sim_device`

---

## Architecture

```
GPU 0 (sim_device):                    GPU 1 (train_device):
┌─────────────────────────┐            ┌─────────────────────────┐
│ IsaacLab (16 envs)      │            │ Trainable modules:      │
│ Inference copies:       │            │   encoder, rssm, decoder│
│   encoder, rssm, actor  │            │   actor, value, reward  │
│ Self-play opponent:     │            │   cont, slow_value      │
│   encoder, rssm, actor  │            │ Frozen copies (shared   │
│                         │            │   .data with trainable) │
│ act() runs here         │            │ Optimizer + GradScaler  │
└─────────────────────────┘            │ update() runs here      │
         ↑                             └─────────────────────────┘
         │ periodic weight sync                    ↑
         │ (train → inference copies)              │
         └──────────────────── batch from CPU buffer
```

Single-GPU mode (default): `_inference_*` aliases `_frozen_*` — zero overhead, identical to current code.

---

## Implementation in `dreamer.py`

### `__init__`

```python
self.train_device = torch.device(config.train_device or config.device)
self.sim_device = torch.device(config.sim_device or config.device)
self.device = self.sim_device  # trainer uses agent.device for env-side tensors
self.multi_gpu = (self.sim_device != self.train_device)
```

### `_create_inference_copies()` — new method

Called after `clone_and_freeze()` (which sets up `_frozen_*` on `train_device`).

```python
def _create_inference_copies(self):
    if not self.multi_gpu:
        # Single GPU: reuse frozen copies (shared .data, zero overhead)
        self._inference_encoder = self._frozen_encoder
        self._inference_rssm = self._frozen_rssm
        self._inference_actor = self._frozen_actor
        return
    # Multi-GPU: independent copies on sim_device
    self._inference_encoder = copy.deepcopy(self.encoder).to(self.sim_device).eval()
    self._inference_rssm = copy.deepcopy(self.rssm).to(self.sim_device).eval()
    self._inference_actor = copy.deepcopy(self.actor).to(self.sim_device).eval()
    # Fix RSSM._device so initial() creates tensors on sim_device
    self._inference_rssm._device = self.sim_device
    for m in (self._inference_encoder, self._inference_rssm, self._inference_actor):
        for p in m.parameters():
            p.requires_grad_(False)
    # Optionally compile for fast inference on sim_device
    if self._compile:
        self._inference_encoder = torch.compile(self._inference_encoder, mode="reduce-overhead")
        self._inference_rssm = torch.compile(self._inference_rssm, mode="reduce-overhead")
        self._inference_actor = torch.compile(self._inference_actor, mode="reduce-overhead")
```

### `_sync_inference_copies()` — new method

Called at the end of `update()`, after optimizer step:

```python
def _sync_inference_copies(self):
    if not self.multi_gpu:
        return  # frozen copies share .data, always up to date
    # load_state_dict handles cross-device (train→sim) automatically.
    # For compiled modules, this updates the underlying parameters that
    # the compiled graph shares — no recompilation needed.
    self._inference_encoder.load_state_dict(self.encoder.state_dict())
    self._inference_rssm.load_state_dict(self.rssm.state_dict())
    self._inference_actor.load_state_dict(self.actor.state_dict())
```

### `act()` (lines 254-282)

Replace `_frozen_*` with `_inference_*`:
```python
embed = self._inference_encoder(p_obs)
stoch, deter, _ = self._inference_rssm.obs_step(prev_stoch, prev_deter, prev_action, embed, obs["is_first"])
feat = self._inference_rssm.get_feat(stoch, deter)
action_dist = self._inference_actor(feat)
```

### `get_initial_state()` (lines 285-292)

Currently uses `self.rssm.initial(B)` (the trainable copy on `train_device`). In multi-GPU mode this would produce tensors on the wrong GPU. Switch to inference RSSM:
```python
stoch, deter = self._inference_rssm.initial(B)
action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.sim_device)
```

Note: when `multi_gpu=False`, `_inference_rssm` is `_frozen_rssm` which shares `.data` with `self.rssm`. The `_frozen_rssm._device` is set during `clone_and_freeze()` via deepcopy from `self.rssm` — it inherits the same `_device`. So `initial()` already produces tensors on the correct device. Explicitly use `self.sim_device` for the action tensor (replaces `self.device` which is already `sim_device`).

### `update()` method

- `autocast` must use `self.train_device.type`:
  ```python
  with autocast(device_type=self.train_device.type, dtype=torch.float16):
  ```
- Call `self._sync_inference_copies()` at the end

### `to()` override (lines 247-251)

After `super().to()` and `clone_and_freeze()`, also recreate inference copies:
```python
def to(self, *args, **kwargs):
    super().to(*args, **kwargs)
    self.clone_and_freeze()
    self._create_inference_copies()
    return self
```

### `clone_and_freeze()` — unchanged

Frozen copies still share `.data` with trainable modules on `train_device`. Used by `_cal_grad()` for imagination rollout and target computation. No changes needed.

---

## Implementation in `train_dreamer.py`

### Device resolution (~line 328)

```python
sim_device = getattr(config, "sim_device", None) or config.device
train_device = getattr(config, "train_device", None) or config.device
```

### Propagate into sub-configs

```python
# Model config needs both devices
with open_dict(config.model):
    config.model.sim_device = sim_device
    config.model.train_device = train_device
    config.model.device = train_device  # modules created on train_device

# Buffer samples go to training GPU
with open_dict(config.buffer):
    config.buffer.device = train_device
```

(Use `OmegaConf.update` or `open_dict` context manager since OmegaConf configs are read-only.)

### Agent creation (~line 400-404)

```python
agent = Dreamer(config.model, obs_space, act_space)
# .to(train_device) moves all nn.Module params, then clone_and_freeze + _create_inference_copies
agent.to(train_device)
```

### GpuEnvStepper

Pass `sim_device` through env_config so `make_envs` can use it:
```python
env_config.sim_device = sim_device
```

### Self-play (~line 454-455)

```python
_self_play_wrapper.set_opponent(agent, device=sim_device)
```

### Checkpoint loading (~line 414)

```python
checkpoint = torch.load(checkpoint_path, map_location=train_device)
```

---

## Implementation in `dreamer_self_play.py`

### `set_opponent()` (line 110)

Accept optional `device` parameter:
```python
def set_opponent(self, agent, device=None):
    self._device = device or agent.device
    self._copy_weights(agent)
    num_envs = self.env.unwrapped.num_envs
    self._reset_opponent_state(num_envs)
```

### `_copy_weights()` (lines 224-262)

Currently deepcopies without `.to(self._device)` — works in single-GPU but breaks in multi-GPU since `agent.encoder` would be on `train_device`. First call — deepcopy, move to `self._device`, THEN compile:
```python
if self._opponent_encoder is None:
    enc = copy.deepcopy(agent.encoder).to(self._device)
    rssm = copy.deepcopy(agent.rssm).to(self._device)
    actor = copy.deepcopy(agent.actor).to(self._device)
    # Fix RSSM._device for initial() on sim_device
    rssm._device = self._device
    for module in (enc, rssm, actor):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    self._opponent_encoder_orig = enc
    self._opponent_rssm_orig = rssm
    self._opponent_actor_orig = actor
    if self._compile:
        # Compile AFTER .to(device) so CUDA graphs target sim GPU
        self._opponent_encoder = torch.compile(enc, mode="reduce-overhead")
        self._opponent_rssm = torch.compile(rssm, mode="reduce-overhead")
        self._opponent_actor = torch.compile(actor, mode="reduce-overhead")
    else:
        self._opponent_encoder = enc
        self._opponent_rssm = rssm
        self._opponent_actor = actor
```

Subsequent calls — `load_state_dict` handles cross-device automatically:
```python
else:
    # state_dict may be on train_device; load_state_dict copies to target device
    self._opponent_encoder_orig.load_state_dict(agent.encoder.state_dict())
    self._opponent_rssm_orig.load_state_dict(agent.rssm.state_dict())
    self._opponent_actor_orig.load_state_dict(agent.actor.state_dict())
```

---

## Implementation in `envs/__init__.py`

### `make_envs()` (lines 6-25)

Use `sim_device` for the stepper's `agent_device`:
```python
if suite == "isaaclab":
    vec_env = config.isaac_vec_env
    device = getattr(config, "sim_device", None) or getattr(config, "device", "cuda:0")
    stepper = GpuEnvStepper(vec_env, device)
    return stepper, stepper, vec_env.observation_space, vec_env.action_space
```

---

## Files to modify

| File | Changes |
|------|---------|
| `config.yaml` | Add `sim_device: null`, `train_device: null` |
| `vision-pre-training-config.yaml` | Add `sim_device: null`, `train_device: null` |
| `r2dreamer/dreamer.py` | Dual device in `__init__`, `_create_inference_copies`, `_sync_inference_copies`, update `act()`, `get_initial_state()`, `update()`, `to()` |
| `dreamer_self_play.py` | `set_opponent` device param, `.to(device)` before compile in `_copy_weights` |
| `r2dreamer/envs/__init__.py` | `make_envs` reads sim_device for GpuEnvStepper |
| `train_dreamer.py` | Resolve devices, propagate to sub-configs, pass sim_device to self-play and env |

---

## Verification

1. **Backward compat**: Default `sim_device: null, train_device: null` → both resolve to `device`. `multi_gpu=False` → `_inference_*` aliases `_frozen_*`. Zero overhead, identical behavior.

2. **Multi-GPU**: Set `sim_device: "cuda:0"`, `train_device: "cuda:1"`:
   - `nvidia-smi` shows IsaacLab memory on GPU 0, training activations on GPU 1
   - `act()` runs on cuda:0 with compiled inference copies
   - `_cal_grad()` runs on cuda:1 with compiled training graph
   - Weight sync after each `update()` (load_state_dict cross-device)
   - Self-play opponent on cuda:0
   - Checkpointing and resume works

3. **torch.compile**: Inference copies compiled on `sim_device` CUDA graphs. Training `_cal_grad` compiled on `train_device` CUDA graphs. `load_state_dict` on compiled modules updates underlying params without recompilation (validated by existing self-play pattern).

---

## Caveats & things to watch

1. **`self.device` ambiguity**: The plan sets `self.device = self.sim_device`. Any training-side code that uses `self.device` instead of `self.train_device` will silently create tensors on the wrong GPU. Before implementing, audit all `self.device` usages in `dreamer.py` (especially inside `_cal_grad()` and `update()`) and replace with `self.train_device` where appropriate.

2. **Weight sync cost**: `_sync_inference_copies()` does 3× `load_state_dict` across GPUs after every `update()`. For large models this cross-device copy could add non-trivial latency. Consider profiling the cost and potentially syncing every N updates — the frozen copies already introduce staleness, so slightly more lag may be acceptable.

3. **Buffer cross-device inserts**: The buffer lives on `train_device`, but data arrives from sim on `sim_device`. Verify that the buffer's `add()`/`store()` method handles cross-device tensors correctly (most PyTorch ops do this implicitly, but worth checking).

4. **RSSM `_device` private attribute**: The plan patches `_inference_rssm._device` and `_opponent_rssm._device` directly. This relies on RSSM's internal implementation. If `initial()` ever changes how it determines its device, this will silently break. Consider adding a `device` parameter to `RSSM.initial()` instead.

5. **No async overlap**: The plan is fully synchronous — `act()` blocks on `sim_device`, `update()` blocks on `train_device`, sync blocks on cross-device copy. Overlapping sim steps with training via CUDA streams is a future optimization, but the synchronous version is the right starting point.
