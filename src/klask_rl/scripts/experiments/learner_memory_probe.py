"""Peak GPU memory and time of the learner's update (no simulator), for choosing the micro-batch size of the R2/PFSP
runs before a launch.

Composes the run config with Hydra as train_dreamer.py composes it (default config-sprite-r2-pfsp-eff-v2.yaml, the R2
pilot v2), with the arm's overrides taken from the launcher's own case line (run_scripts/r2_pfsp_v2_run.sh: exact,
vanilla, exact_dynbp, vanilla_dynbp), derives the model fields train_dreamer.py derives (opponent separation,
mirroring, timing, devices, micro-batch), builds the Dreamer (size50M, 128 px), the self-play opponent's encoder/RSSM/
actor copies (option A recompute in the exact arms), fills the config's learned-priority buffer (mirrored rows) with
random pixel data, and runs updates. Allocator setting as the launcher (PYTORCH_CUDA_ALLOC_CONF expandable_segments).

    cd <code copy> && python.sh scripts/experiments/learner_memory_probe.py --device cuda:0 --arm exact_dynbp --micro 10
    ... --arm exact --micro 10                       # the pilot exact (reinforce), for the difference
    ... --arm exact --actor_grad dynamics            # same as exact_dynbp
    ... --config config-sprite-r2-pfsp.yaml --arm exact --no-compile   # the older pilot config

Compilation follows the config (model.compile: true, reduce-overhead in eff-v2) unless --compile/--no-compile.
Warm-up: updates run until the optimizer has stepped once (LaProp allocates its state, two float32 copies of the
weights, at its first step; with float16 loss scaling the first updates can overflow and be skipped - with
model.actor_grad=dynamics about ten of them, the scaler halving 65536 -> ~64), at least --warmup updates; then
--updates measured updates. Reported: peak reserved/allocated over everything (compile and warm-up included) and over
the measured updates, the median update time, the grad scale. The simulator (~7.4 GiB in the R2 runs), the inference
copies and other processes' CUDA contexts come on top.
"""

import argparse
import copy
import os
import pathlib
import shlex
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # as the launcher; before CUDA init

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
DREAMER = SCRIPTS / "dreamer"
R2 = DREAMER / "r2dreamer"
LAUNCHER = SCRIPTS / "experiments" / "run_scripts" / "r2_pfsp_v2_run.sh"
DEFAULT_CONFIG = "config-sprite-r2-pfsp-eff-v2"


def launcher_arm_overrides(arm, launcher=LAUNCHER):
    """The Hydra overrides of ``arm`` in the launcher's ``case`` block (``<arm>) OVR="..." ;;``)."""
    for line in pathlib.Path(launcher).read_text().splitlines():
        body = line.strip()
        if body.startswith(f"{arm})") and "OVR=" in body:
            value = body.split("OVR=", 1)[1].split(";;")[0].strip()
            return shlex.split(value)[0].split() if value else []
    raise SystemExit(f"arm {arm!r} not found in {launcher}")


def launcher_arms(launcher=LAUNCHER):
    arms = []
    for line in pathlib.Path(launcher).read_text().splitlines():
        body = line.strip()
        if ")" in body and "OVR=" in body and not body.startswith("*"):
            arms.append(body.split(")", 1)[0])
    return arms


def compose_run_config(config_name, overrides):
    """The run config as train_dreamer.py composes it (``${script_dir:}`` = scripts/dreamer)."""
    from hydra import compose, initialize_config_dir

    if not OmegaConf.has_resolver("script_dir"):
        OmegaConf.register_new_resolver("script_dir", lambda: str(DREAMER), use_cache=True)
    with initialize_config_dir(config_dir=str(DREAMER), version_base=None):
        return compose(config_name=config_name.removesuffix(".yaml"), overrides=list(overrides))


def derive_model_fields(cfg, device):
    """What train_dreamer.py sets on the composed config before building the agent and the buffer."""
    osep = OmegaConf.to_container(cfg.opponent_separation, resolve=True)
    tm = osep.get("trajectory_mirroring", {})
    mirror = bool(tm.get("enabled", False)) if isinstance(tm, dict) else bool(tm)
    mirror_ws = bool(tm.get("warm_start", False)) if isinstance(tm, dict) else False
    aligned = str(osep.get("action_timing", "legacy")) == "aligned"
    mbs = int(getattr(cfg, "micro_batch_size", cfg.batch_size))
    rows = int(cfg.batch_size) * (2 if mirror else 1)
    if rows % mbs != 0:
        raise SystemExit(f"batch rows ({rows}) must be divisible by micro_batch_size ({mbs})")
    with open_dict(cfg):
        cfg.micro_batch_size = mbs
    with open_dict(cfg.model):
        cfg.model.sim_device = device
        cfg.model.train_device = device
        cfg.model.train_devices = [device]
        cfg.model.device = device
        cfg.model.micro_batch_size = mbs
        prio = OmegaConf.select(cfg, "buffer.prioritized")
        if prio is not None and bool(prio.enable) and str(prio.mode) == "learned":
            cfg.model.prio_burn_in = int(getattr(prio, "burn_in", 8))
    with open_dict(cfg.buffer):
        cfg.buffer.device = device
    OmegaConf.update(cfg, "buffer.mirror", mirror, force_add=True)
    OmegaConf.update(cfg, "buffer.mirror_opp_warm_start", mirror_ws, force_add=True)
    OmegaConf.update(cfg, "buffer.aligned_opponent_action", aligned, force_add=True)
    cfg.model.opponent_separation = bool(osep.get("enabled", False))
    cfg.model.trajectory_mirroring = mirror
    cfg.model.imag_opponent = str(osep.get("imag_opponent", "random"))
    OmegaConf.update(cfg, "model.aligned_opponent_action", aligned, force_add=True)
    return cfg


def probe_config(config_name, arm, device, micro, actor_grad=None, compile_=None, checkpoint=None, extra=()):
    overrides = [f"device={device}", f"sim_device={device}", f"train_devices=[{device}]", f"micro_batch_size={micro}"]
    overrides += launcher_arm_overrides(arm)
    if actor_grad is not None:
        overrides.append(f"model.actor_grad={actor_grad}")
    if checkpoint is not None:
        overrides.append(f"model.actor_grad_checkpoint={'true' if checkpoint else 'false'}")
    if compile_ is not None:
        overrides.append(f"model.compile={'true' if compile_ else 'false'}")
    overrides += list(extra)
    return derive_model_fields(compose_run_config(config_name, overrides), device), overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="run config in scripts/dreamer (default: R2 pilot v2)")
    parser.add_argument("--arm", default="exact", help=f"launcher arm ({', '.join(launcher_arms())})")
    parser.add_argument("--micro", type=int, default=10)
    parser.add_argument("--batch", type=int, default=None, help="replay rows before mirroring (default: the config's)")
    parser.add_argument("--actor_grad", choices=["reinforce", "dynamics", "mix"], default=None,
                        help="model.actor_grad (default: the arm's / config's)")
    parser.add_argument("--actor_grad_checkpoint", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                        help="default: the config's model.compile")
    parser.add_argument("--warmup", type=int, default=2, help="at least this many updates before measuring")
    parser.add_argument("--max_warmup", type=int, default=40, help="give up waiting for a first optimizer step")
    parser.add_argument("--updates", type=int, default=4, help="measured updates after the warm-up")
    parser.add_argument("--extra", nargs="*", default=[], help="more Hydra overrides (as EXTRA_OVR)")
    args = parser.parse_args()
    extra = list(args.extra) + ([f"batch_size={args.batch}"] if args.batch else [])
    cfg, overrides = probe_config(args.config, args.arm, args.device, args.micro, args.actor_grad, args.compile,
                                  args.actor_grad_checkpoint, extra)
    sys.path.insert(0, str(R2))
    from buffer import PrioritizedBuffer
    from dreamer import Dreamer
    from tensordict import TensorDict

    import tools

    tools.set_seed_everywhere(int(cfg.seed))
    size = tuple(int(x) for x in cfg.env.size)
    obs_space = gym.spaces.Dict({"image": gym.spaces.Box(0, 255, (*size, 3), dtype=np.uint8)})
    act_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    agent = Dreamer(cfg.model, obs_space, act_space).to(args.device)
    if agent.opponent_separation:
        # The self-play wrapper's opponent modules (frozen; option A uses the encoder).
        enc, rssm, actor = (copy.deepcopy(m).eval() for m in (agent.encoder, agent.rssm, agent.actor))
        for mod in (enc, rssm, actor):
            for p in mod.parameters():
                p.requires_grad_(False)
        agent.set_imag_opponent_networks(rssm, actor, enc)
    bcfg = cfg.buffer
    prio = bcfg.prioritized
    envs, rows = int(cfg.env.env_num), 400
    config = SimpleNamespace(
        device=args.device, storage_device="cpu", batch_size=int(cfg.batch_size), batch_length=int(cfg.batch_length),
        max_size=envs * rows, mirror=bool(bcfg.mirror), mirror_opp_warm_start=bool(bcfg.mirror_opp_warm_start),
        aligned_opponent_action=bool(bcfg.aligned_opponent_action), slice_sampling=str(bcfg.slice_sampling),
        prioritized=SimpleNamespace(alpha=float(prio.alpha), beta=float(prio.beta),
                                    baseline_priority=float(prio.baseline_priority), mode=str(prio.mode),
                                    uniform_frac=float(prio.uniform_frac), bucket_stride=int(prio.bucket_stride),
                                    eps=float(prio.eps), stats_decay=float(prio.stats_decay),
                                    reward_agg=tuple(prio.reward_agg), value_agg=tuple(prio.value_agg)),
    )
    buf = PrioritizedBuffer(config)
    S, K, D = agent.rssm._stoch, agent.rssm._discrete, agent.rssm._deter
    for t in range(rows):
        end = t % 300 == 299
        rew = (torch.where(torch.rand(envs, 1) < 0.5, 1000.0, -1000.0) if end else torch.zeros(envs, 1))
        buf.add_transition(TensorDict({
            "image": torch.randint(0, 255, (envs, *size, 3), dtype=torch.uint8),
            "opponent_image": torch.randint(0, 255, (envs, *size, 3), dtype=torch.uint8),
            "action": torch.rand(envs, 2) * 2 - 1, "opponent_action": torch.rand(envs, 2) * 2 - 1,
            "policy": torch.zeros(envs, 1), "opponent": torch.zeros(envs, 1),
            "reward": rew, "opponent_reward": -rew,
            "is_first": torch.full((envs, 1), t % 300 == 0), "is_terminal": torch.full((envs, 1), end),
            "is_last": torch.full((envs, 1), end), "stoch": torch.zeros(envs, S, K),
            "deter": torch.zeros(envs, D), "episode": torch.full((envs,), t // 300, dtype=torch.int64),
        }, batch_size=[envs]))
    print(f"config {args.config}, overrides: {' '.join(overrides)}")
    print(f"actor_grad={agent.actor_grad} (checkpoint {agent.actor_grad_checkpoint}), compile={agent._compile} "
          f"({agent._compile_mode}), rows {int(cfg.batch_size) * (2 if bool(bcfg.mirror) else 1)} x "
          f"{int(cfg.batch_length)}, micro_batch_size {agent.micro_batch_size}, "
          f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)
    if args.device.startswith("cpu"):
        print("(CPU: configuration check only; no memory statistics)")
        return

    def stepped():
        return sum(len(s) for s in agent._optimizer.state.values()) > 0

    torch.cuda.reset_peak_memory_stats(args.device)
    warm, scales, mets = 0, [], {}
    t_start = time.perf_counter()
    while warm < args.warmup or not stepped():
        if warm >= args.max_warmup:
            print(f"WARNING: no optimizer step after {warm} updates (grad scale {scales}); the peak below lacks the "
                  "optimizer state (2 float32 copies of the weights)", flush=True)
            break
        mets = agent.update(buf)
        scales.append(float(mets.get("opt/grad_scale", float("nan"))))
        warm += 1
    torch.cuda.synchronize(args.device)
    warm_s = time.perf_counter() - t_start
    peak_all = torch.cuda.max_memory_reserved(args.device) / 2**30
    alloc_all = torch.cuda.max_memory_allocated(args.device) / 2**30
    torch.cuda.reset_peak_memory_stats(args.device)
    times = []
    for _ in range(args.updates):
        torch.cuda.synchronize(args.device)
        t0 = time.perf_counter()
        mets = agent.update(buf)
        torch.cuda.synchronize(args.device)
        times.append(time.perf_counter() - t0)
        scales.append(float(mets.get("opt/grad_scale", float("nan"))))
    peak = torch.cuda.max_memory_reserved(args.device) / 2**30
    alloc = torch.cuda.max_memory_allocated(args.device) / 2**30
    peak_all, alloc_all = max(peak_all, peak), max(alloc_all, alloc)
    print(f"arm={args.arm} actor_grad={agent.actor_grad} micro={agent.micro_batch_size} compile={agent._compile}: "
          f"peak reserved {peak_all:.2f} GiB (measured updates {peak:.2f}), peak allocated {alloc_all:.2f} GiB "
          f"(measured {alloc:.2f}); update {np.median(times) if times else float('nan'):.2f} s median of "
          f"{len(times)} (warm-up {warm} updates, {warm_s:.0f} s); optimizer stepped: {stepped()}; "
          f"opp_recompute {mets.get('timing/opp_recompute_ms', float('nan')):.0f} ms; grad scale {scales}", flush=True)


if __name__ == "__main__":
    main()
