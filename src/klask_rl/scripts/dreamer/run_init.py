"""How a training run starts (branch r2-ft-v2hw, 2026-10-14). No simulator imports: CPU-testable.

:func:`start_plan` decides from the logdir and the config:

  resume    ``<logdir>/latest.pt`` exists: the run's OWN checkpoint, its saved replay buffer (``replay_buffer.latest``),
            PFSP pool and self-play score window. Always wins, so a crash-restart of a fine-tune continues from its own
            latest.pt and never goes back to the checkpoint it was started from.
  finetune  no latest.pt, ``finetune_from.checkpoint`` set: the FULL trainer state of that checkpoint (weights,
            optimizer, LR scheduler, GradScaler, slow-critic / EMA counters, the step and the curriculum /
            initialisation counters: :func:`load_trainer_state`, the same restore as a resume), but a fresh, empty
            replay buffer, a fresh PFSP pool (member 0 = these weights, :func:`start_pfsp_pool`) and an empty self-play
            score window. The opponent starts as a copy of these weights. The checkpoint must lie inside the logdir
            (a copy: the source run's files are never opened by the trainer) and carry ``finetune_from.expect_step``
            when that is set.
  init      no latest.pt, ``init_checkpoint`` set: weights only, step 0 (unchanged behaviour).
  scratch   none of these.

``python run_init.py --print-step CKPT`` prints a checkpoint's step (memory-mapped read; used by the launcher).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

LATEST = -1  # = pfsp_pool.LATEST (the learner's current weights)


@dataclass(frozen=True)
class StartPlan:
    mode: str  # resume | finetune | init | scratch
    checkpoint: Path | None  # the file the trainer state comes from (resume / finetune) or the weights (init)
    restore_buffer: bool  # load <logdir>/<replay_buffer.latest>
    restore_pool: bool  # load the checkpoint's pfsp_state (pool members under <logdir>/pfsp)
    restore_scores: bool  # load the checkpoint's self-play score window


def start_plan(logdir, finetune_checkpoint=None, init_checkpoint=None) -> StartPlan:
    logdir = Path(logdir).expanduser()
    if finetune_checkpoint is not None and init_checkpoint is not None:
        raise ValueError("finetune_from.checkpoint and init_checkpoint are alternatives; set only one")
    latest = logdir / "latest.pt"
    if latest.exists():
        return StartPlan("resume", latest, restore_buffer=True, restore_pool=True, restore_scores=True)
    if finetune_checkpoint is not None:
        path = Path(finetune_checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"finetune_from.checkpoint not found: {path}")
        if not path.resolve().is_relative_to(logdir.resolve()):
            raise ValueError(
                f"finetune_from.checkpoint must be a copy inside the logdir {logdir} (the source run's files are not "
                f"opened by the trainer), got {path}"
            )
        return StartPlan("finetune", path, restore_buffer=False, restore_pool=False, restore_scores=False)
    if init_checkpoint is not None:
        return StartPlan("init", Path(init_checkpoint).expanduser(), False, False, False)
    return StartPlan("scratch", None, False, False, False)


def check_finetune_checkpoint(checkpoint: dict, expect_step=None) -> int:
    """The fine-tune source must be a full trainer checkpoint (with ``expect_step``: of that step)."""
    missing = [k for k in ("agent_state_dict", "optims_state_dict", "step") if k not in checkpoint]
    if missing:
        raise ValueError(f"finetune_from.checkpoint is not a full trainer checkpoint (missing {missing})")
    step = int(checkpoint["step"])
    if expect_step is not None and step != int(expect_step):
        raise ValueError(f"finetune_from.checkpoint is at step {step}, expected {int(expect_step)} (finetune_from.expect_step)")
    return step


def load_trainer_state(agent, checkpoint: dict, load_optim, log=print) -> dict:
    """Restore weights, optimizer states, LR scheduler, GradScaler and the slow-critic / EMA counters from a full
    trainer checkpoint (the resume restore of train_dreamer.py, shared with the fine-tune start). ``load_optim`` is
    ``tools.recursively_load_optim_state_dict``. Returns the step and the curriculum / initialisation counters."""
    missing, unexpected = agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
    if missing or unexpected:
        log(f"  Checkpoint key mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            log(f"    Missing: {missing}")
        if unexpected:
            log(f"    Unexpected: {unexpected}")
    load_optim(agent, checkpoint["optims_state_dict"])
    # LR scheduler (warmup does not restart), GradScaler (AMP scale continues), slow target update counter.
    if "scheduler_state_dict" in checkpoint:
        agent._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # A bfloat16 run (model.amp_dtype) has a disabled scaler: it ignores a float16 state and saves an empty one, which a
    # float16 run cannot load.
    if checkpoint.get("scaler_state_dict"):
        agent._scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if "slow_value_updates" in checkpoint:
        agent._slow_value_updates = checkpoint["slow_value_updates"]
    # DreamerPro EMA update counter (only present with the dreamerpro config).
    if "ema_updates" in checkpoint and hasattr(agent, "_ema_updates"):
        agent._ema_updates = checkpoint["ema_updates"]
    return {
        "step": checkpoint.get("step", 0),
        "curriculum_step": checkpoint.get("curriculum_step", 0),
        "init_step": checkpoint.get("init_step", 0),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def restored_entry(plan: StartPlan, checkpoint, key):
    """``checkpoint[key]`` if this start restores it, else None: the PFSP pool (``pfsp_state``) and the self-play score
    window (``selfplay_score_buffer``) come back on a resume only, never from a fine-tune source."""
    flags = {"pfsp_state": plan.restore_pool, "selfplay_score_buffer": plan.restore_scores}
    if key not in flags:
        raise KeyError(key)
    return checkpoint.get(key) if checkpoint is not None and flags[key] else None


def start_pfsp_pool(pool, saved_pfsp, include_initial, resume_step, snapshot_fn) -> tuple[dict, bool]:
    """Restore the pool from ``saved_pfsp`` (resume), else start it: member 0 = ``snapshot_fn()`` at ``resume_step``
    (``include_initial``), or empty. Returns ``({"active", "last_snapshot"}, restored)``."""
    if saved_pfsp is not None:
        pool.load_state_dict(saved_pfsp["pool"])
        return {"active": int(saved_pfsp["active"]), "last_snapshot": int(saved_pfsp["last_snapshot"])}, True
    if include_initial:
        return {"active": pool.add(resume_step, snapshot_fn()), "last_snapshot": int(resume_step)}, False
    return {"active": LATEST, "last_snapshot": int(resume_step)}, False


def buffer_dir_to_restore(plan: StartPlan, logdir) -> Path | None:
    """The saved replay buffer a start loads: the one named by ``replay_buffer.latest`` on a resume, never otherwise."""
    if not plan.restore_buffer:
        return None
    logdir = Path(logdir)
    pointer = logdir / "replay_buffer.latest"
    target = logdir / (pointer.read_text().strip() if pointer.exists() else "replay_buffer")
    return target if target.exists() else None


def checkpoint_step(path) -> int:
    import torch

    try:
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except RuntimeError:  # not a zip-format file: read it fully
        ck = torch.load(path, map_location="cpu", weights_only=False)
    return int(ck["step"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-step", metavar="CKPT", required=True)
    print(checkpoint_step(parser.parse_args().print_step))
