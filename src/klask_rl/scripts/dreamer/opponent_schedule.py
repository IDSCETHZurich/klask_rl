"""Fixed opponent schedules for controlled opponent-switch experiments.

A schedule replaces the score-gated self-play update with predetermined frozen
opponents that take over at fixed environment steps::

    self_play:
      enabled: true
      schedule:
        - {start: 0,       kind: dreamer, checkpoint: logs/.../checkpoint_200000.pt,
           config: logs/.../.hydra/config.yaml}
        - {start: 1600000, kind: ppo, checkpoint: logs/.../klask_ppo_nn_v1.1.pth,
           config: logs/.../params/agent.yaml}

Opponents can only change at the trainer's eval hook, so every ``start`` must be
a multiple of ``trainer.eval_every``. Pure Python (no Isaac Lab / torch), so the
parsing and lookup are unit-testable.
"""

from dataclasses import dataclass
from pathlib import Path

KINDS = ("dreamer", "ppo")


@dataclass(frozen=True)
class OpponentPhase:
    index: int
    start: int
    kind: str
    checkpoint: str
    config: str


def resolve_path(value, root):
    """Resolve a schedule path: absolute paths under the container mount
    ``/workspace/klask_rl`` are remapped to ``root``; relative paths are taken
    relative to ``root`` (the project directory containing ``logs/``)."""
    path = Path(str(value)).expanduser()
    root = Path(root)
    if path.is_relative_to("/workspace/klask_rl"):
        path = root / path.relative_to("/workspace/klask_rl")
    elif not path.is_absolute():
        path = root / path
    return path.resolve()


def parse_schedule(entries, eval_every, root, check_files=True):
    """Validate a schedule and return a tuple of :class:`OpponentPhase`.

    Raises ``ValueError`` on an empty schedule, a first start other than 0,
    non-increasing starts, starts that are not multiples of ``eval_every``,
    unknown kinds, a missing config (both kinds need one) or missing files.
    """
    if not entries:
        raise ValueError("Opponent schedule is empty")
    eval_every = int(float(eval_every))
    if eval_every <= 0:
        raise ValueError("trainer.eval_every must be positive for an opponent schedule")
    phases = []
    for index, entry in enumerate(entries):
        entry = dict(entry)
        start = int(float(entry["start"]))
        kind = str(entry["kind"])
        if kind not in KINDS:
            raise ValueError(f"Phase {index}: unknown kind {kind!r}; expected one of {KINDS}")
        if start % eval_every:
            raise ValueError(
                f"Phase {index}: start {start} is not a multiple of trainer.eval_every={eval_every}; "
                "opponents can only change at the eval hook"
            )
        if not entry.get("checkpoint") or not entry.get("config"):
            raise ValueError(f"Phase {index}: 'checkpoint' and 'config' are required")
        checkpoint = resolve_path(entry["checkpoint"], root)
        config = resolve_path(entry["config"], root)
        if check_files:
            for label, path in (("checkpoint", checkpoint), ("config", config)):
                if not path.is_file():
                    raise ValueError(f"Phase {index}: {label} not found: {path}")
        phases.append(OpponentPhase(index, start, kind, str(checkpoint), str(config)))
    if phases[0].start != 0:
        raise ValueError("The first opponent phase must start at step 0")
    starts = [phase.start for phase in phases]
    if any(b <= a for a, b in zip(starts, starts[1:])):
        raise ValueError(f"Opponent phase starts must be strictly increasing: {starts}")
    if phases[0].kind != "dreamer":
        # The Dreamer opponent networks (also used for imagination) are built
        # from the first phase; a PPO-first schedule would leave them undefined.
        raise ValueError("The first opponent phase must be a Dreamer snapshot")
    return tuple(phases)


def phase_at(phases, step):
    """Index of the phase active at environment ``step``."""
    step = int(step)
    current = 0
    for phase in phases:
        if phase.start <= step:
            current = phase.index
        else:
            break
    return current


def last_dreamer_phase(phases, index):
    """Index of the most recent Dreamer phase at or before ``index`` (the
    imagination opponent during a PPO phase)."""
    for phase in reversed(phases[: index + 1]):
        if phase.kind == "dreamer":
            return phase.index
    raise ValueError("No Dreamer phase at or before phase %d" % index)


def eval_anchor(step, eval_every):
    """Last multiple of ``eval_every`` at or before ``step``.

    A resumed trainer schedules its next eval ``eval_every`` steps after the
    resume step. Anchoring its eval counter here keeps the eval hook, and so
    every opponent switch, on the schedule's grid after any resume.
    """
    eval_every = int(float(eval_every))
    return (int(step) // eval_every) * eval_every


def opponent_fingerprint(modules):
    """Order-sensitive checksum of module weights (detects identical opponents)."""
    import torch

    total = 0.0
    with torch.no_grad():
        for module in modules:
            for index, param in enumerate(module.parameters()):
                total += float((param.double().flatten() * (1 + (index % 7) * 1e-3)).sum())
    return total

