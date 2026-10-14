"""Prioritized fictitious self-play (PFSP) over the learner's own snapshots.

The pool stores snapshots of the learner (encoder, RSSM and actor state dicts) on
disk under ``root``. At every rotation the next opponent is

  - with probability ``p_latest``: the learner's current weights (``LATEST``), or
  - otherwise a pool member i, drawn with weight ``(1 - x_i) ** power + floor``,

where x_i is the learner's Beta-smoothed score (W + D/2) / N against member i
(prior: ``prior_games`` games at score 0.5, so unplayed members count as 0.5).
Hard opponents (low x_i) are played most; the floor keeps beaten ones in the
rotation so their play style is not forgotten.

``score_mode="decisive"`` scores on decisive games only, x_i = (W + p/2) / (W + L + p): with mostly
drawn games, (W + D/2) / N stays near 0.5 for every member and the sampling becomes uniform
(2026-10-11: the pilot's 31 members all scored 0.49-0.68 at 75% draws).

``selection="thompson"`` (with ``score_mode="decisive"``): at every pool rotation each member's win rate is
drawn from its Beta posterior, x_i ~ Beta(W_i + p/2, L_i + p/2), and the member is chosen with the weights
(1 - x_i) ** power + floor of these draws. Members with few games have wide posteriors and are explored
automatically (no fixed game threshold); clearly beaten members are rarely drawn.

Only metadata (ids, steps, game counts, RNG state) goes into training checkpoints;
the snapshots themselves are written once.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

LATEST = -1


class PFSPPool:
    def __init__(self, root, p_latest=0.5, power=2.0, floor=0.05, prior_games=2.0, seed=0, score_mode="wdl",
                 selection="weights"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.p_latest = float(p_latest)
        self.power = float(power)
        self.floor = float(floor)
        self.prior_games = float(prior_games)
        if score_mode not in ("wdl", "decisive"):
            raise ValueError(f"PFSP score_mode must be 'wdl' or 'decisive', got {score_mode!r}")
        self.score_mode = score_mode
        if selection not in ("weights", "thompson"):
            raise ValueError(f"PFSP selection must be 'weights' or 'thompson', got {selection!r}")
        if selection == "thompson" and score_mode != "decisive":
            raise ValueError("PFSP selection 'thompson' samples decisive-game win rates: needs score_mode 'decisive'")
        self.selection = selection
        self.members: list[dict] = []
        self.latest = {"wins": 0, "draws": 0, "losses": 0}
        self._rng = random.Random(seed)

    # -- snapshots ---------------------------------------------------------------
    def add(self, step: int, state: dict) -> int:
        """Store a snapshot ``{module_name: state_dict}`` and return its member id."""
        mid = len(self.members)
        path = self.root / f"member_{mid:03d}_step{int(step)}.pt"
        torch.save(state, path)
        self.members.append({"id": mid, "step": int(step), "file": path.name, "wins": 0, "draws": 0, "losses": 0})
        self.save_meta()
        return mid

    def load(self, mid: int) -> dict:
        return torch.load(self.root / self.members[mid]["file"], map_location="cpu", weights_only=True)

    # -- results -----------------------------------------------------------------
    def record(self, mid: int, wins: int, draws: int, losses: int) -> None:
        entry = self.latest if mid == LATEST else self.members[mid]
        entry["wins"] += int(wins)
        entry["draws"] += int(draws)
        entry["losses"] += int(losses)

    def score(self, mid: int) -> float:
        """Learner's smoothed score against ``mid``: (W + D/2) / N, or W / (W + L) with ``score_mode="decisive"``."""
        e = self.latest if mid == LATEST else self.members[mid]
        if self.score_mode == "decisive":
            return (e["wins"] + 0.5 * self.prior_games) / (e["wins"] + e["losses"] + self.prior_games)
        games = e["wins"] + e["draws"] + e["losses"]
        return (e["wins"] + 0.5 * e["draws"] + 0.5 * self.prior_games) / (games + self.prior_games)

    @staticmethod
    def games(entry: dict) -> int:
        return entry["wins"] + entry["draws"] + entry["losses"]

    def _thompson_weights(self, rng) -> list[float]:
        a = 0.5 * self.prior_games
        return [(1.0 - rng.betavariate(m["wins"] + a, m["losses"] + a)) ** self.power + self.floor for m in self.members]

    def weights(self) -> list[float]:
        return [(1.0 - self.score(m["id"])) ** self.power + self.floor for m in self.members]

    def probabilities(self, samples: int = 400) -> dict[int, float]:
        """Selection probability of every option (LATEST and each member); Monte Carlo for Thompson sampling
        (with its own RNG, so logging does not change the training draws)."""
        if not self.members:
            return {LATEST: 1.0}
        if self.selection == "thompson":
            rng = random.Random(len(self.members) * 7919 + sum(self.games(m) for m in self.members))
            counts = [0.0] * len(self.members)
            for _ in range(samples):
                w = self._thompson_weights(rng)
                total = sum(w)
                for i, wi in enumerate(w):
                    counts[i] += wi / total
            w = counts
        else:
            w = self.weights()
        total = sum(w)
        probs = {m["id"]: (1.0 - self.p_latest) * wi / total for m, wi in zip(self.members, w)}
        probs[LATEST] = self.p_latest
        return probs

    def choose(self) -> int:
        if not self.members or self._rng.random() < self.p_latest:
            return LATEST
        w = self._thompson_weights(self._rng) if self.selection == "thompson" else self.weights()
        return self._rng.choices([m["id"] for m in self.members], weights=w, k=1)[0]

    # -- persistence ---------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "members": [dict(m) for m in self.members],
            "latest": dict(self.latest),
            "rng": self._rng.getstate(),
            "settings": {
                "p_latest": self.p_latest, "power": self.power, "floor": self.floor, "prior_games": self.prior_games,
                "score_mode": self.score_mode, "selection": self.selection,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        missing = [m["file"] for m in state["members"] if not (self.root / m["file"]).is_file()]
        if missing:
            raise FileNotFoundError(f"PFSP pool files missing under {self.root}: {missing[:3]}")
        self.members = [dict(m) for m in state["members"]]
        self.latest = dict(state["latest"])
        self._rng.setstate(_as_rng_state(state["rng"]))
        self.save_meta()

    def save_meta(self) -> None:
        meta = {"members": self.members, "latest": self.latest, "p_latest": self.p_latest,
                "power": self.power, "floor": self.floor, "prior_games": self.prior_games,
                "score_mode": self.score_mode, "selection": self.selection}
        tmp = self.root / "pool.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(self.root / "pool.json")


def _as_rng_state(state):
    """random.Random states round-trip through torch.save as nested lists."""
    version, internal, gauss = state
    return (version, tuple(internal), gauss)


def outcome_counts(scores) -> tuple[int, int, int]:
    """(wins, draws, losses) from the self-play wrapper's +1 / 0 / -1 game scores."""
    wins = sum(1 for s in scores if s > 0.5)
    losses = sum(1 for s in scores if s < -0.5)
    return wins, len(scores) - wins - losses, losses
