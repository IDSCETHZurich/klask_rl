"""Per-game evaluation metrics tracker shared by PPO and Dreamer eval scripts.

Owns:
  * The existing termination accounting (total games, per-termination counts,
    Player wins / Opponent wins / Draws / win-rate, live tqdm postfix).
  * The new per-game metrics: ball-contact counts, ball-speed stats, half-occupancy
    fractions, game length, plus the integer outcome code that lets a downstream
    plotting script slice any metric by termination type.

Outcome priority on simultaneous flags (rare but possible):
    goal_scored > goal_conceded > player_in_goal > opponent_in_goal > time_out
The summed-bool ``term_counts`` totals match the existing eval scripts' behavior
(a tick with two flags increments two counters); the per-game ``outcome`` code
collapses to a single canonical category via the priority above.

Known small bias from auto-reset:
    ``update()`` runs after ``env.step()``. For envs that terminated on this
    step, IsaacLab manager-based envs auto-reset in-place inside ``step()``, so
    the state we read for them is the *post-reset* state of the next episode.
    This adds one tick of post-reset data (typically ball at rest) to the just-
    finished episode's accumulators, slightly under-biasing ``mean_ball_speed``
    and shifting half-occupancy fractions by O(1/N). ``game_length_s`` is
    unaffected (the step *did* happen). ``max_ball_speed`` and contact counts
    are also effectively unaffected because the reset state has zero velocity
    and pegs are far from the ball.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg

from klask_rl.tasks.manager_based.klask_rl.utils_manager_based import (
    ball_in_own_half,
    ball_speed,
    collision_player_ball_simple,
)

OUTCOME_NAMES = ("goal_scored", "goal_conceded", "player_in_goal", "opponent_in_goal", "time_out")
OUTCOME_PRIORITY = OUTCOME_NAMES

TERM_NAME_MAP = {
    "goal_scored": "player_scored",
    "goal_conceded": "opponent_scored",
    "player_in_goal": "player_in_goal",
    "opponent_in_goal": "opponent_in_goal",
    "time_out": "time_expired",
}


def _mean_or_nan(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _fmt(value: float, fmt: str = "{:.3f}") -> str:
    if not np.isfinite(value):
        return "N/A"
    return fmt.format(value)


class EvalMetricsTracker:
    """Vectorized per-game metrics + termination accounting.

    Lifecycle per evaluation:
        tracker = EvalMetricsTracker(env)
        loop:
            env.step(...)
            tracker.update()
            if dones.any():
                tracker.finalize(dones)
                pbar.set_postfix(**tracker.live_postfix())
        summary_lines += tracker.summary_body_lines()
        tracker.save_npz(path)

    Spawn a fresh tracker per evaluation run (e.g., per Dreamer checkpoint) so the
    per-game lists do not bleed across runs.
    """

    def __init__(self, env, contact_threshold: float = 0.017, dt: float | None = None):
        unwrapped = env.unwrapped
        # The utility functions in utils_manager_based.py read env.scene[...] /
        # env.num_envs directly, so they need the unwrapped ManagerBasedRLEnv,
        # not the RL-Games / Dreamer wrappers.
        self._env = unwrapped
        self.num_envs: int = int(unwrapped.num_envs)
        self.device: torch.device = torch.device(unwrapped.device)
        if dt is not None:
            self.dt: float = float(dt)
        else:
            self.dt = float(getattr(unwrapped, "step_dt", 0.02))
        self._contact_eps: float = float(contact_threshold)

        self._term_manager = unwrapped.termination_manager

        self._ball_cfg = SceneEntityCfg("ball")
        self._ball_cfg.resolve(unwrapped.scene)
        self._player_cfg = SceneEntityCfg("klask", body_names=["Peg_1"])
        self._player_cfg.resolve(unwrapped.scene)
        self._opponent_cfg = SceneEntityCfg("klask", body_names=["Peg_2"])
        self._opponent_cfg.resolve(unwrapped.scene)

        N = self.num_envs
        d = self.device
        self._step_count = torch.zeros(N, dtype=torch.int64, device=d)
        self._player_contacts = torch.zeros(N, dtype=torch.int64, device=d)
        self._opponent_contacts = torch.zeros(N, dtype=torch.int64, device=d)
        self._prev_player_contact = torch.zeros(N, dtype=torch.bool, device=d)
        self._prev_opponent_contact = torch.zeros(N, dtype=torch.bool, device=d)
        self._max_ball_speed = torch.zeros(N, dtype=torch.float32, device=d)
        self._sum_ball_speed = torch.zeros(N, dtype=torch.float32, device=d)
        self._steps_player_half = torch.zeros(N, dtype=torch.int64, device=d)
        self._steps_opponent_half = torch.zeros(N, dtype=torch.int64, device=d)

        self.total_games: int = 0
        self.term_counts: dict[str, int] = {v: 0 for v in TERM_NAME_MAP.values()}

        self.player_contacts: list[int] = []
        self.opponent_contacts: list[int] = []
        self.max_ball_speed: list[float] = []
        self.mean_ball_speed: list[float] = []
        self.frac_player_half: list[float] = []
        self.frac_opponent_half: list[float] = []
        self.game_length_s: list[float] = []
        self.outcome: list[int] = []
        self.termination_flags: list[list[bool]] = []
        self._outcome_counts = [0] * len(OUTCOME_NAMES)

    def _zero_envs(self, mask: torch.Tensor) -> None:
        self._step_count[mask] = 0
        self._player_contacts[mask] = 0
        self._opponent_contacts[mask] = 0
        self._prev_player_contact[mask] = False
        self._prev_opponent_contact[mask] = False
        self._max_ball_speed[mask] = 0.0
        self._sum_ball_speed[mask] = 0.0
        self._steps_player_half[mask] = 0
        self._steps_opponent_half[mask] = 0

    def update(self) -> None:
        """Per-step accumulation. Call after every ``env.step()``."""
        env = self._env
        p_contact = collision_player_ball_simple(env, self._player_cfg, self._ball_cfg, eps=self._contact_eps).bool()
        o_contact = collision_player_ball_simple(env, self._opponent_cfg, self._ball_cfg, eps=self._contact_eps).bool()
        b_speed = ball_speed(env, self._ball_cfg)
        in_player_half = ball_in_own_half(env, self._ball_cfg, own_half_sign=-1.0).bool()
        in_opponent_half = ball_in_own_half(env, self._ball_cfg, own_half_sign=+1.0).bool()

        self._player_contacts += (p_contact & ~self._prev_player_contact).long()
        self._opponent_contacts += (o_contact & ~self._prev_opponent_contact).long()
        self._prev_player_contact = p_contact
        self._prev_opponent_contact = o_contact

        self._max_ball_speed = torch.maximum(self._max_ball_speed, b_speed)
        self._sum_ball_speed += b_speed

        self._steps_player_half += in_player_half.long()
        self._steps_opponent_half += in_opponent_half.long()
        self._step_count += 1

    def finalize(self, dones: torch.Tensor) -> None:
        """Flush per-game stats for terminated envs and update totals."""
        done_mask_self = dones.bool().to(self.device)
        num_done = int(done_mask_self.sum().item())
        if num_done == 0:
            return
        td = self._term_manager._term_dones
        done_mask_term = done_mask_self.to(td.device)

        term_name_to_idx = {name: i for i, name in enumerate(self._term_manager._term_names)}
        # Keep the canonical code even when an earlier termination is disabled.
        priority_indices = [
            (OUTCOME_NAMES.index(n), term_name_to_idx[n]) for n in OUTCOME_PRIORITY if n in term_name_to_idx
        ]
        td_done_cpu = td[done_mask_term].detach().cpu()
        outcome_codes = []
        for row in range(num_done):
            code = next((code for code, idx in priority_indices if td_done_cpu[row, idx]), None)
            if code is None:
                raise RuntimeError("Finished game has no recognized termination; cannot assign W/L/D.")
            outcome_codes.append(code)

        # Validate the entire batch before mutating any game-level accounting.
        self.total_games += num_done
        for i, term_name in enumerate(self._term_manager._term_names):
            if term_name in TERM_NAME_MAP:
                count_key = TERM_NAME_MAP[term_name]
                self.term_counts[count_key] += int(td_done_cpu[:, i].sum().item())

        done_envs = done_mask_self.nonzero(as_tuple=True)[0].tolist()

        step_cpu = self._step_count.detach().cpu().tolist()
        pc_cpu = self._player_contacts.detach().cpu().tolist()
        oc_cpu = self._opponent_contacts.detach().cpu().tolist()
        max_cpu = self._max_ball_speed.detach().cpu().tolist()
        sum_cpu = self._sum_ball_speed.detach().cpu().tolist()
        sph_cpu = self._steps_player_half.detach().cpu().tolist()
        soh_cpu = self._steps_opponent_half.detach().cpu().tolist()

        for row, env_idx in enumerate(done_envs):
            steps = max(step_cpu[env_idx], 1)
            self.player_contacts.append(int(pc_cpu[env_idx]))
            self.opponent_contacts.append(int(oc_cpu[env_idx]))
            self.max_ball_speed.append(float(max_cpu[env_idx]))
            self.mean_ball_speed.append(float(sum_cpu[env_idx]) / steps)
            self.frac_player_half.append(float(sph_cpu[env_idx]) / steps)
            self.frac_opponent_half.append(float(soh_cpu[env_idx]) / steps)
            self.game_length_s.append(float(step_cpu[env_idx]) * self.dt)

            outcome_code = outcome_codes[row]
            self.outcome.append(int(outcome_code))
            self._outcome_counts[outcome_code] += 1
            self.termination_flags.append(td_done_cpu[row].bool().tolist())

        self._zero_envs(done_mask_self)

    def live_postfix(self) -> dict:
        p_wins = self.player_wins
        o_wins = self.opponent_wins
        draws = self.draws
        wr = f"{p_wins / self.total_games * 100:.1f}%" if self.total_games > 0 else "N/A"
        return {"P_wins": p_wins, "O_wins": o_wins, "Draws": draws, "P_wr": wr}

    @property
    def player_wins(self) -> int:
        return self._outcome_counts[0] + self._outcome_counts[3]

    @property
    def opponent_wins(self) -> int:
        return self._outcome_counts[1] + self._outcome_counts[2]

    @property
    def draws(self) -> int:
        return self._outcome_counts[4]

    def summary_body_lines(self) -> list[str]:
        tc = self.term_counts
        lines = [
            f"  Player scored (goal_scored)      : {tc['player_scored']}",
            f"  Opponent scored (goal_conceded)   : {tc['opponent_scored']}",
            f"  Player fell in goal (player_in)   : {tc['player_in_goal']}",
            f"  Opponent fell in goal (opp_in)    : {tc['opponent_in_goal']}",
            f"  Time expired (time_out)           : {tc['time_expired']}",
            "-" * 50,
            f"  Player wins  : {self.player_wins}",
            f"  Opponent wins: {self.opponent_wins}",
            f"  Draws        : {self.draws}",
        ]
        if self.total_games > 0:
            lines.append(f"  Player win rate: {self.player_wins / self.total_games * 100:.1f}%")

        outcome = np.asarray(self.outcome, dtype=np.int64)
        scored_mask = outcome == 0  # goal_scored
        conceded_mask = outcome == 1  # goal_conceded
        timeout_mask = outcome == 4  # time_out

        def _slice_mean(values: list[float], mask: np.ndarray) -> float:
            if len(values) == 0 or not mask.any():
                return float("nan")
            return float(np.asarray(values, dtype=np.float64)[mask].mean())

        lines += [
            "-" * 50,
            "  --- Per-game metrics (mean across games) ---",
            f"  Player-ball contacts / game        : {_fmt(_mean_or_nan(self.player_contacts), '{:.2f}')}",
            f"  Opponent-ball contacts / game      : {_fmt(_mean_or_nan(self.opponent_contacts), '{:.2f}')}",
            f"  Player contacts | player scored    : {_fmt(_slice_mean(self.player_contacts, scored_mask), '{:.2f}')}",
            f"  Player contacts | opponent scored  : {_fmt(_slice_mean(self.player_contacts, conceded_mask), '{:.2f}')}",
            f"  Opponent contacts | player scored  : {_fmt(_slice_mean(self.opponent_contacts, scored_mask), '{:.2f}')}",
            f"  Opponent contacts | opponent scored: {_fmt(_slice_mean(self.opponent_contacts, conceded_mask), '{:.2f}')}",
            f"  Max ball speed [m/s] / game        : {_fmt(_mean_or_nan(self.max_ball_speed))}",
            f"  Mean ball speed [m/s] / game       : {_fmt(_mean_or_nan(self.mean_ball_speed))}",
            f"  Frac time ball in player half      : {_fmt(_mean_or_nan(self.frac_player_half))}",
            f"  Frac time ball in opponent half    : {_fmt(_mean_or_nan(self.frac_opponent_half))}",
            f"  Frac player half | time_out        : {_fmt(_slice_mean(self.frac_player_half, timeout_mask))}",
            f"  Frac opponent half | time_out      : {_fmt(_slice_mean(self.frac_opponent_half, timeout_mask))}",
            f"  Time until player scored [s]       : {_fmt(_slice_mean(self.game_length_s, scored_mask))}",
            f"  Time until player conceded [s]     : {_fmt(_slice_mean(self.game_length_s, conceded_mask))}",
            f"  Game length [s]                    : {_fmt(_mean_or_nan(self.game_length_s))}",
        ]
        return lines

    def save_npz(self, path: str, metadata: dict | None = None, *, extra_arrays: dict | None = None) -> None:
        """Save per-game arrays + scalar metadata to a numpy ``.npz`` file.

        Args:
            path: Output file path.
            metadata: Optional free-form string metadata (e.g. checkpoint paths).
                Each key is stored under ``meta_<key>`` in the npz.
            extra_arrays: Additional aligned per-game arrays, e.g. environment IDs.
        """
        data = {
            "outcome_schema_version": np.asarray(2, dtype=np.int64),
            "termination_names": np.asarray(self._term_manager._term_names),
            "termination_flags": np.asarray(self.termination_flags, dtype=bool).reshape(
                -1, len(self._term_manager._term_names)
            ),
            "player_contacts": np.asarray(self.player_contacts, dtype=np.int64),
            "opponent_contacts": np.asarray(self.opponent_contacts, dtype=np.int64),
            "max_ball_speed": np.asarray(self.max_ball_speed, dtype=np.float32),
            "mean_ball_speed": np.asarray(self.mean_ball_speed, dtype=np.float32),
            "frac_player_half": np.asarray(self.frac_player_half, dtype=np.float32),
            "frac_opponent_half": np.asarray(self.frac_opponent_half, dtype=np.float32),
            "game_length_s": np.asarray(self.game_length_s, dtype=np.float32),
            "outcome": np.asarray(self.outcome, dtype=np.int8),
            "outcome_legend": np.asarray(OUTCOME_NAMES),
            "dt": np.asarray(self.dt, dtype=np.float64),
            "contact_threshold": np.asarray(self._contact_eps, dtype=np.float64),
            "num_games": np.asarray(self.total_games, dtype=np.int64),
        }
        for k, v in self.term_counts.items():
            data[f"term_count_{k}"] = np.asarray(v, dtype=np.int64)
        if metadata:
            for k, v in metadata.items():
                data[f"meta_{k}"] = np.asarray(str(v))
        for key, values in (extra_arrays or {}).items():
            if key in data or len(values) != self.total_games:
                raise ValueError(f"Invalid per-game array: {key}")
            data[key] = np.asarray(values)
        np.savez(path, **data)
