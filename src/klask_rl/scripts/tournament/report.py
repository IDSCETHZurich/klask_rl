"""Recompute tournament statistics from individual games, without Isaac Lab."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PAYOFF = {
    "goal_scored": 1.0,
    "goal_conceded": 0.0,
    "player_in_goal": 0.0,
    "opponent_in_goal": 1.0,
    "time_out": 0.5,
}


def load_payoffs(path, *, reverse=False):
    with np.load(path, allow_pickle=False) as data:
        codes = data["outcome"]
        legend = data["outcome_legend"].tolist()
        if codes.ndim != 1 or codes.dtype.kind not in "iu":
            raise ValueError(f"{path}: outcome must be a 1D integer array")
        if np.any(codes < 0) or np.any(codes >= len(legend)):
            raise ValueError(f"{path}: invalid outcome codes")
        if len(codes) != int(data["num_games"]):
            raise ValueError(f"{path}: num_games does not match outcomes")
        if "outcome_schema_version" not in data:
            raise ValueError(
                f"{path}: legacy outcome codes may be shifted when terms were disabled; "
                "verify the original active term list before migrating this file"
            )
        values = np.asarray([PAYOFF[name] for name in legend])[codes]
    return 1.0 - values if reverse else values


def statistics(payoffs, samples=10000, seed=2026):
    values = np.asarray(payoffs, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isin(values, [0, 0.5, 1]).all():
        raise ValueError("Expected a nonempty 1D array of game payoffs in {0, 0.5, 1}")
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    n = len(values)
    wins, losses, draws = (int((values == x).sum()) for x in (1, 0, 0.5))
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    # Explicit sampling of individual games with replacement. Bounded memory.
    batch = max(1, min(256, 2_000_000 // n))
    for start in range(0, samples, batch):
        size = min(batch, samples - start)
        means[start : start + size] = rng.choice(values, size=(size, n), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "N": n,
        "W": wins,
        "L": losses,
        "D": draws,
        "score": float(values.mean()),
        "decisive_ratio": wins / losses if losses else ("infinity" if wins else "undefined"),
        "score_ci_low": float(low),
        "score_ci_high": float(high),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def generate_report(directory):
    directory = Path(directory)
    plan = json.loads((directory / "manifest.json").read_text())
    rows, missing = [], []
    for match in plan["matches"]:
        payoffs = []
        complete = True
        for job in [j for j in plan["jobs"] if j["match"] == match["id"]]:
            path = directory / job["id"] / "games.npz"
            if not path.exists():
                missing.append(job["id"])
                complete = False
                continue
            with np.load(path, allow_pickle=False) as data:
                if str(data["meta_run_id"]) != plan["run_id"] or str(data["meta_job_id"]) != job["id"]:
                    raise ValueError(f"Provenance mismatch: {path}")
                leg_complete = str(data["meta_complete"]) == "True" and int(data["num_games"]) == job["games"]
                if not leg_complete:
                    missing.append(job["id"] + " (partial)")
                    complete = False
            payoffs.append(load_payoffs(path, reverse=job["seat0"] != match["a"]))
        values = np.concatenate(payoffs) if payoffs else np.empty(0)
        if not len(values):
            continue
        rows.append(
            {
                "match": match["id"],
                "agent": match["a"],
                "opponent": match["b"],
                "condition": match["condition"],
                "primary": match["primary"],
                "complete": complete,
                **statistics(values, plan["bootstrap_samples"], plan["bootstrap_seed"]),
            }
        )
    output = {
        "run_id": plan["run_id"],
        "complete": not missing,
        "missing_or_partial": missing,
        "results": rows,
    }
    (directory / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if rows:
        with (directory / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Tournament results",
        "",
        (
            "All counts and intervals use individual games. "
            "Primary D-RSSM conditions use zeroed opponent input. "
            "W/L is infinity when L=0<W and undefined when W=L=0."
        ),
        "",
    ]
    if missing:
        lines += ["**INCOMPLETE — missing/partial legs:** " + ", ".join(missing), ""]
    ablation = {r["condition"]: r for r in rows if r["agent"] == "D-RSSM" and r["opponent"] == "PPO-B"}
    if ablation:
        lines += [
            "## D-RSSM vs PPO-B",
            "",
            "| Metric | Exact (diagnostic) | Zero (primary/deployable) |",
            "|---|---:|---:|",
        ]
        for key in (
            "N",
            "W",
            "L",
            "D",
            "score",
            "decisive_ratio",
            "score_ci_low",
            "score_ci_high",
            "complete",
        ):
            vals = [ablation.get(c, {}).get(key, "pending") for c in ("exact", "zero")]
            lines.append(f"| {key} | {vals[0]} | {vals[1]} |")
    lines += [
        "",
        "## All evaluated conditions",
        "",
        "| Agent | Opponent | Condition | N | W | L | D | Score | W/L | 95% score CI | Complete |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['agent']} | {row['opponent']} | {row['condition']} | {row['N']} | "
            f"{row['W']} | {row['L']} | {row['D']} | {row['score']:.5f} | {row['decisive_ratio']} | "
            f"[{row['score_ci_low']:.5f}, {row['score_ci_high']:.5f}] | {row['complete']} |"
        )
    lines += [
        "",
        (
            "## Complete primary round robin (row agent perspective)"
            if sum(m["primary"] for m in plan["matches"]) == 3
            else "## Selected primary matchups (row agent perspective)"
        ),
        "",
        "| Agent | Opponent | N | W | L | D | Score | W/L | 95% score CI | Complete |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        if not row["primary"]:
            continue
        for reverse in (False, True):
            w, l = (row["L"], row["W"]) if reverse else (row["W"], row["L"])
            a, b = (row["opponent"], row["agent"]) if reverse else (row["agent"], row["opponent"])
            score = 1 - row["score"] if reverse else row["score"]
            lo, hi = (
                (1 - row["score_ci_high"], 1 - row["score_ci_low"])
                if reverse
                else (row["score_ci_low"], row["score_ci_high"])
            )
            ratio = w / l if l else ("infinity" if w else "undefined")
            lines.append(
                f"| {a} | {b} | {row['N']} | {w} | {l} | {row['D']} | {score:.5f} | "
                f"{ratio} | [{lo:.5f}, {hi:.5f}] | {row['complete']} |"
            )
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = generate_report(args.directory)
    print((args.directory / "report.md").read_text())
    raise SystemExit(0 if result["complete"] else 1)
