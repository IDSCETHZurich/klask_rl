"""Recompute basic reports/statistics from raw CSVs. Figures are opt-in only."""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tournament"))
from raw_data import utc_now, write_csv, write_json
from report import statistics


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean_interval(values, samples, seed):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite, nonempty observations")
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for offset in range(0, samples, 128):
        means[offset : offset + 128] = rng.choice(values, (min(128, samples - offset), len(values))).mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "independent_samples": len(values),
    }


def games_summary(directory, samples, seed):
    results = []
    lookup = {}
    for ledger in directory.rglob("evaluations.csv"):
        for row in read_rows(ledger):
            location = Path(row["evaluation_directory"])
            if not location.is_absolute():
                location = ledger.parent / location
            elif not location.exists():
                location = ledger.parent / location.name
            lookup[str(location.resolve())] = row
    for path in sorted(directory.rglob("manifest.json")):
        plan = json.loads(path.read_text())
        if "jobs" not in plan or "matches" not in plan:
            continue
        for match in plan["matches"]:
            values, complete, game_keys = [], True, set()
            for job in (job for job in plan["jobs"] if job["match"] == match["id"]):
                csv_path = path.parent / job["id"] / "games.csv"
                if not csv_path.exists():
                    complete = False
                    continue
                rows = read_rows(csv_path)
                if len(rows) > job["games"]:
                    raise ValueError(f"More games than requested: {csv_path}")
                complete &= len(rows) == job["games"]
                for row in rows:
                    if row["run_id"] != plan["run_id"] or row["job_id"] != job["id"]:
                        raise ValueError(f"CSV provenance mismatch: {csv_path}")
                    if any(str(job[key]) != row[key] for key in ("match", "seat0", "seat1", "condition", "seed")):
                        raise ValueError(f"CSV settings mismatch: {csv_path}")
                    key = (row["job_id"], row["env_id"], row["episode_id"])
                    if key in game_keys:
                        raise ValueError(f"Duplicate game: {key}")
                    game_keys.add(key)
                    payoff = float(row["payoff_seat0"])
                    expected = {0: 1, 1: 0, 2: 0, 3: 1, 4: 0.5}[int(row["outcome"])]
                    if payoff != expected:
                        raise ValueError("CSV payoff disagrees with canonical outcome")
                    values.append(payoff if job["seat0"] == match["a"] else 1 - payoff)
            if not values:
                continue
            curve = lookup.get(str(path.parent.resolve()), {})
            experiment = plan.get("experiment", {})
            results.append(
                dict(
                    run_id=plan["run_id"],
                    match=match["id"],
                    agent=match["a"],
                    opponent=match["b"],
                    condition=match["condition"],
                    complete=complete,
                    algorithm=curve.get("algorithm", experiment.get("algorithm", "")),
                    training_seed=curve.get("seed", experiment.get("training_seed", "")),
                    environment_steps=curve.get("environment_steps", experiment.get("environment_steps", "")),
                    requested_steps=curve.get("requested_steps", experiment.get("requested_steps", "")),
                    checkpoint=curve.get("checkpoint", experiment.get("checkpoint", "")),
                    **statistics(values, samples, seed),
                )
            )
    return results


def conditioning_summary(paths, samples, seed):
    # Stream matching exact/random/zero rows, then retain only within-game
    # sums. Image diagnostics can contain millions of correlated frame rows.
    metrics = ("image_nll", "image_nll_per_pixel", "posterior_image_nll", "kl_nats")
    names = ("checkpoint", "domain", "opponent", "timing", "horizon")
    conditions = ("exact", "random", "zero")
    pending, seen = {}, set()
    totals = defaultdict(lambda: defaultdict(lambda: np.zeros(len(metrics) + 1)))
    for path in paths:
        with Path(path).open(newline="") as stream:
            for row in csv.DictReader(stream):
                game = tuple(row[key] for key in ("dataset_run_id", "job_id", "trajectory_id"))
                group = tuple(int(row[key]) if key == "horizon" else row[key] for key in names)
                pair = (game, group, row["origin"], row["draw"])
                condition = row["condition"]
                if condition not in conditions:
                    raise ValueError(f"Unknown condition: {condition}")
                identity = hashlib.sha256(repr((pair, condition)).encode()).digest()
                if identity in seen:
                    raise ValueError(f"Duplicate diagnostic observation in {path}; analyze repeated runs separately")
                seen.add(identity)
                values = np.array([float(row[metric]) for metric in metrics])
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite diagnostic loss: {path}")
                cell = pending.setdefault(pair, {})
                cell[condition] = values
                if len(cell) == len(conditions):
                    for label in conditions:
                        total = totals[(group, label)][game]
                        total[:-1] += cell[label]
                        total[-1] += 1
                    for label in ("random", "zero"):
                        total = totals[(group, label + "-exact")][game]
                        total[:-1] += cell[label] - cell["exact"]
                        total[-1] += 1
                    del pending[pair]
    if pending:
        raise ValueError("Unpaired exact/intervention samples; diagnostic CSV may be incomplete")
    summaries, gaps = [], []
    for (group, label), games in sorted(totals.items()):
        means = np.stack([total[:-1] / total[-1] for total in games.values()])
        for index, metric in enumerate(metrics):
            result = dict(zip(names, group), metric=metric, **mean_interval(means[:, index], samples, seed))
            if label in conditions:
                summaries.append(dict(result, condition=label))
            else:
                gaps.append(dict(result, contrast=label))
    return summaries, gaps


def learning_curve_summary(games, samples, seed):
    groups = defaultdict(list)
    for row in games:
        if row["algorithm"] and row["complete"]:
            for target in filter(None, row["requested_steps"].split(";")):
                groups[(row["algorithm"], row["opponent"], int(target))].append(row)
    results = []
    for (algorithm, opponent, target), rows in sorted(groups.items()):
        if len({row["training_seed"] for row in rows}) != len(rows):
            raise ValueError("Duplicate seed at one curve threshold; analyze experiments separately")
        values = [row["score"] for row in rows]
        results.append(
            dict(
                algorithm=algorithm,
                opponent=opponent,
                requested_steps=target,
                actual_steps_min=min(int(row["environment_steps"]) for row in rows),
                actual_steps_max=max(int(row["environment_steps"]) for row in rows),
                seeds=len(rows),
                complete_three_seeds=len(rows) == 3,
                score_std=float(np.std(values, ddof=1)) if len(values) > 1 else "",
                **mean_interval(values, samples, seed),
            )
        )
    return results


def table(rows, columns):
    if not rows:
        return "No completed observations available.\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key]) for key in columns)
            + " |"
        )
    return "\n".join(lines) + "\n"


def plot(gaps, curves, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ("image_nll_per_pixel", "kl_nats"):
        fig, ax = plt.subplots()
        groups = defaultdict(list)
        for row in gaps:
            if row["metric"] == metric:
                groups[(row["domain"], row["timing"], row["contrast"])].append(row)
        for label, rows in groups.items():
            rows = sorted(rows, key=lambda row: row["horizon"])
            ax.plot(
                [r["horizon"] for r in rows],
                [r["mean"] for r in rows],
                label=" / ".join(label),
            )
        if groups:
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set(xlabel="Open-loop horizon (control steps)", ylabel=f"Gap: {metric}")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(output / f"diagnostic_{metric}.png")
        plt.close(fig)
    if curves:
        fig, ax = plt.subplots()
        for algorithm in sorted({row["algorithm"] for row in curves}):
            rows = sorted(
                (row for row in curves if row["algorithm"] == algorithm),
                key=lambda row: row["requested_steps"],
            )
            ax.plot(
                [row["actual_steps_max"] for row in rows],
                [row["mean"] for row in rows],
                "o-",
                label=algorithm,
            )
        ax.set(
            xlabel="Actual environment transitions (maximum across seeds)",
            ylabel="Mean seed score",
            ylim=(0, 1),
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "diagnostic_learning_curve.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Optional basic diagnostic PNGs, never generated by default",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("Positive bootstrap samples required")
    args.output.mkdir(parents=True, exist_ok=True)
    games = games_summary(args.input, args.bootstrap_samples, args.seed)
    summaries, gaps = conditioning_summary(
        sorted(args.input.rglob("conditioning.csv")), args.bootstrap_samples, args.seed
    )
    curves = learning_curve_summary(games, args.bootstrap_samples, args.seed)
    for name, rows in (
        ("games_summary", games),
        ("conditioning_summary", summaries),
        ("conditioning_gaps", gaps),
        ("learning_curve", curves),
    ):
        if rows:
            write_csv(args.output / (name + ".csv"), rows, list(rows[0]))
    metadata_files = sorted(
        path for path in args.input.rglob("metadata.json") if args.output.resolve() not in path.resolve().parents
    )
    notes = "\n".join(f"- [{path.relative_to(args.input)}]({path.resolve()})" for path in metadata_files)
    incomplete = [
        str(path.relative_to(args.input))
        for path in metadata_files
        if json.loads(path.read_text()).get("complete") is False
    ]
    report = (
        "# Basic experiment report\n\n"
        + ("**INCOMPLETE source runs:** " + ", ".join(incomplete) + "\n\n" if incomplete else "")
        + "Source data remain in CSV; JSON metadata link checkpoint hashes, configurations, seeds, commands, timing and step counts.\n\n"
        "Gameplay: W/L/D count individual games; score=(W+0.5D)/N. 95% intervals resample individual game payoffs. "
        "Rows show the named agent's perspective; reverse results are W↔L and score→1-score.\n\n"
        + table(
            games,
            [
                "agent",
                "opponent",
                "condition",
                "training_seed",
                "environment_steps",
                "N",
                "W",
                "L",
                "D",
                "score",
                "decisive_ratio",
                "score_ci_low",
                "score_ci_high",
                "complete",
            ],
        )
        + "\n## Conditioning gaps\n\n"
        "Positive means corrupted input has higher loss than exact. Average origins/draws within each game first, then bootstrap "
        "whole game means. Horizons and conditions are paired, not independent samples. MSEDist image NLL is the repository's "
        "sum of squared pixel errors on image/255, not a calibrated likelihood. KL is raw categorical posterior-to-prior KL "
        "without free-nat clipping. Prior rollouts never consume future images; target images are used only for diagnostics.\n\n"
        + table(
            gaps,
            [
                "domain",
                "opponent",
                "timing",
                "horizon",
                "metric",
                "contrast",
                "mean",
                "ci_low",
                "ci_high",
                "independent_samples",
            ],
        )
        + "\nAbsolute diagnostic losses (equal weight per game):\n\n"
        + table(
            summaries,
            [
                "domain",
                "opponent",
                "timing",
                "horizon",
                "condition",
                "metric",
                "mean",
                "ci_low",
                "ci_high",
                "independent_samples",
            ],
        )
        + "\n## Baseline learning curves\n\n"
        "Checkpoints use total environment transitions, not vector iterations. Requested 9M thresholds may be exceeded by one "
        "complete PPO rollout/FastSAC vector step. Curves average independent seed scores; seed intervals resample seeds, "
        "not pooled games. Three seeds provide limited precision. Training reward is not tournament score.\n\n"
        + table(
            curves,
            [
                "algorithm",
                "requested_steps",
                "actual_steps_min",
                "actual_steps_max",
                "seeds",
                "mean",
                "score_std",
                "ci_low",
                "ci_high",
                "complete_three_seeds",
            ],
        )
        + "\n## Interpretation limits\n\n"
        "ID membership requires documented opponent provenance; an unverified lineage checkpoint is a proxy. "
        "These experiments measure inference sensitivity, not the necessity of conditioning for training stability. "
        "That claim also needs a training ablation. A near-zero gap requires an equivalence margin; nonsignificance alone is insufficient.\n\n"
        "## Metadata\n\n" + notes + "\n"
    )
    (args.output / "report.md").write_text(report)
    write_json(
        args.output / "analysis_metadata.json",
        {
            "timestamp": utc_now(),
            "argv": sys.argv,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "input": str(args.input.resolve()),
            "plots": args.plots,
        },
    )
    if args.plots:
        plot(gaps, curves, args.output)
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
