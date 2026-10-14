"""Multi-checkpoint head-to-head aggregate summary.

Pure stdlib + numpy, no Isaac Lab imports. Lives in ``scripts/`` (not inside
the ``klask_rl`` package) on purpose, mirroring ``eval_plot.py``: importing the
package eagerly loads ``klask_rl.tasks`` which pulls in IsaacLab/Omniverse,
defeating the "runnable without Isaac Sim" goal of this module.

Reads ``h2h_results_<ts>.npz`` files written by
``EvalMetricsTracker.save_npz`` and prints / writes the same aggregate
win-rate table that ``evaluate_dreamer.py`` produces inline at the end of a
multi-checkpoint run.

Usage:
    # Standalone CLI (no Isaac Sim required):
    python scripts/eval_aggregate.py logs/dreamer/head_to_head_results/

    # From a script that already has scripts/ on sys.path:
    from eval_aggregate import aggregate_from_folder
    aggregate_from_folder("logs/dreamer/head_to_head_results/")

    # From the eval scripts (evaluate_dreamer.py) we load it lazily via
    # importlib.util so we don't depend on sys.path tweaks.
"""

from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import numpy as np


COLOR_PLAYER = "#2ca02c"
COLOR_OPPONENT = "#d62728"
COLOR_DRAW = "#7f7f7f"

# Per-termination palette: greens for player-positive outcomes, reds for
# opponent-positive outcomes, gray for time-out.
TERM_SERIES = (
    ("term_count_player_scored", "Player scored", "#2ca02c", "o"),
    ("term_count_opponent_scored", "Opponent scored", "#d62728", "s"),
    ("term_count_player_in_goal", "Player fell in goal", "#ff7f0e", "v"),
    ("term_count_opponent_in_goal", "Opponent fell in goal", "#1f77b4", "^"),
    ("term_count_time_expired", "Time expired", "#7f7f7f", "D"),
)


def _load_h2h_row(npz_path: str) -> dict:
    """Load one h2h_results_<ts>.npz and derive an aggregate-table row.

    Win/loss/draw formulas mirror ``EvalMetricsTracker.player_wins`` /
    ``opponent_wins`` / ``draws`` so the standalone path produces the same
    counts as the inline path.
    """
    data = np.load(npz_path, allow_pickle=False)

    def _scalar(key: str) -> int:
        if key not in data.files:
            return 0
        return int(np.asarray(data[key]).item())

    player_wins = _scalar("term_count_player_scored") + _scalar("term_count_opponent_in_goal")
    opponent_wins = _scalar("term_count_opponent_scored") + _scalar("term_count_player_in_goal")
    draws = _scalar("term_count_time_expired")
    total_games = _scalar("num_games")
    if _scalar("outcome_schema_version") >= 2:
        # Raw termination counters can overlap; each game has one outcome.
        outcomes = data["outcome"]
        player_wins = int(np.isin(outcomes, [0, 3]).sum())
        opponent_wins = int(np.isin(outcomes, [1, 2]).sum())
        draws = int((outcomes == 4).sum())

    if "meta_player_checkpoint" in data.files:
        checkpoint = str(data["meta_player_checkpoint"])
    else:
        checkpoint = npz_path

    term_counts = {key: _scalar(key) for key, _, _, _ in TERM_SERIES}

    return {
        "checkpoint": checkpoint,
        "total_games": total_games,
        "player_wins": player_wins,
        "opponent_wins": opponent_wins,
        "draws": draws,
        "term_counts": term_counts,
        "npz_path": npz_path,
    }


def _format_aggregate_table(rows: list[dict], invocation_cmd: str | None = None) -> str:
    """Render the fixed-width aggregate table.

    Layout is a verbatim port of the inline block in evaluate_dreamer.py so the
    pre- and post-refactor outputs match byte-for-byte when invocation_cmd is
    supplied. The standalone CLI passes ``invocation_cmd=None``, which omits
    the ``Command:`` line.
    """
    lines = [
        "\n" + "=" * 70,
        "  AGGREGATE RESULTS ACROSS ALL CHECKPOINTS",
        "=" * 70,
    ]
    if invocation_cmd is not None:
        lines.append(f"  Command: {invocation_cmd}")
    lines += [
        "-" * 70,
        f"  {'Checkpoint':<45} {'Win%':>6}  {'W':>5}  {'L':>5}  {'D':>5}",
        "-" * 70,
    ]
    for r in rows:
        ckpt_name = os.path.basename(r["checkpoint"])
        wr = f"{r['player_wins'] / r['total_games'] * 100:.1f}%" if r["total_games"] > 0 else "N/A"
        lines.append(
            f"  {ckpt_name:<45} {wr:>6}  {r['player_wins']:>5}  {r['opponent_wins']:>5}  {r['draws']:>5}"
        )
    lines.append("=" * 70 + "\n")
    return "\n".join(lines)


def _draw_line_plot(
    rows: list[dict],
    series: list[tuple[list[int], str, str, str]],
    out_path: str,
    title: str,
) -> str:
    """Shared scaffolding for line plots indexed by checkpoint.

    ``series`` is a list of ``(y_values, label, color, marker)`` tuples.
    Imports matplotlib lazily so callers that only need the text table don't
    pay the import cost.
    """
    import matplotlib

    if not _is_interactive_backend():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    x = list(range(n))
    labels = [os.path.basename(r["checkpoint"]) for r in rows]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.4 * n + 4.0), 4.8))
    for y, label, color, marker in series:
        ax.plot(x, y, marker=marker, linewidth=1.6, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Games")
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="best", fontsize=10, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=12, weight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _plot_winrates(rows: list[dict], out_path: str, title_suffix: str | None = None) -> str:
    """Three-line plot: player wins / opponent wins / draws across checkpoints."""
    series = [
        ([r["player_wins"] for r in rows], "Player wins", COLOR_PLAYER, "o"),
        ([r["opponent_wins"] for r in rows], "Opponent wins", COLOR_OPPONENT, "s"),
        ([r["draws"] for r in rows], "Draws", COLOR_DRAW, "^"),
    ]
    title = "Aggregate win-rate across checkpoints"
    if title_suffix:
        title = f"{title}  ·  {title_suffix}"
    return _draw_line_plot(rows, series, out_path, title)


def _plot_term_counts(rows: list[dict], out_path: str, title_suffix: str | None = None) -> str:
    """Five-line plot: per-termination type counts across checkpoints.

    Series: player scored, opponent scored, player fell in goal,
    opponent fell in goal, time expired.
    """
    series = [
        ([r["term_counts"][key] for r in rows], label, color, marker)
        for key, label, color, marker in TERM_SERIES
    ]
    title = "Termination breakdown across checkpoints"
    if title_suffix:
        title = f"{title}  ·  {title_suffix}"
    return _draw_line_plot(rows, series, out_path, title)


def _is_interactive_backend() -> bool:
    import matplotlib

    backend = matplotlib.get_backend().lower()
    non_interactive = ("agg", "pdf", "ps", "svg", "cairo", "template")
    return not any(backend.startswith(b) for b in non_interactive)


def aggregate_from_npz_files(
    npz_paths: list[str],
    out_path: str | None = None,
    invocation_cmd: str | None = None,
    plot_path: str | None = None,
    term_plot_path: str | None = None,
) -> tuple[str, str | None, str | None, str | None]:
    """Build the aggregate win-rate table from a list of h2h_results_*.npz files.

    Args:
        npz_paths: Ordered list of ``.npz`` file paths. The output table rows
            follow the same order as the input list.
        out_path: When given, the rendered text is written to this path.
        invocation_cmd: Optional command string rendered as a ``Command:`` line
            in the table header (used by inline calls from eval scripts).
        plot_path: When given, render the win/loss/draw line plot to this PNG.
        term_plot_path: When given, render the per-termination breakdown line
            plot (5 series) to this PNG.

    Returns:
        ``(text, out_path, plot_path, term_plot_path)`` — the rendered table
        and the paths it was written to (``None`` for whichever side effect
        was skipped).
    """
    rows = [_load_h2h_row(p) for p in npz_paths]
    text = _format_aggregate_table(rows, invocation_cmd=invocation_cmd)
    if out_path is not None:
        with open(out_path, "w") as f:
            f.write(text)
    if plot_path is not None:
        _plot_winrates(rows, plot_path)
    if term_plot_path is not None:
        _plot_term_counts(rows, term_plot_path)
    return text, out_path, plot_path, term_plot_path


def aggregate_from_folder(
    folder: str,
    out_path: str | None = None,
    plot: bool = True,
    plot_path: str | None = None,
    term_plot_path: str | None = None,
) -> tuple[str, str | None, str | None, str | None]:
    """Discover h2h_results_*.npz files in folder and aggregate them.

    Args:
        folder: Directory containing one or more ``h2h_results_<ts>.npz``
            files (typically ``logs/<algo>/head_to_head_results/``).
        out_path: When given, override the default text output path.
            Default: ``<folder>/h2h_aggregate_<now>.txt``.
        plot: When True (default), also write the win/loss/draw line plot and
            the per-termination breakdown plot.
        plot_path: When given, override the default win-rate plot output path.
            Default: ``<folder>/h2h_aggregate_<now>.png``.
        term_plot_path: When given, override the default termination plot path.
            Default: ``<folder>/h2h_aggregate_<now>_terminations.png``.

    Returns:
        ``(text, out_path_written, plot_path_written, term_plot_path_written)``.

    Raises:
        FileNotFoundError: when no matching ``.npz`` files are found in folder.
    """
    npz_paths = sorted(glob.glob(os.path.join(folder, "h2h_results_*.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No h2h_results_*.npz files found in: {folder}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_path is None:
        out_path = os.path.join(folder, f"h2h_aggregate_{timestamp}.txt")
    if plot:
        if plot_path is None:
            plot_path = os.path.join(folder, f"h2h_aggregate_{timestamp}.png")
        if term_plot_path is None:
            term_plot_path = os.path.join(folder, f"h2h_aggregate_{timestamp}_terminations.png")
    else:
        plot_path = None
        term_plot_path = None

    return aggregate_from_npz_files(
        npz_paths, out_path=out_path, plot_path=plot_path, term_plot_path=term_plot_path,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-checkpoint head-to-head results from a folder of h2h_results_*.npz files.",
    )
    parser.add_argument("folder", type=str, help="Directory containing h2h_results_<ts>.npz files.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output text path (default: <folder>/h2h_aggregate_<now>.txt).",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help="Output PNG path for the win/loss/draw line plot (default: <folder>/h2h_aggregate_<now>.png).",
    )
    parser.add_argument(
        "--terminations-out",
        type=str,
        default=None,
        help=(
            "Output PNG path for the per-termination line plot "
            "(default: <folder>/h2h_aggregate_<now>_terminations.png)."
        ),
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generating both line plots.")
    args = parser.parse_args()

    text, out_path, plot_path, term_plot_path = aggregate_from_folder(
        args.folder,
        out_path=args.out,
        plot=not args.no_plot,
        plot_path=args.plot_out,
        term_plot_path=args.terminations_out,
    )
    print(text)
    if out_path:
        print(f"[INFO] Aggregate results saved to:  {out_path}")
    if plot_path:
        print(f"[INFO] Win-rate plot saved to:      {plot_path}")
    if term_plot_path:
        print(f"[INFO] Termination plot saved to:   {term_plot_path}")


if __name__ == "__main__":
    _main()
