"""Box-plot generation from h2h_results_<ts>.npz files.

Pure numpy + matplotlib — no Isaac Lab imports. Lives in ``scripts/`` (not
inside the ``klask_rl`` package) on purpose: importing the package eagerly
loads ``klask_rl.tasks`` which pulls in IsaacLab/Omniverse, defeating the
"runnable without Isaac Sim" goal of this module.

Visual style mirrors ``scripts/actuator_model/sim2real_seed_analysis.py``:
single colour (green) box, ``alpha=0.35`` fill, black median, matching
whisker/cap colour, ``showfliers=False``, dotted y-grid.

One chart per metric category — all boxes inside a chart share the same
y-axis. The categories are:
    * Ball contacts        (6 boxes: player/opponent × {all, player-scored, opponent-scored})
    * Ball speed           (2 boxes: max, mean)
    * Half occupancy       (2 boxes: player half, opponent half)
    * Game time            (3 boxes: all, until player scored, until player conceded)
    * Outcome distribution (bar chart of the 5 termination types)

Stacked vertically into a single PNG saved next to the npz.

Usage:
    # As a standalone CLI (no Isaac Sim required):
    python scripts/plot_eval_boxplots.py path/to/h2h_results_xxx.npz

    # From a script that already has scripts/ on sys.path:
    from eval_plot import plot_boxplots
    plot_boxplots("path/to/h2h_results_xxx.npz")

    # From the eval scripts (play_klask.py, evaluate_dreamer.py) we load it
    # lazily via importlib.util so we don't depend on sys.path tweaks.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

import numpy as np


COLOR = "green"


def _is_interactive_backend() -> bool:
    backend = matplotlib.get_backend().lower()
    non_interactive = ("agg", "pdf", "ps", "svg", "cairo", "template")
    return not any(backend.startswith(b) for b in non_interactive)


def _outcome_index(outcome_legend: np.ndarray, name: str) -> int | None:
    for i, n in enumerate(outcome_legend):
        if str(n) == name:
            return int(i)
    return None


def _read_meta(npz_data) -> dict[str, str]:
    meta: dict[str, str] = {}
    for key in npz_data.files:
        if key.startswith("meta_"):
            meta[key[len("meta_") :]] = str(npz_data[key])
    return meta


def _format_matchup(meta: dict[str, str]) -> str | None:
    """Two-line matchup block: ``Player: <path>`` then ``Opponent: <path>``."""
    player = meta.get("player_checkpoint", "")
    opponent = meta.get("opponent_checkpoint", "")
    if not player and not opponent:
        return None
    return f"Player: {player}\nOpponent: {opponent}"


def _box(ax, data, labels, ylabel="", title=""):
    """Draw a single chart with one or more boxes sharing the same y-axis.

    Style mirrors ``_draw_velocity_rmse_boxplot`` from
    ``sim2real_seed_analysis.py``: green fill at ``alpha=0.35``, matching
    edge / whisker / cap colour, black median, no fliers.
    Empty groups render as "n=0" placeholders so the layout stays consistent.
    """
    n_boxes = len(data)
    positions = list(range(n_boxes))
    non_empty_idx = [i for i, d in enumerate(data) if d is not None and len(d) > 0]

    if non_empty_idx:
        plot_data = [data[i] for i in non_empty_idx]
        plot_positions = [positions[i] for i in non_empty_idx]
        bp = ax.boxplot(
            plot_data,
            positions=plot_positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": COLOR, "linewidth": 1.1},
            capprops={"color": COLOR, "linewidth": 1.1},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLOR)
            patch.set_alpha(0.35)
            patch.set_edgecolor(COLOR)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, n_boxes - 0.4)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate sample size below each box.
    y_min, y_max = ax.get_ylim()
    y_text = y_min - (y_max - y_min) * 0.08
    for i, d in enumerate(data):
        n = 0 if d is None else len(d)
        ax.text(positions[i], y_text, f"n={n}", ha="center", va="top", fontsize=8.5, color="gray")


def _bar(ax, labels, counts, ylabel="games", title=""):
    """Bar chart in the same green style."""
    bars = ax.bar(
        range(len(labels)), counts,
        color=COLOR, edgecolor=COLOR, alpha=0.35, width=0.55,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ymax = max(counts) if counts else 0
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ymax * 0.02 if ymax > 0 else 0.1),
            str(c), ha="center", va="bottom", fontsize=9.5, color="black",
        )
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.15)


def plot_boxplots(
    npz_path: str,
    out_path: str | None = None,
    show: bool = False,
    title_suffix: str | None = None,
) -> str:
    """Generate a 5-row figure, one chart per metric category.

    Args:
        npz_path: Path to ``h2h_results_<ts>.npz`` written by ``EvalMetricsTracker.save_npz``.
        out_path: Output PNG path. Defaults to ``<npz path>_boxplots.png``.
        show: If True and the matplotlib backend is interactive, display the figure.
        title_suffix: Optional string appended to the global suptitle.

    Returns:
        Absolute path of the saved PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    npz_path = os.path.abspath(npz_path)
    if out_path is None:
        out_path = npz_path[:-4] + "_boxplots.png" if npz_path.endswith(".npz") else npz_path + "_boxplots.png"
    out_path = os.path.abspath(out_path)

    data = np.load(npz_path, allow_pickle=False)

    player_contacts = np.asarray(data["player_contacts"])
    opponent_contacts = np.asarray(data["opponent_contacts"])
    max_ball_speed = np.asarray(data["max_ball_speed"])
    mean_ball_speed = np.asarray(data["mean_ball_speed"])
    frac_player_half = np.asarray(data["frac_player_half"])
    frac_opponent_half = np.asarray(data["frac_opponent_half"])
    game_length_s = np.asarray(data["game_length_s"])
    outcome = np.asarray(data["outcome"])
    outcome_legend = np.asarray(data["outcome_legend"])
    num_games = int(data["num_games"]) if "num_games" in data.files else len(outcome)

    scored_idx = _outcome_index(outcome_legend, "goal_scored")
    conceded_idx = _outcome_index(outcome_legend, "goal_conceded")
    scored_mask = outcome == scored_idx if scored_idx is not None else np.zeros_like(outcome, dtype=bool)
    conceded_mask = outcome == conceded_idx if conceded_idx is not None else np.zeros_like(outcome, dtype=bool)

    matchup = _format_matchup(_read_meta(data))

    fig, axes = plt.subplots(5, 1, figsize=(14, 22))

    # ---------- 1. ball contacts (6 boxes) ----------
    _box(
        axes[0],
        data=[
            player_contacts,
            player_contacts[scored_mask],
            player_contacts[conceded_mask],
            opponent_contacts,
            opponent_contacts[scored_mask],
            opponent_contacts[conceded_mask],
        ],
        labels=[
            "player\nall games",
            "player\n| player scored",
            "player\n| opponent scored",
            "opponent\nall games",
            "opponent\n| player scored",
            "opponent\n| opponent scored",
        ],
        ylabel="contacts / game",
        title="Ball contacts",
    )

    # ---------- 2. ball speed (2 boxes) ----------
    _box(
        axes[1],
        data=[max_ball_speed, mean_ball_speed],
        labels=["max", "mean"],
        ylabel="m/s",
        title="Ball speed",
    )

    # ---------- 3. half occupancy (2 boxes) ----------
    _box(
        axes[2],
        data=[frac_player_half, frac_opponent_half],
        labels=["player half", "opponent half"],
        ylabel="fraction of game",
        title="Ball half occupancy",
    )

    # ---------- 4. game time (3 boxes) ----------
    _box(
        axes[3],
        data=[
            game_length_s,
            game_length_s[scored_mask],
            game_length_s[conceded_mask],
        ],
        labels=[
            "all games",
            "until player scored",
            "until player conceded",
        ],
        ylabel="seconds",
        title="Game time",
    )

    # ---------- 5. outcome distribution (bar chart) ----------
    bar_labels = [str(n) for n in outcome_legend]
    counts = [int((outcome == i).sum()) for i in range(len(outcome_legend))]
    _bar(axes[4], bar_labels, counts, ylabel="games", title="Outcome distribution")

    # ---------- layout, global suptitle, matchup, legend ----------
    # Reserve top space manually for the suptitle + multi-line matchup so
    # they don't overlap with the first chart's title.
    top = 0.91 if matchup else 0.95
    fig.subplots_adjust(top=top, bottom=0.04, left=0.08, right=0.97, hspace=0.55)

    suffix_str = f"  ·  {title_suffix}" if title_suffix else ""
    title_line = f"Klask eval metrics  ·  {num_games} games{suffix_str}"
    fig.suptitle(title_line, fontsize=11, weight="normal", color="#0f172a", x=0.05, ha="left", y=0.985)

    if matchup:
        fig.text(
            0.05, 0.965, matchup,
            ha="left", va="top",
            fontsize=8.5, family="monospace", color="#334155", linespacing=1.5,
        )

    legend_handles = [Patch(facecolor=COLOR, edgecolor=COLOR, alpha=0.35, label="per-game distribution")]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.97, 0.985), frameon=True, fontsize=10)

    fig.savefig(out_path, dpi=130)

    if show and _is_interactive_backend():
        plt.show()
    else:
        plt.close(fig)

    return out_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-category box-plot figure from an eval-metrics npz.")
    parser.add_argument("--npz", type=str, required=True, help="Path to h2h_results_<ts>.npz")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (default: <npz>_boxplots.png)")
    parser.add_argument("--show", action="store_true", help="Display the figure if backend is interactive.")
    parser.add_argument("--title-suffix", type=str, default=None, help="Optional suffix appended to suptitle.")
    args = parser.parse_args()
    saved = plot_boxplots(args.npz, out_path=args.out, show=args.show, title_suffix=args.title_suffix)
    print(f"[INFO] Box-plot saved to: {saved}")


if __name__ == "__main__":
    _main()
