"""CLI wrapper to generate box-plots from an eval-metrics npz file.

Does not launch Isaac Sim — runnable with vanilla Python:
    python scripts/plot_eval_boxplots.py path/to/h2h_results_<ts>.npz
"""

import argparse

from eval_plot import plot_boxplots


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot box-plots from an h2h_results_<ts>.npz file.")
    parser.add_argument("npz", type=str, help="Path to h2h_results_<ts>.npz")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (default: <npz>_boxplots.png)")
    parser.add_argument("--show", action="store_true", help="Display the figure if backend is interactive.")
    parser.add_argument("--title-suffix", type=str, default=None, help="Optional suffix appended to suptitle.")
    args = parser.parse_args()
    saved = plot_boxplots(args.npz, out_path=args.out, show=args.show, title_suffix=args.title_suffix)
    print(f"[INFO] Box-plot saved to: {saved}")


if __name__ == "__main__":
    main()
