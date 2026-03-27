from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def weight_decay(init_weight, step, decay_rate):
    return init_weight * np.exp(-decay_rate * step)


def sigmoid_weight(init_weight, step, rate, midpoint):
    logits = np.clip(-rate * (step - midpoint), -60.0, 60.0)
    return init_weight / (1.0 + np.exp(logits))


if __name__ == "__main__":
    init_weight = 1.0
    max_steps = 10_000_000
    steps = np.arange(1, max_steps, 1000).reshape(-1, 1)
    decay_rates = np.logspace(-7, -4, num=8).reshape(1, -1)
    midpoint = max_steps / 2.0

    plot_specs = [
        {
            "title": "Exponential Decay of Weights",
            "label_prefix": "decay_rate",
            "func": lambda rate: weight_decay(init_weight, steps, rate),
        },
        {
            "title": "Sigmoid Scaling of Weights",
            "label_prefix": "sigmoid_rate",
            "func": lambda rate: sigmoid_weight(init_weight, steps, rate, midpoint),
        },
    ]

    fig, axes = plt.subplots(1, len(plot_specs), figsize=(7 * len(plot_specs), 5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, spec in zip(axes, plot_specs):
        for rate in decay_rates.ravel():
            weights = spec["func"](rate)
            ax.plot(steps, weights.ravel(), label=f"{spec['label_prefix']}={rate:.1e}")

        ax.set_xlabel("Steps")
        ax.set_ylabel("Weight")
        ax.set_title(spec["title"])
        ax.legend()
        ax.grid()

    fig.tight_layout()

    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = Path(__file__).with_name("rewards_decay.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Non-interactive backend ({plt.get_backend()}); saved figure to {output_path}")
    else:
        plt.show()
