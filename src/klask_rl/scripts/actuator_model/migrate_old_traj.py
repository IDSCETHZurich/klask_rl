from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IN_PATH = "/workspace/klask_rl/scripts/actuator_model/data/original/sim2real_trajectories_5.npz"
OUT_PATH = "/workspace/klask_rl/scripts/actuator_model/data/real_trajectories_05.npz"

OBS_LABELS = [
    "Player Pos X (m)",
    "Player Pos Y (m)",
    "Player Vel X (m/s)",
    "Player Vel Y (m/s)",
    "Opponent Pos X (m)",
    "Opponent Pos Y (m)",
    "Opponent Vel X (m/s)",
    "Opponent Vel Y (m/s)",
    "Ball Pos X (m)",
    "Ball Pos Y (m)",
    "Ball Vel X (m/s)",
    "Ball Vel Y (m/s)",
]


def plot_observations(obs, title, out_png):
    time_steps = range(obs.shape[0])
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(title, fontsize=16, weight="bold")

    for i, ax in enumerate(axes.flatten()):
        ax.plot(time_steps, obs[:, i], linewidth=1.5)
        ax.set_title(OBS_LABELS[i], fontsize=10)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel(OBS_LABELS[i].split("(")[-1].rstrip(")"))
        ax.grid(True, linestyle=":", alpha=0.6)

    backend = plt.get_backend().lower()
    if "agg" in backend:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Non-interactive backend ({plt.get_backend()}); saved figure to {out_png}")
    else:
        plt.show()
    plt.close(fig)


data = np.load(IN_PATH, allow_pickle=True)
obs = data["observations_real"]
actions = data["actions"]

# Plot original observations for verification
plot_observations(
    obs,
    title="Original Observations — sim2real_trajectories_5",
    out_png=Path(__file__).with_name("migrate_orig_obs.png"),
)

# Save migrated file
np.savez(OUT_PATH, actions=actions, obs=obs)
print(f"Saved to {OUT_PATH}")
