"""Actuator model analysis: frequency response (Bode) and step response.

Loads one or more trained ActuatorNetwork checkpoints and produces:
    1. Single-input Bode plots   (one figure per amplitude)
    2. Simultaneous-excitation Bode plots (one figure per amplitude)
    3. Step response plots       (one figure per amplitude)

Models are overlaid in the same figures for manual comparison.

Example usage:
    python analyse_model.py \
        --models model_old.pt model_new.pt \
        --labels "old data" "new data" \
        --amplitudes 0.1 0.3 0.5 \
        --output_dir results/
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from actuator_network import ActuatorNetwork

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _save_or_show(fig, name, output_dir, dpi):
    backend = plt.get_backend().lower()
    if "agg" in backend:
        output_path = output_dir / name
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def load_model(checkpoint_path, device):
    """Load an ActuatorNetwork from a .pt state-dict checkpoint.

    Model dimensions are inferred from the weight shapes so no
    hidden_dim argument is needed.
    """
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hidden_dim = state_dict["fc1.weight"].shape[0]
    input_dim = state_dict["fc1.weight"].shape[1]
    output_dim = state_dict["fc_out.weight"].shape[0]
    model = ActuatorNetwork(input_dim, output_dim, hidden_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded {checkpoint_path} | input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}")
    return model


def run_model_synthetic(model, commands_sequence, device):
    """Run *model* on one or more synthetic command sequences in parallel.

    Args:
        model: An ActuatorNetwork in eval mode.
        commands_sequence: numpy array of shape (T, 2) for a single simulation
            or (B, T, 2) for a batch of B independent simulations (all same length).
        device: torch device.

    Returns:
        numpy array of shape (T, 2) or (B, T, 2) matching the input shape.
    """
    single = commands_sequence.ndim == 2
    if single:
        commands_sequence = commands_sequence[np.newaxis]  # (1, T, 2)

    B, T, _ = commands_sequence.shape
    input_dim = next(model.parameters()).shape[1]

    # Determine whether the model uses state history.
    # num_history = h, cmd = 2h, state = 2(h-1), total = 4h-2
    if input_dim > 20:
        h = (input_dim + 2) // 4
        num_cmd_entries = h * 2
        num_state_entries = (h - 1) * 2
    else:
        h = input_dim // 2
        num_cmd_entries = h * 2
        num_state_entries = 0

    command_buffer = torch.zeros(B, num_cmd_entries, device=device)
    if num_state_entries > 0:
        state_buffer = torch.zeros(B, num_state_entries, device=device)
    else:
        state_buffer = None

    cmds_tensor = torch.as_tensor(commands_sequence, dtype=torch.float32, device=device)
    outputs = torch.empty(B, T, 2, device=device)

    with torch.no_grad():
        for t in range(T):
            command_buffer[:, 2:] = command_buffer[:, :-2].clone()
            command_buffer[:, :2] = cmds_tensor[:, t]

            pred = model(command_buffer, state_buffer)

            if state_buffer is not None:
                state_buffer[:, 2:] = state_buffer[:, :-2].clone()
                state_buffer[:, :2] = pred

            outputs[:, t] = pred

    result = outputs.cpu().numpy()
    return result[0] if single else result


# ---------------------------------------------------------------------------
# Lock-in detection
# ---------------------------------------------------------------------------


def _lockin(signal, freq, dt, discard_steps, num_analysis_cycles):
    """Extract amplitude and phase of *signal* at *freq* via lock-in detection.

    Args:
        signal: 1-D numpy array (full simulation output for one component).
        freq: excitation frequency in Hz.
        dt: timestep in seconds.
        discard_steps: number of leading samples to skip (transient).
        num_analysis_cycles: number of complete cycles to use for analysis.

    Returns:
        (amplitude, phase_rad)
    """
    samples_per_cycle = 1.0 / (freq * dt)
    n_samples = int(round(num_analysis_cycles * samples_per_cycle))
    sig = signal[discard_steps : discard_steps + n_samples]
    # Truncate to actual available length (rounding may cause off-by-one).
    n_samples = len(sig)
    t = np.arange(n_samples) * dt
    cos_ref = np.cos(2 * np.pi * freq * t)
    sin_ref = np.sin(2 * np.pi * freq * t)
    # For input sin(ωt): sin-correlation gives in-phase, cos-correlation gives quadrature.
    in_phase = (2.0 / n_samples) * np.sum(sig * sin_ref)
    quadrature = (2.0 / n_samples) * np.sum(sig * cos_ref)
    amplitude = np.sqrt(in_phase**2 + quadrature**2)
    # Phase = atan2(quadrature, in_phase) gives lag as negative.
    phase = np.arctan2(quadrature, in_phase)
    return amplitude, phase


# ---------------------------------------------------------------------------
# Frequency response — single-input sweeps
# ---------------------------------------------------------------------------


def compute_frequency_response_single(model, freqs, dt, amplitudes, num_cycles, discard_cycles, device, label=""):
    """Sweep sinusoids on one axis at a time and extract all 4 transfer functions.

    All amplitudes are batched together in a single GPU pass per frequency.

    Args:
        amplitudes: list/array of excitation amplitudes.

    Returns:
        dict[amplitude] → dict[tf_name] → {'gain_dB': array, 'phase_deg': array}
    """
    n_analysis = num_cycles - discard_cycles
    tf_names = ["vx_ux", "vx_uy", "vy_ux", "vy_uy"]
    n_amps = len(amplitudes)

    # Pre-allocate results per amplitude.
    results = {
        amp: {name: {"gain_dB": np.empty(len(freqs)), "phase_deg": np.empty(len(freqs))} for name in tf_names}
        for amp in amplitudes
    }

    for i, f in enumerate(tqdm(freqs, desc=f"  Bode single ({label})", leave=False)):
        n_steps = int(round(num_cycles / (f * dt)))
        t = np.arange(n_steps) * dt
        discard_steps = int(round(discard_cycles / (f * dt)))
        sin_wave = np.sin(2 * np.pi * f * t)

        # Batch: 2 experiments (v_x, v_y) × n_amps amplitudes.
        cmds = np.zeros((2 * n_amps, n_steps, 2), dtype=np.float32)
        for a_idx, amp in enumerate(amplitudes):
            cmds[2 * a_idx, :, 0] = amp * sin_wave       # v_x excitation
            cmds[2 * a_idx + 1, :, 1] = amp * sin_wave   # v_y excitation
        out = run_model_synthetic(model, cmds, device)  # (2*n_amps, n_steps, 2)

        for a_idx, amp in enumerate(amplitudes):
            out_vx = out[2 * a_idx]      # v_x excitation output
            out_vy = out[2 * a_idx + 1]  # v_y excitation output

            for out_idx, tf in zip([0, 1], ["vx_ux", "vx_uy"]):
                amp_out, phase_out = _lockin(out_vx[:, out_idx], f, dt, discard_steps, n_analysis)
                results[amp][tf]["gain_dB"][i] = 20.0 * np.log10(amp_out / amp + 1e-12)
                results[amp][tf]["phase_deg"][i] = np.degrees(phase_out)

            for out_idx, tf in zip([0, 1], ["vy_ux", "vy_uy"]):
                amp_out, phase_out = _lockin(out_vy[:, out_idx], f, dt, discard_steps, n_analysis)
                results[amp][tf]["gain_dB"][i] = 20.0 * np.log10(amp_out / amp + 1e-12)
                results[amp][tf]["phase_deg"][i] = np.degrees(phase_out)

    # Unwrap phases.
    for amp in amplitudes:
        for tf in tf_names:
            results[amp][tf]["phase_deg"] = np.degrees(np.unwrap(np.radians(results[amp][tf]["phase_deg"])))

    return results


# ---------------------------------------------------------------------------
# Frequency response — simultaneous (quadrature) excitation
# ---------------------------------------------------------------------------


def compute_frequency_response_simultaneous(model, freqs, dt, amplitudes, num_cycles, discard_cycles, device, label=""):
    """Excite both axes simultaneously in quadrature and measure combined outputs.

    v_x_in = A·sin(2πft),  v_y_in = A·cos(2πft)

    All amplitudes are batched together in a single GPU pass per frequency.

    Returns:
        dict[amplitude] → dict[out_name] → {'gain_dB': array, 'phase_deg': array}
    """
    n_analysis = num_cycles - discard_cycles
    out_names = ["vx_out", "vy_out"]
    n_amps = len(amplitudes)

    results = {
        amp: {name: {"gain_dB": np.empty(len(freqs)), "phase_deg": np.empty(len(freqs))} for name in out_names}
        for amp in amplitudes
    }

    for i, f in enumerate(tqdm(freqs, desc=f"  Bode simul ({label})", leave=False)):
        n_steps = int(round(num_cycles / (f * dt)))
        t = np.arange(n_steps) * dt
        discard_steps = int(round(discard_cycles / (f * dt)))
        sin_wave = np.sin(2 * np.pi * f * t)
        cos_wave = np.cos(2 * np.pi * f * t)

        # Batch: 1 experiment × n_amps amplitudes.
        cmds = np.zeros((n_amps, n_steps, 2), dtype=np.float32)
        for a_idx, amp in enumerate(amplitudes):
            cmds[a_idx, :, 0] = amp * sin_wave
            cmds[a_idx, :, 1] = amp * cos_wave
        out = run_model_synthetic(model, cmds, device)  # (n_amps, n_steps, 2)

        for a_idx, amp in enumerate(amplitudes):
            for out_idx, name in enumerate(out_names):
                amp_out, phase_out = _lockin(out[a_idx, :, out_idx], f, dt, discard_steps, n_analysis)
                results[amp][name]["gain_dB"][i] = 20.0 * np.log10(amp_out / amp + 1e-12)
                results[amp][name]["phase_deg"][i] = np.degrees(phase_out)

    for amp in amplitudes:
        for name in out_names:
            results[amp][name]["phase_deg"] = np.degrees(np.unwrap(np.radians(results[amp][name]["phase_deg"])))

    return results


_SIMUL_LABELS = {
    "vx_out": r"$v_{x,out}$",
    "vy_out": r"$v_{y,out}$",
}
_SIMUL_ORDER = ["vx_out", "vy_out"]


def plot_bode_simultaneous(results_per_model, labels, freqs, title, filename, output_dir, dpi):
    """Plot Bode for simultaneous excitation: 2 rows (mag/phase) × 2 cols (v_x_out, v_y_out)."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(title, fontsize=16, weight="bold")

    for col, name in enumerate(_SIMUL_ORDER):
        ax_mag = axes[0, col]
        ax_phase = axes[1, col]

        for idx, (res, label) in enumerate(zip(results_per_model, labels)):
            color = colors[idx % len(colors)]
            ax_mag.semilogx(freqs, res[name]["gain_dB"], color=color, label=label, linewidth=1.5)
            ax_phase.semilogx(freqs, res[name]["phase_deg"], color=color, label=label, linewidth=1.5)

        ax_mag.set_title(_SIMUL_LABELS[name])
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_mag.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_phase.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        if col == 0:
            ax_mag.legend(fontsize=9)

    _save_or_show(fig, filename, output_dir, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step response
# ---------------------------------------------------------------------------


def compute_step_response(model, amplitudes, dt, num_steps, device):
    """Compute step responses for three excitation cases, batched across amplitudes.

    Returns:
        dict[amplitude] → dict with keys 'step_vx', 'step_vy', 'step_vxy',
        each an (num_steps, 2) array.
    """
    n_amps = len(amplitudes)
    # Batch: 3 experiments × n_amps amplitudes.
    cmds = np.zeros((3 * n_amps, num_steps, 2), dtype=np.float32)
    for a_idx, amp in enumerate(amplitudes):
        cmds[3 * a_idx, :, 0] = amp          # v_x only
        cmds[3 * a_idx + 1, :, 1] = amp      # v_y only
        cmds[3 * a_idx + 2, :, :] = amp      # v_x + v_y
    out = run_model_synthetic(model, cmds, device)  # (3*n_amps, num_steps, 2)

    return {
        amp: {
            "step_vx": out[3 * a_idx],
            "step_vy": out[3 * a_idx + 1],
            "step_vxy": out[3 * a_idx + 2],
        }
        for a_idx, amp in enumerate(amplitudes)
    }


# ---------------------------------------------------------------------------
# Plotting — Bode
# ---------------------------------------------------------------------------

_TF_LABELS = {
    "vx_ux": r"$v_{x,in} \rightarrow v_{x,out}$",
    "vx_uy": r"$v_{x,in} \rightarrow v_{y,out}$",
    "vy_ux": r"$v_{y,in} \rightarrow v_{x,out}$",
    "vy_uy": r"$v_{y,in} \rightarrow v_{y,out}$",
}
_TF_GRID = [
    # row 0: v_x excitation,  row 1: v_y excitation
    # col 0: v_x out,         col 1: v_y out
    ["vx_ux", "vx_uy"],
    ["vy_ux", "vy_uy"],
]


def plot_bode(results_per_model, labels, freqs, title, filename, output_dir, dpi):
    """Plot Bode diagram as a 2×2 transfer function grid, each cell with magnitude + phase.

    Layout:
        rows = excitation (v_x_in, v_y_in)
        cols = output (v_x_out, v_y_out)
        each cell = magnitude on top, phase on bottom (shared x-axis)
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    fig.suptitle(title, fontsize=16, weight="bold")

    for grid_row in range(2):
        for grid_col in range(2):
            tf = _TF_GRID[grid_row][grid_col]
            ax_mag = axes[grid_row * 2, grid_col]
            ax_phase = axes[grid_row * 2 + 1, grid_col]

            for idx, (res, label) in enumerate(zip(results_per_model, labels)):
                color = colors[idx % len(colors)]
                ax_mag.semilogx(freqs, res[tf]["gain_dB"], color=color, label=label, linewidth=1.5)
                ax_phase.semilogx(freqs, res[tf]["phase_deg"], color=color, label=label, linewidth=1.5)

            ax_mag.set_title(_TF_LABELS[tf])
            ax_mag.set_ylabel("Magnitude (dB)")
            ax_mag.grid(True, which="both", linestyle=":", alpha=0.6)
            ax_mag.axhline(0, color="gray", linewidth=0.8, linestyle="--")

            ax_phase.set_ylabel("Phase (deg)")
            ax_phase.set_xlabel("Frequency (Hz)")
            ax_phase.grid(True, which="both", linestyle=":", alpha=0.6)
            ax_phase.axhline(0, color="gray", linewidth=0.8, linestyle="--")

            if grid_row == 0 and grid_col == 0:
                ax_mag.legend(fontsize=9)

    _save_or_show(fig, filename, output_dir, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting — Step response
# ---------------------------------------------------------------------------

_STEP_ROWS = [
    ("step_vx", "Step $v_{x,in}$ only"),
    ("step_vy", "Step $v_{y,in}$ only"),
    ("step_vxy", "Step $v_{x,in} + v_{y,in}$"),
]


def plot_step_response(results_per_model, labels, dt, amplitude, filename, output_dir, dpi):
    """Plot 3x2 step response grid.

    Args:
        results_per_model: list of dicts (one per model), each from compute_step_response.
        labels: list of string labels.
        dt: timestep.
        amplitude: command amplitude (for reference line).
        filename: output filename.
        output_dir: Path.
        dpi: int.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(f"Step Response (amplitude = {amplitude} m/s)", fontsize=16, weight="bold")

    for row, (key, row_title) in enumerate(_STEP_ROWS):
        for col, out_label in enumerate(["$v_{x,out}$", "$v_{y,out}$"]):
            ax = axes[row, col]
            for idx, (res, label) in enumerate(zip(results_per_model, labels)):
                data = res[key]
                t = np.arange(data.shape[0]) * dt
                color = colors[idx % len(colors)]
                ax.plot(t, data[:, col], color=color, label=label, linewidth=1.5)

            ax.axhline(amplitude, color="gray", linewidth=1, linestyle="--", alpha=0.7)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="-", alpha=0.4)
            ax.set_title(f"{row_title} → {out_label}")
            ax.set_ylabel("Velocity (m/s)")
            ax.grid(True, linestyle=":", alpha=0.6)

            if row == 0 and col == 0:
                ax.legend(fontsize=9)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    _save_or_show(fig, filename, output_dir, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyse actuator model(s): Bode plots and step response.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", required=True, type=Path, help="One or more .pt checkpoint paths.")
    parser.add_argument("--labels", nargs="+", required=True, help="One label per model (for plot legends).")
    # Frequency response.
    parser.add_argument("--freq_min", type=float, default=0.1, help="Min sweep frequency (Hz).")
    parser.add_argument("--freq_max", type=float, default=20.0, help="Max sweep frequency (Hz).")
    parser.add_argument("--freq_points", type=int, default=100, help="Number of log-spaced frequency points.")
    parser.add_argument("--num_cycles", type=int, default=50, help="Sinusoidal cycles per frequency.")
    parser.add_argument("--discard_cycles", type=int, default=10, help="Transient cycles to discard.")
    # Step response.
    parser.add_argument("--step_duration", type=float, default=2.0, help="Step response duration (seconds).")
    # Shared.
    parser.add_argument(
        "--amplitudes",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
        help="Excitation amplitudes (m/s).",
    )
    parser.add_argument("--dt", type=float, default=0.02, help="Timestep (seconds).")
    parser.add_argument(
        "--output_dir", type=Path, default=Path(__file__).parent / "evaluation", help="Directory for saved figures."
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")

    args = parser.parse_args()

    if len(args.models) != len(args.labels):
        parser.error("Number of --models must match number of --labels.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models.
    models = [load_model(p, device) for p in args.models]

    freqs = np.logspace(np.log10(args.freq_min), np.log10(args.freq_max), args.freq_points)
    num_steps = int(args.step_duration / args.dt)

    amplitudes = args.amplitudes

    # --- Frequency response: single-input ---
    # Each call batches all amplitudes; returns dict[amp] → dict[tf] → {...}
    print("\nComputing single-input frequency response...")
    fr_single_per_model = []
    for model, label in zip(models, args.labels):
        fr_single_per_model.append(compute_frequency_response_single(
            model, freqs, args.dt, amplitudes,
            args.num_cycles, args.discard_cycles, device, label=label,
        ))

    # --- Frequency response: simultaneous ---
    print("\nComputing simultaneous (quadrature) frequency response...")
    fr_sim_per_model = []
    for model, label in zip(models, args.labels):
        fr_sim_per_model.append(compute_frequency_response_simultaneous(
            model, freqs, args.dt, amplitudes,
            args.num_cycles, args.discard_cycles, device, label=label,
        ))

    # --- Step response ---
    print("\nComputing step response...")
    step_per_model = []
    for model, label in zip(models, args.labels):
        print(f"  Step response ({label})")
        step_per_model.append(compute_step_response(model, amplitudes, args.dt, num_steps, device))

    # --- Plot per amplitude ---
    for amp in amplitudes:
        print(f"\nPlotting amplitude = {amp} m/s")
        amp_tag = f"{amp:.2f}".replace(".", "_")

        plot_bode(
            [r[amp] for r in fr_single_per_model],
            args.labels, freqs,
            title=f"Frequency Response — Single Input (A = {amp} m/s)",
            filename=f"analyse_model_01_bode_single_A{amp_tag}.png",
            output_dir=args.output_dir, dpi=args.dpi,
        )
        plot_bode_simultaneous(
            [r[amp] for r in fr_sim_per_model],
            args.labels, freqs,
            title=f"Frequency Response — Simultaneous Quadrature (A = {amp} m/s)",
            filename=f"analyse_model_02_bode_simultaneous_A{amp_tag}.png",
            output_dir=args.output_dir, dpi=args.dpi,
        )
        plot_step_response(
            [r[amp] for r in step_per_model],
            args.labels, args.dt, amp,
            filename=f"analyse_model_03_step_A{amp_tag}.png",
            output_dir=args.output_dir, dpi=args.dpi,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
