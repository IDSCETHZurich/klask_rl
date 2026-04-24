import argparse
import multiprocessing as mp
import random
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from queue import Empty

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from actuator_network import ActuatorNetwork
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ActuatorDataset(Dataset):

    def __init__(self, X_commands, X_states, Y, Y_prev, commands):
        super().__init__()

        self.X_commands = X_commands
        self.X_states = X_states
        self.Y = Y
        self.Y_prev = Y_prev
        self.commands = commands

    def __len__(self):
        return self.X_commands.shape[0]

    def __getitem__(self, index):
        x_states = -1 if self.X_states is None else self.X_states[index]
        return self.X_commands[index], x_states, self.Y[index], self.Y_prev[index], self.commands[index]


def smoothness_loss(preds, batch_y_prev):
    """Penalize rapid changes in the predictions over time."""
    diff = torch.cat([(preds[:, 0] - batch_y_prev[:, 0]).unsqueeze(1), preds[:, 1:] - preds[:, :-1]], dim=1)
    return torch.mean(torch.abs(diff))


def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_seed_training(
    seed,
    data_path,
    run_name,
    n_splits,
    epochs,
    lr,
    batch_size,
    smoothness_weight,
    hidden_dim,
    verbose,
    detect_anomaly,
    checkpoints_dir,
    log_dir,
    progress_queue,
):
    set_seed_everywhere(seed)
    data = np.load(data_path)
    X_commands, Y, Y_prev, commands = data["X_commands"], data["Y"], data["Y_prev"], data["commands"]
    X_states = data["X_states"] if "X_states" in data.keys() else None

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed_{seed}.log"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file, redirect_stdout(log_file), redirect_stderr(
        log_file
    ):
        best_state, cv_losses, best_fold_loss = train_model_with_cv(
            X_commands,
            X_states,
            Y,
            Y_prev,
            commands,
            n_splits=n_splits,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            smoothness_weight=smoothness_weight,
            hidden_dim=hidden_dim,
            verbose=verbose,
            detect_anomaly=detect_anomaly,
            progress_queue=progress_queue,
            seed=seed,
        )

    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoints_dir / f"model_{run_name}_seed{seed}.pt"
    torch.save(best_state, model_path)

    return {
        "seed": seed,
        "model_path": str(model_path),
        "log_path": str(log_path),
        "best_fold_loss": float(best_fold_loss),
        "avg_cv_loss": float(np.mean(cv_losses)),
    }


def train_model_with_cv(
    X_commands,
    X_states,
    Y,
    Y_prev,
    commands,
    n_splits=5,
    epochs=20,
    lr=1e-3,
    batch_size=512,
    smoothness_weight=0.1,
    hidden_dim=32,
    verbose=False,
    device=None,
    num_workers=0,
    detect_anomaly=False,
    progress_queue=None,
    seed=None,
):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_cuda = device.type == "cuda"
    if verbose:
        print(f"Using device: {device}")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_fold = 1

    fold_val_losses = []
    best_fold_state = None
    best_fold_loss = float("inf")

    # Ensure the data is float32
    X_commands = X_commands.astype(np.float32)
    if X_states is not None:
        X_states = X_states.astype(np.float32)
    Y = Y.astype(np.float32)
    Y_prev = Y_prev.astype(np.float32)
    commands = commands.astype(np.float32)
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    for train_index, val_index in tscv.split(X_commands):
        if use_cuda:
            torch.cuda.empty_cache()
        if verbose:
            print(f"\nTime Series CV Fold {cv_fold}:")
        X_commands_train, X_commands_val = X_commands[train_index], X_commands[val_index]
        if X_states is not None:
            X_states_train, X_states_val = X_states[train_index], X_states[val_index]
        else:
            X_states_train, X_states_val = None, None
        Y_train, Y_val = Y[train_index], Y[val_index]
        Y_prev_train, Y_prev_val = Y_prev[train_index], Y_prev[val_index]
        commands_train, commands_val = commands[train_index], commands[val_index]

        train_dataset = ActuatorDataset(
            torch.from_numpy(X_commands_train),
            torch.from_numpy(X_states_train) if X_states_train is not None else None,
            torch.from_numpy(Y_train),
            torch.from_numpy(Y_prev_train),
            torch.from_numpy(commands_train),
        )
        val_dataset = ActuatorDataset(
            torch.from_numpy(X_commands_val),
            torch.from_numpy(X_states_val) if X_states_val is not None else None,
            torch.from_numpy(Y_val),
            torch.from_numpy(Y_prev_val),
            torch.from_numpy(commands_val),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda
        )

        # Instantiate a new model using the input dimension from the data:
        input_shape = X_commands.shape[1] if X_states is None else X_commands.shape[1] + X_states.shape[1]
        model = ActuatorNetwork(input_shape, Y.shape[-1], hidden_dim=hidden_dim).to(device)
        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train for this fold
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x_commands, batch_x_states, batch_y, batch_y_prev, batch_commands in train_loader:
                batch_x_commands = batch_x_commands.to(device, non_blocking=use_cuda)
                batch_y = batch_y.to(device, non_blocking=use_cuda)
                batch_y_prev = batch_y_prev.to(device, non_blocking=use_cuda)
                batch_commands = batch_commands.to(device, non_blocking=use_cuda)
                if len(batch_x_states.shape) == 2:
                    batch_x_states = batch_x_states.to(device, non_blocking=use_cuda)
                else:
                    batch_x_states = None
                optimizer.zero_grad()
                preds_all = torch.empty_like(batch_y)
                for t in range(batch_y.shape[1]):
                    # adjust history window:
                    if t > 0:
                        # use command sequence from dataset:
                        batch_x_commands[:, 2:] = batch_x_commands.clone()[:, :-2]
                        batch_x_commands[:, :2] = batch_commands[:, t]
                        # use states autoregressively:
                        if batch_x_states is not None and batch_x_states[0] is not None:

                            batch_x_states[:, 2:] = batch_x_states.clone()[:, :-2]
                            batch_x_states[:, :2] = preds

                    # make prediction:
                    preds = model(batch_x_commands, batch_x_states)

                    preds_all[:, t] = preds

                loss = mse_loss(preds_all, batch_y) + smoothness_weight * smoothness_loss(preds_all, batch_y_prev)
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.item() * batch_x_commands.size(0)
            epoch_loss /= len(train_dataset)
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}", flush=True)
            if progress_queue is not None and seed is not None:
                progress_queue.put({
                    "seed": seed,
                    "phase": "cv",
                    "fold": cv_fold,
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "loss": float(epoch_loss),
                })

        # Validate
        model.eval()
        val_loss = 0.0
        val_loss_mse = 0.0
        val_loss_smoothness = 0.0
        with torch.no_grad():
            for batch_x_commands, batch_x_states, batch_y, batch_y_prev, batch_commands in val_loader:
                batch_x_commands = batch_x_commands.to(device, non_blocking=use_cuda)
                batch_y = batch_y.to(device, non_blocking=use_cuda)
                batch_y_prev = batch_y_prev.to(device, non_blocking=use_cuda)
                batch_commands = batch_commands.to(device, non_blocking=use_cuda)
                if len(batch_x_states.shape) == 2:
                    batch_x_states = batch_x_states.to(device, non_blocking=use_cuda)
                else:
                    batch_x_states = None
                preds_all = torch.empty_like(batch_y)
                for t in range(batch_y.shape[1]):
                    # adjust history window:
                    if t > 0:
                        # use command sequence from dataset:
                        batch_x_commands = batch_x_commands.clone()
                        batch_x_commands[:, 2:] = batch_x_commands[:, :-2]
                        batch_x_commands[:, :2] = batch_commands[:, t]

                        # use states autoregressively:
                        if batch_x_states is not None:
                            batch_x_states[:, 2:] = batch_x_states.clone()[:, :-2]
                            batch_x_states[:, :2] = preds

                    # make prediction:
                    preds = model(batch_x_commands, batch_x_states)
                    preds_all[:, t] = preds

                mse = mse_loss(preds_all, batch_y)
                smoothness = smoothness_loss(preds_all, batch_y_prev)
                loss = mse + smoothness_weight * smoothness
                val_loss += loss.item() * batch_x_commands.size(0)
                val_loss_mse += mse.item() * batch_x_commands.size(0)
                val_loss_smoothness += smoothness.item() * batch_x_commands.size(0)
        val_loss /= len(val_dataset)
        val_loss_mse /= len(val_dataset)
        val_loss_smoothness /= len(val_dataset)
        if verbose:
            print(
                f"  Fold {cv_fold} Validation Loss: {val_loss:.6f}, MSE loss: {val_loss_mse:.4f}, smoothness loss:"
                f" {val_loss_smoothness:.4f}",
                flush=True,
            )
        fold_val_losses.append(val_loss)

        if val_loss < best_fold_loss:
            best_fold_loss = val_loss
            best_fold_state = model.state_dict()

        cv_fold += 1

    avg_val_loss = np.mean(fold_val_losses)
    if verbose:
        print(f"\nAverage Validation Loss across folds: {avg_val_loss:.4f}", flush=True)

    full_dataset = ActuatorDataset(
        torch.from_numpy(X_commands),
        torch.from_numpy(X_states) if X_states is not None else None,
        torch.from_numpy(Y),
        torch.from_numpy(Y_prev),
        torch.from_numpy(commands),
    )
    full_loader = DataLoader(
        full_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_cuda
    )

    input_shape = X_commands.shape[1] if X_states is None else X_commands.shape[1] + X_states.shape[1]
    model = ActuatorNetwork(input_shape, Y.shape[-1], hidden_dim=hidden_dim).to(device)
    model.load_state_dict(best_fold_state)  # Load best weights from CV

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x_commands, batch_x_states, batch_y, batch_y_prev, batch_commands in full_loader:
            batch_x_commands = batch_x_commands.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)
            batch_y_prev = batch_y_prev.to(device, non_blocking=use_cuda)
            batch_commands = batch_commands.to(device, non_blocking=use_cuda)
            if batch_x_states is not None and len(batch_x_states.shape) == 2:
                batch_x_states = batch_x_states.to(device, non_blocking=use_cuda)
            else:
                batch_x_states = None

            optimizer.zero_grad()
            preds_all = torch.empty_like(batch_y)
            for t in range(batch_y.shape[1]):
                if t > 0:
                    batch_x_commands[:, 2:] = batch_x_commands.clone()[:, :-2]
                    batch_x_commands[:, :2] = batch_commands[:, t]
                    if batch_x_states is not None:
                        batch_x_states[:, 2:] = batch_x_states.clone()[:, :-2]
                        batch_x_states[:, :2] = preds

                preds = model(batch_x_commands, batch_x_states)
                preds_all[:, t] = preds

            loss = mse_loss(preds_all, batch_y) + smoothness_weight * smoothness_loss(preds_all, batch_y_prev)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x_commands.size(0)

        epoch_loss /= len(full_dataset)
        print(f"  Final Training Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}", flush=True)
        if progress_queue is not None and seed is not None:
            progress_queue.put({
                "seed": seed,
                "phase": "full",
                "fold": None,
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "loss": float(epoch_loss),
            })

    print("Training on full dataset completed.", flush=True)
    return model.state_dict(), fold_val_losses, best_fold_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the actuator model across multiple seeds.")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(
            "/workspace/klask_rl/logs/actuator_model/data/train_traj/train/data_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states_train.npz"
        ),
        help="Path to the .npz dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.resolve(),
        help=(
            "Base output directory. Logs go to <output-dir>/logs/seed_runs and checkpoints to <output-dir>/checkpoints."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=0, help="First seed value.")
    parser.add_argument("--seed-count", type=int, default=8, help="Number of seeds to train.")
    parser.add_argument("--seed-workers", type=int, default=4, help="Number of parallel worker processes.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of time-series CV folds.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per fold and for final training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--smoothness-weight", type=float, default=0.0, help="Smoothness loss weight.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer dimension of the actuator network.")
    args = parser.parse_args()

    data_file = args.data_file
    run_name = Path(data_file).stem

    seed_start = args.seed_start
    seed_count = args.seed_count
    seed_workers = args.seed_workers
    logs_dir = args.output_dir / "logs" / "seed_runs"

    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    seeds = list(range(seed_start, seed_start + seed_count))
    results = []
    total_epochs = (args.n_splits + 1) * args.epochs

    manager = mp.Manager()
    progress_queue = manager.Queue()
    stop_event = threading.Event()
    progress_bars = {
        seed: tqdm(total=total_epochs, position=index, desc=f"seed {seed}", leave=True)
        for index, seed in enumerate(seeds)
    }

    def consume_progress():
        while not stop_event.is_set() or not progress_queue.empty():
            try:
                message = progress_queue.get(timeout=0.1)
            except Empty:
                continue

            seed = message.get("seed")
            bar = progress_bars.get(seed)
            if bar is None:
                continue

            bar.update(1)
            phase = message.get("phase")
            fold = message.get("fold")
            epoch = message.get("epoch")
            loss = message.get("loss")
            if phase == "cv":
                bar.set_postfix_str(f"cv f{fold} e{epoch} loss={loss:.6f}")
            else:
                bar.set_postfix_str(f"full e{epoch} loss={loss:.6f}")

    progress_thread = threading.Thread(target=consume_progress, daemon=True)
    progress_thread.start()

    with ProcessPoolExecutor(max_workers=seed_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(
                run_seed_training,
                seed,
                data_file,
                run_name,
                args.n_splits,
                args.epochs,
                args.lr,
                args.batch_size,
                args.smoothness_weight,
                args.hidden_dim,
                True,
                True,
                str(checkpoints_dir),
                str(logs_dir),
                progress_queue,
            )
            for seed in seeds
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            tqdm.write(
                f"Seed {result['seed']} done | best_fold_loss={result['best_fold_loss']:.6f},"
                f" avg_cv_loss={result['avg_cv_loss']:.6f} | log={result['log_path']}"
            )

    stop_event.set()
    progress_thread.join()
    for seed, bar in progress_bars.items():
        if bar.n < bar.total:
            bar.n = bar.total
            bar.refresh()
        bar.close()

    results = sorted(results, key=lambda item: item["seed"])
    print("\n=== Summary ===", flush=True)
    for result in results:
        print(
            f"Seed {result['seed']}: best_fold_loss={result['best_fold_loss']:.6f},"
            f" avg_cv_loss={result['avg_cv_loss']:.6f}, checkpoint={result['model_path']}",
            flush=True,
        )

    avg_best = np.mean([item["best_fold_loss"] for item in results])
    avg_cv = np.mean([item["avg_cv_loss"] for item in results])
    print(f"\nMean best_fold_loss={avg_best:.6f}, Mean avg_cv_loss={avg_cv:.6f}", flush=True)
