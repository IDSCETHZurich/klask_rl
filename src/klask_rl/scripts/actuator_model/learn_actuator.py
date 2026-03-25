import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit

from pathlib import Path
from actuator_network import ActuatorNetwork


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
        

def train_model_with_cv(X_commands, X_states, Y, Y_prev, commands, n_splits=5, epochs=20, lr=1e-3, batch_size=512, smoothness_weight=0.1, hidden_dim=32, verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_fold = 1

    fold_val_losses = []
    best_fold_state = None
    best_fold_loss = float('inf')

    # Ensure the data is float32
    X_commands = X_commands.astype(np.float32)
    if X_states is not None:
        X_states = X_states.astype(np.float32)
    Y = Y.astype(np.float32)
    Y_prev = Y_prev.astype(np.float32)
    commands = commands.astype(np.float32)
    torch.autograd.set_detect_anomaly(True)


    for train_index, val_index in tscv.split(X_commands):
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

        train_dataset = ActuatorDataset(torch.from_numpy(X_commands_train), torch.from_numpy(X_states_train) if X_states_train is not None else None, torch.from_numpy(Y_train), torch.from_numpy(Y_prev_train), torch.from_numpy(commands_train))
        val_dataset   = ActuatorDataset(torch.from_numpy(X_commands_val), torch.from_numpy(X_states_val) if X_states_val is not None else None, torch.from_numpy(Y_val), torch.from_numpy(Y_prev_val), torch.from_numpy(commands_val))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
                batch_x_commands, batch_y, batch_y_prev, batch_commands = batch_x_commands.to(device), batch_y.to(device), batch_y_prev.to(device), batch_commands.to(device)
                if len(batch_x_states.shape) == 2:
                    batch_x_states = batch_x_states.to(device)
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
                epoch_loss = epoch_loss+ loss.item() * batch_x_commands.size(0)
            epoch_loss /= len(train_dataset)
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {1000*epoch_loss:.4f}")

        # Validate
        model.eval()
        val_loss = 0.0
        val_loss_mse = 0.0
        val_loss_smoothness = 0.0
        with torch.no_grad():
            for batch_x_commands, batch_x_states, batch_y, batch_y_prev, batch_commands in val_loader:
                batch_x_commands, batch_y, batch_y_prev, batch_commands = batch_x_commands.to(device), batch_y.to(device), batch_y_prev.to(device), batch_commands.to(device)
                if len(batch_x_states.shape) == 2:
                    batch_x_states = batch_x_states.to(device)
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
            print(f"  Fold {cv_fold} Validation Loss: {1000*val_loss:.4f}, MSE loss: {val_loss_mse:.4f}, smoothness loss: {val_loss_smoothness:.4f}")
        fold_val_losses.append(val_loss)

        if val_loss < best_fold_loss:
            best_fold_loss = val_loss
            best_fold_state = model.state_dict()

        cv_fold += 1

    avg_val_loss = np.mean(fold_val_losses)
    if verbose:
        print(f"\nAverage Validation Loss across folds: {avg_val_loss:.4f}")

    full_dataset = ActuatorDataset(
        torch.from_numpy(X_commands),
        torch.from_numpy(X_states) if X_states is not None else None,
        torch.from_numpy(Y),
        torch.from_numpy(Y_prev),
        torch.from_numpy(commands)
    )
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    input_shape = X_commands.shape[1] if X_states is None else X_commands.shape[1] + X_states.shape[1]
    model = ActuatorNetwork(input_shape, Y.shape[-1], hidden_dim=hidden_dim).to(device)
    model.load_state_dict(best_fold_state)  # Load best weights from CV

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x_commands, batch_x_states, batch_y, batch_y_prev, batch_commands in full_loader:
            batch_x_commands, batch_y, batch_y_prev, batch_commands = (
                batch_x_commands.to(device),
                batch_y.to(device),
                batch_y_prev.to(device),
                batch_commands.to(device)
            )
            if batch_x_states is not None and len(batch_x_states.shape) == 2:
                batch_x_states = batch_x_states.to(device)
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
        print(f"  Final Training Epoch {epoch+1}/{epochs}, Loss: {1000*epoch_loss:.4f}")

    print("Training on full dataset completed.")
    return model.state_dict(),fold_val_losses, best_fold_loss

if __name__ == "__main__":

    data_file = "data_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states.npz"
    best_val_loss = 10000.0

    run_name = data_file[5:-4]
    data_file = Path(__file__).parent.resolve() / "data" / data_file

    data = np.load(data_file)
    X_commands, Y, Y_prev, commands = data["X_commands"], data["Y"], data["Y_prev"], data["commands"]
    if "X_states" in data.keys():
        X_states = data["X_states"]
    else:
        X_states = None

    best_state, cv_losses, best_fold_loss = train_model_with_cv(X_commands, 
                                                                X_states, 
                                                                Y, 
                                                                Y_prev, 
                                                                commands, 
                                                                n_splits=5, 
                                                                epochs=20, 
                                                                lr=1e-3, 
                                                                batch_size=512, 
                                                                smoothness_weight=0.0,
                                                                hidden_dim=64,
                                                                verbose=True)
    if best_fold_loss < best_val_loss:
        best_val_loss = best_fold_loss
        best_run = run_name
        print(f"New best run: {best_run}, val loss {best_val_loss}")
    
    # Save best model state from cross-validation
    model_path = f"checkpoints/model_{run_name}.pt"
    torch.save(best_state, model_path)
    print(f"Best model state saved to {model_path}")