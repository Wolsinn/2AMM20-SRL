from pathlib import Path
import sys
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
MODEL_DIR    = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_pointgoal"
MODEL_PATH   = MODEL_DIR / "cost_predictor.pt"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safety.cost_model import CostPredictor
from safety.cost_targets import (
    concatenate_episode_ranges,
    discounted_cost_to_go,
    episode_ranges,
    stratified_episode_split,
)


DEVICE = torch.device("cpu")

TRAIN_FRACTION = 0.8
SPLIT_SEED = 42
BATCH_SIZE = 1024
EPOCHS = 20
LEARNING_RATE = 1e-3


def load_data():

    with h5py.File(DATASET_PATH, "r") as f:

        observations = np.asarray(
            f["observations"][:],
            dtype=np.float32,
        )

        actions = np.asarray(
            f["actions"][:],
            dtype=np.float32,
        )

        costs = np.asarray(
            f["costs"][:],
            dtype=np.float32,
        )

        terminals = np.asarray(
            f["terminals"][:],
            dtype=bool,
        )

        timeouts = np.asarray(
            f["timeouts"][:],
            dtype=bool,
        )

    return observations, actions, costs, terminals, timeouts


def main():

    torch.manual_seed(0)
    np.random.seed(0)

    observations, actions, costs, terminals, timeouts = load_data()

    all_episodes = episode_ranges(terminals, timeouts)
    train_ranges, validation_ranges = stratified_episode_split(
        costs,
        terminals,
        timeouts,
        validation_fraction=1.0 - TRAIN_FRACTION,
        seed=SPLIT_SEED,
    )
    num_episodes = len(all_episodes)
    train_episodes = len(train_ranges)
    validation_episodes = len(validation_ranges)

    print(f"Total episodes: {num_episodes}")
    print(f"Training episodes: {train_episodes}")
    print(f"Validation episodes: {validation_episodes}")

    normalization = np.load(MODEL_DIR / "normalization.npz")
    mean = normalization["mean"]
    std = normalization["std"]
    observations = (observations - mean) / std

    targets = discounted_cost_to_go(costs, terminals, timeouts, gamma=0.99)

    train_states = concatenate_episode_ranges(observations, train_ranges)
    train_actions = concatenate_episode_ranges(actions, train_ranges)
    train_targets = concatenate_episode_ranges(targets, train_ranges)
    validation_states = concatenate_episode_ranges(observations, validation_ranges)
    validation_actions = concatenate_episode_ranges(actions, validation_ranges)
    validation_targets = concatenate_episode_ranges(targets, validation_ranges)
    
    train_dataset = TensorDataset(
        torch.from_numpy(train_states),
        torch.from_numpy(train_actions),
        torch.from_numpy(train_targets),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    state_dim  = train_states.shape[1]
    action_dim = train_actions.shape[1]

    model = CostPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_fn = nn.MSELoss()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0
        count = 0

        for (state, action, target) in train_loader:

            state  = state.to(DEVICE)
            action = action.to(DEVICE)
            target = target.to(DEVICE)

            prediction = model(state, action)
            loss = loss_fn(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += (loss.item() * len(state))
            count += len(state)

        train_loss = total_loss / count

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        model.eval()

        validation_squared_error = 0.0
        validation_count = 0

        validation_dataset = TensorDataset(
            torch.from_numpy(validation_states),
            torch.from_numpy(validation_actions),
            torch.from_numpy(validation_targets),
        )

        validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        with torch.no_grad():

            for state, action, target in validation_loader:
                prediction = model(state, action)

                error = prediction - target
                validation_squared_error += (torch.sum(error ** 2).item())
                validation_count += len(state)

        validation_loss = (validation_squared_error / validation_count)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train MSE={train_loss:.6f} | "
            f"validation MSE={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    # ---------------------------------------------------------
    # Save best model
    # ---------------------------------------------------------

    model.load_state_dict(best_state)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "train_episodes": train_episodes,
            "validation_episodes": validation_episodes,
            "split_seed": SPLIT_SEED,
            "validation_fraction": 1.0 - TRAIN_FRACTION,
            "validation_episode_ranges": validation_ranges,
            "gamma": 0.99,
        },
        MODEL_PATH,
    )

    print()
    print(f"Best validation MSE: {best_validation_loss:.6f}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()