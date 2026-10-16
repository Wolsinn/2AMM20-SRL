from pathlib import Path
import argparse
import sys
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
MODEL_DIR    = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_pointgoal"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safety.cost_model import CostPredictor
from safety.cost_targets import (
    concatenate_episode_ranges,
    discounted_cost_to_go,
    # episode_ranges,                # no longer needed here
    # stratified_episode_split,       # replaced by three_way_episode_split
    three_way_episode_split,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.20
TEST_SPLIT_SEED = 42        # fixed: the test episodes are identical for every seed
# TRAIN_FRACTION = 0.8      # replaced by the three-way split
# SPLIT_SEED = 42           # replaced by TEST_SPLIT_SEED and the per-run seed
BATCH_SIZE = 1024
EPOCHS = 20
LEARNING_RATE = 1e-3
GAMMA = 0.9 


def load_data():
    with h5py.File(DATASET_PATH, "r") as f:
        observations = np.asarray(f["observations"][:], dtype=np.float32)
        actions = np.asarray(f["actions"][:], dtype=np.float32)
        costs = np.asarray(f["costs"][:], dtype=np.float32)
        terminals = np.asarray(f["terminals"][:], dtype=bool)
        timeouts = np.asarray(f["timeouts"][:], dtype=bool)

    return observations, actions, costs, terminals, timeouts


def train_one_seed(seed, observations, actions, targets, costs, terminals, timeouts, args_num_seeds):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ranges, validation_ranges, test_ranges = three_way_episode_split(
        costs,
        terminals,
        timeouts,
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
        test_seed=TEST_SPLIT_SEED,
        split_seed=seed,
    )
    print(
        f"\nSeed {seed} | episodes train/validation/test = "
        f"{len(train_ranges)}/{len(validation_ranges)}/{len(test_ranges)}"
    )

    train_states = concatenate_episode_ranges(observations, train_ranges)
    train_actions = concatenate_episode_ranges(actions, train_ranges)
    train_targets = concatenate_episode_ranges(targets, train_ranges)
    validation_states = concatenate_episode_ranges(observations, validation_ranges)
    validation_actions = concatenate_episode_ranges(actions, validation_ranges)
    validation_targets = concatenate_episode_ranges(targets, validation_ranges)
    # The test episodes are NOT loaded here. They are only used in evaluate_cost_predictor.py.

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_states),
            torch.from_numpy(train_actions),
            torch.from_numpy(train_targets),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation_states),
            torch.from_numpy(validation_actions),
            torch.from_numpy(validation_targets),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    state_dim = train_states.shape[1]
    action_dim = train_actions.shape[1]

    model = CostPredictor(state_dim=state_dim, action_dim=action_dim, hidden_dim=256).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_validation_loss = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        count = 0

        for state, action, target in train_loader:
            state, action, target = state.to(DEVICE), action.to(DEVICE), target.to(DEVICE)

            prediction = model(state, action)
            loss = loss_fn(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(state)
            count += len(state)

        train_loss = total_loss / count

        model.eval()
        validation_squared_error = 0.0
        validation_count = 0

        with torch.no_grad():
            for state, action, target in validation_loader:
                state, action, target = state.to(DEVICE), action.to(DEVICE), target.to(DEVICE)
                error = model(state, action) - target
                validation_squared_error += torch.sum(error ** 2).item()
                validation_count += len(state)

        validation_loss = validation_squared_error / validation_count

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train MSE={train_loss:.6f} | "
            f"validation MSE={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)

    model_filename = "cost_predictor.pt" if args_num_seeds == 1 else f"cost_predictor_seed{seed}.pt"
    model_path = MODEL_DIR / model_filename

    torch.save(
        {
            "model": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "seed": seed,
            "split_seed": seed,
            "test_split_seed": TEST_SPLIT_SEED,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": TEST_FRACTION,
            "train_episode_ranges": train_ranges,
            "validation_episode_ranges": validation_ranges,
            "test_episode_ranges": test_ranges,
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation_loss,
            "gamma": GAMMA,
        },
        model_path,
    )

    print(f"Seed {seed} | best epoch {best_epoch} | best validation MSE {best_validation_loss:.6f}")
    print(f"Saved model to {model_path}")

    return best_validation_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=5)
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    observations, actions, costs, terminals, timeouts = load_data()

    normalization = np.load(MODEL_DIR / "normalization.npz")
    observations = ((observations - normalization["mean"]) / normalization["std"]).astype(np.float32)

    targets = discounted_cost_to_go(costs, terminals, timeouts, gamma=GAMMA)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_losses = [
        train_one_seed(seed, observations, actions, targets, costs, terminals, timeouts, args.num_seeds)
        for seed in range(args.num_seeds)
    ]

    print()
    print(f"Best validation MSE over {len(best_losses)} seeds: "
          f"{np.mean(best_losses):.6f} ± {np.std(best_losses, ddof=1) if len(best_losses) > 1 else 0.0:.6f}")


if __name__ == "__main__":
    main()