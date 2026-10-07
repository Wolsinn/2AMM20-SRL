from pathlib import Path
import sys

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
MODEL_DIR    = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_pointgoal"
MODEL_PATH   = MODEL_DIR / "cost_predictor.pt"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safety.cost_model import CostPredictor
from safety.cost_targets import discounted_cost_to_go


DEVICE = torch.device("cpu")


def main():
    normalization = np.load(MODEL_DIR / "normalization.npz")
    mean = normalization["mean"]
    std = normalization["std"]

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = CostPredictor(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model"])

    model.eval()

    squared_error = 0.0
    absolute_error = 0.0

    target_sum = 0.0
    prediction_sum = 0.0

    target_squared_sum = 0.0
    prediction_squared_sum = 0.0
    target_prediction_sum = 0.0

    count = 0
    batch_size = 4096

    with h5py.File(DATASET_PATH, "r") as f:
        episode_length = 1000
        split_episode  = int(0.8 * len(f["observations"]))
        start_index    = split_episode * episode_length

        observations = np.asarray(
            f["observations"][start_index:],
            dtype=np.float32,
        )

        actions = np.asarray(
            f["actions"][start_index:],
            dtype=np.float32,
        )

        costs = np.asarray(
            f["costs"][start_index:],
            dtype=np.float32,
        )

        terminals = np.asarray(
            f["terminals"][start_index:],
            dtype=bool,
        )

        timeouts = np.asarray(
            f["timeouts"][start_index:],
            dtype=bool,
        )

        observations = (observations - mean) / std

        targets = discounted_cost_to_go(costs, terminals, timeouts, gamma=0.99)

        with torch.no_grad():

            for start in range(0, len(observations), batch_size):

                end = min(start + batch_size, len(observations),)

                states = torch.from_numpy(
                    observations[start:end]
                ).to(DEVICE)

                actions_batch = torch.from_numpy(
                    actions[start:end]
                ).to(DEVICE)

                predictions = model(
                    states,
                    actions_batch,
                ).cpu().numpy()

                target = targets[start:end]
                error = predictions - target

                squared_error += np.sum(error ** 2)
                absolute_error += np.sum(np.abs(error))

                target_sum += np.sum(target)
                prediction_sum += np.sum(predictions)

                target_squared_sum     += np.sum(target ** 2)
                prediction_squared_sum += np.sum(predictions ** 2)
                target_prediction_sum  += np.sum(target * predictions)

                count += len(target)

    mse = squared_error / count
    mae = absolute_error / count
    target_mean = target_sum / count
    prediction_mean = prediction_sum / count

    target_variance = (
        target_squared_sum / count
        - target_mean ** 2
    )

    prediction_variance = (
        prediction_squared_sum / count
        - prediction_mean ** 2
    )

    covariance = (
        target_prediction_sum / count
        - target_mean * prediction_mean
    )

    denominator = np.sqrt(target_variance * prediction_variance)

    if denominator > 0:
        correlation = covariance / denominator
    else:
        correlation = float("nan")

    print("Cost predictor validation")
    print("=========================")
    print(f"Transitions:     {count}")
    print(f"MSE:             {mse:.6f}")
    print(f"MAE:             {mae:.6f}")
    print(f"Correlation:     {correlation:.6f}")
    print(f"Target mean:     {target_mean:.6f}")
    print(f"Prediction mean: {prediction_mean:.6f}")


if __name__ == "__main__":
    main()