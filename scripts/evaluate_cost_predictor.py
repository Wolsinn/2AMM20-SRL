from pathlib import Path
import sys

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyCarButton1Gymnasium-v0-250-2656.hdf5"
# MODEL_DIR    = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_pointgoal"
MODEL_DIR = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_carbutton1"
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
BATCH_SIZE = 4096


def evaluate_ranges(file, ranges, targets, mean, std, model):
    observations = concatenate_episode_ranges(file["observations"], ranges).astype(np.float32)
    actions = concatenate_episode_ranges(file["actions"], ranges).astype(np.float32)
    labels = concatenate_episode_ranges(targets, ranges)
    observations = ((observations - mean) / std).astype(np.float32)

    predictions = []
    with torch.no_grad():
        for start in range(0, len(observations), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(observations))
            states = torch.from_numpy(observations[start:end]).to(DEVICE)
            batch_actions = torch.from_numpy(actions[start:end]).to(DEVICE)
            predictions.append(model(states, batch_actions).cpu().numpy())

    predictions = np.concatenate(predictions)
    errors = predictions - labels
    target_mean = float(labels.mean())
    target_std = float(labels.std())
    prediction_std = float(predictions.std())
    correlation = (
        float(np.corrcoef(labels, predictions)[0, 1])
        if target_std > 0 and prediction_std > 0
        else float("nan")
    )

    return {
        "transitions": len(labels),
        "target_mean": target_mean,
        "prediction_mean": float(predictions.mean()),
        "mse": float(np.mean(errors ** 2)),
        "mae": float(np.mean(np.abs(errors))),
        "correlation": correlation,
        "zero_baseline_mse": float(np.mean(labels ** 2)),
        "mean_baseline_mse": float(np.mean((labels - target_mean) ** 2)),
    }


def print_metrics(title, metrics):
    print(title)
    print("-" * len(title))
    print(f"Transitions:       {metrics['transitions']}")
    print(f"Target mean:       {metrics['target_mean']:.6f}")
    print(f"Prediction mean:   {metrics['prediction_mean']:.6f}")
    print(f"MSE:               {metrics['mse']:.6f}")
    print(f"MAE:               {metrics['mae']:.6f}")
    print(f"Correlation:       {metrics['correlation']:.6f}")
    print(f"Zero baseline MSE: {metrics['zero_baseline_mse']:.6f}")
    print(f"Mean baseline MSE: {metrics['mean_baseline_mse']:.6f}")


def main():
    normalization = np.load(MODEL_DIR / "normalization.npz")
    mean = normalization["mean"]
    std = normalization["std"]

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = CostPredictor(checkpoint["state_dim"], checkpoint["action_dim"]).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    validation_fraction = checkpoint.get("validation_fraction", 0.2)
    with h5py.File(DATASET_PATH, "r") as file:
        costs = np.asarray(file["costs"][:], dtype=np.float32)
        terminals = np.asarray(file["terminals"][:], dtype=bool)
        timeouts = np.asarray(file["timeouts"][:], dtype=bool)
        targets = discounted_cost_to_go(costs, terminals, timeouts, gamma=checkpoint.get("gamma", 0.99))
        all_ranges = episode_ranges(terminals, timeouts)
        _, stratified_ranges = stratified_episode_split(
            costs,
            terminals,
            timeouts,
            validation_fraction=validation_fraction,
            seed=checkpoint.get("split_seed", 42),
        )
        validation_ranges = checkpoint.get("validation_episode_ranges", stratified_ranges)
        validation_ranges = [tuple(bounds) for bounds in validation_ranges]
        chronological_start = int(len(all_ranges) * (1.0 - validation_fraction))

        validation_metrics = evaluate_ranges(file, validation_ranges, targets, mean, std, model)
        stress_metrics = evaluate_ranges(file, all_ranges[chronological_start:], targets, mean, std, model)

    print("Cost predictor evaluation")
    print("=========================")
    print_metrics("Stratified episode holdout", validation_metrics)
    print_metrics("\nChronological stress holdout", stress_metrics)


if __name__ == "__main__":
    main()