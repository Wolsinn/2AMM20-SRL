from pathlib import Path
import argparse
import sys

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
# DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyCarButton1Gymnasium-v0-250-2656.hdf5"
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyCarButton2Gymnasium-v0-300-3755.hdf5"
# MODEL_DIR    = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_pointgoal"
# MODEL_DIR = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_carbutton1"
MODEL_DIR = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_carbutton2"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safety.cost_model import CostPredictor
from safety.cost_targets import concatenate_episode_ranges, discounted_cost_to_go
# from safety.cost_targets import episode_ranges, stratified_episode_split   # old split, unused


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4096


def predict(model, states, actions):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(states), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(states))
            batch_states = torch.from_numpy(states[start:end]).to(DEVICE)
            batch_actions = torch.from_numpy(actions[start:end]).to(DEVICE)
            predictions.append(model(batch_states, batch_actions).cpu().numpy())
    return np.concatenate(predictions)


def evaluate_ranges(observations, actions, targets, ranges, model, taus):
    states = concatenate_episode_ranges(observations, ranges).astype(np.float32)
    batch_actions = concatenate_episode_ranges(actions, ranges).astype(np.float32)
    labels = concatenate_episode_ranges(targets, ranges)

    predictions = predict(model, states, batch_actions)
    errors = predictions - labels

    target_mean = float(labels.mean())
    target_std = float(labels.std())
    prediction_std = float(predictions.std())
    mse = float(np.mean(errors ** 2))
    mean_baseline_mse = float(np.mean((labels - target_mean) ** 2))
    correlation = (
        float(np.corrcoef(labels, predictions)[0, 1])
        if target_std > 0 and prediction_std > 0
        else float("nan")
    )

    metrics = {
        "transitions": len(labels),
        "target_mean": target_mean,
        "prediction_mean": float(predictions.mean()),
        "mse": mse,
        "mae": float(np.mean(np.abs(errors))),
        "r2": 1.0 - mse / mean_baseline_mse if mean_baseline_mse > 0 else float("nan"),
        "correlation": correlation,
        "zero_baseline_mse": float(np.mean(labels ** 2)),
        "mean_baseline_mse": mean_baseline_mse,
    }

    # MSE restricted to transitions with future cost (target > 0)
    positive = labels > 0
    if positive.any():
        positive_labels = labels[positive]
        metrics["positive_transitions"] = int(positive.sum())
        metrics["mse_positive"] = float(np.mean(errors[positive] ** 2))
        metrics["positive_mean_baseline_mse"] = float(
            np.mean((positive_labels - positive_labels.mean()) ** 2)
        )
        metrics["positive_zero_baseline_mse"] = float(np.mean(positive_labels ** 2))
    else:
        metrics["positive_transitions"] = 0
        metrics["mse_positive"] = float("nan")
        metrics["positive_mean_baseline_mse"] = float("nan")
        metrics["positive_zero_baseline_mse"] = float("nan")

    # Ranking quality for the event "target > tau"
    for tau in taus:
        event = labels > tau
        metrics[f"positive_rate_tau{tau:g}"] = float(event.mean())
        if 0 < event.sum() < len(event):
            metrics[f"auroc_tau{tau:g}"] = float(roc_auc_score(event, predictions))
            metrics[f"auprc_tau{tau:g}"] = float(average_precision_score(event, predictions))
        else:
            metrics[f"auroc_tau{tau:g}"] = float("nan")
            metrics[f"auprc_tau{tau:g}"] = float("nan")

    return metrics

METRICS_TO_PRINT = ("mse", "mae", "r2")
def print_metrics(title, metrics):
    print(title)
    print("-" * len(title))

    for name in METRICS_TO_PRINT:
        print(f"{name:<15}{metrics[name]:.6f}")

    # Print AUROC and AUPRC for each requested threshold
    for name, value in metrics.items():
        if name.startswith(("positive_rate_tau", "auroc_tau", "auprc_tau")):
            print(f"{name:<15}{value:.6f}")




def print_summary(title, results):
    print(title)
    print("-" * len(title))

    metric_names = list(METRICS_TO_PRINT)
    metric_names += [
        name for name in results[0]
        if name.startswith(("positive_rate_tau", "auroc_tau", "auprc_tau"))
    ]

    for name in metric_names:
        values = np.array(
            [result[name] for result in results], dtype=np.float64
        )
        mean = np.nanmean(values) if not np.all(np.isnan(values)) else float("nan")
        std = np.nanstd(values, ddof=1) if len(values) > 1 else 0.0
        print(f"{name:<30}{mean:.6f} ± {std:.6f}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Use 'test' only for the final reported numbers.",
    )
    parser.add_argument("--taus", nargs="+", type=float, default=[0.0])
    args = parser.parse_args()

    normalization = np.load(MODEL_DIR / "normalization.npz")
    mean = normalization["mean"]
    std = normalization["std"]

    with h5py.File(DATASET_PATH, "r") as file:
        observations = np.asarray(file["observations"][:], dtype=np.float32)
        actions = np.asarray(file["actions"][:], dtype=np.float32)
        costs = np.asarray(file["costs"][:], dtype=np.float32)
        terminals = np.asarray(file["terminals"][:], dtype=bool)
        timeouts = np.asarray(file["timeouts"][:], dtype=bool)

    observations = ((observations - mean) / std).astype(np.float32)


    results = []
    reference_ranges = None
    single_seed = args.num_seeds == 1

    for seed in range(args.num_seeds):
        checkpoint_name = (
            "cost_predictor.pt"
            if single_seed
            else f"cost_predictor_seed{seed}.pt"
        )
        checkpoint = torch.load(
            MODEL_DIR / checkpoint_name,
            map_location=DEVICE,
        )

        model = CostPredictor(
            checkpoint["state_dim"], checkpoint["action_dim"]
        ).to(DEVICE)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        # Leakage check: the three splits must be disjoint
        split_sets = {
            name: {
                tuple(bounds)
                for bounds in checkpoint[f"{name}_episode_ranges"]
            }
            for name in ("train", "validation", "test")
        }
        assert not (split_sets["train"] & split_sets["validation"])
        assert not (split_sets["train"] & split_sets["test"])
        assert not (split_sets["validation"] & split_sets["test"])

        ranges = [
            tuple(bounds)
            for bounds in checkpoint[f"{args.split}_episode_ranges"]
        ]

        # The test set must be identical for every seed
        if args.split == "test":
            if reference_ranges is None:
                reference_ranges = ranges
            elif ranges != reference_ranges:
                raise RuntimeError(f"Seed {seed} uses a different test set")

        targets = discounted_cost_to_go(
            costs, terminals, timeouts, gamma=checkpoint["gamma"]
        )
        metrics = evaluate_ranges(
            observations, actions, targets, ranges, model, args.taus
        )
        results.append(metrics)

        title = (
            f"{args.split.capitalize()} set"
            if single_seed
            else f"Seed {seed} | {args.split} set"
        )
        print_metrics(title, metrics)
        print()

    summary_title = (
        f"{args.split.capitalize()} set"
        if single_seed
        else f"Mean ± std over {len(results)} seeds | {args.split} set"
    )

    if single_seed:
        print_metrics(summary_title, results[0])
    else:
        print_summary(summary_title, results)


    # ---- Disabled: chronological stress holdout -------------------------------
    # With a random training split, the last 20% of episodes were mostly in the
    # training set, so this was not a true holdout.
    # all_ranges = episode_ranges(terminals, timeouts)
    # chronological_start = int(len(all_ranges) * (1.0 - validation_fraction))
    # stress_metrics = evaluate_ranges(..., all_ranges[chronological_start:], ...)
    # print_metrics("\nChronological stress holdout", stress_metrics)


if __name__ == "__main__":
    main()