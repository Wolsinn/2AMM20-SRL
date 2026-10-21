"""Evaluate the Safe-POR K-step safety-violation risk model.

This script evaluates the held-out validation portion of the DSRL dataset using
exactly the train/validation split stored in ``risk_predictor.pt``.

The risk model predicts:

    P(violation within K steps | s_t, a_t)

The script reports both probability-quality metrics (BCE, Brier score,
ROC-AUC, PR-AUC, ECE) and safety-threshold metrics that are directly relevant
to the Safe-POR filter (recall, false-negative rate, specificity, intervention
rate, etc.).
"""

from pathlib import Path
import argparse
import sys

import h5py
import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#
# This file lives in:
#
#   scripts/evaluation/evaluate_risk_model.py
#
# Therefore:
#
#   parents[0] -> scripts/evaluation
#   parents[1] -> scripts
#   parents[2] -> repository root
#
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dsrl"
    / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
)

DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "runs"
    / "por_dsrl_pointgoal"
)

# Allow imports from src/.
sys.path.insert(0, str(SRC_DIR))

from safety.risk_model import RiskPredictor
from safety.risk_targets import k_step_violation_labels


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def parse_thresholds(value):
    """Parse a comma-separated list of probability thresholds.

    Parameters
    ----------
    value : str
        Example: ``"0.05,0.10,0.20,0.30,0.50"``.

    Returns
    -------
    list[float]
        Sorted unique thresholds in the interval [0, 1].
    """

    try:
        thresholds = [
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "thresholds must be comma-separated numbers"
        ) from exc

    if not thresholds:
        raise argparse.ArgumentTypeError(
            "at least one threshold must be supplied"
        )

    if any(
        threshold < 0.0 or threshold > 1.0
        for threshold in thresholds
    ):
        raise argparse.ArgumentTypeError(
            "all thresholds must be between 0 and 1"
        )

    return sorted(set(thresholds))


def binary_classification_metrics(
    targets,
    probabilities,
    threshold,
):
    """Compute binary safety metrics at one risk threshold.

    Safe-POR accepts actions when:

        predicted_risk <= threshold

    and intervenes when:

        predicted_risk > threshold

    Therefore, a predicted positive in this function means that the safety
    layer would classify the state-action pair as unsafe and intervene.

    Parameters
    ----------
    targets : np.ndarray
        Binary ground-truth K-step violation labels.

    probabilities : np.ndarray
        Predicted violation probabilities.

    threshold : float
        Safe-POR risk threshold epsilon.

    Returns
    -------
    dict
        Confusion-matrix counts and derived classification/safety metrics.
    """

    targets = np.asarray(
        targets,
        dtype=bool,
    )

    # Match RiskSafetyFilter exactly: risk <= epsilon is accepted,
    # risk > epsilon triggers intervention.
    predicted_unsafe = (
        probabilities > threshold
    )

    tp = int(np.sum(
        predicted_unsafe & targets
    ))

    fp = int(np.sum(
        predicted_unsafe & ~targets
    ))

    tn = int(np.sum(
        ~predicted_unsafe & ~targets
    ))

    fn = int(np.sum(
        ~predicted_unsafe & targets
    ))

    total = tp + fp + tn + fn

    def safe_divide(numerator, denominator):
        if denominator == 0:
            return float("nan")
        return numerator / denominator

    accuracy = safe_divide(
        tp + tn,
        total,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    false_negative_rate = safe_divide(
        fn,
        tp + fn,
    )

    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )

    if (
        np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0.0
    ):
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = float("nan")

    intervention_rate = safe_divide(
        tp + fp,
        total,
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_negative_rate": false_negative_rate,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
        "intervention_rate": intervention_rate,
    }


def roc_auc_score_numpy(targets, probabilities):
    """Compute ROC-AUC without requiring scikit-learn.

    Returns NaN if the validation set contains only one class.
    """

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    positives = int(np.sum(targets == 1))
    negatives = int(np.sum(targets == 0))

    if positives == 0 or negatives == 0:
        return float("nan")

    # Sort from highest predicted risk to lowest.
    order = np.argsort(-probabilities)
    sorted_targets = targets[order]
    sorted_probabilities = probabilities[order]

    cumulative_tp = np.cumsum(
        sorted_targets == 1
    )

    cumulative_fp = np.cumsum(
        sorted_targets == 0
    )

    # Evaluate the ROC curve only after the final member of each group of
    # identical scores. This handles ties correctly.
    distinct_indices = np.where(
        np.diff(sorted_probabilities) != 0.0
    )[0]

    threshold_indices = np.concatenate(
        [
            distinct_indices,
            np.array(
                [len(sorted_probabilities) - 1],
                dtype=np.int64,
            ),
        ]
    )

    tpr = cumulative_tp[
        threshold_indices
    ] / positives

    fpr = cumulative_fp[
        threshold_indices
    ] / negatives

    # ROC starts at (0, 0).
    tpr = np.concatenate(
        [[0.0], tpr]
    )

    fpr = np.concatenate(
        [[0.0], fpr]
    )

    return float(
        np.trapz(tpr, fpr)
    )


def pr_auc_score_numpy(targets, probabilities):
    """Compute trapezoidal area under the precision-recall curve.

    Returns NaN when the validation set contains no positive labels.
    """

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    positives = int(np.sum(targets == 1))

    if positives == 0:
        return float("nan")

    order = np.argsort(-probabilities)
    sorted_targets = targets[order]
    sorted_probabilities = probabilities[order]

    cumulative_tp = np.cumsum(
        sorted_targets == 1
    )

    cumulative_fp = np.cumsum(
        sorted_targets == 0
    )

    distinct_indices = np.where(
        np.diff(sorted_probabilities) != 0.0
    )[0]

    threshold_indices = np.concatenate(
        [
            distinct_indices,
            np.array(
                [len(sorted_probabilities) - 1],
                dtype=np.int64,
            ),
        ]
    )

    tp = cumulative_tp[
        threshold_indices
    ].astype(np.float64)

    fp = cumulative_fp[
        threshold_indices
    ].astype(np.float64)

    recall = tp / positives

    precision = tp / np.maximum(
        tp + fp,
        1.0,
    )

    # Conventionally, precision is 1 at recall 0 before any prediction is
    # accepted as positive.
    recall = np.concatenate(
        [[0.0], recall]
    )

    precision = np.concatenate(
        [[1.0], precision]
    )

    return float(
        np.trapz(precision, recall)
    )


def calibration_metrics(
    targets,
    probabilities,
    num_bins,
):
    """Compute expected calibration error and per-bin statistics.

    A well-calibrated risk model should approximately satisfy:

        predicted probability ~= observed violation frequency

    within groups of similarly scored samples.

    Returns
    -------
    ece : float
        Expected calibration error.

    bins : list[dict]
        Per-bin count, mean predicted risk, and observed violation rate.
    """

    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    # Values exactly equal to 1 should belong to the final bin.
    bin_indices = np.digitize(
        probabilities,
        bin_edges[1:-1],
        right=False,
    )

    total = len(targets)
    ece = 0.0
    rows = []

    for bin_index in range(num_bins):

        mask = (
            bin_indices == bin_index
        )

        count = int(np.sum(mask))

        if count == 0:
            mean_probability = float("nan")
            observed_rate = float("nan")
        else:
            mean_probability = float(
                np.mean(probabilities[mask])
            )

            observed_rate = float(
                np.mean(targets[mask])
            )

            ece += (
                count / total
            ) * abs(
                mean_probability
                - observed_rate
            )

        rows.append(
            {
                "lower": float(
                    bin_edges[bin_index]
                ),
                "upper": float(
                    bin_edges[bin_index + 1]
                ),
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )

    return float(ece), rows


def load_validation_data(
    dataset_path,
    train_end,
):
    """Load only the held-out portion of the DSRL dataset.

    ``train_end`` is read from the training checkpoint. It is the first
    validation transition and should coincide with the start of an episode.

    Returns
    -------
    observations, actions, costs, terminals, timeouts : np.ndarray
        Validation arrays only.
    """

    with h5py.File(dataset_path, "r") as file:

        required_keys = [
            "observations",
            "actions",
            "costs",
            "terminals",
            "timeouts",
        ]

        for key in required_keys:
            if key not in file:
                raise KeyError(
                    f"Required dataset field '{key}' "
                    f"is missing from {dataset_path}."
                )

        num_transitions = len(
            file["observations"]
        )

        if not 0 < train_end < num_transitions:
            raise ValueError(
                "Checkpoint train_transitions must fall strictly "
                "inside the dataset. "
                f"Got {train_end} for a dataset with "
                f"{num_transitions} transitions."
            )

        # The split used in train_risk_model.py is episode-based. Verify
        # that the final training transition actually closes an episode.
        previous_terminal = bool(
            file["terminals"][train_end - 1]
        )

        previous_timeout = bool(
            file["timeouts"][train_end - 1]
        )

        if not (
            previous_terminal
            or previous_timeout
        ):
            raise ValueError(
                "The saved training split does not end on an episode "
                "boundary. Evaluation labels could cross from the "
                "training set into the validation set."
            )

        observations = np.asarray(
            file["observations"][train_end:],
            dtype=np.float32,
        )

        actions = np.asarray(
            file["actions"][train_end:],
            dtype=np.float32,
        )

        costs = np.asarray(
            file["costs"][train_end:],
            dtype=np.float32,
        ).reshape(-1)

        terminals = np.asarray(
            file["terminals"][train_end:],
            dtype=bool,
        ).reshape(-1)

        timeouts = np.asarray(
            file["timeouts"][train_end:],
            dtype=bool,
        ).reshape(-1)

    return (
        observations,
        actions,
        costs,
        terminals,
        timeouts,
    )


def run_model(
    model,
    observations,
    actions,
    batch_size,
):
    """Run batched model inference and return logits and probabilities."""

    logits_parts = []
    probability_parts = []

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(observations),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(observations),
            )

            states = torch.from_numpy(
                observations[start:end]
            ).to(
                DEVICE,
                non_blocking=True,
            )

            action_batch = torch.from_numpy(
                actions[start:end]
            ).to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(
                states,
                action_batch,
            )

            probabilities = torch.sigmoid(
                logits
            )

            logits_parts.append(
                logits.cpu().numpy()
            )

            probability_parts.append(
                probabilities.cpu().numpy()
            )

    return (
        np.concatenate(logits_parts),
        np.concatenate(probability_parts),
    )


def build_argument_parser():
    """Construct command-line arguments for evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Safe-POR K-step risk predictor "
            "on the held-out DSRL validation set."
        )
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the DSRL HDF5 dataset.",
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=(
            "POR run directory containing normalization.npz "
            "and risk_predictor.pt."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Inference batch size.",
    )

    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds(
            "0.05,0.10,0.20,0.30,0.50"
        ),
        help=(
            "Comma-separated Safe-POR risk thresholds. "
            "Example: --thresholds 0.05,0.10,0.20,0.30,0.50"
        ),
    )

    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help=(
            "Number of equal-width probability bins used for "
            "expected calibration error."
        ),
    )

    return parser


def main():

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be at least 1"
        )

    if args.calibration_bins < 2:
        parser.error(
            "--calibration-bins must be at least 2"
        )

    dataset_path = (
        args.dataset_path.resolve()
    )

    run_dir = (
        args.run_dir.resolve()
    )

    model_path = (
        run_dir / "risk_predictor.pt"
    )

    normalization_path = (
        run_dir / "normalization.npz"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"DSRL dataset not found: {dataset_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            "Risk predictor checkpoint not found. "
            "Train the model first with train_risk_model.py.\n"
            f"Expected: {model_path}"
        )

    if not normalization_path.exists():
        raise FileNotFoundError(
            f"POR normalization file not found: {normalization_path}"
        )

    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset_path}")
    print(f"Risk model: {model_path}")
    print()

    # ------------------------------------------------------------------
    # Load checkpoint.
    # ------------------------------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )

    required_checkpoint_fields = [
        "model",
        "state_dim",
        "action_dim",
        "hidden_dim",
        "horizon",
        "cost_threshold",
        "train_transitions",
    ]

    for field in required_checkpoint_fields:
        if field not in checkpoint:
            raise KeyError(
                f"Checkpoint is missing required field '{field}'."
            )

    horizon = int(
        checkpoint["horizon"]
    )

    cost_threshold = float(
        checkpoint["cost_threshold"]
    )

    train_end = int(
        checkpoint["train_transitions"]
    )

    print(f"K-step horizon: {horizon}")
    print(
        "Cost threshold: "
        f"{cost_threshold}"
    )
    print(
        "First validation transition: "
        f"{train_end}"
    )
    print()

    # ------------------------------------------------------------------
    # Load exactly the held-out validation transitions.
    # ------------------------------------------------------------------

    print(
        "Loading validation split..."
    )

    (
        observations,
        actions,
        costs,
        terminals,
        timeouts,
    ) = load_validation_data(
        dataset_path=dataset_path,
        train_end=train_end,
    )

    print(
        f"Validation transitions: {len(observations)}"
    )

    # ------------------------------------------------------------------
    # Reconstruct labels using the same definition used during training.
    # ------------------------------------------------------------------

    targets = k_step_violation_labels(
        costs=costs,
        terminals=terminals,
        timeouts=timeouts,
        horizon=horizon,
        cost_threshold=cost_threshold,
    )

    positive_rate = float(
        np.mean(targets)
    )

    print(
        "Validation positive-label rate: "
        f"{positive_rate:.4%}"
    )
    print()

    # ------------------------------------------------------------------
    # Normalize states exactly as POR and the risk model training script do.
    # ------------------------------------------------------------------

    normalization = np.load(
        normalization_path
    )

    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        normalization["std"],
        dtype=np.float32,
    )

    if observations.shape[1] != len(mean):
        raise ValueError(
            "Observation dimension does not match normalization.npz."
        )

    observations -= mean
    observations /= std

    # ------------------------------------------------------------------
    # Build the trained model.
    # ------------------------------------------------------------------

    if observations.shape[1] != int(
        checkpoint["state_dim"]
    ):
        raise ValueError(
            "Validation state dimension does not match the checkpoint."
        )

    if actions.shape[1] != int(
        checkpoint["action_dim"]
    ):
        raise ValueError(
            "Validation action dimension does not match the checkpoint."
        )

    model = RiskPredictor(
        state_dim=int(
            checkpoint["state_dim"]
        ),
        action_dim=int(
            checkpoint["action_dim"]
        ),
        hidden_dim=int(
            checkpoint["hidden_dim"]
        ),
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model"]
    )

    # ------------------------------------------------------------------
    # Inference.
    # ------------------------------------------------------------------

    print(
        "Running risk-model inference..."
    )

    logits, probabilities = run_model(
        model=model,
        observations=observations,
        actions=actions,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Probability-quality metrics.
    # ------------------------------------------------------------------

    targets_tensor = torch.from_numpy(
        targets
    ).to(DEVICE)

    logits_tensor = torch.from_numpy(
        logits
    ).to(DEVICE)

    bce = float(
        nn.functional.binary_cross_entropy_with_logits(
            logits_tensor,
            targets_tensor,
            reduction="mean",
        ).item()
    )

    brier = float(
        np.mean(
            (probabilities - targets) ** 2
        )
    )

    roc_auc = roc_auc_score_numpy(
        targets,
        probabilities,
    )

    pr_auc = pr_auc_score_numpy(
        targets,
        probabilities,
    )

    ece, calibration_rows = calibration_metrics(
        targets=targets,
        probabilities=probabilities,
        num_bins=args.calibration_bins,
    )

    print()
    print("Risk predictor validation")
    print("=========================")
    print(
        f"Transitions:        {len(targets)}"
    )
    print(
        f"Positive labels:    {int(np.sum(targets))} "
        f"({positive_rate:.4%})"
    )
    print(f"BCE:                {bce:.6f}")
    print(f"Brier score:        {brier:.6f}")
    print(f"ROC-AUC:            {roc_auc:.6f}")
    print(f"PR-AUC:             {pr_auc:.6f}")
    print(f"ECE:                {ece:.6f}")
    print(
        "Mean prediction:    "
        f"{float(np.mean(probabilities)):.6f}"
    )
    print()

    # ------------------------------------------------------------------
    # Threshold sweep.
    #
    # These values matter directly to Safe-POR because epsilon determines
    # whether the filter accepts the POR action or searches for a replacement.
    # ------------------------------------------------------------------

    print("Safe-POR threshold sweep")
    print("=========================")
    print(
        " eps   recall    FNR    precision  specificity  intervention   TP     FN     FP"
    )

    for threshold in args.thresholds:

        metrics = binary_classification_metrics(
            targets=targets,
            probabilities=probabilities,
            threshold=threshold,
        )

        print(
            f"{threshold:5.2f}  "
            f"{metrics['recall']:7.4f}  "
            f"{metrics['false_negative_rate']:7.4f}  "
            f"{metrics['precision']:9.4f}  "
            f"{metrics['specificity']:11.4f}  "
            f"{metrics['intervention_rate']:12.4f}  "
            f"{metrics['tp']:6d}  "
            f"{metrics['fn']:6d}  "
            f"{metrics['fp']:6d}"
        )

    print()

    # ------------------------------------------------------------------
    # Calibration table.
    #
    # If the model is well calibrated, the two rightmost columns should be
    # reasonably similar for sufficiently populated bins.
    # ------------------------------------------------------------------

    print("Calibration")
    print("===========")
    print(
        " probability bin     count   mean predicted   observed violation rate"
    )

    for row in calibration_rows:

        if row["count"] == 0:
            continue

        print(
            f"[{row['lower']:.2f}, {row['upper']:.2f})"
            f"  {row['count']:9d}"
            f"  {row['mean_probability']:14.6f}"
            f"  {row['observed_rate']:23.6f}"
        )

    print()
    print(
        "Interpretation: for Safe-POR, high recall and a low false-negative "
        "rate are especially important because false negatives are truly "
        "unsafe state-action pairs that the filter would incorrectly accept."
    )


if __name__ == "__main__":
    main()
