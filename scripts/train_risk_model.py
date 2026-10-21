"""Train the Safe-POR K-step safety-violation risk model."""

from pathlib import Path
import argparse
import random
import sys

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

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


def set_seed(seed):
    """
    Seed Python, NumPy, and PyTorch for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(dataset_path):
    """
    Load the DSRL fields required for risk-model training.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Path to the DSRL HDF5 dataset.

    Returns
    -------
    observations : np.ndarray
        State observations.

    actions : np.ndarray
        Actions taken in each state.

    costs : np.ndarray
        Safety costs associated with each transition.

    terminals : np.ndarray
        Terminal-transition indicators.

    timeouts : np.ndarray
        Episode-timeout indicators.
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

        observations = np.asarray(
            file["observations"][:],
            dtype=np.float32,
        )

        actions = np.asarray(
            file["actions"][:],
            dtype=np.float32,
        )

        costs = np.asarray(
            file["costs"][:],
            dtype=np.float32,
        ).reshape(-1)

        terminals = np.asarray(
            file["terminals"][:],
            dtype=bool,
        ).reshape(-1)

        timeouts = np.asarray(
            file["timeouts"][:],
            dtype=bool,
        ).reshape(-1)

    num_transitions = len(observations)

    if not (
        len(actions)
        == len(costs)
        == len(terminals)
        == len(timeouts)
        == num_transitions
    ):
        raise ValueError(
            "All loaded DSRL arrays must contain "
            "the same number of transitions."
        )

    return (
        observations,
        actions,
        costs,
        terminals,
        timeouts,
    )


def episode_train_validation_split(
    terminals,
    timeouts,
    train_fraction,
):
    """
    Find a train/validation split located exactly on an episode boundary.

    Splitting by complete episodes prevents transitions from the same episode
    from appearing in both the training and validation datasets.

    Parameters
    ----------
    terminals : np.ndarray
        Terminal indicators.

    timeouts : np.ndarray
        Timeout indicators.

    train_fraction : float
        Fraction of complete episodes used for training.

    Returns
    -------
    train_end : int
        End-exclusive transition index of the training dataset.

    train_episodes : int
        Number of training episodes.

    validation_episodes : int
        Number of validation episodes.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train_fraction must be strictly between 0 and 1"
        )

    terminals = np.asarray(
        terminals,
        dtype=bool,
    ).reshape(-1)

    timeouts = np.asarray(
        timeouts,
        dtype=bool,
    ).reshape(-1)

    boundaries = terminals | timeouts

    # If transition i ends an episode, i + 1 is the end-exclusive
    # array index of that episode.
    episode_ends = (
        np.flatnonzero(boundaries) + 1
    ).astype(np.int64)

    num_transitions = len(boundaries)

    # Some datasets may not explicitly mark the final transition.
    # Treat the end of the dataset as the final episode boundary.
    if (
        len(episode_ends) == 0
        or episode_ends[-1] != num_transitions
    ):
        episode_ends = np.concatenate(
            [
                episode_ends,
                np.array(
                    [num_transitions],
                    dtype=np.int64,
                ),
            ]
        )

    num_episodes = len(episode_ends)

    if num_episodes < 2:
        raise ValueError(
            "At least two episodes are required for "
            "a train/validation split."
        )

    train_episodes = int(
        train_fraction * num_episodes
    )

    # Ensure both datasets contain at least one episode.
    train_episodes = min(
        max(train_episodes, 1),
        num_episodes - 1,
    )

    train_end = int(
        episode_ends[train_episodes - 1]
    )

    validation_episodes = (
        num_episodes - train_episodes
    )

    return (
        train_end,
        train_episodes,
        validation_episodes,
    )


def resolve_positive_weight(
    value,
    train_targets,
):
    """
    Determine the positive-class weight for BCEWithLogitsLoss.

    Parameters
    ----------
    value : str
        Either a numeric value such as ``"1.0"`` or ``"auto"``.

    train_targets : np.ndarray
        Binary training labels.

    Returns
    -------
    float
        Positive-class weight.

    Notes
    -----
    A weight of 1.0 corresponds to ordinary binary cross-entropy and is
    recommended initially because the model output is intended to represent:

        P(violation within K steps | s, a)

    ``auto`` uses:

        number of negative labels / number of positive labels

    This can help with severe class imbalance, but class weighting can affect
    probability calibration.
    """

    if value.lower() == "auto":

        positives = float(
            np.sum(train_targets == 1.0)
        )

        negatives = float(
            np.sum(train_targets == 0.0)
        )

        if positives == 0.0:
            raise ValueError(
                "Cannot compute automatic positive-class "
                "weight because the training set contains "
                "no positive labels."
            )

        return negatives / positives

    weight = float(value)

    if weight <= 0.0:
        raise ValueError(
            "pos_weight must be greater than 0"
        )

    return weight


def evaluate_model(
    model,
    states,
    actions,
    targets,
    batch_size,
    device,
):
    """
    Evaluate the risk predictor on the validation dataset.

    Validation uses unweighted BCE even when weighted BCE is used for
    training. This keeps validation focused on the quality of the predicted
    probabilities.

    The Brier score is also reported:

        mean((predicted_probability - target)^2)

    which is useful for probability-based models.

    Returns
    -------
    dict
        BCE, Brier score, accuracy, precision, recall, and F1.
    """

    model.eval()

    loss_fn = nn.BCEWithLogitsLoss(
        reduction="sum"
    )

    total_bce = 0.0
    total_brier = 0.0
    total_count = 0

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    with torch.no_grad():

        for start in range(
            0,
            len(states),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(states),
            )

            state = torch.from_numpy(
                states[start:end]
            ).to(device)

            action = torch.from_numpy(
                actions[start:end]
            ).to(device)

            target = torch.from_numpy(
                targets[start:end]
            ).to(device)

            logits = model(
                state,
                action,
            )

            probabilities = torch.sigmoid(
                logits
            )

            total_bce += loss_fn(
                logits,
                target,
            ).item()

            total_brier += torch.sum(
                (probabilities - target) ** 2
            ).item()

            total_count += len(state)

            # A threshold of 0.5 is only used here for reporting
            # classification metrics. The Safe-POR safety threshold
            # is a separate hyperparameter.
            predicted = (
                probabilities >= 0.5
            )

            target_bool = (
                target >= 0.5
            )

            true_positive += torch.sum(
                predicted & target_bool
            ).item()

            false_positive += torch.sum(
                predicted & ~target_bool
            ).item()

            true_negative += torch.sum(
                ~predicted & ~target_bool
            ).item()

            false_negative += torch.sum(
                ~predicted & target_bool
            ).item()

    if total_count == 0:
        raise ValueError(
            "Validation set is empty"
        )

    accuracy = (
        true_positive + true_negative
    ) / total_count

    precision_denominator = (
        true_positive + false_positive
    )

    recall_denominator = (
        true_positive + false_negative
    )

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )

    return {
        "bce": total_bce / total_count,
        "brier": total_brier / total_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_argument_parser():
    """
    Construct command-line arguments for risk-model training.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train a K-step safety-violation "
            "risk predictor for Safe-POR."
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
            "and receiving risk_predictor.pt."
        ),
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help=(
            "K in the K-step target. A state-action pair is "
            "labelled unsafe if a cost occurs in "
            "c_t through c_{t+K-1}."
        ),
    )

    parser.add_argument(
        "--cost-threshold",
        type=float,
        default=0.0,
        help=(
            "A transition is considered a safety violation "
            "when cost > this value."
        ),
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help=(
            "Fraction of complete episodes used for training."
        ),
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help=(
            "Width of each hidden layer in RiskPredictor."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--pos-weight",
        type=str,
        default="1.0",
        help=(
            "Positive-class weight for BCEWithLogitsLoss. "
            "Use 1.0 for ordinary probabilistic training, "
            "or 'auto' to use N_negative / N_positive."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser


def main():

    parser = build_argument_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate command-line arguments.
    # ------------------------------------------------------------------

    if args.horizon < 1:
        parser.error(
            "--horizon must be at least 1"
        )

    if args.hidden_dim < 1:
        parser.error(
            "--hidden-dim must be at least 1"
        )

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be at least 1"
        )

    if args.epochs < 1:
        parser.error(
            "--epochs must be at least 1"
        )

    if args.learning_rate <= 0.0:
        parser.error(
            "--learning-rate must be greater than 0"
        )

    set_seed(args.seed)

    dataset_path = (
        args.dataset_path.resolve()
    )

    run_dir = (
        args.run_dir.resolve()
    )

    normalization_path = (
        run_dir / "normalization.npz"
    )

    model_path = (
        run_dir / "risk_predictor.pt"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"DSRL dataset not found: {dataset_path}"
        )

    if not normalization_path.exists():
        raise FileNotFoundError(
            "POR normalization file not found. "
            "Train POR first or provide the correct --run-dir.\n"
            f"Expected: {normalization_path}"
        )

    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset_path}")
    print(f"POR run directory: {run_dir}")
    print(f"K-step horizon: {args.horizon}")
    print()

    # ------------------------------------------------------------------
    # Load DSRL data.
    # ------------------------------------------------------------------

    print("Loading DSRL dataset...")

    (
        observations,
        actions,
        costs,
        terminals,
        timeouts,
    ) = load_data(
        dataset_path
    )

    print(
        f"Loaded {len(observations)} transitions."
    )

    # ------------------------------------------------------------------
    # Create K-step risk labels.
    #
    # For transition t:
    #
    # y_t = 1
    #
    # if any violation occurs in:
    #
    # c_t, ..., c_{t+K-1}
    #
    # risk_targets.py ensures that this window never crosses an episode
    # boundary.
    # ------------------------------------------------------------------

    print(
        "Building K-step violation labels..."
    )

    targets = k_step_violation_labels(
        costs=costs,
        terminals=terminals,
        timeouts=timeouts,
        horizon=args.horizon,
        cost_threshold=args.cost_threshold,
    )

    # ------------------------------------------------------------------
    # Split into training and validation data by complete episodes.
    # ------------------------------------------------------------------

    (
        train_end,
        train_episodes,
        validation_episodes,
    ) = episode_train_validation_split(
        terminals=terminals,
        timeouts=timeouts,
        train_fraction=args.train_fraction,
    )

    # ------------------------------------------------------------------
    # Apply exactly the same state normalization that POR uses.
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

    if (
        observations.shape[1] != len(mean)
        or len(mean) != len(std)
    ):
        raise ValueError(
            "Observation dimension does not match "
            "the POR normalization file."
        )

    # Normalize in place to reduce memory use.
    observations -= mean
    observations /= std

    # ------------------------------------------------------------------
    # Construct train/validation splits.
    # ------------------------------------------------------------------

    train_states = (
        observations[:train_end]
    )

    train_actions = (
        actions[:train_end]
    )

    train_targets = (
        targets[:train_end]
    )

    validation_states = (
        observations[train_end:]
    )

    validation_actions = (
        actions[train_end:]
    )

    validation_targets = (
        targets[train_end:]
    )

    train_positive_rate = float(
        np.mean(train_targets)
    )

    validation_positive_rate = float(
        np.mean(validation_targets)
    )

    print(
        f"Training episodes: {train_episodes}"
    )

    print(
        f"Validation episodes: "
        f"{validation_episodes}"
    )

    print(
        f"Training transitions: "
        f"{len(train_states)}"
    )

    print(
        f"Validation transitions: "
        f"{len(validation_states)}"
    )

    print(
        "Training positive-label rate: "
        f"{train_positive_rate:.4%}"
    )

    print(
        "Validation positive-label rate: "
        f"{validation_positive_rate:.4%}"
    )

    print()

    # ------------------------------------------------------------------
    # Training DataLoader.
    # ------------------------------------------------------------------

    train_dataset = TensorDataset(
        torch.from_numpy(train_states),
        torch.from_numpy(train_actions),
        torch.from_numpy(train_targets),
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    # ------------------------------------------------------------------
    # Build the risk model.
    # ------------------------------------------------------------------

    state_dim = (
        train_states.shape[1]
    )

    action_dim = (
        train_actions.shape[1]
    )

    model = RiskPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # ------------------------------------------------------------------
    # Binary classification loss.
    #
    # RiskPredictor.forward() returns logits, so BCEWithLogitsLoss should
    # be used directly. Do NOT apply sigmoid before this loss.
    # ------------------------------------------------------------------

    positive_weight = (
        resolve_positive_weight(
            args.pos_weight,
            train_targets,
        )
    )

    pos_weight_tensor = torch.tensor(
        [positive_weight],
        dtype=torch.float32,
        device=DEVICE,
    )

    train_loss_fn = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor,
        )
    )

    print(
        "Training positive-class weight: "
        f"{positive_weight:.6f}"
    )

    if not np.isclose(
        positive_weight,
        1.0,
    ):
        print(
            "Note: positive-class weighting may improve "
            "minority-class learning, but can affect "
            "probability calibration."
        )

    print()
    print(
        "Starting risk-model training..."
    )

    # ------------------------------------------------------------------
    # Training.
    # ------------------------------------------------------------------

    best_validation_bce = float("inf")
    best_validation_metrics = None
    best_state = None
    best_epoch = None

    for epoch in range(
        args.epochs
    ):

        model.train()

        total_train_loss = 0.0
        train_count = 0

        for (
            state,
            action,
            target,
        ) in train_loader:

            state = state.to(
                DEVICE,
                non_blocking=True,
            )

            action = action.to(
                DEVICE,
                non_blocking=True,
            )

            target = target.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(
                state,
                action,
            )

            loss = train_loss_fn(
                logits,
                target,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()
            optimizer.step()

            total_train_loss += (
                loss.item() * len(state)
            )

            train_count += len(state)

        train_loss = (
            total_train_loss
            / train_count
        )

        # --------------------------------------------------------------
        # Validation.
        # --------------------------------------------------------------

        validation_metrics = evaluate_model(
            model=model,
            states=validation_states,
            actions=validation_actions,
            targets=validation_targets,
            batch_size=args.batch_size,
            device=DEVICE,
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train BCE={train_loss:.6f} | "
            f"val BCE={validation_metrics['bce']:.6f} | "
            f"Brier={validation_metrics['brier']:.6f} | "
            f"acc={validation_metrics['accuracy']:.4f} | "
            f"precision={validation_metrics['precision']:.4f} | "
            f"recall={validation_metrics['recall']:.4f} | "
            f"F1={validation_metrics['f1']:.4f}"
        )

        # --------------------------------------------------------------
        # Save the parameters with the lowest unweighted validation BCE.
        #
        # This is preferable to selecting based only on accuracy because
        # Safe-POR consumes the actual predicted probabilities.
        # --------------------------------------------------------------

        if (
            validation_metrics["bce"]
            < best_validation_bce
        ):

            best_validation_bce = (
                validation_metrics["bce"]
            )

            best_validation_metrics = (
                validation_metrics.copy()
            )

            best_epoch = epoch + 1

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

    # ------------------------------------------------------------------
    # Save best model.
    # ------------------------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "Training finished without producing "
            "a model checkpoint."
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": best_state,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,

            # Definition of the safety target.
            "horizon": args.horizon,
            "cost_threshold": (
                args.cost_threshold
            ),

            # Dataset split information.
            "train_fraction": (
                args.train_fraction
            ),
            "train_episodes": (
                train_episodes
            ),
            "validation_episodes": (
                validation_episodes
            ),
            "train_transitions": (
                len(train_states)
            ),
            "validation_transitions": (
                len(validation_states)
            ),

            # Useful for diagnosing class imbalance.
            "train_positive_rate": (
                train_positive_rate
            ),
            "validation_positive_rate": (
                validation_positive_rate
            ),
            "pos_weight": (
                positive_weight
            ),

            # Reproducibility and model selection.
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation_bce": (
                best_validation_bce
            ),
            "best_validation_brier": (
                best_validation_metrics[
                    "brier"
                ]
            ),
        },
        model_path,
    )

    print()
    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        "Best validation BCE: "
        f"{best_validation_bce:.6f}"
    )

    print(
        "Best validation Brier score: "
        f"{best_validation_metrics['brier']:.6f}"
    )

    print(
        f"Saved risk model to {model_path}"
    )


if __name__ == "__main__":
    main()