import sys
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import trange


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POR_DIR = PROJECT_ROOT / "external" / "POR"

sys.path.insert(0, str(POR_DIR))

from por import POR
from policy import GaussianPolicy
from value_functions import TwinV
from util import sample_batch, torchify


# DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyCarButton1Gymnasium-v0-250-2656.hdf5"
# RUN_DIR = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_pointgoal"
RUN_DIR = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_carbutton1"

DEVICE = torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_por_terminals(dataset):
    terminals = dataset["terminals"]
    timeouts = dataset["timeouts"]

    dataset["terminals"] = np.maximum(terminals, timeouts)

    return dataset


def load_dataset(max_transitions=None):
    import h5py

    dataset = {}

    with h5py.File(DATASET_PATH, "r") as file:
        total = len(file["observations"])
        n = total if max_transitions is None else min(max_transitions, total)

        dataset["observations"] = np.asarray(
            file["observations"][:n],
            dtype=np.float32
        )

        dataset["actions"] = np.asarray(
            file["actions"][:n],
            dtype=np.float32
        )

        dataset["next_observations"] = np.asarray(
            file["next_observations"][:n],
            dtype=np.float32
        )

        dataset["rewards"] = np.asarray(
            file["rewards"][:n],
            dtype=np.float32
        )

        dataset["terminals"] = np.asarray(
            file["terminals"][:n],
            dtype=np.float32
        )

        dataset["timeouts"] = np.asarray(
            file["timeouts"][:n],
            dtype=np.float32,
        )

        dataset["costs"] = np.asarray(
            file["costs"][:n],
            dtype=np.float32,
        )

    return dataset


def prepare_dataset(dataset):
    required_keys = ["observations", "actions", "next_observations", "rewards", "terminals",]

    for key in required_keys:
        if key not in dataset:
            raise KeyError(f"Required dataset field '{key}' is missing.")

    obs_dim = dataset["observations"].shape[1]

    mean = dataset["observations"].mean(axis=0)
    std = dataset["observations"].std(axis=0) + 1e-3

    dataset["observations"] -= mean
    dataset["observations"] /= std

    dataset["next_observations"] -= mean
    dataset["next_observations"] /= std

    for key, value in dataset.items():
        if key == "costs":
            continue

        dataset[key] = torchify(value)

    return dataset, mean, std, obs_dim


def build_por(obs_dim, act_dim, train_steps):
    policy = GaussianPolicy(
        obs_dim + obs_dim,
        act_dim,
        hidden_dim=1024,
        n_hidden=2,
    )

    goal_policy = GaussianPolicy(
        obs_dim,
        obs_dim,
        hidden_dim=256,
        n_hidden=2,
    )

    value_function = TwinV(
        obs_dim,
        layer_norm=False,
        hidden_dim=256,
        n_hidden=2,
    )

    por = POR(
        vf=value_function,
        policy=policy,
        goal_policy=goal_policy,
        max_steps=train_steps,
        tau=0.9,
        alpha=10.0,
        discount=0.99,
        value_lr=1e-4,
        policy_lr=1e-4,
    )

    return por, policy, goal_policy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-transitions", type=int, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    train_steps = args.steps
    batch_size = 256

    print("Loading DSRL dataset...")

    dataset = load_dataset(max_transitions=args.max_transitions)
    dataset = prepare_por_terminals(dataset)

    print(
        f"Loaded {len(dataset['observations'])} transitions."
    )

    dataset, mean, std, obs_dim = prepare_dataset(dataset)

    act_dim = dataset["actions"].shape[1]

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")
    print("Building POR...")

    por, policy, goal_policy = build_por(obs_dim, act_dim, train_steps)

    print("Starting POR training...")

    for step in trange(train_steps):
        batch = sample_batch(dataset, batch_size)

        por.por_residual_update(
            observations=batch["observations"],
            actions=batch["actions"],
            next_observations=batch["next_observations"],
            rewards=batch["rewards"],
            terminals=batch["terminals"],
        )

    print("Training finished.")

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        policy.state_dict(),
        RUN_DIR / "policy_network.pt",
    )

    torch.save(
        goal_policy.state_dict(),
        RUN_DIR / "goal_network.pt",
    )

    np.savez(
        RUN_DIR / "normalization.npz",
        mean=mean,
        std=std,
    )

    print(f"Saved trained models to {RUN_DIR}")


if __name__ == "__main__":
    main()