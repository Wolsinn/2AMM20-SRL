from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DATASET_PATH = Path(PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5")
# DATASET_PATH = Path(PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyCarButton1Gymnasium-v0-250-2656.hdf5")
DATASET_PATH = Path(PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyCarButton2Gymnasium-v0-300-3755.hdf5")
def load_dsrl_dataset(dataset_path=DATASET_PATH):
    dataset = {}

    with h5py.File(dataset_path, "r") as file:
        for key in file.keys():
            dataset[key] = np.array(file[key])

    required_keys = [ "observations", "actions", "next_observations", "rewards", "terminals", ]

    for key in required_keys:
        if key not in dataset:
            raise KeyError(f"Required dataset field '{key}' is missing.")

    return dataset


def print_dataset_summary(dataset):
    print("DSRL dataset")
    print("=" * 50)

    for key, value in dataset.items():
        print(
            f"{key:20s}"
            f"shape={value.shape}"
            f" dtype={value.dtype}"
        )

    print()


if __name__ == "__main__":
    dataset = load_dsrl_dataset()
    print_dataset_summary(dataset)