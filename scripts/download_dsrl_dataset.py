import os
from pathlib import Path
import h5py
import gymnasium as gym
import dsrl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "dsrl"
# DATASET_FILENAME = "SafetyCarButton1Gymnasium-v0-250-2656.hdf5"
DATASET_FILENAME = "SafetyCarButton2Gymnasium-v0-300-3755.hdf5"
# ENV_NAME = "OfflineCarButton1-v0"
ENV_NAME = "OfflineCarButton2-v0"

def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Tell DSRL where to store datasets, if supported by this installation.
    os.environ["DSRL_DATASET_DIR"] = str(DATASET_DIR)


    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Environment: {ENV_NAME}")

    env = gym.make(ENV_NAME)

    try:
        dataset = env.get_dataset()

        print("\nDataset contents:")
        for key, value in dataset.items():
            if hasattr(value, "shape"):
                print(
                    f"{key:20s} "
                    f"shape={value.shape} "
                    f"dtype={value.dtype}"
                )

        output_path = DATASET_DIR / DATASET_FILENAME

        with h5py.File(output_path, "w") as f:
            for key, value in dataset.items():
                f.create_dataset(
                    key,
                    data=value,
                    compression="gzip",
                )

        print(f"\nDataset saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GiB")

    finally:
        env.close()


if __name__ == "__main__":
    main()