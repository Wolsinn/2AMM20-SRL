from pathlib import Path
import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyCarButton1Gymnasium-v0-250-2656.hdf5"

CHUNK_SIZE = 10_000


def main():
    with h5py.File(DATASET_PATH, "r") as f:

        observations = f["observations"]
        next_observations = f["next_observations"]
        terminals = f["terminals"]
        timeouts = f["timeouts"]

        n = len(observations)

        print(f"Transitions: {n}")
        print(f"Terminal transitions: {np.count_nonzero(terminals)}")
        print(f"Timeout transitions: {np.count_nonzero(timeouts)}")

        normal_checked = 0
        normal_matches = 0

        for start in range(0, n - 1, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n - 1)

            current_next = next_observations[start:end]
            following_obs = observations[start + 1:end + 1]
            following_timeouts = timeouts[start:end]

            normal_mask = ~following_timeouts.astype(bool)

            if not np.any(normal_mask):
                continue

            matches = np.isclose(
                current_next[normal_mask],
                following_obs[normal_mask],
                atol=1e-6,
            ).all(axis=1)

            normal_checked += len(matches)
            normal_matches += int(matches.sum())

        print("\nNormal transition consistency")
        print("--------------------------------")
        print(f"Checked: {normal_checked}")

        if normal_checked > 0:
            print(f"Matches: {normal_matches}")
            print(f"Match rate: {normal_matches / normal_checked:.6f}")


        timeout_indices = np.flatnonzero(
            np.asarray(timeouts[:], dtype=bool)
        )

        print("\nTimeout boundaries")
        print("------------------")

        for idx in timeout_indices[:10]:

            stored_next_obs = next_observations[idx]
            next_episode_obs = observations[idx + 1]

            same = np.isclose(stored_next_obs, next_episode_obs, atol=1e-6).all()

            print(
                f"transition={idx:7d} | "
                f"next_obs == next_episode_first_obs: "
                f"{same}"
            )


        episode_lengths = np.diff(
            np.concatenate(
                [
                    np.array([-1]),
                    timeout_indices,
                ]
            )
        )

        print("\nEpisode lengths")
        print("----------------")
        print(f"Number of episodes: {len(episode_lengths)}")
        print(f"Unique lengths: {np.unique(episode_lengths)}")
        print(f"Minimum length: {episode_lengths.min()}")
        print(f"Maximum length: {episode_lengths.max()}")


if __name__ == "__main__":
    main()