from pathlib import Path
import sys
import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POR_DIR = PROJECT_ROOT / "external" / "POR"

sys.path.insert(0, str(POR_DIR))

from policy import GaussianPolicy


DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dsrl" / "SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
RUN_DIR = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_pointgoal"
RESULT_PATH = PROJECT_ROOT / "experiments" / "results" / "por_guide_analysis.csv"

DEVICE = torch.device("cpu")

BATCH_SIZE = 4096


def load_normalization():
    normalization = np.load(RUN_DIR / "normalization.npz")
    mean = normalization["mean"].astype(np.float32)
    std = normalization["std"].astype(np.float32)

    return mean, std


def load_goal_policy(obs_dim):
    goal_policy = GaussianPolicy(
        obs_dim,
        obs_dim,
        hidden_dim=256,
        n_hidden=2,
    )

    state_dict = torch.load(
        RUN_DIR / "goal_network.pt",
        map_location=DEVICE,
    )

    goal_policy.load_state_dict(state_dict)
    goal_policy.eval()

    return goal_policy


def update_group_statistics(values, costs, statistics):
    safe = costs == 0
    unsafe = costs > 0

    for name, mask in (("safe", safe), ("unsafe", unsafe)):

        group = values[mask]

        if len(group) == 0:
            continue

        stats = statistics[name]
        stats["count"] += len(group)
        stats["sum"] += float(group.sum())
        stats["sum_sq"] += float(np.square(group).sum())
        stats["values"].append(group)


def calculate_statistics(values):

    if not values:
        return None

    values = np.concatenate(values)

    return {
        "count":  len(values),
        "mean":   float(np.mean(values)),
        "std":    float(np.std(values)),
        "median": float(np.median(values)),
        "p90":    float(np.percentile(values, 90)),
        "p95":    float(np.percentile(values, 95)),
        "p99":    float(np.percentile(values, 99)),
    }


def main():
    mean, std = load_normalization()

    with h5py.File(DATASET_PATH, "r") as f:

        observations = f["observations"]
        next_observations = f["next_observations"]
        costs = f["costs"]

        dataset_size = len(observations)

        obs_dim = observations.shape[1]

        print(f"Dataset transitions: {dataset_size}")
        print(f"Observation dimension: {obs_dim}")

        goal_policy = load_goal_policy(obs_dim)

        safe_distances = []
        unsafe_distances = []
        all_distances = []

        # For correlation:
        sum_distance = 0.0
        sum_cost = 0.0
        sum_distance_sq = 0.0
        sum_cost_sq = 0.0
        sum_distance_cost = 0.0
        correlation_count = 0

        with torch.no_grad():

            for start in range(0, dataset_size, BATCH_SIZE):

                end = min(
                    start + BATCH_SIZE,
                    dataset_size,
                )

                obs = np.asarray(
                    observations[start:end],
                    dtype=np.float32,
                )

                next_obs = np.asarray(
                    next_observations[
                        start:end
                    ],
                    dtype=np.float32,
                )

                batch_costs = np.asarray(
                    costs[start:end],
                    dtype=np.float32,
                )

                obs = (obs - mean) / std

                next_obs = (next_obs - mean) / std

                obs_tensor = torch.from_numpy(obs)

                next_obs_tensor = (
                    torch.from_numpy(
                        next_obs
                    )
                )

                goals = goal_policy.act(
                    obs_tensor,
                    deterministic=True,
                )

                distances = (
                    torch.linalg.vector_norm(
                        goals - next_obs_tensor,
                        dim=1,
                    )
                    .numpy()
                )

                all_distances.append(distances)

                safe_mask = (batch_costs == 0)
                unsafe_mask = (batch_costs > 0)

                if safe_mask.any():
                    safe_distances.append(distances[safe_mask])

                if unsafe_mask.any():
                    unsafe_distances.append(distances[unsafe_mask])

                # Running correlation statistics
                d = distances.astype(np.float64)
                c = batch_costs.astype(np.float64)

                sum_distance += d.sum()
                sum_cost += c.sum()

                sum_distance_sq += (np.square(d).sum())
                sum_cost_sq += (np.square(c).sum())

                sum_distance_cost += (d * c).sum()

                correlation_count += len(d)

                if (start == 0 or (start // BATCH_SIZE) % 100 == 0):
                    print(
                        f"Processed "
                        f"{end:,}/{dataset_size:,}"
                    )

    print()
    print("Guide-state analysis")
    print("====================")

    all_stats    = calculate_statistics(all_distances)
    safe_stats   = calculate_statistics(safe_distances)
    unsafe_stats = calculate_statistics(unsafe_distances)

    print()
    print("Overall guide deviation")
    print("-----------------------")
    print(f"Count:  {all_stats['count']:,}")
    print(f"Mean:   {all_stats['mean']:.6f}")
    print(f"Std:    {all_stats['std']:.6f}")
    print(f"Median: {all_stats['median']:.6f}")
    print(f"P90:    {all_stats['p90']:.6f}")
    print(f"P95:    {all_stats['p95']:.6f}")
    print(f"P99:    {all_stats['p99']:.6f}")

    print()
    print("Safe transitions")
    print("----------------")
    print(f"Count:  {safe_stats['count']:,}")
    print(f"Mean:   {safe_stats['mean']:.6f}")
    print(f"Std:    {safe_stats['std']:.6f}")
    print(f"Median: {safe_stats['median']:.6f}")
    print(f"P90:    {safe_stats['p90']:.6f}")
    print(f"P95:    {safe_stats['p95']:.6f}")

    print()
    print("Costly transitions")
    print("------------------")
    print(f"Count:  {unsafe_stats['count']:,}")
    print(f"Mean:   {unsafe_stats['mean']:.6f}")
    print(f"Std:    {unsafe_stats['std']:.6f}")
    print(f"Median: {unsafe_stats['median']:.6f}")
    print(f"P90:    {unsafe_stats['p90']:.6f}")
    print(f"P95:    {unsafe_stats['p95']:.6f}")

    n = correlation_count

    numerator = n * sum_distance_cost - sum_distance * sum_cost

    denominator = np.sqrt(
        (n * sum_distance_sq - sum_distance**2)
        *
        (n * sum_cost_sq - sum_cost**2)
    )

    correlation = (
        numerator / denominator
        if denominator > 0
        else np.nan
    )

    print()
    print("Guide deviation / cost relationship")
    print("-----------------------------------")
    print(f"Pearson correlation: {correlation:.6f}")
    
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULT_PATH, "w") as file:
        file.write("group,count,mean,std,median,p90,p95,p99\n")

        for name, stats in ( ("all", all_stats), ("safe", safe_stats), ("unsafe", unsafe_stats) ):

            file.write(
                f"{name},"
                f"{stats['count']},"
                f"{stats['mean']},"
                f"{stats['std']},"
                f"{stats['median']},"
                f"{stats['p90']},"
                f"{stats['p95']},"
                f"{stats['p99']}\n"
            )

        file.write(
            f"\ncorrelation,"
            f"{correlation}\n"
        )

    print()
    print(f"Saved results to {RESULT_PATH}")


if __name__ == "__main__":
    main()