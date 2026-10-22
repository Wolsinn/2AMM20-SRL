import numpy as np


def episode_ranges(terminals, timeouts):
    terminals = np.asarray(terminals, dtype=bool)
    timeouts = np.asarray(timeouts, dtype=bool)
    if terminals.ndim != 1 or timeouts.shape != terminals.shape:
        raise ValueError("terminals and timeouts must be matching 1D arrays")
    if len(terminals) == 0:
        return []

    ends = np.flatnonzero(terminals | timeouts)

    if ends.size == 0 or ends[-1] != len(terminals) - 1:
        ends = np.append(ends, len(terminals) - 1)

    starts = np.concatenate(([0], ends[:-1] + 1))

    return [(int(start), int(end) + 1) for start, end in zip(starts, ends)]


def stratified_episode_split(costs, terminals, timeouts, validation_fraction=0.2, seed=42):
    costs = np.asarray(costs, dtype=np.float32)
    ranges = episode_ranges(terminals, timeouts)

    if costs.ndim != 1 or (ranges and ranges[-1][1] != len(costs)):
        raise ValueError("costs, terminals, and timeouts must have matching lengths")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(ranges) < 2:
        raise ValueError("At least two episodes are required for a split")

    episode_costs = np.asarray(
        [costs[start:end].sum(dtype=np.float64) for start, end in ranges]
    )
    positive = episode_costs > 0
    strata = np.zeros(len(ranges), dtype=np.int64)

    if np.any(positive):
        quantiles = np.quantile(episode_costs[positive], [0.25, 0.5, 0.75])
        strata[positive] = 1 + np.searchsorted(
            quantiles,
            episode_costs[positive],
            side="right",
        )

    rng = np.random.default_rng(seed)
    train_ids = []
    validation_ids = []

    for stratum in np.unique(strata):
        ids = rng.permutation(np.flatnonzero(strata == stratum))

        if len(ids) < 2:
            train_ids.extend(ids.tolist())
            continue

        validation_count = int(round(len(ids) * validation_fraction))
        validation_count = min(max(validation_count, 1), len(ids) - 1)
        validation_ids.extend(ids[:validation_count].tolist())
        train_ids.extend(ids[validation_count:].tolist())

    if not train_ids or not validation_ids:
        raise ValueError("Could not create non-empty train and validation splits")

    return (
        [ranges[index] for index in sorted(train_ids)],
        [ranges[index] for index in sorted(validation_ids)],
    )


def concatenate_episode_ranges(values, ranges):
    if not ranges:
        raise ValueError("Cannot concatenate an empty set of episode ranges")

    return np.concatenate([np.asarray(values[start:end]) for start, end in ranges], axis=0)


def discounted_cost_to_go(costs, terminals, timeouts, gamma=0.3):
    """
    Calculate discounted cost-to-go separately for every episode.

    An episode boundary occurs when either
    a terminal state or a timeout is reached.

    G_t = c_t + gamma * G_{t+1}
    """

    costs     = np.asarray(costs, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=bool)
    timeouts  = np.asarray(timeouts, dtype=bool)

    boundaries = terminals | timeouts

    targets = np.zeros_like(costs)

    running = 0.0

    for t in reversed(range(len(costs))):

        running = costs[t] + gamma * running
        targets[t] = running

        if boundaries[t]:
            running = 0.0

    return targets

# ADDED CHANGES
def stratified_split_ranges(costs, ranges, holdout_fraction, seed):
    """
    Split a list of episode ranges into (kept, holdout), stratified by episode cost.
    Strata: zero-cost episodes, plus quartiles of episode cost among episodes with cost.
    """
    costs = np.asarray(costs, dtype=np.float32)

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if len(ranges) < 2:
        raise ValueError("At least two episodes are required for a split")

    episode_costs = np.asarray(
        [costs[start:end].sum(dtype=np.float64) for start, end in ranges]
    )
    positive = episode_costs > 0
    strata = np.zeros(len(ranges), dtype=np.int64)

    if np.any(positive):
        quantiles = np.quantile(episode_costs[positive], [0.25, 0.5, 0.75])
        strata[positive] = 1 + np.searchsorted(
            quantiles, episode_costs[positive], side="right"
        )

    rng = np.random.default_rng(seed)
    kept_ids = []
    holdout_ids = []

    for stratum in np.unique(strata):
        ids = rng.permutation(np.flatnonzero(strata == stratum))

        if len(ids) < 2:
            kept_ids.extend(ids.tolist())
            continue

        holdout_count = int(round(len(ids) * holdout_fraction))
        holdout_count = min(max(holdout_count, 1), len(ids) - 1)
        holdout_ids.extend(ids[:holdout_count].tolist())
        kept_ids.extend(ids[holdout_count:].tolist())

    if not kept_ids or not holdout_ids:
        raise ValueError("Could not create non-empty splits")

    return (
        [ranges[index] for index in sorted(kept_ids)],
        [ranges[index] for index in sorted(holdout_ids)],
    )


def three_way_episode_split(
    costs, terminals, timeouts,
    validation_fraction=0.10, test_fraction=0.20,
    test_seed=42, split_seed=0,
):
    """
    Returns (train, validation, test) episode ranges.

    The test set is drawn first with a FIXED seed, so it is identical for every
    run. Train/validation are then split from the remaining episodes with
    `split_seed`, which varies per run.
    """
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    ranges = episode_ranges(terminals, timeouts)
    if ranges and ranges[-1][1] != len(costs):
        raise ValueError("costs, terminals, and timeouts must have matching lengths")

    trainval_ranges, test_ranges = stratified_split_ranges(
        costs, ranges, test_fraction, test_seed
    )
    train_ranges, validation_ranges = stratified_split_ranges(
        costs,
        trainval_ranges,
        validation_fraction / (1.0 - test_fraction),  # 0.10 / 0.80 -> 10% of all episodes
        split_seed,
    )
    return train_ranges, validation_ranges, test_ranges