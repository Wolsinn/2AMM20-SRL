import numpy as np


def k_step_violation_labels(
    costs,
    terminals,
    timeouts,
    horizon,
    cost_threshold=0.0,
):
    """
    Construct binary K-step safety-violation labels.

    For transition t, the label is 1 if at least one safety violation occurs
    within the following horizon:

        c_t, c_{t+1}, ..., c_{t+horizon-1}

    Otherwise, the label is 0.

    The prediction window is truncated at the end of the current episode.
    Costs from a following episode are never used when constructing a label.

    The current cost c_t is included because action a_t causes the transition
    associated with c_t. Therefore the model learns:

        P(violation within K steps | s_t, a_t)

    Parameters
    ----------
    costs : array-like
        Safety cost for every transition.
        Expected shape is (N,) or (N, 1).

    terminals : array-like
        Boolean terminal indicator for every transition.

    timeouts : array-like
        Boolean timeout indicator for every transition.

    horizon : int
        Number of transitions included in the prediction window.

        horizon = 1:
            Only c_t is checked.

        horizon = 5:
            c_t through c_{t+4} are checked, unless the episode ends first.

    cost_threshold : float, optional
        A transition is considered a violation when:

            cost > cost_threshold

        Default is 0.0, so any positive cost is treated as a violation.

    Returns
    -------
    np.ndarray
        Float32 array of shape (N,).

        1.0 means at least one violation occurs within the horizon.
        0.0 means no violation occurs within the horizon.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    # Flatten the arrays so both (N,) and (N, 1) dataset formats work.
    costs = np.asarray(
        costs,
        dtype=np.float32,
    ).reshape(-1)

    terminals = np.asarray(
        terminals,
        dtype=bool,
    ).reshape(-1)

    timeouts = np.asarray(
        timeouts,
        dtype=bool,
    ).reshape(-1)

    # All arrays must describe the same transitions.
    if not (
        len(costs)
        == len(terminals)
        == len(timeouts)
    ):
        raise ValueError(
            "costs, terminals, and timeouts must contain "
            "the same number of transitions"
        )

    num_transitions = len(costs)

    labels = np.zeros(
        num_transitions,
        dtype=np.float32,
    )

    if num_transitions == 0:
        return labels

    # Convert raw costs to simple violation indicators.
    #
    # Example:
    #
    # costs       = [0, 0, 1, 0]
    # violations  = [F, F, T, F]
    #
    violations = costs > cost_threshold

    # An episode ends whenever either termination condition occurs.
    boundaries = terminals | timeouts

    # ---------------------------------------------------------------
    # Process the dataset one episode at a time.
    #
    # This is important because a state near the end of episode A
    # must never receive a positive label because of a violation in
    # episode B.
    # ---------------------------------------------------------------

    episode_start = 0

    boundary_indices = np.flatnonzero(boundaries)

    # Also treat the final dataset transition as an episode end in case
    # the dataset does not explicitly mark it as terminal/timeout.
    episode_ends = np.concatenate(
        [
            boundary_indices,
            np.array(
                [num_transitions - 1],
                dtype=np.int64,
            ),
        ]
    )

    # Avoid processing the final index twice when it is already a boundary.
    episode_ends = np.unique(episode_ends)

    for episode_end in episode_ends:

        if episode_end < episode_start:
            continue

        episode_violations = violations[
            episode_start : episode_end + 1
        ]

        episode_length = len(episode_violations)

        # -----------------------------------------------------------
        # Prefix sums allow each K-step query to be answered quickly.
        #
        # This avoids explicitly checking K future values for every
        # transition, which would be expensive for the ~2 million
        # transitions in the DSRL dataset.
        # -----------------------------------------------------------

        prefix = np.empty(
            episode_length + 1,
            dtype=np.int64,
        )

        prefix[0] = 0

        prefix[1:] = np.cumsum(
            episode_violations,
            dtype=np.int64,
        )

        # Local transition indices inside this episode.
        starts = np.arange(episode_length)

        # The prediction horizon may not extend beyond the episode.
        ends_exclusive = np.minimum(
            starts + horizon,
            episode_length,
        )

        # Number of violations in:
        #
        # [t, t + horizon)
        #
        violation_counts = (
            prefix[ends_exclusive]
            - prefix[starts]
        )

        labels[
            episode_start : episode_end + 1
        ] = (
            violation_counts > 0
        ).astype(np.float32)

        episode_start = episode_end + 1

        if episode_start >= num_transitions:
            break

    return labels