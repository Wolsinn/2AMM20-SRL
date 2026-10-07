import numpy as np


def discounted_cost_to_go(costs, terminals, timeouts, gamma=0.99):
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