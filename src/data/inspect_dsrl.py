import numpy as np


def print_dataset_statistics(dataset):
    print("Dataset statistics")
    print("===================")

    for key, value in dataset.items():
        if not isinstance(value, np.ndarray):
            continue

        print()
        print(key)
        print("  shape:", value.shape)
        print("  dtype:", value.dtype)

        if np.issubdtype(value.dtype, np.number):
            print("  min:", np.min(value))
            print("  max:", np.max(value))
            print("  mean:", np.mean(value))
            print("  std:", np.std(value))


def count_costs(dataset):
    costs = dataset["costs"]

    print()
    print("Safety statistics")
    print("=================")
    print("Total transitions:", len(costs))
    print("Transitions with cost > 0:", np.sum(costs > 0))
    print("Fraction with cost > 0:", np.mean(costs > 0))
    print("Total recorded cost:", np.sum(costs))


def main():
    import gymnasium as gym
    import dsrl

    env_name = "OfflinePointGoal1-v0"

    env = gym.make(env_name)
    dataset = env.get_dataset()

    print_dataset_statistics(dataset)
    count_costs(dataset)

    env.close()


if __name__ == "__main__":
    main()