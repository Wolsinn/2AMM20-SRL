import os
import gymnasium as gym
import dsrl


ENV_NAME = "OfflinePointGoal1-v0"


def main():
    print("DSRL dataset directory:")
    print(os.environ.get("DSRL_DATASET_DIR"))

    env = gym.make(ENV_NAME)

    dataset = env.get_dataset()

    print()
    print("Environment:", ENV_NAME)
    print()

    for key, value in dataset.items():
        if hasattr(value, "shape"):
            print(f"{key:20s} shape={value.shape} dtype={value.dtype}")
        else:
            print(f"{key:20s} type={type(value)}")

    env.close()


if __name__ == "__main__":
    main()