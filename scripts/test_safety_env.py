import safety_gymnasium


ENV_NAME = "SafetyPointGoal1-v0"


def main():
    env = safety_gymnasium.make(ENV_NAME)

    observation, info = env.reset(seed=42)

    print("Environment:", ENV_NAME)
    print("Observation shape:", observation.shape)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    total_reward = 0.0
    total_cost = 0.0

    for step in range(100):
        action = env.action_space.sample()

        observation, reward, cost, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_cost += cost

        print(
            f"step={step:03d} "
            f"reward={reward:.4f} "
            f"cost={cost}"
        )

        if terminated or truncated:
            break

    print()
    print("Episode finished")
    print("Steps:", step + 1)
    print("Total reward:", total_reward)
    print("Total cost:", total_cost)

    env.close()


if __name__ == "__main__":
    main()