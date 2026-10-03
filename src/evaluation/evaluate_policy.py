import numpy as np
import safety_gymnasium


def evaluate_policy(policy, env_name, num_episodes=10):
    rewards = []
    costs = []
    successes = []

    for episode in range(num_episodes):
        env = safety_gymnasium.make(env_name)

        observation, info = env.reset(seed=episode)

        episode_reward = 0.0
        episode_cost = 0.0

        for _ in range(1000):
            action = policy(observation)

            observation, reward, cost, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_cost += cost

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        costs.append(episode_cost)

        success = episode_reward > 0
        successes.append(success)

        env.close()

    return {
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "cost_mean": np.mean(costs),
        "cost_std": np.std(costs),
        "success_rate": np.mean(successes),
    }