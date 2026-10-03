import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POR_DIR = PROJECT_ROOT / "external" / "POR"

sys.path.insert(0, str(POR_DIR))

from policy import GaussianPolicy


RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "runs"
    / "por_dsrl_pointgoal"
)


DEVICE = torch.device("cpu")


ENV_NAME = "SafetyPointGoal1-v0"


def build_policies(obs_dim, act_dim):
    policy = GaussianPolicy(
        obs_dim + obs_dim,
        act_dim,
        hidden_dim=1024,
        n_hidden=2,
    )

    goal_policy = GaussianPolicy(
        obs_dim,
        obs_dim,
        hidden_dim=256,
        n_hidden=2,
    )

    return policy, goal_policy


def load_policies(obs_dim, act_dim):
    policy, goal_policy = build_policies(
        obs_dim,
        act_dim,
    )

    policy.load_state_dict(
        torch.load(
            RUN_DIR / "policy_network.pt",
            map_location=DEVICE,
        )
    )

    goal_policy.load_state_dict(
        torch.load(
            RUN_DIR / "goal_network.pt",
            map_location=DEVICE,
        )
    )

    policy.eval()
    goal_policy.eval()

    return policy, goal_policy


def load_normalization():
    normalization = np.load(
        RUN_DIR / "normalization.npz"
    )

    mean = normalization["mean"]
    std = normalization["std"]

    return mean, std


def normalize_observation(observation, mean, std):
    return (
        observation - mean
    ) / std


def select_action(
    policy,
    goal_policy,
    observation,
    mean,
    std,
):
    observation = normalize_observation(
        observation,
        mean,
        std,
    )

    observation_tensor = torch.tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        goal = goal_policy(
            observation_tensor
        )

        action = policy(
            torch.cat(
                [
                    observation_tensor,
                    goal,
                ],
                dim=-1,
            )
        )

    action = action.squeeze(0).cpu().numpy()

    return action


def evaluate(num_episodes=5):
    env = gym.make(ENV_NAME)

    observation_space = env.observation_space
    action_space = env.action_space

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")

    mean, std = load_normalization()

    print("Loaded normalization statistics.")

    policy, goal_policy = load_policies(
        obs_dim,
        act_dim,
    )

    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    episode_violations = []

    for episode in range(num_episodes):
        observation, info = env.reset()

        total_reward = 0.0
        total_cost = 0.0
        steps = 0

        while True:
            action = select_action(
                policy,
                goal_policy,
                observation,
                mean,
                std,
            )

            action = np.clip(
                action,
                action_space.low,
                action_space.high,
            )

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += float(reward)

            cost = float(
                info.get("cost", 0.0)
            )

            total_cost += cost

            steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_lengths.append(steps)
        episode_violations.append(
            float(total_cost > 0)
        )

        print(
            f"Episode {episode + 1}: "
            f"reward={total_reward:.3f}, "
            f"cost={total_cost:.3f}, "
            f"length={steps}"
        )

    env.close()

    print()
    print("Evaluation results")
    print("------------------")

    print(
        f"Mean reward: "
        f"{np.mean(episode_rewards):.3f}"
    )

    print(
        f"Mean cost: "
        f"{np.mean(episode_costs):.3f}"
    )

    print(
        f"Violation rate: "
        f"{np.mean(episode_violations):.3f}"
    )

    print(
        f"Mean episode length: "
        f"{np.mean(episode_lengths):.1f}"
    )


if __name__ == "__main__":
    evaluate(
        num_episodes=5
    )