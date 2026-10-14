from pathlib import Path
import sys
import argparse
import csv
import numpy as np
import torch
import datetime as dt
import safety_gymnasium
import gymnasium
import dsrl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POR_DIR = PROJECT_ROOT / "external" / "POR"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

sys.path.insert(0, str(POR_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from policy import GaussianPolicy
from safety.cost_model import CostPredictor
from safety.safe_action import SafeActionCorrector


RUN_DIR = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_pointgoal"

DEVICE = torch.device("cpu")

ENV_NAME = "SafetyPointGoal1-v0"
SCORE_ENV_NAME = "OfflinePointGoal1Gymnasium-v0"  


def load_cost_model():
    checkpoint = torch.load(
        RUN_DIR / "cost_predictor.pt",
        map_location="cpu",
    )

    model = CostPredictor(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
    )

    model.load_state_dict(checkpoint["model"])

    model.eval()

    return model


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
    policy, goal_policy = build_policies(obs_dim, act_dim)

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
    normalization = np.load(RUN_DIR / "normalization.npz")
    mean = normalization["mean"]
    std = normalization["std"]

    return mean, std


def select_action(policy, goal_policy, observation, mean, std, corrector=None, sampling_seed=None):

    normalized_observation = (observation - mean) / std

    observation_tensor = torch.tensor(
        normalized_observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        goal = goal_policy.act(
            observation_tensor,
            deterministic=True,
        )

        policy_input = torch.cat(
            [
                observation_tensor,
                goal,
            ],
            dim=-1,
        )

        proposed_action = policy.act(
            policy_input,
            deterministic=True,
        )

    proposed_action = proposed_action.squeeze(0).cpu().numpy()

    if corrector is None:
        return proposed_action, None

    result = corrector.correct(
        observation=normalized_observation,
        proposed_action=proposed_action,
        sampling_seed=sampling_seed,
    )

    return result["action"], result


def evaluate(num_episodes=5, mode="por", risk_threshold=5.0, num_candidates=50, candidate_std=0.10, seed=0, target_cost=10):
    env = safety_gymnasium.make(ENV_NAME)
    score_env = gymnasium.make(SCORE_ENV_NAME)
    score_env.set_target_cost(target_cost)

    observation_space = env.observation_space
    action_space = env.action_space

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")

    mean, std = load_normalization()

    print("Loaded normalization statistics.")

    policy, goal_policy = load_policies(obs_dim, act_dim)

    corrector = None

    if mode == "safe":
        cost_model = load_cost_model()

        corrector = SafeActionCorrector(
            cost_model=cost_model,
            action_low=action_space.low,
            action_high=action_space.high,
            risk_threshold=risk_threshold,
            num_candidates=num_candidates,
            candidate_std=candidate_std,
            device=DEVICE,
        )

        print()
        print("Safety correction enabled")
        print(f"Risk threshold: {risk_threshold}")
        print(f"Candidate actions: {num_candidates}")
        print(f"Candidate std: {candidate_std}")

    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    episode_violations = []
    episode_norm_rewards = []
    episode_norm_costs = []
    episode_rows = []
    evaluation_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    total_corrected = 0
    total_safe_candidate_found = 0
    total_action_changed = 0
    correction_distances = []
    proposed_risks = []
    selected_risks = []

    total_unsafe_steps = 0
    total_steps = 0

    for episode in range(num_episodes):
        episode_seed = seed + episode
        observation, info = env.reset(seed=episode_seed)

        total_reward = 0.0
        total_cost = 0.0
        steps = 0
        unsafe_steps = 0

        while True:
            action, correction_info = select_action(
                policy,
                goal_policy,
                observation,
                mean,
                std,
                corrector=corrector,
                sampling_seed=int(
                    np.random.SeedSequence([seed, episode, steps]).generate_state(1)[0]
                ),
            )

            if correction_info is not None:
                proposed_risks.append(correction_info["proposed_risk"])
                selected_risks.append(correction_info["selected_risk"])
                if correction_info["search_attempted"]:
                    total_corrected += 1
                    total_safe_candidate_found += int(correction_info["safe_candidate_found"])
                    total_action_changed += int(correction_info["distance"] > 1e-8)
                    correction_distances.append(correction_info["distance"])

            action = np.clip(
                action,
                action_space.low,
                action_space.high,
            )

            observation, reward, cost, terminated, truncated, info = env.step(action)
            
            total_steps += 1

            if cost > 0:
                unsafe_steps += 1
                total_unsafe_steps += 1

            total_reward += float(reward)
            total_cost += float(cost)
            steps += 1

            if terminated or truncated:
                break

        norm_reward, norm_cost = score_env.get_normalized_score(total_reward, total_cost)
        episode_norm_rewards.append(norm_reward)
        episode_norm_costs.append(norm_cost)
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_lengths.append(steps)
        episode_violations.append(float(total_cost > 0))
        episode_rows.append(
            {
                "mode": mode,
                "seed": seed,
                "episode": episode,
                "episode_seed": episode_seed,
                "risk_threshold": risk_threshold if mode == "safe" else "",
                "num_candidates": num_candidates if mode == "safe" else "",
                "candidate_std": candidate_std if mode == "safe" else "",
                "reward": total_reward,
                "cost": total_cost,
                "unsafe_steps": unsafe_steps,
                "length": steps,
                "violated": int(total_cost > 0),
                "target_cost": target_cost,
                "normalized_reward": norm_reward,
                "normalized_cost": norm_cost,
                "within_target": int(norm_cost <= 1.0),
            }
        )

        print(
            f"Episode {episode + 1}: "
            f"reward={total_reward:.3f}, "
            f"cost={total_cost:.3f}, "
            f"length={steps}"
        )

    env.close()
    score_env.close()

    csv_path = RESULTS_DIR / f"por_dsrl_{mode}_seed{seed}_{evaluation_id}.csv"

    with csv_path.open("x", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    print(f"CSV_OUTPUT={csv_path}")

    timestep_violation_rate = (
        total_unsafe_steps / total_steps
        if total_steps > 0
        else 0.0
    )

    episode_violation_rate = np.mean( episode_violations )

    print("\nEvaluation results")
    print("------------------")
    print(f"Mean reward: {np.mean(episode_rewards):.3f}")
    print(f"Mean cost: {np.mean(episode_costs):.3f}")
    print(f"Normalized reward: {np.mean(episode_norm_rewards):.3f}")
    print(f"Normalized cost: {np.mean(episode_norm_costs):.3f} (target cost = {target_cost:g})")
    print(f"Episodes within cost target: {np.mean(np.array(episode_norm_costs) <= 1.0):.3f}")
    print(f"Episode violation rate: {episode_violation_rate:.3f}")
    print(f"Timestep violation rate: {timestep_violation_rate:.3f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")

    if corrector is not None:

        print("\nSafety correction")
        print("-----------------")
        print(f"Correction trigger rate: {total_corrected / total_steps:.3f}")
        print(f"Actions changed rate: {total_action_changed / total_steps:.3f}")
        print(f"Search attempts: {total_corrected}")

        if total_corrected:
            safe_search_rate = total_safe_candidate_found / total_corrected
            print(f"Safe candidate success (triggered searches): {safe_search_rate:.3f}")
            print(f"No-safe-candidate fallback rate: {1.0 - safe_search_rate:.3f}")

    if correction_distances:
        print(f"Mean correction distance: {np.mean(correction_distances):.6f}")
        print(f"Median correction distance: {np.median(correction_distances):.6f}")
        print(f"Mean proposed risk: {np.mean(proposed_risks):.6f}")
        print(f"Mean selected risk: {np.mean(selected_risks):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["por", "safe"], default="por")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--risk-threshold", type=float, default=5.0)
    parser.add_argument("--num-candidates", type=int, default=50)
    parser.add_argument("--candidate-std", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-cost", type=float, default=10.0)

    args = parser.parse_args()

    evaluate(
        num_episodes=args.episodes,
        mode=args.mode,
        risk_threshold=args.risk_threshold,
        num_candidates=args.num_candidates,
        candidate_std=args.candidate_std,
        seed=args.seed,
        target_cost=args.target_cost
    )