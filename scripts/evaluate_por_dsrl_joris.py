from pathlib import Path
import sys
import argparse

import numpy as np
import torch
import safety_gymnasium


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POR_DIR = PROJECT_ROOT / "external" / "POR"

sys.path.insert(0, str(POR_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from policy import GaussianPolicy
from safety.cost_model import CostPredictor
from safety.safe_action import SafeActionCorrector
from safety.risk_model import RiskPredictor
from safety.risk_safety_filter import RiskSafetyFilter


RUN_DIR = PROJECT_ROOT / "experiments" / "runs" / "por_dsrl_pointgoal"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_NAME = "SafetyPointGoal1-v0"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_cost_model():
    """Load the original discounted-cost safety model."""

    checkpoint = torch.load(
        RUN_DIR / "cost_predictor.pt",
        map_location=DEVICE,
    )

    model = CostPredictor(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(DEVICE)
    model.eval()

    return model


def load_risk_model():
    """Load the K-step violation-probability risk model."""

    checkpoint = torch.load(
        RUN_DIR / "risk_predictor.pt",
        map_location=DEVICE,
    )

    model = RiskPredictor(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(DEVICE)
    model.eval()

    return model


def build_policies(obs_dim, act_dim):
    """Construct the POR execute and guide policies."""

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
    """Load trained POR execute and guide policies."""

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

    policy.to(DEVICE)
    goal_policy.to(DEVICE)

    policy.eval()
    goal_policy.eval()

    return policy, goal_policy


def load_normalization():
    """Load the state normalization statistics used during POR training."""

    normalization = np.load(
        RUN_DIR / "normalization.npz"
    )

    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        normalization["std"],
        dtype=np.float32,
    )

    return mean, std


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def select_action(
    policy,
    goal_policy,
    observation,
    mean,
    std,
    corrector=None,
):
    """
    Generate a POR action and optionally pass it through a safety layer.

    POR first generates a desired next state using the guide policy:

        s -> g(s)

    The execute policy then generates the proposed action:

        (s, g(s)) -> a_POR

    If a safety corrector is provided, a_POR is evaluated and may be
    replaced before being sent to the environment.
    """

    normalized_observation = (
        observation - mean
    ) / std

    observation_tensor = torch.as_tensor(
        normalized_observation,
        dtype=torch.float32,
        device=DEVICE,
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

    proposed_action = (
        proposed_action
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    # POR baseline: execute the proposed action unchanged.
    if corrector is None:
        return proposed_action, None

    # Safe-POR: allow the selected safety layer to inspect/correct it.
    result = corrector.correct(
        observation=normalized_observation,
        proposed_action=proposed_action,
    )

    return result["action"], result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    num_episodes=5,
    mode="por",
    cost_threshold=1.0,
    risk_threshold=0.20,
    num_candidates=50,
    candidate_std=0.10,
    search_radius=0.10,
):
    """
    Evaluate POR or one of the two Safe-POR variants.

    Modes
    -----
    por
        Original POR without a safety layer.

    safe_cost
        Original cost-based safety layer:
        - CostPredictor
        - discounted future cost score
        - Gaussian candidate perturbations controlled by candidate_std

    safe_risk
        New K-step risk-based safety layer:
        - RiskPredictor
        - predicted probability of a violation within K steps
        - radius-bounded candidate search controlled by search_radius
    """

    env = safety_gymnasium.make(
        ENV_NAME
    )

    observation_space = env.observation_space
    action_space = env.action_space

    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    print(
        f"Observation dimension: {obs_dim}"
    )
    print(
        f"Action dimension: {act_dim}"
    )

    mean, std = load_normalization()

    print(
        "Loaded normalization statistics."
    )

    policy, goal_policy = load_policies(
        obs_dim,
        act_dim,
    )

    # ------------------------------------------------------------------
    # Configure the selected safety layer.
    # ------------------------------------------------------------------

    corrector = None

    if mode == "safe_cost":

        cost_model = load_cost_model()

        corrector = SafeActionCorrector(
            cost_model=cost_model,
            action_low=action_space.low,
            action_high=action_space.high,
            risk_threshold=cost_threshold,
            num_candidates=num_candidates,
            candidate_std=candidate_std,
            device=DEVICE,
        )

        print()
        print(
            "Cost-based safety correction enabled"
        )
        print(
            f"Cost threshold: {cost_threshold}"
        )
        print(
            f"Candidate actions: {num_candidates}"
        )
        print(
            f"Candidate std: {candidate_std}"
        )

    elif mode == "safe_risk":

        risk_model = load_risk_model()

        corrector = RiskSafetyFilter(
            risk_model=risk_model,
            action_low=action_space.low,
            action_high=action_space.high,
            risk_threshold=risk_threshold,
            num_candidates=num_candidates,
            search_radius=search_radius,
            device=DEVICE,
        )

        print()
        print(
            "K-step risk-based safety correction enabled"
        )
        print(
            f"Risk threshold: {risk_threshold}"
        )
        print(
            f"Candidate actions: {num_candidates}"
        )
        print(
            f"Search radius: {search_radius}"
        )

    else:

        print()
        print(
            "POR baseline: no safety correction"
        )

    # ------------------------------------------------------------------
    # Metrics.
    # ------------------------------------------------------------------

    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    episode_violations = []

    total_corrected = 0
    total_safe_candidate_found = 0

    # Only actual interventions contribute to correction distance.
    correction_distances = []

    # Safety scores are recorded for every evaluated step.
    proposed_scores = []
    selected_scores = []

    total_unsafe_steps = 0
    total_steps = 0

    # ------------------------------------------------------------------
    # Environment rollouts.
    #
    # The episode number is used as the environment seed so that POR,
    # safe_cost, and safe_risk can be evaluated on matching seeds.
    # ------------------------------------------------------------------

    for episode in range(
        num_episodes
    ):

        observation, info = env.reset(
            seed=episode
        )

        total_reward = 0.0
        total_cost = 0.0
        steps = 0
        unsafe_steps = 0

        while True:

            action, correction_info = (
                select_action(
                    policy,
                    goal_policy,
                    observation,
                    mean,
                    std,
                    corrector=corrector,
                )
            )

            # ----------------------------------------------------------
            # Record safety-layer diagnostics.
            # ----------------------------------------------------------

            if correction_info is not None:

                proposed_scores.append(
                    correction_info[
                        "proposed_risk"
                    ]
                )

                selected_scores.append(
                    correction_info[
                        "selected_risk"
                    ]
                )

                if correction_info[
                    "corrected"
                ]:

                    total_corrected += 1

                    correction_distances.append(
                        correction_info[
                            "distance"
                        ]
                    )

                    # Count only cases where an intervention occurred and
                    # the candidate search found an action satisfying the
                    # relevant safety threshold.
                    if correction_info[
                        "safe_candidate_found"
                    ]:
                        total_safe_candidate_found += 1

            # Final clipping before sending the action to the environment.
            action = np.clip(
                action,
                action_space.low,
                action_space.high,
            )

            (
                observation,
                reward,
                cost,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            total_steps += 1

            if float(cost) > 0.0:

                unsafe_steps += 1
                total_unsafe_steps += 1

            total_reward += float(
                reward
            )

            total_cost += float(
                cost
            )

            steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(
            total_reward
        )

        episode_costs.append(
            total_cost
        )

        episode_lengths.append(
            steps
        )

        episode_violations.append(
            float(total_cost > 0.0)
        )

        print(
            f"Episode {episode + 1}: "
            f"reward={total_reward:.3f}, "
            f"cost={total_cost:.3f}, "
            f"unsafe_steps={unsafe_steps}, "
            f"length={steps}"
        )

    env.close()

    # ------------------------------------------------------------------
    # Aggregate environment metrics.
    # ------------------------------------------------------------------

    timestep_violation_rate = (
        total_unsafe_steps / total_steps
        if total_steps > 0
        else 0.0
    )

    episode_violation_rate = float(
        np.mean(episode_violations)
    )

    print()
    print(
        "Evaluation results"
    )
    print(
        "------------------"
    )

    print(
        f"Mean reward: "
        f"{np.mean(episode_rewards):.3f}"
    )

    print(
        f"Mean cost: "
        f"{np.mean(episode_costs):.3f}"
    )

    print(
        f"Episode violation rate: "
        f"{episode_violation_rate:.3f}"
    )

    print(
        f"Timestep violation rate: "
        f"{timestep_violation_rate:.3f}"
    )

    print(
        f"Mean episode length: "
        f"{np.mean(episode_lengths):.1f}"
    )

    # ------------------------------------------------------------------
    # Aggregate safety-layer metrics.
    # ------------------------------------------------------------------

    if corrector is not None:

        correction_rate = (
            total_corrected / total_steps
            if total_steps > 0
            else 0.0
        )

        safe_candidate_rate = (
            total_safe_candidate_found
            / total_corrected
            if total_corrected > 0
            else 0.0
        )

        print()
        print(
            "Safety correction"
        )
        print(
            "-----------------"
        )

        print(
            f"Correction rate: "
            f"{correction_rate:.3f}"
        )

        print(
            "Safe candidate found after intervention: "
            f"{safe_candidate_rate:.3f}"
        )

        if correction_distances:

            print(
                "Mean correction distance: "
                f"{np.mean(correction_distances):.6f}"
            )

            print(
                "Median correction distance: "
                f"{np.median(correction_distances):.6f}"
            )

        if proposed_scores:

            # The meaning of "score" depends on the safety model:
            #
            # safe_cost:
            #     predicted discounted future cost
            #
            # safe_risk:
            #     predicted K-step violation probability
            #
            print(
                "Mean proposed safety score: "
                f"{np.mean(proposed_scores):.6f}"
            )

            print(
                "Mean selected safety score: "
                f"{np.mean(selected_scores):.6f}"
            )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate POR, the original cost-filtered Safe-POR, "
            "or the K-step risk-filtered Safe-POR."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "por",
            "safe_cost",
            "safe_risk",
        ],
        default="por",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--cost-threshold",
        type=float,
        default=1.0,
        help=(
            "Maximum predicted discounted future cost accepted "
            "by the original cost-based safety filter."
        ),
    )

    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.20,
        help=(
            "Maximum predicted K-step violation probability "
            "accepted by the new risk-based safety filter."
        ),
    )

    parser.add_argument(
        "--num-candidates",
        type=int,
        default=50,
        help=(
            "Number of candidate actions considered when a "
            "safety intervention is required."
        ),
    )

    parser.add_argument(
        "--candidate-std",
        type=float,
        default=0.10,
        help=(
            "Gaussian perturbation standard deviation used only "
            "by the original cost-based safety filter."
        ),
    )

    parser.add_argument(
        "--search-radius",
        type=float,
        default=0.10,
        help=(
            "L2 candidate-search radius used only by the new "
            "K-step risk-based safety filter."
        ),
    )

    args = parser.parse_args()

    evaluate(
        num_episodes=args.episodes,
        mode=args.mode,
        cost_threshold=args.cost_threshold,
        risk_threshold=args.risk_threshold,
        num_candidates=args.num_candidates,
        candidate_std=args.candidate_std,
        search_radius=args.search_radius,
    )