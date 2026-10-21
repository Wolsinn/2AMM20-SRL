# Sweep_full.py 

from pathlib import Path
import argparse
import csv
import datetime as dt
import sys

import numpy as np
import torch
import safety_gymnasium
import dsrl

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import train_cost_predictor as tcp
import evaluate_cost_predictor as ecp
import evaluate_por_dsrl as ev            # <- RENAME to the file that contains evaluate()
from safety.cost_targets import concatenate_episode_ranges, discounted_cost_to_go

GAMMAS = [0.3, 0.5, 0.75, 0.9, 0.99]
QUANTILES = [0.50, 0.70, 0.80, 0.90, 0.95]
NUM_CANDIDATES = 100
CANDIDATE_STD = 1
NUM_EPISODES = 30
TARGET_COST = 10.0
COST_MODEL_SEED = 0          # train/val split seed of the cost model (test split is fixed at 42)
BATCH = 4096

TABLE_COLS = [
    "gamma", "quantile", "risk_threshold", "norm_reward_mean", "norm_cost_mean",
    "frac_within_target", "episode_violation_rate", "trigger_rate", "safe_candidate_success",
]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def policy_risks(cost_model, policy, goal_policy, states, low, high):
    """Cost-model risk of the POR policy's own (deterministic) actions on the given states."""
    proposed = []
    with torch.no_grad():
        for s in range(0, len(states), BATCH):
            obs = torch.from_numpy(states[s:s + BATCH])
            goal = goal_policy.act(obs, deterministic=True)
            action = policy.act(torch.cat([obs, goal], dim=-1), deterministic=True)
            proposed.append(action.cpu().numpy())
    proposed = np.clip(np.concatenate(proposed), low, high).astype(np.float32)
    return ecp.predict(cost_model, states, proposed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="first episode seed (episodes use seed..seed+N-1)")
    args = parser.parse_args()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ev.RESULTS_DIR / f"sweep_full_{stamp}"
    ckpt_dir = ev.RUN_DIR / "sweep"
    out_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- data (normalized exactly as in training) ----
    observations, actions, costs, terminals, timeouts = tcp.load_data()
    norm = np.load(ev.RUN_DIR / "normalization.npz")
    observations = ((observations - norm["mean"]) / norm["std"]).astype(np.float32)

    # ---- env info + POR policies ----
    env = safety_gymnasium.make(ev.ENV_NAME)
    low, high = env.action_space.low, env.action_space.high
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    env.close()
    policy, goal_policy = ev.load_policies(obs_dim, act_dim)

    summary_rows, episode_rows = [], []

    def record(config, summary, rows):
        summary_rows.append({**config, **summary})
        episode_rows.extend({"gamma": config["gamma"], "quantile": config["quantile"], **r} for r in rows)
        write_csv(out_dir / "summary.csv", summary_rows)       # rewritten after every config
        write_csv(out_dir / "episodes.csv", episode_rows)

    empty_cost = {"cost_val_mse": "", "cost_val_r2": "", "cost_val_auroc": "", "cost_val_auprc": ""}

    # ---- baseline: POR without safety layer ----
    print("\n=== Baseline (POR, no safety layer) ===")
    summary, rows = ev.evaluate(
        num_episodes=NUM_EPISODES, mode="por", seed=args.seed,
        target_cost=TARGET_COST, write_csv=False,
    )
    record({"gamma": "", "quantile": "", "risk_threshold": "", **empty_cost}, summary, rows)

    # ---- sweep ----
    for gamma in GAMMAS:
        print(f"\n=== gamma={gamma:g}: training cost model ===")
        targets = discounted_cost_to_go(costs, terminals, timeouts, gamma=gamma)
        ckpt_path = ckpt_dir / f"cost_predictor_gamma{gamma:g}.pt"
        tcp.train_one_seed(
            COST_MODEL_SEED, observations, actions, targets, costs, terminals, timeouts, 1,
            gamma=gamma, model_path=ckpt_path,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt["gamma"] == gamma
        cost_model = ev.load_cost_model(ckpt_path)
        val_ranges = [tuple(b) for b in ckpt["validation_episode_ranges"]]

        # cost-model quality on the validation episodes
        m = ecp.evaluate_ranges(observations, actions, targets, val_ranges, cost_model, [0.0])
        cost_metrics = {
            "cost_val_mse": m["mse"], "cost_val_r2": m["r2"],
            "cost_val_auroc": m["auroc_tau0"], "cost_val_auprc": m["auprc_tau0"],
        }

        # thresholds = quantiles of the risk assigned to POR's own actions on validation states
        val_states = concatenate_episode_ranges(observations, val_ranges).astype(np.float32)
        risks = policy_risks(cost_model, policy, goal_policy, val_states, low, high)
        thresholds = np.quantile(risks, QUANTILES)
        print(f"gamma={gamma:g} thresholds: " + ", ".join(f"q{q:g}={t:.4f}" for q, t in zip(QUANTILES, thresholds)))

        for q, thr in zip(QUANTILES, thresholds):
            print(f"\n--- gamma={gamma:g} quantile={q:g} threshold={thr:.4f} ---")
            summary, rows = ev.evaluate(
                num_episodes=NUM_EPISODES, mode="safe", risk_threshold=float(thr),
                num_candidates=NUM_CANDIDATES, candidate_std=CANDIDATE_STD,
                seed=args.seed, target_cost=TARGET_COST, cost_model_path=ckpt_path, write_csv=False,
            )
            record({"gamma": gamma, "quantile": q, "risk_threshold": float(thr), **cost_metrics}, summary, rows)

    # ---- console table ----
    print("\n\nSWEEP SUMMARY (baseline first)")
    print(" ".join(f"{c[:14]:>14}" for c in TABLE_COLS))
    for row in summary_rows:
        print(" ".join(f"{row[c]:>14.4f}" if isinstance(row[c], float) else f"{str(row[c]):>14}" for c in TABLE_COLS))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()