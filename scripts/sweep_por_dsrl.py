from pathlib import Path
from collections import defaultdict
import argparse
import csv
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_por_dsrl.py"
DEFAULT_OUTPUT = RESULTS_DIR / "paired_sweep.csv"

METRICS = ("reward", "cost", "violated", "unsafe_steps")


def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def run_evaluation(command):
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    csv_line = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith("CSV_OUTPUT="):
            csv_line = line.strip()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    if csv_line is None:
        raise RuntimeError("Evaluator completed without reporting its CSV output path")
    return Path(csv_line.split("=", 1)[1].strip())


def main():
    parser = argparse.ArgumentParser(
        description="Run paired POR and safe-policy evaluations on identical episode seeds."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[2.5, 3.5, 5.0])
    parser.add_argument("--candidate-stds", nargs="+", type=float, default=[0.10])
    parser.add_argument("--num-candidates", type=int, default=75)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = args.output.resolve()
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if output_path.exists() or partial_path.exists():
        parser.error(
            f"Output or partial output already exists: {output_path}; "
            "choose a new --output path to avoid mixing experiments"
        )

    partial_path.parent.mkdir(parents=True, exist_ok=True)
    common = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--episodes",
        str(args.episodes),
    ]
    evaluation_paths = []

    try:
        for seed in args.seeds:
            evaluation_paths.append(
                run_evaluation(common + ["--mode", "por", "--seed", str(seed)])
            )
            for threshold in args.thresholds:
                for candidate_std in args.candidate_stds:
                    evaluation_paths.append(
                        run_evaluation(
                            common
                            + [
                                "--mode",
                                "safe",
                                "--seed",
                                str(seed),
                                "--risk-threshold",
                                str(threshold),
                                "--num-candidates",
                                str(args.num_candidates),
                                "--candidate-std",
                                str(candidate_std),
                            ]
                        )
                    )

        rows = [row for path in evaluation_paths for row in read_rows(path)]
        with partial_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        baselines = {
            (int(row["seed"]), int(row["episode"])): row
            for row in rows
            if row["mode"] == "por"
        }
        paired = defaultdict(lambda: {metric: [] for metric in METRICS})
        for row in rows:
            if row["mode"] != "safe":
                continue
            key = (
                float(row["risk_threshold"]),
                int(row["num_candidates"]),
                float(row["candidate_std"]),
            )
            baseline = baselines[(int(row["seed"]), int(row["episode"]))]
            for metric in METRICS:
                paired[key][metric].append(float(row[metric]) - float(baseline[metric]))

        print("\nPaired safe-minus-POR episode deltas")
        print("====================================")
        for key, deltas in sorted(paired.items()):
            threshold, candidates, candidate_std = key
            reward_delta = sum(deltas["reward"]) / len(deltas["reward"])
            cost_delta = sum(deltas["cost"]) / len(deltas["cost"])
            violation_delta = sum(deltas["violated"]) / len(deltas["violated"])
            print(
                f"threshold={threshold:g}, candidates={candidates}, std={candidate_std:g} | "
                f"n={len(deltas['cost'])} | "
                f"reward delta={reward_delta:.3f} | "
                f"cost delta={cost_delta:.3f} | "
                f"episode-violation delta={violation_delta:.3f}"
            )

        partial_path.replace(output_path)
        print(f"\nPer-episode results saved to {output_path}")
    except Exception:
        print(f"Partial results, if any, remain at {partial_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
