"""
run_seeds.py
Run each coordinator across multiple seeds and report mean ± std.

Usage:
    python run_seeds.py                          # 5 seeds, all 3 coordinators
    python run_seeds.py --seeds 0 1 2 3 4 9
    python run_seeds.py --coordinators confidence margin
"""
import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path("artifacts/results")
SCRIPT = Path(__file__).parent / "run_pipeline.py"


def run_one(coordinator: str, seed: int, distill_epochs: int) -> dict:
    exp_name = f"seed{seed}_{coordinator}"
    result_path = RESULTS_DIR / f"{exp_name}.json"

    if result_path.exists():
        print(f"  [skip] {exp_name} already exists, loading cached result.")
        with open(result_path) as f:
            return json.load(f)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--exp-name", exp_name,
        "--coordinator", coordinator,
        "--seed", str(seed),
        "--distill-epochs", str(distill_epochs),
    ]
    print(f"  Running: {' '.join(cmd[2:])}")
    result = subprocess.run(cmd, cwd=Path(SCRIPT).parent, capture_output=False)
    if result.returncode != 0:
        print(f"  [ERROR] Run failed for {exp_name}")
        return {}

    with open(result_path) as f:
        return json.load(f)


def summarize(results: list[dict]) -> dict:
    if not results:
        return {}
    keys = ["acc_client_a", "acc_client_b", "acc_teacher", "acc_student"]
    import statistics
    summary = {}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            summary[k] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--coordinators", nargs="+",
        choices=["confidence", "margin", "average"],
        default=["confidence", "margin", "average"],
    )
    parser.add_argument("--distill-epochs", type=int, default=5)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for coord in args.coordinators:
        print(f"\n=== Coordinator: {coord} ===")
        results = []
        for seed in args.seeds:
            r = run_one(coord, seed, args.distill_epochs)
            if r:
                results.append(r)
        all_summaries[coord] = summarize(results)

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS: mean (std) over seeds:", args.seeds)
    print("=" * 70)
    header = f"{'Coordinator':<12}  {'Client A':>10}  {'Client B':>10}  {'Teacher':>10}  {'Student':>10}"
    print(header)
    print("-" * 70)
    for coord, s in all_summaries.items():
        def fmt(key):
            if key not in s:
                return "  n/a     "
            return f"{s[key]['mean']:.4f}±{s[key]['std']:.4f}"
        print(
            f"{coord:<12}  {fmt('acc_client_a'):>18}  {fmt('acc_client_b'):>18}"
            f"  {fmt('acc_teacher'):>18}  {fmt('acc_student'):>18}"
        )
    print("=" * 70)

    # Save summary JSON
    out = RESULTS_DIR / "seeds_summary.json"
    with open(out, "w") as f:
        json.dump({"seeds": args.seeds, "coordinators": all_summaries}, f, indent=2)
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
