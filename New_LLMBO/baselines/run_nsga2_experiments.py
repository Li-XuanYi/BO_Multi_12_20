"""
run_nsga2_experiments.py — Multi-seed NSGA-II experiment launcher
=================================================================
Runs NSGA-II with multiple random seeds, saves per-seed results and
an aggregated report.json for paper comparison.

Usage:
    pixi run python baselines/run_nsga2_experiments.py --seeds 0 1 2 3 4 --n-evals 56
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.nsga2_runner import NSGA2Runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed NSGA-II experiments")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Random seeds",
    )
    parser.add_argument(
        "--n-evals", type=int, default=56,
        help="Total PyBaMM evaluations per run",
    )
    parser.add_argument(
        "--pop-size", type=int, default=20,
        help="NSGA-II population size",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Output directory (auto-generated if omitted)",
    )
    return parser.parse_args()


def run_single(seed: int, n_evals: int, pop_size: int, output_dir: Path) -> Dict:
    logger.info("=== Seed %d ===", seed)
    runner = NSGA2Runner(seed=seed, n_evals=n_evals, pop_size=pop_size)
    runner.run()
    summary = runner.save_results(str(output_dir))
    return summary


def _seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed{seed}" / "nsga2"


def build_report(records: List[Dict], seeds: List[int], n_evals: int, pop_size: int) -> Dict:
    hv_values = [r["canonical_hv"] for r in records]
    display_hvs = [r["display_hv"] for r in records]
    pareto_sizes = [r["pareto_size"] for r in records]

    aggregates = {
        "canonical_hv": {
            "mean": statistics.mean(hv_values),
            "std": statistics.stdev(hv_values) if len(hv_values) > 1 else 0.0,
            "min": min(hv_values),
            "max": max(hv_values),
        },
        "display_hv": {
            "mean": statistics.mean(display_hvs),
            "std": statistics.stdev(display_hvs) if len(display_hvs) > 1 else 0.0,
        },
        "pareto_size": {
            "mean": statistics.mean(pareto_sizes),
            "std": statistics.stdev(pareto_sizes) if len(pareto_sizes) > 1 else 0.0,
        },
    }

    return {
        "algorithm": "nsga2",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "config": {
            "n_evals": n_evals,
            "pop_size": pop_size,
            "seeds": seeds,
        },
        "records": [
            {
                "seed": r["seed"],
                "canonical_hv": r["canonical_hv"],
                "display_hv": r["display_hv"],
                "hypervolume_raw": r["hypervolume_raw"],
                "pareto_size": r["pareto_size"],
                "n_total": r["n_total"],
                "n_feasible": r["n_feasible"],
                "summary_path": str(_seed_dir(Path("."), r["seed"])),
            }
            for r in records
        ],
        "aggregates": aggregates,
    }


def main():
    args = parse_args()

    if args.output_root is None:
        date_str = datetime.now().strftime("%Y_%m_%d")
        args.output_root = PROJECT_ROOT / "optimized_experiments" / f"nsga2_{len(args.seeds)}seeds_{args.n_evals}evals_{date_str}"

    output_root: Path = args.output_root
    logger.info("Output root: %s", output_root)

    dir_fn = lambda s: _seed_dir(output_root, s)
    records: List[Dict] = []

    for seed in args.seeds:
        seed_dir = dir_fn(seed)
        try:
            summary = run_single(seed, args.n_evals, args.pop_size, seed_dir)
            records.append(summary)
            logger.info("Seed %d done: canonical_hv=%.6f, pareto=%d",
                        seed, summary["canonical_hv"], summary["pareto_size"])
        except Exception as e:
            logger.error("Seed %d failed: %s", seed, e, exc_info=True)

    report = build_report(records, args.seeds, args.n_evals, args.pop_size)

    # Fix summary_path to be relative
    for rec in report["records"]:
        rec["summary_path"] = str(output_root / f"seed{rec['seed']}" / "nsga2" / "summary.json")

    report_path = output_root / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report saved: %s", report_path)

    # Print summary
    agg = report["aggregates"]
    print(f"\n{'='*50}")
    print(f"NSGA-II Results ({len(records)} seeds)")
    print(f"{'='*50}")
    print(f"  canonical_hv: {agg['canonical_hv']['mean']:.6f} ± {agg['canonical_hv']['std']:.6f}")
    print(f"  display_hv:   {agg['display_hv']['mean']:.6f} ± {agg['display_hv']['std']:.6f}")
    print(f"  pareto_size:  {agg['pareto_size']['mean']:.1f} ± {agg['pareto_size']['std']:.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
