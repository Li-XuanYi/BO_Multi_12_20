"""
run_disk_python.py - Run DISK experiments with Python native implementation
============================================================================

Usage:
    python Compare_Exp/run_disk_python.py

This runs DISK algorithm with seeds 8409-8413, 50 evaluations each.
Results are saved to Compare_Exp/experiment_records/
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

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Compare_Exp.Exp.disk_pimd_algorithms import DISKOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DISK experiments (Python native)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[8409, 8410, 8411, 8412, 8413])
    parser.add_argument("--n-evals", type=int, default=50)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--param-set", type=str, default="Chen2020",
                       choices=["Chen2020", "Ecker2015", "ORegan2022"])
    parser.add_argument("--wmax", type=int, default=60)
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def run_single(seed: int, args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    """Run single seed experiment."""
    output_dir = output_root / f"seed{seed}" / "disk_Chen2020"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting DISK with seed {seed}")

    optimizer = DISKOptimizer(
        seed=seed,
        n_evals=args.n_evals,
        population_size=args.population_size,
        param_set=args.param_set,
        wmax=args.wmax,
        alpha=args.alpha,
    )

    optimizer.run()
    summary = optimizer.save_results(str(output_dir))
    summary["output_dir"] = str(output_dir)

    logger.info(f"Seed {seed} complete: HV={summary['canonical_hv']:.4f}")
    return summary


def build_report(records: List[Dict], args: argparse.Namespace) -> Dict[str, Any]:
    """Build experiment report."""
    hv_values = [r["canonical_hv"] for r in records]
    pareto_sizes = [r["pareto_size"] for r in records]
    n_feasibles = [r["n_feasible"] for r in records]

    def aggregate(values: List[float]) -> Dict[str, float]:
        return {
            "mean": statistics.mean(values) if values else 0.0,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    return {
        "algorithm": "disk_python",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "config": {
            "n_evals": args.n_evals,
            "population_size": args.population_size,
            "param_set": args.param_set,
            "wmax": args.wmax,
            "alpha": args.alpha,
            "seeds": args.seeds,
        },
        "records": records,
        "aggregates": {
            "canonical_hv": aggregate(hv_values),
            "pareto_size": aggregate(pareto_sizes),
            "n_feasible": aggregate(n_feasibles),
        },
    }


def main() -> int:
    args = parse_args()

    # Create output directory
    if args.output_root is None:
        date_str = datetime.now().strftime("%Y_%m_%d")
        args.output_root = (
            PROJECT_ROOT
            / "Compare_Exp"
            / "experiment_records"
            / f"disk_python_{args.param_set}_5seeds_{args.n_evals}evals_{date_str}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DISK Experiments (Python Native)")
    logger.info("=" * 60)
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Evaluations: {args.n_evals}")
    logger.info(f"Population: {args.population_size}")
    logger.info(f"Output: {args.output_root}")
    logger.info("")

    # Run experiments
    records = []
    for seed in args.seeds:
        try:
            summary = run_single(seed, args, args.output_root)
            records.append(summary)
        except Exception as e:
            logger.error(f"Seed {seed} failed: {e}", exc_info=True)

    # Build and save report
    if records:
        report = build_report(records, args)
        report_path = args.output_root / "report_5seeds.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"\nReport saved: {report_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Results Summary")
        logger.info("=" * 60)
        for r in records:
            logger.info(f"Seed {r['seed']}: HV={r['canonical_hv']:.4f}, Pareto={r['pareto_size']}")

        agg = report["aggregates"]["canonical_hv"]
        logger.info(f"\nHV Mean: {agg['mean']:.4f} ± {agg['std']:.4f}")
        logger.info(f"HV Range: [{agg['min']:.4f}, {agg['max']:.4f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
