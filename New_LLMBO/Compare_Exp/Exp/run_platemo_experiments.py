"""
run_platemo_experiments.py - Multi-seed launcher for PlatEMO DISK/PIMD.

Example:
    python Compare_Exp/Exp/run_platemo_experiments.py --algorithm DISK --seeds 8409 --n-evals 56
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Compare_Exp.Exp.platemo_runner import PlatEMORunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PlatEMO DISK/PIMD experiments")
    parser.add_argument("--algorithm", choices=["DISK", "PIMD", "disk", "pimd"], default="DISK")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-evals", type=int, default=56)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument(
        "--algorithm-parameters",
        type=float,
        nargs="+",
        default=None,
        help="DISK: wmax alpha; PIMD: wmax eta. Defaults follow PlatEMO.",
    )
    parser.add_argument("--platemo-root", type=Path, default=None)
    parser.add_argument("--matlab-command", type=str, default="matlab")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def _seed_dir(root: Path, seed: int, algorithm: str) -> Path:
    return root / f"seed{seed}" / f"platemo_{algorithm.lower()}"


def run_single(args: argparse.Namespace, seed: int, output_dir: Path) -> Dict:
    runner = PlatEMORunner(
        algorithm=args.algorithm,
        seed=seed,
        n_evals=args.n_evals,
        population_size=args.population_size,
        algorithm_parameters=args.algorithm_parameters,
        platemo_root=args.platemo_root,
        matlab_command=args.matlab_command,
        python_executable=args.python_executable,
    )
    runner.run()
    return runner.save_results(str(output_dir))


def build_report(records: List[Dict], failures: List[Dict[str, Any]], args: argparse.Namespace) -> Dict:
    hv_values = [r["canonical_hv"] for r in records]
    display_hvs = [r["display_hv"] for r in records]
    pareto_sizes = [r["pareto_size"] for r in records]

    def aggregate(values: List[float]) -> Dict[str, float]:
        return {
            "mean": statistics.mean(values) if values else 0.0,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }

    return {
        "algorithm": f"platemo_{args.algorithm.lower()}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "config": {
            "n_evals": args.n_evals,
            "population_size": args.population_size,
            "algorithm_parameters": args.algorithm_parameters,
            "seeds": args.seeds,
            "platemo_root": str(args.platemo_root) if args.platemo_root else None,
            "matlab_command": args.matlab_command,
        },
        "records": records,
        "failures": failures,
        "aggregates": {
            "canonical_hv": aggregate(hv_values),
            "display_hv": aggregate(display_hvs),
            "pareto_size": aggregate(pareto_sizes),
        },
    }


def main() -> None:
    args = parse_args()
    args.algorithm = args.algorithm.upper()
    if args.output_root is None:
        date_str = datetime.now().strftime("%Y_%m_%d")
        args.output_root = (
            PROJECT_ROOT
            / "optimized_experiments"
            / f"platemo_{args.algorithm.lower()}_{len(args.seeds)}seeds_{args.n_evals}evals_{date_str}"
        )

    records: List[Dict] = []
    failures: List[Dict[str, Any]] = []
    for seed in args.seeds:
        out_dir = _seed_dir(args.output_root, seed, args.algorithm)
        try:
            summary = run_single(args, seed, out_dir)
            summary["summary_path"] = str(out_dir / "summary.json")
            records.append(summary)
            logger.info("Seed %d done: canonical_hv=%.6f", seed, summary["canonical_hv"])
        except Exception as exc:
            logger.error("Seed %d failed: %s", seed, exc, exc_info=True)
            failures.append(
                {
                    "seed": seed,
                    "algorithm": f"platemo_{args.algorithm.lower()}",
                    "error": str(exc),
                    "output_dir": str(out_dir),
                }
            )

    report = build_report(records, failures, args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()
