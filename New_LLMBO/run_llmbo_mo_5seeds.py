"""
run_llmbo_mo_5seeds.py — Run LLMBO-MO multi-seed on Chen2020 for paper comparison
=================================================================================
Runs LLMBO-MO with DeepSeek-V4-Flash on 5 seeds (8409, 8410, 8411, 8412, 8413)
for the Chen2020 battery parameter set.

Usage:
    python run_llmbo_mo_5seeds.py
    python run_llmbo_mo_5seeds.py --seeds 8409 8410 8411 8412 8413 --n-evals 56
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_API_BASE = "https://api.chat.csu.edu.cn/v1"
DEFAULT_API_KEY = "sk-d1ee7a7d3e594831be6ad87b4d367e4c"
DEFAULT_MODEL = "deepseek-v4-flash"

LLMBO_CONFIG_BASE = {
    # Use the paper's closest preset as base
    "experiment_preset": "warmstart_region_lifted_gp_force_pool_tuned",
    "n_warmstart": 3,
    "n_random_init": 3,
    "max_iterations": 50,
    "n_candidates": 15,
    "n_select": 1,

    # LLM API configuration
    "llm_backend": "openai",       # OpenAI-compatible API
    "llm_model": DEFAULT_MODEL,
    "llm_api_base": DEFAULT_API_BASE,
    "llm_api_key": DEFAULT_API_KEY,
    "llm_n_samples": 3,
    "llm_temperature": 0.7,

    # Battery parameters
    "battery_param_set": "Chen2020",
    "soc_start": 0.0,
    "soc_end": 0.8,

    # Warmstart
    "enable_warmstart_portfolio": True,
    "warmstart_pool_size": 16,
    "warmstart_context_level": "full",
    "warmstart_max_tokens": 2500,
    "warmstart_cache_mode": "read_write",

    # Acquisition
    "acquisition_strategy": "ei_lbfgsb",

    # Objective transformation
    "target_transform_mode": "none",
    "objective_preprocess_mode": "minmax",

    # Weight generation
    "weight_strategy": "riesz_relaxed_cycle",
    "weight_simplex_divisions": 10,
    "weight_count": 30,
}


def build_config(seed: int, n_evals: int, output_dir: Path) -> Dict[str, Any]:
    """Build LLMBO-MO config dictionary for a given seed."""
    cfg = dict(LLMBO_CONFIG_BASE)
    cfg["seed"] = seed
    cfg["max_iterations"] = n_evals - cfg["n_warmstart"] - cfg["n_random_init"]

    # Set up checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg["checkpoint_dir"] = str(checkpoint_dir)
    cfg["checkpoint_every"] = 999999  # effectively disable checkpoint saves

    # Override with EXPERIMENT_PRESETS
    from llmbo.optimizer import EXPERIMENT_PRESETS
    preset = EXPERIMENT_PRESETS.get("warmstart_region_lifted_gp_force_pool_tuned", {})
    cfg.update(preset)
    # Re-apply our custom overrides on top
    cfg["n_warmstart"] = 3
    cfg["n_random_init"] = 3
    cfg["checkpoint_dir"] = str(checkpoint_dir)
    cfg["checkpoint_every"] = 999999

    return cfg


def run_single(seed: int, n_evals: int, output_dir: Path) -> Dict[str, Any]:
    """Run a single LLMBO-MO experiment for one seed."""
    logger.info("=" * 60)
    logger.info("LLMBO-MO: seed=%d, n_evals=%d, model=%s", seed, n_evals, DEFAULT_MODEL)
    logger.info("=" * 60)

    cfg = build_config(seed, n_evals, output_dir)
    optimizer = BayesOptimizer(config=cfg)
    db = optimizer.run()
    optimizer.save_results(str(output_dir))

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        # Build minimal summary from optimizer state
        hv = db.current_hypervolume() if hasattr(db, "current_hypervolume") else 0.0
        summary = {
            "canonical_hv": hv,
            "n_total": db.size if hasattr(db, "size") else n_evals,
            "n_feasible": db.n_feasible if hasattr(db, "n_feasible") else 0,
            "pareto_size": db.pareto_size if hasattr(db, "pareto_size") else 0,
            "seed": seed,
        }

    logger.info("Seed %d done: canonical_hv=%.6f, pareto_size=%d",
                seed, summary.get("canonical_hv", -1), summary.get("pareto_size", -1))
    summary.setdefault("seed", seed)
    return summary


def build_report(records: List[Dict], seeds: List[int], n_evals: int) -> Dict[str, Any]:
    """Build aggregated report from per-seed records."""
    hv_values = [r["canonical_hv"] for r in records]
    pareto_sizes = [r["pareto_size"] for r in records]

    aggregates = {
        "canonical_hv": {
            "mean": statistics.mean(hv_values),
            "std": statistics.stdev(hv_values) if len(hv_values) > 1 else 0.0,
            "min": min(hv_values),
            "max": max(hv_values),
        },
        "pareto_size": {
            "mean": statistics.mean(pareto_sizes),
            "std": statistics.stdev(pareto_sizes) if len(pareto_sizes) > 1 else 0.0,
        },
    }

    return {
        "algorithm": "llmbo_mo",
        "date": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "config": {
            "n_evals": n_evals,
            "model": DEFAULT_MODEL,
            "api_base": DEFAULT_API_BASE,
            "seeds": seeds,
        },
        "records": [{"seed": r["seed"], "canonical_hv": r["canonical_hv"],
                      "pareto_size": r["pareto_size"]} for r in records],
        "aggregates": aggregates,
    }


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed LLMBO-MO experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[8409, 8410, 8411, 8412, 8413],
                        help="Random seeds")
    parser.add_argument("--n-evals", type=int, default=56,
                        help="Total evaluations per run")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Output directory (auto-generated if omitted)")
    args = parser.parse_args()

    if args.output_root is None:
        date_str = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        args.output_root = PROJECT_ROOT / "optimized_experiments" / f"llmbo_mo_5seeds_{args.n_evals}evals_{date_str}"

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output root: %s", output_root)

    records: List[Dict] = []

    for seed in args.seeds:
        seed_dir = output_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary = run_single(seed, args.n_evals, seed_dir)
            records.append(summary)
        except Exception as e:
            logger.error("Seed %d failed: %s", seed, e, exc_info=True)

    report = build_report(records, args.seeds, args.n_evals)
    report_path = output_root / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", report_path)

    # Print summary
    agg = report["aggregates"]
    print(f"\n{'=' * 50}")
    print(f"LLMBO-MO Results ({len(records)} seeds, {args.n_evals} evals)")
    print(f"{'=' * 50}")
    print(f"  canonical_hv: {agg['canonical_hv']['mean']:.6f} ± {agg['canonical_hv']['std']:.6f}")
    print(f"  pareto_size:  {agg['pareto_size']['mean']:.1f} ± {agg['pareto_size']['std']:.1f}")
    print(f"  per seed:")
    for r in report["records"]:
        print(f"    seed {r['seed']}: HV={r['canonical_hv']:.6f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
