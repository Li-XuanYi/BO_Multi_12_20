"""
chen2020_5seed_experiment.py — Chen2020 5-seed comparison: LLMBO-MO vs ParEGO
=============================================================================
Runs ParEGO (matlab_reference) and LLMBO-MO on 5 seeds (8409-8413),
Chen2020 parameter set, 56 evaluations each (6 init + 50 BO iterations).

LLMBO-MO uses DeepSeek V4 Flash via api.deepseek.com.

Usage:
    python chen2020_5seed_experiment.py [--seeds 8409 8410 8411 8412 8413]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer, EXPERIMENT_PRESETS
from DataBase.database import ObservationDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("chen2020_experiment")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SEEDS = [8409, 8410, 8411, 8412, 8413]
N_EVALS = 56          # 6 init + 50 BO
N_RANDOM_INIT = 6     # for ParEGO
N_WARMSTART = 3       # for LLMBO-MO
N_RANDOM_INIT_LLMBO = 3  # for LLMBO-MO
N_ITERATIONS = 50

# DeepSeek V4 Flash API config
LLM_CONFIG = {
    "llm_backend": "openai",
    "llm_model": "deepseek-v4-flash",
    "llm_api_base": "https://api.deepseek.com",
    "llm_api_key": "sk-9538336f41ce46ae8758f68fde5bebf2",
    "llm_n_samples": 1,
    "llm_temperature": 0.3,
    "warmstart_temperature": 0.3,
    "warmstart_max_retries": 3,
    "warmstart_max_tokens": 2500,
    "warmstart_context_level": "full",
    "region_preference_max_tokens": 4096,
    "region_preference_prompt_version": "default",
}

# Base output directory
OUTPUT_ROOT = Path("chen2020_5seed_exp")
PAREGO_OUTPUT = OUTPUT_ROOT / "parego_matlab_reference"
LLMBO_OUTPUT = OUTPUT_ROOT / "llmbo_mo"


def build_parego_config(seed: int, n_evals: int) -> Dict[str, Any]:
    """Build ParEGO matlab_reference configuration."""
    n_random_init = 6
    n_iterations = n_evals - n_random_init

    return {
        "experiment_preset": "parego_matlab_reference",
        "max_iterations": n_iterations,
        "n_warmstart": 0,
        "n_random_init": n_random_init,
        "n_candidates": 1,
        "n_select": 1,
        # Mock LLM (ParEGO doesn't use LLM)
        "llm_backend": "mock",
        "llm_model": "mock",
        "llm_api_base": "",
        "llm_api_key": "",
        "llm_n_samples": 1,
        "llm_temperature": 0.7,
        "battery_param_set": "Chen2020",
        "warmstart_context_level": "full",
        "warmstart_max_tokens": 2500,
        "warmstart_max_retries": 3,
        "warmstart_temperature": None,
        "w_sample_seed": seed,
        "init_seed": seed,
        "checkpoint_dir": str(OUTPUT_ROOT / "checkpoints"),
        "checkpoint_every": 9999,
        # Disable all LLM features
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": False,
        "enable_warmstart_portfolio": False,
        "target_transform_mode": "none",
    }


def build_llmbo_config(seed: int, n_evals: int) -> Dict[str, Any]:
    """Build LLMBO-MO configuration matching the paper's setup."""
    n_random_init = N_RANDOM_INIT_LLMBO  # 3
    n_warmstart = N_WARMSTART  # 3
    n_iterations = n_evals - n_warmstart - n_random_init  # 50

    return {
        "experiment_preset": "warmstart_region_lifted_gp_force_pool_tuned",
        "max_iterations": n_iterations,
        "n_warmstart": n_warmstart,
        "n_random_init": n_random_init,
        "n_candidates": 15,
        "n_select": 1,
        **LLM_CONFIG,
        "battery_param_set": "Chen2020",
        "w_sample_seed": seed,
        "init_seed": seed,
        "checkpoint_dir": str(OUTPUT_ROOT / "checkpoints" / f"seed{seed}"),
        "checkpoint_every": 9999,
        # Enable LLM-guided BO features
        "enable_iterative_guidance": True,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": True,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": True,
        "enable_warmstart_portfolio": True,
        "warmstart_pool_size": 16,
        "warmstart_cache_path": None,
        "warmstart_cache_mode": "read_write",
        "target_transform_mode": "none",
        "objective_preprocess_mode": "minmax",
        "weight_strategy": "riesz_relaxed_cycle",
        "weight_count": 30,
        "region_lift_mode": "heuristic_correlation",
        "region_lift_control_mode": "none",
        "region_lift_external_influence_mode": "force_pool",
        "region_lift_include_raw_candidates": False,
        "region_lift_lambda_max": 0.20,
        "region_lift_n_anchors": 64,
        "region_lift_active_until": 16,
        "region_lift_min_width": 0.03,
        "region_lift_max_width": 0.80,
        "region_lift_trust_init": 0.7,
        "region_lift_anchor_weighting": "ei_softmax",
        "region_lift_anchor_temperature": 0.35,
        "region_lift_require_inside": True,
        "region_lift_candidate_oversample": 16,
        "region_lift_point_current_probe_levels": 3,
        "region_lift_point_current_probe_keep": 2,
        "region_lift_dsoc_margin": 0.01,
        "ei_n_external_restarts": 32,
        "region_lift_lgbo_shift_source": "posterior_covariance",
        "region_lift_lgbo_min_variance": 1e-12,
    }


def save_llmbo_results(optimizer: BayesOptimizer, output_dir: Path, seed: int) -> Dict:
    """Save LLMBO-MO experiment results via the optimizer's built-in method."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db = optimizer._db
    if db is None:
        raise RuntimeError("Optimizer has no database — did run() succeed?")

    # Use built-in save if available
    try:
        optimizer.save_results(str(output_dir))
    except Exception:
        pass

    # Read the saved summary
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        return summary

    # Fallback: build summary manually
    display_hv = db.compute_hypervolume()
    raw_hv = db.compute_hypervolume_raw()
    canonical_hv = raw_hv / db.hv_max if db.hv_max > 1e-12 else 0.0

    summary = {
        "seed": seed,
        "n_total": db.size,
        "n_feasible": db.n_feasible,
        "pareto_size": db.pareto_size,
        "canonical_hv": canonical_hv,
        "display_hv": display_hv,
        "hypervolume_raw": raw_hv,
        "timestamp": datetime.now().isoformat(),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def save_parego_results(db: ObservationDB, output_dir: Path, seed: int) -> Dict:
    """Save ParEGO experiment results by reading from the ObservationDB."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    display_hv = db.compute_hypervolume()
    raw_hv = db.compute_hypervolume_raw()
    canonical_hv = raw_hv / db.hv_max if db.hv_max > 1e-12 else 0.0

    summary = {
        "algorithm": "parego_matlab_reference",
        "seed": seed,
        "n_evals": N_EVALS,
        "param_set": "Chen2020",
        "n_total": db.size,
        "n_feasible": db.n_feasible,
        "pareto_size": db.pareto_size,
        "canonical_hv": canonical_hv,
        "display_hv": display_hv,
        "hypervolume_raw": raw_hv,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save database
    db.save(str(output_dir / "database.json"))

    # Save Pareto front
    pareto = db.get_pareto_front()
    pf_data = []
    for o in pareto:
        pf_data.append({
            "theta": o.theta.tolist() if hasattr(o.theta, 'tolist') else list(o.theta),
            "objectives": o.objectives.tolist() if hasattr(o.objectives, 'tolist') else list(o.objectives),
        })
    with open(output_dir / "pareto_front.json", "w") as f:
        json.dump(pf_data, f, indent=2)

    return summary


def run_single_parego(seed: int) -> Optional[Dict]:
    """Run ParEGO for a single seed."""
    logger.info("=" * 50)
    logger.info(f"ParEGO seed={seed} | START")
    t0 = time.perf_counter()

    try:
        cfg = build_parego_config(seed, N_EVALS)
        optimizer = BayesOptimizer(config=cfg)
        db = optimizer.run()

        output_dir = PAREGO_OUTPUT / f"seed{seed}"
        summary = save_parego_results(db, output_dir, seed)

        elapsed = time.perf_counter() - t0
        logger.info(f"ParEGO seed={seed} | DONE | HV={summary['canonical_hv']:.6f} | {elapsed:.1f}s")
        return summary

    except Exception as e:
        logger.error(f"ParEGO seed={seed} | FAILED: {e}")
        traceback.print_exc()
        return None


def run_single_llmbo(seed: int) -> Optional[Dict]:
    """Run LLMBO-MO for a single seed."""
    logger.info("=" * 50)
    logger.info(f"LLMBO-MO seed={seed} | START")
    t0 = time.perf_counter()

    try:
        cfg = build_llmbo_config(seed, N_EVALS)
        optimizer = BayesOptimizer(config=cfg)
        optimizer.run()

        output_dir = LLMBO_OUTPUT / f"seed{seed}"
        summary = save_llmbo_results(optimizer, output_dir, seed)

        elapsed = time.perf_counter() - t0
        hv = summary.get('canonical_hv', -1)
        logger.info(f"LLMBO-MO seed={seed} | DONE | HV={hv:.6f} | {elapsed:.1f}s")
        return summary

    except Exception as e:
        logger.error(f"LLMBO-MO seed={seed} | FAILED: {e}")
        traceback.print_exc()
        return None


def run_experiments(seeds: List[int], methods: List[str] = None):
    """Run multi-seed experiments."""
    if methods is None:
        methods = ["parego", "llmbo_mo"]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, List] = {}

    for method in methods:
        logger.info("=" * 70)
        logger.info(f"RUNNING: {method} x {len(seeds)} seeds")
        logger.info("=" * 70)

        results = []
        runner = run_single_parego if method == "parego" else run_single_llmbo

        for seed in seeds:
            summary = runner(seed)
            if summary:
                results.append(summary)
            else:
                logger.error(f"{method} seed={seed} returned no result!")

        all_results[method] = results

        # Print intermediate summary
        if results:
            hvs = [r.get('canonical_hv', 0) for r in results]
            logger.info(f"\n=== {method} SUMMARY ===")
            logger.info(f"  Completed: {len(results)}/{len(seeds)}")
            logger.info(f"  Mean HV:   {statistics.mean(hvs):.6f}")
            logger.info(f"  Std HV:    {statistics.stdev(hvs):.6f}" if len(hvs) > 1 else f"  Std HV:    N/A (n=1)")
            logger.info(f"  Min HV:    {min(hvs):.6f}")
            logger.info(f"  Max HV:    {max(hvs):.6f}")

    # Print final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: LLMBO-MO vs ParEGO on Chen2020 (5 seeds)")
    print("=" * 70)

    for method, results in all_results.items():
        if results:
            hvs = [r.get('canonical_hv', 0) for r in results]
            print(f"\n{method}:")
            for r in results:
                print(f"  seed {r.get('seed', '?'):5}: HV={r.get('canonical_hv', -1):.6f}")
            print(f"  mean={statistics.mean(hvs):.6f}  std={statistics.stdev(hvs):.6f}")

    # Save final report
    report = {
        "timestamp": datetime.now().isoformat(),
        "seeds": seeds,
        "n_evals": N_EVALS,
        "param_set": "Chen2020",
        "llm_model": LLM_CONFIG["llm_model"],
        "results": {
            method: {
                "mean_hv": statistics.mean([r.get('canonical_hv', 0) for r in results]),
                "std_hv": statistics.stdev([r.get('canonical_hv', 0) for r in results]) if len(results) > 1 else 0,
                "per_seed": results,
            }
            for method, results in all_results.items() if results
        }
    }

    report_path = OUTPUT_ROOT / "final_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nFinal report saved to: {report_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Chen2020 5-seed comparison experiment")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Random seeds",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=["parego", "llmbo_mo"],
        choices=["parego", "llmbo_mo"],
        help="Methods to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Output: {OUTPUT_ROOT.absolute()}")
    logger.info(f"LLM model: {LLM_CONFIG['llm_model']}")
    run_experiments(args.seeds, args.methods)
