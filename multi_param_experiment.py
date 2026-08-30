"""
Multi-parameter experiment: LLMBO-MO vs ParEGO on Chen2020 + ORegan2022
======================================================================
Uses DeepSeek v4-flash API via api.deepseek.com.
Runs 5 seeds (8409-8413), 56 evaluations each, on both parameter sets.

Usage:
    cd d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO
    python multi_param_experiment.py
"""
from __future__ import annotations
import argparse, json, logging, os, statistics, sys, time, traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer, EXPERIMENT_PRESETS
from DataBase.database import ObservationDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multi_param_exp")

# Configuration
DEFAULT_SEEDS = [8409, 8410, 8411, 8412, 8413]
N_EVALS = 56
N_RANDOM_INIT = 6
N_WARMSTART = 3
N_RANDOM_INIT_LLMBO = 3
N_ITERATIONS = 50

# DeepSeek V4 Flash via api.deepseek.com
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

OUTPUT_BASE = Path("multi_param_exp")


def build_llmbo_config(seed: int, param_set: str) -> Dict[str, Any]:
    n_iter = N_EVALS - N_WARMSTART - N_RANDOM_INIT_LLMBO
    return {
        "experiment_preset": "warmstart_region_lifted_gp_force_pool_tuned",
        "max_iterations": n_iter,
        "n_warmstart": N_WARMSTART,
        "n_random_init": N_RANDOM_INIT_LLMBO,
        "n_candidates": 15,
        "n_select": 1,
        **LLM_CONFIG,
        "battery_param_set": param_set,
        "w_sample_seed": seed,
        "init_seed": seed,
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
        "checkpoint_dir": str(OUTPUT_BASE / "checkpoints" / param_set / f"seed{seed}"),
        "checkpoint_every": 9999,
    }


def build_parego_config(seed: int, param_set: str) -> Dict[str, Any]:
    n_iter = N_EVALS - N_RANDOM_INIT
    return {
        "experiment_preset": "parego_matlab_reference",
        "max_iterations": n_iter,
        "n_warmstart": 0,
        "n_random_init": N_RANDOM_INIT,
        "n_candidates": 1,
        "n_select": 1,
        "llm_backend": "mock",
        "llm_model": "mock",
        "llm_api_base": "",
        "llm_api_key": "",
        "llm_n_samples": 1,
        "llm_temperature": 0.7,
        "battery_param_set": param_set,
        "warmstart_context_level": "full",
        "warmstart_max_tokens": 2500,
        "warmstart_max_retries": 3,
        "warmstart_temperature": None,
        "w_sample_seed": seed,
        "init_seed": seed,
        "checkpoint_dir": str(OUTPUT_BASE / "checkpoints"),
        "checkpoint_every": 9999,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": False,
        "enable_warmstart_portfolio": False,
        "target_transform_mode": "none",
    }


def save_results(optimizer_or_db, output_dir: Path, seed: int, method: str, param_set: str) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(optimizer_or_db, BayesOptimizer):
        db = optimizer_or_db.database
        if db is None:
            raise RuntimeError("Optimizer has no database")
        try:
            optimizer_or_db.save_results(str(output_dir))
        except Exception:
            pass
    else:
        db = optimizer_or_db

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary["method"] = method
        summary["param_set"] = param_set
        return summary

    display_hv = db.compute_hypervolume()
    raw_hv = db.compute_hypervolume_raw()
    canonical_hv = raw_hv / db.hv_max if db.hv_max > 1e-12 else 0.0

    summary = {
        "method": method,
        "param_set": param_set,
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

    db.save(str(output_dir / "database.json"))
    pareto = db.get_pareto_front()
    pf_data = [{"theta": list(o.theta), "objectives": list(o.objectives)} for o in pareto]
    with open(output_dir / "pareto_front.json", "w") as f:
        json.dump(pf_data, f, indent=2)
    return summary


def run_parego(seed: int, param_set: str) -> Optional[Dict]:
    logger.info(f"[ParEGO] {param_set} seed={seed}")
    t0 = time.perf_counter()
    try:
        cfg = build_parego_config(seed, param_set)
        optimizer = BayesOptimizer(config=cfg)
        db = optimizer.run()
        out = OUTPUT_BASE / param_set / "parego" / f"seed{seed}"
        s = save_results(db, out, seed, "parego", param_set)
        logger.info(f"[ParEGO] {param_set} seed={seed} done: HV={s['canonical_hv']:.4f} ({time.perf_counter()-t0:.0f}s)")
        return s
    except Exception as e:
        logger.error(f"[ParEGO] {param_set} seed={seed} FAILED: {e}")
        traceback.print_exc()
        return None


def run_llmbo(seed: int, param_set: str) -> Optional[Dict]:
    logger.info(f"[LLMBO] {param_set} seed={seed}")
    t0 = time.perf_counter()
    try:
        cfg = build_llmbo_config(seed, param_set)
        optimizer = BayesOptimizer(config=cfg)
        optimizer.run()
        out = OUTPUT_BASE / param_set / "llmbo" / f"seed{seed}"
        s = save_results(optimizer, out, seed, "llmbo", param_set)
        logger.info(f"[LLMBO] {param_set} seed={seed} done: HV={s['canonical_hv']:.4f} ({time.perf_counter()-t0:.0f}s)")
        return s
    except Exception as e:
        logger.error(f"[LLMBO] {param_set} seed={seed} FAILED: {e}")
        traceback.print_exc()
        return None


def run_all(param_sets: List[str], seeds: List[int]):
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": datetime.now().isoformat(), "param_sets": {}}

    for ps in param_sets:
        logger.info(f"\n{'='*60}\nPARAM SET: {ps}\n{'='*60}")
        parego_results = []
        llmbo_results = []

        for seed in seeds:
            r = run_parego(seed, ps)
            if r: parego_results.append(r)

        for seed in seeds:
            r = run_llmbo(seed, ps)
            if r: llmbo_results.append(r)

        # Summary
        p_hvs = [r["canonical_hv"] for r in parego_results]
        l_hvs = [r["canonical_hv"] for r in llmbo_results]

        if p_hvs and l_hvs:
            p_mean, l_mean = statistics.mean(p_hvs), statistics.mean(l_hvs)
            improvement = (l_mean - p_mean) / p_mean * 100
            logger.info(f"\n{ps} SUMMARY:")
            logger.info(f"  ParEGO:   {p_mean:.4f} ± {statistics.stdev(p_hvs):.4f}")
            logger.info(f"  LLMBO-MO: {l_mean:.4f} ± {statistics.stdev(l_hvs):.4f}")
            logger.info(f"  Improvement: {improvement:+.2f}%")
        else:
            improvement = 0

        report["param_sets"][ps] = {
            "parego": {"mean": statistics.mean(p_hvs) if p_hvs else 0, "std": statistics.stdev(p_hvs) if len(p_hvs)>1 else 0, "per_seed": parego_results},
            "llmbo": {"mean": statistics.mean(l_hvs) if l_hvs else 0, "std": statistics.stdev(l_hvs) if len(l_hvs)>1 else 0, "per_seed": llmbo_results},
            "improvement_pct": improvement,
        }

    with open(OUTPUT_BASE / "report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Final print
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for ps, data in report["param_sets"].items():
        print(f"\n{ps}:")
        print(f"  ParEGO:   {data['parego']['mean']:.4f} ± {data['parego']['std']:.4f}")
        print(f"  LLMBO-MO: {data['llmbo']['mean']:.4f} ± {data['llmbo']['std']:.4f}")
        print(f"  Improvement: {data['improvement_pct']:+.2f}%")
    print(f"\nReport: {OUTPUT_BASE / 'report.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param-sets", nargs="+", default=["Chen2020", "ORegan2022"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    logger.info(f"Param sets: {args.param_sets}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"LLM: {LLM_CONFIG['llm_model']} via {LLM_CONFIG['llm_api_base']}")
    run_all(args.param_sets, args.seeds)


if __name__ == "__main__":
    main()
