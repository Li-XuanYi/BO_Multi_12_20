"""
run_three_algo_comparison.py — 三算法对比实验运行脚本 (NSGA-II, ParEGO, LLMBO-MO)
==================================================================================
运行5组对比实验，每组包含三种算法：
- NSGA-II (pymoo实现)
- ParEGO (matlab_reference变体)
- LLMBO-MO (使用指定API配置)

每组50轮迭代（加上随机初始化），使用随机种子范围10-5000。

Usage:
    python Compare_Exp/Exp/run_three_algo_comparison.py --seeds 389 822 2323 4097 4304 --n-evals 56

LLMBO-MO配置:
- API Key: sk-HmCBUaZaKtzEFmFSmGBZb9hIcALBDZFAyhGbyNU5VLB7FMyb
- API Base: https://api.nuwaapi.com/v1
- Model: gpt-4.1
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
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Compare_Exp.Exp.nsga2_runner import NSGA2Runner
from Compare_Exp.Exp.parego_runner import ParEGORunner
from llmbo.optimizer import BayesOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 实验配置常量
# ═══════════════════════════════════════════════════════════════════════════

LLMBO_CONFIG = {
    # 使用最优preset作为基础
    "experiment_preset": "warmstart_region_lifted_gp_force_pool_tuned",
    "max_iterations": 50,
    "n_warmstart": 5,
    "n_random_init": 6,
    "n_candidates": 15,
    "n_select": 1,
    # LLM API配置
    "llm_backend": "openai",
    "llm_model": "gpt-4.1",
    "llm_api_base": "https://api.nuwaapi.com/v1",
    "llm_api_key": "sk-HmCBUaZaKtzEFmFSmGBZb9hIcALBDZFAyhGbyNU5VLB7FMyb",
    "llm_n_samples": 3,
    "llm_temperature": 0.7,
    # 电池参数
    "battery_param_set": "Chen2020",
    "warmstart_context_level": "full",
    "warmstart_max_tokens": 2500,
    # 功能开关 - 使用最优配置
    "enable_iterative_guidance": True,
    "enable_gp_llm_coupling": False,
    "enable_acq_prior_coupling": True,
    "enable_proposal_sampler": False,
    "enable_llm_rerank": False,
    # Region Lifted GP配置 - 最优参数
    "enable_region_lifted_gp": True,
    "region_lift_variant": "wider_active16_ext32",
    "region_lift_warmstart_max_tokens": 2500,
    "region_lift_reflection_max_tokens": 2000,
    "region_lift_wider_active_region_size": 16,
    "region_lift_wider_extension_size": 32,
    "region_lift_external_influence_mode": "force_pool",
    "region_lift_include_raw_candidates": False,
    "region_lift_lambda_max": 0.20,
    "region_lift_n_anchors": 64,
    "region_lift_active_until": 16,
    "region_lift_max_plain_ei_gap": 0.25,
    "region_lift_min_volume": 1e-8,
    "region_lift_max_volume": 0.08,
    "region_lift_min_width": 0.03,
    "region_lift_trust_init": 0.7,
    "region_lift_anchor_weighting": "ei_softmax",
    "region_lift_anchor_temperature": 0.35,
    "region_lift_require_inside": True,
    "region_lift_min_sigma_ratio": 0.85,
    "region_lift_candidate_oversample": 16,
    "region_lift_point_current_probe_levels": 3,
    "region_lift_point_current_probe_keep": 2,
    "region_lift_close_distance": 0.03,
    "region_lift_dsoc_margin": 0.01,
    # 检查点配置
    "checkpoint_dir": str(PROJECT_ROOT / "checkpoints"),
    "checkpoint_every": 9999,
}

PAREGO_CONFIG = {
    "variant": "matlab_reference",
    "n_random_init": 6,
}

NSGA2_CONFIG = {
    "pop_size": 20,
}


# ═══════════════════════════════════════════════════════════════════════════
# LLMBO-MO Runner
# ═══════════════════════════════════════════════════════════════════════════

class LLMBORunner:
    """LLMBO-MO runner with full configuration."""

    def __init__(
        self,
        seed: int = 0,
        n_evals: int = 56,
        n_warmstart: int = 5,
        n_random_init: int = 6,
    ):
        self.seed = seed
        self.n_evals = n_evals
        self.n_warmstart = n_warmstart
        self.n_random_init = n_random_init
        self.n_iterations = n_evals - n_warmstart - n_random_init
        self.db: Optional[Any] = None
        self.optimizer: Optional[BayesOptimizer] = None
        self._result_summary: Optional[Dict] = None

    def run(self) -> Any:
        """Run LLMBO-MO optimization."""
        logger.info("=" * 60)
        logger.info("LLMBO-MO Runner: seed=%d, n_evals=%d", self.seed, self.n_evals)
        logger.info("=" * 60)

        # Build configuration
        cfg_dict = LLMBO_CONFIG.copy()
        cfg_dict["max_iterations"] = self.n_iterations
        cfg_dict["n_warmstart"] = self.n_warmstart
        cfg_dict["n_random_init"] = self.n_random_init
        cfg_dict["w_sample_seed"] = self.seed
        cfg_dict["init_seed"] = self.seed

        self.optimizer = BayesOptimizer(config=cfg_dict)
        self.db = self.optimizer.run()

        logger.info(
            "LLMBO-MO done: %d evals, %d feasible, %d Pareto, HV=%.4f",
            self.db.size, self.db.n_feasible, self.db.pareto_size,
            self.db.compute_hypervolume()
        )
        return self.db

    def save_results(self, output_dir: str) -> Dict:
        """Save experiment results in compatible format."""
        if self.db is None:
            raise RuntimeError("Must call run() before save_results()")

        os.makedirs(output_dir, exist_ok=True)

        display_hv = self.db.compute_hypervolume()
        raw_hv = self.db.compute_hypervolume_raw()
        canonical_hv = raw_hv / self.db.hv_max if self.db.hv_max > 1e-12 else 0.0

        # Build hv_trace from database observations
        hv_trace = []
        eval_count = 0
        for obs in self.db.observations:
            eval_count += 1
            hv_trace.append({
                "eval_index": eval_count,
                "phase": "init" if obs.source != "bo" else "bo",
                "iteration": obs.iteration,
                "source": obs.source,
                "theta": obs.theta.tolist() if hasattr(obs.theta, 'tolist') else list(obs.theta),
                "feasible": obs.feasible,
                "hypervolume": display_hv,
                "display_hv": display_hv,
                "canonical_hv": canonical_hv,
                "hypervolume_raw": raw_hv,
                "pareto_size": self.db.pareto_size,
                "n_total": self.db.size,
                "n_feasible": self.db.n_feasible,
                "elapsed_s": 0.0,
            })

        summary = {
            "algorithm": "llmbo_mo",
            "seed": self.seed,
            "n_evals": self.n_evals,
            "n_warmstart": self.n_warmstart,
            "n_random_init": self.n_random_init,
            "n_iterations": self.n_iterations,
            "n_total": self.db.size,
            "n_feasible": self.db.n_feasible,
            "pareto_size": self.db.pareto_size,
            "hypervolume": display_hv,
            "display_hv": display_hv,
            "canonical_hv": canonical_hv,
            "hypervolume_raw": raw_hv,
            "hv_trace": hv_trace,
            "best_per_objective": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Best per objective
        best = self.db.get_best_per_objective()
        from DataBase.database import OBJECTIVE_NAMES
        for name in OBJECTIVE_NAMES:
            if name in best:
                o = best[name]
                idx = list(OBJECTIVE_NAMES).index(name)
                summary["best_per_objective"][name] = {
                    "value": float(o.objectives[idx]),
                    "theta": o.theta.tolist() if hasattr(o.theta, 'tolist') else list(o.theta),
                }

        # Save summary.json
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save database.json
        self.db.save(os.path.join(output_dir, "database.json"))

        # Save pareto_front.json
        pareto = self.db.get_pareto_front()
        pf_data = []
        for o in pareto:
            pf_data.append({
                "theta": o.theta.tolist() if hasattr(o.theta, 'tolist') else list(o.theta),
                "objectives": o.objectives.tolist() if hasattr(o.objectives, 'tolist') else list(o.objectives),
            })
        with open(os.path.join(output_dir, "pareto_front.json"), "w") as f:
            json.dump(pf_data, f, indent=2)

        self._result_summary = summary
        logger.info("Results saved to %s", output_dir)
        return summary

    def get_summary(self) -> Optional[Dict]:
        """Get the result summary after run()."""
        return self._result_summary


# ═══════════════════════════════════════════════════════════════════════════
# 命令行参数解析
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three-algorithm comparison experiments (NSGA-II, ParEGO, LLMBO-MO)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[389, 822, 2323, 4097, 4304],
        help="Random seeds for experiments (default: 5 random seeds)",
    )
    parser.add_argument(
        "--n-evals",
        type=int,
        default=56,
        help="Total PyBaMM evaluations per run (default: 56 = 5 warmstart + 6 random init + 50 BO)",
    )
    parser.add_argument(
        "--n-warmstart",
        type=int,
        default=5,
        help="Number of LLM warmstart samples (default: 5)",
    )
    parser.add_argument(
        "--n-random-init",
        type=int,
        default=6,
        help="Number of random initialization points (default: 6)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=20,
        help="NSGA-II population size (default: 20)",
    )
    parser.add_argument(
        "--parego-variant",
        type=str,
        default="matlab_reference",
        choices=["baseline", "matlab_reference"],
        help="ParEGO variant (default: matlab_reference)",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["nsga2", "parego", "llmbo"],
        choices=["nsga2", "parego", "llmbo", "all"],
        help="Algorithms to run (default: all three)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory (auto-generated if omitted)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have results",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 实验运行函数
# ═══════════════════════════════════════════════════════════════════════════

def run_nsga2(seed: int, n_evals: int, pop_size: int, output_dir: Path) -> Dict:
    """Run NSGA-II experiment."""
    logger.info("[NSGA-II] Seed %d starting...", seed)
    runner = NSGA2Runner(seed=seed, n_evals=n_evals, pop_size=pop_size)
    runner.run()
    summary = runner.save_results(str(output_dir))
    logger.info(
        "[NSGA-II] Seed %d done: canonical_hv=%.6f, pareto=%d",
        seed, summary["canonical_hv"], summary["pareto_size"]
    )
    return summary


def run_parego(seed: int, n_evals: int, n_random_init: int, variant: str, output_dir: Path) -> Dict:
    """Run ParEGO experiment."""
    logger.info("[ParEGO] Seed %d starting...", seed)
    runner = ParEGORunner(
        seed=seed,
        n_evals=n_evals,
        n_random_init=n_random_init,
        variant=variant,
    )
    runner.run()
    summary = runner.save_results(str(output_dir))
    logger.info(
        "[ParEGO] Seed %d done: canonical_hv=%.6f, pareto=%d",
        seed, summary["canonical_hv"], summary["pareto_size"]
    )
    return summary


def run_llmbo(seed: int, n_evals: int, n_warmstart: int, n_random_init: int, output_dir: Path) -> Dict:
    """Run LLMBO-MO experiment."""
    logger.info("[LLMBO-MO] Seed %d starting...", seed)
    runner = LLMBORunner(
        seed=seed,
        n_evals=n_evals,
        n_warmstart=n_warmstart,
        n_random_init=n_random_init,
    )
    runner.run()
    summary = runner.save_results(str(output_dir))
    logger.info(
        "[LLMBO-MO] Seed %d done: canonical_hv=%.6f, pareto=%d",
        seed, summary["canonical_hv"], summary["pareto_size"]
    )
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════════════════════════════════

def build_algorithm_report(
    records: List[Dict],
    algorithm: str,
    seeds: List[int],
    config: Dict
) -> Dict:
    """Build report for a single algorithm."""
    hv_values = [r["canonical_hv"] for r in records]
    display_hvs = [r["display_hv"] for r in records]
    pareto_sizes = [r["pareto_size"] for r in records]

    aggregates = {
        "canonical_hv": {
            "mean": statistics.mean(hv_values) if hv_values else 0.0,
            "std": statistics.stdev(hv_values) if len(hv_values) > 1 else 0.0,
            "min": min(hv_values) if hv_values else 0.0,
            "max": max(hv_values) if hv_values else 0.0,
        },
        "display_hv": {
            "mean": statistics.mean(display_hvs) if display_hvs else 0.0,
            "std": statistics.stdev(display_hvs) if len(display_hvs) > 1 else 0.0,
        },
        "pareto_size": {
            "mean": statistics.mean(pareto_sizes) if pareto_sizes else 0.0,
            "std": statistics.stdev(pareto_sizes) if len(pareto_sizes) > 1 else 0.0,
        },
    }

    return {
        "algorithm": algorithm,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "config": config,
        "records": [
            {
                "seed": r["seed"],
                "canonical_hv": r["canonical_hv"],
                "display_hv": r["display_hv"],
                "hypervolume_raw": r["hypervolume_raw"],
                "pareto_size": r["pareto_size"],
                "n_total": r["n_total"],
                "n_feasible": r["n_feasible"],
            }
            for r in records
        ],
        "aggregates": aggregates,
    }


def build_comparison_report(
    nsga2_records: List[Dict],
    parego_records: List[Dict],
    llmbo_records: List[Dict],
    seeds: List[int],
    args: argparse.Namespace
) -> Dict:
    """Build comprehensive comparison report."""
    nsga2_report = build_algorithm_report(
        nsga2_records, "nsga2", seeds,
        {"n_evals": args.n_evals, "pop_size": args.pop_size}
    )
    parego_report = build_algorithm_report(
        parego_records, f"parego_{args.parego_variant}", seeds,
        {"n_evals": args.n_evals, "n_random_init": args.n_random_init, "variant": args.parego_variant}
    )
    llmbo_report = build_algorithm_report(
        llmbo_records, "llmbo_mo", seeds,
        {"n_evals": args.n_evals, "n_warmstart": args.n_warmstart, "n_random_init": args.n_random_init}
    )

    # Cross-algorithm comparison
    comparison = {
        "nsga2": nsga2_report["aggregates"],
        "parego": parego_report["aggregates"],
        "llmbo": llmbo_report["aggregates"],
    }

    # Find best algorithm per metric
    best_by_metric = {}
    for metric in ["canonical_hv", "display_hv"]:
        values = {
            "nsga2": nsga2_report["aggregates"][metric]["mean"],
            "parego": parego_report["aggregates"][metric]["mean"],
            "llmbo": llmbo_report["aggregates"][metric]["mean"],
        }
        best_algo = max(values, key=values.get)
        best_by_metric[metric] = {
            "best_algorithm": best_algo,
            "value": values[best_algo],
        }

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "config": {
            "seeds": seeds,
            "n_evals": args.n_evals,
            "n_warmstart": args.n_warmstart,
            "n_random_init": args.n_random_init,
            "pop_size": args.pop_size,
            "parego_variant": args.parego_variant,
        },
        "algorithms": {
            "nsga2": nsga2_report,
            "parego": parego_report,
            "llmbo": llmbo_report,
        },
        "comparison": comparison,
        "best_by_metric": best_by_metric,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Handle algorithm list
    if "all" in args.algorithms:
        algorithms = ["nsga2", "parego", "llmbo"]
    else:
        algorithms = args.algorithms

    # Set output directory
    if args.output_root is None:
        date_str = datetime.now().strftime("%Y_%m_%d")
        args.output_root = (
            PROJECT_ROOT / "Compare_Exp" / "experiment_records"
            / f"three_algo_comparison_{len(args.seeds)}seeds_{args.n_evals}evals_{date_str}"
        )

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Three-Algorithm Comparison Experiment")
    logger.info("=" * 70)
    logger.info("Algorithms: %s", algorithms)
    logger.info("Seeds: %s", args.seeds)
    logger.info("Evaluations per run: %d", args.n_evals)
    logger.info("Output root: %s", output_root)
    logger.info("=" * 70)

    # Storage for results
    nsga2_records = []
    parego_records = []
    llmbo_records = []

    # Run experiments
    for seed in args.seeds:
        logger.info("\n" + "=" * 70)
        logger.info("Running experiments for seed %d", seed)
        logger.info("=" * 70)

        seed_dir = output_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # NSGA-II
        if "nsga2" in algorithms:
            nsga2_dir = seed_dir / "nsga2"
            summary_path = nsga2_dir / "summary.json"
            if args.skip_existing and summary_path.exists():
                logger.info("[NSGA-II] Seed %d: skipping (exists)", seed)
                with open(summary_path) as f:
                    summary = json.load(f)
            else:
                try:
                    summary = run_nsga2(seed, args.n_evals, args.pop_size, nsga2_dir)
                except Exception as e:
                    logger.error("[NSGA-II] Seed %d failed: %s", seed, e, exc_info=True)
                    summary = None
            if summary:
                nsga2_records.append(summary)

        # ParEGO
        if "parego" in algorithms:
            parego_dir = seed_dir / f"parego_{args.parego_variant}"
            summary_path = parego_dir / "summary.json"
            if args.skip_existing and summary_path.exists():
                logger.info("[ParEGO] Seed %d: skipping (exists)", seed)
                with open(summary_path) as f:
                    summary = json.load(f)
            else:
                try:
                    summary = run_parego(
                        seed, args.n_evals, args.n_random_init,
                        args.parego_variant, parego_dir
                    )
                except Exception as e:
                    logger.error("[ParEGO] Seed %d failed: %s", seed, e, exc_info=True)
                    summary = None
            if summary:
                parego_records.append(summary)

        # LLMBO-MO
        if "llmbo" in algorithms:
            llmbo_dir = seed_dir / "llmbo_mo"
            summary_path = llmbo_dir / "summary.json"
            if args.skip_existing and summary_path.exists():
                logger.info("[LLMBO-MO] Seed %d: skipping (exists)", seed)
                with open(summary_path) as f:
                    summary = json.load(f)
            else:
                try:
                    summary = run_llmbo(
                        seed, args.n_evals, args.n_warmstart,
                        args.n_random_init, llmbo_dir
                    )
                except Exception as e:
                    logger.error("[LLMBO-MO] Seed %d failed: %s", seed, e, exc_info=True)
                    summary = None
            if summary:
                llmbo_records.append(summary)

    # Generate comparison report
    report = build_comparison_report(
        nsga2_records, parego_records, llmbo_records,
        args.seeds, args
    )

    report_path = output_root / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("\nComparison report saved: %s", report_path)

    # Print summary
    print("\n" + "=" * 70)
    print("Three-Algorithm Comparison Results")
    print("=" * 70)

    for algo_name, algo_data in report["algorithms"].items():
        agg = algo_data["aggregates"]
        print(f"\n{algo_name.upper()}:")
        print(f"  canonical_hv: {agg['canonical_hv']['mean']:.6f} ± {agg['canonical_hv']['std']:.6f}")
        print(f"  display_hv:   {agg['display_hv']['mean']:.6f} ± {agg['display_hv']['std']:.6f}")
        print(f"  pareto_size:  {agg['pareto_size']['mean']:.1f} ± {agg['pareto_size']['std']:.1f}")

    print("\n" + "=" * 70)
    print("Best by metric:")
    for metric, data in report["best_by_metric"].items():
        print(f"  {metric}: {data['best_algorithm']} ({data['value']:.6f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
