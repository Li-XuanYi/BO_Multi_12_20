"""
比较实验：Experimental Prompt vs Detailed Prompt vs Baseline
- 10 seeds
- 10 iterations each
- Model: deepseek-v3-thinking
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer
from utils.model_labels import canonical_model_label


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiment_records" / "prompt_comparison_10seeds_10iter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare experimental vs detailed warmstart prompt vs baseline."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that stores all runs and the final report.json",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of BO iterations per run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[8409, 8410, 8411, 8412, 8413, 8414, 8415, 8416, 8417, 8418],
        help="Random seeds for experiments",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("LLM_API_KEY", ""),
        help="LLM API key",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://api.chat.csu.edu.cn/v1",
        help="API base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3-thinking",
        help="Model name",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline runs",
    )
    parser.add_argument(
        "--skip-detailed",
        action="store_true",
        help="Skip detailed prompt runs",
    )
    parser.add_argument(
        "--skip-experimental",
        action="store_true",
        help="Skip experimental prompt runs",
    )
    return parser.parse_args()


def _mean(values: List[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _run_record(seed: int, summary: Dict[str, Any]) -> Dict[str, Any]:
    warmstart_trace = summary.get("warmstart_trace") or []
    warmstart_hv_last = None
    if warmstart_trace:
        warmstart_hv_last = float(warmstart_trace[-1].get("hypervolume", 0.0))

    return {
        "seed": int(seed),
        "hypervolume": float(summary.get("hypervolume", 0.0)),
        "warmstart_hv_last": warmstart_hv_last,
        "n_feasible": int(summary.get("n_feasible", 0)),
    }


def _aggregate_run_set(run_dirs: List[Path], seeds: List[int]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for seed, run_dir in zip(seeds, run_dirs):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append(_run_record(seed, summary))

    hv_values = [float(run["hypervolume"]) for run in runs]
    init_hv_values = [
        float(run["warmstart_hv_last"])
        for run in runs
        if run["warmstart_hv_last"] is not None
    ]
    return {
        "mean_hv": _mean(hv_values),
        "std_hv": _std(hv_values),
        "min_hv": float(min(hv_values)) if hv_values else 0.0,
        "max_hv": float(max(hv_values)) if hv_values else 0.0,
        "mean_init_hv": _mean(init_hv_values),
        "runs": runs,
    }


def _comparison(lhs: Dict[str, Any], rhs: Dict[str, Any]) -> Dict[str, Any]:
    lhs_by_seed = {int(run["seed"]): run for run in lhs.get("runs", [])}
    rhs_by_seed = {int(run["seed"]): run for run in rhs.get("runs", [])}
    shared_seeds = sorted(set(lhs_by_seed) & set(rhs_by_seed))

    deltas: List[float] = []
    pct_deltas: List[float] = []
    wins = 0
    for seed in shared_seeds:
        lhs_hv = float(lhs_by_seed[seed]["hypervolume"])
        rhs_hv = float(rhs_by_seed[seed]["hypervolume"])
        delta = lhs_hv - rhs_hv
        deltas.append(delta)
        if abs(rhs_hv) > 1e-12:
            pct_deltas.append(delta / rhs_hv * 100.0)
        if delta > 0:
            wins += 1

    return {
        "mean_delta": _mean(deltas),
        "mean_pct": _mean(pct_deltas),
        "wins": int(wins),
        "total": int(len(shared_seeds)),
    }


def _common_config(output_dir: Path, iterations: int, seed: int) -> Dict[str, Any]:
    return {
        "max_iterations": int(iterations),
        "n_candidates": 15,
        "n_select": 2,
        "w_sample_seed": int(seed),
        "init_seed": int(2026 + seed),
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_llm_rerank": False,
        "enable_proposal_sampler": False,
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "checkpoint_every": 99,
    }


def _baseline_config(output_dir: Path, iterations: int, seed: int) -> Dict[str, Any]:
    cfg = _common_config(output_dir, iterations, seed)
    cfg.update({
        "n_warmstart": 0,
        "n_random_init": 6,
        "llm_backend": "mock",
        "llm_api_key": "",
    })
    return cfg


def _warmstart_config(
    output_dir: Path,
    iterations: int,
    seed: int,
    api_key: str,
    api_base: str,
    model: str,
    prompt_version: str = "full",  # "full" for detailed, "experimental" for experimental
) -> Dict[str, Any]:
    cfg = _common_config(output_dir, iterations, seed)
    cfg.update({
        "n_warmstart": 6,
        "n_random_init": 0,
        "warmstart_batch_size": 6,
        "warmstart_max_attempts": 1,
        "llm_backend": "openai",
        "llm_model": model,
        "llm_api_base": api_base,
        "llm_api_key": api_key,
        "llm_n_samples": 1,
        "llm_temperature": 0.0,
        "warmstart_temperature": 0.0,
        "warmstart_prompt_version": prompt_version,
        "llm_safe_dsoc_sum_max": 0.695,
        "warmstart_cache_path": None,
        "warmstart_cache_mode": "disabled",
    })
    return cfg


def _run_single(output_dir: Path, cfg: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = BayesOptimizer(config=cfg)
    optimizer.run()
    optimizer.save_results(str(output_dir))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key and not args.skip_experimental and not args.skip_detailed:
        raise RuntimeError("API key required. Set LLM_API_KEY or pass --api-key.")

    # Setup directories
    baseline_dirs = [output_root / f"baseline_seed{seed}" for seed in args.seeds]
    detailed_dirs = [output_root / f"detailed_prompt_seed{seed}" for seed in args.seeds]
    experimental_dirs = [output_root / f"experimental_prompt_seed{seed}" for seed in args.seeds]

    # Run baseline
    if not args.skip_baseline:
        print("\n=== Running Baseline (no LLM warmstart) ===")
        for seed, output_dir in zip(args.seeds, baseline_dirs):
            print(f"[baseline] seed={seed} -> {output_dir}")
            cfg = _baseline_config(output_dir, args.iterations, seed)
            _run_single(output_dir, cfg)

    # Run detailed prompt
    if not args.skip_detailed:
        print("\n=== Running Detailed Prompt ===")
        for seed, output_dir in zip(args.seeds, detailed_dirs):
            print(f"[detailed] seed={seed} -> {output_dir}")
            cfg = _warmstart_config(
                output_dir, args.iterations, seed,
                api_key, args.api_base, args.model,
                prompt_version="full"
            )
            _run_single(output_dir, cfg)

    # Run experimental prompt
    if not args.skip_experimental:
        print("\n=== Running Experimental Prompt ===")
        for seed, output_dir in zip(args.seeds, experimental_dirs):
            print(f"[experimental] seed={seed} -> {output_dir}")
            cfg = _warmstart_config(
                output_dir, args.iterations, seed,
                api_key, args.api_base, args.model,
                prompt_version="experimental"
            )
            _run_single(output_dir, cfg)

    # Generate report
    report: Dict[str, Any] = {
        "meta": {
            "iterations": int(args.iterations),
            "seeds": [int(seed) for seed in args.seeds],
            "output_root": str(output_root),
            "model": args.model,
            "model_display": canonical_model_label(args.model),
            "api_base": args.api_base,
        }
    }

    if not args.skip_baseline:
        report["baseline"] = _aggregate_run_set(baseline_dirs, args.seeds)
    if not args.skip_detailed:
        report["detailed_prompt"] = _aggregate_run_set(detailed_dirs, args.seeds)
    if not args.skip_experimental:
        report["experimental_prompt"] = _aggregate_run_set(experimental_dirs, args.seeds)

    # Comparisons
    if not args.skip_baseline and not args.skip_detailed:
        report["detailed_vs_baseline"] = _comparison(
            report["detailed_prompt"], report["baseline"]
        )
    if not args.skip_baseline and not args.skip_experimental:
        report["experimental_vs_baseline"] = _comparison(
            report["experimental_prompt"], report["baseline"]
        )
    if not args.skip_detailed and not args.skip_experimental:
        report["experimental_vs_detailed"] = _comparison(
            report["experimental_prompt"], report["detailed_prompt"]
        )

    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[report] {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
