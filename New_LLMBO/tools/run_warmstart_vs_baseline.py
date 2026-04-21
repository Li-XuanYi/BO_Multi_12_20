from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "optimized_experiments" / "warmstart_prompt_v2_10iter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired multi-seed experiments for real WarmStart vs baseline."
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
        default=[0, 1, 2],
        help="Paired random seeds used for baseline and warmstart runs",
    )
    parser.add_argument(
        "--allow-mock-warmstart",
        action="store_true",
        help="Allow warmstart runs to fall back to mock if no API key is present",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Do not run the baseline side",
    )
    parser.add_argument(
        "--skip-warmstart",
        action="store_true",
        help="Do not run the warmstart side",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip execution and only rebuild report.json from existing run folders",
    )
    return parser.parse_args()


def _mean(values: List[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _count_hv_violations(summary: Dict[str, Any]) -> int:
    hv_trace = summary.get("hv_trace") or []
    values = [float(item.get("hypervolume", 0.0)) for item in hv_trace]
    violations = 0
    for prev, curr in zip(values, values[1:]):
        if curr + 1e-12 < prev:
            violations += 1
    return violations


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
        "hv_violations": _count_hv_violations(summary),
        "candidate_source_counts": dict(summary.get("last_candidate_source_counts") or {}),
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
    cfg.update(
        {
            "n_warmstart": 0,
            "n_random_init": 6,
            "llm_backend": "mock",
            "llm_api_key": "",
        }
    )
    return cfg


def _warmstart_config(
    output_dir: Path,
    iterations: int,
    seed: int,
    api_key: str,
    *,
    model: str,
    api_base: str,
) -> Dict[str, Any]:
    cfg = _common_config(output_dir, iterations, seed)
    cfg.update(
        {
            "n_warmstart": 3,
            "n_random_init": 3,
            "warmstart_batch_size": 3,
            "warmstart_max_attempts": 1,
            "llm_backend": "openai",
            "llm_model": model,
            "llm_api_base": api_base,
            "llm_api_key": api_key,
            "llm_n_samples": 1,
            "llm_temperature": 0.0,
            "warmstart_temperature": 0.0,
            "llm_safe_dsoc_sum_max": 0.695,
        }
    )
    return cfg


def _run_single(output_dir: Path, cfg: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = BayesOptimizer(config=cfg)
    optimizer.run()
    optimizer.save_results(str(output_dir))


def _print_run_banner(name: str, seed: int, output_dir: Path) -> None:
    print(f"[run] {name} seed={seed} -> {output_dir}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    api_base = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.nuwaapi.com/v1"
    model = os.getenv("LLM_MODEL") or "gpt-4.1-mini"

    if not args.skip_warmstart and not args.allow_mock_warmstart and not api_key:
        raise RuntimeError(
            "WarmStart requires a real API key. Set LLM_API_KEY/OPENAI_API_KEY or pass --allow-mock-warmstart."
        )

    baseline_dirs = [output_root / f"baseline_strict_seed{seed}" for seed in args.seeds]
    warmstart_dirs = [output_root / f"warmstart_prompt_v2_seed{seed}" for seed in args.seeds]

    if not args.summarize_only:
        if not args.skip_baseline:
            for seed, output_dir in zip(args.seeds, baseline_dirs):
                _print_run_banner("baseline_strict", seed, output_dir)
                cfg = _baseline_config(output_dir, args.iterations, seed)
                _run_single(output_dir, cfg)

        if not args.skip_warmstart:
            for seed, output_dir in zip(args.seeds, warmstart_dirs):
                _print_run_banner("warmstart_prompt_v2", seed, output_dir)
                if args.allow_mock_warmstart and not api_key:
                    cfg = _baseline_config(output_dir, args.iterations, seed)
                    cfg.update(
                        {
                            "n_warmstart": 3,
                            "n_random_init": 3,
                            "warmstart_batch_size": 3,
                            "warmstart_max_attempts": 1,
                        }
                    )
                else:
                    cfg = _warmstart_config(
                        output_dir,
                        args.iterations,
                        seed,
                        api_key,
                        model=model,
                        api_base=api_base,
                    )
                _run_single(output_dir, cfg)

    report: Dict[str, Any] = {
        "meta": {
            "iterations": int(args.iterations),
            "seeds": [int(seed) for seed in args.seeds],
            "output_root": str(output_root),
            "model": model,
            "api_base": api_base,
            "warmstart_backend": "mock" if (args.allow_mock_warmstart and not api_key) else "openai",
        }
    }

    if not args.skip_baseline:
        report["baseline_strict"] = _aggregate_run_set(baseline_dirs, args.seeds)
    if not args.skip_warmstart:
        report["warmstart_prompt_v2"] = _aggregate_run_set(warmstart_dirs, args.seeds)
    if not args.skip_baseline and not args.skip_warmstart:
        report["comparison_vs_baseline_warmstart_prompt_v2"] = _comparison(
            report["warmstart_prompt_v2"],
            report["baseline_strict"],
        )

    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
