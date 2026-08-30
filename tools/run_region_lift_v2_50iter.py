from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer
from utils.model_labels import canonical_model_label


DEFAULT_VARIANTS = [
    "strict_baseline",
    "parego_baseline",
    "warmstart_plain_ei",
    "warmstart_region_lifted_gp",
]
CORRECTED_REGION_VARIANTS = [
    "warmstart_region_lgbo_proposition1",
    "sham_region_lgbo_proposition1",
    "random_region_lgbo_proposition1",
]
SUPPORTED_VARIANTS = DEFAULT_VARIANTS + [
    "warmstart_region_lifted_gp_guarded_pool",
    "warmstart_region_lifted_gp_force_pool_tuned",
] + CORRECTED_REGION_VARIANTS
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "optimized_experiments"
    / f"baseline_warmstart_llmgp_50iter_seed01234_{date.today().isoformat().replace('-', '_')}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 50-iteration, 5-seed experiments for baseline / warmstart / warmstart+LLMGP."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for all seed/variant run folders and report_5seeds.json",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="BO iterations per run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds used for all variants",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=SUPPORTED_VARIANTS,
        help="Preset variants to run",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip execution and rebuild report_5seeds.json from existing run folders",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose summary.json already exists",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        help="LLM model for warmstart variants",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.nuwaapi.com/v1"),
        help="LLM API base URL for warmstart variants",
    )
    parser.add_argument(
        "--two-stage-gate",
        action="store_true",
        help="Run a 2-seed pilot first and only run the final 5-seed stage if the pilot passes.",
    )
    parser.add_argument(
        "--pilot-iterations",
        type=int,
        default=20,
        help="Iterations used by the pilot stage when --two-stage-gate is enabled.",
    )
    parser.add_argument(
        "--pilot-seeds",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Seeds used by the pilot stage when --two-stage-gate is enabled.",
    )
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.fmean(items)) if items else 0.0


def _median(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.median(items)) if items else 0.0


def _variance(values: Iterable[float]) -> float:
    items = list(values)
    if len(items) < 2:
        return 0.0
    return float(statistics.pvariance(items))


def _worst_quartile(values: Iterable[float]) -> float:
    items = sorted(float(v) for v in values)
    if not items:
        return 0.0
    count = max(1, len(items) // 4)
    return float(_mean(items[:count]))


def _count_hv_violations(summary: Dict[str, Any]) -> int:
    hv_trace = summary.get("hv_trace") or []
    values = [float(item.get("hypervolume_raw", item.get("hypervolume", 0.0))) for item in hv_trace]
    violations = 0
    for prev, curr in zip(values, values[1:]):
        if curr + 1e-12 < prev:
            violations += 1
    return violations


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_record(seed: int, variant: str, run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {
            "seed": int(seed),
            "variant": variant,
            "status": "missing",
            "summary_path": str(summary_path),
        }

    summary = _load_json(summary_path)
    fallback_distribution = dict(summary.get("region_lift_fallback_reasons") or {})
    return {
        "seed": int(seed),
        "variant": variant,
        "status": "ok",
        "summary_path": str(summary_path),
        "display_hv": float(summary.get("display_hv", summary.get("hypervolume", 0.0))),
        "canonical_hv": float(summary.get("canonical_hv", summary.get("hypervolume_canonical", 0.0))),
        "hypervolume_raw": float(summary.get("hypervolume_raw", 0.0)),
        "hv_violations": int(_count_hv_violations(summary)),
        "pareto_size": int(summary.get("pareto_size", 0)),
        "n_total": int(summary.get("n_total", 0)),
        "n_feasible": int(summary.get("n_feasible", 0)),
        "lift_accept_rate": float(
            summary.get("lift_accept_rate", summary.get("region_lift_accept_rate", 0.0))
        ),
        "acquisition_used_lift_count": int(summary.get("acquisition_used_lift_count", 0)),
        "acquisition_used_lift_rate": float(summary.get("acquisition_used_lift_rate", 0.0)),
        "selection_guard_pass_count": int(summary.get("selection_guard_pass_count", 0)),
        "selection_guard_pass_rate": float(summary.get("selection_guard_pass_rate", 0.0)),
        "effective_lift_accept_rate": float(summary.get("effective_lift_accept_rate", 0.0)),
        "effective_lift_accept_count": int(summary.get("effective_lift_accept_count", 0)),
        "effective_selection_change_count": int(
            summary.get("effective_selection_change_count", summary.get("effective_lift_accept_count", 0))
        ),
        "effective_selection_change_rate": float(
            summary.get("effective_selection_change_rate", summary.get("effective_lift_accept_rate", 0.0))
        ),
        "plain_candidate_inside_region_count": int(summary.get("plain_candidate_inside_region_count", 0)),
        "diagnostic_override_candidate_count": int(summary.get("diagnostic_override_candidate_count", 0)),
        "region_pool_influenced_acquisition_count": int(summary.get("region_pool_influenced_acquisition_count", 0)),
        "region_influence_gate_pass_count": int(summary.get("region_influence_gate_pass_count", 0)),
        "inactive_window_skipped_count": int(summary.get("inactive_window_skipped_count", 0)),
        "zero_shift_accept_count": int(summary.get("zero_shift_accept_count", 0)),
        "fallback_distribution": fallback_distribution,
    }


def _aggregate_variant(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_runs = [record for record in records if record.get("status") == "ok"]
    canonical = [float(record["canonical_hv"]) for record in ok_runs]
    display = [float(record["display_hv"]) for record in ok_runs]
    raw = [float(record["hypervolume_raw"]) for record in ok_runs]
    accept_rate = [float(record.get("lift_accept_rate", 0.0)) for record in ok_runs]
    acquisition_used_rate = [
        float(record.get("acquisition_used_lift_rate", 0.0)) for record in ok_runs
    ]
    guard_pass_rate = [
        float(record.get("selection_guard_pass_rate", 0.0)) for record in ok_runs
    ]
    effective_rate = [float(record.get("effective_lift_accept_rate", 0.0)) for record in ok_runs]
    effective_change_rate = [
        float(record.get("effective_selection_change_rate", 0.0)) for record in ok_runs
    ]
    fallback_counter: Counter[str] = Counter()
    for record in ok_runs:
        fallback_counter.update(record.get("fallback_distribution") or {})

    return {
        "n_runs": int(len(ok_runs)),
        "mean_canonical_hv": _mean(canonical),
        "median_canonical_hv": _median(canonical),
        "worst_quartile_canonical_hv": _worst_quartile(canonical),
        "variance_canonical_hv": _variance(canonical),
        "mean_display_hv": _mean(display),
        "mean_raw_hv": _mean(raw),
        "hv_violations_total": int(sum(int(record.get("hv_violations", 0)) for record in ok_runs)),
        "mean_lift_accept_rate": _mean(accept_rate),
        "mean_acquisition_used_lift_rate": _mean(acquisition_used_rate),
        "acquisition_used_lift_count_total": int(
            sum(int(record.get("acquisition_used_lift_count", 0)) for record in ok_runs)
        ),
        "mean_selection_guard_pass_rate": _mean(guard_pass_rate),
        "selection_guard_pass_count_total": int(
            sum(int(record.get("selection_guard_pass_count", 0)) for record in ok_runs)
        ),
        "mean_effective_lift_accept_rate": _mean(effective_rate),
        "effective_lift_accept_count_total": int(sum(int(record.get("effective_lift_accept_count", 0)) for record in ok_runs)),
        "mean_effective_selection_change_rate": _mean(effective_change_rate),
        "effective_selection_change_count_total": int(
            sum(int(record.get("effective_selection_change_count", 0)) for record in ok_runs)
        ),
        "plain_candidate_inside_region_count_total": int(
            sum(int(record.get("plain_candidate_inside_region_count", 0)) for record in ok_runs)
        ),
        "diagnostic_override_candidate_count_total": int(
            sum(int(record.get("diagnostic_override_candidate_count", 0)) for record in ok_runs)
        ),
        "region_pool_influenced_acquisition_count_total": int(
            sum(int(record.get("region_pool_influenced_acquisition_count", 0)) for record in ok_runs)
        ),
        "region_influence_gate_pass_count_total": int(
            sum(int(record.get("region_influence_gate_pass_count", 0)) for record in ok_runs)
        ),
        "inactive_window_skipped_count_total": int(
            sum(int(record.get("inactive_window_skipped_count", 0)) for record in ok_runs)
        ),
        "zero_shift_accept_count_total": int(sum(int(record.get("zero_shift_accept_count", 0)) for record in ok_runs)),
        "fallback_distribution": dict(fallback_counter),
        "runs": ok_runs,
        "missing_runs": [record for record in records if record.get("status") != "ok"],
    }


def _comparison(lhs: List[Dict[str, Any]], rhs: List[Dict[str, Any]]) -> Dict[str, Any]:
    lhs_by_seed = {int(record["seed"]): record for record in lhs if record.get("status") == "ok"}
    rhs_by_seed = {int(record["seed"]): record for record in rhs if record.get("status") == "ok"}
    shared_seeds = sorted(set(lhs_by_seed) & set(rhs_by_seed))
    deltas: List[float] = []
    pct_deltas: List[float] = []
    wins = 0
    for seed in shared_seeds:
        lhs_hv = float(lhs_by_seed[seed]["canonical_hv"])
        rhs_hv = float(rhs_by_seed[seed]["canonical_hv"])
        delta = lhs_hv - rhs_hv
        deltas.append(delta)
        if abs(rhs_hv) > 1e-12:
            pct_deltas.append(delta / rhs_hv * 100.0)
        if delta > 0.0:
            wins += 1
    return {
        "lhs": lhs[0]["variant"] if lhs else None,
        "rhs": rhs[0]["variant"] if rhs else None,
        "shared_seeds": shared_seeds,
        "mean_canonical_hv_delta": _mean(deltas),
        "mean_canonical_hv_pct_delta": _mean(pct_deltas),
        "wins": int(wins),
        "total": int(len(shared_seeds)),
    }


def _build_config(
    *,
    variant: str,
    output_dir: Path,
    shared_random_init_cache: Path,
    shared_warmstart_cache: Path,
    iterations: int,
    seed: int,
    api_key: str,
    api_base: str,
    model: str,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "experiment_preset": variant,
        "max_iterations": int(iterations),
        "w_sample_seed": int(seed),
        "init_seed": int(2026 + seed),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "checkpoint_every": 99,
        "random_init_cache_path": str(shared_random_init_cache),
    }
    if variant in {"strict_baseline", "parego_baseline"}:
        cfg.update(
            {
                "llm_backend": "mock",
                "llm_api_key": "",
            }
        )
    else:
        cfg.update(
            {
                "llm_backend": "openai",
                "llm_model": model,
                "llm_api_base": api_base,
                "llm_api_key": api_key,
                "llm_n_samples": 1,
                "llm_temperature": 0.0,
                "warmstart_temperature": 0.0,
                "warmstart_cache_path": str(shared_warmstart_cache),
            }
        )
        if variant == "warmstart_plain_ei":
            cfg.update(
                {
                    "warmstart_cache_mode": "write",
                    "warmstart_cache_use_selected": False,
                }
            )
        elif variant == "warmstart_region_lifted_gp":
            cfg.update(
                {
                    "warmstart_cache_mode": "read",
                    "warmstart_cache_use_selected": True,
                }
            )
        elif variant in {
            "warmstart_region_lifted_gp_guarded_pool",
            "warmstart_region_lifted_gp_force_pool_tuned",
            *CORRECTED_REGION_VARIANTS,
        }:
            cfg.update(
                {
                    "warmstart_cache_mode": "read",
                    "warmstart_cache_use_selected": True,
                }
            )
    return cfg


def _run_single(output_dir: Path, cfg: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = BayesOptimizer(config=cfg)
    optimizer.run()
    optimizer.save_results(str(output_dir))


def _print_banner(variant: str, seed: int, output_dir: Path) -> None:
    print(f"[run] variant={variant} seed={seed} -> {output_dir}")


def _write_report(
    args: argparse.Namespace,
    output_root: Path,
    *,
    iterations: Optional[int] = None,
    seeds: Optional[List[int]] = None,
    variants: Optional[List[str]] = None,
    report_name: str = "report_5seeds.json",
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    report_variants = list(variants or args.variants)
    report_seeds = [int(seed) for seed in (seeds or args.seeds)]
    for variant in report_variants:
        variant_records: List[Dict[str, Any]] = []
        for seed in report_seeds:
            run_dir = output_root / f"seed{seed}" / variant
            record = _run_record(seed, variant, run_dir)
            records.append(record)
            variant_records.append(record)
        by_variant[variant] = variant_records

    report: Dict[str, Any] = {
        "meta": {
            "iterations": int(args.iterations if iterations is None else iterations),
            "seeds": report_seeds,
            "variants": report_variants,
            "output_root": str(output_root),
            "api_base": args.api_base,
            "model": args.model,
            "model_display": canonical_model_label(args.model),
        },
        "records": records,
        "aggregates": {variant: _aggregate_variant(variant_records) for variant, variant_records in by_variant.items()},
        "comparisons": {},
    }

    if "warmstart_plain_ei" in by_variant and "strict_baseline" in by_variant:
        report["comparisons"]["warmstart_plain_ei_vs_strict_baseline"] = _comparison(
            by_variant["warmstart_plain_ei"],
            by_variant["strict_baseline"],
        )
    if "parego_baseline" in by_variant and "strict_baseline" in by_variant:
        report["comparisons"]["parego_baseline_vs_strict_baseline"] = _comparison(
            by_variant["parego_baseline"],
            by_variant["strict_baseline"],
        )
    if "parego_baseline" in by_variant and "warmstart_plain_ei" in by_variant:
        report["comparisons"]["parego_baseline_vs_warmstart_plain_ei"] = _comparison(
            by_variant["parego_baseline"],
            by_variant["warmstart_plain_ei"],
        )
    if "parego_baseline" in by_variant and "warmstart_region_lifted_gp" in by_variant:
        report["comparisons"]["parego_baseline_vs_warmstart_region_lifted_gp"] = _comparison(
            by_variant["parego_baseline"],
            by_variant["warmstart_region_lifted_gp"],
        )
    if "warmstart_region_lifted_gp" in by_variant and "warmstart_plain_ei" in by_variant:
        report["comparisons"]["warmstart_region_lifted_gp_vs_warmstart_plain_ei"] = _comparison(
            by_variant["warmstart_region_lifted_gp"],
            by_variant["warmstart_plain_ei"],
        )
    if "warmstart_region_lifted_gp" in by_variant and "strict_baseline" in by_variant:
        report["comparisons"]["warmstart_region_lifted_gp_vs_strict_baseline"] = _comparison(
            by_variant["warmstart_region_lifted_gp"],
            by_variant["strict_baseline"],
        )
    if "warmstart_region_lifted_gp_guarded_pool" in by_variant and "warmstart_plain_ei" in by_variant:
        report["comparisons"]["warmstart_region_lifted_gp_guarded_pool_vs_warmstart_plain_ei"] = _comparison(
            by_variant["warmstart_region_lifted_gp_guarded_pool"],
            by_variant["warmstart_plain_ei"],
        )
    if "warmstart_region_lifted_gp_force_pool_tuned" in by_variant and "warmstart_plain_ei" in by_variant:
        report["comparisons"]["warmstart_region_lifted_gp_force_pool_tuned_vs_warmstart_plain_ei"] = _comparison(
            by_variant["warmstart_region_lifted_gp_force_pool_tuned"],
            by_variant["warmstart_plain_ei"],
        )
    for region_variant in CORRECTED_REGION_VARIANTS:
        if region_variant in by_variant and "warmstart_plain_ei" in by_variant:
            report["comparisons"][f"{region_variant}_vs_warmstart_plain_ei"] = _comparison(
                by_variant[region_variant],
                by_variant["warmstart_plain_ei"],
            )
        if region_variant in by_variant and "strict_baseline" in by_variant:
            report["comparisons"][f"{region_variant}_vs_strict_baseline"] = _comparison(
                by_variant[region_variant],
                by_variant["strict_baseline"],
            )
    corrected = "warmstart_region_lgbo_proposition1"
    for control in ("sham_region_lgbo_proposition1", "random_region_lgbo_proposition1"):
        if corrected in by_variant and control in by_variant:
            report["comparisons"][f"{corrected}_vs_{control}"] = _comparison(
                by_variant[corrected],
                by_variant[control],
            )

    report_path = output_root / report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    report["report_path"] = str(report_path)
    return report


def _gate_conditions_met(report: Dict[str, Any]) -> Dict[str, Any]:
    aggregates = report.get("aggregates") or {}
    comparisons = report.get("comparisons") or {}
    corrected_name = "warmstart_region_lgbo_proposition1"
    region_name = corrected_name if corrected_name in aggregates else "warmstart_region_lifted_gp"
    region = aggregates.get(region_name) or {}
    delta_vs_strict = float(
        (comparisons.get(f"{region_name}_vs_strict_baseline") or {}).get("mean_canonical_hv_delta", -1.0)
    )
    delta_vs_plain = float(
        (comparisons.get(f"{region_name}_vs_warmstart_plain_ei") or {}).get("mean_canonical_hv_delta", -1.0)
    )
    if region_name == corrected_name:
        region_signal_total = min(
            int(region.get("acquisition_used_lift_count_total", 0)),
            int(region.get("selection_guard_pass_count_total", 0)),
        )
        control_key = f"{corrected_name}_vs_sham_region_lgbo_proposition1"
        control_comparison = comparisons.get(control_key)
        delta_vs_control = (
            None
            if control_comparison is None
            else float(control_comparison.get("mean_canonical_hv_delta", -1.0))
        )
        control_passed = delta_vs_control is None or delta_vs_control >= 0.0
    else:
        region_signal_total = int(region.get("plain_candidate_inside_region_count_total", 0)) + int(
            region.get("diagnostic_override_candidate_count_total", 0)
        )
        delta_vs_control = None
        control_passed = True
    passed = bool(
        delta_vs_strict > 0.0
        and delta_vs_plain >= 0.0
        and region_signal_total > 0
        and control_passed
    )
    return {
        "passed": passed,
        "region_variant": region_name,
        "delta_vs_strict": delta_vs_strict,
        "delta_vs_plain": delta_vs_plain,
        "delta_vs_control": delta_vs_control,
        "region_signal_total": region_signal_total,
        # Backward-compatible output key used by older report consumers.
        "region_diag_total": region_signal_total,
    }


def _run_stage(
    *,
    output_root: Path,
    iterations: int,
    seeds: List[int],
    variants: List[str],
    args: argparse.Namespace,
    api_key: str,
    report_name: str,
) -> Dict[str, Any]:
    if not args.summarize_only:
        for seed in seeds:
            seed_root = output_root / f"seed{seed}"
            shared_random = seed_root / "shared_random_init_cache.json"
            shared_warm = seed_root / "shared_warmstart_cache.json"
            for variant in variants:
                run_dir = seed_root / variant
                summary_path = run_dir / "summary.json"
                if args.skip_existing and summary_path.exists():
                    print(f"[skip] variant={variant} seed={seed} already has {summary_path}")
                    continue
                _print_banner(variant, seed, run_dir)
                cfg = _build_config(
                    variant=variant,
                    output_dir=run_dir,
                    shared_random_init_cache=shared_random,
                    shared_warmstart_cache=shared_warm,
                    iterations=iterations,
                    seed=seed,
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                )
                _run_single(run_dir, cfg)

    return _write_report(
        args,
        output_root,
        iterations=iterations,
        seeds=seeds,
        variants=variants,
        report_name=report_name,
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    llm_variants = [
        variant
        for variant in args.variants
        if variant not in {"strict_baseline", "parego_baseline"}
    ]
    if llm_variants and not args.summarize_only and not api_key:
        raise RuntimeError(
            "WarmStart / WarmStart+LLMGP require LLM_API_KEY or OPENAI_API_KEY when not using --summarize-only."
        )

    if args.two_stage_gate:
        pilot_root = output_root / f"pilot_{len(args.pilot_seeds)}seeds_{int(args.pilot_iterations)}iter"
        pilot_report = _run_stage(
            output_root=pilot_root,
            iterations=int(args.pilot_iterations),
            seeds=[int(seed) for seed in args.pilot_seeds],
            variants=list(args.variants),
            args=args,
            api_key=api_key,
            report_name="report_2seeds.json",
        )
        gate = _gate_conditions_met(pilot_report)
        gate_path = output_root / "gate_result.json"
        gate_payload = {
            "pilot_output_root": str(pilot_root),
            "pilot_report_path": pilot_report.get("report_path"),
            **gate,
        }
        gate_path.write_text(json.dumps(gate_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[gate] {json.dumps(gate_payload, ensure_ascii=False)}")
        if not gate["passed"]:
            return

        final_root = output_root / f"final_{len(args.seeds)}seeds_{int(args.iterations)}iter"
        _run_stage(
            output_root=final_root,
            iterations=int(args.iterations),
            seeds=[int(seed) for seed in args.seeds],
            variants=list(args.variants),
            args=args,
            api_key=api_key,
            report_name="report_5seeds.json",
        )
        return

    _run_stage(
        output_root=output_root,
        iterations=int(args.iterations),
        seeds=[int(seed) for seed in args.seeds],
        variants=list(args.variants),
        args=args,
        api_key=api_key,
        report_name="report_5seeds.json",
    )


if __name__ == "__main__":
    main()
