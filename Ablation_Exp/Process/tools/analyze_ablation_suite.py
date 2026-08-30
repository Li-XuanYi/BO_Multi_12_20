"""Audit and aggregate the strongest archived ablation experiment batches.

The historical reports use more than one hypervolume display convention and
do not consistently expose paired statistical tests.  This script reads the
per-run ``summary.json`` files, normalises reporting to ``canonical_hv``,
verifies the evaluation protocol, checks paired initial designs, records
configuration confounds, and writes a reviewer-facing report.

No optimiser or LLM API call is made by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    color: str
    directory_pattern: str


@dataclass(frozen=True)
class ComparisonSpec:
    key: str
    lhs: str
    rhs: str
    expect_same_initialization: bool = False


@dataclass(frozen=True)
class GroupSpec:
    key: str
    title: str
    report_path: str
    expected_evaluations: int
    variants: Tuple[VariantSpec, ...]
    comparisons: Tuple[ComparisonSpec, ...]
    role: str


FOUR_WAY_VARIANTS = (
    VariantSpec("baseline", "Plain BO", "#3B4A54", "seed{seed}/baseline"),
    VariantSpec(
        "baseline_warmstart",
        "Warm start",
        "#2E8BC8",
        "seed{seed}/baseline_warmstart",
    ),
    VariantSpec(
        "baseline_llm_region",
        "Region",
        "#F0A33A",
        "seed{seed}/baseline_llm_region",
    ),
    VariantSpec("llmbo_mo", "Full", "#D45162", "seed{seed}/llmbo_mo"),
)

FOUR_WAY_COMPARISONS = (
    ComparisonSpec("warm_vs_plain", "baseline_warmstart", "baseline"),
    ComparisonSpec(
        "region_vs_plain",
        "baseline_llm_region",
        "baseline",
        expect_same_initialization=True,
    ),
    ComparisonSpec("full_vs_plain", "llmbo_mo", "baseline"),
    ComparisonSpec(
        "full_vs_warm",
        "llmbo_mo",
        "baseline_warmstart",
        expect_same_initialization=True,
    ),
    ComparisonSpec("full_vs_region", "llmbo_mo", "baseline_llm_region"),
)

GROUPS = (
    GroupSpec(
        key="same_batch_component_bundle",
        title="Same-batch component bundle",
        report_path=(
            "Ablation_Exp/experiment_records/"
            "adaptive4_5seeds_50iter_deepseek_v3_2026_05_22/report_5seeds.json"
        ),
        expected_evaluations=56,
        variants=FOUR_WAY_VARIANTS,
        comparisons=FOUR_WAY_COMPARISONS,
        role="Primary five-seed configuration ablation",
    ),
    GroupSpec(
        key="shared_warmstart_region_increment",
        title="Shared-initialisation Region increment",
        report_path=(
            "Ablation_Exp/experiment_records/"
            "warmstart_vs_llmbo_paired_5seeds_50iter_deepseek_v3_2026_05_23/"
            "report_5seeds.json"
        ),
        expected_evaluations=56,
        variants=(
            VariantSpec(
                "baseline_warmstart",
                "Warm start",
                "#2E8BC8",
                "seed{seed}/baseline_warmstart",
            ),
            VariantSpec("llmbo_mo", "Warm + Region", "#D45162", "seed{seed}/llmbo_mo"),
        ),
        comparisons=(
            ComparisonSpec(
                "region_increment",
                "llmbo_mo",
                "baseline_warmstart",
                expect_same_initialization=True,
            ),
        ),
        role="Paired check of the Region preset after a shared warm start",
    ),
    GroupSpec(
        key="independent_seed_robustness",
        title="Independent-seed robustness batch",
        report_path=(
            "Ablation_Exp/experiment_records/"
            "ablation_4way_5randomseeds_50iter_deepseek_v3_2026_05_14_180222_"
            "seeds_56702_53604_97885_98126_37310/report_5seeds.json"
        ),
        expected_evaluations=56,
        variants=FOUR_WAY_VARIANTS,
        comparisons=FOUR_WAY_COMPARISONS,
        role="Robustness/failure-mode replication on independent seeds",
    ),
    GroupSpec(
        key="warmstart_prompt",
        title="Warm-start prompt ablation",
        report_path="experiment_records/prompt_comparison_v3_10seeds_10iter/report.json",
        expected_evaluations=26,
        variants=(
            VariantSpec("baseline", "Random init", "#3B4A54", "baseline_seed{seed}"),
            VariantSpec(
                "detailed_prompt",
                "Detailed prompt",
                "#8E7CC3",
                "detailed_prompt_seed{seed}",
            ),
            VariantSpec(
                "experimental_prompt",
                "Experimental prompt",
                "#00A676",
                "experimental_prompt_seed{seed}",
            ),
        ),
        comparisons=(
            ComparisonSpec("detailed_vs_random", "detailed_prompt", "baseline"),
            ComparisonSpec("experimental_vs_random", "experimental_prompt", "baseline"),
            ComparisonSpec(
                "experimental_vs_detailed",
                "experimental_prompt",
                "detailed_prompt",
            ),
        ),
        role="Prompt-content ablation under the 26-evaluation batch protocol",
    ),
)


SAFE_CONFIG_KEYS = (
    "experiment_preset",
    "max_iterations",
    "n_select",
    "n_candidates",
    "n_warmstart",
    "n_random_init",
    "w_sample_seed",
    "init_seed",
    "battery_param_set",
    "llm_backend",
    "llm_model",
    "llm_temperature",
    "warmstart_temperature",
    "warmstart_prompt_version",
    "warmstart_cache_mode",
    "warmstart_cache_use_selected",
    "enable_region_lifted_gp",
    "region_lift_mode",
    "region_lift_active_until",
    "region_lift_apply_override",
    "region_lift_external_influence_mode",
    "ei_n_restarts",
    "ei_n_random_samples",
    "ei_n_external_restarts",
)

# These fields describe evaluation/acquisition budget rather than the intended
# WarmStart/Region intervention.  Any difference is therefore a confound.
CONTROL_CONFIG_KEYS = (
    "max_iterations",
    "n_select",
    "n_candidates",
    "w_sample_seed",
    "init_seed",
    "battery_param_set",
    "ei_n_restarts",
    "ei_n_random_samples",
    "ei_n_external_restarts",
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only non-secret fields needed for reproducibility auditing."""
    return {key: config.get(key) for key in SAFE_CONFIG_KEYS}


def _canonical_hv(summary: Mapping[str, Any]) -> float:
    for key in ("canonical_hv", "hypervolume_canonical"):
        value = summary.get(key)
        if value is not None:
            return float(value)
    raise KeyError("summary does not contain canonical_hv/hypervolume_canonical")


def _count_hv_violations(summary: Mapping[str, Any]) -> int:
    values: List[float] = []
    for item in summary.get("hv_trace") or []:
        if not isinstance(item, Mapping):
            continue
        value = item.get("canonical_hv", item.get("hypervolume_canonical"))
        if value is not None:
            values.append(float(value))
    return sum(curr + 1e-12 < prev for prev, curr in zip(values, values[1:]))


def _initialisation_fingerprint(summary_path: Path, n_initial: int) -> Optional[str]:
    database_path = summary_path.parent / "database.json"
    if not database_path.exists():
        database_path = summary_path.parent / "db_final.json"
    if not database_path.exists():
        return None
    payload = _read_json(database_path)
    observations = payload.get("observations") or []
    rows = []
    for item in observations[:n_initial]:
        rows.append(
            {
                "theta": [float(value) for value in item.get("theta", [])],
                "objectives": [float(value) for value in item.get("objectives", [])],
                "feasible": bool(item.get("feasible", False)),
            }
        )
    if len(rows) != n_initial:
        return None
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summary_path(group: GroupSpec, variant: VariantSpec, seed: int) -> Path:
    report_dir = PROJECT_ROOT / group.report_path
    return report_dir.parent / variant.directory_pattern.format(seed=seed) / "summary.json"


def load_group(group: GroupSpec) -> Dict[str, Any]:
    report_path = PROJECT_ROOT / group.report_path
    report = _read_json(report_path)
    meta = report.get("meta") or {}
    seeds = [int(seed) for seed in meta.get("seeds") or []]
    if not seeds:
        raise ValueError("No seeds declared in {}".format(report_path))

    observations: Dict[str, Dict[int, Dict[str, Any]]] = {}
    errors: List[str] = []
    warnings: List[str] = []
    for variant in group.variants:
        by_seed: Dict[int, Dict[str, Any]] = {}
        for seed in seeds:
            path = _summary_path(group, variant, seed)
            if not path.exists():
                errors.append("missing summary: {}".format(_relative(path)))
                continue
            summary = _read_json(path)
            config = summary.get("config") or {}
            n_total = int(summary.get("n_total", 0))
            n_feasible = int(summary.get("n_feasible", 0))
            n_initial = int(config.get("n_warmstart", 0) or 0) + int(
                config.get("n_random_init", 0) or 0
            )
            if n_total != group.expected_evaluations:
                errors.append(
                    "{} seed {} has {} evaluations, expected {}".format(
                        variant.key, seed, n_total, group.expected_evaluations
                    )
                )
            if n_feasible != n_total:
                warnings.append(
                    "{} seed {} has {}/{} feasible evaluations".format(
                        variant.key, seed, n_feasible, n_total
                    )
                )
            hv_violations = _count_hv_violations(summary)
            if hv_violations:
                errors.append(
                    "{} seed {} has {} decreasing canonical-HV steps".format(
                        variant.key, seed, hv_violations
                    )
                )
            by_seed[seed] = {
                "seed": seed,
                "canonical_hv": _canonical_hv(summary),
                "display_hv": float(summary.get("display_hv", summary.get("hypervolume", 0.0))),
                "pareto_size": int(summary.get("pareto_size", 0)),
                "n_total": n_total,
                "n_feasible": n_feasible,
                "summary_path": _relative(path),
                "initialisation_fingerprint": _initialisation_fingerprint(path, n_initial),
                "config": _safe_config(config),
                "telemetry": {
                    "region_lift_attempt_count": int(
                        summary.get("region_lift_attempt_count", 0) or 0
                    ),
                    "region_lift_accept_count": int(
                        summary.get("region_lift_accept_count", 0) or 0
                    ),
                    "effective_lift_accept_count": int(
                        summary.get("effective_lift_accept_count", 0) or 0
                    ),
                    "region_pool_influenced_acquisition_count": int(
                        summary.get("region_pool_influenced_acquisition_count", 0) or 0
                    ),
                    "fallback_reasons": dict(
                        summary.get("region_lift_fallback_reasons") or {}
                    ),
                },
            }
        observations[variant.key] = by_seed

    return {
        "key": group.key,
        "title": group.title,
        "role": group.role,
        "source_report": _relative(report_path),
        "source_meta": {
            key: meta.get(key)
            for key in (
                "experiment",
                "variant_set",
                "iterations",
                "seeds",
                "model",
                "model_display",
                "llm_backend",
                "created_at",
            )
        },
        "expected_evaluations": group.expected_evaluations,
        "seeds": seeds,
        "observations": observations,
        "integrity": {
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
        },
    }


def _sample_summary(values: Sequence[float]) -> Dict[str, Any]:
    items = [float(value) for value in values]
    n = len(items)
    if not items:
        return {
            "n": 0,
            "mean": None,
            "sample_std": None,
            "median": None,
            "min": None,
            "max": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = float(statistics.fmean(items))
    std = float(statistics.stdev(items)) if n > 1 else 0.0
    if n > 1:
        half = float(student_t.ppf(0.975, n - 1)) * std / math.sqrt(n)
    else:
        half = 0.0
    return {
        "n": n,
        "mean": mean,
        "sample_std": std,
        "median": float(statistics.median(items)),
        "min": min(items),
        "max": max(items),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def exact_sign_flip_p(deltas: Sequence[float]) -> float:
    """Two-sided exact paired randomisation p-value using mean difference."""
    values = [float(value) for value in deltas if abs(float(value)) > 1e-15]
    if not values:
        return 1.0
    observed = abs(sum(values))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(sum(sign * value for sign, value in zip(signs, values)))
        if statistic + 1e-15 >= observed:
            extreme += 1
    return float(extreme) / float(total)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm step-down family-wise correction preserving input order."""
    m = len(p_values)
    if not m:
        return []
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, float(p_values[index]) * (m - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _comparison_control_mismatches(
    lhs: Mapping[int, Mapping[str, Any]],
    rhs: Mapping[int, Mapping[str, Any]],
    seeds: Sequence[int],
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for key in CONTROL_CONFIG_KEYS:
        mismatches = []
        for seed in seeds:
            lhs_value = (lhs[seed].get("config") or {}).get(key)
            rhs_value = (rhs[seed].get("config") or {}).get(key)
            if lhs_value != rhs_value:
                mismatches.append({"seed": seed, "lhs": lhs_value, "rhs": rhs_value})
        if mismatches:
            result[key] = mismatches
    return result


def _paired_comparison(
    spec: ComparisonSpec,
    observations: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> Dict[str, Any]:
    lhs = observations[spec.lhs]
    rhs = observations[spec.rhs]
    seeds = sorted(set(lhs) & set(rhs))
    deltas = [float(lhs[seed]["canonical_hv"]) - float(rhs[seed]["canonical_hv"]) for seed in seeds]
    summary = _sample_summary(deltas)
    wins = sum(delta > 1e-15 for delta in deltas)
    ties = sum(abs(delta) <= 1e-15 for delta in deltas)
    losses = len(deltas) - wins - ties
    rhs_values = [float(rhs[seed]["canonical_hv"]) for seed in seeds]
    relative = [
        delta / value * 100.0
        for delta, value in zip(deltas, rhs_values)
        if abs(value) > 1e-15
    ]
    initial_matches = [
        lhs[seed].get("initialisation_fingerprint") is not None
        and lhs[seed].get("initialisation_fingerprint")
        == rhs[seed].get("initialisation_fingerprint")
        for seed in seeds
    ]
    std = float(summary["sample_std"] or 0.0)
    effect_size = float(summary["mean"]) / std if std > 1e-15 else None
    return {
        "key": spec.key,
        "lhs": spec.lhs,
        "rhs": spec.rhs,
        "shared_seeds": seeds,
        "per_seed_delta": [
            {"seed": seed, "delta": delta} for seed, delta in zip(seeds, deltas)
        ],
        "delta": summary,
        "mean_relative_delta_pct": float(statistics.fmean(relative)) if relative else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "cohen_dz": effect_size,
        "exact_sign_flip_p": exact_sign_flip_p(deltas),
        "holm_p_within_group": None,
        "expect_same_initialization": spec.expect_same_initialization,
        "same_initialization_count": sum(initial_matches),
        "same_initialization_total": len(initial_matches),
        "initialization_requirement_met": (
            all(initial_matches) if spec.expect_same_initialization else None
        ),
        "control_config_mismatches": _comparison_control_mismatches(lhs, rhs, seeds),
    }


def _aggregate_telemetry(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    fallback: Counter[str] = Counter()
    for record in records:
        telemetry = record.get("telemetry") or {}
        fallback.update(telemetry.get("fallback_reasons") or {})
    keys = (
        "region_lift_attempt_count",
        "region_lift_accept_count",
        "effective_lift_accept_count",
        "region_pool_influenced_acquisition_count",
    )
    return {
        "run_count": len(records),
        **{
            "{}_total".format(key): sum(
                int((record.get("telemetry") or {}).get(key, 0) or 0)
                for record in records
            )
            for key in keys
        },
        "fallback_reasons": dict(sorted(fallback.items())),
    }


def _nominal_factorial_effects(group: Mapping[str, Any]) -> List[Dict[str, Any]]:
    required = {"baseline", "baseline_warmstart", "baseline_llm_region", "llmbo_mo"}
    observations = group["observations"]
    if not required.issubset(observations):
        return []
    seeds = sorted(set.intersection(*(set(observations[key]) for key in required)))
    effects: Dict[str, List[float]] = {
        "warmstart_main_contrast": [],
        "region_main_contrast": [],
        "interaction_contrast": [],
    }
    for seed in seeds:
        y00 = float(observations["baseline"][seed]["canonical_hv"])
        y10 = float(observations["baseline_warmstart"][seed]["canonical_hv"])
        y01 = float(observations["baseline_llm_region"][seed]["canonical_hv"])
        y11 = float(observations["llmbo_mo"][seed]["canonical_hv"])
        effects["warmstart_main_contrast"].append(((y10 - y00) + (y11 - y01)) / 2.0)
        effects["region_main_contrast"].append(((y01 - y00) + (y11 - y10)) / 2.0)
        effects["interaction_contrast"].append(y11 - y10 - y01 + y00)
    output = []
    for key, values in effects.items():
        output.append(
            {
                "key": key,
                "seeds": seeds,
                "values": values,
                "estimate": _sample_summary(values),
                "exact_sign_flip_p": exact_sign_flip_p(values),
                "causal_interpretation_allowed": False,
                "reason": (
                    "Region-bearing configurations use a different external-EI restart budget "
                    "in the archived runs."
                ),
            }
        )
    return output


def analyse_group(group_spec: GroupSpec, loaded: Dict[str, Any]) -> Dict[str, Any]:
    observations = loaded["observations"]
    variant_summaries: Dict[str, Any] = {}
    for variant in group_spec.variants:
        records = observations[variant.key]
        values = [float(records[seed]["canonical_hv"]) for seed in sorted(records)]
        pareto = [float(records[seed]["pareto_size"]) for seed in sorted(records)]
        feasible = [
            float(records[seed]["n_feasible"]) / max(1.0, float(records[seed]["n_total"]))
            for seed in sorted(records)
        ]
        variant_summaries[variant.key] = {
            "label": variant.label,
            "color": variant.color,
            "canonical_hv": _sample_summary(values),
            "pareto_size": _sample_summary(pareto),
            "feasible_rate": _sample_summary(feasible),
            "values_by_seed": [
                {"seed": seed, "value": float(records[seed]["canonical_hv"])}
                for seed in sorted(records)
            ],
            "telemetry": _aggregate_telemetry(records.values()),
            "safe_config_values": {
                key: sorted(
                    {
                        json.dumps(record["config"].get(key), sort_keys=True)
                        for record in records.values()
                    }
                )
                for key in SAFE_CONFIG_KEYS
            },
        }

    comparisons = [
        _paired_comparison(comparison, observations) for comparison in group_spec.comparisons
    ]
    adjusted = holm_adjust([item["exact_sign_flip_p"] for item in comparisons])
    for item, p_value in zip(comparisons, adjusted):
        item["holm_p_within_group"] = p_value

    return {
        **{key: value for key, value in loaded.items() if key != "observations"},
        "observations": observations,
        "variant_summaries": variant_summaries,
        "comparisons": comparisons,
        "nominal_factorial_effects": _nominal_factorial_effects(loaded),
    }


def _fmt(value: Optional[float], digits: int = 5) -> str:
    if value is None:
        return "--"
    return ("{:.%df}" % digits).format(float(value))


def _comparison_by_key(group: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    for item in group["comparisons"]:
        if item["key"] == key:
            return item
    raise KeyError(key)


def _render_findings(groups: Mapping[str, Mapping[str, Any]]) -> List[str]:
    findings: List[str] = []
    primary = groups["same_batch_component_bundle"]
    warm = _comparison_by_key(primary, "warm_vs_plain")
    full_warm = _comparison_by_key(primary, "full_vs_warm")
    findings.append(
        "同批配置研究中，Warm 相对 Plain 的配对均值差为 {}（{}/{} seeds 胜出）；"
        "Full 相对 Warm 为 {}。Region-bearing arms 的 external EI restart 从 16 增至 32，"
        "且 Warm/Full 初始集未完全共享，因此这些数值是配置差异而非 lift 的独立因果效应。".format(
            _fmt(warm["delta"]["mean"]),
            warm["wins"],
            len(warm["shared_seeds"]),
            _fmt(full_warm["delta"]["mean"]),
        )
    )

    paired = groups["shared_warmstart_region_increment"]
    region = _comparison_by_key(paired, "region_increment")
    findings.append(
        "共享 WarmStart 初始点后，加入整个 Region preset 的均值差为 {}（{}/{} seeds 胜出）。"
        "该批次确认了初始点完全一致，但仍存在 16 对 32 external restart 的搜索预算混杂，"
        "所以只能判定 Region 配置没有观察到稳定增益。".format(
            _fmt(region["delta"]["mean"]),
            region["wins"],
            len(region["shared_seeds"]),
        )
    )

    robust = groups["independent_seed_robustness"]
    robust_warm = _comparison_by_key(robust, "warm_vs_plain")
    parse_fail = sum(
        int(summary["telemetry"]["fallback_reasons"].get("parse_fail", 0))
        for summary in robust["variant_summaries"].values()
    )
    findings.append(
        "独立随机 seed 复验中，Warm 相对 Plain 的均值差变为 {}；Region 相关运行累计 {} 次 "
        "parse_fail，且有效 lift 为零。该组作为负结果/鲁棒性边界保留，不用于证明 Region 有效。".format(
            _fmt(robust_warm["delta"]["mean"]), parse_fail
        )
    )

    prompt = groups["warmstart_prompt"]
    exp_random = _comparison_by_key(prompt, "experimental_vs_random")
    exp_detailed = _comparison_by_key(prompt, "experimental_vs_detailed")
    findings.append(
        "短预算 prompt 消融中，Experimental 相对 Random 的 canonical sHV 差为 {}（{}/{}），"
        "相对 Detailed 为 {}（{}/{}）；组内 Holm 校正 p 分别为 {} 和 {}。"
        "这支持 experimental prompt 在 Chen2020、26-evaluation 协议中的效果，但不能外推到 "
        "Region/lift 或 56-evaluation 主协议。".format(
            _fmt(exp_random["delta"]["mean"]),
            exp_random["wins"],
            len(exp_random["shared_seeds"]),
            _fmt(exp_detailed["delta"]["mean"]),
            exp_detailed["wins"],
            len(exp_detailed["shared_seeds"]),
            _fmt(exp_random["holm_p_within_group"], 4),
            _fmt(exp_detailed["holm_p_within_group"], 4),
        )
    )
    return findings


def write_variant_csv(path: Path, groups: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "group",
        "variant",
        "label",
        "n",
        "evaluations_per_run",
        "canonical_hv_mean",
        "canonical_hv_sample_std",
        "ci95_low",
        "ci95_high",
        "pareto_size_mean",
        "feasible_rate_mean",
        "effective_lift_count_total",
        "parse_fail_total",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            for key, item in group["variant_summaries"].items():
                hv = item["canonical_hv"]
                telemetry = item["telemetry"]
                writer.writerow(
                    {
                        "group": group["key"],
                        "variant": key,
                        "label": item["label"],
                        "n": hv["n"],
                        "evaluations_per_run": group["expected_evaluations"],
                        "canonical_hv_mean": hv["mean"],
                        "canonical_hv_sample_std": hv["sample_std"],
                        "ci95_low": hv["ci95_low"],
                        "ci95_high": hv["ci95_high"],
                        "pareto_size_mean": item["pareto_size"]["mean"],
                        "feasible_rate_mean": item["feasible_rate"]["mean"],
                        "effective_lift_count_total": telemetry[
                            "effective_lift_accept_count_total"
                        ],
                        "parse_fail_total": telemetry["fallback_reasons"].get(
                            "parse_fail", 0
                        ),
                    }
                )


def write_comparison_csv(path: Path, groups: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "group",
        "comparison",
        "lhs",
        "rhs",
        "n",
        "mean_delta",
        "sample_std_delta",
        "ci95_low",
        "ci95_high",
        "wins",
        "ties",
        "losses",
        "exact_sign_flip_p",
        "holm_p_within_group",
        "same_initialization_count",
        "same_initialization_total",
        "control_config_mismatch_keys",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            for item in group["comparisons"]:
                delta = item["delta"]
                writer.writerow(
                    {
                        "group": group["key"],
                        "comparison": item["key"],
                        "lhs": item["lhs"],
                        "rhs": item["rhs"],
                        "n": delta["n"],
                        "mean_delta": delta["mean"],
                        "sample_std_delta": delta["sample_std"],
                        "ci95_low": delta["ci95_low"],
                        "ci95_high": delta["ci95_high"],
                        "wins": item["wins"],
                        "ties": item["ties"],
                        "losses": item["losses"],
                        "exact_sign_flip_p": item["exact_sign_flip_p"],
                        "holm_p_within_group": item["holm_p_within_group"],
                        "same_initialization_count": item["same_initialization_count"],
                        "same_initialization_total": item["same_initialization_total"],
                        "control_config_mismatch_keys": ";".join(
                            item["control_config_mismatches"].keys()
                        ),
                    }
                )


def plot_groups(path_png: Path, path_pdf: Path, groups: Sequence[Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    for panel, (ax, group, spec) in enumerate(zip(axes.ravel(), groups, GROUPS)):
        variants = list(spec.variants)
        positions = np.arange(len(variants), dtype=float)
        common_seeds = sorted(
            set.intersection(
                *(set(group["observations"][variant.key]) for variant in variants)
            )
        )
        for seed in common_seeds:
            values = [
                float(group["observations"][variant.key][seed]["canonical_hv"])
                for variant in variants
            ]
            ax.plot(positions, values, color="#B7BDC3", linewidth=0.8, alpha=0.6, zorder=1)
        for position, variant in zip(positions, variants):
            values = np.asarray(
                [
                    group["observations"][variant.key][seed]["canonical_hv"]
                    for seed in common_seeds
                ],
                dtype=float,
            )
            offsets = np.linspace(-0.055, 0.055, max(1, len(values)))
            ax.scatter(
                position + offsets,
                values,
                s=30,
                color=variant.color,
                edgecolor="#222222",
                linewidth=0.35,
                alpha=0.86,
                zorder=3,
            )
            summary = group["variant_summaries"][variant.key]["canonical_hv"]
            mean = float(summary["mean"])
            low = float(summary["ci95_low"])
            high = float(summary["ci95_high"])
            ax.errorbar(
                position,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="D",
                markersize=6.2,
                color="#111111",
                markerfacecolor=variant.color,
                capsize=3.5,
                linewidth=1.2,
                zorder=4,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([variant.label for variant in variants], rotation=16, ha="right")
        ax.set_ylabel("Canonical sHV")
        ax.set_title(
            "({}) {}\n{} evaluations/run, {} paired seeds".format(
                chr(ord("a") + panel),
                group["title"],
                group["expected_evaluations"],
                len(common_seeds),
            ),
            loc="left",
            fontsize=11,
        )
        ax.grid(True, axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Archived ablation suite: paired runs, mean and 95% t interval",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(path_png, dpi=260, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    groups = report["groups"]
    lines = [
        "# 消融实验统一复核报告",
        "",
        "本报告重新读取逐 seed `summary.json`，统一使用 canonical sHV 和 sample SD。",
        "分析过程未调用优化器或 LLM API；实验数值来自已完成的真实归档运行。",
        "",
        "## 核心结论",
        "",
    ]
    for finding in report["findings"]:
        lines.append("- {}".format(finding))
    lines.extend(["", "## 分组结果", ""])
    for group in groups:
        lines.extend(
            [
                "### {}".format(group["title"]),
                "",
                "- 角色：{}".format(group["role"]),
                "- 来源：`{}`".format(group["source_report"]),
                "- 协议：{} evaluations/run；seeds={}".format(
                    group["expected_evaluations"], group["seeds"]
                ),
                "- 完整性：{}".format(
                    "通过" if group["integrity"]["passed"] else "未通过"
                ),
                "",
                "| Variant | n | canonical sHV (mean ± sample SD) | Pareto size | Feasible |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for key, item in group["variant_summaries"].items():
            hv = item["canonical_hv"]
            lines.append(
                "| {} | {} | {} ± {} | {} | {:.1%} |".format(
                    item["label"],
                    hv["n"],
                    _fmt(hv["mean"]),
                    _fmt(hv["sample_std"]),
                    _fmt(item["pareto_size"]["mean"], 2),
                    float(item["feasible_rate"]["mean"]),
                )
            )
        lines.extend(
            [
                "",
                "| Paired contrast (lhs-rhs) | Mean Δ | 95% CI | W/T/L | Exact p | Holm p | Config confound |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in group["comparisons"]:
            delta = item["delta"]
            mismatch = ", ".join(item["control_config_mismatches"].keys()) or "none"
            lines.append(
                "| {} | {} | [{}, {}] | {}/{}/{} | {} | {} | {} |".format(
                    item["key"],
                    _fmt(delta["mean"]),
                    _fmt(delta["ci95_low"]),
                    _fmt(delta["ci95_high"]),
                    item["wins"],
                    item["ties"],
                    item["losses"],
                    _fmt(item["exact_sign_flip_p"], 4),
                    _fmt(item["holm_p_within_group"], 4),
                    mismatch,
                )
            )
        if group["integrity"]["errors"] or group["integrity"]["warnings"]:
            lines.extend(["", "完整性备注："])
            for message in group["integrity"]["errors"]:
                lines.append("- ERROR: {}".format(message))
            for message in group["integrity"]["warnings"]:
                lines.append("- WARNING: {}".format(message))
        lines.append("")
    lines.extend(
        [
            "## 结论边界",
            "",
            "1. 五 seed 组的 exact paired randomisation test 最小双侧 p 值为 0.0625，",
            "   因此结果应作描述性配置比较，不作显著性或等效性声明。",
            "2. 历史 Region arms 与对照的 external EI restart 预算不一致，nominal factorial contrast",
            "   不能解释为 posterior lift 的独立因果效应。",
            "3. Prompt 组是 6+20=26 evaluations 的短预算批协议；只能在该设置内解释。",
            "4. 所有结果均为 Chen2020 仿真和未校准退化代理，不代表物理或实验室验证。",
            "",
            "## 产物",
            "",
            "- `report.json`：完整逐 seed 统计、配置审计与 telemetry",
            "- `variant_summary.csv`：各变体汇总",
            "- `paired_comparisons.csv`：配对差值、CI 与 exact/Holm p",
            "- `ablation_suite.png` / `ablation_suite.pdf`：四组配对可视化",
        ]
    )
    # A BOM keeps Chinese text readable in Windows PowerShell 5.1/Notepad while
    # remaining valid UTF-8 for modern editors and Markdown renderers.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def build_report(output_dir: Path) -> Dict[str, Any]:
    analysed = [analyse_group(spec, load_group(spec)) for spec in GROUPS]
    by_key = {group["key"]: group for group in analysed}
    report: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "metric": "canonical_hv",
            "standard_deviation": "sample (ddof=1)",
            "confidence_interval": "two-sided 95% Student-t interval",
            "paired_test": "two-sided exact sign-flip randomisation test on mean delta",
            "multiplicity": "Holm correction within each experiment group",
            "analysis_only": True,
            "optimizer_or_llm_calls": 0,
        },
        "findings": _render_findings(by_key),
        "groups": analysed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_variant_csv(output_dir / "variant_summary.csv", analysed)
    write_comparison_csv(output_dir / "paired_comparisons.csv", analysed)
    plot_groups(
        output_dir / "ablation_suite.png",
        output_dir / "ablation_suite.pdf",
        analysed,
    )
    write_markdown(output_dir / "report.md", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and aggregate archived ablation experiment groups."
    )
    default_output = (
        PROJECT_ROOT
        / "Ablation_Exp"
        / "analysis"
        / "ablation_suite_{}".format(date.today().isoformat().replace("-", "_"))
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    report = build_report(output_dir)
    integrity_ok = all(group["integrity"]["passed"] for group in report["groups"])
    print(
        json.dumps(
            {
                "event": "ablation_analysis_complete",
                "output_dir": str(output_dir),
                "groups": len(report["groups"]),
                "integrity_ok": integrity_ok,
                "report": str(output_dir / "report.md"),
                "figure": str(output_dir / "ablation_suite.png"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if integrity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
