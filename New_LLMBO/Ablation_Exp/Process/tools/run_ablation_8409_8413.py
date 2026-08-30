from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer
from utils.model_labels import canonical_model_label


DEFAULT_SEEDS = [8409, 8410, 8411, 8412, 8413]
DEFAULT_ITERATIONS = 50
DEFAULT_MODEL = "deepseek-v3-thinking"
DEFAULT_API_BASE = "https://api.chat.csu.edu.cn/v1"


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    preset: str
    needs_llm: bool
    color: str
    overrides: Mapping[str, Any] = field(default_factory=dict)
    warmstart_cache_mode: Optional[str] = None
    warmstart_cache_use_selected: bool = False


VARIANTS: List[VariantSpec] = [
    VariantSpec(
        key="baseline",
        label="Baseline",
        preset="strict_baseline",
        needs_llm=False,
        color="#3B4A54",
        overrides={"n_warmstart": 0, "n_random_init": 6},
    ),
    VariantSpec(
        key="baseline_warmstart",
        label="Baseline+WarmStart",
        preset="warmstart_plain_ei",
        needs_llm=True,
        color="#2E8BC8",
        overrides={"n_warmstart": 3, "n_random_init": 3},
        warmstart_cache_mode="write",
        warmstart_cache_use_selected=False,
    ),
    VariantSpec(
        key="baseline_llm_region",
        label="Baseline+LLM_Region",
        preset="warmstart_region_lifted_gp_force_pool_tuned",
        needs_llm=True,
        color="#F0A33A",
        overrides={"n_warmstart": 0, "n_random_init": 6, "region_lift_active_until": 16},
    ),
    VariantSpec(
        key="llmbo_mo",
        label="LLMBO-MO",
        preset="warmstart_region_lifted_gp_force_pool_tuned",
        needs_llm=True,
        color="#D45162",
        overrides={
            "n_warmstart": 3,
            "n_random_init": 3,
            "region_lift_active_until": 24,
            "region_lift_apply_override": True,
            "region_lift_override_uses_diagnostic_pool": True,
            "region_lift_external_influence_mode": "diagnostic_only",
        },
        warmstart_cache_mode="read",
        warmstart_cache_use_selected=True,
    ),
]


def _default_output_root() -> Path:
    date_tag = date.today().isoformat().replace("-", "_")
    return (
        PROJECT_ROOT
        / "Ablation_Exp"
        / "experiment_records"
        / f"ablation_4way_5seeds_{DEFAULT_ITERATIONS}iter_deepseek_v3_thinking_{date_tag}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run four-way ablation experiments for seeds 8409-8413."
    )
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_API_BASE,
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failures in the report and continue with remaining runs.",
    )
    return parser.parse_args()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _json_load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.fmean(items)) if items else 0.0


def _std(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.stdev(items)) if len(items) > 1 else 0.0


def _median(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.median(items)) if items else 0.0


def _variant_by_key(key: str) -> VariantSpec:
    for spec in VARIANTS:
        if spec.key == key:
            return spec
    raise KeyError(key)


def _run_dir(output_root: Path, seed: int, spec: VariantSpec) -> Path:
    return output_root / f"seed{int(seed)}" / spec.key


def _shared_random_cache(output_root: Path, seed: int, n_random_init: int) -> Path:
    return output_root / f"seed{int(seed)}" / f"shared_random_init_{int(n_random_init)}.json"


def _shared_warmstart_cache(output_root: Path, seed: int) -> Path:
    return output_root / f"seed{int(seed)}" / "shared_warmstart_cache.json"


def _build_config(
    *,
    output_root: Path,
    output_dir: Path,
    spec: VariantSpec,
    seed: int,
    iterations: int,
    api_key: str,
    api_base: str,
    model: str,
) -> Dict[str, Any]:
    n_random_init = int(spec.overrides.get("n_random_init", 6))
    cfg: Dict[str, Any] = {
        "experiment_preset": spec.preset,
        "max_iterations": int(iterations),
        "n_candidates": 15,
        "n_select": 1,
        "w_sample_seed": int(seed),
        "init_seed": 2026 + int(seed),
        "random_init_cache_path": str(_shared_random_cache(output_root, seed, n_random_init)),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "checkpoint_every": 99,
    }
    cfg.update(dict(spec.overrides))

    if spec.needs_llm:
        cfg.update(
            {
                "llm_backend": "openai",
                "llm_model": model,
                "llm_api_base": api_base,
                "llm_api_key": api_key,
                "llm_n_samples": 1,
                "llm_temperature": 0.0,
                "warmstart_temperature": 0.0,
                "warmstart_batch_size": 3,
                "warmstart_pool_size": 6,
                "warmstart_max_attempts": 1,
                "warmstart_max_retries": 1,
                "region_preference_max_tokens": 4096,
            }
        )
        if int(cfg.get("n_warmstart", 0)) > 0:
            cfg.update(
                {
                    "warmstart_cache_path": str(_shared_warmstart_cache(output_root, seed)),
                    "warmstart_cache_mode": spec.warmstart_cache_mode or "read_write",
                    "warmstart_cache_use_selected": bool(spec.warmstart_cache_use_selected),
                }
            )
    else:
        cfg.update({"llm_backend": "mock", "llm_api_key": ""})
    return cfg


def _run_single(output_dir: Path, cfg: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = BayesOptimizer(config=cfg)
    optimizer.run()
    optimizer.save_results(str(output_dir))


def _record_failure(output_dir: Path, *, seed: int, spec: VariantSpec, exc: BaseException) -> None:
    payload = {
        "seed": int(seed),
        "variant": spec.key,
        "label": spec.label,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "failed_at": datetime.now().isoformat(),
    }
    _json_dump(output_dir / "failure.json", payload)


def _count_hv_violations(summary: Mapping[str, Any]) -> int:
    trace = summary.get("hv_trace") or []
    values = [
        float(item.get("canonical_hv", item.get("hypervolume_canonical", item.get("hypervolume", 0.0))))
        for item in trace
        if isinstance(item, Mapping)
    ]
    return int(sum(1 for prev, curr in zip(values, values[1:]) if curr + 1e-12 < prev))


def _record_for_run(seed: int, spec: VariantSpec, run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    failure_path = run_dir / "failure.json"
    base = {
        "seed": int(seed),
        "variant": spec.key,
        "label": spec.label,
        "preset": spec.preset,
        "summary_path": str(summary_path),
        "run_dir": str(run_dir),
    }
    if summary_path.exists():
        summary = _json_load(summary_path)
        fallback_distribution = dict(summary.get("region_lift_fallback_reasons") or {})
        return {
            **base,
            "status": "ok",
            "display_hv": float(summary.get("display_hv", summary.get("hypervolume", 0.0))),
            "canonical_hv": float(summary.get("canonical_hv", summary.get("hypervolume_canonical", 0.0))),
            "hypervolume_raw": float(summary.get("hypervolume_raw", 0.0)),
            "pareto_size": int(summary.get("pareto_size", 0)),
            "n_total": int(summary.get("n_total", 0)),
            "n_feasible": int(summary.get("n_feasible", 0)),
            "hv_violations": _count_hv_violations(summary),
            "region_lift_accept_rate": float(summary.get("region_lift_accept_rate", 0.0)),
            "effective_lift_accept_rate": float(summary.get("effective_lift_accept_rate", 0.0)),
            "effective_lift_accept_count": int(summary.get("effective_lift_accept_count", 0)),
            "region_pool_influenced_acquisition_count": int(
                summary.get("region_pool_influenced_acquisition_count", 0)
            ),
            "plain_candidate_inside_region_count": int(summary.get("plain_candidate_inside_region_count", 0)),
            "diagnostic_override_candidate_count": int(summary.get("diagnostic_override_candidate_count", 0)),
            "fallback_distribution": fallback_distribution,
        }
    if failure_path.exists():
        failure = _json_load(failure_path)
        return {**base, **failure, "status": "failed"}
    return {**base, "status": "missing"}


def _aggregate(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ok = [record for record in records if record.get("status") == "ok"]
    canonical = [float(record["canonical_hv"]) for record in ok]
    display = [float(record["display_hv"]) for record in ok]
    raw = [float(record["hypervolume_raw"]) for record in ok]
    pareto = [float(record["pareto_size"]) for record in ok]
    feasible = [float(record["n_feasible"]) for record in ok]
    fallback_counter: Counter[str] = Counter()
    for record in ok:
        fallback_counter.update(record.get("fallback_distribution") or {})
    return {
        "n_runs": int(len(ok)),
        "n_missing_or_failed": int(len(records) - len(ok)),
        "canonical_hv": {
            "mean": _mean(canonical),
            "std": _std(canonical),
            "median": _median(canonical),
            "min": min(canonical) if canonical else 0.0,
            "max": max(canonical) if canonical else 0.0,
            "values": canonical,
        },
        "display_hv": {"mean": _mean(display), "std": _std(display), "values": display},
        "hypervolume_raw": {"mean": _mean(raw), "std": _std(raw), "values": raw},
        "pareto_size": {"mean": _mean(pareto), "std": _std(pareto), "values": pareto},
        "n_feasible": {"mean": _mean(feasible), "std": _std(feasible), "values": feasible},
        "hv_violations_total": int(sum(int(record.get("hv_violations", 0)) for record in ok)),
        "region_lift_accept_rate_mean": _mean(float(record.get("region_lift_accept_rate", 0.0)) for record in ok),
        "effective_lift_accept_rate_mean": _mean(
            float(record.get("effective_lift_accept_rate", 0.0)) for record in ok
        ),
        "effective_lift_accept_count_total": int(
            sum(int(record.get("effective_lift_accept_count", 0)) for record in ok)
        ),
        "region_pool_influenced_acquisition_count_total": int(
            sum(int(record.get("region_pool_influenced_acquisition_count", 0)) for record in ok)
        ),
        "plain_candidate_inside_region_count_total": int(
            sum(int(record.get("plain_candidate_inside_region_count", 0)) for record in ok)
        ),
        "diagnostic_override_candidate_count_total": int(
            sum(int(record.get("diagnostic_override_candidate_count", 0)) for record in ok)
        ),
        "fallback_distribution": dict(fallback_counter),
        "runs": list(ok),
        "missing_or_failed_runs": [dict(record) for record in records if record.get("status") != "ok"],
    }


def _comparison(lhs_records: Sequence[Mapping[str, Any]], rhs_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    lhs_ok = {int(record["seed"]): record for record in lhs_records if record.get("status") == "ok"}
    rhs_ok = {int(record["seed"]): record for record in rhs_records if record.get("status") == "ok"}
    shared = sorted(set(lhs_ok) & set(rhs_ok))
    deltas = []
    pct_deltas = []
    wins = 0
    for seed in shared:
        lhs_hv = float(lhs_ok[seed]["canonical_hv"])
        rhs_hv = float(rhs_ok[seed]["canonical_hv"])
        delta = lhs_hv - rhs_hv
        deltas.append(delta)
        if abs(rhs_hv) > 1e-12:
            pct_deltas.append(delta / rhs_hv * 100.0)
        if delta > 0.0:
            wins += 1
    lhs_label = str(lhs_records[0]["label"]) if lhs_records else None
    rhs_label = str(rhs_records[0]["label"]) if rhs_records else None
    return {
        "lhs": lhs_label,
        "rhs": rhs_label,
        "shared_seeds": shared,
        "mean_canonical_hv_delta": _mean(deltas),
        "std_canonical_hv_delta": _std(deltas),
        "mean_canonical_hv_pct_delta": _mean(pct_deltas),
        "wins": int(wins),
        "total": int(len(shared)),
        "per_seed_delta": [
            {
                "seed": int(seed),
                "delta": float(
                    float(lhs_ok[seed]["canonical_hv"]) - float(rhs_ok[seed]["canonical_hv"])
                ),
            }
            for seed in shared
        ],
    }


def _write_manifest(output_root: Path, args: argparse.Namespace) -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "experiment": "ablation_4way",
        "iterations": int(args.iterations),
        "seeds": [int(seed) for seed in args.seeds],
        "api_base": str(args.api_base),
        "model": str(args.model),
        "model_display": canonical_model_label(str(args.model)),
        "api_key_storage": "not written to disk; read from LLM_API_KEY/OPENAI_API_KEY at runtime",
        "variants": [
            {
                "key": spec.key,
                "label": spec.label,
                "preset": spec.preset,
                "needs_llm": bool(spec.needs_llm),
                "overrides": dict(spec.overrides),
            }
            for spec in VARIANTS
        ],
    }
    _json_dump(output_root / "manifest.json", payload)


def _write_report(output_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(seed) for seed in args.seeds]
    records: List[Dict[str, Any]] = []
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for spec in VARIANTS:
        variant_records = []
        for seed in seeds:
            record = _record_for_run(seed, spec, _run_dir(output_root, seed, spec))
            records.append(record)
            variant_records.append(record)
        by_variant[spec.key] = variant_records

    comparisons: Dict[str, Any] = {}
    baseline_records = by_variant["baseline"]
    for spec in VARIANTS:
        if spec.key == "baseline":
            continue
        comparisons[f"{spec.key}_vs_baseline"] = _comparison(by_variant[spec.key], baseline_records)
    comparisons["llmbo_mo_vs_baseline_warmstart"] = _comparison(
        by_variant["llmbo_mo"], by_variant["baseline_warmstart"]
    )
    comparisons["llmbo_mo_vs_baseline_llm_region"] = _comparison(
        by_variant["llmbo_mo"], by_variant["baseline_llm_region"]
    )
    comparisons["baseline_llm_region_vs_baseline_warmstart"] = _comparison(
        by_variant["baseline_llm_region"], by_variant["baseline_warmstart"]
    )

    report = {
        "meta": {
            "experiment": "ablation_4way",
            "iterations": int(args.iterations),
            "seeds": seeds,
            "output_root": str(output_root),
            "api_base": str(args.api_base),
            "model": str(args.model),
            "model_display": canonical_model_label(str(args.model)),
            "hv_metric": "canonical_hv",
            "created_at": datetime.now().isoformat(),
        },
        "variants": [
            {
                "key": spec.key,
                "label": spec.label,
                "preset": spec.preset,
                "needs_llm": bool(spec.needs_llm),
                "overrides": dict(spec.overrides),
            }
            for spec in VARIANTS
        ],
        "records": records,
        "aggregates": {key: _aggregate(items) for key, items in by_variant.items()},
        "comparisons": comparisons,
    }
    _json_dump(output_root / "report_5seeds.json", report)
    return report


def _summary_paths(output_root: Path, spec: VariantSpec, seeds: Sequence[int]) -> List[Path]:
    return [_run_dir(output_root, int(seed), spec) / "summary.json" for seed in seeds]


def _extract_trace(summary: Mapping[str, Any]) -> np.ndarray:
    trace = summary.get("hv_trace") or []
    values = []
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        value = item.get("canonical_hv", item.get("hypervolume_canonical", None))
        if value is None:
            raw = item.get("hypervolume_raw")
            value = item.get("hypervolume") if raw is None else None
        if value is not None:
            values.append(float(value))
    if values:
        return np.asarray(values, dtype=float)
    final_value = summary.get("canonical_hv", summary.get("hypervolume_canonical", 0.0))
    return np.asarray([float(final_value)], dtype=float)


def _plot_hv_box(report: Mapping[str, Any], image_dir: Path) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [spec.label for spec in VARIANTS]
    colors = [spec.color for spec in VARIANTS]
    data = [
        np.asarray(
            ((report.get("aggregates") or {}).get(spec.key) or {}).get("canonical_hv", {}).get("values", []),
            dtype=float,
        )
        for spec in VARIANTS
    ]
    image_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(1, len(data) + 1)
    box = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor("#222222")
    rng = np.random.default_rng(8409)
    for pos, values, color in zip(positions, data, colors):
        if values.size == 0:
            continue
        offsets = rng.normal(0.0, 0.035, size=values.size)
        ax.scatter(pos + offsets, values, s=42, color=color, edgecolor="#222222", linewidth=0.35, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Canonical HV")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    png = image_dir / "ablation_canonical_hv_box.png"
    pdf = image_dir / "ablation_canonical_hv_box.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _plot_hv_convergence(output_root: Path, seeds: Sequence[int], image_dir: Path) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for spec in VARIANTS:
        traces = []
        for path in _summary_paths(output_root, spec, seeds):
            if not path.exists():
                continue
            traces.append(_extract_trace(_json_load(path)))
        if not traces:
            continue
        min_len = min(len(trace) for trace in traces)
        stack = np.vstack([trace[:min_len] for trace in traces])
        x = np.arange(1, min_len + 1)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        ax.plot(x, mean, color=spec.color, linewidth=2.0, label=spec.label)
        ax.fill_between(x, mean - std, mean + std, color=spec.color, alpha=0.14, linewidth=0)
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Canonical HV")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    png = image_dir / "ablation_hv_convergence.png"
    pdf = image_dir / "ablation_hv_convergence.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _extract_pareto_objectives(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: List[List[float]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping) and "objectives" in item:
                rows.append([float(v) for v in item["objectives"][:3]])
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)) and len(item) >= 3:
                rows.append([float(v) for v in item[:3]])
    elif isinstance(payload, Mapping):
        for item in payload.get("pareto_front", []):
            if isinstance(item, Mapping) and "objectives" in item:
                rows.append([float(v) for v in item["objectives"][:3]])
    return np.asarray(rows, dtype=float) if rows else np.empty((0, 3), dtype=float)


def _plot_pareto_3d(output_root: Path, seeds: Sequence[int], image_dir: Path) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    for spec in VARIANTS:
        rows = []
        for seed in seeds:
            path = _run_dir(output_root, int(seed), spec) / "pareto_front.json"
            if path.exists():
                points = _extract_pareto_objectives(path)
                if points.size:
                    rows.append(points)
        if not rows:
            continue
        points = np.vstack(rows)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=18,
            alpha=0.68,
            color=spec.color,
            label=spec.label,
            depthshade=False,
        )
    ax.set_xlabel("Charging time / s", labelpad=8)
    ax.set_ylabel("Temp rise / K", labelpad=8)
    ax.set_zlabel("Aging / %", labelpad=8)
    ax.view_init(elev=20, azim=-126)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    png = image_dir / "ablation_pareto_3d_all_seeds.png"
    pdf = image_dir / "ablation_pareto_3d_all_seeds.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _write_plot_config(report: Mapping[str, Any], output_root: Path, image_dir: Path, artifacts: Mapping[str, Any]) -> None:
    payload = {
        "report_path": str(output_root / "report_5seeds.json"),
        "image_dir": str(image_dir),
        "artifacts": artifacts,
        "metric": "canonical_hv",
        "groups": [
            {
                "key": spec.key,
                "label": spec.label,
                "color": spec.color,
                "values": ((report.get("aggregates") or {}).get(spec.key) or {})
                .get("canonical_hv", {})
                .get("values", []),
            }
            for spec in VARIANTS
        ],
    }
    _json_dump(output_root / "plot_manifest.json", payload)


def generate_plots(output_root: Path, args: argparse.Namespace, report: Mapping[str, Any]) -> Dict[str, Any]:
    image_dir = output_root / "images"
    artifacts = {
        "hv_box": _plot_hv_box(report, image_dir),
        "hv_convergence": _plot_hv_convergence(output_root, args.seeds, image_dir),
        "pareto_3d": _plot_pareto_3d(output_root, args.seeds, image_dir),
    }
    _write_plot_config(report, output_root, image_dir, artifacts)
    return artifacts


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not args.summarize_only and not api_key:
        raise RuntimeError("Set LLM_API_KEY or OPENAI_API_KEY before running real LLM ablation experiments.")

    _write_manifest(output_root, args)
    seeds = [int(seed) for seed in args.seeds]

    if not args.summarize_only:
        for seed in seeds:
            for spec in VARIANTS:
                run_dir = _run_dir(output_root, seed, spec)
                summary_path = run_dir / "summary.json"
                if args.skip_existing and summary_path.exists():
                    print(f"[skip] seed={seed} variant={spec.key} -> {summary_path}", flush=True)
                    continue
                print(f"[run] seed={seed} variant={spec.key} ({spec.label}) -> {run_dir}", flush=True)
                cfg = _build_config(
                    output_root=output_root,
                    output_dir=run_dir,
                    spec=spec,
                    seed=seed,
                    iterations=int(args.iterations),
                    api_key=api_key,
                    api_base=str(args.api_base),
                    model=str(args.model),
                )
                try:
                    _run_single(run_dir, cfg)
                except Exception as exc:
                    _record_failure(run_dir, seed=seed, spec=spec, exc=exc)
                    print(f"[failed] seed={seed} variant={spec.key}: {type(exc).__name__}: {exc}", flush=True)
                    if not args.continue_on_error:
                        raise

    report = _write_report(output_root, args)
    artifacts = generate_plots(output_root, args, report)
    print(
        json.dumps(
            {
                "event": "done",
                "output_root": str(output_root),
                "report": str(output_root / "report_5seeds.json"),
                "plots": artifacts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
