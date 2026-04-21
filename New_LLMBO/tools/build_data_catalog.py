from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis_runs" / "unified_data_catalog"
SCAN_ROOTS = [
    "optimized_experiments",
    "api_experiments",
    "fixed_experiments",
    "analysis_runs",
    "results",
]


@dataclass
class PathInfo:
    collection: str
    experiment_family: str
    variant_group: str
    run_name: str
    relative_dir: str


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_paths(names: Iterable[str]) -> Iterable[Path]:
    for root_name in names:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        yield root


def _path_info(path: Path) -> PathInfo:
    rel = path.relative_to(PROJECT_ROOT)
    parts = rel.parts[:-1]
    collection = parts[0] if parts else ""
    subdirs = list(parts[1:])

    if not subdirs:
        return PathInfo(
            collection=collection,
            experiment_family=collection,
            variant_group="",
            run_name=collection,
            relative_dir=str(rel.parent).replace("\\", "/"),
        )

    experiment_family = subdirs[0]
    run_name = subdirs[-1]
    variant_group = "/".join(subdirs[1:-1]) if len(subdirs) > 2 else ""
    if len(subdirs) == 2:
        variant_group = ""

    return PathInfo(
        collection=collection,
        experiment_family=experiment_family,
        variant_group=variant_group,
        run_name=run_name,
        relative_dir=str(rel.parent).replace("\\", "/"),
    )


def _count_hv_violations(summary: Dict[str, Any]) -> int:
    hv_trace = summary.get("hv_trace") or []
    hv_values = []
    for item in hv_trace:
        value = _safe_float(item.get("hypervolume"))
        if value is not None:
            hv_values.append(value)
    violations = 0
    for prev, curr in zip(hv_values, hv_values[1:]):
        if curr + 1e-12 < prev:
            violations += 1
    return violations


def _source_counts(summary: Dict[str, Any]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for item in summary.get("hv_trace") or []:
        source = str(item.get("source") or "")
        if source:
            counts[source] += 1
    return dict(sorted(counts.items()))


def _warmstart_last(trace: Any, key: str) -> Optional[float]:
    if not isinstance(trace, list) or not trace:
        return None
    return _safe_float(trace[-1].get(key))


def build_run_manifest() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in _iter_paths(SCAN_ROOTS):
        for summary_path in root.rglob("summary.json"):
            info = _path_info(summary_path)
            summary = _read_json(summary_path)
            config = summary.get("config") or {}
            run_dir = summary_path.parent

            row = {
                "collection": info.collection,
                "experiment_family": info.experiment_family,
                "variant_group": info.variant_group,
                "run_name": info.run_name,
                "relative_dir": info.relative_dir,
                "summary_path": str(summary_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "database_path": str((run_dir / "database.json").relative_to(PROJECT_ROOT)).replace("\\", "/")
                if (run_dir / "database.json").exists() else "",
                "db_final_path": str((run_dir / "db_final.json").relative_to(PROJECT_ROOT)).replace("\\", "/")
                if (run_dir / "db_final.json").exists() else "",
                "pareto_front_path": str((run_dir / "pareto_front.json").relative_to(PROJECT_ROOT)).replace("\\", "/")
                if (run_dir / "pareto_front.json").exists() else "",
                "hypervolume": _safe_float(summary.get("hypervolume")),
                "hypervolume_raw": _safe_float(summary.get("hypervolume_raw")),
                "n_total": _safe_int(summary.get("n_total")),
                "n_feasible": _safe_int(summary.get("n_feasible")),
                "pareto_size": _safe_int(summary.get("pareto_size")),
                "init_hv": _warmstart_last(summary.get("warmstart_trace"), "hypervolume"),
                "init_hv_raw": _warmstart_last(summary.get("warmstart_trace"), "hypervolume_raw"),
                "warmstart_trace_len": len(summary.get("warmstart_trace") or []),
                "hv_trace_len": len(summary.get("hv_trace") or []),
                "hv_violations": _count_hv_violations(summary),
                "w_sample_seed": _safe_int(config.get("w_sample_seed")),
                "init_seed": _safe_int(config.get("init_seed")),
                "max_iterations": _safe_int(config.get("max_iterations")),
                "n_warmstart": _safe_int(config.get("n_warmstart")),
                "n_random_init": _safe_int(config.get("n_random_init")),
                "n_candidates": _safe_int(config.get("n_candidates")),
                "n_select": _safe_int(config.get("n_select")),
                "llm_backend": config.get("llm_backend"),
                "llm_model": config.get("llm_model"),
                "llm_safe_dsoc_sum_max": _safe_float(config.get("llm_safe_dsoc_sum_max")),
                "enable_iterative_guidance": bool(config.get("enable_iterative_guidance", False)),
                "enable_gp_llm_coupling": bool(config.get("enable_gp_llm_coupling", False)),
                "enable_acq_prior_coupling": bool(config.get("enable_acq_prior_coupling", False)),
                "enable_llm_rerank": bool(config.get("enable_llm_rerank", False)),
                "enable_proposal_sampler": bool(config.get("enable_proposal_sampler", False)),
                "llm_rerank_gate_mode": config.get("llm_rerank_gate_mode"),
                "llm_rerank_const_gate": _safe_float(config.get("llm_rerank_const_gate")),
                "last_guidance_mode": (summary.get("last_guidance") or {}).get("mode"),
                "last_guidance_confidence": _safe_float((summary.get("last_guidance") or {}).get("confidence")),
                "last_rerank_active": bool((summary.get("last_llm_rerank_summary") or {}).get("active", False)),
                "last_rerank_applied": bool((summary.get("last_llm_rerank_summary") or {}).get("applied", False)),
                "rerank_telemetry_count": len(summary.get("rerank_telemetry") or []),
                "eval_source_counts": _json_text(_source_counts(summary)),
                "last_candidate_source_counts": _json_text(summary.get("last_candidate_source_counts") or {}),
            }
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["collection"],
            row["experiment_family"],
            row["variant_group"],
            row["run_name"],
        )
    )
    return rows


def _report_group_row(
    report_path: Path,
    info: PathInfo,
    group_name: str,
    payload: Dict[str, Any],
    *,
    structured: bool,
) -> Dict[str, Any]:
    aggregate = payload.get("aggregate") if structured else payload
    if not isinstance(aggregate, dict):
        aggregate = {}
    runs = payload.get("runs") if structured else payload.get("runs")
    if not isinstance(runs, list):
        runs = []

    return {
        "collection": info.collection,
        "experiment_family": info.experiment_family,
        "variant_group": info.variant_group,
        "report_dir": info.relative_dir,
        "report_path": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "group_name": group_name,
        "structured_report": structured,
        "run_count": len(runs),
        "mean_hv": _safe_float(aggregate.get("mean_hv", aggregate.get("mean_final_hv"))),
        "std_hv": _safe_float(aggregate.get("std_hv", aggregate.get("std_final_hv"))),
        "mean_hv_raw": _safe_float(aggregate.get("mean_hv_raw", aggregate.get("mean_final_hv_raw"))),
        "std_hv_raw": _safe_float(aggregate.get("std_hv_raw", aggregate.get("std_final_hv_raw"))),
        "mean_init_hv": _safe_float(aggregate.get("mean_init_hv")),
        "std_init_hv": _safe_float(aggregate.get("std_init_hv")),
        "mean_pareto_size": _safe_float(aggregate.get("mean_pareto_size")),
        "std_pareto_size": _safe_float(aggregate.get("std_pareto_size")),
        "mean_hv_violations": _safe_float(aggregate.get("mean_hv_violations")),
        "rerank_applied_count": _safe_int(aggregate.get("rerank_applied_count")),
        "rerank_active_count": _safe_int(aggregate.get("rerank_active_count")),
        "group_payload": _json_text(payload),
    }


def _report_comparison_row(
    report_path: Path,
    info: PathInfo,
    name: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "collection": info.collection,
        "experiment_family": info.experiment_family,
        "variant_group": info.variant_group,
        "report_dir": info.relative_dir,
        "report_path": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "comparison_name": name,
        "mean_delta": _safe_float(payload.get("mean_delta", payload.get("abs_diff", payload.get("vs_baseline_mean_delta")))),
        "mean_pct": _safe_float(payload.get("mean_pct", payload.get("rel_diff_pct", payload.get("vs_baseline_mean_pct")))),
        "wins": _safe_int(payload.get("wins", payload.get("wins_vs_plain"))),
        "total": _safe_int(payload.get("total")),
        "comparison_payload": _json_text(payload),
    }


def build_report_manifests() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []

    for root in _iter_paths(SCAN_ROOTS):
        for report_path in root.rglob("report.json"):
            info = _path_info(report_path)
            report = _read_json(report_path)
            if "groups" in report and isinstance(report.get("groups"), dict):
                for group_name, payload in report["groups"].items():
                    if not isinstance(payload, dict):
                        continue
                    group_rows.append(
                        _report_group_row(report_path, info, group_name, payload, structured=True)
                    )
                    for key, value in payload.items():
                        if key.startswith("comparison") and isinstance(value, dict):
                            comparison_rows.append(
                                _report_comparison_row(
                                    report_path,
                                    info,
                                    f"{group_name}.{key}",
                                    value,
                                )
                            )
                continue

            for key, value in report.items():
                if key == "meta":
                    continue
                if not isinstance(value, dict):
                    continue
                if key.startswith("comparison"):
                    comparison_rows.append(_report_comparison_row(report_path, info, key, value))
                else:
                    group_rows.append(
                        _report_group_row(report_path, info, key, value, structured=False)
                    )

    group_rows.sort(
        key=lambda row: (
            row["collection"],
            row["experiment_family"],
            row["group_name"],
        )
    )
    comparison_rows.sort(
        key=lambda row: (
            row["collection"],
            row["experiment_family"],
            row["comparison_name"],
        )
    )
    return group_rows, comparison_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_inventory_summary(
    run_rows: List[Dict[str, Any]],
    group_rows: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_collection: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "run_count": 0,
        "report_group_count": 0,
        "report_comparison_count": 0,
        "experiment_families": set(),
    })

    for row in run_rows:
        entry = by_collection[row["collection"]]
        entry["run_count"] += 1
        entry["experiment_families"].add(row["experiment_family"])

    for row in group_rows:
        entry = by_collection[row["collection"]]
        entry["report_group_count"] += 1
        entry["experiment_families"].add(row["experiment_family"])

    for row in comparison_rows:
        entry = by_collection[row["collection"]]
        entry["report_comparison_count"] += 1
        entry["experiment_families"].add(row["experiment_family"])

    summary = {
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "totals": {
            "run_count": len(run_rows),
            "report_group_count": len(group_rows),
            "report_comparison_count": len(comparison_rows),
        },
        "by_collection": {},
    }

    for collection, entry in sorted(by_collection.items()):
        summary["by_collection"][collection] = {
            "run_count": entry["run_count"],
            "report_group_count": entry["report_group_count"],
            "report_comparison_count": entry["report_comparison_count"],
            "experiment_family_count": len(entry["experiment_families"]),
            "experiment_families": sorted(entry["experiment_families"]),
        }

    return summary


def build_readme(
    inventory: Dict[str, Any],
    run_rows: List[Dict[str, Any]],
    group_rows: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
) -> str:
    latest_runs = sorted(
        run_rows,
        key=lambda row: (
            row["collection"],
            row["experiment_family"],
            row["run_name"],
        ),
    )[-8:]

    lines = [
        "# Unified Data Catalog",
        "",
        "这个目录是对仓库实验结果的统一索引，不改动原始 `summary.json / report.json / database.json` 文件。",
        "",
        "## 文件说明",
        "",
        "- `run_manifest.csv/json`: 每个 `summary.json` 一行，适合按 seed、HV、配置字段做筛选。",
        "- `report_group_manifest.csv/json`: 每个 `report.json` 里的实验组聚合结果一行。",
        "- `report_comparison_manifest.csv/json`: 每个 `report.json` 里的 comparison 项一行。",
        "- `inventory_summary.json`: 全仓库结果文件数量、来源目录和实验族统计。",
        "",
        "## 当前规模",
        "",
        f"- 运行级条目: `{inventory['totals']['run_count']}`",
        f"- 报告组条目: `{inventory['totals']['report_group_count']}`",
        f"- 对比条目: `{inventory['totals']['report_comparison_count']}`",
        "",
        "## 推荐用法",
        "",
        "- 先看 `run_manifest.csv`，按 `experiment_family / run_name / w_sample_seed / hypervolume` 过滤。",
        "- 要看均值结果时看 `report_group_manifest.csv`。",
        "- 要查历史对比结论时看 `report_comparison_manifest.csv`。",
        "- 原始细节仍回跳到对应 `summary_path / database_path / pareto_front_path`。",
        "",
        "## 最近索引到的部分运行",
        "",
    ]

    for row in latest_runs:
        lines.append(
            f"- `{row['collection']}/{row['experiment_family']}/{row['run_name']}`: "
            f"HV={row['hypervolume']}, seed={row['w_sample_seed']}, summary=`{row['summary_path']}`"
        )

    lines.extend(
        [
            "",
            "## 字段约定",
            "",
            "- `experiment_family`: 一级实验族目录，例如 `llmei_vs_plain_v1`。",
            "- `variant_group`: 更深层的分组目录；没有时为空。",
            "- `run_name`: 最终运行目录名。",
            "- `init_hv`: `warmstart_trace` 的最后一个超体积值。",
            "- `hv_violations`: `hv_trace` 中超体积下降次数，正常应为 0。",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    run_rows = build_run_manifest()
    group_rows, comparison_rows = build_report_manifests()
    inventory = build_inventory_summary(run_rows, group_rows, comparison_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(OUTPUT_DIR / "run_manifest.csv", run_rows)
    write_json(OUTPUT_DIR / "run_manifest.json", run_rows)

    write_csv(OUTPUT_DIR / "report_group_manifest.csv", group_rows)
    write_json(OUTPUT_DIR / "report_group_manifest.json", group_rows)

    write_csv(OUTPUT_DIR / "report_comparison_manifest.csv", comparison_rows)
    write_json(OUTPUT_DIR / "report_comparison_manifest.json", comparison_rows)

    write_json(OUTPUT_DIR / "inventory_summary.json", inventory)
    (OUTPUT_DIR / "README.md").write_text(
        build_readme(inventory, run_rows, group_rows, comparison_rows),
        encoding="utf-8",
    )

    print(json.dumps(inventory, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
