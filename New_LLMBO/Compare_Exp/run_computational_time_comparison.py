from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Compare_Exp.Exp.nsga2_runner import NSGA2Runner
from Compare_Exp.Exp.parego_runner import ParEGORunner
from llmbo.optimizer import BayesOptimizer

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [8409, 8410, 8411, 8412, 8413]
DEFAULT_API_BASE = "https://api.chat.csu.edu.cn/v1"
DEFAULT_LLM_MODEL = "deepseek-v3-thinking"
ALGORITHM_ORDER: List[Tuple[str, str]] = [
    ("parego", "ParEGO"),
    ("nsga2", "NSGA-II"),
    ("llmbo_mo", "LLMBO-MO"),
]


def _default_output_root(iterations: int) -> Path:
    date_tag = datetime.now().strftime("%Y_%m_%d")
    return (
        PROJECT_ROOT
        / "Compare_Exp"
        / "experiment_records"
        / f"computational_time_3algo_{len(DEFAULT_SEEDS)}seeds_{iterations}iter_{date_tag}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark computational time for ParEGO, NSGA-II, and LLMBO-MO."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["parego", "nsga2", "llmbo_mo"],
        choices=["parego", "nsga2", "llmbo_mo", "all"],
    )
    parser.add_argument("--param-set", type=str, default="Chen2020", choices=["Chen2020", "Ecker2015", "ORegan2022"])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")

    parser.add_argument("--parego-variant", type=str, default="matlab_reference", choices=["baseline", "matlab_reference"])
    parser.add_argument("--parego-random-init", type=int, default=6)
    parser.add_argument("--nsga2-pop-size", type=int, default=20)
    parser.add_argument("--nsga2-evals", type=int, default=None)

    parser.add_argument("--llmbo-preset", type=str, default="warmstart_region_lifted_gp_force_pool_tuned")
    parser.add_argument("--llm-warmstart", type=int, default=3)
    parser.add_argument("--llm-random-init", type=int, default=3)
    parser.add_argument("--llm-candidates", type=int, default=15)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-samples", type=int, default=1)
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_API_BASE,
    )
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL)

    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.fmean(items)) if items else 0.0


def _std(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.stdev(items)) if len(items) > 1 else 0.0


def _runtime_from_summary(summary: Dict[str, Any]) -> Optional[float]:
    timing = summary.get("timing") or {}
    value = timing.get("runtime_s", summary.get("runtime_s"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mark_timing(
    summary_path: Path,
    summary: Dict[str, Any],
    *,
    runtime_s: float,
    algorithm: str,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    summary["algorithm_key"] = algorithm
    summary["runtime_s"] = float(runtime_s)
    summary["output_dir"] = str(output_dir)
    summary["timing"] = {
        "runtime_s": float(runtime_s),
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "clock": "time.perf_counter",
        "scope": "optimizer.run plus save_results for one seed",
    }
    summary["seed"] = int(seed)
    _write_json(summary_path, summary)
    return summary


def _run_parego(seed: int, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    n_evals = int(args.iterations) + int(args.parego_random_init)
    runner = ParEGORunner(
        seed=int(seed),
        n_evals=n_evals,
        n_random_init=int(args.parego_random_init),
        variant=str(args.parego_variant),
        param_set=str(args.param_set),
    )
    runner.run()
    return runner.save_results(str(output_dir))


def _run_nsga2(seed: int, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    n_evals = int(args.nsga2_evals or (int(args.iterations) + int(args.parego_random_init)))
    runner = NSGA2Runner(
        seed=int(seed),
        n_evals=n_evals,
        pop_size=int(args.nsga2_pop_size),
        param_set=str(args.param_set),
    )
    runner.run()
    return runner.save_results(str(output_dir))


def _llmbo_config(seed: int, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("Set LLM_API_KEY or OPENAI_API_KEY before running LLMBO-MO timing.")

    seed_root = output_dir.parent
    return {
        "experiment_preset": str(args.llmbo_preset),
        "max_iterations": int(args.iterations),
        "n_warmstart": int(args.llm_warmstart),
        "n_random_init": int(args.llm_random_init),
        "n_candidates": int(args.llm_candidates),
        "n_select": 1,
        "w_sample_seed": int(seed),
        "init_seed": int(2026 + seed),
        "objective_preprocess_mode": "minmax",
        "battery_param_set": str(args.param_set),
        "llm_backend": "openai",
        "llm_model": str(args.model),
        "llm_api_base": str(args.api_base),
        "llm_api_key": api_key,
        "llm_n_samples": int(args.llm_samples),
        "llm_temperature": float(args.llm_temperature),
        "warmstart_temperature": float(args.llm_temperature),
        "warmstart_cache_path": str(seed_root / "shared_warmstart_cache.json"),
        "warmstart_cache_mode": "read_write",
        "warmstart_cache_use_selected": True,
        "random_init_cache_path": str(seed_root / "shared_random_init_cache.json"),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "checkpoint_every": 9999,
    }


def _run_llmbo(seed: int, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    cfg = _llmbo_config(seed, args, output_dir)
    optimizer = BayesOptimizer(config=cfg)
    optimizer.run()
    optimizer.save_results(str(output_dir))
    return _read_json(output_dir / "summary.json")


def _run_or_load(seed: int, algorithm: str, args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    output_dir = output_root / f"seed{int(seed)}" / algorithm
    summary_path = output_dir / "summary.json"

    if (args.skip_existing or args.summarize_only) and summary_path.exists():
        summary = _read_json(summary_path)
        if _runtime_from_summary(summary) is not None:
            logger.info("[%s] seed=%s using existing timing", algorithm, seed)
            return summary
        if args.summarize_only:
            raise RuntimeError(f"Existing summary has no runtime_s: {summary_path}")

    if args.summarize_only:
        raise RuntimeError(f"Missing timed summary: {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[%s] seed=%s starting -> %s", algorithm, seed, output_dir)
    t0 = time.perf_counter()
    if algorithm == "parego":
        summary = _run_parego(seed, args, output_dir)
    elif algorithm == "nsga2":
        summary = _run_nsga2(seed, args, output_dir)
    elif algorithm == "llmbo_mo":
        summary = _run_llmbo(seed, args, output_dir)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    runtime_s = time.perf_counter() - t0
    logger.info("[%s] seed=%s finished in %.2fs", algorithm, seed, runtime_s)
    return _mark_timing(
        summary_path,
        summary,
        runtime_s=runtime_s,
        algorithm=algorithm,
        seed=seed,
        output_dir=output_dir,
    )


def _record_from_summary(seed: int, algorithm: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    runtime_s = _runtime_from_summary(summary)
    if runtime_s is None:
        raise RuntimeError(f"No runtime_s in summary for {algorithm} seed={seed}")
    return {
        "seed": int(seed),
        "algorithm": algorithm,
        "runtime_s": float(runtime_s),
        "canonical_hv": float(summary.get("canonical_hv", summary.get("hypervolume_canonical", 0.0)) or 0.0),
        "display_hv": float(summary.get("display_hv", summary.get("hypervolume", 0.0)) or 0.0),
        "pareto_size": int(summary.get("pareto_size", 0) or 0),
        "n_total": int(summary.get("n_total", summary.get("n_evals", 0)) or 0),
        "n_feasible": int(summary.get("n_feasible", 0) or 0),
        "summary_path": str(Path(summary.get("output_dir", "") or ".") / "summary.json"),
    }


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    runtime = [float(r["runtime_s"]) for r in records]
    hv = [float(r["canonical_hv"]) for r in records]
    pareto = [float(r["pareto_size"]) for r in records]
    return {
        "n_runs": len(records),
        "runtime_s": {
            "mean": _mean(runtime),
            "std": _std(runtime),
            "min": min(runtime) if runtime else 0.0,
            "max": max(runtime) if runtime else 0.0,
        },
        "canonical_hv": {"mean": _mean(hv), "std": _std(hv)},
        "pareto_size": {"mean": _mean(pareto), "std": _std(pareto)},
    }


def _build_report(
    args: argparse.Namespace,
    output_root: Path,
    algorithms: List[str],
    records_by_algorithm: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    return {
        "meta": {
            "experiment": "computational_time_comparison",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "iterations": int(args.iterations),
            "seeds": [int(s) for s in args.seeds],
            "algorithms": algorithms,
            "param_set": str(args.param_set),
            "output_root": str(output_root),
            "api_base": str(args.api_base),
            "llm_model": str(args.model),
            "api_key": "<redacted>" if (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")) else "",
            "timing_scope": "wall time per seed, including optimizer.run and save_results",
            "budget_note": (
                "ParEGO and NSGA-II use iterations + parego_random_init total simulator evaluations; "
                "LLMBO-MO uses iterations BO steps plus llm_warmstart and llm_random_init initialization points."
            ),
        },
        "config": {
            "parego_variant": str(args.parego_variant),
            "parego_random_init": int(args.parego_random_init),
            "nsga2_pop_size": int(args.nsga2_pop_size),
            "nsga2_evals": int(args.nsga2_evals or (int(args.iterations) + int(args.parego_random_init))),
            "llmbo_preset": str(args.llmbo_preset),
            "llm_warmstart": int(args.llm_warmstart),
            "llm_random_init": int(args.llm_random_init),
            "llm_candidates": int(args.llm_candidates),
            "llm_temperature": float(args.llm_temperature),
            "llm_samples": int(args.llm_samples),
        },
        "records": {
            algorithm: records_by_algorithm.get(algorithm, [])
            for algorithm in algorithms
        },
        "aggregates": {
            algorithm: _aggregate(records_by_algorithm.get(algorithm, []))
            for algorithm in algorithms
        },
    }


def _format_time_cell(aggregate: Dict[str, Any], best_mean: float) -> Tuple[str, bool]:
    timing = aggregate.get("runtime_s") or {}
    mean = float(timing.get("mean", 0.0) or 0.0)
    std = float(timing.get("std", 0.0) or 0.0)
    return f"{mean:.1f}\n($\\pm${std:.1f})", abs(mean - best_mean) <= 1e-9


def _plot_time_table(report: Dict[str, Any], output_root: Path) -> Dict[str, str]:
    image_root = (
        PROJECT_ROOT
        / "Compare_Exp"
        / "images"
        / Path(str(output_root.name)).name
    )
    image_root.mkdir(parents=True, exist_ok=True)

    algorithms = [item for item in ALGORITHM_ORDER if item[0] in report["meta"]["algorithms"]]
    means = [
        float((report["aggregates"][key].get("runtime_s") or {}).get("mean", 0.0) or 0.0)
        for key, _ in algorithms
    ]
    best_mean = min(means) if means else 0.0

    blue = "#0000ff"
    fig, ax = plt.subplots(figsize=(9.4, 2.35), dpi=220)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.hlines([0.92, 0.36], xmin=0.0, xmax=1.0, colors=blue, linewidths=1.8)

    left_x = 0.17
    algo_start = 0.43
    algo_end = 0.92
    xs = [algo_start + i * ((algo_end - algo_start) / max(1, len(algorithms) - 1)) for i in range(len(algorithms))]
    title = f"{report['meta']['param_set']}, {report['meta']['iterations']}-round optimization"

    text_common = {"color": blue, "fontname": "Times New Roman"}
    ax.text(left_x, 0.72, "Case", ha="center", va="center", fontsize=22, **text_common)
    ax.text((algo_start + algo_end) / 2, 0.72, title, ha="center", va="center", fontsize=21, **text_common)

    for x, (_, label) in zip(xs, algorithms):
        ax.text(x, 0.48, label, ha="center", va="center", fontsize=21, **text_common)

    ax.text(
        left_x,
        0.14,
        "Computational time (s)",
        ha="center",
        va="center",
        fontsize=21,
        **text_common,
    )
    for x, (key, _) in zip(xs, algorithms):
        cell, is_best = _format_time_cell(report["aggregates"][key], best_mean)
        ax.text(
            x,
            0.14,
            cell,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold" if is_best else "normal",
            linespacing=1.45,
            **text_common,
        )

    png_path = image_root / "computational_time_table.png"
    pdf_path = image_root / "computational_time_table.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args()

    algorithms = ["parego", "nsga2", "llmbo_mo"] if "all" in args.algorithms else list(args.algorithms)
    output_root = Path(args.output_root) if args.output_root is not None else _default_output_root(args.iterations)
    output_root.mkdir(parents=True, exist_ok=True)

    records_by_algorithm: Dict[str, List[Dict[str, Any]]] = {algorithm: [] for algorithm in algorithms}
    failures: List[Dict[str, str]] = []

    logger.info("Output root: %s", output_root)
    logger.info("Algorithms: %s", algorithms)
    logger.info("Seeds: %s", args.seeds)
    logger.info("Iterations: %s", args.iterations)

    for seed in args.seeds:
        for algorithm in algorithms:
            try:
                summary = _run_or_load(int(seed), algorithm, args, output_root)
                records_by_algorithm[algorithm].append(_record_from_summary(int(seed), algorithm, summary))
            except Exception as exc:
                logger.error("[%s] seed=%s failed: %s", algorithm, seed, exc, exc_info=True)
                failures.append({"algorithm": algorithm, "seed": str(seed), "error": str(exc)})

    report = _build_report(args, output_root, algorithms, records_by_algorithm)
    if failures:
        report["failures"] = failures

    report_path = output_root / "computational_time_report.json"
    _write_json(report_path, report)
    image_paths = _plot_time_table(report, output_root)
    report["image_paths"] = image_paths
    _write_json(report_path, report)

    logger.info("Report saved: %s", report_path)
    logger.info("PNG saved: %s", image_paths["png"])
    logger.info("PDF saved: %s", image_paths["pdf"])

    print("\nComputational time summary")
    for key, label in ALGORITHM_ORDER:
        if key not in report["aggregates"]:
            continue
        timing = report["aggregates"][key]["runtime_s"]
        print(f"  {label}: {timing['mean']:.1f} +/- {timing['std']:.1f} s ({timing['min']:.1f}-{timing['max']:.1f})")
    if failures:
        print(f"  failures: {len(failures)}")
    print(f"  report: {report_path}")
    print(f"  figure: {image_paths['png']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
