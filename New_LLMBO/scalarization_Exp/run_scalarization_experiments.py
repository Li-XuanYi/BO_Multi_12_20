from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer
from llmbo.scalarization import OBJECTIVE_PREPROCESS_MODES, canonicalize_objective_preprocess_mode
from utils.model_labels import canonical_model_label


DEFAULT_SEEDS = [8409, 8410, 8411, 8412, 8413]
DEFAULT_MODES = ["minmax", "zscore", "none"]
MODEL_NAME = "deepseek-v3-thinking"


def _default_output_root() -> Path:
    date_tag = date.today().isoformat().replace("-", "_")
    return (
        PROJECT_ROOT
        / "scalarization_Exp"
        / "experiment_records"
        / f"scalarization_llmbo_mo_5seeds_50iter_gpt41nano_{date_tag}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLMBO-MO scalarization preprocessing experiments.")
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--modes", type=str, nargs="+", default=DEFAULT_MODES, choices=OBJECTIVE_PREPROCESS_MODES)
    parser.add_argument("--param-set", type=str, default="Chen2020", choices=["Chen2020", "Ecker2015", "ORegan2022"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.nuwaapi.com/v1",
    )
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.fmean(items)) if items else 0.0


def _std(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.stdev(items)) if len(items) > 1 else 0.0


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(output_root: Path, seed: int, mode: str) -> Path:
    return output_root / f"seed{int(seed)}" / canonicalize_objective_preprocess_mode(mode)


def _build_config(
    *,
    output_dir: Path,
    seed: int,
    mode: str,
    iterations: int,
    param_set: str,
    api_key: str,
    api_base: str,
    shared_random_init_cache: Path,
    shared_warmstart_cache: Path,
) -> Dict[str, Any]:
    return {
        "experiment_preset": "warmstart_region_lifted_gp_force_pool_tuned",
        "max_iterations": int(iterations),
        "n_warmstart": 3,
        "n_random_init": 3,
        "n_candidates": 15,
        "n_select": 1,
        "w_sample_seed": int(seed),
        "init_seed": 2026 + int(seed),
        "objective_preprocess_mode": canonicalize_objective_preprocess_mode(mode),
        "battery_param_set": param_set,
        "llm_backend": "openai",
        "llm_model": MODEL_NAME,
        "llm_api_base": api_base,
        "llm_api_key": api_key,
        "llm_n_samples": 1,
        "llm_temperature": 0.0,
        "warmstart_temperature": 0.0,
        "random_init_cache_path": str(shared_random_init_cache),
        "warmstart_cache_path": str(shared_warmstart_cache),
        "warmstart_cache_mode": "read_write",
        "warmstart_cache_use_selected": True,
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "checkpoint_every": 99,
    }


def _run_single(output_dir: Path, cfg: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = BayesOptimizer(config=cfg)
    optimizer.run()
    optimizer.save_results(str(output_dir))


def _record_for_run(seed: int, mode: str, run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {
            "seed": int(seed),
            "mode": mode,
            "status": "missing",
            "summary_path": str(summary_path),
        }

    summary = _load_json(summary_path)
    return {
        "seed": int(seed),
        "mode": mode,
        "status": "ok",
        "summary_path": str(summary_path),
        "canonical_hv": float(summary.get("canonical_hv", summary.get("hypervolume_canonical", 0.0))),
        "display_hv": float(summary.get("display_hv", summary.get("hypervolume", 0.0))),
        "hypervolume_raw": float(summary.get("hypervolume_raw", 0.0)),
        "pareto_size": int(summary.get("pareto_size", 0)),
        "n_total": int(summary.get("n_total", 0)),
        "n_feasible": int(summary.get("n_feasible", 0)),
        "objective_preprocess_mode": str((summary.get("config") or {}).get("objective_preprocess_mode", mode)),
    }


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [record for record in records if record.get("status") == "ok"]
    canonical = [float(record["canonical_hv"]) for record in ok]
    display = [float(record["display_hv"]) for record in ok]
    raw = [float(record["hypervolume_raw"]) for record in ok]
    pareto = [float(record["pareto_size"]) for record in ok]
    return {
        "n_runs": int(len(ok)),
        "canonical_hv": {
            "mean": _mean(canonical),
            "std": _std(canonical),
            "min": min(canonical) if canonical else 0.0,
            "max": max(canonical) if canonical else 0.0,
        },
        "display_hv": {"mean": _mean(display), "std": _std(display)},
        "hypervolume_raw": {"mean": _mean(raw), "std": _std(raw)},
        "pareto_size": {"mean": _mean(pareto), "std": _std(pareto)},
        "runs": ok,
        "missing_runs": [record for record in records if record.get("status") != "ok"],
    }


def _pairwise_delta(lhs: List[Dict[str, Any]], rhs: List[Dict[str, Any]]) -> Dict[str, Any]:
    lhs_ok = {int(record["seed"]): record for record in lhs if record.get("status") == "ok"}
    rhs_ok = {int(record["seed"]): record for record in rhs if record.get("status") == "ok"}
    shared = sorted(set(lhs_ok) & set(rhs_ok))
    deltas = [float(lhs_ok[seed]["canonical_hv"]) - float(rhs_ok[seed]["canonical_hv"]) for seed in shared]
    wins = sum(1 for value in deltas if value > 0.0)
    return {
        "lhs": lhs[0]["mode"] if lhs else None,
        "rhs": rhs[0]["mode"] if rhs else None,
        "shared_seeds": shared,
        "mean_canonical_hv_delta": _mean(deltas),
        "std_canonical_hv_delta": _std(deltas),
        "wins": int(wins),
        "total": int(len(shared)),
    }


def _write_report(args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    modes = [canonicalize_objective_preprocess_mode(mode) for mode in args.modes]
    seeds = [int(seed) for seed in args.seeds]
    records: List[Dict[str, Any]] = []
    by_mode: Dict[str, List[Dict[str, Any]]] = {}

    for mode in modes:
        mode_records = []
        for seed in seeds:
            record = _record_for_run(seed, mode, _run_dir(output_root, seed, mode))
            records.append(record)
            mode_records.append(record)
        by_mode[mode] = mode_records

    comparisons: Dict[str, Any] = {}
    for lhs in modes:
        for rhs in modes:
            if lhs == rhs:
                continue
            comparisons[f"{lhs}_vs_{rhs}"] = _pairwise_delta(by_mode[lhs], by_mode[rhs])

    report = {
        "meta": {
            "experiment": "scalarization_objective_preprocess",
            "iterations": int(args.iterations),
            "seeds": seeds,
            "modes": modes,
            "output_root": str(output_root),
            "model": MODEL_NAME,
            "model_display": canonical_model_label(MODEL_NAME),
            "api_base": args.api_base,
            "hv_metric": "canonical_hv",
        },
        "records": records,
        "aggregates": {mode: _aggregate(mode_records) for mode, mode_records in by_mode.items()},
        "comparisons": comparisons,
    }
    report_path = output_root / "report_5seeds.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    modes = [canonicalize_objective_preprocess_mode(mode) for mode in args.modes]
    seeds = [int(seed) for seed in args.seeds]
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not args.summarize_only and not api_key:
        raise RuntimeError("Set LLM_API_KEY or OPENAI_API_KEY before running real API experiments.")

    if not args.summarize_only:
        for seed in seeds:
            seed_root = output_root / f"seed{seed}"
            shared_random = seed_root / "shared_random_init_cache.json"
            shared_warmstart = seed_root / "shared_warmstart_cache.json"
            for mode in modes:
                run_dir = _run_dir(output_root, seed, mode)
                summary_path = run_dir / "summary.json"
                if args.skip_existing and summary_path.exists():
                    print(f"[skip] seed={seed} mode={mode} already has {summary_path}")
                    continue
                print(f"[run] seed={seed} mode={mode} -> {run_dir}")
                cfg = _build_config(
                    output_dir=run_dir,
                    seed=seed,
                    mode=mode,
                    iterations=int(args.iterations),
                    param_set=args.param_set,
                    api_key=api_key,
                    api_base=str(args.api_base),
                    shared_random_init_cache=shared_random,
                    shared_warmstart_cache=shared_warmstart,
                )
                _run_single(run_dir, cfg)

    _write_report(args, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
