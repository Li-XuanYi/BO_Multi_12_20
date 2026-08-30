"""Run controlled Chen2020 sensitivity studies with archived LLM replay.

The study holds the six initialization points, per-iteration Riesz weights,
and raw region-preference payloads fixed for each seed.  It therefore makes no
online LLM calls.  Two factors are varied independently:

1. the LGBO posterior mean-shift budget ``region_lift_lgbo_shift_mean_budget``;
2. objective preprocessing (dynamic min-max, z-score scaling, or no scaling).

The source archive is the five-seed adaptive four-way experiment from
2026-05-22.  Only the archived Full arm is used as replay material.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer
from llmbo.region_lifted_gp import LLMRegionPreference, parse_region_preference_payload


DEFAULT_ARCHIVE_ROOT = (
    PROJECT_ROOT
    / "Ablation_Exp"
    / "experiment_records"
    / "adaptive4_5seeds_50iter_deepseek_v3_2026_05_22"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Ablation_Exp"
    / "experiment_records"
    / "replayed_sensitivity_5seeds_50iter_2026_08_06"
)
DEFAULT_SEEDS = [8409, 8410, 8411, 8412, 8413]
DEFAULT_BUDGETS = [0.005, 0.0125, 0.025, 0.05, 0.1]
DEFAULT_NORMALIZATIONS = ["minmax", "zscore", "none"]
BASE_BUDGET = 0.025
BASE_NORMALIZATION = "minmax"
N_INIT = 6


@dataclass(frozen=True)
class StudySetting:
    key: str
    shift_budget: float
    normalization: str
    panels: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "shift_budget": float(self.shift_budget),
            "normalization": self.normalization,
            "panels": list(self.panels),
        }


class RegionPreferenceReplay:
    """Serve archived raw payloads by zero-based BO iteration."""

    def __init__(self, payloads: Mapping[int, Mapping[str, Any]]) -> None:
        self._payloads = {int(k): dict(v) for k, v in payloads.items()}
        self.calls: List[int] = []

    def query(self, state: Mapping[str, Any]) -> LLMRegionPreference:
        iteration = int(state.get("iteration", len(self.calls)))
        self.calls.append(iteration)
        payload = self._payloads.get(iteration)
        if payload is None:
            return LLMRegionPreference.none("replay_missing")
        return parse_region_preference_payload(payload, log_level=logging.DEBUG)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def _budget_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _setting_key(shift_budget: float, normalization: str) -> str:
    return f"budget_{_budget_token(shift_budget)}__{normalization}"


def build_settings(
    budgets: Iterable[float],
    normalizations: Iterable[str],
) -> List[StudySetting]:
    merged: Dict[tuple[float, str], set[str]] = {}
    for budget in budgets:
        merged.setdefault((float(budget), BASE_NORMALIZATION), set()).add("shift_budget")
    for normalization in normalizations:
        merged.setdefault((BASE_BUDGET, str(normalization)), set()).add("normalization")
    return [
        StudySetting(
            key=_setting_key(budget, normalization),
            shift_budget=budget,
            normalization=normalization,
            panels=tuple(sorted(panels)),
        )
        for (budget, normalization), panels in sorted(merged.items())
    ]


def _archive_paths(archive_root: Path, seed: int) -> tuple[Path, Path]:
    run_root = archive_root / f"seed{int(seed)}" / "llmbo_mo"
    return run_root / "summary.json", run_root / "database.json"


def _load_replay_material(archive_root: Path, seed: int) -> Dict[str, Any]:
    summary_path, database_path = _archive_paths(archive_root, seed)
    if not summary_path.exists() or not database_path.exists():
        raise FileNotFoundError(f"Missing Full-arm archive for seed {seed}: {summary_path}")

    summary = _json_load(summary_path)
    database = _json_load(database_path)
    observations = list(database.get("observations") or [])
    if len(observations) < N_INIT:
        raise ValueError(f"Seed {seed} archive has fewer than {N_INIT} observations")
    initial = observations[:N_INIT]
    if not all(str(item.get("source")) == "llm_warmstart" for item in initial):
        raise ValueError(f"Seed {seed} first {N_INIT} observations are not all LLM warm starts")
    init_points = [list(map(float, item["theta"])) for item in initial]

    raw_payloads: Dict[int, Dict[str, Any]] = {}
    for index, item in enumerate(summary.get("region_lift_telemetry") or []):
        preference = item.get("preference") if isinstance(item, Mapping) else None
        raw_response = preference.get("raw_response") if isinstance(preference, Mapping) else None
        if index < int(summary.get("config", {}).get("region_lift_active_until", 12)):
            if not isinstance(raw_response, Mapping):
                raise ValueError(f"Seed {seed} lacks a raw region payload at iteration {index}")
            raw_payloads[index] = dict(raw_response)

    base_config = dict(summary.get("config") or {})
    if int(base_config.get("w_sample_seed", -1)) != int(seed):
        raise ValueError(f"Seed mismatch in archived config for seed {seed}")
    if str(base_config.get("battery_param_set", "Chen2020")) != "Chen2020":
        raise ValueError(f"Unexpected battery parameterization for seed {seed}")

    return {
        "base_config": base_config,
        "init_points": init_points,
        "raw_payloads": raw_payloads,
        "summary_path": summary_path,
        "database_path": database_path,
        "source_final_sHV": float(summary["canonical_hv"]),
    }


def _build_config(
    material: Mapping[str, Any],
    setting: StudySetting,
    seed: int,
    iterations: int,
    run_dir: Path,
) -> Dict[str, Any]:
    config = dict(material["base_config"])
    config.update(
        {
            "max_iterations": int(iterations),
            "fixed_init_points": material["init_points"],
            "fixed_init_source": "archived_llm_warmstart_replay",
            "llm_backend": "mock",
            "llm_model": "archived-deepseek-v3-preference-replay",
            "llm_api_base": "",
            "llm_api_key": "",
            "enable_iterative_guidance": False,
            "enable_gp_llm_coupling": False,
            "enable_acq_prior_coupling": False,
            "enable_proposal_sampler": False,
            "enable_llm_rerank": False,
            "enable_region_lifted_gp": True,
            "enable_warmstart_portfolio": False,
            "warmstart_cache_path": None,
            "warmstart_cache_mode": "disabled",
            "objective_preprocess_mode": setting.normalization,
            "region_lift_lgbo_shift_mean_budget": float(setting.shift_budget),
            "region_lift_active_until": min(
                int(config.get("region_lift_active_until", 12)), int(iterations)
            ),
            "w_sample_seed": int(seed),
            "init_seed": int(seed),
            "checkpoint_dir": str(run_dir / "checkpoints"),
            "checkpoint_every": max(int(iterations) + 1, 9999),
        }
    )
    return config


def _summary_is_complete(summary_path: Path, expected_total: int) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = _json_load(summary_path)
        return (
            int(summary.get("n_total", -1)) == int(expected_total)
            and math.isfinite(float(summary.get("canonical_hv")))
            and len(summary.get("hv_trace") or []) == int(expected_total)
        )
    except Exception:
        return False


def _run_one(job: Mapping[str, Any]) -> Dict[str, Any]:
    seed = int(job["seed"])
    setting = StudySetting(**job["setting"])
    archive_root = Path(job["archive_root"])
    output_root = Path(job["output_root"])
    iterations = int(job["iterations"])
    skip_existing = bool(job["skip_existing"])
    run_dir = output_root / f"seed{seed}" / setting.key
    summary_path = run_dir / "summary.json"
    expected_total = N_INIT + iterations

    if skip_existing and _summary_is_complete(summary_path, expected_total):
        summary = _json_load(summary_path)
        return {
            "status": "skipped",
            "seed": seed,
            "setting": setting.to_dict(),
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
            "canonical_hv": float(summary["canonical_hv"]),
            "duration_s": float(summary.get("sensitivity_replay", {}).get("duration_s", 0.0)),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        material = _load_replay_material(archive_root, seed)
        config = _build_config(material, setting, seed, iterations, run_dir)
        replay = RegionPreferenceReplay(material["raw_payloads"])

        optimizer = BayesOptimizer(config=config)
        optimizer.setup()
        optimizer.llm.query_region_preference = replay.query
        optimizer.run_initialization()
        optimizer.initialize_acquisition()
        optimizer.run_optimization_loop()
        optimizer.save_results(str(run_dir))

        duration = time.perf_counter() - start
        summary = _json_load(summary_path)
        expected_replay_calls = min(
            int(config.get("region_lift_active_until", 0)), int(iterations)
        )
        if int(summary.get("n_total", -1)) != expected_total:
            raise RuntimeError(
                f"Expected {expected_total} evaluations, got {summary.get('n_total')}"
            )
        if replay.calls != list(range(expected_replay_calls)):
            raise RuntimeError(
                f"Replay iteration mismatch: expected 0..{expected_replay_calls - 1}, "
                f"got {replay.calls}"
            )

        provenance = {
            "protocol": "fixed-initialization and fixed-region-payload replay",
            "online_llm_calls": 0,
            "seed": seed,
            "setting": setting.to_dict(),
            "iterations": iterations,
            "evaluations": expected_total,
            "replayed_region_iterations": replay.calls,
            "source_summary": str(material["summary_path"]),
            "source_database": str(material["database_path"]),
            "source_summary_sha256": _sha256(material["summary_path"]),
            "source_database_sha256": _sha256(material["database_path"]),
            "source_final_sHV": material["source_final_sHV"],
            "duration_s": duration,
            "completed_at": _utc_now(),
        }
        summary["sensitivity_replay"] = provenance
        _json_dump(summary_path, summary)
        _json_dump(run_dir / "run_manifest.json", provenance)
        return {
            "status": "ok",
            "seed": seed,
            "setting": setting.to_dict(),
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
            "canonical_hv": float(summary["canonical_hv"]),
            "duration_s": duration,
        }
    except Exception as exc:
        failure = {
            "status": "failed",
            "seed": seed,
            "setting": setting.to_dict(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at": _utc_now(),
        }
        _json_dump(run_dir / "failure.json", failure)
        return failure


def _aggregate(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result["setting"]["key"]), []).append(result)

    settings: Dict[str, Any] = {}
    for key, rows in sorted(grouped.items()):
        valid = [row for row in rows if row.get("status") in {"ok", "skipped"}]
        values = [float(row["canonical_hv"]) for row in valid]
        settings[key] = {
            "setting": rows[0]["setting"],
            "n_runs": len(valid),
            "n_failed": len(rows) - len(valid),
            "final_sHV": {
                "mean": statistics.mean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": [
                    {"seed": int(row["seed"]), "value": float(row["canonical_hv"])}
                    for row in sorted(valid, key=lambda item: int(item["seed"]))
                ],
            },
            "duration_s_total": sum(float(row.get("duration_s", 0.0)) for row in valid),
            "runs": list(rows),
        }
    return settings


def _environment_manifest() -> Dict[str, Any]:
    versions: Dict[str, str] = {}
    for name in ("numpy", "scipy", "sklearn", "pybamm", "pymoo"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "not-recorded"))
        except Exception:
            versions[name] = "unavailable"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or "not-recorded",
        "packages": versions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--budget-values", type=float, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--normalization-modes",
        nargs="+",
        choices=DEFAULT_NORMALIZATIONS,
        default=DEFAULT_NORMALIZATIONS,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = build_settings(args.budget_values, args.normalization_modes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "seed": int(seed),
            "setting": {
                "key": setting.key,
                "shift_budget": setting.shift_budget,
                "normalization": setting.normalization,
                "panels": setting.panels,
            },
            "archive_root": str(args.archive_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "iterations": int(args.iterations),
            "skip_existing": bool(args.skip_existing),
        }
        for seed in args.seeds
        for setting in settings
    ]

    manifest = {
        "study": "Chen2020 replayed LLMBO-MO sensitivity",
        "created_at": _utc_now(),
        "archive_root": str(args.archive_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "seeds": list(map(int, args.seeds)),
        "iterations": int(args.iterations),
        "evaluations_per_run": N_INIT + int(args.iterations),
        "settings": [setting.to_dict() for setting in settings],
        "controls": {
            "fixed_per_seed_initialization": True,
            "fixed_per_seed_raw_region_payloads": True,
            "fixed_per_seed_weight_rng": True,
            "online_llm_calls": 0,
            "battery_parameterization": "Chen2020",
            "reporting_metric": "canonical sHV with fixed Chen2020 reporting box",
        },
        "environment": _environment_manifest(),
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in (
                PROJECT_ROOT / "llmbo" / "optimizer.py",
                PROJECT_ROOT / "llmbo" / "scalarization.py",
                PROJECT_ROOT / "llmbo" / "region_lifted_gp.py",
                PROJECT_ROOT / "DataBase" / "database.py",
                Path(__file__).resolve(),
            )
        },
    }
    _json_dump(args.output_root / "manifest.json", manifest)

    results: List[Dict[str, Any]] = []
    workers = max(1, min(int(args.workers), len(jobs)))
    if workers == 1:
        iterator = map(_run_one, jobs)
        for result in iterator:
            results.append(result)
            print(
                f"[{len(results)}/{len(jobs)}] {result.get('status')} "
                f"seed={result.get('seed')} setting={result.get('setting', {}).get('key')} "
                f"sHV={result.get('canonical_hv', 'NA')}",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_run_one, job): job for job in jobs}
            for future in concurrent.futures.as_completed(future_map):
                result = future.result()
                results.append(result)
                print(
                    f"[{len(results)}/{len(jobs)}] {result.get('status')} "
                    f"seed={result.get('seed')} setting={result.get('setting', {}).get('key')} "
                    f"sHV={result.get('canonical_hv', 'NA')}",
                    flush=True,
                )

    report = {
        "meta": manifest,
        "completed_at": _utc_now(),
        "settings": _aggregate(results),
        "failures": [result for result in results if result.get("status") == "failed"],
    }
    _json_dump(args.output_root / "report.json", report)
    print(f"Report: {args.output_root / 'report.json'}", flush=True)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
