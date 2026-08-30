from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer
from llmbo.riesz_cache import load_or_generate_riesz


OLD_SEED8409_SUMMARY = (
    PROJECT_ROOT
    / "Compare_Exp"
    / "experiment_records"
    / "seed8409_llmbo_vs_parego_50iter"
    / "seed8409"
    / "llmbo_mo"
    / "summary.json"
)


def _load_reference_config() -> Dict[str, Any]:
    if not OLD_SEED8409_SUMMARY.exists():
        return {}
    with OLD_SEED8409_SUMMARY.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload.get("config") or {})


def _api_defaults() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    try:
        from Compare_Exp.Exp.run_three_algo_comparison import LLMBO_CONFIG

        defaults.update(
            {
                "llm_backend": LLMBO_CONFIG.get("llm_backend", "openai"),
                "llm_api_base": LLMBO_CONFIG.get("llm_api_base"),
                "llm_api_key": LLMBO_CONFIG.get("llm_api_key"),
            }
        )
    except Exception:
        pass

    env_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    env_base = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if env_key:
        defaults["llm_api_key"] = env_key
    if env_base:
        defaults["llm_api_base"] = env_base
    return defaults


def _build_config(output_dir: Path, seed: int) -> Dict[str, Any]:
    cfg = _load_reference_config()
    cfg.update(_api_defaults())
    cfg.update(
        {
            "experiment_preset": "warmstart_region_lifted_gp_force_pool_tuned",
            "max_iterations": 50,
            "n_warmstart": 3,
            "n_random_init": 3,
            "n_candidates": 15,
            "n_select": 1,
            "w_sample_seed": int(seed),
            "init_seed": 2026 + int(seed),
            "weight_strategy": "riesz_relaxed_cycle",
            "weight_sampling_mode": "cycle_without_replacement",
            "riesz_n_div": 10,
            "riesz_s": 2.0,
            "riesz_n_iter": 300,
            "riesz_lr": 5e-3,
            "riesz_seed": 42,
            "weight_eps_min": 0.01,
            "llm_backend": "openai",
            "llm_model": cfg.get("llm_model") or "gpt-4.1-mini",
            "llm_n_samples": 1,
            "llm_temperature": 0.0,
            "warmstart_temperature": 0.0,
            "checkpoint_dir": str(output_dir / "checkpoints"),
            "checkpoint_every": 99,
        }
    )
    return cfg


def _weight_diagnostics(cfg: Dict[str, Any]) -> Dict[str, Any]:
    W = load_or_generate_riesz(
        n_obj=3,
        n_div=int(cfg["riesz_n_div"]),
        s=float(cfg["riesz_s"]),
        n_iter=int(cfg["riesz_n_iter"]),
        lr=float(cfg["riesz_lr"]),
        seed=int(cfg["riesz_seed"]),
    )
    W_round = np.round(W, 6)
    return {
        "shape": list(W.shape),
        "unique_count": int(np.unique(W_round, axis=0).shape[0]),
        "min": W.min(axis=0).tolist(),
        "max": W.max(axis=0).tolist(),
        "mean": W.mean(axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run seed8409 LLMBO-MO after Riesz weight fix.")
    parser.add_argument("--seed", type=int, default=8409)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "optimized_experiments" / "llmbo_mo_seed8409_50iter_fixed_riesz_2026_05_09",
    )
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"seed{args.seed}" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_config(output_dir, args.seed)
    weight_info = _weight_diagnostics(cfg)
    print(
        json.dumps(
            {
                "event": "start",
                "seed": args.seed,
                "output_dir": str(output_dir),
                "weight_info": weight_info,
                "max_iterations": cfg["max_iterations"],
                "n_warmstart": cfg["n_warmstart"],
                "n_random_init": cfg["n_random_init"],
                "llm_model": cfg["llm_model"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    optimizer = BayesOptimizer(config=cfg)
    db = optimizer.run()
    optimizer.save_results(str(output_dir))

    metadata = {
        "seed": int(args.seed),
        "output_dir": str(output_dir),
        "weight_info": weight_info,
        "n_total": db.size,
        "n_feasible": db.n_feasible,
        "pareto_size": db.pareto_size,
        "display_hv": db.compute_hypervolume(),
        "canonical_hv": db.compute_hypervolume_canonical(),
        "hypervolume_raw": db.compute_hypervolume_raw(),
        "finished_at": datetime.now().isoformat(),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps({"event": "done", **metadata}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
