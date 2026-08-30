#!/usr/bin/env python
"""
exp/hparam_search.py
=====================
Optuna-based hyperparameter search for LLAMBO-MO.

Usage:
    # Quick test (2 trials, minimal budget)
    python exp/hparam_search.py --n-trials 2 --budget-iters 5 --n-warmstart 3

    # Full search
    python exp/hparam_search.py --n-trials 50 --study-name my_study

    # Resume interrupted study
    python exp/hparam_search.py --study-name my_study --n-trials 100

    # Override search space via CLI
    python exp/hparam_search.py --n-trials 20 --search-space narrow

Goal: Maximize final canonical_hv from each experiment run.
Storage: SQLite at exp/optuna_results/{study_name}.db (interrupt-safe, migratable).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

# 加载 .env（确保 schema 中的 os.getenv 能读到 LLM 配置）
try:
    from dotenv import load_dotenv
    _dotenv = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_dotenv if _dotenv.exists() else None)
except ImportError:
    pass

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("optuna not installed. Run: conda install optuna or pip install optuna")
    sys.exit(1)

from config.schema import (
    Config,
    BOConfig,
    GPConfig,
    MOBOConfig,
    AcquisitionConfig,
    LLMConfig,
    LLMWarmStartConfig,
    create_minimal_config,
)
from config.presets import EXPERIMENT_PRESETS
from main import build_optimizer_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 全局 verbose 标志，由 CLI --verbose 设置
_VERBOSE = False
_VERBOSE_LOGGER = logging.getLogger("llm")
_VERBOSE_LOGGER.setLevel(logging.DEBUG)

RESULT_DIR = ROOT / "exp" / "optuna_results"


# ═══════════════════════════════════════════════════════════════════════════
# Search space definitions
# ═══════════════════════════════════════════════════════════════════════════

SEARCH_SPACES = {
    "default": {
        # GP
        "gp_kernel_nu": {"type": "categorical", "choices": [1.5, 2.5, 3.5]},
        "gp_alpha_log": {"type": "float", "low": -7, "high": -3},  # log10(alpha)
        "gp_n_restarts": {"type": "int", "low": 3, "high": 10},
        # MOBO
        "mobo_eta": {"type": "float", "low": 0.01, "high": 0.20},
        "mobo_n_weights": {"type": "int", "low": 10, "high": 30},
        # Acquisition
        "acq_n_cand": {"type": "int", "low": 10, "high": 30},
        "acq_n_select": {"type": "int", "low": 1, "high": 3},
        # BO warmstart
        "bo_n_warmstart": {"type": "int", "low": 3, "high": 15},
        "bo_n_random_init": {"type": "int", "low": 3, "high": 10},
        # LLM temperature
        "llm_temperature": {"type": "float", "low": 0.3, "high": 1.0},
    },
    "narrow": {
        # Tighter range around known good values
        "gp_kernel_nu": {"type": "categorical", "choices": [2.5]},
        "gp_alpha_log": {"type": "float", "low": -6, "high": -4},
        "gp_n_restarts": {"type": "int", "low": 3, "high": 7},
        "mobo_eta": {"type": "float", "low": 0.03, "high": 0.10},
        "mobo_n_weights": {"type": "int", "low": 12, "high": 25},
        "acq_n_cand": {"type": "int", "low": 12, "high": 25},
        "acq_n_select": {"type": "int", "low": 1, "high": 2},
        "bo_n_warmstart": {"type": "int", "low": 5, "high": 10},
        "bo_n_random_init": {"type": "int", "low": 3, "high": 6},
        "llm_temperature": {"type": "float", "low": 0.5, "high": 0.8},
    },
    "wide": {
        # Explore aggressively
        "gp_kernel_nu": {"type": "categorical", "choices": [0.5, 1.5, 2.5, 3.5, 4.5]},
        "gp_alpha_log": {"type": "float", "low": -8, "high": -2},
        "gp_n_restarts": {"type": "int", "low": 1, "high": 15},
        "mobo_eta": {"type": "float", "low": 0.005, "high": 0.30},
        "mobo_n_weights": {"type": "int", "low": 5, "high": 40},
        "acq_n_cand": {"type": "int", "low": 5, "high": 40},
        "acq_n_select": {"type": "int", "low": 1, "high": 5},
        "bo_n_warmstart": {"type": "int", "low": 1, "high": 20},
        "bo_n_random_init": {"type": "int", "low": 0, "high": 15},
        "llm_temperature": {"type": "float", "low": 0.1, "high": 1.5},
    },
}


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    t = spec["type"]
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"])
    elif t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    elif t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown param type: {t}")


# ═══════════════════════════════════════════════════════════════════════════
# Build Config from Optuna suggestions
# ═══════════════════════════════════════════════════════════════════════════

def build_trial_config(
    trial: optuna.Trial,
    space_name: str,
    n_iterations: int,
    seed: int,
) -> Config:
    space = SEARCH_SPACES[space_name]
    vals = {k: _suggest(trial, k, v) for k, v in space.items()}

    return Config(
        bo=BOConfig(
            n_iterations=n_iterations,
            n_warmstart=int(vals["bo_n_warmstart"]),
            n_random_init=int(vals["bo_n_random_init"]),
        ),
        gp=GPConfig(
            kernel_nu=float(vals["gp_kernel_nu"]),
            alpha=10 ** vals["gp_alpha_log"],
            n_restarts_optimizer=int(vals["gp_n_restarts"]),
        ),
        mobo=MOBOConfig(
            eta=float(vals["mobo_eta"]),
            n_weights=int(vals["mobo_n_weights"]),
        ),
        acquisition=AcquisitionConfig(
            n_cand=int(vals["acq_n_cand"]),
            n_select=int(vals["acq_n_select"]),
        ),
        llm=LLMConfig(
            warmstart=LLMWarmStartConfig(
                temperature=float(vals["llm_temperature"]),
            ),
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Trial runner
# ═══════════════════════════════════════════════════════════════════════════

def run_trial(
    trial: optuna.Trial,
    config: Config,
    preset: str | None,
    trial_dir: Path,
) -> float:
    """Run one experiment, return canonical_hv."""
    from llmbo.optimizer import BayesOptimizer

    trial_dir.mkdir(parents=True, exist_ok=True)

    # Verbose 模式：为当前 trial 设置 DEBUG 级别的文件日志
    trial_fh = None
    if _VERBOSE:
        log_path = trial_dir / "debug.log"
        trial_fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        trial_fh.setLevel(logging.DEBUG)
        trial_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
        # 为关键模块开启 DEBUG
        for mod_name in ("llm", "llmbo", "llm.llm_interface", "llmbo.optimizer"):
            mod_logger = logging.getLogger(mod_name)
            mod_logger.setLevel(logging.DEBUG)
            mod_logger.addHandler(trial_fh)

    args = argparse.Namespace(preset=preset, mock=False)
    flat = build_optimizer_config(config, args, trial_dir)

    optimizer = BayesOptimizer(config=flat)
    t0 = time.time()
    optimizer.run()
    elapsed = time.time() - t0

    optimizer.save_results(str(trial_dir))

    # Read summary for objective
    summary_path = trial_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"summary.json not found after trial: {trial_dir}")

    with open(summary_path) as f:
        summary = json.load(f)

    canonical_hv = summary.get("canonical_hv", summary.get("hypervolume_canonical", 0.0))

    # Save trial metadata
    meta = {
        "trial_number": trial.number,
        "canonical_hv": canonical_hv,
        "elapsed_s": elapsed,
        "params": dict(trial.params),
        "preset": preset,
        "n_total": summary.get("n_total"),
        "n_feasible": summary.get("n_feasible"),
        "pareto_size": summary.get("pareto_size"),
    }
    with open(trial_dir / "trial_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 关闭 trial 级别的 debug file handler
    if trial_fh is not None:
        trial_fh.close()
        for mod_name in ("llm", "llmbo", "llm.llm_interface", "llmbo.optimizer"):
            logging.getLogger(mod_name).removeHandler(trial_fh)

    return float(canonical_hv)


# ═══════════════════════════════════════════════════════════════════════════
# Objective
# ═══════════════════════════════════════════════════════════════════════════

def make_objective(
    study_name: str,
    space_name: str,
    n_iterations: int,
    preset: str | None,
    base_seed: int,
):
    def objective(trial: optuna.Trial) -> float:
        seed = base_seed + trial.number
        np.random.seed(seed)

        config = build_trial_config(trial, space_name, n_iterations, seed)
        trial_dir = RESULT_DIR / study_name / f"trial_{trial.number:03d}"

        logger.info(
            "Trial %d starting | dir=%s | preset=%s | seed=%d",
            trial.number, trial_dir, preset, seed,
        )

        try:
            hv = run_trial(trial, config, preset, trial_dir)
        except Exception as exc:
            logger.error("Trial %d failed: %s", trial.number, exc)
            raise

        logger.info("Trial %d done | canonical_hv=%.6f", trial.number, hv)
        return hv

    return objective


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for LLAMBO-MO")
    p.add_argument("--study-name", type=str, default="hparam_search",
                   help="Optuna study name (default: hparam_search)")
    p.add_argument("--n-trials", type=int, default=20,
                   help="Number of Optuna trials (default: 20)")
    p.add_argument("--search-space", type=str, default="default",
                   choices=list(SEARCH_SPACES.keys()),
                   help="Search space definition (default: default)")
    p.add_argument("--budget-iters", type=int, default=None,
                   help="Override BO iterations per trial (for quick testing)")
    p.add_argument("--n-warmstart", type=int, default=None,
                   help="Override warmstart budget (for quick testing)")
    p.add_argument("--preset", type=str, default=None,
                   choices=list(EXPERIMENT_PRESETS.keys()),
                   help="Fix experiment preset for all trials")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (default: 42)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print first trial config and exit (no experiment run)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging with LLM I/O, saved per-trial log files")
    return p


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    global _VERBOSE
    _VERBOSE = args.verbose
    if _VERBOSE:
        logger.info("Verbose mode: DEBUG logs → trial_XXX/debug.log (includes LLM I/O)")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = RESULT_DIR / f"{args.study_name}.db"
    storage = f"sqlite:///{db_path}"

    n_iterations = args.budget_iters if args.budget_iters else 50
    if args.n_warmstart:
        logger.info("Note: --n-warmstart overrides search space for bo_n_warmstart")

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    if args.dry_run:
        trial = study.ask()
        config = build_trial_config(trial, args.search_space, n_iterations, args.seed)
        print(json.dumps(config.model_dump(), indent=2, default=str))
        return 0

    print(f"\n{'=' * 60}")
    print(f"Study: {args.study_name}")
    print(f"Storage: {db_path}")
    print(f"Search space: {args.search_space}")
    print(f"Trials: {args.n_trials}")
    print(f"Budget/trial: {n_iterations} BO iterations")
    print(f"Preset: {args.preset or '(none, default Config)'}")
    print(f"Existing trials: {len(study.trials)}")
    print(f"{'=' * 60}\n")

    objective = make_objective(
        study_name=args.study_name,
        space_name=args.search_space,
        n_iterations=n_iterations,
        preset=args.preset,
        base_seed=args.seed,
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"Study complete: {args.study_name}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best canonical_hv: {study.best_value:.6f}")
    print(f"Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"  {k}: {v}")
    print(f"Total trials: {len(study.trials)}")
    print(f"{'=' * 60}")

    # Save best config as JSON
    best_config_path = RESULT_DIR / args.study_name / "best_config.json"
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_config_path, "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_hv": study.best_value,
            "best_params": study.best_params,
        }, f, indent=2)
    print(f"Best config saved to: {best_config_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
