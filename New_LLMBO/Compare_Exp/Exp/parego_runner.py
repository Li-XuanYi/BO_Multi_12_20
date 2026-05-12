"""
parego_runner.py — ParEGO baseline runner for paper comparison
===============================================================
Runs ParEGO (Pareto Efficient Global Optimization) as a standalone baseline.
Uses Tchebycheff scalarization with Riesz s-energy or Das-Dennis weight sets.

This runner wraps llmbo.BayesOptimizer with ParEGO-specific presets to ensure
compatible output format with NSGA-II and LLAMBO-MO experiments.

Usage:
    from Compare_Exp.Exp.parego_runner import ParEGORunner
    runner = ParEGORunner(seed=8409, n_evals=56, variant="matlab_reference")
    runner.run()
    runner.save_results("output_dir/")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.optimizer import BayesOptimizer, EXPERIMENT_PRESETS
from DataBase.database import ObservationDB
from utils.constants import DEFAULT_BOUNDS

logger = logging.getLogger(__name__)


class ParEGORunner:
    """
    ParEGO baseline runner compatible with paper comparison framework.

    Two variants are supported:
    - "baseline": Original ParEGO with Riesz s-energy weights (weight_count=30)
    - "matlab_reference": MATLAB-style ParEGO with Das-Dennis weights (n_div=30)
    """

    VARIANT_PRESETS = {
        "baseline": "parego_baseline",
        "matlab_reference": "parego_matlab_reference",
    }

    def __init__(
        self,
        seed: int = 0,
        n_evals: int = 56,
        variant: str = "baseline",
        n_random_init: int = 6,
        param_set: str = "Chen2020",
    ):
        """
        Initialize ParEGO runner.

        Args:
            seed: Random seed for reproducibility
            n_evals: Total number of evaluations (including random init)
            variant: "baseline" or "matlab_reference"
            n_random_init: Number of random initialization points
            param_set: Battery parameter set ("Chen2020", "Ecker2015", "ORegan2022")
        """
        self.seed = seed
        self.n_evals = n_evals
        self.n_random_init = n_random_init
        self.variant = variant
        self.param_set = param_set

        if variant not in self.VARIANT_PRESETS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(self.VARIANT_PRESETS.keys())}")

        self.preset_name = self.VARIANT_PRESETS[variant]
        self.db: Optional[ObservationDB] = None
        self.optimizer: Optional[BayesOptimizer] = None
        self._result_summary: Optional[Dict] = None

    def run(self) -> ObservationDB:
        """Run ParEGO optimization."""
        logger.info("=" * 60)
        logger.info("ParEGO Runner: seed=%d, n_evals=%d, variant=%s", self.seed, self.n_evals, self.variant)
        logger.info("=" * 60)

        # Build configuration from preset
        cfg = self._build_config()
        self.optimizer = BayesOptimizer(config=cfg)

        # Run optimization
        self.db = self.optimizer.run()

        logger.info("ParEGO done: %d evals, %d feasible, %d Pareto, HV=%.4f",
                    self.db.size, self.db.n_feasible, self.db.pareto_size,
                    self.db.compute_hypervolume())
        return self.db

    def _build_config(self) -> Dict[str, Any]:
        """Build optimizer configuration from preset."""
        n_iterations = self.n_evals - self.n_random_init

        base_config = {
            "experiment_preset": self.preset_name,
            "max_iterations": n_iterations,
            "n_warmstart": 0,
            "n_random_init": self.n_random_init,
            "n_candidates": 1,
            "n_select": 1,
            "llm_backend": "mock",
            "llm_model": "gpt-4.1-mini",
            "llm_api_base": "https://api.nuwaapi.com/v1",
            "llm_api_key": "",
            "llm_n_samples": 1,
            "llm_temperature": 0.7,
            "battery_param_set": self.param_set,
            "warmstart_context_level": "full",
            "warmstart_max_tokens": 2500,
            "warmstart_max_retries": 3,
            "warmstart_temperature": None,
            "w_sample_seed": self.seed,
            "init_seed": self.seed,
            "checkpoint_dir": str(Path.cwd() / "checkpoints"),
            "checkpoint_every": 9999,
            "enable_iterative_guidance": False,
            "enable_gp_llm_coupling": False,
            "enable_acq_prior_coupling": False,
            "enable_proposal_sampler": False,
            "enable_llm_rerank": False,
            "enable_region_lifted_gp": False,
        }

        return base_config

    def save_results(self, output_dir: str) -> Dict:
        """Save experiment results in compatible format."""
        if self.db is None:
            raise RuntimeError("Must call run() before save_results()")

        os.makedirs(output_dir, exist_ok=True)

        display_hv = self.db.compute_hypervolume()
        raw_hv = self.db.compute_hypervolume_raw()
        canonical_hv = raw_hv / self.db.hv_max if self.db.hv_max > 1e-12 else 0.0

        # Build hv_trace from database observations
        hv_trace = []
        eval_count = 0
        for obs in self.db.get_all():
            eval_count += 1
            hv_trace.append({
                "eval_index": eval_count,
                "phase": "init" if obs.source != "bo" else "bo",
                "iteration": obs.iteration,
                "source": obs.source,
                "theta": obs.theta.tolist() if hasattr(obs.theta, 'tolist') else list(obs.theta),
                "feasible": obs.feasible,
                "hypervolume": display_hv,
                "display_hv": display_hv,
                "canonical_hv": canonical_hv,
                "hypervolume_raw": raw_hv,
                "pareto_size": self.db.pareto_size,
                "n_total": self.db.size,
                "n_feasible": self.db.n_feasible,
                "elapsed_s": 0.0,
            })

        summary = {
            "algorithm": f"parego_{self.variant}",
            "seed": self.seed,
            "n_evals": self.n_evals,
            "n_random_init": self.n_random_init,
            "variant": self.variant,
            "preset": self.preset_name,
            "param_set": self.param_set,
            "n_total": self.db.size,
            "n_feasible": self.db.n_feasible,
            "pareto_size": self.db.pareto_size,
            "hypervolume": display_hv,
            "display_hv": display_hv,
            "canonical_hv": canonical_hv,
            "hypervolume_raw": raw_hv,
            "hv_trace": hv_trace,
            "best_per_objective": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Best per objective
        best = self.db.get_best_per_objective()
        from DataBase.database import OBJECTIVE_NAMES
        for name in OBJECTIVE_NAMES:
            if name in best:
                o = best[name]
                idx = list(OBJECTIVE_NAMES).index(name)
                summary["best_per_objective"][name] = {
                    "value": float(o.objectives[idx]),
                    "theta": o.theta.tolist() if hasattr(o.theta, 'tolist') else list(o.theta),
                }

        # Save summary.json
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save database.json
        self.db.save(os.path.join(output_dir, "database.json"))

        # Save pareto_front.json
        pareto = self.db.get_pareto_front()
        pf_data = []
        for o in pareto:
            pf_data.append({
                "theta": o.theta.tolist() if hasattr(o.theta, 'tolist') else list(o.theta),
                "objectives": o.objectives.tolist() if hasattr(o.objectives, 'tolist') else list(o.objectives),
            })
        with open(os.path.join(output_dir, "pareto_front.json"), "w") as f:
            json.dump(pf_data, f, indent=2)

        self._result_summary = summary
        logger.info("Results saved to %s", output_dir)
        return summary

    def get_summary(self) -> Optional[Dict]:
        """Get the result summary after run()."""
        return self._result_summary
