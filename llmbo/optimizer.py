from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from config.presets import EXPERIMENT_PRESETS
from DataBase.database import DEFAULT_BOUNDS, ObservationDB
from llm.llm_interface import IterationGuidance, build_llm_interface
from llmbo.acquisition import (
    AcquisitionPrior,
    build_acquisition_function,
    build_ei_candidate_pool,
    expected_improvement,
    select_topm_for_rerank,
)
from llmbo.constraint_policy import build_constraint_policy
from llmbo.gp_model import build_gp_stack
from llmbo.proposal import ProposalTrainingRecord, build_proposal_sampler
from llmbo.region_lifted_gp import (
    LGBORegionLiftBuildResult,
    LLMRegionPreference,
    RegionLiftConfig,
    build_lgbo_region_lift,
    evaluate_region_lift_on_pool,
    is_lgbo_region_lift_mode,
    parse_region_preference_payload,
    sample_region_candidates,
)
from llmbo.rerank import (
    RerankState,
    TrialTelemetry,
    rerank_topm_with_llm,
)
from llmbo.scalarization import (
    canonical_hv_from_raw,
    canonicalize_objective_preprocess_mode,
    compute_objective_preprocess_context,
    compute_parego_reference_from_raw,
    compute_tchebycheff_from_raw_with_ideal,
    log_transform_objectives,
)
from pybamm_simulator import PyBaMMSimulator
from utils.constants import (
    DSOC_SUM_MAX as CANONICAL_DSOC_SUM_MAX,
    IDEAL_POINT,
    LLM_SAFE_DSOC_SUM_MAX,
    REF_POINT,
)
from utils.model_labels import canonical_model_label

logger = logging.getLogger(__name__)

PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]
DSOC_SUM_MAX = CANONICAL_DSOC_SUM_MAX


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


DEFAULT_CONFIG = {
    "experiment_preset": None,
    "max_iterations": 20,
    "n_warmstart": 10,
    "n_random_init": 3,
    "init_strategy": "manual",
    "init_budget": None,
    "warmstart_ratio": 0.5,
    "fixed_init_points": None,
    "fixed_init_source": "shared_init",
    "n_candidates": 15,
    "n_select": 1,
    "warmstart_batch_size": 10,
    "warmstart_max_attempts": 4,
    "warmstart_hv_log_interval": 5,
    "enable_warmstart_portfolio": True,
    "warmstart_pool_size": 16,
    "warmstart_diversity_weight": 0.45,
    "warmstart_soft_penalty_weight": 0.65,
    "warmstart_monotone_bonus": 0.08,
    "warmstart_archive_bonus_weight": 0.0,
    "warmstart_boundary_probe_limit": 1,
    "warmstart_cache_path": None,
    "warmstart_cache_mode": "read_write",
    "warmstart_cache_use_selected": False,
    "random_init_cache_path": None,
    "llm_backend": os.getenv("LLM_BACKEND", "openai"),
    "llm_model": os.getenv("LLM_MODEL", "gpt-4.1-mini"),
    "llm_api_base": os.getenv("LLM_API_BASE") or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.nuwaapi.com/v1"),
    "llm_api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
    "llm_n_samples": 3,
    "llm_temperature": 0.7,
    "llm_thinking_mode": None,
    "battery_param_set": "Chen2020",
    "warmstart_context_level": "full",
    "warmstart_prompt_version": None,
    "warmstart_max_tokens": 4096,
    "region_preference_max_tokens": 4096,
    "region_preference_prompt_version": "default",
    "warmstart_max_retries": 3,
    "warmstart_temperature": None,
    "kernel_nu": 2.5,
    "gp_alpha": 1e-6,
    "gp_normalize_y": True,
    "gp_n_restarts_optimizer": 5,
    "target_transform_mode": "none",
    "ei_n_restarts": 12,
    "ei_n_random_samples": 96,
    "ei_n_external_restarts": 16,
    "riesz_n_div": 10,
    "riesz_s": 2.0,
    "riesz_n_iter": 300,
    "riesz_lr": 5e-3,
    "riesz_seed": 42,
    "weight_strategy": "riesz_relaxed_cycle",
    "weight_simplex_divisions": 10,
    "weight_count": 30,
    "weight_eps_min": 0.01,
    "weight_sampling_mode": "cycle_without_replacement",
    "scalarization_mode": "log_ideal_gap",
    "objective_preprocess_mode": "minmax",
    "acquisition_strategy": "ei_lbfgsb",
    "parego_lcb_variance_weight": 0.5,
    "parego_de_population": 30,
    "parego_de_maxiter": 200,
    "parego_invert_weights": False,
    "parego_use_model_standardized_lcb": False,
    "w_sample_seed": 0,
    "init_seed": 2026,
    "eta": 0.05,
    "enable_iterative_guidance": True,
    "enable_gp_llm_coupling": False,
    "enable_acq_prior_coupling": True,
    "guidance_grid_size": 64,
    "guidance_point_grid_size": 25,
    "guidance_point_local_scale": 0.75,
    "guidance_probe_size": 128,
    "guidance_hotspots": 5,
    "guidance_top_scalar_k": 3,
    "lambda_max": 1.0,
    "lambda_min": 0.0,
    "lambda_decay_rate": 0.75,
    "coupling_history_similarity_threshold": 0.85,
    "coupling_history_fallback_score": 0.75,
    "llm_safe_dsoc_sum_max": LLM_SAFE_DSOC_SUM_MAX,
    "enable_proposal_sampler": False,
    "proposal_type": "weighted_gmm",
    "proposal_min_train_size": 8,
    "proposal_n_components": 3,
    "proposal_n_samples": 24,
    "proposal_local_mix": 0.30,
    "proposal_cov_floor": 1e-3,
    "proposal_elite_fraction": 0.35,
    "proposal_weight_epsilon": 1e-3,
    "proposal_near_constraint_lambda": 8.0,
    "proposal_monotone_penalty_lambda": 4.0,
    "proposal_safe_dsoc_sum_max": LLM_SAFE_DSOC_SUM_MAX,
    "proposal_enforce_monotone_profile": False,
    "guidance_prior_alpha": 0.30,
    "proposal_prior_alpha": 0.20,
    "proposal_prior_warmup_span": 8,
    "acq_risk_safe_weight": 0.20,
    "acq_risk_hard_weight": 3.00,
    "acq_risk_monotone_weight": 0.40,
    "enable_llm_rerank": False,
    "llm_rerank_mode": "none",
    "llm_rerank_top_m": 5,
    "llm_rerank_gamma_quantile": 0.20,
    "llm_rerank_eps": 1e-12,
    "llm_rerank_max_log_ei_gap": 0.20,
    "llm_rerank_gate": 0.10,
    "llm_rerank_max_bonus": 0.05,
    "llm_rerank_q_bad_threshold": 0.60,
    "llm_rerank_min_confidence": 0.50,
    "llm_rerank_parse_fail_open": True,
    "llm_rerank_score_mode": "unsafe_legacy_const_gate",
    "llm_rerank_const_gate": 0.25,
    "llm_rerank_gate_window": 5,
    "llm_rerank_min_ei": 1e-10,
    "llm_rerank_entropy_threshold": 0.80,
    "llm_rerank_fail_open_to_plain_ei": True,
    "enable_region_lifted_gp": False,
    "region_lift_mode": "heuristic_correlation",
    "region_lift_control_mode": "none",
    "region_lift_apply_override": False,
    "region_lift_override_uses_diagnostic_pool": False,
    "region_lift_external_influence_mode": "diagnostic_only",
    "region_lift_include_raw_candidates": True,
    "region_lift_lambda_max": 0.25,
    "region_lift_min_confidence": 0.60,
    "region_lift_n_anchors": 32,
    "region_lift_max_shift_std": 0.25,
    "region_lift_active_until": 12,
    "region_lift_anneal": "linear_decay",
    "region_lift_max_plain_ei_gap": 0.25,
    "region_lift_log_ei_eps": 1e-12,
    "region_lift_kernel_jitter": 1e-6,
    "region_lift_min_norm_sq": 1e-12,
    "region_lift_min_volume": 1e-5,
    "region_lift_max_volume": 0.25,
    "region_lift_min_width": 0.03,
    "region_lift_max_width": 0.80,
    "region_lift_close_distance": 0.05,
    "region_lift_max_close_fraction": 0.5,
    "region_lift_min_feasible_anchor_ratio": 0.6,
    "region_lift_near_region_tol": 0.05,
    "region_lift_trust_init": 0.5,
    "region_lift_trust_beta": 0.2,
    "region_lift_anchor_weighting": "ei_softmax",
    "region_lift_anchor_temperature": 0.35,
    "region_lift_require_inside": True,
    "region_lift_min_sigma_ratio": 0.85,
    "region_lift_candidate_oversample": 8,
    "region_lift_point_current_probe_levels": 0,
    "region_lift_point_current_probe_keep": 0,
    "region_lift_dsoc_margin": 0.02,
    "region_lift_guard_min_anchor_consistency": 0.35,
    "region_lift_guard_min_reliability": 0.20,
    "region_lift_guard_max_plain_ei_gap": 0.25,
    "region_lift_guard_require_inside": True,
    "region_lift_guard_require_positive_corr": True,
    "region_lift_lgbo_min_variance": 1e-12,
    "region_lift_lgbo_shift_source": "posterior_covariance",
    "region_lift_confidence_scale": 1.0,
    "region_lift_lgbo_shift_mean_budget": 0.0,
    "region_lift_lgbo_max_shift_std": 0.0,
    "region_lift_lgbo_apply_anneal": False,
    "region_lift_hard_confidence_gate": False,
    "region_lift_adaptive_confidence_enabled": False,
    "region_lift_adaptive_confidence_floor": 0.35,
    "region_lift_adaptive_base_scale": 0.85,
    "region_lift_adaptive_width_min_factor": 0.80,
    "region_lift_adaptive_repeat_min_factor": 0.85,
    "region_lift_adaptive_late_min_factor": 0.85,
    "region_lift_adaptive_width_start": 0.30,
    "region_lift_adaptive_repeat_distance": 0.18,
    "region_lift_random_width_norm": 0.15,
    "region_lift_random_confidence": 0.5,
    "checkpoint_dir": "checkpoints",
    "checkpoint_every": 5,
    "battery_model": "LG INR21700-M50 (Chen2020)",
    "soc_start": 0.0,
    "soc_end": 0.8,
    "dsoc_sum_max": DSOC_SUM_MAX,
}


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if np.allclose(v.sum(), 1.0) and np.all(v >= 0.0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = float(cssv[rho] - 1.0) / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def generate_das_dennis_weight_set(
    n_obj: int = 3,
    n_div: int = 10,
    eps_min: float = 0.01,
) -> np.ndarray:
    """Generate the evenly spread simplex weight set used by classic ParEGO."""

    def _das_dennis(divisions: int, dimensions: int) -> List[List[int]]:
        if dimensions == 1:
            return [[divisions]]
        points: List[List[int]] = []
        for i in range(divisions + 1):
            for rest in _das_dennis(divisions - i, dimensions - 1):
                points.append([i] + rest)
        return points

    W = np.array(_das_dennis(int(n_div), int(n_obj)), dtype=float)
    W = W / float(n_div)
    W = np.maximum(W, float(eps_min))
    return W / W.sum(axis=1, keepdims=True)


def generate_reference_parego_weight_set(
    n_obj: int = 3,
    n_weights: int = 30,
    seed: int = 42,
    eps_min: float = 0.01,
) -> np.ndarray:
    """Build a simple, evenly spread reference weight set for ParEGO."""
    target = max(int(n_weights), int(n_obj))
    n_div = 1
    base = generate_das_dennis_weight_set(n_obj=n_obj, n_div=n_div, eps_min=eps_min)
    while len(base) < target:
        n_div += 1
        base = generate_das_dennis_weight_set(n_obj=n_obj, n_div=n_div, eps_min=eps_min)

    if len(base) == target:
        return base

    rng = np.random.default_rng(seed)
    first_idx = int(rng.integers(0, len(base)))
    chosen = [first_idx]
    remaining = [idx for idx in range(len(base)) if idx != first_idx]

    while len(chosen) < target and remaining:
        chosen_points = base[chosen]
        best_idx = remaining[0]
        best_dist = -1.0
        for idx in remaining:
            dist = float(np.min(np.linalg.norm(base[idx][None, :] - chosen_points, axis=1)))
            if dist > best_dist:
                best_dist = dist
                best_idx = idx
        chosen.append(best_idx)
        remaining.remove(best_idx)

    return base[chosen]


def is_usable_simplex_weight_set(W: np.ndarray, n_obj: int = 3) -> bool:
    """Check that W is a valid simplex weight matrix (non-negative, rows sum to ~1)."""
    if not isinstance(W, np.ndarray) or W.ndim != 2 or W.shape[1] != n_obj:
        return False
    if np.any(W < -1e-6):
        return False
    row_sums = W.sum(axis=1)
    return bool(np.allclose(row_sums, 1.0, atol=1e-4))


def generate_riesz_weight_set(
    n_obj: int = 3,
    n_div: int = 10,
    s: float = 2.0,
    n_iter: int = 300,
    lr: float = 5e-3,
    seed: int = 42,
    eps_min: float = 0.01,
) -> np.ndarray:
    """
    Generate a Riesz-relaxed weight set on the probability simplex.

    This starts from a Das-Dennis grid and applies projected gradient steps on
    the Riesz energy to spread the weights more evenly across the simplex.
    """

    W = generate_das_dennis_weight_set(
        n_obj=n_obj,
        n_div=n_div,
        eps_min=eps_min,
    )

    for _ in range(int(n_iter)):
        grad = np.zeros_like(W)
        for i in range(len(W)):
            diff = W[i] - W
            dist2 = np.sum(diff ** 2, axis=1)
            dist2[i] = np.inf
            factor = float(s) / (dist2 ** ((float(s) + 2.0) / 2.0) + 1e-15)
            factor[i] = 0.0
            grad[i] = np.sum(factor[:, None] * diff, axis=0)

        W = W + float(lr) * grad
        for i in range(len(W)):
            W[i] = _project_to_simplex(W[i])
        W = np.maximum(W, eps_min)
        W = W / W.sum(axis=1, keepdims=True)

    return W


class BayesOptimizer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        user_cfg = dict(config or {})
        preset_name = user_cfg.get("experiment_preset") or user_cfg.get("preset")
        cfg = dict(DEFAULT_CONFIG)
        if preset_name:
            if str(preset_name) not in EXPERIMENT_PRESETS:
                available = ", ".join(sorted(EXPERIMENT_PRESETS))
                raise ValueError(f"Unknown experiment preset '{preset_name}'. Available: {available}")
            cfg.update(EXPERIMENT_PRESETS[str(preset_name)])
        cfg.update(user_cfg)
        cfg["experiment_preset"] = preset_name
        cfg["objective_preprocess_mode"] = canonicalize_objective_preprocess_mode(
            cfg.get("objective_preprocess_mode", "minmax")
        )
        self.cfg = cfg
        if "enable_gp_llm_coupling" not in user_cfg and "enable_region_lift" in user_cfg:
            self.cfg["enable_gp_llm_coupling"] = bool(user_cfg["enable_region_lift"])
        if "n_candidates" in user_cfg and self.cfg.get("n_candidates") and "ei_n_random_samples" not in user_cfg:
            self.cfg["ei_n_random_samples"] = max(64, int(self.cfg["n_candidates"]) * 8)

        seed = self.cfg.get("w_sample_seed")
        self._rng = np.random.default_rng(seed)
        self._weight_order: List[int] = []

        self.param_bounds = {k: tuple(v) for k, v in DEFAULT_BOUNDS.items()}
        self.simulator: Optional[PyBaMMSimulator] = None
        self.database: Optional[ObservationDB] = None
        self.llm: Any = None
        self.psi_fn: Any = None
        self.gp: Any = None
        self.af: Any = None
        self.proposal: Any = None
        self.constraint_policy = build_constraint_policy(self.cfg)
        self._weight_set: Optional[np.ndarray] = None
        self._warmstart_hv_trace: List[Dict[str, Any]] = []
        self._warmstart_portfolio_summary: Dict[str, Any] = {}
        self._hv_eval_trace: List[Dict[str, Any]] = []
        self._y_tilde_min = np.zeros(3, dtype=float)
        self._y_tilde_max = np.ones(3, dtype=float)
        self._previous_guidance: Optional[Dict[str, Any]] = None
        self._last_coupling_summary: Optional[Dict[str, Any]] = None
        self._last_proposal_summary: Optional[Dict[str, Any]] = None
        self._last_acq_prior_summary: Optional[Dict[str, Any]] = None
        self._last_rerank_summary: Optional[Dict[str, Any]] = None
        self._last_region_lift_summary: Optional[Dict[str, Any]] = None
        self._previous_region_thinking: Optional[str] = None
        self._last_region_adoption_note: Optional[Dict[str, Any]] = None
        self._last_candidate_source_counts: Dict[str, int] = {}
        self._rerank_telemetry: List[Dict[str, Any]] = []
        self._region_lift_telemetry: List[Dict[str, Any]] = []
        self._region_lift_trust: float = float(self.cfg.get("region_lift_trust_init", 0.5))
        self._region_influence_gate_open: bool = False

        Path(self.cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    def setup(self) -> None:
        logger.info("=" * 60)
        logger.info("Setting up 5D GP-LLM-coupled MOBO optimizer")
        logger.info("=" * 60)

        self.simulator = PyBaMMSimulator()
        self.param_bounds = {
            key: tuple(bounds) for key, bounds in getattr(self.simulator, "param_bounds", DEFAULT_BOUNDS).items()
        }

        self.database = ObservationDB(
            param_bounds=self.param_bounds,
            ref_point=REF_POINT.copy(),
            ideal_point=IDEAL_POINT.copy(),
            normalize=True,
        )

        backend = str(self.cfg["llm_backend"]).lower()
        api_key = str(self.cfg.get("llm_api_key") or "")
        if backend != "mock" and not api_key:
            logger.warning("No valid LLM API key found; falling back to mock warmstart backend")
            backend = "mock"
        self.cfg["llm_backend"] = backend

        self.llm = build_llm_interface(
            param_bounds=self.param_bounds,
            backend=backend,
            model=self.cfg["llm_model"],
            api_base=self.cfg["llm_api_base"],
            api_key=api_key,
            n_samples=self.cfg["llm_n_samples"],
            temperature=self.cfg["llm_temperature"],
            thinking_mode=self.cfg.get("llm_thinking_mode"),
            battery_model=getattr(self.simulator, "battery_name", self.cfg["battery_model"]),
            battery_param_set=str(
                self.cfg.get("battery_param_set", getattr(self.simulator, "param_set", "Chen2020"))
            ),
            warmstart_context_level=str(self.cfg.get("warmstart_context_level", "full")),
            warmstart_prompt_version=self.cfg.get("warmstart_prompt_version"),
            warmstart_max_tokens=int(self.cfg.get("warmstart_max_tokens", 4096)),
            region_preference_max_tokens=int(self.cfg.get("region_preference_max_tokens", 4096)),
            region_preference_prompt_version=str(self.cfg.get("region_preference_prompt_version", "default")),
            warmstart_max_retries=int(self.cfg.get("warmstart_max_retries", 3)),
            warmstart_temperature=self.cfg.get("warmstart_temperature"),
            soc_start=float(self.cfg.get("soc_start", getattr(self.simulator, "soc_start", 0.0))),
            soc_end=float(self.cfg.get("soc_end", getattr(self.simulator, "soc_end", 0.8))),
            dsoc_sum_max=float(self.cfg.get("dsoc_sum_max", getattr(self.simulator, "dsoc_sum_max", DSOC_SUM_MAX))),
            safe_dsoc_sum_max=float(self.cfg.get("llm_safe_dsoc_sum_max", LLM_SAFE_DSOC_SUM_MAX)),
            enable_warmstart_portfolio=bool(self.cfg.get("enable_warmstart_portfolio", True)),
            warmstart_pool_size=int(self.cfg.get("warmstart_pool_size", 16)),
            warmstart_diversity_weight=float(self.cfg.get("warmstart_diversity_weight", 0.45)),
            warmstart_soft_penalty_weight=float(self.cfg.get("warmstart_soft_penalty_weight", 0.65)),
            warmstart_monotone_bonus=float(self.cfg.get("warmstart_monotone_bonus", 0.08)),
            warmstart_archive_bonus_weight=float(self.cfg.get("warmstart_archive_bonus_weight", 0.0)),
            warmstart_boundary_probe_limit=int(self.cfg.get("warmstart_boundary_probe_limit", 1)),
            warmstart_cache_path=self.cfg.get("warmstart_cache_path"),
            warmstart_cache_mode=str(self.cfg.get("warmstart_cache_mode", "read_write")),
            warmstart_cache_use_selected=bool(self.cfg.get("warmstart_cache_use_selected", False)),
        )

        self.psi_fn, _, _, self.gp = build_gp_stack(
            param_bounds=self.param_bounds,
            kernel_nu=self.cfg["kernel_nu"],
            alpha=self.cfg["gp_alpha"],
            normalize_y=self.cfg["gp_normalize_y"],
            n_restarts_optimizer=self.cfg["gp_n_restarts_optimizer"],
            target_transform_mode=self.cfg.get("target_transform_mode", "none"),
            random_state=self.cfg.get("w_sample_seed"),
        )

        self.af = build_acquisition_function(
            gp=self.gp,
            psi_fn=self.psi_fn,
            param_bounds=self.param_bounds,
            n_select=self.cfg["n_select"],
            n_restarts_optimizer=self.cfg["ei_n_restarts"],
            n_random_candidates=self.cfg["ei_n_random_samples"],
            n_external_local_restarts=self.cfg.get("ei_n_external_restarts", 0),
            random_seed=self.cfg.get("w_sample_seed"),
            acquisition_strategy=self.cfg.get("acquisition_strategy", "ei_lbfgsb"),
            parego_lcb_variance_weight=self.cfg.get("parego_lcb_variance_weight", 0.5),
            parego_de_population=self.cfg.get("parego_de_population", 30),
            parego_de_maxiter=self.cfg.get("parego_de_maxiter", 200),
            parego_use_model_standardized_lcb=self.cfg.get("parego_use_model_standardized_lcb", False),
        )

        if bool(self.cfg.get("enable_proposal_sampler", False)):
            self.proposal = build_proposal_sampler(
                param_bounds=self.param_bounds,
                config=self.cfg,
                random_state=self.cfg.get("w_sample_seed"),
            )
            logger.info("Proposal sampler initialized: type=%s", self.cfg.get("proposal_type", "weighted_gmm"))
        else:
            self.proposal = None

        weight_strategy = str(self.cfg.get("weight_strategy", "riesz_relaxed_cycle")).lower()
        weight_eps_min = float(self.cfg.get("weight_eps_min", 0.01))
        if weight_strategy == "parego_reference_cycle":
            self._weight_set = generate_reference_parego_weight_set(
                n_obj=3,
                n_weights=int(self.cfg.get("weight_count", 30)),
                seed=int(self.cfg.get("riesz_seed", 42)),
                eps_min=weight_eps_min,
            )
            logger.info("Reference ParEGO weight set ready: shape=%s", self._weight_set.shape)
        elif weight_strategy == "parego_das_dennis_cycle":
            self._weight_set = generate_das_dennis_weight_set(
                n_obj=3,
                n_div=int(self.cfg.get("weight_simplex_divisions", 10)),
                eps_min=weight_eps_min,
            )
            logger.info("ParEGO Das-Dennis weight set ready: shape=%s", self._weight_set.shape)
        elif weight_strategy == "riesz_relaxed_cycle":
            from llmbo.riesz_cache import load_or_generate_riesz

            self._weight_set = load_or_generate_riesz(
                n_obj=3,
                n_div=self.cfg["riesz_n_div"],
                s=self.cfg["riesz_s"],
                n_iter=self.cfg["riesz_n_iter"],
                lr=self.cfg["riesz_lr"],
                seed=self.cfg["riesz_seed"],
            )
            logger.info("Riesz weight set ready: shape=%s", self._weight_set.shape)
        else:
            raise ValueError(f"Unsupported weight_strategy: {weight_strategy}")

    def run_initialization(self) -> None:
        n_warmstart, n_random_init = self._resolve_init_counts()
        logger.info("=" * 60)
        logger.info(
            "Initialization: strategy=%s warmstart=%d random_init=%d",
            self.cfg.get("init_strategy", "manual"),
            n_warmstart,
            n_random_init,
        )
        logger.info("=" * 60)

        scheduled: List[Tuple[str, np.ndarray]] = []
        fixed_init_points = self.cfg.get("fixed_init_points")

        if fixed_init_points is not None:
            fixed = np.atleast_2d(np.asarray(fixed_init_points, dtype=float))
            logger.info("Using %d fixed initialization points from config", fixed.shape[0])
            source = str(self.cfg.get("fixed_init_source", "shared_init"))
            scheduled.extend((source, self._repair_theta(theta)) for theta in fixed)
        else:
            if n_warmstart > 0:
                warmstart_points = self.llm.generate_warmstart_candidates(
                    n=n_warmstart,
                    batch_size=int(self.cfg["warmstart_batch_size"]),
                    max_attempts=int(self.cfg["warmstart_max_attempts"]),
                )
                if hasattr(self.llm, "get_warmstart_summary"):
                    try:
                        self._warmstart_portfolio_summary = self.llm.get_warmstart_summary()
                    except Exception:
                        self._warmstart_portfolio_summary = {}
                scheduled.extend(("llm_warmstart", theta) for theta in warmstart_points)

            if n_random_init > 0:
                random_points = self._get_random_init_points(
                    n_random_init,
                    seed=int(self.cfg.get("init_seed", self.cfg.get("w_sample_seed", 2026) or 2026)),
                )
                scheduled.extend(("random_init", theta) for theta in random_points)

        scheduled = self._deduplicate_tagged_points(scheduled)
        hv_trace: List[Dict[str, Any]] = []
        log_interval = max(1, int(self.cfg.get("warmstart_hv_log_interval", 5)))

        for i, (source, theta) in enumerate(scheduled, start=1):
            logger.info("Init [%d/%d] src=%s theta=%s", i, len(scheduled), source, np.round(theta, 4))
            t0 = time.perf_counter()
            result = self.simulator.evaluate(theta)
            elapsed = time.perf_counter() - t0
            self.database.add_from_simulator(
                theta=theta,
                result=result,
                source=source,
                iteration=0,
            )
            self._record_hv_snapshot(
                phase="init",
                iteration=0,
                source=source,
                theta=theta,
                feasible=bool(result["feasible"]),
                elapsed_s=elapsed,
            )
            logger.info(
                "  -> feasible=%s obj=%s (%.1fs)",
                result["feasible"],
                np.round(result["raw_objectives"], 6),
                elapsed,
            )

            if i % log_interval == 0 or i == len(scheduled):
                hv_raw = self.database.compute_hypervolume_raw()
                hv_canonical = canonical_hv_from_raw(hv_raw, self.database.hv_max)
                hv_display = self.database.compute_hypervolume()
                hv_trace.append(
                    {
                        "n_evaluated": i,
                        "hypervolume": hv_display,
                        "display_hv": hv_display,
                        "canonical_hv": hv_canonical,
                        "hypervolume_canonical": hv_canonical,
                        "hypervolume_raw": hv_raw,
                        "pareto_size": self.database.pareto_size,
                    }
                )

        self._warmstart_hv_trace = hv_trace
        if self.database.n_feasible == 0:
            raise RuntimeError("Initialization produced no feasible observations")

    def _resolve_init_counts(self) -> Tuple[int, int]:
        strategy = str(self.cfg.get("init_strategy", "manual")).lower()
        if strategy == "manual":
            return int(self.cfg.get("n_warmstart", 0)), int(self.cfg.get("n_random_init", 0))

        budget = self.cfg.get("init_budget")
        if budget is None:
            budget = int(self.cfg.get("n_warmstart", 0)) + int(self.cfg.get("n_random_init", 0))
        budget = max(0, int(budget))

        if strategy == "warmstart_only":
            return budget, 0
        if strategy == "random_only":
            return 0, budget
        if strategy == "mixed":
            ratio = float(self.cfg.get("warmstart_ratio", 0.5))
            ratio = min(max(ratio, 0.0), 1.0)
            n_warmstart = int(round(budget * ratio))
            n_warmstart = min(max(n_warmstart, 0), budget)
            return n_warmstart, budget - n_warmstart

        raise ValueError(f"Unsupported init_strategy: {strategy}")

    def initialize_acquisition(self) -> None:
        self._update_dynamic_bounds()
        w_init = np.full(3, 1.0 / 3.0, dtype=float)
        _, Y_raw = self.database.get_train_XY(feasible_only=True, normalize_X=False, normalize_Y=False)
        ideal_point_raw = self._compute_dynamic_ideal_point(Y_raw) if Y_raw.size else None
        self.database.update_tchebycheff_context(
            w_vec=w_init,
            y_min=self._y_tilde_min,
            y_max=self._y_tilde_max,
            ideal_point_raw=ideal_point_raw,
            eta=float(self.cfg["eta"]),
            scalarization_mode=str(self.cfg.get("scalarization_mode", "log_ideal_gap")),
            objective_preprocess_mode=str(self.cfg.get("objective_preprocess_mode", "minmax")),
            parego_invert_weights=bool(self.cfg.get("parego_invert_weights", False)),
        )
        self.af.initialize(self.database, llm_prior=self.llm)

    def run_optimization_loop(self) -> None:
        logger.info("=" * 60)
        logger.info("Optimization loop: %d iterations", self.cfg["max_iterations"])
        logger.info("=" * 60)
        self._region_influence_gate_open = False

        for t in range(int(self.cfg["max_iterations"])):
            iter_start = time.perf_counter()
            logger.info("--- Iteration %d ---", t)

            if self.database.n_feasible < 2:
                logger.warning("Not enough feasible points for GP; adding bootstrap LHS point")
                theta = self._lhs_candidates(1, seed=1000 + t)[0]
                result = self.simulator.evaluate(theta)
                self.database.add_from_simulator(theta=theta, result=result, source="bootstrap", iteration=t + 1)
                self._record_hv_snapshot(
                    phase="bo",
                    iteration=t + 1,
                    source="bootstrap",
                    theta=theta,
                    feasible=bool(result["feasible"]),
                )
                self.database.record_iteration_stats(extra={"t": t, "w_vec": None, "n_new_evals": 1})
                continue

            w_vec = self._next_weight()
            self._update_dynamic_bounds()
            X_train, Y_raw = self.database.get_train_XY(feasible_only=True, normalize_X=False, normalize_Y=False)
            ideal_point_raw = self._compute_dynamic_ideal_point(Y_raw)
            self.database.update_tchebycheff_context(
                w_vec=w_vec,
                y_min=self._y_tilde_min,
                y_max=self._y_tilde_max,
                ideal_point_raw=ideal_point_raw,
                eta=float(self.cfg["eta"]),
                scalarization_mode=str(self.cfg.get("scalarization_mode", "log_ideal_gap")),
                objective_preprocess_mode=str(self.cfg.get("objective_preprocess_mode", "minmax")),
                parego_invert_weights=bool(self.cfg.get("parego_invert_weights", False)),
            )
            scalar_y = self._compute_scalarized_targets(
                Y_raw=Y_raw,
                w_vec=w_vec,
                ideal_point_raw=ideal_point_raw,
            )
            self.gp.fit(X_train, scalar_y, w_vec=w_vec, t=t)

            guidance = None
            coupling = None
            acq_prior = None
            X_candidates = None
            proposal_candidates = np.empty((0, len(PARAM_KEYS)), dtype=float)
            region_acquisition_candidates = np.empty((0, len(PARAM_KEYS)), dtype=float)
            region_restart_candidates = np.empty((0, len(PARAM_KEYS)), dtype=float)
            diagnostic_region_candidates = np.empty((0, len(PARAM_KEYS)), dtype=float)
            hotspot_candidates = np.empty((0, len(PARAM_KEYS)), dtype=float)
            uncertainty_hotspots: List[Dict[str, Any]] = []
            region_preference: Optional[LLMRegionPreference] = None
            region_acquisition_lift = None
            region_lift_pre_acq_summary: Optional[Dict[str, Any]] = None
            region_pool_influenced_acquisition = False
            region_influence_mode = self._region_influence_mode()
            self._previous_guidance = None
            self._last_coupling_summary = None
            self._last_acq_prior_summary = None
            self._last_rerank_summary = None
            self._last_region_lift_summary = None
            self._last_candidate_source_counts = {}
            guidance_candidates = None

            proposal_records = self._build_proposal_training_records(scalar_y=scalar_y)
            self._last_proposal_summary = None
            if self.proposal is not None:
                self._last_proposal_summary = self.proposal.fit(proposal_records)
                proposal_candidates = self._sample_proposal_candidates(theta_best=self.database.get_theta_best())

            if bool(self.cfg.get("enable_region_lifted_gp", False)):
                if self._region_lift_window_active(t):
                    if self._is_fixed_random_region_lift_control():
                        region_preference = self._query_region_preference_random_fallback(t=t)
                    else:
                        region_preference = self._query_region_preference(
                            t=t,
                            w_vec=w_vec,
                            scalar_y=scalar_y,
                            ideal_point_raw=ideal_point_raw,
                        )
                    if self._is_lgbo_region_lift_mode():
                        lgbo_build = self._build_lgbo_acquisition_lift(
                            preference=region_preference,
                            t=t,
                        )
                        region_acquisition_lift = lgbo_build.coupling
                        region_lift_pre_acq_summary = dict(lgbo_build.telemetry)
                    else:
                        diagnostic_region_candidates = self._sample_region_candidates_from_preference(
                            preference=region_preference,
                            t=t,
                        )
                        region_pool_influenced_acquisition = self._should_influence_acquisition_with_region(t=t)
                        if region_pool_influenced_acquisition:
                            if bool(self.cfg.get("region_lift_include_raw_candidates", True)):
                                region_acquisition_candidates = diagnostic_region_candidates.copy()
                            else:
                                region_restart_candidates = diagnostic_region_candidates.copy()
                else:
                    region_preference = LLMRegionPreference.none("inactive_window_skipped")

            if bool(self.cfg.get("enable_iterative_guidance", True)):
                uncertainty_hotspots = self._compute_uncertainty_hotspots(t)
                guidance_state = self._build_guidance_state(
                    t=t,
                    w_vec=w_vec,
                    scalar_y=scalar_y,
                    ideal_point_raw=ideal_point_raw,
                    proposal_summary=self._last_proposal_summary,
                    uncertainty_hotspots=uncertainty_hotspots,
                )
                guidance = self.llm.query_iteration_guidance(guidance_state)

                if guidance is not None:
                    self._previous_guidance = guidance.to_dict()
                    guidance_payload_candidates = self._build_gp_llm_coupling_from_guidance(
                        guidance,
                        t,
                        w_vec=w_vec,
                    )
                    if bool(self.cfg.get("enable_gp_llm_coupling", True)):
                        coupling, guidance_candidates = guidance_payload_candidates
                        self._last_coupling_summary = coupling.to_dict()
                        logger.info(
                            "  Guidance mode=%s confidence=%.3f coupling_lambda=%.6f gate=%.3f effective=%.6f",
                            guidance.mode,
                            guidance.confidence,
                            float(coupling.lambda_value),
                            float(coupling.gate),
                            float(coupling.strength),
                        )
                    else:
                        _, guidance_candidates = guidance_payload_candidates
                        logger.info(
                            "  Guidance mode=%s confidence=%.3f GP-LLM coupling=disabled",
                            guidance.mode,
                            guidance.confidence,
                        )

                hotspot_candidates = np.array(
                    [hotspot["theta"] for hotspot in uncertainty_hotspots],
                    dtype=float,
                ) if uncertainty_hotspots else np.empty((0, len(PARAM_KEYS)))

            tagged_candidates: List[Tuple[str, np.ndarray]] = []
            if proposal_candidates.size:
                tagged_candidates.extend(("proposal", row) for row in proposal_candidates)
            if region_acquisition_candidates.size:
                tagged_candidates.extend(("region", row) for row in region_acquisition_candidates)
            if guidance_candidates is not None:
                tagged_candidates.extend(("guidance", row) for row in guidance_candidates)
            if hotspot_candidates.size:
                tagged_candidates.extend(("hotspot", row) for row in hotspot_candidates)
            if tagged_candidates:
                tagged_candidates = self._deduplicate_tagged_points(tagged_candidates)
                self._last_candidate_source_counts = self._summarize_tagged_points(tagged_candidates)
                X_candidates = np.vstack([theta for _, theta in tagged_candidates])

            if bool(self.cfg.get("enable_acq_prior_coupling", True)):
                acq_prior = self._build_acquisition_prior(
                    t=t,
                    guidance=guidance,
                    guidance_candidates=guidance_candidates,
                    proposal_candidates=proposal_candidates,
                )
                self._last_acq_prior_summary = None if acq_prior is None else acq_prior.to_dict()

            active_acquisition_lift = (
                region_acquisition_lift
                if region_acquisition_lift is not None
                else (None if bool(self.cfg.get("enable_region_lifted_gp", False)) else coupling)
            )
            acq_result = self.af.step(
                X_candidates=X_candidates,
                X_external_restarts=region_restart_candidates if region_restart_candidates.size else None,
                database=self.database,
                t=t,
                w_vec=w_vec,
                lift=active_acquisition_lift,
                prior=acq_prior,
            )
            debug_plain_indices = list(acq_result.debug.get("plain_selected_indices_without_lift", []))
            plain_selected_indices = (
                [int(idx) for idx in debug_plain_indices]
                if debug_plain_indices else list(acq_result.selected_indices)
            )
            if acq_result.all_alpha_base is not None and plain_selected_indices:
                base_scores = np.asarray(acq_result.all_alpha_base, dtype=float).ravel()
                plain_selected_scores = np.asarray(
                    [base_scores[idx] for idx in plain_selected_indices],
                    dtype=float,
                )
            else:
                plain_selected_scores = np.asarray(acq_result.selected_scores, dtype=float).copy()
            acq_result = self._maybe_apply_region_lifted_gp(
                t=t,
                w_vec=w_vec,
                scalar_y=scalar_y,
                ideal_point_raw=ideal_point_raw,
                acq_result=acq_result,
                plain_selected_indices=plain_selected_indices,
                plain_selected_scores=plain_selected_scores,
                preference=region_preference,
                diagnostic_region_candidates=diagnostic_region_candidates,
                region_pool_influenced_acquisition=region_pool_influenced_acquisition,
                region_influence_mode=region_influence_mode,
                pre_acquisition_summary=region_lift_pre_acq_summary,
                acquisition_lift=region_acquisition_lift,
            )
            self._update_region_influence_gate_from_summary()
            rerank_input_indices = list(acq_result.selected_indices)
            acq_result = self._maybe_apply_llm_rerank(
                t=t,
                w_vec=w_vec,
                scalar_y=scalar_y,
                acq_result=acq_result,
                plain_selected_indices=plain_selected_indices,
                plain_selected_scores=plain_selected_scores,
            )

            n_new = 0
            guidance_payload = (
                json.dumps(guidance.to_dict(), ensure_ascii=False)
                if guidance is not None else None
            )
            rerank_selected_map: Dict[int, Dict[str, Any]] = {}
            topm_candidate_map: Dict[int, Dict[str, Any]] = {}
            if self._last_rerank_summary and self._last_rerank_summary.get("rows"):
                rerank_selected_map = {
                    int(row["idx"]): row
                    for row in self._last_rerank_summary.get("rows", [])
                }
            if self._last_rerank_summary and self._last_rerank_summary.get("topm_candidates"):
                topm_candidate_map = {
                    int(row["idx"]): row
                    for row in self._last_rerank_summary.get("topm_candidates", [])
                    if isinstance(row, dict) and "idx" in row
                }
            for rank, theta in enumerate(acq_result.selected_thetas):
                logger.info("  Evaluate rank=%d theta=%s", rank, np.round(theta, 4))
                hv_before_raw = float(self.database.compute_hypervolume_raw())
                t_eval = time.perf_counter()
                sim_result = self.simulator.evaluate(theta)
                elapsed_eval = time.perf_counter() - t_eval
                selected_idx = int(acq_result.selected_indices[rank])
                plain_idx = int(plain_selected_indices[min(rank, len(plain_selected_indices) - 1)]) if plain_selected_indices else selected_idx
                before_rerank_idx = (
                    int(rerank_input_indices[min(rank, len(rerank_input_indices) - 1)])
                    if rerank_input_indices else selected_idx
                )
                plain_score = float(plain_selected_scores[min(rank, len(plain_selected_scores) - 1)]) if len(plain_selected_scores) else None
                rerank_row = rerank_selected_map.get(selected_idx, {})
                plain_row = rerank_selected_map.get(plain_idx, topm_candidate_map.get(plain_idx, {}))
                rerank_candidate = topm_candidate_map.get(selected_idx, {})
                if str(self.cfg.get("acquisition_strategy", "ei_lbfgsb")).lower() == "parego_lcb_de":
                    acq_type = "LCB_de"
                else:
                    acq_type = (
                        "EI_region_lifted_gp"
                        if self._last_region_lift_summary is not None and self._last_region_lift_summary.get("accepted", False)
                        else (
                            "EI_llm_rerank"
                            if self._last_rerank_summary is not None and self._last_rerank_summary.get("applied", False)
                            else (
                                "EI_gp_llm_coupled"
                                if coupling is not None
                                else ("EI_prior" if acq_prior is not None and acq_prior.is_active() else "EI")
                            )
                        )
                    )
                self.database.add_from_simulator(
                    theta=theta,
                    result=sim_result,
                    source="bo",
                    iteration=t + 1,
                    acq_value=float(acq_result.selected_scores[rank]),
                    acq_type=acq_type,
                    gp_pred={
                        "mean_coupled": float(acq_result.all_mean[acq_result.selected_indices[rank]]),
                        "mean_base": float(acq_result.all_mean_base[acq_result.selected_indices[rank]]),
                        "std": float(acq_result.all_std[acq_result.selected_indices[rank]]),
                        "coupling_lambda": (
                            float(active_acquisition_lift.lambda_value)
                            if active_acquisition_lift is not None else 0.0
                        ),
                        "coupling_mode": (
                            active_acquisition_lift.mode
                            if active_acquisition_lift is not None
                            else (guidance.mode if guidance is not None else None)
                        ),
                        "prior_bonus": (
                            float(acq_result.all_prior_bonus[acq_result.selected_indices[rank]])
                            if acq_result.all_prior_bonus is not None else 0.0
                        ),
                        "risk_penalty": (
                            float(acq_result.all_risk_penalty[acq_result.selected_indices[rank]])
                            if acq_result.all_risk_penalty is not None else 0.0
                        ),
                        "rerank_q_good": float(rerank_row.get("q_good", 0.0)) if rerank_row else None,
                        "rerank_gate": (
                            float(self._last_rerank_summary.get("gate", 0.0))
                            if self._last_rerank_summary else 0.0
                        ),
                        "region_lift_selected_source": (
                            None if not self._last_region_lift_summary
                            else self._last_region_lift_summary.get("selected_source")
                        ),
                        "region_lift_lambda_t": (
                            None if not self._last_region_lift_summary
                            else self._last_region_lift_summary.get("lambda_t")
                        ),
                        "region_lift_max_shift_z": (
                            None if not self._last_region_lift_summary
                            else self._last_region_lift_summary.get("max_shift_z")
                        ),
                    },
                    llm_rationale=guidance_payload,
                )
                self._record_hv_snapshot(
                    phase="bo",
                    iteration=t + 1,
                    source="bo",
                    theta=theta,
                    feasible=bool(sim_result["feasible"]),
                    elapsed_s=elapsed_eval,
                    acq_value=float(acq_result.selected_scores[rank]),
                )
                logger.info(
                    "    -> feasible=%s obj=%s acq=%.6f (%.1fs)",
                    sim_result["feasible"],
                    np.round(sim_result["raw_objectives"], 6),
                    float(acq_result.selected_scores[rank]),
                    elapsed_eval,
                )
                hv_after_raw = float(self.database.compute_hypervolume_raw())
                telemetry = TrialTelemetry(
                    iter_id=int(t),
                    w_vec=np.asarray(w_vec, dtype=float).tolist(),
                    tau_t=float(self._last_rerank_summary.get("tau_t", np.nan)) if self._last_rerank_summary else float("nan"),
                    selected_idx_before_rerank=int(before_rerank_idx),
                    selected_idx_after_rerank=int(selected_idx),
                    g_value=float(self._last_rerank_summary.get("gate", 0.0)) if self._last_rerank_summary else 0.0,
                    llm_called=bool(self._last_rerank_summary and self._last_rerank_summary.get("llm_called", False)),
                    llm_entropy_mean=(
                        None if not self._last_rerank_summary
                        else self._last_rerank_summary.get("entropy_mean")
                    ),
                    llm_q_selected=None if not rerank_row else float(rerank_row.get("q_good", 0.0)),
                    score_plain_selected=plain_score,
                    score_rerank_selected=float(acq_result.selected_scores[rank]),
                    hv_before=float(hv_before_raw),
                    hv_after=float(hv_after_raw),
                    hv_gain=float(hv_after_raw - hv_before_raw),
                    feasible=bool(sim_result["feasible"]),
                    plain_ei=None if not plain_row else plain_row.get("ei"),
                    rerank_ei=None if not rerank_candidate else rerank_candidate.get("ei"),
                    ei_ratio=(
                        None
                        if not plain_row or not rerank_candidate or float(plain_row.get("ei", 0.0)) <= 1e-12
                        else float(rerank_candidate.get("ei", 0.0)) / float(plain_row.get("ei", 1.0))
                    ),
                    log_ei_gap=None if not rerank_row else rerank_row.get("log_ei_gap_to_best"),
                    plain_mu=None if not plain_row else plain_row.get("mu_fw"),
                    plain_sigma=None if not plain_row else plain_row.get("sigma_fw"),
                    rerank_mu=None if not rerank_candidate else rerank_candidate.get("mu_fw"),
                    rerank_sigma=None if not rerank_candidate else rerank_candidate.get("sigma_fw"),
                    plain_rank_by_ei=None if not plain_row else plain_row.get("rank_by_ei"),
                    rerank_rank_by_ei=None if not rerank_candidate else rerank_candidate.get("rank_by_ei"),
                    llm_q_plain=None if not plain_row else plain_row.get("q_good"),
                    llm_q_rerank=None if not rerank_row else rerank_row.get("q_good"),
                    llm_conf_plain=None if not plain_row else plain_row.get("confidence"),
                    llm_conf_rerank=None if not rerank_row else rerank_row.get("confidence"),
                    selected_changed=bool(before_rerank_idx != selected_idx),
                    fallback_reason=None if not self._last_rerank_summary else self._last_rerank_summary.get("fallback_reason"),
                    rerank_mode=None if not self._last_rerank_summary else self._last_rerank_summary.get("rerank_mode"),
                )
                self._rerank_telemetry.append(telemetry.to_dict())
                if rank == 0 and self._last_region_lift_summary is not None:
                    self._last_region_lift_summary = {
                        **self._last_region_lift_summary,
                        "evaluated_theta": np.asarray(theta, dtype=float).tolist(),
                        "hv_before_raw": float(hv_before_raw),
                        "hv_after_raw": float(hv_after_raw),
                    }
                    self._finalize_region_lift_trust(hv_gain=float(hv_after_raw - hv_before_raw))
                n_new += 1

            iter_elapsed = time.perf_counter() - iter_start
            self.database.record_iteration_stats(
                extra={
                    "t": t,
                    "w_vec": w_vec.tolist(),
                    "n_new_evals": n_new,
                    "iter_time_s": round(iter_elapsed, 2),
                    "llm_guidance": self._previous_guidance,
                    "gp_llm_coupling": self._last_coupling_summary,
                    "acq_prior": self._last_acq_prior_summary,
                    "llm_rerank": self._last_rerank_summary,
                    "region_lifted_gp": self._last_region_lift_summary,
                    "proposal_summary": self._last_proposal_summary,
                    "candidate_source_counts": self._last_candidate_source_counts,
                }
            )
            logger.info(
                "Iteration %d complete: HV=%.6f |PF|=%d n=%d (%.1fs)",
                t,
                self.database.compute_hypervolume(),
                self.database.pareto_size,
                self.database.size,
                iter_elapsed,
            )

            if (t + 1) % int(self.cfg["checkpoint_every"]) == 0:
                self._save_checkpoint(t)

    def run(self) -> ObservationDB:
        self.setup()
        self.run_initialization()
        self.initialize_acquisition()
        self.run_optimization_loop()
        logger.info("Optimization finished: HV=%.6f", self.database.compute_hypervolume())
        return self.database

    def _maybe_apply_region_lifted_gp(
        self,
        *,
        t: int,
        w_vec: np.ndarray,
        scalar_y: np.ndarray,
        ideal_point_raw: np.ndarray,
        acq_result: Any,
        plain_selected_indices: List[int],
        plain_selected_scores: np.ndarray,
        preference: Optional[LLMRegionPreference],
        diagnostic_region_candidates: np.ndarray,
        region_pool_influenced_acquisition: bool,
        region_influence_mode: str,
        pre_acquisition_summary: Optional[Dict[str, Any]] = None,
        acquisition_lift: Optional[Any] = None,
    ) -> Any:
        if not bool(self.cfg.get("enable_region_lifted_gp", False)):
            return acq_result
        region_preference = preference or LLMRegionPreference.none("missing_preference")
        if self._is_lgbo_region_lift_mode():
            self._last_region_lift_summary = self._build_lgbo_post_acquisition_summary(
                t=t,
                acq_result=acq_result,
                plain_selected_indices=plain_selected_indices,
                plain_selected_scores=plain_selected_scores,
                preference=region_preference,
                pre_acquisition_summary=pre_acquisition_summary,
                acquisition_lift=acquisition_lift,
            )
            return acq_result

        override_enabled = bool(self.cfg.get("region_lift_apply_override", False))
        override_uses_diagnostic_pool = bool(
            self.cfg.get("region_lift_override_uses_diagnostic_pool", False)
        )
        plain_index_before = int(plain_selected_indices[0]) if plain_selected_indices else None
        summary_base = {
            "override_enabled": override_enabled,
            "override_uses_diagnostic_pool": override_uses_diagnostic_pool,
            "iteration": int(t),
            "preference": region_preference.to_dict(),
            "region_pool_influenced_acquisition": bool(region_pool_influenced_acquisition),
            "region_influence_mode": region_influence_mode,
            "region_influence_gate_passed": False,
            "inactive_window_skipped": False,
            "diagnostic_region_candidate_count": int(
                np.atleast_2d(np.asarray(diagnostic_region_candidates, dtype=float)).shape[0]
            ) if np.asarray(diagnostic_region_candidates).size else 0,
        }
        if not self._region_lift_window_active(t):
            self._last_region_lift_summary = {
                "active": False,
                "accepted": False,
                "selected_source": "plain_ei",
                "fallback_reason": "inactive_window_skipped",
                "selected_index_before": plain_index_before,
                "selected_index_after": plain_index_before,
                **summary_base,
                "inactive_window_skipped": True,
            }
            return acq_result
        if acq_result.candidate_pool is None or len(acq_result.candidate_pool) == 0:
            self._last_region_lift_summary = {
                "active": True,
                "accepted": False,
                "selected_source": "fallback",
                "fallback_reason": "empty_candidate_pool",
                "selected_index_before": plain_index_before,
                "selected_index_after": plain_index_before,
                **summary_base,
            }
            return acq_result

        config = RegionLiftConfig.from_config(self.cfg)
        existing_X = np.vstack([obs.theta for obs in self.database.get_all()]) if self.database.size else np.empty((0, len(PARAM_KEYS)))
        candidate_pool = np.atleast_2d(np.asarray(acq_result.candidate_pool, dtype=float))
        selected_theta_before = (
            np.asarray(acq_result.selected_thetas[0], dtype=float).copy()
            if acq_result.selected_thetas else candidate_pool[0].copy()
        )
        diagnostic_pool = candidate_pool
        selection_pool = candidate_pool
        plain_index_override = plain_index_before
        if (not override_enabled) or override_uses_diagnostic_pool:
            diagnostic_pool, plain_index_override = self._build_region_diagnostic_pool(
                candidate_pool=candidate_pool,
                plain_selected_theta=selected_theta_before,
                plain_selected_index=plain_index_before,
                diagnostic_region_candidates=diagnostic_region_candidates,
            )
            if override_enabled and override_uses_diagnostic_pool:
                selection_pool = diagnostic_pool
        result = evaluate_region_lift_on_pool(
            gp=self.gp,
            candidate_pool=diagnostic_pool,
            f_min_y=float(self.database.get_f_min()),
            preference=region_preference,
            existing_X=existing_X,
            bounds=self.param_bounds,
            config=config,
            trust=float(self._region_lift_trust),
            bo_iteration=int(t),
            plain_index_override=plain_index_override,
        )
        summary = dict(result.telemetry)
        summary.update(
            {
                "accepted": bool(result.accepted),
                "selected_index_before": plain_index_before,
                "selected_score_before": float(plain_selected_scores[0]) if len(plain_selected_scores) else None,
                "selected_index_after": int(result.selected_index),
                "diagnostic_candidate_pool_size": int(diagnostic_pool.shape[0]),
                "acquisition_candidate_pool_size": int(candidate_pool.shape[0]),
                "selection_candidate_pool_size": int(selection_pool.shape[0]),
                **summary_base,
            }
        )
        gate_passed = self._region_influence_gate_passes(summary)
        summary["region_influence_gate_passed"] = bool(gate_passed)
        if not override_enabled:
            summary["diagnostic_selected_source"] = str(summary.get("selected_source", result.selected_source))
            summary["diagnostic_fallback_reason"] = result.fallback_reason
            summary["diagnostic_override_candidate_available"] = bool(result.accepted)
            summary["accepted"] = False
            summary["selected_source"] = "plain_ei"
            summary["fallback_reason"] = "override_disabled" if bool(result.accepted) else (result.fallback_reason or "diagnostic_only")
            if plain_selected_indices:
                summary["selected_index_after"] = plain_index_before
        self._last_region_lift_summary = summary
        if override_enabled and result.accepted:
            idx = int(result.selected_index)
            if idx < 0 or idx >= len(selection_pool):
                summary["accepted"] = False
                summary["selected_source"] = "plain_ei"
                summary["fallback_reason"] = "override_index_out_of_range"
                self._last_region_lift_summary = summary
                return acq_result
            if len(selection_pool) != len(candidate_pool):
                self._align_acquisition_result_with_selection_pool(acq_result, selection_pool)
            acq_result.selected_indices = [idx]
            acq_result.selected_thetas = [selection_pool[idx].copy()]
            fallback_score = (
                float(acq_result.all_ei[idx])
                if idx < len(np.asarray(acq_result.all_ei).ravel())
                else float(plain_selected_scores[0]) if len(plain_selected_scores) else 0.0
            )
            score = float(summary.get("lifted_ei_at_lift", fallback_score))
            acq_result.selected_scores = np.asarray([score], dtype=float)
            acq_result.lift_summary = summary
        return acq_result

    def _is_lgbo_region_lift_mode(self) -> bool:
        return is_lgbo_region_lift_mode(RegionLiftConfig.from_config(self.cfg))

    def _is_fixed_random_region_lift_control(self) -> bool:
        return (
            self._is_lgbo_region_lift_mode()
            and str(self.cfg.get("region_lift_control_mode", "none") or "none").lower() == "fixed_random"
        )

    def _build_lgbo_acquisition_lift(
        self,
        *,
        preference: LLMRegionPreference,
        t: int,
    ) -> LGBORegionLiftBuildResult:
        result = build_lgbo_region_lift(
            gp=self.gp,
            preference=preference,
            bounds=self.param_bounds,
            config=RegionLiftConfig.from_config(self.cfg),
            bo_iteration=int(t),
        )
        result.telemetry.update(
            {
                "trust_before": float(self._region_lift_trust),
                "trust_after": float(self._region_lift_trust),
                "trust_update_reason": "pending",
                "preference": result.preference.to_dict(),
                "llm_called_for_region": not self._is_fixed_random_region_lift_control(),
            }
        )
        if isinstance(preference.raw_response, dict):
            adaptive = preference.raw_response.get("_adaptive_confidence")
            if isinstance(adaptive, dict):
                result.telemetry.update(adaptive)
        return result

    def _query_region_preference_random_fallback(self, *, t: int) -> LLMRegionPreference:
        lo = np.array([self.param_bounds[key][0] for key in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[key][1] for key in PARAM_KEYS], dtype=float)
        span = np.maximum(hi - lo, 1e-12)
        width_norm_raw = self.cfg.get("region_lift_random_width_norm", 0.15)
        if isinstance(width_norm_raw, (list, tuple, np.ndarray)):
            width_norm = np.asarray(width_norm_raw, dtype=float).ravel()
            if width_norm.size != len(PARAM_KEYS):
                width_norm = np.full(len(PARAM_KEYS), 0.15, dtype=float)
        else:
            width_norm = np.full(len(PARAM_KEYS), float(width_norm_raw), dtype=float)
        width_norm = np.clip(width_norm, 1e-6, 1.0)
        width = np.minimum(width_norm * span, span)
        center_min = lo + 0.5 * width
        center_max = hi - 0.5 * width
        seed_base = int(self.cfg.get("w_sample_seed") or 0)
        sampler = qmc.Sobol(d=len(PARAM_KEYS), scramble=True, seed=seed_base + int(t) + 104729)
        unit = sampler.random_base2(m=8)
        margin = max(float(self.cfg.get("region_lift_dsoc_margin", 0.0)), 0.0)
        safe_limit = float(DSOC_SUM_MAX) - margin
        chosen_lb = None
        chosen_ub = None
        for row in unit:
            center = center_min + row * (center_max - center_min)
            lb = center - 0.5 * width
            ub = center + 0.5 * width
            if ub[3] + ub[4] <= safe_limit + 1e-12:
                chosen_lb, chosen_ub = lb, ub
                break
        if chosen_lb is None or chosen_ub is None:
            center = 0.5 * (center_min + center_max)
            excess = max((center[3] + center[4] + 0.5 * (width[3] + width[4])) - safe_limit, 0.0)
            center[3:5] -= 0.5 * excess
            center = np.clip(center, center_min, center_max)
            chosen_lb = center - 0.5 * width
            chosen_ub = center + 0.5 * width
        payload = {
            "kind": "region",
            "coordinate_space": "raw",
            "preference_direction": "promising",
            "lb": {key: float(chosen_lb[idx]) for idx, key in enumerate(PARAM_KEYS)},
            "ub": {key: float(chosen_ub[idx]) for idx, key in enumerate(PARAM_KEYS)},
            "confidence": float(np.clip(self.cfg.get("region_lift_random_confidence", 0.5), 0.0, 1.0)),
            "preference_type": "random_control",
            "reason": "fixed random LGBO control region",
            "mechanistic_thinking": "Random control: no LLM mechanism is used.",
            "llm_called_for_region": False,
            "random_control_type": "fixed_random",
        }
        preference = parse_region_preference_payload(payload)
        preference.parser_status = "ok"
        return preference

    def _build_lgbo_post_acquisition_summary(
        self,
        *,
        t: int,
        acq_result: Any,
        plain_selected_indices: List[int],
        plain_selected_scores: np.ndarray,
        preference: LLMRegionPreference,
        pre_acquisition_summary: Optional[Dict[str, Any]],
        acquisition_lift: Optional[Any],
    ) -> Dict[str, Any]:
        debug_plain_indices = list(getattr(acq_result, "debug", {}).get("plain_selected_indices_without_lift", []))
        plain_index = (
            int(debug_plain_indices[0])
            if debug_plain_indices else int(plain_selected_indices[0]) if plain_selected_indices else None
        )
        lifted_index = (
            int(acq_result.selected_indices[0])
            if getattr(acq_result, "selected_indices", None) else plain_index
        )
        used_lift = acquisition_lift is not None
        summary = dict(pre_acquisition_summary or {})
        structural_fallback = summary.get("structural_fallback_reason") or summary.get("fallback_reason")

        plain_ei_at_plain = self._score_at_index(getattr(acq_result, "all_ei_base", None), plain_index, None)
        plain_ei_at_lifted = self._score_at_index(getattr(acq_result, "all_ei_base", None), lifted_index, None)
        lifted_ei_at_lifted = self._score_at_index(getattr(acq_result, "all_ei", None), lifted_index, None)
        eps = float(self.cfg.get("region_lift_log_ei_eps", 1e-12))
        plain_ei_gap = None
        if plain_ei_at_plain is not None and plain_ei_at_lifted is not None:
            plain_ei_gap = float(
                np.log(max(float(plain_ei_at_plain), eps))
                - np.log(max(float(plain_ei_at_lifted), eps))
            )

        candidate_pool = getattr(acq_result, "candidate_pool", None)
        lifted_theta = None
        if candidate_pool is not None and lifted_index is not None and 0 <= lifted_index < len(candidate_pool):
            lifted_theta = np.asarray(candidate_pool[lifted_index], dtype=float)
        inside = None
        local_lb = getattr(acquisition_lift, "local_lb", None) if used_lift else None
        local_ub = getattr(acquisition_lift, "local_ub", None) if used_lift else None
        if lifted_theta is not None and local_lb is not None and local_ub is not None:
            lo = np.array([self.param_bounds[key][0] for key in PARAM_KEYS], dtype=float)
            hi = np.array([self.param_bounds[key][1] for key in PARAM_KEYS], dtype=float)
            tol = max(float(self.cfg.get("region_lift_near_region_tol", 0.0)), 0.0) * (hi - lo)
            lb = np.asarray(local_lb, dtype=float).ravel()
            ub = np.asarray(local_ub, dtype=float).ravel()
            inside = bool(np.all(lifted_theta >= lb - tol) and np.all(lifted_theta <= ub + tol))

        sigma_ratio = None
        plain_sigma = self._score_at_index(getattr(acq_result, "all_std", None), plain_index, None)
        lifted_sigma = self._score_at_index(getattr(acq_result, "all_std", None), lifted_index, None)
        if plain_sigma is not None and lifted_sigma is not None:
            sigma_ratio = float(lifted_sigma / max(float(plain_sigma), 1e-12))

        shift_z_all = None
        selected_shift_z = None
        try:
            mean_base = np.asarray(acq_result.all_mean_base, dtype=float).ravel()
            mean_lifted = np.asarray(acq_result.all_mean, dtype=float).ravel()
            _, y_std = self.gp.target_standardization()
            shift_z_all = (mean_base - mean_lifted) / max(float(y_std), 1e-12)
            if lifted_index is not None and 0 <= int(lifted_index) < len(shift_z_all):
                selected_shift_z = float(shift_z_all[int(lifted_index)])
        except Exception:
            shift_z_all = None

        changed = bool(
            used_lift
            and plain_index is not None
            and lifted_index is not None
            and int(plain_index) != int(lifted_index)
        )
        guard_reason = None
        if changed:
            if plain_ei_at_plain is None or plain_ei_gap is None:
                guard_reason = "missing_plain_ei_counterfactual"
            elif float(plain_ei_at_plain) <= eps:
                guard_reason = "flat_plain_ei"
            elif plain_ei_gap > float(self.cfg.get("region_lift_max_plain_ei_gap", 0.25)):
                guard_reason = "plain_ei_gap"
            elif selected_shift_z is None:
                guard_reason = "missing_shift_counterfactual"
            elif selected_shift_z <= eps:
                guard_reason = "nonpositive_selected_shift"
            elif bool(self.cfg.get("region_lift_require_inside", True)) and inside is not True:
                guard_reason = "outside_region"
            elif sigma_ratio is None:
                guard_reason = "missing_sigma_counterfactual"
            elif sigma_ratio < float(self.cfg.get("region_lift_min_sigma_ratio", 0.85)):
                guard_reason = "low_sigma_ratio"

        if guard_reason is not None and plain_index is not None and candidate_pool is not None:
            acq_result.selected_indices = [int(plain_index)]
            acq_result.selected_thetas = [np.asarray(candidate_pool[plain_index], dtype=float).copy()]
            plain_score = self._score_at_index(
                getattr(acq_result, "all_alpha_base", None),
                plain_index,
                float(plain_selected_scores[0]) if len(plain_selected_scores) else 0.0,
            )
            acq_result.selected_scores = np.asarray([float(plain_score or 0.0)], dtype=float)

        selected_index_after = (
            int(acq_result.selected_indices[0])
            if getattr(acq_result, "selected_indices", None) else plain_index
        )
        accepted = bool(used_lift and guard_reason is None)
        summary.update(
            {
                "active": bool(self._region_lift_window_active(t)),
                "accepted": accepted,
                "selected_source": "lgbo_lifted_gp" if accepted else "plain_ei",
                "fallback_reason": (
                    None if accepted else guard_reason or structural_fallback or "missing_lgbo_lift"
                ),
                "structural_fallback_reason": structural_fallback,
                "acquisition_used_lift": bool(used_lift),
                "selection_guard_passed": bool(used_lift and guard_reason is None),
                "selection_guard_reason": guard_reason,
                "selected_index_before": plain_index,
                "selected_index_after": selected_index_after,
                "plain_selected_idx": plain_index,
                "lifted_selected_idx_before_guard": lifted_index,
                "selected_changed_by_lift": changed,
                "effective_selection_change": bool(accepted and changed),
                "selected_score_before": self._score_at_index(
                    getattr(acq_result, "all_alpha_base", None),
                    plain_index,
                    float(plain_selected_scores[0]) if len(plain_selected_scores) else None,
                ),
                "selected_score_after": (
                    float(acq_result.selected_scores[0])
                    if getattr(acq_result, "selected_scores", None) is not None
                    and len(acq_result.selected_scores) else None
                ),
                "iteration": int(t),
                "region_lift_mode": str(self.cfg.get("region_lift_mode", "heuristic_correlation")),
                "region_lift_control_mode": str(self.cfg.get("region_lift_control_mode", "none")),
                "region_lift_lgbo_shift_source": str(
                    getattr(acquisition_lift, "shift_source", self.cfg.get("region_lift_lgbo_shift_source", "prior_kernel"))
                ),
                "plain_ei_at_plain": plain_ei_at_plain,
                "plain_ei_at_lifted": plain_ei_at_lifted,
                "lifted_ei_at_lifted": lifted_ei_at_lifted,
                "plain_ei_gap": plain_ei_gap,
                "plain_ei_gap_exceeded": bool(
                    plain_ei_gap is not None
                    and plain_ei_gap > float(self.cfg.get("region_lift_max_plain_ei_gap", 0.25))
                ),
                "lift_candidate_inside_region": inside,
                "sigma_ratio_vs_plain": sigma_ratio,
                "selected_shift_z": selected_shift_z,
                "override_enabled": bool(self.cfg.get("region_lift_apply_override", False)),
                "override_uses_diagnostic_pool": bool(self.cfg.get("region_lift_override_uses_diagnostic_pool", False)),
                "region_pool_influenced_acquisition": False,
                "region_influence_mode": "diagnostic_only",
                "region_influence_gate_passed": False,
                "inactive_window_skipped": not bool(self._region_lift_window_active(t)),
                "diagnostic_region_candidate_count": 0,
                "acquisition_candidate_pool_size": int(len(candidate_pool)) if candidate_pool is not None else 0,
                "preference": preference.to_dict(),
            }
        )
        if used_lift:
            summary["anchor_weighting_mode"] = "uniform"
            summary["lgbo_covariance_source"] = "posterior_standardized"
            summary["lgbo_shift_kernel_source"] = (
                "posterior_standardized_cross_covariance"
                if str(getattr(acquisition_lift, "shift_source", "")).lower() == "posterior_covariance"
                else "prior_latent_standardized"
            )
            try:
                if shift_z_all is not None and len(shift_z_all):
                    summary.update(
                        {
                            "lgbo_shift_min": float(np.min(shift_z_all)),
                            "lgbo_shift_max": float(np.max(shift_z_all)),
                            "lgbo_shift_mean": float(np.mean(shift_z_all)),
                            "max_shift_z": float(np.max(np.abs(shift_z_all))),
                            "mean_shift_z": float(np.mean(shift_z_all)),
                        }
                    )
            except Exception:
                pass
        acq_result.lift_summary = summary
        return summary

    @staticmethod
    def _score_at_index(values: Any, index: Optional[int], default: Optional[float]) -> Optional[float]:
        if values is None or index is None:
            return default
        try:
            arr = np.asarray(values, dtype=float).ravel()
            idx = int(index)
            if idx < 0 or idx >= len(arr):
                return default
            return float(arr[idx])
        except Exception:
            return default

    def _align_acquisition_result_with_selection_pool(self, acq_result: Any, selection_pool: np.ndarray) -> None:
        pool = np.atleast_2d(np.asarray(selection_pool, dtype=float))
        try:
            mean_base, std = self.gp.predict(pool)
            mean_base = np.asarray(mean_base, dtype=float).ravel()
            std = np.asarray(std, dtype=float).ravel()
            f_min = self._model_target_value(float(self.database.get_f_min()))
            ei_base = expected_improvement(mean_base, std, f_min)
            acq_result.all_mean = mean_base.copy()
            acq_result.all_mean_base = mean_base.copy()
            acq_result.all_std = std
            acq_result.all_ei = ei_base.copy()
            acq_result.all_ei_base = ei_base.copy()
            acq_result.all_wcharge = np.ones(len(pool), dtype=float)
            acq_result.all_alpha = ei_base.copy()
            acq_result.all_alpha_base = ei_base.copy()
            acq_result.all_prior_bonus = np.zeros(len(pool), dtype=float)
            acq_result.all_risk_penalty = np.zeros(len(pool), dtype=float)
        except Exception as exc:
            logger.debug("Could not recompute expanded Region override pool: %s", exc)

            def _pad(values: Any, fill: float = 0.0) -> np.ndarray:
                arr = np.asarray([] if values is None else values, dtype=float).ravel()
                if len(arr) >= len(pool):
                    return arr[: len(pool)]
                return np.pad(arr, (0, len(pool) - len(arr)), constant_values=float(fill))

            acq_result.all_mean = _pad(getattr(acq_result, "all_mean", None))
            acq_result.all_mean_base = _pad(getattr(acq_result, "all_mean_base", None))
            acq_result.all_std = _pad(getattr(acq_result, "all_std", None), fill=1e-12)
            acq_result.all_ei = _pad(getattr(acq_result, "all_ei", None))
            acq_result.all_ei_base = _pad(getattr(acq_result, "all_ei_base", None))
            acq_result.all_wcharge = _pad(getattr(acq_result, "all_wcharge", None), fill=1.0)
            acq_result.all_alpha = _pad(getattr(acq_result, "all_alpha", None))
            acq_result.all_alpha_base = _pad(getattr(acq_result, "all_alpha_base", None))
            acq_result.all_prior_bonus = _pad(getattr(acq_result, "all_prior_bonus", None))
            acq_result.all_risk_penalty = _pad(getattr(acq_result, "all_risk_penalty", None))
        acq_result.candidate_pool = pool.copy()
        if isinstance(getattr(acq_result, "debug", None), dict):
            acq_result.debug["n_pool"] = int(len(pool))
            acq_result.debug["region_override_pool_expanded"] = True

    def _region_influence_mode(self) -> str:
        raw = str(self.cfg.get("region_lift_external_influence_mode", "diagnostic_only") or "diagnostic_only").lower()
        if raw in {"diagnostic_only", "guarded_pool", "force_pool"}:
            return raw
        return "diagnostic_only"

    def _region_lift_window_active(self, t: int) -> bool:
        active_until = max(int(self.cfg.get("region_lift_active_until", 0)), 0)
        return int(t) < active_until

    def _should_influence_acquisition_with_region(self, *, t: int) -> bool:
        if self._is_lgbo_region_lift_mode():
            return False
        if not self._region_lift_window_active(t):
            return False
        mode = self._region_influence_mode()
        if mode == "force_pool":
            return True
        if mode == "guarded_pool":
            return bool(self._region_influence_gate_open)
        return False

    def _update_region_influence_gate_from_summary(self) -> None:
        mode = self._region_influence_mode()
        if mode != "guarded_pool" or self._last_region_lift_summary is None:
            self._region_influence_gate_open = False
            return
        self._region_influence_gate_open = bool(self._last_region_lift_summary.get("region_influence_gate_passed", False))

    def _region_influence_gate_passes(self, summary: Dict[str, Any]) -> bool:
        if bool(summary.get("inactive_window_skipped", False)):
            return False
        if float(summary.get("lambda_t", 0.0) or 0.0) <= 0.0:
            return False
        if not bool(summary.get("diagnostic_override_candidate_available", summary.get("accepted", False))):
            return False
        gap = summary.get("plain_ei_gap")
        if gap is None or float(gap) > float(self.cfg.get("region_lift_guard_max_plain_ei_gap", 0.25)):
            return False
        if float(summary.get("anchor_consistency", 0.0) or 0.0) < float(
            self.cfg.get("region_lift_guard_min_anchor_consistency", 0.35)
        ):
            return False
        if float(summary.get("region_reliability", 0.0) or 0.0) < float(
            self.cfg.get("region_lift_guard_min_reliability", 0.20)
        ):
            return False
        if bool(self.cfg.get("region_lift_guard_require_inside", True)) and not bool(
            summary.get("lift_candidate_inside_region", False)
        ):
            return False
        if bool(self.cfg.get("region_lift_guard_require_positive_corr", True)) and float(
            summary.get("corr_at_lift", 0.0) or 0.0
        ) <= 0.0:
            return False
        return True

    def _build_region_diagnostic_pool(
        self,
        *,
        candidate_pool: np.ndarray,
        plain_selected_theta: np.ndarray,
        plain_selected_index: Optional[int],
        diagnostic_region_candidates: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[int]]:
        base = np.atleast_2d(np.asarray(candidate_pool, dtype=float))
        extras = (
            np.atleast_2d(np.asarray(diagnostic_region_candidates, dtype=float))
            if np.asarray(diagnostic_region_candidates).size else np.empty((0, len(PARAM_KEYS)), dtype=float)
        )
        if extras.size == 0:
            return base.copy(), plain_selected_index

        merged_points = [row.copy() for row in base]
        merged_points.extend(np.asarray(row, dtype=float).copy() for row in extras)
        merged = np.vstack(self._deduplicate_points(merged_points))
        plain_idx = self._find_point_index(merged, plain_selected_theta)
        if plain_idx is None:
            merged = np.vstack([np.asarray(plain_selected_theta, dtype=float), merged])
            plain_idx = 0
        return merged, plain_idx

    @staticmethod
    def _find_point_index(pool: np.ndarray, theta: np.ndarray, tol: float = 1e-9) -> Optional[int]:
        X = np.atleast_2d(np.asarray(pool, dtype=float))
        target = np.asarray(theta, dtype=float).ravel()
        for idx, row in enumerate(X):
            if np.allclose(row, target, atol=tol, rtol=0.0):
                return int(idx)
        return None

    def _query_region_preference(
        self,
        *,
        t: int,
        w_vec: np.ndarray,
        scalar_y: np.ndarray,
        ideal_point_raw: np.ndarray,
    ) -> LLMRegionPreference:
        state = self._build_region_preference_state(
            t=t,
            w_vec=w_vec,
            scalar_y=scalar_y,
            ideal_point_raw=ideal_point_raw,
        )
        try:
            preference = self.llm.query_region_preference(state)
            if preference is not None:
                preference = self._apply_adaptive_region_confidence(preference, t=t)
            return preference
        except Exception as exc:
            logger.warning("Region preference query failed: %s", exc)
            return LLMRegionPreference.none("query_exception")

    def _apply_adaptive_region_confidence(
        self,
        preference: LLMRegionPreference,
        *,
        t: int,
    ) -> LLMRegionPreference:
        if not bool(self.cfg.get("region_lift_adaptive_confidence_enabled", False)):
            return preference
        if preference.kind not in {"point", "region"}:
            return preference

        raw_confidence = float(np.clip(preference.confidence, 0.0, 1.0))
        base_scale = float(np.clip(self.cfg.get("region_lift_adaptive_base_scale", 0.85), 0.0, 1.0))
        floor = float(np.clip(self.cfg.get("region_lift_adaptive_confidence_floor", 0.35), 0.0, 1.0))
        width_factor = 1.0
        repeat_factor = 1.0
        width_mean = None
        center_norm = None
        center_distance = None

        try:
            lo = np.array([self.param_bounds[key][0] for key in PARAM_KEYS], dtype=float)
            hi = np.array([self.param_bounds[key][1] for key in PARAM_KEYS], dtype=float)
            span = np.maximum(hi - lo, 1e-12)
            if preference.kind == "region" and preference.lb and preference.ub:
                lb = np.array([float(preference.lb[key]) for key in PARAM_KEYS], dtype=float)
                ub = np.array([float(preference.ub[key]) for key in PARAM_KEYS], dtype=float)
                width_norm = np.clip((ub - lb) / span, 0.0, 1.0)
                center_norm_arr = np.clip((0.5 * (lb + ub) - lo) / span, 0.0, 1.0)
                width_mean = float(np.mean(width_norm))
            elif preference.point:
                point = np.array([float(preference.point[key]) for key in PARAM_KEYS], dtype=float)
                center_norm_arr = np.clip((point - lo) / span, 0.0, 1.0)
                width_mean = 0.0
            else:
                raise ValueError("missing preference coordinates")
            center_norm = center_norm_arr.tolist()

            width_start = float(np.clip(self.cfg.get("region_lift_adaptive_width_start", 0.30), 0.0, 1.0))
            width_min_factor = float(
                np.clip(self.cfg.get("region_lift_adaptive_width_min_factor", 0.80), 0.0, 1.0)
            )
            width_limit = max(float(self.cfg.get("region_lift_max_width", 0.80)), width_start + 1e-12)
            width_pressure = float(
                np.clip((float(width_mean) - width_start) / max(width_limit - width_start, 1e-12), 0.0, 1.0)
            )
            width_factor = 1.0 - (1.0 - width_min_factor) * width_pressure

            previous = self._last_region_adoption_note if isinstance(self._last_region_adoption_note, dict) else {}
            prev_center = previous.get("region_center_norm")
            prev_hv_gain = float(previous.get("hv_gain_raw", 0.0) or 0.0)
            if prev_center is not None and prev_hv_gain <= 0.0:
                prev_arr = np.asarray(prev_center, dtype=float).ravel()
                if prev_arr.size == len(PARAM_KEYS):
                    center_distance = float(np.linalg.norm(center_norm_arr - prev_arr) / math.sqrt(len(PARAM_KEYS)))
                    repeat_distance = float(
                        np.clip(self.cfg.get("region_lift_adaptive_repeat_distance", 0.18), 1e-9, 1.0)
                    )
                    if center_distance <= repeat_distance:
                        repeat_factor = float(
                            np.clip(self.cfg.get("region_lift_adaptive_repeat_min_factor", 0.85), 0.0, 1.0)
                        )
        except Exception as exc:
            preference.risk_flags.append(f"adaptive_confidence_width_error:{type(exc).__name__}")

        active_until = max(int(self.cfg.get("region_lift_active_until", 0)), 1)
        progress = float(np.clip(int(t) / max(active_until - 1, 1), 0.0, 1.0))
        late_min_factor = float(
            np.clip(self.cfg.get("region_lift_adaptive_late_min_factor", 0.85), 0.0, 1.0)
        )
        late_factor = 1.0 - (1.0 - late_min_factor) * progress
        unbounded = raw_confidence * base_scale * width_factor * repeat_factor * late_factor
        effective = float(np.clip(max(floor, unbounded), 0.0, 1.0))
        preference.confidence = effective
        adaptive_telemetry = {
            "llm_raw_confidence": raw_confidence,
            "effective_confidence": effective,
            "confidence_base_scale": base_scale,
            "confidence_width_factor": float(width_factor),
            "confidence_repeat_factor": float(repeat_factor),
            "confidence_late_factor": float(late_factor),
            "effective_confidence_floor_applied": bool(effective > unbounded + 1e-12),
            "adaptive_region_width_norm_mean": width_mean,
            "adaptive_region_center_norm": center_norm,
            "adaptive_region_repeat_center_distance": center_distance,
        }
        if not isinstance(preference.raw_response, dict):
            preference.raw_response = {}
        preference.raw_response["_adaptive_confidence"] = adaptive_telemetry
        preference.risk_flags.append(f"adaptive_confidence:{raw_confidence:.3f}->{effective:.3f}")
        return preference

    def _sample_region_candidates_from_preference(
        self,
        *,
        preference: Optional[LLMRegionPreference],
        t: int,
    ) -> np.ndarray:
        region_preference = preference or LLMRegionPreference.none("missing_preference")
        config = RegionLiftConfig.from_config(self.cfg)
        target_n = max(int(config.region_lift_n_anchors), 1)
        oversample = max(int(self.cfg.get("region_lift_candidate_oversample", 1)), 1)
        candidates = sample_region_candidates(
            region_preference,
            self.param_bounds,
            config,
            n_candidates=target_n * oversample,
        )
        point_probes = self._build_point_current_probe_candidates(region_preference)
        if point_probes:
            if len(candidates) == 0:
                candidates = np.vstack(point_probes)
            else:
                candidates = np.vstack([np.atleast_2d(np.asarray(candidates, dtype=float)), np.vstack(point_probes)])
        if len(candidates) == 0:
            return np.empty((0, len(PARAM_KEYS)), dtype=float)
        repaired = [self._repair_theta(row) for row in np.atleast_2d(np.asarray(candidates, dtype=float))]
        unique = self._deduplicate_points(repaired)
        if not unique:
            return np.empty((0, len(PARAM_KEYS)), dtype=float)
        forced_probe_keep = max(int(self.cfg.get("region_lift_point_current_probe_keep", 0)), 0)
        forced_probes = point_probes[-forced_probe_keep:] if forced_probe_keep > 0 and point_probes else []
        ranked_budget = max(target_n - len(forced_probes), 1)
        ranked = self._rank_region_candidates_with_gp(unique, max_keep=ranked_budget)
        if forced_probes:
            ranked = self._deduplicate_points(ranked + [self._repair_theta(row) for row in forced_probes])
        if not ranked:
            return np.empty((0, len(PARAM_KEYS)), dtype=float)
        return np.vstack(ranked)

    def _build_point_current_probe_candidates(
        self,
        preference: LLMRegionPreference,
    ) -> List[np.ndarray]:
        if preference.kind != "point" or not preference.point:
            return []
        levels = max(int(self.cfg.get("region_lift_point_current_probe_levels", 0)), 0)
        if levels <= 0:
            return []

        try:
            point = np.array([float(preference.point[key]) for key in PARAM_KEYS], dtype=float)
        except Exception:
            return []

        lo = np.array([self.param_bounds[key][0] for key in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[key][1] for key in PARAM_KEYS], dtype=float)
        current_hi = hi[:3]
        current_lo = lo[:3]
        base = self._repair_theta(point)
        alphas = np.linspace(1.0 / float(levels), 1.0, levels)
        probes: List[np.ndarray] = []
        for alpha in alphas:
            full_current = base.copy()
            full_current[:3] = base[:3] + float(alpha) * (current_hi - base[:3])
            probes.append(self._repair_theta(full_current))

            front_loaded = base.copy()
            front_loaded[0] = base[0] + float(alpha) * (current_hi[0] - base[0])
            front_loaded[1] = base[1] + float(alpha) * (current_hi[1] - base[1])
            front_loaded[2] = max(base[2], current_lo[2])
            probes.append(self._repair_theta(front_loaded))

        return self._deduplicate_points(probes)

    def _rank_region_candidates_with_gp(
        self,
        candidates: List[np.ndarray],
        *,
        max_keep: int,
    ) -> List[np.ndarray]:
        unique = self._deduplicate_points([self._repair_theta(row) for row in candidates])
        if not unique:
            return []
        max_keep = max(int(max_keep), 1)
        if len(unique) <= max_keep or self.gp is None or self.database is None or self.database.size == 0:
            return unique[:max_keep]

        try:
            X = np.vstack(unique)
            mean_z, sigma_z = self.gp.predict_standardized(X)
            y_mean, y_std = self.gp.target_standardization()
            f_min_model = self._model_target_value(float(self.database.get_f_min()))
            f_min_z = (float(f_min_model) - float(y_mean)) / float(y_std)
            ei = expected_improvement(mean_z, sigma_z, f_min_z)
        except Exception as exc:
            logger.debug("Failed to GP-rank region candidates: %s", exc)
            return unique[:max_keep]

        lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        X_norm = (X - lo) / (hi - lo + 1e-12)
        existing = np.vstack([obs.theta for obs in self.database.get_all()]) if self.database.size else np.empty((0, len(PARAM_KEYS)))
        if existing.size:
            existing_norm = (existing - lo) / (hi - lo + 1e-12)
            novelty = np.min(np.linalg.norm(X_norm[:, None, :] - existing_norm[None, :, :], axis=2), axis=1)
        else:
            novelty = np.ones(X.shape[0], dtype=float)
        log_ei = np.log(np.maximum(np.asarray(ei, dtype=float), 1e-12))
        candidate_score = (
            self._zscore_feature(log_ei)
            + 0.35 * self._zscore_feature(np.asarray(sigma_z, dtype=float))
            + 0.20 * self._zscore_feature(novelty)
        )
        order = np.argsort(np.asarray(candidate_score, dtype=float))[::-1]
        min_sep = 0.5 * float(self.cfg.get("region_lift_close_distance", 0.05))

        selected: List[int] = []
        for idx in order:
            if any(np.linalg.norm(X_norm[idx] - X_norm[j]) < min_sep for j in selected):
                continue
            selected.append(int(idx))
            if len(selected) >= max_keep:
                break
        if len(selected) < max_keep:
            for idx in order:
                if int(idx) in selected:
                    continue
                selected.append(int(idx))
                if len(selected) >= max_keep:
                    break
        return [X[idx].copy() for idx in selected]

    @staticmethod
    def _zscore_feature(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).ravel()
        if len(arr) <= 1:
            return np.zeros_like(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std <= 1e-12:
            return np.zeros_like(arr)
        return np.clip((arr - mean) / std, -3.0, 3.0)

    def _model_target_value(self, value: float) -> float:
        try:
            transformed = self.gp.transform_targets(np.array([float(value)], dtype=float))
            return float(np.asarray(transformed, dtype=float).ravel()[0])
        except Exception:
            return float(value)

    def _build_region_preference_state(
        self,
        *,
        t: int,
        w_vec: np.ndarray,
        scalar_y: np.ndarray,
        ideal_point_raw: np.ndarray,
    ) -> Dict[str, Any]:
        X_train, Y_raw = self.database.get_train_XY(feasible_only=True, normalize_X=False, normalize_Y=False)
        scalar = np.asarray(scalar_y, dtype=float).ravel()
        order = np.argsort(scalar)[: min(5, len(scalar))]
        top_rows = []
        for idx in order:
            top_rows.append(
                {
                    "theta": np.asarray(X_train[idx], dtype=float).round(6).tolist(),
                    "raw_objectives": np.asarray(Y_raw[idx], dtype=float).round(6).tolist(),
                    "scalar_y": float(scalar[idx]),
                }
            )
        return {
            "iteration": int(t),
            "w_vec": np.asarray(w_vec, dtype=float).round(6).tolist(),
            "ideal_point_raw": np.asarray(ideal_point_raw, dtype=float).round(6).tolist(),
            "y_min": np.asarray(self._y_tilde_min, dtype=float).round(6).tolist(),
            "y_max": np.asarray(self._y_tilde_max, dtype=float).round(6).tolist(),
            "eta": float(self.cfg.get("eta", 0.05)),
            "f_min": float(self.database.get_f_min()),
            "hv_feedback": self.database.get_hv_feedback_summary(window=3),
            "boundary_failures": self.database.get_boundary_failure_stats(),
            "previous_region_thinking": self._previous_region_thinking,
            "previous_thinking": self._previous_region_thinking,
            "last_region_adoption_note": self._last_region_adoption_note,
            "adoption_note": self._last_region_adoption_note,
            "top_scalar_points": top_rows,
            "recent_observations": [
                {
                    "theta": obs.theta.round(6).tolist(),
                    "objectives": obs.objectives.round(6).tolist(),
                    "feasible": bool(obs.feasible),
                    "source": obs.source,
                }
                for obs in self.database.get_all()[-5:]
            ],
        }

    def _finalize_region_lift_trust(self, *, hv_gain: float) -> None:
        if self._last_region_lift_summary is None:
            return
        summary = dict(self._last_region_lift_summary)
        trust_before = float(summary.get("trust_before", self._region_lift_trust))
        trust_after = float(self._region_lift_trust)
        if self._is_lgbo_region_lift_mode():
            summary["trust_before"] = trust_before
            summary["trust_after"] = trust_after
            summary["trust_update_reason"] = "skipped_lgbo_mode"
            summary["hv_gain_raw"] = float(hv_gain)
            self._last_region_lift_summary = summary
            self._region_lift_telemetry.append(summary)
            self._update_region_prompt_memory(summary)
            return

        reason = "skipped_not_lifted"
        beta = float(np.clip(self.cfg.get("region_lift_trust_beta", 0.2), 0.0, 1.0))

        if summary.get("selected_source") == "lifted":
            recent = [
                float(item.get("hv_gain_raw", 0.0))
                for item in self._region_lift_telemetry[-5:]
                if item.get("selected_source") == "lifted"
            ]
            threshold = float(np.median(recent)) if recent else 0.0
            inside = bool(summary.get("lift_candidate_inside_region", False))
            if inside and float(hv_gain) > threshold:
                success_t = 1.0
                reason = "lifted_inside_improved"
            elif inside:
                success_t = 0.5
                reason = "lifted_inside_unclear"
            elif float(hv_gain) > threshold:
                success_t = 0.25
                reason = "lifted_outside_improved"
            else:
                success_t = 0.0
                reason = "lifted_no_improvement"
            trust_after = float(np.clip((1.0 - beta) * trust_before + beta * success_t, 0.0, 1.0))
        elif summary.get("fallback_reason") in {
            "parse_fail",
            "invalid_json",
            "invalid_kind",
            "invalid_region_bounds",
            "invalid_point",
            "low_confidence",
            "non_raw_coordinate_space",
            "non_promising_direction",
            "bad_region_volume",
            "bad_region_width",
            "low_feasible_anchor_ratio",
        }:
            trust_after = float(np.clip(trust_before * (1.0 - 0.25 * beta), 0.0, 1.0))
            reason = "small_decay_after_invalid_preference"

        self._region_lift_trust = trust_after
        summary["trust_before"] = trust_before
        summary["trust_after"] = trust_after
        summary["trust_update_reason"] = reason
        summary["hv_gain_raw"] = float(hv_gain)
        self._last_region_lift_summary = summary
        self._region_lift_telemetry.append(summary)
        self._update_region_prompt_memory(summary)

    def _update_region_prompt_memory(self, summary: Dict[str, Any]) -> None:
        preference = summary.get("preference") if isinstance(summary, dict) else None
        thinking = ""
        if isinstance(preference, dict):
            thinking = str(preference.get("mechanistic_thinking") or preference.get("reason") or "")
        self._previous_region_thinking = thinking[:500] if thinking else None
        self._last_region_adoption_note = {
            "iteration": int(summary.get("iteration", summary.get("t", -1)) or -1),
            "suggestion_used": bool(
                summary.get("acquisition_used_lift", False) or summary.get("selected_source") == "lifted"
            ),
            "selected_source": summary.get("selected_source"),
            "fallback_reason": summary.get("fallback_reason"),
            "actual_theta": summary.get("evaluated_theta"),
            "hv_gain_raw": float(summary.get("hv_gain_raw", 0.0) or 0.0),
            "region_center_norm": summary.get("region_center_norm"),
            "region_width_norm_mean": summary.get("region_width_norm_mean"),
            "selected_changed_by_lift": bool(summary.get("selected_changed_by_lift", False)),
        }

    def save_results(self, output_dir: str = "results") -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        self.database.save(str(output / "database.json"))
        self.database.save(str(output / "db_final.json"))

        pareto = [
            {
                "theta": obs.theta.tolist(),
                "objectives": obs.objectives.tolist(),
                "source": obs.source,
                "iteration": obs.iteration,
            }
            for obs in self.database.get_pareto_front()
        ]
        with open(output / "pareto_front.json", "w", encoding="utf-8") as f:
            json.dump(pareto, f, indent=2, ensure_ascii=False)

        hv_raw = self.database.compute_hypervolume_raw()
        hv_canonical = canonical_hv_from_raw(hv_raw, self.database.hv_max)
        hv_display = self.database.compute_hypervolume()
        rerank_summary = self._summarize_rerank_telemetry()
        region_lift_summary = self._summarize_region_lift_telemetry()
        summary = {
            "n_total": self.database.size,
            "n_feasible": self.database.n_feasible,
            "pareto_size": self.database.pareto_size,
            "hypervolume": hv_display,
            "display_hv": hv_display,
            "canonical_hv": hv_canonical,
            "hypervolume_canonical": hv_canonical,
            "hypervolume_raw": hv_raw,
            "warmstart_trace": self._warmstart_hv_trace,
            "warmstart_portfolio_summary": self._warmstart_portfolio_summary,
            "hv_trace": self._hv_eval_trace,
            "last_guidance": self._previous_guidance,
            "last_gp_llm_coupling": self._last_coupling_summary,
            "last_proposal_summary": self._last_proposal_summary,
            "last_acq_prior_summary": self._last_acq_prior_summary,
            "last_llm_rerank_summary": self._last_rerank_summary,
            "last_region_lifted_gp_summary": self._last_region_lift_summary,
            "last_candidate_source_counts": self._last_candidate_source_counts,
            "rerank_telemetry": self._rerank_telemetry,
            "region_lift_telemetry": self._region_lift_telemetry,
            "llm_model_display": canonical_model_label(self.cfg.get("llm_model")),
            "config": self._jsonable_config(),
            **rerank_summary,
            **region_lift_summary,
        }
        with open(output / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to %s", output)

    def _save_checkpoint(self, t: int) -> None:
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.database.save(str(ckpt_dir / f"db_t{t:04d}.json"))
        with open(ckpt_dir / f"af_t{t:04d}.json", "w", encoding="utf-8") as f:
            json.dump(self.af.save_state(), f, indent=2)
        with open(ckpt_dir / f"summary_t{t:04d}.json", "w", encoding="utf-8") as f:
            hv_raw = self.database.compute_hypervolume_raw()
            hv_canonical = canonical_hv_from_raw(hv_raw, self.database.hv_max)
            hv_display = self.database.compute_hypervolume()
            json.dump(
                {
                    "t": t,
                    "n_total": self.database.size,
                    "n_feasible": self.database.n_feasible,
                    "pareto_size": self.database.pareto_size,
                    "hypervolume": hv_display,
                    "display_hv": hv_display,
                    "canonical_hv": hv_canonical,
                    "hypervolume_canonical": hv_canonical,
                    "hypervolume_raw": hv_raw,
                    "last_guidance": self._previous_guidance,
                    "last_gp_llm_coupling": self._last_coupling_summary,
                    "last_proposal_summary": self._last_proposal_summary,
                    "last_acq_prior_summary": self._last_acq_prior_summary,
                    "last_llm_rerank_summary": self._last_rerank_summary,
                    "last_region_lifted_gp_summary": self._last_region_lift_summary,
                    "last_candidate_source_counts": self._last_candidate_source_counts,
                    "llm_model_display": canonical_model_label(self.cfg.get("llm_model")),
                    "config": self._jsonable_config(),
                },
                f,
                indent=2,
            )

    def _update_dynamic_bounds(self) -> None:
        feasible = self.database.get_feasible()
        if not feasible:
            self._y_tilde_min = np.zeros(3, dtype=float)
            self._y_tilde_max = np.ones(3, dtype=float)
            return

        Y_raw = np.array([obs.objectives for obs in feasible], dtype=float)
        Y_tilde = log_transform_objectives(Y_raw)
        self._y_tilde_min, self._y_tilde_max = compute_objective_preprocess_context(
            Y_tilde,
            self.database.ideal_point,
            self.database.ref_point,
            preprocess_mode=str(self.cfg.get("objective_preprocess_mode", "minmax")),
        )

    def _compute_scalarized_targets(
        self,
        *,
        Y_raw: np.ndarray,
        w_vec: np.ndarray,
        ideal_point_raw: np.ndarray,
    ) -> np.ndarray:
        mode = str(self.cfg.get("scalarization_mode", "log_ideal_gap") or "log_ideal_gap").lower()
        if mode == "parego_reference":
            return compute_parego_reference_from_raw(
                Y_raw=Y_raw,
                w_vec=w_vec,
                eta=float(self.cfg.get("eta", 0.05)),
                eps_min=1e-6,
                invert_weights=bool(self.cfg.get("parego_invert_weights", False)),
            )
        return compute_tchebycheff_from_raw_with_ideal(
            Y_raw=Y_raw,
            w_vec=w_vec,
            ideal_point_raw=ideal_point_raw,
            y_min=self._y_tilde_min,
            y_max=self._y_tilde_max,
            eta=float(self.cfg.get("eta", 0.05)),
            preprocess_mode=str(self.cfg.get("objective_preprocess_mode", "minmax")),
        )

    def _next_weight(self) -> np.ndarray:
        sampling_mode = str(self.cfg.get("weight_sampling_mode", "cycle_without_replacement") or "cycle_without_replacement").lower()
        if sampling_mode == "random_with_replacement":
            idx = int(self._rng.integers(0, len(self._weight_set)))
            return self._weight_set[idx]
        if not self._weight_order:
            order = self._rng.permutation(len(self._weight_set))
            self._weight_order = order.tolist()
        return self._weight_set[self._weight_order.pop()]

    def _compute_dynamic_ideal_point(self, Y_raw: np.ndarray) -> np.ndarray:
        Y_raw = np.atleast_2d(np.asarray(Y_raw, dtype=float))
        if Y_raw.size == 0:
            return np.asarray(self.database.ideal_point, dtype=float).copy()
        return Y_raw.min(axis=0)

    def _sobol_grid(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        n_points: int,
        seed: int,
    ) -> np.ndarray:
        lb = np.asarray(lb, dtype=float).ravel()
        ub = np.asarray(ub, dtype=float).ravel()
        n_points = max(1, int(n_points))
        span = ub - lb
        adj_ub = np.where(span <= 1e-12, lb + 1e-9, ub)
        m = int(np.ceil(np.log2(n_points)))
        sampler = qmc.Sobol(d=len(lb), scramble=True, seed=seed)
        sample = sampler.random_base2(m=m)[:n_points]
        scaled = qmc.scale(sample, lb, adj_ub)
        return np.clip(scaled, lb, ub)

    def _estimate_search_sigma(self) -> np.ndarray:
        lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        stagnation = int(self.database.get_stagnation_count())
        sigma_scale = 1.0 + 0.20 * min(stagnation, 3)
        return np.maximum((hi - lo) * 0.15 * sigma_scale, 1e-3)

    def _compute_uncertainty_hotspots(self, t: int) -> List[Dict[str, Any]]:
        n_probe = int(self.cfg.get("guidance_probe_size", 128))
        top_k = int(self.cfg.get("guidance_hotspots", 5))
        lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        probe = self._sobol_grid(lo, hi, n_probe, seed=int(self.cfg.get("w_sample_seed", 0) or 0) + 1000 + t)
        _, std = self.gp.predict(probe)
        order = np.argsort(std)[::-1][:top_k]
        return [
            {
                "theta": probe[idx].tolist(),
                "std": float(std[idx]),
            }
            for idx in order
        ]

    def _compute_sensitivity_summary(self) -> str:
        """Per-parameter linear sensitivity from feasible observations (OLS)."""
        feasible = self.database.get_feasible()
        if len(feasible) < 6:
            return "none"

        X = np.array([obs.theta for obs in feasible], dtype=float)
        Y = np.array([obs.objectives for obs in feasible], dtype=float)
        lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        span = hi - lo
        X_norm = (X - lo) / span

        obj_info = [("time", "s"), ("temp", "K"), ("aging", "%")]
        param_units = ["A", "A", "A", "SOC", "SOC"]

        lines = []
        for j in range(3):
            y = Y[:, j]
            A = np.column_stack([X_norm, np.ones(len(y))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            betas = coeffs[:5]
            per_unit = betas / span
            sorted_idx = np.argsort(np.abs(per_unit))[::-1]

            parts = []
            for k in sorted_idx[:3]:
                val = per_unit[k]
                # Skip near-zero sensitivities (noise floor)
                if abs(val) < 1e-6 * max(np.abs(per_unit)):
                    continue
                if abs(val) >= 1:
                    fmt = f"{val:+.1f}"
                elif abs(val) >= 0.01:
                    fmt = f"{val:+.3f}"
                else:
                    fmt = f"{val:+.6f}"
                parts.append(f"d{obj_info[j][0]}/d{PARAM_KEYS[k]}≈{fmt} {obj_info[j][1]}/{param_units[k]}")
            lines.append("  " + ", ".join(parts))

        return ("Parameter sensitivity (OLS, top-3 per objective):\n"
                + "\n".join(lines)) if lines else "none"

    def _build_guidance_state(
        self,
        *,
        t: int,
        w_vec: np.ndarray,
        scalar_y: np.ndarray,
        ideal_point_raw: np.ndarray,
        proposal_summary: Optional[Dict[str, Any]],
        uncertainty_hotspots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        selective_history_summary = self._build_selective_history_summary(
            w_vec=w_vec,
            scalar_y=scalar_y,
            proposal_summary=proposal_summary,
        )
        sensitivity_summary = self._compute_sensitivity_summary()
        top_scalar_protocols = self._build_top_scalar_protocols_summary(
            scalar_y=scalar_y,
            top_k=int(self.cfg.get("guidance_top_scalar_k", 3)),
        )
        hv_feedback = self.database.get_hv_feedback_summary(window=3)
        similar_weight_guidance = self.database.get_similar_weight_guidance_stats(
            w_vec=w_vec,
            similarity_threshold=float(self.cfg.get("coupling_history_similarity_threshold", 0.85)),
            fallback_score=float(self.cfg.get("coupling_history_fallback_score", 0.75)),
        )
        boundary_failure_stats = self.database.get_boundary_failure_stats(
            safe_dsoc_sum_max=float(self.cfg.get("llm_safe_dsoc_sum_max", LLM_SAFE_DSOC_SUM_MAX)),
            hard_dsoc_sum_max=float(DSOC_SUM_MAX),
            recent_window=10,
        )
        return {
            "iteration": t + 1,
            "max_iterations": int(self.cfg["max_iterations"]),
            "w_vec": np.asarray(w_vec, dtype=float).tolist(),
            "theta_best": self.database.get_theta_best().tolist(),
            "f_min": float(self.database.get_f_min()),
            "eta": float(self.cfg.get("eta", 0.05)),
            "mu": self.database.get_theta_best().tolist(),
            "sigma": self._estimate_search_sigma().tolist(),
            "y_min": np.asarray(self._y_tilde_min, dtype=float).tolist(),
            "y_max": np.asarray(self._y_tilde_max, dtype=float).tolist(),
            "stagnation_count": int(self.database.get_stagnation_count()),
            "database": self.database,
            "uncertainty_hotspots": uncertainty_hotspots,
            "previous_guidance": self._previous_guidance,
            "ideal_point": np.asarray(ideal_point_raw, dtype=float).tolist(),
            "current_hv": float(hv_feedback["current_hv"]),
            "hv_delta_last_3": float(hv_feedback["hv_delta_last_k"]),
            "hv_feedback_summary": str(hv_feedback["summary"]),
            "pareto_size": int(self.database.pareto_size),
            "scalarization_formula": self._build_scalarization_formula_text(),
            "top_scalar_protocols": top_scalar_protocols,
            "similar_weight_guidance_success": str(similar_weight_guidance["summary"]),
            "boundary_failure_stats": str(boundary_failure_stats["summary"]),
            "proposal_summary": proposal_summary or {},
            "selective_history_summary": selective_history_summary,
            "sensitivity_summary": sensitivity_summary,
            "safe_dsoc_sum_max": float(self.cfg.get("llm_safe_dsoc_sum_max", LLM_SAFE_DSOC_SUM_MAX)),
            "hard_dsoc_sum_max": float(DSOC_SUM_MAX),
        }

    def _build_gp_llm_coupling_from_guidance(
        self,
        guidance: IterationGuidance,
        t: int,
        *,
        w_vec: np.ndarray,
    ) -> Tuple[Any, np.ndarray]:
        local_center = None
        local_lb = None
        local_ub = None
        local_sigma = None
        if guidance.mode == "region":
            local_lb = self._repair_theta(np.asarray(guidance.lb, dtype=float))
            local_ub = self._repair_theta(np.asarray(guidance.ub, dtype=float))
            grid = self._sobol_grid(
                local_lb,
                local_ub,
                n_points=int(self.cfg.get("guidance_grid_size", 64)),
                seed=int(self.cfg.get("w_sample_seed", 0) or 0) + 2000 + t,
            )
            weights = np.full(grid.shape[0], 1.0 / max(grid.shape[0], 1), dtype=float)
        else:
            local_center = self._repair_theta(guidance.representative_point())
            local_sigma = self._guidance_local_sigma()
            lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
            hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
            local_lb = np.maximum(lo, local_center - 2.0 * local_sigma)
            local_ub = np.minimum(hi, local_center + 2.0 * local_sigma)
            grid = self._sobol_grid(
                local_lb,
                local_ub,
                n_points=int(self.cfg.get("guidance_point_grid_size", 25)),
                seed=int(self.cfg.get("w_sample_seed", 0) or 0) + 3000 + t,
            )
            grid = np.vstack([local_center[None, :], grid])
            grid = np.vstack(self._deduplicate_points([self._repair_theta(row) for row in grid]))
            diff = (grid - local_center[np.newaxis, :]) / local_sigma[np.newaxis, :]
            weights = np.exp(-0.5 * np.sum(diff ** 2, axis=1))

        gate_info = self._compute_coupling_gate(
            t=t,
            w_vec=w_vec,
            guidance=guidance,
            guidance_candidates=grid,
        )

        coupling = self.gp.build_preference_coupling(
            grid=grid,
            weights=weights,
            confidence=float(guidance.confidence),
            mode=guidance.mode,
            t=t,
            lambda_max=float(self.cfg.get("lambda_max", 1.0)),
            lambda_min=float(self.cfg.get("lambda_min", 0.0)),
            decay_rate=float(self.cfg.get("lambda_decay_rate", 0.75)),
            gate=float(gate_info["gate"]),
            align_score=float(gate_info["align_score"]),
            history_score=float(gate_info["history_score"]),
            hv_score=float(gate_info["hv_score"]),
            stage_score=float(gate_info["stage_score"]),
            local_center=local_center,
            local_lb=local_lb,
            local_ub=local_ub,
            local_sigma=local_sigma,
        )
        return coupling, grid

    def _build_acquisition_prior(
        self,
        *,
        t: int,
        guidance: Optional[IterationGuidance],
        guidance_candidates: Optional[np.ndarray],
        proposal_candidates: np.ndarray,
    ) -> Optional[AcquisitionPrior]:
        proposal_alpha = 0.0
        proposal_anchor = 0.0
        proposal_scale = 1.0
        proposal_scorer = None
        agreement = 1.0
        max_iterations = max(int(self.cfg.get("max_iterations", 1)), 1)
        feasible_count = len(self.database.get_feasible())

        if self.proposal is not None and self.proposal.is_ready():
            proposal_scorer = self.proposal.score
            warmup_span = max(int(self.cfg.get("proposal_prior_warmup_span", 8)), 1)
            min_train = max(int(self.cfg.get("proposal_min_train_size", 8)), 0)
            data_factor = np.clip((feasible_count - min_train + 1) / warmup_span, 0.0, 1.0)
            iter_factor = 0.35 + 0.65 * ((t + 1) / max_iterations)
            proposal_alpha = float(self.cfg.get("proposal_prior_alpha", 0.20)) * float(data_factor) * float(iter_factor)
            proposal_probe = proposal_candidates if proposal_candidates.size else self._repair_theta(self.database.get_theta_best())[None, :]
            proposal_scores = np.asarray(self.proposal.score(proposal_probe), dtype=float).ravel()
            proposal_anchor = float(np.median(proposal_scores)) if len(proposal_scores) else 0.0
            proposal_scale = float(np.std(proposal_scores)) if len(proposal_scores) > 1 else 1.0
            proposal_scale = max(proposal_scale, 1e-3)

        guidance_alpha = 0.0
        guidance_mode = None
        guidance_center = None
        guidance_lb = None
        guidance_ub = None
        guidance_sigma = None
        if guidance is not None:
            stage_factor = max(0.0, 1.0 - (t / max_iterations))
            if proposal_scorer is not None:
                if guidance_candidates is not None and len(guidance_candidates):
                    probe = np.atleast_2d(np.asarray(guidance_candidates, dtype=float))
                else:
                    probe = self._repair_theta(guidance.representative_point())[None, :]
                guide_scores = np.asarray(self.proposal.score(probe), dtype=float).ravel()
                centered = (guide_scores - proposal_anchor) / max(proposal_scale, 1e-6)
                agreement = float(np.mean(_stable_sigmoid(centered))) if len(centered) else 1.0
            guidance_alpha = (
                float(self.cfg.get("guidance_prior_alpha", 0.30))
                * float(np.clip(guidance.confidence, 0.0, 1.0))
                * float(stage_factor)
                * float(agreement)
            )
            guidance_mode = guidance.mode
            if guidance.mode == "region":
                guidance_lb = self._repair_theta(np.asarray(guidance.lb, dtype=float))
                guidance_ub = self._repair_theta(np.asarray(guidance.ub, dtype=float))
            else:
                guidance_center = self._repair_theta(guidance.representative_point())
                guidance_sigma = self._guidance_local_sigma()

        prior = AcquisitionPrior(
            proposal_scorer=proposal_scorer,
            proposal_alpha=float(proposal_alpha),
            proposal_anchor=float(proposal_anchor),
            proposal_scale=float(proposal_scale),
            guidance_alpha=float(guidance_alpha),
            guidance_mode=guidance_mode,
            guidance_center=guidance_center,
            guidance_lb=guidance_lb,
            guidance_ub=guidance_ub,
            guidance_sigma=guidance_sigma,
            safe_dsoc_sum_max=float(self.cfg.get("llm_safe_dsoc_sum_max", LLM_SAFE_DSOC_SUM_MAX)),
            hard_dsoc_sum_max=float(DSOC_SUM_MAX),
            safe_risk_weight=float(self.cfg.get("acq_risk_safe_weight", 0.20)),
            hard_risk_weight=float(self.cfg.get("acq_risk_hard_weight", 3.00)),
            monotone_risk_weight=float(self.cfg.get("acq_risk_monotone_weight", 0.40)),
            agreement=float(agreement),
        )
        return prior if prior.is_active() else None

    def _compute_canonical_hv(self) -> float:
        return canonical_hv_from_raw(
            self.database.compute_hypervolume_raw(),
            getattr(self.database, "hv_max", 1.0),
        )

    def _compute_recent_hv_gain_mean(self, window: int) -> float:
        stats = self.database.get_iteration_stats()
        window = max(int(window), 1)
        if len(stats) < 2:
            return 0.0

        hv_max = max(float(getattr(self.database, "hv_max", 1.0)), 1e-12)
        tail = stats[-(window + 1):]
        gains: List[float] = []
        for prev, curr in zip(tail[:-1], tail[1:]):
            prev_raw = float(prev.get("hypervolume_raw", 0.0))
            curr_raw = float(curr.get("hypervolume_raw", prev_raw))
            gains.append((curr_raw - prev_raw) / hv_max)
        return float(np.mean(gains)) if gains else 0.0

    def _compute_recent_violation_rate(self, window: int) -> float:
        all_obs = self.database.get_all()
        n_recent = max(int(window), 1) * max(int(self.cfg.get("n_select", 1)), 1)
        recent = all_obs[-n_recent:]
        if not recent:
            return 0.0
        violations = sum(1 for obs in recent if not obs.feasible)
        return float(violations / len(recent))

    def _compute_recent_llm_uncertainty(self, window: int) -> float:
        window = max(int(window), 1)
        if not self._rerank_telemetry:
            return 0.0
        tail = self._rerank_telemetry[-window:]
        values = [
            float(item["llm_entropy_mean"])
            for item in tail
            if item.get("llm_called") and item.get("llm_entropy_mean") is not None
        ]
        return float(np.mean(values)) if values else 0.0

    def _build_rerank_history_summary(self, window: int) -> List[Dict[str, Any]]:
        stats = self.database.get_iteration_stats()
        if not stats:
            return []
        hv_max = max(float(getattr(self.database, "hv_max", 1.0)), 1e-12)
        tail = stats[-max(int(window), 1):]
        summary: List[Dict[str, Any]] = []
        for stat in tail:
            summary.append(
                {
                    "iteration": int(stat.get("t", stat.get("iteration", 0))),
                    "canonical_hv": float(stat.get("hypervolume_raw", 0.0)) / hv_max,
                    "pareto_size": int(stat.get("pareto_size", 0)),
                    "n_feasible": int(stat.get("n_feasible", 0)),
                    "n_new_evals": int(stat.get("n_new_evals", 0)),
                    "llm_rerank_applied": bool((stat.get("llm_rerank") or {}).get("applied", False)),
                }
            )
        return summary

    def _build_rerank_state(
        self,
        *,
        t: int,
        w_vec: np.ndarray,
        scalar_y: np.ndarray,
    ) -> RerankState:
        gate_window = max(int(self.cfg.get("llm_rerank_gate_window", 5)), 1)
        gamma = float(np.clip(self.cfg.get("llm_rerank_gamma_quantile", 0.20), 0.01, 0.99))
        scalar = np.asarray(scalar_y, dtype=float).ravel()
        tau_t = float(np.quantile(scalar, gamma)) if len(scalar) else float(self.database.get_f_min())
        boundary_failure = self.database.get_boundary_failure_stats(
            safe_dsoc_sum_max=float(self.cfg.get("llm_safe_dsoc_sum_max", LLM_SAFE_DSOC_SUM_MAX)),
            hard_dsoc_sum_max=float(DSOC_SUM_MAX),
            recent_window=gate_window,
        )
        similar_weight = self.database.get_similar_weight_guidance_stats(
            w_vec=w_vec,
            similarity_threshold=float(self.cfg.get("coupling_history_similarity_threshold", 0.85)),
            fallback_score=float(self.cfg.get("coupling_history_fallback_score", 0.75)),
        )
        hv_feedback = self.database.get_hv_feedback_summary(window=gate_window)

        return RerankState(
            iter_id=int(t),
            w_vec=np.asarray(w_vec, dtype=float).tolist(),
            tau_t=float(tau_t),
            scalar_best=float(np.min(scalar)) if len(scalar) else float(self.database.get_f_min()),
            hv_current=float(self._compute_canonical_hv()),
            hv_gain_recent_mean=float(self._compute_recent_hv_gain_mean(gate_window)),
            violation_rate_recent=float(self._compute_recent_violation_rate(gate_window)),
            llm_uncertainty_recent=float(self._compute_recent_llm_uncertainty(gate_window)),
            safe_margin_summary={
                "boundary_failure_stats": boundary_failure,
                "similar_weight_guidance": similar_weight,
                "hv_feedback": hv_feedback,
            },
            history_summary=self._build_rerank_history_summary(gate_window),
        )

    def _augment_rerank_candidates(self, candidates: List[Any]) -> List[Any]:
        soft_limit = float(self.cfg.get("llm_safe_dsoc_sum_max", 0.65))
        hard_limit = float(DSOC_SUM_MAX)
        augmented: List[Any] = []
        for candidate in candidates:
            x = np.asarray(candidate.x, dtype=float).ravel()
            dsoc_sum = float(x[3] + x[4]) if x.size >= 5 else None
            monotone_flag = None
            if x.size >= 3:
                monotone_flag = bool(x[0] >= x[1] >= x[2])
            augmented.append(
                dataclasses.replace(
                    candidate,
                    dSOC_sum=dsoc_sum,
                    margin_to_soft_limit=(
                        None if dsoc_sum is None else float(soft_limit - dsoc_sum)
                    ),
                    hard_violation_flag=(
                        None if dsoc_sum is None else bool(dsoc_sum > hard_limit + 1e-12)
                    ),
                    monotone_flag=monotone_flag,
                )
            )
        return augmented

    def _rerank_mode(self) -> str:
        mode = str(self.cfg.get("llm_rerank_mode", "none") or "none")
        if mode == "const_gate":
            return "unsafe_legacy_const_gate"
        return mode

    def _rerank_fail_open(self, mode: str) -> bool:
        if mode in {"ei_preserving_tiebreak", "risk_veto_only"}:
            return True
        return bool(self.cfg.get("llm_rerank_parse_fail_open", self.cfg.get("llm_rerank_fail_open_to_plain_ei", True)))

    def _summarize_rerank_telemetry(self) -> Dict[str, Any]:
        if not self._rerank_telemetry:
            return {
                "rerank_applied_count": 0,
                "rerank_changed_count": 0,
                "rerank_fail_open_count": 0,
                "mean_ei_ratio_when_changed": None,
                "mean_hv_gain_when_changed": None,
            }

        applied = [item for item in self._rerank_telemetry if item.get("llm_called")]
        changed = [item for item in self._rerank_telemetry if item.get("selected_changed")]
        fail_open = [item for item in self._rerank_telemetry if item.get("fallback_reason")]
        ei_ratios = [float(item["ei_ratio"]) for item in changed if item.get("ei_ratio") is not None]
        hv_gains = [float(item["hv_gain"]) for item in changed if item.get("hv_gain") is not None]
        return {
            "rerank_applied_count": int(len(applied)),
            "rerank_changed_count": int(len(changed)),
            "rerank_fail_open_count": int(len(fail_open)),
            "mean_ei_ratio_when_changed": None if not ei_ratios else float(np.mean(ei_ratios)),
            "mean_hv_gain_when_changed": None if not hv_gains else float(np.mean(hv_gains)),
        }

    def _summarize_region_lift_telemetry(self) -> Dict[str, Any]:
        if not self._region_lift_telemetry:
            return {
                "region_lift_attempt_count": 0,
                "region_lift_accept_count": 0,
                "lift_accept_rate": 0.0,
                "acquisition_used_lift_count": 0,
                "acquisition_used_lift_rate": 0.0,
                "selection_guard_pass_count": 0,
                "selection_guard_pass_rate": 0.0,
                "effective_lift_accept_count": 0,
                "effective_lift_accept_rate": 0.0,
                "effective_selection_change_count": 0,
                "effective_selection_change_rate": 0.0,
                "region_lift_fallback_count": 0,
                "region_lift_fallback_reasons": {},
                "plain_candidate_inside_region_count": 0,
                "plain_candidate_inside_region_rate": 0.0,
                "diagnostic_override_candidate_count": 0,
                "region_pool_influenced_acquisition_count": 0,
                "region_pool_influenced_acquisition_rate": 0.0,
                "region_influence_gate_pass_count": 0,
                "inactive_window_skipped_count": 0,
                "zero_shift_accept_count": 0,
                "mean_region_lift_hv_gain": None,
            }
        attempts = list(self._region_lift_telemetry)
        accepted = [item for item in attempts if bool(item.get("accepted", False))]
        acquisition_used = [
            item for item in attempts if bool(item.get("acquisition_used_lift", False))
        ]
        guard_passed = [
            item for item in attempts if bool(item.get("selection_guard_passed", False))
        ]
        fallbacks = [item for item in attempts if item.get("fallback_reason")]
        plain_inside = [item for item in attempts if bool(item.get("plain_candidate_inside_region", False))]
        diagnostic_available = [
            item for item in attempts
            if bool(item.get("diagnostic_override_candidate_available", False))
        ]
        influenced = [
            item for item in attempts
            if bool(item.get("region_pool_influenced_acquisition", False))
        ]
        gated = [
            item for item in attempts
            if bool(item.get("region_influence_gate_passed", False))
        ]
        inactive_skips = [
            item for item in attempts
            if bool(item.get("inactive_window_skipped", False))
        ]
        fallback_counter: Counter[str] = Counter(
            str(item.get("fallback_reason"))
            for item in fallbacks
            if item.get("fallback_reason")
        )
        effective = [
            item
            for item in accepted
            if bool(
                item.get(
                    "effective_selection_change",
                    int(item.get("selected_index_after", -1))
                    != int(item.get("selected_index_before", -1)),
                )
            )
        ]
        zero_shift = [
            item
            for item in accepted
            if float(item.get("max_shift_z", 0.0)) <= float(self.cfg.get("region_lift_log_ei_eps", 1e-12))
        ]
        gains = [float(item["hv_gain_raw"]) for item in attempts if item.get("hv_gain_raw") is not None]
        return {
            "region_lift_attempt_count": int(len(attempts)),
            "region_lift_accept_count": int(len(accepted)),
            "lift_accept_rate": float(len(accepted) / max(len(attempts), 1)),
            "acquisition_used_lift_count": int(len(acquisition_used)),
            "acquisition_used_lift_rate": float(len(acquisition_used) / max(len(attempts), 1)),
            "selection_guard_pass_count": int(len(guard_passed)),
            "selection_guard_pass_rate": float(len(guard_passed) / max(len(attempts), 1)),
            "effective_lift_accept_count": int(len(effective)),
            "effective_lift_accept_rate": float(len(effective) / max(len(attempts), 1)),
            "effective_selection_change_count": int(len(effective)),
            "effective_selection_change_rate": float(len(effective) / max(len(attempts), 1)),
            "region_lift_fallback_count": int(len(fallbacks)),
            "region_lift_fallback_reasons": dict(fallback_counter),
            "plain_candidate_inside_region_count": int(len(plain_inside)),
            "plain_candidate_inside_region_rate": float(len(plain_inside) / max(len(attempts), 1)),
            "diagnostic_override_candidate_count": int(len(diagnostic_available)),
            "region_pool_influenced_acquisition_count": int(len(influenced)),
            "region_pool_influenced_acquisition_rate": float(len(influenced) / max(len(attempts), 1)),
            "region_influence_gate_pass_count": int(len(gated)),
            "inactive_window_skipped_count": int(len(inactive_skips)),
            "zero_shift_accept_count": int(len(zero_shift)),
            "mean_region_lift_hv_gain": None if not gains else float(np.mean(gains)),
        }

    def _maybe_apply_llm_rerank(
        self,
        *,
        t: int,
        w_vec: np.ndarray,
        scalar_y: np.ndarray,
        acq_result: Any,
        plain_selected_indices: List[int],
        plain_selected_scores: np.ndarray,
    ) -> Any:
        mode = self._rerank_mode()
        if not bool(self.cfg.get("enable_llm_rerank", False)) or mode == "none":
            return acq_result
        if acq_result.candidate_pool is None or len(acq_result.candidate_pool) == 0:
            self._last_rerank_summary = {
                "active": True,
                "applied": False,
                "fallback_reason": "empty_candidate_pool",
                "rerank_mode": mode,
            }
            return acq_result

        pareto_points = [obs.theta for obs in self.database.get_pareto_front()]
        candidate_infos = build_ei_candidate_pool(
            acq_result.candidate_pool,
            acq_result.all_mean_base if acq_result.all_mean_base is not None else acq_result.all_mean,
            acq_result.all_std,
            acq_result.all_ei,
            theta_best=self.database.get_theta_best(),
            pareto_points=np.vstack(pareto_points) if pareto_points else None,
        )
        candidate_infos = self._augment_rerank_candidates(candidate_infos)
        topm_candidates = select_topm_for_rerank(
            candidate_infos,
            top_m=int(self.cfg.get("llm_rerank_top_m", 5)),
            min_ei=float(self.cfg.get("llm_rerank_min_ei", 1e-10)),
        )
        rerank_state = self._build_rerank_state(t=t, w_vec=w_vec, scalar_y=scalar_y)
        gate_value = float(
            np.clip(
                self.cfg.get(
                    "llm_rerank_const_gate" if mode == "unsafe_legacy_const_gate" else "llm_rerank_gate",
                    0.25 if mode == "unsafe_legacy_const_gate" else 0.10,
                ),
                0.0,
                1.0,
            )
        )
        gate_state = {
            "g_value": gate_value,
            "mode": mode,
            "hv_gain_recent_mean": float(rerank_state.hv_gain_recent_mean),
            "violation_rate_recent": float(rerank_state.violation_rate_recent),
            "llm_uncertainty_recent": float(rerank_state.llm_uncertainty_recent),
        }

        llm_outputs = self.llm.score_candidate_goodness(rerank_state, topm_candidates)
        entropy_mean = float(np.mean([output.entropy() for output in llm_outputs])) if llm_outputs else None
        entropy_threshold = float(self.cfg.get("llm_rerank_entropy_threshold", 0.80))
        fail_open = self._rerank_fail_open(mode)
        if not llm_outputs and fail_open:
            self._last_rerank_summary = {
                "active": True,
                "applied": False,
                "llm_called": False,
                "tau_t": float(rerank_state.tau_t),
                "top_m": int(len(topm_candidates)),
                "gate": float(gate_value),
                "gate_state": gate_state,
                "entropy_mean": entropy_mean,
                "rerank_mode": mode,
                "parse_fail_open": bool(fail_open),
                "selected_indices_before": list(plain_selected_indices),
                "selected_indices_after": list(plain_selected_indices),
                "fallback_reason": "empty_llm_output",
                "rows": [],
                "eligible_indices": [],
                "topm_candidates": [candidate.to_dict() for candidate in topm_candidates],
            }
            return acq_result
        if entropy_mean is not None and entropy_mean > entropy_threshold and fail_open:
            self._last_rerank_summary = {
                "active": True,
                "applied": False,
                "llm_called": True,
                "tau_t": float(rerank_state.tau_t),
                "top_m": int(len(topm_candidates)),
                "gate": float(gate_value),
                "gate_state": gate_state,
                "entropy_mean": float(entropy_mean),
                "rerank_mode": mode,
                "parse_fail_open": bool(fail_open),
                "selected_indices_before": list(plain_selected_indices),
                "selected_indices_after": list(plain_selected_indices),
                "fallback_reason": "high_entropy",
                "rows": [],
                "eligible_indices": [],
                "topm_candidates": [candidate.to_dict() for candidate in topm_candidates],
            }
            return acq_result

        rerank_result = rerank_topm_with_llm(
            topm_candidates=topm_candidates,
            llm_outputs=llm_outputs,
            mode=mode,
            gate=float(gate_value),
            max_log_ei_gap=float(self.cfg.get("llm_rerank_max_log_ei_gap", 0.20)),
            max_bonus=float(self.cfg.get("llm_rerank_max_bonus", 0.05)),
            q_bad_threshold=float(self.cfg.get("llm_rerank_q_bad_threshold", 0.60)),
            min_confidence=float(self.cfg.get("llm_rerank_min_confidence", 0.50)),
            n_select=int(self.cfg.get("n_select", 1)),
            eps=float(self.cfg.get("llm_rerank_eps", 1e-12)),
        )
        if not rerank_result["selected_indices"]:
            self._last_rerank_summary = {
                "active": True,
                "applied": False,
                "llm_called": bool(llm_outputs),
                "tau_t": float(rerank_state.tau_t),
                "top_m": int(len(topm_candidates)),
                "gate": float(gate_value),
                "gate_state": gate_state,
                "entropy_mean": entropy_mean,
                "rerank_mode": mode,
                "parse_fail_open": bool(fail_open),
                "selected_indices_before": list(plain_selected_indices),
                "selected_indices_after": list(plain_selected_indices),
                "fallback_reason": rerank_result.get("fallback_reason", "empty_rerank_result"),
                "rows": [],
                "eligible_indices": list(rerank_result.get("eligible_indices", [])),
                "topm_candidates": [candidate.to_dict() for candidate in topm_candidates],
            }
            return acq_result

        selected_indices = [int(idx) for idx in rerank_result["selected_indices"]]
        acq_result.selected_indices = selected_indices
        acq_result.selected_thetas = [acq_result.candidate_pool[idx].copy() for idx in selected_indices]
        acq_result.selected_scores = np.asarray(rerank_result["selected_scores"], dtype=float)
        self._last_rerank_summary = {
            "active": True,
            "applied": True,
            "llm_called": True,
            "tau_t": float(rerank_state.tau_t),
            "top_m": int(len(topm_candidates)),
            "gate": float(gate_value),
            "gate_state": gate_state,
            "entropy_mean": rerank_result["entropy_mean"],
            "score_mode": mode,
            "rerank_mode": mode,
            "parse_fail_open": bool(fail_open),
            "selected_indices_before": list(plain_selected_indices),
            "selected_scores_before": np.asarray(plain_selected_scores, dtype=float).tolist(),
            "selected_indices_after": selected_indices,
            "selected_scores_after": np.asarray(acq_result.selected_scores, dtype=float).tolist(),
            "rows": rerank_result["rows"],
            "eligible_indices": list(rerank_result.get("eligible_indices", [])),
            "selected_changed": bool(list(plain_selected_indices) != selected_indices),
            "topm_candidates": [candidate.to_dict() for candidate in topm_candidates],
        }
        return acq_result

    def _guidance_local_sigma(self) -> np.ndarray:
        return np.maximum(
            self._estimate_search_sigma() * float(self.cfg.get("guidance_point_local_scale", 0.75)),
            np.array([0.08, 0.08, 0.04, 0.015, 0.015], dtype=float),
        )

    def _build_proposal_training_records(
        self,
        *,
        scalar_y: np.ndarray,
    ) -> List[ProposalTrainingRecord]:
        feasible = self.database.get_feasible()
        scalar = np.asarray(scalar_y, dtype=float).ravel()
        if len(feasible) != len(scalar):
            logger.warning(
                "Proposal record build skipped due to feasible/scalar mismatch: %d vs %d",
                len(feasible),
                len(scalar),
            )
            return []
        if len(feasible) == 0:
            return []

        elite_fraction = float(np.clip(self.cfg.get("proposal_elite_fraction", 0.35), 0.05, 0.95))
        scalar_threshold = float(np.quantile(scalar, elite_fraction))
        weight_eps = max(float(self.cfg.get("proposal_weight_epsilon", 1e-3)), 0.0)
        near_lambda = max(float(self.cfg.get("proposal_near_constraint_lambda", 8.0)), 0.0)
        monotone_lambda = max(float(self.cfg.get("proposal_monotone_penalty_lambda", 4.0)), 0.0)
        safe_limit = float(
            self.cfg.get(
                "proposal_safe_dsoc_sum_max",
                self.cfg.get("llm_safe_dsoc_sum_max", LLM_SAFE_DSOC_SUM_MAX),
            )
        )

        records: List[ProposalTrainingRecord] = []
        for obs, scalar_value in zip(feasible, scalar):
            theta = np.asarray(obs.theta, dtype=float).ravel()
            dsoc_sum = float(theta[3] + theta[4])
            near_penalty = max(0.0, dsoc_sum - safe_limit)
            monotone_penalty = max(0.0, float(theta[1] - theta[0])) + max(0.0, float(theta[2] - theta[1]))
            improvement = max(scalar_threshold - float(scalar_value), 0.0)
            weight = (weight_eps + improvement) * np.exp(
                -near_lambda * near_penalty - monotone_lambda * monotone_penalty
            )
            records.append(
                ProposalTrainingRecord(
                    theta=theta.copy(),
                    scalar_y=float(scalar_value),
                    improvement=float(improvement),
                    feasible=bool(obs.feasible),
                    near_constraint_penalty=float(near_penalty),
                    monotone_penalty=float(monotone_penalty),
                    source=str(obs.source),
                    iteration=int(obs.iteration),
                    weight=float(weight),
                )
            )
        return records

    def _sample_proposal_candidates(self, *, theta_best: np.ndarray) -> np.ndarray:
        if self.proposal is None or not self.proposal.is_ready():
            return np.empty((0, len(PARAM_KEYS)), dtype=float)
        n_samples = max(int(self.cfg.get("proposal_n_samples", 24)), 0)
        if n_samples <= 0:
            return np.empty((0, len(PARAM_KEYS)), dtype=float)
        samples = self.proposal.sample(
            n=n_samples,
            rng=self._rng,
            center=self._repair_theta(theta_best),
        )
        if samples.size == 0:
            return np.empty((0, len(PARAM_KEYS)), dtype=float)
        return np.vstack(self._deduplicate_points([self._repair_theta(row) for row in samples]))

    def _build_selective_history_summary(
        self,
        *,
        w_vec: np.ndarray,
        scalar_y: np.ndarray,
        proposal_summary: Optional[Dict[str, Any]],
    ) -> str:
        feasible = self.database.get_feasible()
        if not feasible:
            return "none"

        scalar = np.asarray(scalar_y, dtype=float).ravel()
        top_k = min(3, len(feasible))
        top_idx = np.argsort(scalar)[:top_k]
        safe_limit = float(
            self.cfg.get(
                "proposal_safe_dsoc_sum_max",
                self.cfg.get("llm_safe_dsoc_sum_max", LLM_SAFE_DSOC_SUM_MAX),
            )
        )
        hard_limit = float(DSOC_SUM_MAX)
        pf = self.database.get_pareto_front()
        guided_obs = [obs for obs in self.database.get_all() if obs.llm_rationale]
        recent_guided = guided_obs[-5:]
        guided_on_pf = 0
        for obs in recent_guided:
            if any(np.allclose(obs.theta, pf_obs.theta, atol=1e-6) for pf_obs in pf):
                guided_on_pf += 1

        all_obs = self.database.get_all()
        risky_recent = [
            obs for obs in reversed(all_obs)
            if (
                (not obs.feasible)
                or (obs.theta[3] + obs.theta[4] >= safe_limit - 1e-9)
                or (obs.theta[1] > obs.theta[0] + 1e-9)
                or (obs.theta[2] > obs.theta[1] + 1e-9)
            )
        ][:3]

        feasible_sums = np.array([obs.theta[3] + obs.theta[4] for obs in feasible], dtype=float)
        lines = [
            f"weight={np.round(np.asarray(w_vec, dtype=float), 4).tolist()}",
            "top_scalar_protocols:",
        ]
        for idx in top_idx:
            obs = feasible[idx]
            lines.append(
                "  "
                f"iter={obs.iteration} src={obs.source} "
                f"theta={np.round(obs.theta, 4).tolist()} "
                f"scalar={float(scalar[idx]):.6f}"
            )
        lines.append(
            "boundary_stats: "
            f"near_safe={int(np.sum(feasible_sums >= safe_limit - 1e-9))}/{len(feasible_sums)}, "
            f"near_hard={int(np.sum(feasible_sums >= hard_limit - 0.02))}/{len(feasible_sums)}"
        )
        lines.append(
            "guidance_effectiveness: "
            f"recent_guided={len(recent_guided)}, on_pareto={guided_on_pf}"
        )
        if proposal_summary:
            lines.append(
                "proposal_summary: "
                f"ready={proposal_summary.get('ready')} "
                f"components={proposal_summary.get('n_components', 0)} "
                f"elite={proposal_summary.get('elite_count', 0)}"
            )
        if risky_recent:
            lines.append("recent_risky_or_failed:")
            for obs in risky_recent:
                lines.append(
                    "  "
                    f"iter={obs.iteration} src={obs.source} feasible={obs.feasible} "
                    f"theta={np.round(obs.theta, 4).tolist()}"
                )
        return "\n".join(lines)

    def _build_scalarization_formula_text(self) -> str:
        eta = float(self.cfg.get("eta", 0.05))
        return (
            "Transform objectives with log10(time) and log10(aging), then compute normalized "
            "gaps to the current ideal point, and minimize "
            f"f_w = max_i(w_i * gap_i) + {eta:.3f} * sum_i(w_i * gap_i). "
            "Lower f_w is better under the current weight and normalization context."
        )

    def _build_top_scalar_protocols_summary(
        self,
        *,
        scalar_y: np.ndarray,
        top_k: int,
    ) -> str:
        feasible = self.database.get_feasible()
        scalar = np.asarray(scalar_y, dtype=float).ravel()
        if not feasible or len(feasible) != len(scalar):
            return "none"

        k = max(min(int(top_k), len(feasible)), 0)
        if k <= 0:
            return "none"

        idxs = np.argsort(scalar)[:k]
        lines: List[str] = []
        for idx in idxs:
            obs = feasible[int(idx)]
            lines.append(
                f"iter={obs.iteration} src={obs.source} "
                f"theta={np.round(obs.theta, 4).tolist()} "
                f"scalar={float(scalar[idx]):.6f}"
            )
        return "\n".join(lines) if lines else "none"

    def _compute_coupling_gate(
        self,
        *,
        t: int,
        w_vec: np.ndarray,
        guidance: IterationGuidance,
        guidance_candidates: np.ndarray,
    ) -> Dict[str, float]:
        max_iterations = max(int(self.cfg.get("max_iterations", 1)), 1)
        probe = np.atleast_2d(np.asarray(guidance_candidates, dtype=float))
        if probe.shape[0] > 8:
            probe = probe[:8]

        f_min = float(self.database.get_f_min())
        mean_probe, std_probe = self.gp.predict(probe)
        z = (f_min - mean_probe) / np.maximum(std_probe, 1e-6)
        align_score = float(np.mean(_stable_sigmoid(z))) if len(z) else 0.5

        history_info = self.database.get_similar_weight_guidance_stats(
            w_vec=w_vec,
            similarity_threshold=float(self.cfg.get("coupling_history_similarity_threshold", 0.85)),
            fallback_score=float(self.cfg.get("coupling_history_fallback_score", 0.75)),
        )
        history_score = float(np.clip(history_info["success_rate"], 0.0, 1.0))

        hv_info = self.database.get_hv_feedback_summary(window=3)
        hv_delta = float(hv_info["hv_delta_last_k"])
        stagnation = int(self.database.get_stagnation_count())
        stalled = 1.0 if (stagnation > 0 or hv_delta <= 1e-3) else 0.0
        hv_score = float(np.clip(0.55 + 0.40 * stalled, 0.0, 1.0))

        stage_ratio = max(0.0, 1.0 - (float(t) / float(max_iterations)))
        stage_score = float(np.clip(0.35 + 0.65 * stage_ratio, 0.0, 1.0))

        gate = float(np.clip(align_score * history_score * hv_score * stage_score, 0.0, 1.0))

        return {
            "gate": gate,
            "align_score": align_score,
            "history_score": history_score,
            "hv_score": hv_score,
            "stage_score": stage_score,
        }

    @staticmethod
    def _summarize_tagged_points(tagged_points: List[Tuple[str, np.ndarray]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for source, _ in tagged_points:
            counts[source] = counts.get(source, 0) + 1
        return counts

    def _get_random_init_points(self, n: int, seed: int = 0) -> List[np.ndarray]:
        cache_path = self.cfg.get("random_init_cache_path")
        if not cache_path:
            return self._lhs_candidates(n, seed=seed)

        path = Path(str(cache_path))
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                points = np.asarray(payload.get("points", payload), dtype=float)
                if points.ndim == 2 and points.shape[0] >= int(n) and points.shape[1] == len(PARAM_KEYS):
                    logger.info("Using cached random init points from %s", path)
                    return [self._repair_theta(row) for row in points[: int(n)]]
            except Exception as exc:
                logger.warning("Failed to read random init cache %s: %s", path, exc)

        points = self._lhs_candidates(n, seed=seed)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "method": "lhs",
                        "n": int(n),
                        "seed": int(seed),
                        "points": [np.asarray(theta, dtype=float).ravel().tolist() for theta in points],
                    },
                    f,
                    indent=2,
                )
            logger.info("Saved random init cache to %s", path)
        except Exception as exc:
            logger.warning("Failed to save random init cache %s: %s", path, exc)
        return points

    def _lhs_candidates(self, n: int, seed: int = 0) -> List[np.ndarray]:
        if n <= 0:
            return []
        lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        rng = np.random.default_rng(seed)
        samples = np.zeros((n, len(PARAM_KEYS)), dtype=float)
        intervals = np.linspace(0.0, 1.0, n + 1)
        for dim in range(len(PARAM_KEYS)):
            perm = rng.permutation(n)
            lower = intervals[perm]
            upper = intervals[perm + 1]
            samples[:, dim] = lower + rng.random(n) * (upper - lower)

        candidates = []
        for row in samples:
            theta = lo + row * (hi - lo)
            candidates.append(self.constraint_policy.repair_hard(theta, bounds=self.param_bounds))
        return candidates

    @staticmethod
    def _deduplicate_tagged_points(
        tagged_points: List[Tuple[str, np.ndarray]]
    ) -> List[Tuple[str, np.ndarray]]:
        deduped: List[Tuple[str, np.ndarray]] = []
        seen = set()
        for source, theta in tagged_points:
            key = tuple(np.round(np.asarray(theta, dtype=float).ravel(), 6))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((source, np.asarray(theta, dtype=float).ravel()))
        return deduped

    @staticmethod
    def _deduplicate_points(points: List[np.ndarray]) -> List[np.ndarray]:
        deduped: List[np.ndarray] = []
        seen = set()
        for theta in points:
            key = tuple(np.round(np.asarray(theta, dtype=float).ravel(), 6))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(np.asarray(theta, dtype=float).ravel())
        return deduped

    def _repair_theta(self, theta: np.ndarray) -> np.ndarray:
        return self.constraint_policy.repair_hard(theta, bounds=self.param_bounds)

    def _jsonable_config(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in self.cfg.items():
            if "api_key" in key.lower():
                result[key] = "<redacted>" if value else ""
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                result[key] = value
        return result

    def _record_hv_snapshot(
        self,
        *,
        phase: str,
        iteration: int,
        source: str,
        theta: np.ndarray,
        feasible: bool,
        elapsed_s: Optional[float] = None,
        acq_value: Optional[float] = None,
    ) -> None:
        if self.database is None:
            return

        hv_raw = self.database.compute_hypervolume_raw()
        hv_canonical = canonical_hv_from_raw(hv_raw, self.database.hv_max)
        hv_display = self.database.compute_hypervolume()
        snapshot = {
            "eval_index": self.database.size,
            "phase": phase,
            "iteration": int(iteration),
            "source": source,
            "theta": np.asarray(theta, dtype=float).ravel().tolist(),
            "feasible": bool(feasible),
            "hypervolume": hv_display,
            "display_hv": hv_display,
            "canonical_hv": hv_canonical,
            "hypervolume_canonical": hv_canonical,
            "hypervolume_raw": hv_raw,
            "pareto_size": self.database.pareto_size,
            "n_total": self.database.size,
            "n_feasible": self.database.n_feasible,
        }
        if elapsed_s is not None:
            snapshot["elapsed_s"] = float(elapsed_s)
        if acq_value is not None:
            snapshot["acq_value"] = float(acq_value)
        self._hv_eval_trace.append(snapshot)

    def get_hv_eval_trace(self) -> List[Dict[str, Any]]:
        return list(self._hv_eval_trace)
