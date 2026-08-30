"""Experiment presets for BayesOptimizer.

Separated from llmbo/optimizer.py to avoid heavy import chain (sklearn, etc.).
"""
from __future__ import annotations

EXPERIMENT_PRESETS: dict[str, dict] = {
    "warmstart_plain_ei": {
        "n_warmstart": 3,
        "n_random_init": 3,
        "enable_warmstart_portfolio": True,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": False,
        "target_transform_mode": "none",
    },
    "warmstart_portfolio_plain_ei": {
        "n_warmstart": 3,
        "n_random_init": 3,
        "enable_warmstart_portfolio": True,
        "warmstart_pool_size": 16,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": False,
        "target_transform_mode": "none",
    },
    "strict_baseline": {
        "n_warmstart": 0,
        "n_random_init": 6,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": False,
        "target_transform_mode": "none",
    },
    "parego_baseline": {
        "n_warmstart": 0,
        "n_random_init": 6,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": False,
        "target_transform_mode": "none",
        "weight_strategy": "parego_reference_cycle",
        "weight_count": 30,
        "acquisition_strategy": "parego_lcb_de",
        "parego_lcb_variance_weight": 0.5,
        "parego_de_population": 30,
        "parego_de_maxiter": 200,
    },
    "parego_matlab_reference": {
        "n_warmstart": 0,
        "n_random_init": 6,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": False,
        "target_transform_mode": "none",
        "weight_strategy": "parego_das_dennis_cycle",
        "weight_simplex_divisions": 30,
        "weight_eps_min": 1e-6,
        "weight_sampling_mode": "random_with_replacement",
        "scalarization_mode": "parego_reference",
        "acquisition_strategy": "parego_lcb_de",
        "parego_lcb_variance_weight": 0.5,
        "parego_de_population": 30,
        "parego_de_maxiter": 200,
        "parego_invert_weights": True,
        "parego_use_model_standardized_lcb": True,
    },
    "warmstart_safe_tiebreak": {
        "n_warmstart": 3,
        "n_random_init": 3,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": True,
        "llm_rerank_mode": "ei_preserving_tiebreak",
        "enable_region_lifted_gp": False,
        "target_transform_mode": "none",
    },
    "warmstart_risk_veto": {
        "n_warmstart": 3,
        "n_random_init": 3,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": True,
        "llm_rerank_mode": "risk_veto_only",
        "enable_region_lifted_gp": False,
        "target_transform_mode": "none",
    },
    "warmstart_region_lifted_gp": {
        "n_warmstart": 3,
        "n_random_init": 3,
        "enable_warmstart_portfolio": True,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": True,
        "target_transform_mode": "none",
        "region_lift_apply_override": False,
        "region_lift_external_influence_mode": "diagnostic_only",
        "region_lift_include_raw_candidates": True,
        "region_lift_lambda_max": 0.20,
        "region_lift_n_anchors": 32,
        "region_lift_active_until": 12,
        "region_lift_max_plain_ei_gap": 0.25,
        "region_lift_min_volume": 1e-5,
        "region_lift_min_width": 0.03,
        "region_lift_trust_init": 0.7,
        "region_lift_anchor_weighting": "ei_softmax",
        "region_lift_anchor_temperature": 0.35,
        "region_lift_require_inside": True,
        "region_lift_min_sigma_ratio": 0.85,
        "region_lift_candidate_oversample": 8,
        "region_lift_point_current_probe_levels": 0,
        "region_lift_point_current_probe_keep": 0,
        "region_lift_dsoc_margin": 0.02,
        "ei_n_external_restarts": 16,
    },
    "warmstart_region_lifted_gp_guarded_pool": {
        "n_warmstart": 3,
        "n_random_init": 3,
        "enable_warmstart_portfolio": True,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": True,
        "target_transform_mode": "none",
        "region_lift_apply_override": False,
        "region_lift_external_influence_mode": "guarded_pool",
        "region_lift_include_raw_candidates": True,
        "region_lift_lambda_max": 0.20,
        "region_lift_n_anchors": 32,
        "region_lift_active_until": 12,
        "region_lift_max_plain_ei_gap": 0.25,
        "region_lift_min_volume": 1e-5,
        "region_lift_min_width": 0.03,
        "region_lift_trust_init": 0.7,
        "region_lift_anchor_weighting": "ei_softmax",
        "region_lift_anchor_temperature": 0.35,
        "region_lift_require_inside": True,
        "region_lift_min_sigma_ratio": 0.85,
        "region_lift_candidate_oversample": 8,
        "region_lift_point_current_probe_levels": 0,
        "region_lift_point_current_probe_keep": 0,
        "region_lift_dsoc_margin": 0.02,
        "ei_n_external_restarts": 16,
    },
    "warmstart_region_lifted_gp_force_pool_tuned": {
        "n_warmstart": 3,
        "n_random_init": 3,
        "enable_warmstart_portfolio": True,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "enable_region_lifted_gp": True,
        "target_transform_mode": "none",
        "region_lift_apply_override": False,
        "region_lift_external_influence_mode": "force_pool",
        "region_lift_include_raw_candidates": False,
        "region_lift_lambda_max": 0.20,
        "region_lift_n_anchors": 64,
        "region_lift_active_until": 16,
        "region_lift_max_plain_ei_gap": 0.25,
        "region_lift_min_volume": 1e-8,
        "region_lift_max_volume": 0.08,
        "region_lift_min_width": 0.03,
        "region_lift_trust_init": 0.7,
        "region_lift_anchor_weighting": "ei_softmax",
        "region_lift_anchor_temperature": 0.35,
        "region_lift_require_inside": True,
        "region_lift_min_sigma_ratio": 0.85,
        "region_lift_candidate_oversample": 16,
        "region_lift_point_current_probe_levels": 3,
        "region_lift_point_current_probe_keep": 2,
        "region_lift_close_distance": 0.03,
        "region_lift_dsoc_margin": 0.01,
        "ei_n_external_restarts": 32,
    },
}


# Corrected Proposition-1 Region-Lift path.  It mirrors warmstart_plain_ei's
# initialization and EI budget so the Region ablation changes only the region
# information path, not the amount of optimizer search.
EXPERIMENT_PRESETS["warmstart_region_lgbo_proposition1"] = {
    **EXPERIMENT_PRESETS["warmstart_plain_ei"],
    "enable_region_lifted_gp": True,
    "region_lift_mode": "lgbo_proposition1",
    "region_lift_control_mode": "none",
    "region_lift_lgbo_shift_source": "posterior_covariance",
    "region_lift_external_influence_mode": "diagnostic_only",
    "region_lift_include_raw_candidates": False,
    "region_lift_apply_override": False,
    "region_lift_anchor_weighting": "uniform",
    "region_lift_n_anchors": 32,
    "region_lift_lgbo_min_variance": 1e-10,
    "region_lift_min_confidence": 0.60,
    "region_lift_hard_confidence_gate": True,
    "region_lift_confidence_scale": 0.25,
    "region_lift_active_until": 12,
    "region_lift_anneal": "linear_decay",
    "region_lift_lgbo_apply_anneal": True,
    "region_lift_lgbo_shift_mean_budget": 0.05,
    "region_lift_lgbo_max_shift_std": 0.15,
    "region_lift_max_plain_ei_gap": 0.25,
    "region_lift_require_inside": True,
    "region_lift_near_region_tol": 0.02,
    "region_lift_min_sigma_ratio": 0.50,
    "region_lift_trust_beta": 0.0,
    "ei_n_external_restarts": 16,
}

# Random-region negative control with the same posterior-covariance mechanism.
EXPERIMENT_PRESETS["random_region_lgbo_proposition1"] = {
    **EXPERIMENT_PRESETS["warmstart_region_lgbo_proposition1"],
    "region_lift_control_mode": "fixed_random",
    "region_lift_random_width_norm": 0.15,
    "region_lift_random_confidence": 0.5,
    "region_lift_hard_confidence_gate": False,
}

# Stronger semantic sham: call the LLM exactly as in the corrected Region arm,
# but deterministically relocate its box while preserving box shape, confidence,
# call count, posterior-covariance mechanism, and optimizer budget.
EXPERIMENT_PRESETS["sham_region_lgbo_proposition1"] = {
    **EXPERIMENT_PRESETS["warmstart_region_lgbo_proposition1"],
    "region_lift_control_mode": "shape_randomized",
}

# Historical LGBO names are retained for exact experiment-script compatibility.
# They remain soft-confidence arms; the guarded 3+3 arm above is the corrected
# preset intended for new Region-vs-Plain ablations.
EXPERIMENT_PRESETS["llm_region_lgbo_prior"] = {
    **EXPERIMENT_PRESETS["warmstart_region_lifted_gp_force_pool_tuned"],
    "n_warmstart": 0,
    "n_random_init": 6,
    "region_lift_mode": "lgbo_proposition1",
    "region_lift_control_mode": "none",
    "region_lift_lgbo_shift_source": "prior_kernel",
    "region_lift_apply_override": False,
    "region_lift_external_influence_mode": "diagnostic_only",
    "region_lift_include_raw_candidates": False,
    "region_lift_anchor_weighting": "uniform",
    "region_lift_lgbo_min_variance": 1e-12,
    "region_lift_trust_beta": 0.0,
    "region_lift_point_current_probe_levels": 0,
    "region_lift_point_current_probe_keep": 0,
    "region_lift_hard_confidence_gate": False,
    "region_lift_lgbo_apply_anneal": False,
    "region_lift_lgbo_max_shift_std": 0.0,
}

EXPERIMENT_PRESETS["llm_region_lgbo_posterior"] = {
    **EXPERIMENT_PRESETS["llm_region_lgbo_prior"],
    "region_lift_lgbo_shift_source": "posterior_covariance",
}

EXPERIMENT_PRESETS["llm_region_lgbo_posterior_calibrated"] = {
    **EXPERIMENT_PRESETS["llm_region_lgbo_posterior"],
    "region_preference_prompt_version": "calibrated",
    "region_lift_confidence_scale": 0.75,
    "region_lift_max_width": 0.55,
    "region_lift_active_until": 8,
}

EXPERIMENT_PRESETS["llm_region_lgbo_posterior_adaptive"] = {
    **EXPERIMENT_PRESETS["llm_region_lgbo_posterior_calibrated"],
    "region_preference_prompt_version": "calibrated_v2",
    "region_lift_confidence_scale": 1.0,
    "region_lift_active_until": 12,
    "region_lift_adaptive_confidence_enabled": True,
    "region_lift_adaptive_confidence_floor": 0.35,
    "region_lift_adaptive_base_scale": 0.85,
    "region_lift_adaptive_width_min_factor": 0.80,
    "region_lift_adaptive_repeat_min_factor": 0.85,
    "region_lift_adaptive_late_min_factor": 0.85,
    "region_lift_adaptive_width_start": 0.30,
    "region_lift_adaptive_repeat_distance": 0.18,
    "region_lift_lgbo_shift_mean_budget": 0.025,
}

EXPERIMENT_PRESETS["warmstart_lgbo_prior"] = {
    **EXPERIMENT_PRESETS["llm_region_lgbo_prior"],
    "n_warmstart": 6,
    "n_random_init": 0,
}
EXPERIMENT_PRESETS["warmstart_lgbo_posterior"] = {
    **EXPERIMENT_PRESETS["warmstart_lgbo_prior"],
    "region_lift_lgbo_shift_source": "posterior_covariance",
}

EXPERIMENT_PRESETS["random_region_lgbo_prior"] = {
    **EXPERIMENT_PRESETS["llm_region_lgbo_prior"],
    "region_lift_control_mode": "fixed_random",
    "region_lift_random_width_norm": 0.15,
    "region_lift_random_confidence": 0.5,
    "region_lift_lgbo_shift_source": "prior_kernel",
}
EXPERIMENT_PRESETS["random_region_lgbo_posterior"] = {
    **EXPERIMENT_PRESETS["random_region_lgbo_prior"],
    "region_lift_lgbo_shift_source": "posterior_covariance",
}

# Historical aliases used by older runners.  They intentionally keep their
# historical 6+0 initialization; use warmstart_region_lgbo_proposition1 for the
# corrected same-budget Region-vs-Plain ablation.
EXPERIMENT_PRESETS["llmbo_mo"] = {
    **EXPERIMENT_PRESETS["warmstart_lgbo_posterior"],
}
EXPERIMENT_PRESETS["LLMBO-MO"] = {
    **EXPERIMENT_PRESETS["warmstart_lgbo_posterior"],
}

EXPERIMENT_PRESETS["baseline_plain_ei"] = {
    **EXPERIMENT_PRESETS["strict_baseline"],
}
