from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.schema import create_minimal_config
from main import build_optimizer_config


def test_build_optimizer_config_mainline_defaults_are_explicit() -> None:
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    args = argparse.Namespace(preset="warmstart_plain_ei", mock=True)
    flat = build_optimizer_config(cfg, args, Path("results"))

    assert flat["experiment_preset"] == "warmstart_plain_ei"
    assert flat["enable_iterative_guidance"] is False
    assert flat["enable_gp_llm_coupling"] is False
    assert flat["enable_acq_prior_coupling"] is False
    assert flat["enable_proposal_sampler"] is False
    assert flat["enable_llm_rerank"] is False
    assert flat["llm_rerank_mode"] == "none"
    assert flat["target_transform_mode"] == "none"
    assert flat["objective_preprocess_mode"] == "minmax"


def test_build_optimizer_config_risk_veto_preset_enables_safe_rerank() -> None:
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    args = argparse.Namespace(preset="warmstart_risk_veto", mock=True)
    flat = build_optimizer_config(cfg, args, Path("results"))

    assert flat["enable_llm_rerank"] is True
    assert flat["llm_rerank_mode"] == "risk_veto_only"
    assert flat["llm_rerank_top_m"] == 5
    assert flat["llm_rerank_parse_fail_open"] is True


def test_build_optimizer_config_parego_baseline_uses_explicit_das_dennis_weights() -> None:
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    args = argparse.Namespace(preset="parego_baseline", mock=True)
    flat = build_optimizer_config(cfg, args, Path("results"))

    assert flat["experiment_preset"] == "parego_baseline"
    assert flat["n_warmstart"] == 0
    assert flat["n_random_init"] == 6
    assert flat["enable_region_lifted_gp"] is False
    assert flat["enable_llm_rerank"] is False
    assert flat["weight_strategy"] == "parego_reference_cycle"
    assert flat["weight_count"] == 30
    assert flat["acquisition_strategy"] == "parego_lcb_de"
    assert flat["parego_lcb_variance_weight"] == 0.5
    assert flat["parego_de_population"] == 30
    assert flat["parego_de_maxiter"] == 200


def test_build_optimizer_config_region_lifted_preset_enables_region_lift() -> None:
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    args = argparse.Namespace(preset="warmstart_region_lifted_gp", mock=True)
    flat = build_optimizer_config(cfg, args, Path("results"))

    assert flat["enable_region_lifted_gp"] is True
    assert flat["target_transform_mode"] == "none"
    assert flat["objective_preprocess_mode"] == "minmax"
    assert flat["region_lift_external_influence_mode"] == "diagnostic_only"
    assert flat["region_lift_active_until"] >= 10
    assert flat["region_lift_anchor_weighting"] == "ei_softmax"
    assert flat["region_lift_require_inside"] is True
    assert flat["region_lift_apply_override"] is False
    assert flat["ei_n_external_restarts"] >= 4
    assert flat["region_lift_dsoc_margin"] > 0.0
    assert flat["enable_iterative_guidance"] is False
    assert flat["enable_gp_llm_coupling"] is False
    assert flat["enable_llm_rerank"] is False


def test_build_optimizer_config_region_lifted_guarded_pool_preset_enables_guarded_influence() -> None:
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    args = argparse.Namespace(preset="warmstart_region_lifted_gp_guarded_pool", mock=True)
    flat = build_optimizer_config(cfg, args, Path("results"))

    assert flat["enable_region_lifted_gp"] is True
    assert flat["region_lift_external_influence_mode"] == "guarded_pool"
    assert flat["region_lift_guard_min_anchor_consistency"] == 0.35
    assert flat["region_lift_guard_min_reliability"] == 0.20


def test_build_optimizer_config_region_lifted_force_pool_tuned_preset_uses_tighter_point_region() -> None:
    cfg = create_minimal_config(n_iterations=5, n_warmstart=3, n_candidates=5)
    args = argparse.Namespace(preset="warmstart_region_lifted_gp_force_pool_tuned", mock=True)
    flat = build_optimizer_config(cfg, args, Path("results"))

    assert flat["enable_region_lifted_gp"] is True
    assert flat["region_lift_external_influence_mode"] == "force_pool"
    assert flat["region_lift_include_raw_candidates"] is False
    assert flat["region_lift_active_until"] == 16
    assert flat["region_lift_n_anchors"] == 64
    assert flat["region_lift_candidate_oversample"] == 16
    assert flat["region_lift_point_current_probe_levels"] == 3
    assert flat["region_lift_point_current_probe_keep"] == 2
    assert flat["ei_n_external_restarts"] == 32
    assert flat["region_lift_min_width"] == 0.03
    assert flat["region_lift_min_volume"] == 1e-8
    assert flat["region_lift_max_volume"] == 0.08
    assert flat["region_lift_close_distance"] == 0.03
    assert flat["region_lift_dsoc_margin"] == 0.01
