from __future__ import annotations

import math

from Ablation_Exp.Process.tools import analyze_ablation_suite as analysis


def test_exact_sign_flip_and_holm_are_deterministic() -> None:
    # With three positive paired differences, only the all-positive and
    # all-negative assignments are at least as extreme: 2 / 2**3 = 0.25.
    assert analysis.exact_sign_flip_p([1.0, 2.0, 3.0]) == 0.25
    assert analysis.exact_sign_flip_p([0.0, 0.0]) == 1.0
    assert analysis.holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_archived_groups_pass_integrity_and_use_canonical_hv() -> None:
    groups = {
        spec.key: analysis.analyse_group(spec, analysis.load_group(spec))
        for spec in analysis.GROUPS
    }
    assert all(group["integrity"]["passed"] for group in groups.values())

    prompt = groups["warmstart_prompt"]
    assert math.isclose(
        prompt["variant_summaries"]["experimental_prompt"]["canonical_hv"]["mean"],
        0.3689000178816712,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert prompt["expected_evaluations"] == 26


def test_confounds_and_shared_initialisation_are_detected() -> None:
    groups = {
        spec.key: analysis.analyse_group(spec, analysis.load_group(spec))
        for spec in analysis.GROUPS[:2]
    }

    primary = groups["same_batch_component_bundle"]
    region_vs_plain = analysis._comparison_by_key(primary, "region_vs_plain")
    assert region_vs_plain["initialization_requirement_met"] is True
    assert "ei_n_external_restarts" in region_vs_plain["control_config_mismatches"]

    paired = groups["shared_warmstart_region_increment"]
    region_increment = analysis._comparison_by_key(paired, "region_increment")
    assert region_increment["same_initialization_count"] == 5
    assert region_increment["same_initialization_total"] == 5
    assert "ei_n_external_restarts" in region_increment["control_config_mismatches"]


def test_safe_config_never_copies_api_credentials() -> None:
    sanitized = analysis._safe_config(
        {
            "llm_model": "example-model",
            "llm_api_key": "must-not-leak",
            "openai_api_key": "must-not-leak-either",
        }
    )
    assert sanitized["llm_model"] == "example-model"
    assert "llm_api_key" not in sanitized
    assert "openai_api_key" not in sanitized
