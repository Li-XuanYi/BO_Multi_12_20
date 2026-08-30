from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.region_prompt import render_region_preference_prompt
from utils.constants import DEFAULT_BOUNDS


def _extract_input_payload(prompt: str) -> dict:
    marker = "Input data: "
    start = prompt.index(marker) + len(marker)
    end = prompt.index("\nJSON only:", start)
    return json.loads(prompt[start:end])


def test_region_prompt_preserves_recent_observations_and_raw_objectives() -> None:
    state = {
        "iteration": 3,
        "w_vec": [0.5, 0.3, 0.2],
        "ideal_point_raw": [3200.0, 3.6, 0.6],
        "objective_preprocess_mode": "minmax",
        "y_min": [3000.0, 3.0, 0.4],
        "y_max": [5000.0, 5.5, 1.0],
        "eta": 0.05,
        "f_min": 0.12,
        "hv_feedback": {"summary": "current_hv=0.42"},
        "boundary_failures": {"summary": "recent_failures=1/5"},
        "previous_region_thinking": "Higher I1 with moderate dSOC can improve time without excessive boundary pressure.",
        "last_region_adoption_note": {
            "iteration": 2,
            "suggestion_used": True,
            "selected_source": "lgbo_lifted_gp",
            "fallback_reason": None,
            "actual_theta": [4.6, 3.8, 2.7, 0.2, 0.17],
            "hv_gain_raw": 0.03,
        },
        "top_scalar_points": [
            {
                "theta": [4.8, 4.0, 2.8, 0.22, 0.18],
                "raw_objectives": [3600.0, 4.0, 0.7],
                "scalar_y": 0.12,
            }
        ],
        "recent_observations": [
            {
                "theta": [4.6, 3.8, 2.7, 0.2, 0.17],
                "objectives": [3700.0, 3.9, 0.65],
                "feasible": True,
                "source": "bo",
            }
        ],
    }

    prompt = render_region_preference_prompt(state=state, param_bounds=DEFAULT_BOUNDS)
    payload = _extract_input_payload(prompt)
    context = payload["current_context"]

    assert payload["objective_schema"]["order"] == [
        "charge_time_seconds",
        "peak_temperature_rise_celsius",
        "aging_or_degradation_proxy",
    ]
    assert payload["objective_schema"]["direction"] == "all objectives are minimized"
    assert "Objective schema" in prompt
    assert "charge_time_seconds" in prompt
    assert "peak_temperature_rise_celsius" in prompt
    assert "aging_or_degradation_proxy" in prompt
    assert context["top_scalar_points"][0]["raw_objectives"] == [3600.0, 4.0, 0.7]
    assert context["recent_observations"][0]["source"] == "bo"
    assert context["recent_observations"][0]["objectives"] == [3700.0, 3.9, 0.65]
    assert context["hv_feedback"]["summary"] == "current_hv=0.42"
    assert context["boundary_failures"]["summary"] == "recent_failures=1/5"
    assert context["previous_region_thinking"].startswith("Higher I1")
    assert context["previous_thinking"].startswith("Higher I1")
    assert context["last_region_adoption_note"]["suggestion_used"] is True
    assert context["last_region_adoption_note"]["actual_theta"] == [4.6, 3.8, 2.7, 0.2, 0.17]
    assert context["adoption_note"]["suggestion_used"] is True
    assert "mechanistic_thinking" in prompt
    assert "Evidence hierarchy" in prompt
    assert "PRIMARY domain knowledge" in prompt
    assert "Neutrality rules" in prompt
    assert "do not assume any specific numeric parameter value is inherently good or bad" in prompt
    assert "Anti-collapse" in prompt
    assert "never center a region on a past observation" in prompt
    assert "Confidence calibration" in prompt
    assert "Previous thinking is weak context only" in prompt
    assert '"point":{"I1":4.0' not in prompt
    assert '"lb":{"I1":3.8' not in prompt


def test_region_prompt_calibrated_version_adds_non_anchoring_confidence_rules() -> None:
    prompt = render_region_preference_prompt(
        state={"iteration": 1, "w_vec": [0.7, 0.2, 0.1], "recent_observations": []},
        param_bounds=DEFAULT_BOUNDS,
        prompt_version="calibrated",
    )

    assert "Calibrated standalone-region rules" in prompt
    assert "Do not default to the 0.60-0.70 confidence band" in prompt
    assert "w_vec" in prompt
    assert '"point":{"I1":4.0' not in prompt
    assert '"lb":{"I1":3.8' not in prompt


def test_region_prompt_calibrated_v2_adds_soft_adaptive_rules_without_parameter_anchors() -> None:
    prompt = render_region_preference_prompt(
        state={
            "iteration": 2,
            "w_vec": [0.2, 0.7, 0.1],
            "last_region_adoption_note": {"hv_gain_raw": -0.01},
        },
        param_bounds=DEFAULT_BOUNDS,
        prompt_version="calibrated_v2",
    )

    assert "Calibrated-v2 standalone-region rules" in prompt
    assert "weak but plausible preference may still be returned with lower confidence" in prompt
    assert "do not mechanically repeat the same region" in prompt
    assert "which objective weight is being served" in prompt
    assert '"point":{"I1":4.0' not in prompt
    assert '"lb":{"I1":3.8' not in prompt
