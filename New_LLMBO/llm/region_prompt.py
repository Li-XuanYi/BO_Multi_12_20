from __future__ import annotations

import json
from typing import Any, Dict, Mapping

PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]


def _compact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key in (
            "theta",
            "point",
            "objectives",
            "raw_objectives",
            "scalar_y",
            "y",
            "score",
            "rank",
            "iteration",
            "selected_source",
            "fallback_reason",
            "suggestion_used",
            "actual_theta",
            "hv_gain_raw",
            "reason",
            "mechanistic_thinking",
            "source",
            "feasible",
        ):
            if key in value:
                out[key] = _compact_value(value[key])
        return out or {str(k): _compact_value(v) for k, v in list(value.items())[:8]}
    if isinstance(value, (list, tuple)):
        return [_compact_value(item) for item in list(value)[:8]]
    if isinstance(value, float):
        return round(float(value), 6)
    return value


def _compact_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    keep = {}
    for key in (
        "iteration",
        "max_iterations",
        "w_vec",
        "ideal_point_raw",
        "objective_preprocess_mode",
        "y_min",
        "y_max",
        "eta",
        "f_min",
        "hv_feedback",
        "boundary_failures",
        "previous_region_thinking",
        "previous_thinking",
        "last_region_adoption_note",
        "adoption_note",
        "best_theta",
        "best_objectives",
        "top_scalar_points",
        "recent_observations",
        "recent_points",
        "uncertainty_hotspots",
    ):
        if key in state:
            keep[key] = _compact_value(state[key])
    if "previous_region_thinking" in keep and "previous_thinking" not in keep:
        keep["previous_thinking"] = keep["previous_region_thinking"]
    if "last_region_adoption_note" in keep and "adoption_note" not in keep:
        keep["adoption_note"] = keep["last_region_adoption_note"]
    return keep


def render_region_preference_prompt(
    *,
    state: Mapping[str, Any],
    param_bounds: Mapping[str, tuple[float, float]],
    prompt_version: str = "default",
) -> str:
    parameters = [
        {
            "name": key,
            "lower": float(param_bounds[key][0]),
            "upper": float(param_bounds[key][1]),
            "unit": "A" if key.startswith("I") else "SOC fraction",
        }
        for key in PARAM_KEYS
    ]
    payload: Dict[str, Any] = {
        "parameter_bounds": parameters,
        "objective_schema": {
            "order": [
                "charge_time_seconds",
                "peak_temperature_rise_celsius",
                "aging_or_degradation_proxy",
            ],
            "direction": "all objectives are minimized",
            "applies_to": [
                "w_vec",
                "ideal_point_raw",
                "y_min",
                "y_max",
                "raw_objectives",
                "objectives",
            ],
        },
        "current_context": _compact_state(state),
    }
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    version = str(prompt_version or "default").strip().lower()
    calibrated_rules = ""
    if version == "calibrated":
        calibrated_rules = (
            "Calibrated standalone-region rules: only return a region when the mechanism is tied to the current "
            "w_vec and at least one recent or top observation supports the direction. If the reasoning is generic, "
            "prefer kind='none' or confidence below 0.50. Do not default to the 0.60-0.70 confidence band. "
            "Use confidence above 0.70 only when mechanism, w_vec emphasis, and observations agree. "
            "Region width policy: stronger evidence should produce narrower boxes, tentative evidence should produce "
            "wider low-confidence boxes or none; avoid broad boxes that cover much of the search space. "
            "Mechanistic thinking must explicitly name which objective weight is being served and why this region is "
            "expected to improve that scalarized objective.\n"
        )
    elif version in {"calibrated_v2", "adaptive"}:
        calibrated_rules = (
            "Calibrated-v2 standalone-region rules: return a point or region only when the mechanism is tied to the "
            "current w_vec and is at least weakly supported by recent_observations or top_scalar_points. Confidence "
            "must vary with evidence strength; do not default to a narrow middle confidence band. High confidence "
            "requires agreement between mechanism, current w_vec emphasis, and observations. A weak but plausible "
            "preference may still be returned with lower confidence instead of kind='none'. Region width policy: "
            "stronger evidence should produce tighter boxes, tentative evidence should produce wider lower-confidence "
            "boxes; avoid broad boxes that cover much of the search space. Mechanistic thinking must explicitly name "
            "which objective weight is being served. Previous thinking is weak context only; if the previous adoption "
            "note shows no improvement, do not mechanically repeat the same region without new supporting evidence.\n"
        )
    return (
        "Return exactly one JSON object and nothing else. Do not include analysis, markdown, or prose. "
        "Start with '{' and end with '}'.\n"
        "Task: provide one promising raw-coordinate point or region for a scalarized minimization objective. "
        "Lower scalarized objective is better. Do not directly choose the next experiment.\n"
        "Objective schema: w_vec and all raw objective arrays use this order: "
        "[charge_time_seconds, peak_temperature_rise_celsius, aging_or_degradation_proxy]. "
        "All three objectives are minimized.\n"
        "Use w_vec to infer the current scalarization emphasis. Prefer a region that is plausible for this "
        "specific weighted objective and that adds useful search signal beyond simply repeating historical best points.\n"
        "Evidence hierarchy: PRIMARY domain knowledge, physical mechanisms, constraints, and units; "
        "SECONDARY historical trial data as auxiliary evidence only. If mechanism implications conflict with "
        "historical points, side with the mechanism.\n"
        "Neutrality rules: do not assume any specific numeric parameter value is inherently good or bad because of this prompt. "
        "Do not use fixed numeric anchors, canned ranges, or copied examples as recommendations. "
        "Choose every numeric value from parameter_bounds, current_context, and your qualitative mechanism assessment.\n"
        "Anti-collapse: never center a region on a past observation unless mechanistically justified. "
        "If you reuse a past setting, explicitly state the mechanism that makes it promising.\n"
        "Previous thinking is weak context only: do not directly repeat the previous region unless new observations support it.\n"
        "Confidence calibration: use high confidence only when mechanism and observations agree, moderate confidence when "
        "mechanism is plausible but evidence is limited, and low confidence or kind='none' when the preference is weak.\n"
        f"{calibrated_rules}"
        "Hard constraints: dSOC1+dSOC2 < 0.70; all values must stay inside parameter_bounds. "
        "Use coordinate_space='raw' and preference_direction='promising'. "
        "For a region, use dictionaries lb and ub with all five parameter names; keep widths moderate, about 3%-35% of each range. "
        "If uncertainty is high, return kind='none' or a low-confidence point/region only when supported by current_context and mechanism.\n"
        "Include mechanistic_thinking as 1-2 short sentences about the mechanism, not step-by-step reasoning. "
        "Required output schemas, with numeric fields chosen by you rather than copied from this instruction:\n"
        "Point schema keys: kind='point', coordinate_space='raw', preference_direction='promising', "
        "point={I1:number,I2:number,I3:number,dSOC1:number,dSOC2:number}, confidence:number in [0,1], "
        "preference_type:string, reason:string, mechanistic_thinking:string.\n"
        "Region schema keys: kind='region', coordinate_space='raw', preference_direction='promising', "
        "lb={I1:number,I2:number,I3:number,dSOC1:number,dSOC2:number}, "
        "ub={I1:number,I2:number,I3:number,dSOC1:number,dSOC2:number}, confidence:number in [0,1], "
        "preference_type:string, reason:string, mechanistic_thinking:string.\n"
        f"Input data: {payload_json}\n"
        "JSON only:"
    )
