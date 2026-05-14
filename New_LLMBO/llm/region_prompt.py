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
            "scalar_y",
            "y",
            "score",
            "rank",
            "iteration",
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
        "best_theta",
        "best_objectives",
        "top_scalar_points",
        "recent_points",
        "uncertainty_hotspots",
    ):
        if key in state:
            keep[key] = _compact_value(state[key])
    return keep


def render_region_preference_prompt(
    *,
    state: Mapping[str, Any],
    param_bounds: Mapping[str, tuple[float, float]],
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
        "current_context": _compact_state(state),
    }
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return (
        "Return exactly one JSON object and nothing else. Do not include analysis, markdown, or prose. "
        "Start with '{' and end with '}'.\n"
        "Task: provide one promising raw-coordinate point or region for a scalarized minimization objective. "
        "Lower scalarized objective is better. Do not directly choose the next experiment.\n"
        "Hard constraints: dSOC1+dSOC2 < 0.70; all values must stay inside parameter_bounds. "
        "Use coordinate_space='raw' and preference_direction='promising'. "
        "For a region, use dictionaries lb and ub with all five parameter names; keep widths moderate, about 3%-35% of each range. "
        "If unsure, return kind='point' near the best trade-off; use kind='none' only if there is no defensible preference.\n"
        "Allowed output shapes:\n"
        "{\"kind\":\"point\",\"coordinate_space\":\"raw\",\"preference_direction\":\"promising\","
        "\"point\":{\"I1\":4.0,\"I2\":3.0,\"I3\":2.5,\"dSOC1\":0.2,\"dSOC2\":0.2},"
        "\"confidence\":0.7,\"preference_type\":\"balanced\",\"reason\":\"short\"}\n"
        "{\"kind\":\"region\",\"coordinate_space\":\"raw\",\"preference_direction\":\"promising\","
        "\"lb\":{\"I1\":3.8,\"I2\":2.8,\"I3\":2.2,\"dSOC1\":0.16,\"dSOC2\":0.14},"
        "\"ub\":{\"I1\":4.8,\"I2\":3.7,\"I3\":2.8,\"dSOC1\":0.26,\"dSOC2\":0.22},"
        "\"confidence\":0.7,\"preference_type\":\"balanced\",\"reason\":\"short\"}\n"
        f"Input data: {payload_json}\n"
        "JSON only:"
    )
