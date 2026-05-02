from __future__ import annotations

import json
from typing import Any, Dict, Mapping

PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]


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
        "task": (
            "Return exactly one promising raw-coordinate point or region for the current "
            "scalarized minimization objective. Do not directly choose the next experiment."
        ),
        "rules": [
            "Lower scalarized objective is better under the current weight vector.",
            "Return a promising region only; do not return avoid-only regions.",
            "If no defensible promising region exists, return kind='none'.",
            "0.70 is the hard dSOC1+dSOC2 feasibility limit.",
            "0.65 is a soft safety margin, not a hard simulator constraint.",
            "I1>=I2>=I3 is a soft preference only.",
            "Use coordinate_space='raw' and parameter-name dictionaries, not arrays.",
            "Avoid degenerate boxes: never use identical lower/upper bounds on a dimension unless you return kind='point'.",
            "Prefer kind='point' over an extremely tiny region.",
            "If you return kind='region', keep it moderate rather than razor-thin; roughly 3%-35% of each parameter range is a good target unless strongly justified.",
        ],
        "parameters": parameters,
        "current_context": dict(state),
        "output_schema": {
            "kind": "point | region | none",
            "coordinate_space": "raw",
            "preference_direction": "promising",
            "point": {"I1": None, "I2": None, "I3": None, "dSOC1": None, "dSOC2": None},
            "lb": {"I1": None, "I2": None, "I3": None, "dSOC1": None, "dSOC2": None},
            "ub": {"I1": None, "I2": None, "I3": None, "dSOC1": None, "dSOC2": None},
            "confidence": "float in [0,1]",
            "preference_type": "balanced | fast_charge | thermal_safe | aging_safe | boundary_probe",
            "reason": "short rationale",
            "risk_flags": ["optional strings"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
