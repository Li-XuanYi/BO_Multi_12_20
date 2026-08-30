"""
Iteration-guidance prompt templates and rendering utilities.

This keeps the iterative guidance prompt out of llm_interface.py so the
architecture stays aligned with warm-start prompting:
  1. Text template on disk
  2. Context builder that formats runtime state
  3. Small renderer that validates placeholder resolution
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from utils.constants import IDEAL_POINT as CANONICAL_IDEAL_POINT, LLM_SAFE_DSOC_SUM_MAX

try:
    from llm.warmstart_prompt import PLACEHOLDER_PATTERN
except ModuleNotFoundError:  # pragma: no cover - allows direct script execution
    from warmstart_prompt import PLACEHOLDER_PATTERN

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "iteration_guidance.txt"
PARAM_KEYS = ("I1", "I2", "I3", "dSOC1", "dSOC2")


class IterationPromptTemplateRenderer:
    """Load and render the iteration guidance template with [PLACEHOLDER] tokens."""

    def __init__(self, template_path: Optional[Path] = None):
        self._template_path = Path(template_path or DEFAULT_TEMPLATE_PATH)
        self._cache: Optional[str] = None

    def load(self) -> str:
        if self._cache is None:
            if not self._template_path.exists():
                raise FileNotFoundError(f"Iteration guidance template not found: {self._template_path}")
            self._cache = self._template_path.read_text(encoding="utf-8")
        return self._cache

    def render(self, context: Mapping[str, str]) -> str:
        rendered = self.load()
        for key, value in context.items():
            rendered = rendered.replace(f"[{key}]", str(value))

        leftovers = PLACEHOLDER_PATTERN.findall(rendered)
        if leftovers:
            missing = ", ".join(sorted(set(leftovers)))
            raise ValueError(
                f"Unresolved placeholders remain in iteration guidance template: {missing}"
            )
        return rendered


@dataclass(frozen=True)
class IterationPromptContextBuilder:
    param_bounds: Dict[str, Tuple[float, float]]
    battery_name: Optional[str] = None
    safe_dsoc_sum_max: float = LLM_SAFE_DSOC_SUM_MAX
    hard_dsoc_sum_max: float = 0.70

    def build(self, state_dict: Mapping[str, Any], pareto_context: str) -> Dict[str, str]:
        w = np.asarray(state_dict.get("w_vec", [1 / 3, 1 / 3, 1 / 3]), dtype=float)
        best = np.asarray(
            state_dict.get("theta_best", [4.0, 3.5, 2.5, 0.25, 0.20]),
            dtype=float,
        )
        ideal = np.asarray(
            state_dict.get("ideal_point", CANONICAL_IDEAL_POINT.tolist()),
            dtype=float,
        )
        y_min = np.asarray(state_dict.get("y_min", [0.0, 0.0, 0.0]), dtype=float)
        y_max = np.asarray(state_dict.get("y_max", [1.0, 1.0, 1.0]), dtype=float)

        hotspots_block = self._format_hotspots(state_dict.get("uncertainty_hotspots", []))
        previous_guidance = self._json_or_none(state_dict.get("previous_guidance"))
        proposal_summary = self._json_or_none(state_dict.get("proposal_summary"))

        current_hv = float(state_dict.get("current_hv", 0.0))
        hv_delta_last_3 = float(state_dict.get("hv_delta_last_3", 0.0))
        pareto_size = int(state_dict.get("pareto_size", 0))

        return {
            "BATTERY_NAME": str(self.battery_name or "LG INR21700-M50"),
            "I1_MIN": f"{self.param_bounds['I1'][0]}",
            "I1_MAX": f"{self.param_bounds['I1'][1]}",
            "I2_MIN": f"{self.param_bounds['I2'][0]}",
            "I2_MAX": f"{self.param_bounds['I2'][1]}",
            "I3_MIN": f"{self.param_bounds['I3'][0]}",
            "I3_MAX": f"{self.param_bounds['I3'][1]}",
            "DSOC1_MIN": f"{self.param_bounds['dSOC1'][0]}",
            "DSOC1_MAX": f"{self.param_bounds['dSOC1'][1]}",
            "DSOC2_MIN": f"{self.param_bounds['dSOC2'][0]}",
            "DSOC2_MAX": f"{self.param_bounds['dSOC2'][1]}",
            "DSOC_HARD_LIMIT": f"{float(self.hard_dsoc_sum_max):.2f}",
            "DSOC_SAFE_LIMIT": f"{float(self.safe_dsoc_sum_max):.2f}",
            "ITERATION": str(int(state_dict.get("iteration", 0))),
            "MAX_ITERATIONS": str(int(state_dict.get("max_iterations", 50))),
            "WEIGHT_VECTOR": (
                f"[time={w[0]:.3f}, temp={w[1]:.3f}, aging={w[2]:.3f}]"
            ),
            "FOCUS_TEXT": self._focus_text(w),
            "SCALARIZATION_FORMULA": str(
                state_dict.get(
                    "scalarization_formula",
                    "Lower scalarized objective is better.",
                )
                or "Lower scalarized objective is better."
            ),
            "IDEAL_POINT": f"[time={ideal[0]:.2f}, temp={ideal[1]:.2f}, aging={ideal[2]:.6f}]",
            "Y_MIN_TILDE": f"[{y_min[0]:.4f}, {y_min[1]:.4f}, {y_min[2]:.4f}]",
            "Y_MAX_TILDE": f"[{y_max[0]:.4f}, {y_max[1]:.4f}, {y_max[2]:.4f}]",
            "ETA_VALUE": f"{float(state_dict.get('eta', 0.05)):.3f}",
            "F_MIN": f"{float(state_dict.get('f_min', 0.5)):.6f}",
            "BEST_PROTOCOL": self._format_protocol(best),
            "STAGNATION_COUNT": str(int(state_dict.get("stagnation_count", 0))),
            "CURRENT_HV": f"{current_hv:.6f}",
            "HV_DELTA_LAST_3": f"{hv_delta_last_3:.6f}",
            "PARETO_SIZE": str(pareto_size),
            "HV_FEEDBACK_SUMMARY": str(
                state_dict.get("hv_feedback_summary", "none") or "none"
            ),
            "PREVIOUS_GUIDANCE": previous_guidance,
            "TOP_SCALAR_PROTOCOLS": str(
                state_dict.get("top_scalar_protocols", "none") or "none"
            ),
            "SIMILAR_WEIGHT_GUIDANCE_SUCCESS": str(
                state_dict.get("similar_weight_guidance_success", "none") or "none"
            ),
            "BOUNDARY_FAILURE_STATS": str(
                state_dict.get("boundary_failure_stats", "none") or "none"
            ),
            "SELECTIVE_HISTORY_SUMMARY": str(
                state_dict.get("selective_history_summary", "none") or "none"
            ),
            "PROPOSAL_SUMMARY": proposal_summary,
            "HOTSPOTS_BLOCK": hotspots_block,
            "SENSITIVITY_SUMMARY": str(state_dict.get("sensitivity_summary", "none") or "none"),
            "PARETO_CONTEXT": str(pareto_context or "none"),
        }

    @staticmethod
    def _format_protocol(theta: np.ndarray) -> str:
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != len(PARAM_KEYS):
            return "none"
        return (
            f"[{theta[0]:.2f}, {theta[1]:.2f}, {theta[2]:.2f}, "
            f"{theta[3]:.3f}, {theta[4]:.3f}]"
        )

    @staticmethod
    def _json_or_none(value: Any) -> str:
        if not value:
            return "none"
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    @staticmethod
    def _focus_text(w: np.ndarray) -> str:
        focus_idx = int(np.argmax(w))
        return {
            0: "Prioritize faster charging time while respecting thermal and aging constraints.",
            1: "Prioritize lower peak temperature even if charging time becomes longer.",
            2: "Prioritize lower aging and gentler late-stage charging.",
        }[focus_idx]

    @staticmethod
    def _format_hotspots(hotspots: List[Mapping[str, Any]]) -> str:
        lines: List[str] = []
        for idx, hotspot in enumerate(hotspots[:5]):
            theta = np.asarray(hotspot.get("theta", []), dtype=float).ravel()
            if theta.size != len(PARAM_KEYS):
                continue
            lines.append(
                f"  hotspot[{idx}] std={float(hotspot.get('std', 0.0)):.4f} "
                f"theta=[{theta[0]:.2f}, {theta[1]:.2f}, {theta[2]:.2f}, {theta[3]:.3f}, {theta[4]:.3f}]"
            )
        return "\n".join(lines) if lines else "  none"


def render_iteration_guidance_prompt(
    state_dict: Mapping[str, Any],
    param_bounds: Dict[str, Tuple[float, float]],
    pareto_context: str,
    *,
    battery_name: Optional[str] = None,
    safe_dsoc_sum_max: float = LLM_SAFE_DSOC_SUM_MAX,
    hard_dsoc_sum_max: float = 0.70,
    template_path: Optional[Path] = None,
) -> str:
    builder = IterationPromptContextBuilder(
        param_bounds=param_bounds,
        battery_name=battery_name,
        safe_dsoc_sum_max=safe_dsoc_sum_max,
        hard_dsoc_sum_max=hard_dsoc_sum_max,
    )
    context = builder.build(state_dict, pareto_context)
    renderer = IterationPromptTemplateRenderer(template_path=template_path)
    prompt = renderer.render(context)
    logger.debug("Iteration guidance prompt rendered (%d chars)", len(prompt))
    return prompt
