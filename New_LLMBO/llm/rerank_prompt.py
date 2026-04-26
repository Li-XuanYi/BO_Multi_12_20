"""
Candidate-rerank prompt templates and rendering utilities.

This mirrors the warm-start / iteration-guidance prompt architecture:
  1. Text template on disk
  2. Context builder that formats runtime state and shortlist candidates
  3. Small renderer that validates placeholder resolution
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from llmbo.rerank import CandidateInfo, RerankState
from utils.constants import LLM_SAFE_DSOC_SUM_MAX

try:
    from llm.warmstart_prompt import PLACEHOLDER_PATTERN
except ModuleNotFoundError:  # pragma: no cover - allows direct script execution
    from warmstart_prompt import PLACEHOLDER_PATTERN

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "candidate_rerank.txt"
PARAM_KEYS = ("I1", "I2", "I3", "dSOC1", "dSOC2")


class CandidateRerankTemplateRenderer:
    """Load and render the candidate-rerank template with [PLACEHOLDER] tokens."""

    def __init__(self, template_path: Optional[Path] = None):
        self._template_path = Path(template_path or DEFAULT_TEMPLATE_PATH)
        self._cache: Optional[str] = None

    def load(self) -> str:
        if self._cache is None:
            if not self._template_path.exists():
                raise FileNotFoundError(f"Candidate rerank template not found: {self._template_path}")
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
                f"Unresolved placeholders remain in candidate rerank template: {missing}"
            )
        return rendered


@dataclass(frozen=True)
class CandidateRerankContextBuilder:
    param_bounds: Dict[str, Tuple[float, float]]
    safe_dsoc_sum_max: float = LLM_SAFE_DSOC_SUM_MAX
    hard_dsoc_sum_max: float = 0.70

    def build(
        self,
        *,
        state: RerankState,
        candidates: Iterable[CandidateInfo],
        scalarization_formula: str,
    ) -> Dict[str, str]:
        candidates = list(candidates)
        return {
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
            "ITERATION": str(int(state.iter_id)),
            "WEIGHT_VECTOR": self._format_weights(state.w_vec),
            "TAU_T": f"{float(state.tau_t):.6f}",
            "SCALAR_BEST": f"{float(state.scalar_best):.6f}",
            "HV_CURRENT": f"{float(state.hv_current):.6f}",
            "HV_GAIN_RECENT_MEAN": f"{float(state.hv_gain_recent_mean):.6f}",
            "VIOLATION_RATE_RECENT": f"{float(state.violation_rate_recent):.4f}",
            "LLM_UNCERTAINTY_RECENT": f"{float(state.llm_uncertainty_recent):.4f}",
            "SCALARIZATION_FORMULA": str(scalarization_formula or "Lower scalarized objective is better."),
            "SAFE_MARGIN_SUMMARY": self._json_or_none(state.safe_margin_summary),
            "HISTORY_SUMMARY": self._format_history(state.history_summary),
            "CANDIDATE_BLOCK": self._format_candidates(candidates),
        }

    @staticmethod
    def _format_weights(w_vec: List[float]) -> str:
        w = np.asarray(w_vec, dtype=float).ravel()
        if w.size != 3:
            return str(list(w_vec))
        return f"[time={w[0]:.3f}, temp={w[1]:.3f}, aging={w[2]:.3f}]"

    @staticmethod
    def _json_or_none(value: Any) -> str:
        if not value:
            return "none"
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    @staticmethod
    def _format_history(history_summary: List[Dict[str, Any]]) -> str:
        if not history_summary:
            return "none"
        lines: List[str] = []
        for item in history_summary[:5]:
            try:
                lines.append(json.dumps(item, ensure_ascii=False))
            except TypeError:
                lines.append(str(item))
        return "\n".join(lines)

    @staticmethod
    def _format_candidates(candidates: List[CandidateInfo]) -> str:
        if not candidates:
            return "none"
        lines: List[str] = []
        for candidate in candidates:
            x = np.asarray(candidate.x, dtype=float).ravel()
            if x.size != len(PARAM_KEYS):
                continue
            dist_to_best = (
                "none" if candidate.dist_to_best is None
                else f"{float(candidate.dist_to_best):.4f}"
            )
            dist_to_pareto = (
                "none" if candidate.dist_to_pareto is None
                else f"{float(candidate.dist_to_pareto):.4f}"
            )
            lines.append(
                f"Candidate {int(candidate.idx)}:\n"
                f"  x = [I1={x[0]:.3f}, I2={x[1]:.3f}, I3={x[2]:.3f}, dSOC1={x[3]:.4f}, dSOC2={x[4]:.4f}]\n"
                f"  dSOC_sum = {'none' if candidate.dSOC_sum is None else f'{float(candidate.dSOC_sum):.4f}'}\n"
                f"  margin_to_soft_limit = {'none' if candidate.margin_to_soft_limit is None else f'{float(candidate.margin_to_soft_limit):.4f}'}\n"
                f"  monotone_flag = {'none' if candidate.monotone_flag is None else str(bool(candidate.monotone_flag)).lower()}\n"
                f"  mu_fw = {float(candidate.mu_fw):.6f}\n"
                f"  sigma_fw = {float(candidate.sigma_fw):.6f}\n"
                f"  EI = {float(candidate.ei):.6e}\n"
                f"  log_ei_gap_to_best = {'none' if candidate.log_ei_gap_to_best is None else f'{float(candidate.log_ei_gap_to_best):.6f}'}\n"
                f"  rank_by_ei = {int(candidate.rank_by_ei)}\n"
                f"  dist_to_best = {dist_to_best}\n"
                f"  dist_to_pareto = {dist_to_pareto}"
            )
        return "\n\n".join(lines) if lines else "none"


def render_candidate_rerank_prompt(
    *,
    state: RerankState,
    candidates: Iterable[CandidateInfo],
    param_bounds: Dict[str, Tuple[float, float]],
    scalarization_formula: str,
    safe_dsoc_sum_max: float = LLM_SAFE_DSOC_SUM_MAX,
    hard_dsoc_sum_max: float = 0.70,
    template_path: Optional[Path] = None,
) -> str:
    builder = CandidateRerankContextBuilder(
        param_bounds=param_bounds,
        safe_dsoc_sum_max=safe_dsoc_sum_max,
        hard_dsoc_sum_max=hard_dsoc_sum_max,
    )
    context = builder.build(
        state=state,
        candidates=candidates,
        scalarization_formula=scalarization_formula,
    )
    renderer = CandidateRerankTemplateRenderer(template_path=template_path)
    prompt = renderer.render(context)
    logger.debug("Candidate rerank prompt rendered (%d chars)", len(prompt))
    return prompt
