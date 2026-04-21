from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np


def _stable_sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    z = np.asarray(x, dtype=float)
    z = np.clip(z, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(-z))
    if np.isscalar(x):
        return float(out)
    return out


@dataclasses.dataclass(frozen=True)
class CandidateInfo:
    idx: int
    x: List[float]
    mu_fw: float
    sigma_fw: float
    ei: float
    rank_by_ei: int
    dist_to_best: Optional[float] = None
    dist_to_pareto: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": int(self.idx),
            "x": list(self.x),
            "mu_fw": float(self.mu_fw),
            "sigma_fw": float(self.sigma_fw),
            "ei": float(self.ei),
            "rank_by_ei": int(self.rank_by_ei),
            "dist_to_best": None if self.dist_to_best is None else float(self.dist_to_best),
            "dist_to_pareto": None if self.dist_to_pareto is None else float(self.dist_to_pareto),
        }


@dataclasses.dataclass(frozen=True)
class RerankState:
    iter_id: int
    w_vec: List[float]
    tau_t: float
    scalar_best: float
    hv_current: float
    hv_gain_recent_mean: float
    violation_rate_recent: float
    llm_uncertainty_recent: float
    safe_margin_summary: Dict[str, Any]
    history_summary: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iter_id": int(self.iter_id),
            "w_vec": list(self.w_vec),
            "tau_t": float(self.tau_t),
            "scalar_best": float(self.scalar_best),
            "hv_current": float(self.hv_current),
            "hv_gain_recent_mean": float(self.hv_gain_recent_mean),
            "violation_rate_recent": float(self.violation_rate_recent),
            "llm_uncertainty_recent": float(self.llm_uncertainty_recent),
            "safe_margin_summary": dict(self.safe_margin_summary),
            "history_summary": list(self.history_summary),
        }


@dataclasses.dataclass(frozen=True)
class RerankOutput:
    idx: int
    q_good: float
    confidence: float
    rationale_short: str

    def entropy(self) -> float:
        q = float(np.clip(self.q_good, 1e-6, 1.0 - 1e-6))
        raw = -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))
        return float(raw / np.log(2.0))

    def centered_score(self) -> float:
        return float(2.0 * np.clip(self.q_good, 0.0, 1.0) - 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": int(self.idx),
            "q_good": float(np.clip(self.q_good, 0.0, 1.0)),
            "confidence": float(np.clip(self.confidence, 0.0, 1.0)),
            "rationale_short": str(self.rationale_short),
            "entropy": float(self.entropy()),
        }


@dataclasses.dataclass(frozen=True)
class GateState:
    g_value: float
    hv_gain_recent_mean: float
    violation_rate_recent: float
    llm_uncertainty_recent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "g_value": float(self.g_value),
            "hv_gain_recent_mean": float(self.hv_gain_recent_mean),
            "violation_rate_recent": float(self.violation_rate_recent),
            "llm_uncertainty_recent": float(self.llm_uncertainty_recent),
        }


@dataclasses.dataclass(frozen=True)
class TrialTelemetry:
    iter_id: int
    w_vec: List[float]
    tau_t: float
    selected_idx_before_rerank: int
    selected_idx_after_rerank: int
    g_value: float
    llm_called: bool
    llm_entropy_mean: Optional[float]
    llm_q_selected: Optional[float]
    score_plain_selected: Optional[float]
    score_rerank_selected: Optional[float]
    hv_before: float
    hv_after: float
    hv_gain: float
    feasible: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iter_id": int(self.iter_id),
            "w_vec": list(self.w_vec),
            "tau_t": float(self.tau_t),
            "selected_idx_before_rerank": int(self.selected_idx_before_rerank),
            "selected_idx_after_rerank": int(self.selected_idx_after_rerank),
            "g_value": float(self.g_value),
            "llm_called": bool(self.llm_called),
            "llm_entropy_mean": None if self.llm_entropy_mean is None else float(self.llm_entropy_mean),
            "llm_q_selected": None if self.llm_q_selected is None else float(self.llm_q_selected),
            "score_plain_selected": None if self.score_plain_selected is None else float(self.score_plain_selected),
            "score_rerank_selected": None if self.score_rerank_selected is None else float(self.score_rerank_selected),
            "hv_before": float(self.hv_before),
            "hv_after": float(self.hv_after),
            "hv_gain": float(self.hv_gain),
            "feasible": bool(self.feasible),
        }


def compute_online_gate(
    hv_gain_recent_mean: float,
    violation_rate_recent: float,
    llm_uncertainty_recent: float,
    lambda_max: float,
    a: float,
    b: float,
    c: float,
) -> GateState:
    g = float(lambda_max) * float(
        _stable_sigmoid(
            float(a) * float(hv_gain_recent_mean)
            - float(b) * float(violation_rate_recent)
            - float(c) * float(llm_uncertainty_recent)
        )
    )
    g = float(np.clip(g, 0.0, float(lambda_max)))
    return GateState(
        g_value=g,
        hv_gain_recent_mean=float(hv_gain_recent_mean),
        violation_rate_recent=float(violation_rate_recent),
        llm_uncertainty_recent=float(llm_uncertainty_recent),
    )


def rerank_topm_with_llm(
    topm_candidates: List[CandidateInfo],
    llm_outputs: List[RerankOutput],
    gate_value: float,
    score_mode: str,
    n_select: int = 1,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    idx_to_output = {int(item.idx): item for item in llm_outputs}
    rows: List[Dict[str, Any]] = []
    for candidate in topm_candidates:
        output = idx_to_output.get(int(candidate.idx))
        if output is None:
            continue
        s_val = output.centered_score()
        if score_mode == "ei_exp_gate":
            rerank_score = float(candidate.ei) * float(np.exp(float(gate_value) * s_val))
        else:
            rerank_score = float(np.log(float(candidate.ei) + float(eps)) + float(gate_value) * s_val)
        rows.append(
            {
                "idx": int(candidate.idx),
                "ei": float(candidate.ei),
                "q_good": float(output.q_good),
                "confidence": float(output.confidence),
                "entropy": float(output.entropy()),
                "centered_score": float(s_val),
                "rerank_score": float(rerank_score),
                "rationale_short": str(output.rationale_short),
            }
        )

    if not rows:
        return {
            "selected_indices": [],
            "selected_scores": np.empty((0,), dtype=float),
            "entropy_mean": None,
            "rows": [],
        }

    order = sorted(rows, key=lambda row: row["rerank_score"], reverse=True)
    selected = order[: max(int(n_select), 1)]
    entropy_mean = float(np.mean([row["entropy"] for row in rows]))
    return {
        "selected_indices": [int(row["idx"]) for row in selected],
        "selected_scores": np.array([float(row["rerank_score"]) for row in selected], dtype=float),
        "entropy_mean": entropy_mean,
        "rows": order,
    }
