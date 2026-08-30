from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class CandidateInfo:
    idx: int
    x: List[float]
    mu_fw: float
    sigma_fw: float
    ei: float
    rank_by_ei: int
    log_ei: Optional[float] = None
    log_ei_gap_to_best: Optional[float] = None
    dist_to_best: Optional[float] = None
    dist_to_pareto: Optional[float] = None
    dSOC_sum: Optional[float] = None
    margin_to_soft_limit: Optional[float] = None
    hard_violation_flag: Optional[bool] = None
    monotone_flag: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": int(self.idx),
            "x": list(self.x),
            "mu_fw": float(self.mu_fw),
            "sigma_fw": float(self.sigma_fw),
            "ei": float(self.ei),
            "rank_by_ei": int(self.rank_by_ei),
            "log_ei": None if self.log_ei is None else float(self.log_ei),
            "log_ei_gap_to_best": None if self.log_ei_gap_to_best is None else float(self.log_ei_gap_to_best),
            "dist_to_best": None if self.dist_to_best is None else float(self.dist_to_best),
            "dist_to_pareto": None if self.dist_to_pareto is None else float(self.dist_to_pareto),
            "dSOC_sum": None if self.dSOC_sum is None else float(self.dSOC_sum),
            "margin_to_soft_limit": None if self.margin_to_soft_limit is None else float(self.margin_to_soft_limit),
            "hard_violation_flag": None if self.hard_violation_flag is None else bool(self.hard_violation_flag),
            "monotone_flag": None if self.monotone_flag is None else bool(self.monotone_flag),
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
    risk_flags: List[str] = dataclasses.field(default_factory=list)

    def entropy(self) -> float:
        q = float(np.clip(self.q_good, 1e-6, 1.0 - 1e-6))
        raw = -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))
        return float(raw / np.log(2.0))

    def centered_score(self) -> float:
        return float(2.0 * np.clip(self.q_good, 0.0, 1.0) - 1.0)

    def effective_confidence(self, min_confidence: float) -> float:
        confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if confidence < float(min_confidence):
            return 0.0
        return float(np.clip(confidence * (1.0 - self.entropy()), 0.0, 1.0))

    def effective_q_good(self, min_confidence: float) -> float:
        if float(np.clip(self.confidence, 0.0, 1.0)) < float(min_confidence):
            return 0.5
        return float(np.clip(self.q_good, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": int(self.idx),
            "candidate_id": int(self.idx),
            "q_good": float(np.clip(self.q_good, 0.0, 1.0)),
            "confidence": float(np.clip(self.confidence, 0.0, 1.0)),
            "rationale_short": str(self.rationale_short),
            "risk_flags": list(self.risk_flags),
            "entropy": float(self.entropy()),
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
    plain_ei: Optional[float] = None
    rerank_ei: Optional[float] = None
    ei_ratio: Optional[float] = None
    log_ei_gap: Optional[float] = None
    plain_mu: Optional[float] = None
    plain_sigma: Optional[float] = None
    rerank_mu: Optional[float] = None
    rerank_sigma: Optional[float] = None
    plain_rank_by_ei: Optional[int] = None
    rerank_rank_by_ei: Optional[int] = None
    llm_q_plain: Optional[float] = None
    llm_q_rerank: Optional[float] = None
    llm_conf_plain: Optional[float] = None
    llm_conf_rerank: Optional[float] = None
    selected_changed: bool = False
    fallback_reason: Optional[str] = None
    rerank_mode: Optional[str] = None

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
            "plain_ei": None if self.plain_ei is None else float(self.plain_ei),
            "rerank_ei": None if self.rerank_ei is None else float(self.rerank_ei),
            "ei_ratio": None if self.ei_ratio is None else float(self.ei_ratio),
            "log_ei_gap": None if self.log_ei_gap is None else float(self.log_ei_gap),
            "plain_mu": None if self.plain_mu is None else float(self.plain_mu),
            "plain_sigma": None if self.plain_sigma is None else float(self.plain_sigma),
            "rerank_mu": None if self.rerank_mu is None else float(self.rerank_mu),
            "rerank_sigma": None if self.rerank_sigma is None else float(self.rerank_sigma),
            "plain_rank_by_ei": None if self.plain_rank_by_ei is None else int(self.plain_rank_by_ei),
            "rerank_rank_by_ei": None if self.rerank_rank_by_ei is None else int(self.rerank_rank_by_ei),
            "llm_q_plain": None if self.llm_q_plain is None else float(self.llm_q_plain),
            "llm_q_rerank": None if self.llm_q_rerank is None else float(self.llm_q_rerank),
            "llm_conf_plain": None if self.llm_conf_plain is None else float(self.llm_conf_plain),
            "llm_conf_rerank": None if self.llm_conf_rerank is None else float(self.llm_conf_rerank),
            "selected_changed": bool(self.selected_changed),
            "fallback_reason": self.fallback_reason,
            "rerank_mode": self.rerank_mode,
        }


def rerank_topm_with_llm(
    topm_candidates: List[CandidateInfo],
    llm_outputs: List[RerankOutput],
    mode: str,
    gate: float,
    max_log_ei_gap: float,
    max_bonus: float,
    q_bad_threshold: float,
    min_confidence: float,
    n_select: int = 1,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    mode = str(mode or "none")
    if mode == "const_gate":
        mode = "unsafe_legacy_const_gate"

    if not topm_candidates:
        return {
            "selected_indices": [],
            "selected_scores": np.empty((0,), dtype=float),
            "entropy_mean": None,
            "rows": [],
            "eligible_indices": [],
            "fallback_reason": "empty_topm",
        }

    idx_to_output = {int(item.idx): item for item in llm_outputs}
    candidate_by_idx = {int(item.idx): item for item in topm_candidates}
    best_log_ei = max(
        float(candidate.log_ei) if candidate.log_ei is not None else float(np.log(float(candidate.ei) + float(eps)))
        for candidate in topm_candidates
    )
    eligible_indices = [
        int(candidate.idx)
        for candidate in topm_candidates
        if (
            mode == "unsafe_legacy_const_gate"
            or (
                best_log_ei
                - (
                    float(candidate.log_ei)
                    if candidate.log_ei is not None
                    else float(np.log(float(candidate.ei) + float(eps)))
                )
            ) <= float(max_log_ei_gap) + 1e-12
        )
    ]
    if mode != "unsafe_legacy_const_gate" and len(eligible_indices) <= 1:
        return {
            "selected_indices": [],
            "selected_scores": np.empty((0,), dtype=float),
            "entropy_mean": None,
            "rows": [],
            "eligible_indices": eligible_indices,
            "fallback_reason": "eligible_too_small",
        }

    rows: List[Dict[str, Any]] = []
    for candidate in topm_candidates:
        if mode != "unsafe_legacy_const_gate" and int(candidate.idx) not in eligible_indices:
            continue
        output = idx_to_output.get(int(candidate.idx))
        if output is None:
            continue
        log_ei = (
            float(candidate.log_ei)
            if candidate.log_ei is not None
            else float(np.log(float(candidate.ei) + float(eps)))
        )
        confidence_raw = float(np.clip(output.confidence, 0.0, 1.0))
        conf_eff = float(output.effective_confidence(min_confidence))
        q_effective = float(output.effective_q_good(min_confidence))
        centered_score = float(2.0 * q_effective - 1.0)
        entropy = float(output.entropy())

        if mode == "risk_veto_only":
            penalty = float(max(0.0, (1.0 - q_effective) - float(q_bad_threshold)))
            rerank_score = log_ei - float(gate) * penalty * conf_eff
        elif mode == "unsafe_legacy_const_gate":
            rerank_score = log_ei + float(gate) * conf_eff * centered_score
        else:
            bonus = float(gate) * conf_eff * (q_effective - 0.5)
            rerank_score = log_ei + float(np.clip(bonus, -float(max_bonus), float(max_bonus)))

        rows.append(
            {
                "idx": int(candidate.idx),
                "candidate_id": int(candidate.idx),
                "ei": float(candidate.ei),
                "log_ei": log_ei,
                "log_ei_gap_to_best": float(best_log_ei - log_ei),
                "q_good": float(output.q_good),
                "q_effective": q_effective,
                "confidence": confidence_raw,
                "confidence_effective": conf_eff,
                "entropy": entropy,
                "centered_score": centered_score,
                "rerank_score": float(rerank_score),
                "rationale_short": str(output.rationale_short),
                "risk_flags": list(output.risk_flags),
                "mu_fw": float(candidate.mu_fw),
                "sigma_fw": float(candidate.sigma_fw),
                "rank_by_ei": int(candidate.rank_by_ei),
                "dist_to_best": candidate.dist_to_best,
                "dist_to_pareto": candidate.dist_to_pareto,
                "dSOC_sum": candidate.dSOC_sum,
                "margin_to_soft_limit": candidate.margin_to_soft_limit,
                "hard_violation_flag": candidate.hard_violation_flag,
                "monotone_flag": candidate.monotone_flag,
            }
        )

    if not rows:
        return {
            "selected_indices": [],
            "selected_scores": np.empty((0,), dtype=float),
            "entropy_mean": None,
            "rows": [],
            "eligible_indices": eligible_indices,
            "fallback_reason": "empty_rerank_rows",
        }

    order = sorted(rows, key=lambda row: row["rerank_score"], reverse=True)
    selected = order[: max(int(n_select), 1)]
    entropy_mean = float(np.mean([row["entropy"] for row in rows]))
    return {
        "selected_indices": [int(row["idx"]) for row in selected],
        "selected_scores": np.array([float(row["rerank_score"]) for row in selected], dtype=float),
        "entropy_mean": entropy_mean,
        "rows": order,
        "eligible_indices": eligible_indices,
        "fallback_reason": None,
        "selected_candidates": [candidate_by_idx[int(row["idx"])] for row in selected if int(row["idx"]) in candidate_by_idx],
    }
