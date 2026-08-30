"""Deterministic portfolio selection for LLM warm-start candidates.

The selector intentionally sits before GP/EI.  It turns an over-generated LLM
candidate pool into a small, complementary warm-start portfolio while keeping
hard simulator semantics unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.constants import (
    DEFAULT_BOUNDS,
    DSOC_SUM_ATOL,
    DSOC_SUM_MAX,
    LLM_SAFE_DSOC_SUM_MAX,
    PARAM_NAMES,
    dsoc_sum_violates_limit,
)


@dataclasses.dataclass
class WarmStartCandidate:
    theta: np.ndarray
    source: str = "llm"
    confidence: float = 0.5
    style: str = "unknown"
    risk_flags: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    rationale: str = ""
    raw_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theta": np.asarray(self.theta, dtype=float).ravel().tolist(),
            "source": self.source,
            "confidence": float(self.confidence),
            "style": self.style,
            "risk_flags": list(self.risk_flags),
            "rationale": self.rationale,
            "raw_index": int(self.raw_index),
        }


@dataclasses.dataclass
class WarmStartSelectionConfig:
    n_select: int
    bounds: Dict[str, Tuple[float, float]] = dataclasses.field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_BOUNDS.items()}
    )
    hard_dsoc_sum_max: float = DSOC_SUM_MAX
    soft_dsoc_sum_max: float = LLM_SAFE_DSOC_SUM_MAX
    diversity_weight: float = 0.45
    soft_penalty_weight: float = 0.65
    monotone_bonus: float = 0.08
    archive_bonus_weight: float = 0.0
    boundary_probe_limit: int = 1
    boundary_probe_margin: float = 0.015
    dedup_decimals: int = 4


def _as_candidate(item: WarmStartCandidate | np.ndarray, raw_index: int) -> WarmStartCandidate:
    if isinstance(item, WarmStartCandidate):
        return item
    return WarmStartCandidate(theta=np.asarray(item, dtype=float), raw_index=raw_index)


def _bounds_arrays(bounds: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.array([bounds[k][0] for k in PARAM_NAMES], dtype=float)
    hi = np.array([bounds[k][1] for k in PARAM_NAMES], dtype=float)
    return lo, hi


def _normalized(theta: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.maximum(hi - lo, 1e-12)
    return (np.asarray(theta, dtype=float).ravel() - lo) / span


def _is_monotone(theta: np.ndarray) -> bool:
    x = np.asarray(theta, dtype=float).ravel()
    return bool(x.size >= 3 and x[0] >= x[1] >= x[2])


def _is_boundary_probe(candidate: WarmStartCandidate, soft_limit: float, margin: float) -> bool:
    theta = np.asarray(candidate.theta, dtype=float).ravel()
    dsoc_sum = float(theta[3] + theta[4]) if theta.size >= 5 else 0.0
    text = " ".join([candidate.style, " ".join(candidate.risk_flags), candidate.rationale]).lower()
    return bool(dsoc_sum >= soft_limit + margin or "boundary" in text or "probe" in text)


def _candidate_quality(
    candidate: WarmStartCandidate,
    *,
    soft_limit: float,
    hard_limit: float,
    soft_penalty_weight: float,
    monotone_bonus: float,
) -> float:
    theta = np.asarray(candidate.theta, dtype=float).ravel()
    confidence = float(np.clip(candidate.confidence, 0.0, 1.0))
    dsoc_sum = float(theta[3] + theta[4])
    denom = max(hard_limit - soft_limit, 1e-12)
    soft_over = max(0.0, dsoc_sum - soft_limit) / denom
    score = confidence - soft_penalty_weight * soft_over
    if _is_monotone(theta):
        score += float(monotone_bonus)
    return float(score)


def _archive_bonus(
    theta: np.ndarray,
    archive_points: Optional[np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
) -> float:
    if archive_points is None or len(archive_points) == 0:
        return 0.0
    x = _normalized(theta, lo, hi)
    archive = np.atleast_2d(np.asarray(archive_points, dtype=float))
    archive_n = np.vstack([_normalized(row, lo, hi) for row in archive])
    return float(np.min(np.linalg.norm(archive_n - x[None, :], axis=1)))


def filter_warmstart_candidates(
    candidates: Iterable[WarmStartCandidate | np.ndarray],
    cfg: WarmStartSelectionConfig,
) -> Tuple[List[WarmStartCandidate], Dict[str, Any]]:
    lo, hi = _bounds_arrays(cfg.bounds)
    valid: List[WarmStartCandidate] = []
    seen = set()
    filtered = {
        "non_finite": 0,
        "shape": 0,
        "bounds": 0,
        "hard_dsoc": 0,
        "duplicate": 0,
    }

    for raw_index, item in enumerate(candidates):
        candidate = _as_candidate(item, raw_index=raw_index)
        theta = np.asarray(candidate.theta, dtype=float).ravel()
        if theta.size != len(PARAM_NAMES):
            filtered["shape"] += 1
            continue
        if not np.all(np.isfinite(theta)):
            filtered["non_finite"] += 1
            continue
        if np.any(theta < lo - 1e-12) or np.any(theta > hi + 1e-12):
            filtered["bounds"] += 1
            continue
        if dsoc_sum_violates_limit(
            theta[3],
            theta[4],
            dsoc_sum_max=cfg.hard_dsoc_sum_max,
            atol=DSOC_SUM_ATOL,
        ):
            filtered["hard_dsoc"] += 1
            continue
        key = tuple(np.round(theta, cfg.dedup_decimals))
        if key in seen:
            filtered["duplicate"] += 1
            continue
        seen.add(key)
        valid.append(dataclasses.replace(candidate, theta=theta))

    summary = {
        "input_count": int(sum(filtered.values()) + len(valid)),
        "valid_count": int(len(valid)),
        "filtered": filtered,
    }
    return valid, summary


def select_warmstart_portfolio(
    candidates: Sequence[WarmStartCandidate | np.ndarray],
    cfg: WarmStartSelectionConfig,
    *,
    archive_points: Optional[np.ndarray] = None,
) -> Tuple[List[WarmStartCandidate], Dict[str, Any]]:
    """Select a small warm-start portfolio from an over-generated pool."""

    valid, summary = filter_warmstart_candidates(candidates, cfg)
    n_select = max(0, int(cfg.n_select))
    if n_select <= 0 or not valid:
        summary.update(
            {
                "requested": n_select,
                "selected_count": 0,
                "selected": [],
                "method": "portfolio_selector",
            }
        )
        return [], summary

    lo, hi = _bounds_arrays(cfg.bounds)
    selected: List[WarmStartCandidate] = []
    remaining = list(valid)
    boundary_selected = 0
    chosen_scores: List[Dict[str, Any]] = []

    while remaining and len(selected) < n_select:
        best_idx: Optional[int] = None
        best_score = -float("inf")
        best_parts: Dict[str, float] = {}

        for idx, candidate in enumerate(remaining):
            is_boundary = _is_boundary_probe(
                candidate,
                cfg.soft_dsoc_sum_max,
                cfg.boundary_probe_margin,
            )
            if is_boundary and boundary_selected >= int(cfg.boundary_probe_limit):
                continue

            quality = _candidate_quality(
                candidate,
                soft_limit=cfg.soft_dsoc_sum_max,
                hard_limit=cfg.hard_dsoc_sum_max,
                soft_penalty_weight=cfg.soft_penalty_weight,
                monotone_bonus=cfg.monotone_bonus,
            )
            diversity = 0.0
            if selected:
                x = _normalized(candidate.theta, lo, hi)
                selected_n = np.vstack([_normalized(item.theta, lo, hi) for item in selected])
                diversity = float(np.min(np.linalg.norm(selected_n - x[None, :], axis=1)))
            archive = _archive_bonus(candidate.theta, archive_points, lo, hi)
            score = (
                quality
                + float(cfg.diversity_weight) * diversity
                + float(cfg.archive_bonus_weight) * archive
            )

            if score > best_score:
                best_score = float(score)
                best_idx = idx
                best_parts = {
                    "quality": float(quality),
                    "diversity": float(diversity),
                    "archive_bonus": float(archive),
                    "score": float(score),
                    "boundary_probe": float(is_boundary),
                }

        if best_idx is None:
            break

        chosen = remaining.pop(best_idx)
        if _is_boundary_probe(chosen, cfg.soft_dsoc_sum_max, cfg.boundary_probe_margin):
            boundary_selected += 1
        selected.append(chosen)
        chosen_scores.append(
            {
                **chosen.to_dict(),
                **best_parts,
                "dSOC_sum": float(np.asarray(chosen.theta, dtype=float)[3:5].sum()),
                "monotone": bool(_is_monotone(chosen.theta)),
            }
        )

    summary.update(
        {
            "requested": n_select,
            "selected_count": int(len(selected)),
            "selected": chosen_scores,
            "boundary_selected": int(boundary_selected),
            "method": "portfolio_selector",
            "config": {
                "hard_dsoc_sum_max": float(cfg.hard_dsoc_sum_max),
                "soft_dsoc_sum_max": float(cfg.soft_dsoc_sum_max),
                "diversity_weight": float(cfg.diversity_weight),
                "soft_penalty_weight": float(cfg.soft_penalty_weight),
                "monotone_bonus": float(cfg.monotone_bonus),
                "archive_bonus_weight": float(cfg.archive_bonus_weight),
                "boundary_probe_limit": int(cfg.boundary_probe_limit),
            },
        }
    )
    return selected, summary


__all__ = [
    "WarmStartCandidate",
    "WarmStartSelectionConfig",
    "filter_warmstart_candidates",
    "select_warmstart_portfolio",
]
