from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from utils.constants import (
    DEFAULT_BOUNDS,
    DSOC_SUM_MAX,
    LLM_SAFE_DSOC_SUM_MAX,
    dsoc_sum_violates_limit,
    project_dsoc_pair,
)


PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]


@dataclass(frozen=True)
class ConstraintPolicy:
    """Centralized hard/soft constraint semantics for protocol proposals.

    The simulator hard feasibility stays at dSOC1 + dSOC2 < hard_dsoc_sum_max.
    The 0.65 safety band and monotone current profile remain soft preferences.
    """

    hard_dsoc_sum_max: float = DSOC_SUM_MAX
    soft_safe_dsoc_sum_max: float = LLM_SAFE_DSOC_SUM_MAX
    monotone_profile_is_soft: bool = True

    def repair_hard(
        self,
        theta: np.ndarray,
        bounds: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        x = np.asarray(theta, dtype=float).ravel().copy()
        bounds = bounds or DEFAULT_BOUNDS
        lo = np.array([bounds[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([bounds[k][1] for k in PARAM_KEYS], dtype=float)
        x = np.clip(x, lo, hi)
        if dsoc_sum_violates_limit(x[3], x[4], dsoc_sum_max=self.hard_dsoc_sum_max):
            x[3], x[4] = project_dsoc_pair(
                x[3],
                x[4],
                dsoc_sum_max=self.hard_dsoc_sum_max,
            )
            x = np.clip(x, lo, hi)
        return x

    def repair_soft(
        self,
        theta: np.ndarray,
        bounds: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        x = self.repair_hard(theta, bounds=bounds)
        if x[3] + x[4] > self.soft_safe_dsoc_sum_max:
            x[3], x[4] = project_dsoc_pair(
                x[3],
                x[4],
                dsoc_sum_max=self.soft_safe_dsoc_sum_max,
            )
            x = self.repair_hard(x, bounds=bounds)
        return x

    def safe_margin(self, theta: np.ndarray) -> float:
        x = np.asarray(theta, dtype=float).ravel()
        return float(self.soft_safe_dsoc_sum_max - (x[3] + x[4]))

    def monotone_violation(self, theta: np.ndarray) -> float:
        x = np.asarray(theta, dtype=float).ravel()
        return float(max(0.0, x[1] - x[0]) + max(0.0, x[2] - x[1]))

    def hard_violation(self, theta: np.ndarray) -> bool:
        x = np.asarray(theta, dtype=float).ravel()
        return bool(dsoc_sum_violates_limit(x[3], x[4], dsoc_sum_max=self.hard_dsoc_sum_max))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hard_dsoc_sum_max": float(self.hard_dsoc_sum_max),
            "soft_safe_dsoc_sum_max": float(self.soft_safe_dsoc_sum_max),
            "monotone_profile_is_soft": bool(self.monotone_profile_is_soft),
        }


def build_constraint_policy(config: Optional[Mapping[str, Any]] = None) -> ConstraintPolicy:
    cfg = dict(config or {})
    return ConstraintPolicy(
        hard_dsoc_sum_max=float(cfg.get("dsoc_sum_max", DSOC_SUM_MAX)),
        soft_safe_dsoc_sum_max=float(cfg.get("llm_safe_dsoc_sum_max", LLM_SAFE_DSOC_SUM_MAX)),
        monotone_profile_is_soft=bool(cfg.get("monotone_profile_is_soft", True)),
    )
