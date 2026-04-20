"""Canonical optimization constants shared across the project."""

from __future__ import annotations

import numpy as np

PARAM_NAMES = ("I1", "I2", "I3", "dSOC1", "dSOC2")

DEFAULT_BOUNDS = {
    "I1": (2.0, 6.0),
    "I2": (2.0, 5.0),
    "I3": (2.0, 3.0),
    "dSOC1": (0.10, 0.40),
    "dSOC2": (0.10, 0.30),
}

# Raw-objective reference used both for HV computation and failed-simulation penalties.
REF_POINT = np.array([7200.0, 40.0, 5.0], dtype=float)
IDEAL_POINT = np.array([1800.0, 0.0, 0.3], dtype=float)
FAILURE_PENALTY = REF_POINT.copy()

DSOC_SUM_MAX = 0.70
DSOC3_MIN = 0.10
LLM_SAFE_DSOC_SUM_MAX = 0.65
DSOC_SUM_ATOL = 1e-9
DSOC_REPAIR_SHRINK = 0.995


def dsoc_sum_violates_limit(
    dsoc1: float,
    dsoc2: float,
    dsoc_sum_max: float = DSOC_SUM_MAX,
    atol: float = DSOC_SUM_ATOL,
) -> bool:
    return float(dsoc1) + float(dsoc2) >= float(dsoc_sum_max) - float(atol)


def dsoc_repair_target(
    dsoc_sum_max: float = DSOC_SUM_MAX,
    shrink: float = DSOC_REPAIR_SHRINK,
    atol: float = DSOC_SUM_ATOL,
) -> float:
    return min(float(dsoc_sum_max) * float(shrink), float(dsoc_sum_max) - float(atol))


def project_dsoc_pair(
    dsoc1: float,
    dsoc2: float,
    dsoc_sum_max: float = DSOC_SUM_MAX,
    shrink: float = DSOC_REPAIR_SHRINK,
    atol: float = DSOC_SUM_ATOL,
) -> tuple[float, float]:
    total = float(dsoc1) + float(dsoc2)
    if total <= 0.0:
        return float(dsoc1), float(dsoc2)
    if not dsoc_sum_violates_limit(dsoc1, dsoc2, dsoc_sum_max=dsoc_sum_max, atol=atol):
        return float(dsoc1), float(dsoc2)
    scale = dsoc_repair_target(dsoc_sum_max=dsoc_sum_max, shrink=shrink, atol=atol) / max(total, 1e-12)
    return float(dsoc1) * scale, float(dsoc2) * scale

__all__ = [
    "DEFAULT_BOUNDS",
    "DSOC_REPAIR_SHRINK",
    "DSOC3_MIN",
    "DSOC_SUM_ATOL",
    "DSOC_SUM_MAX",
    "FAILURE_PENALTY",
    "IDEAL_POINT",
    "LLM_SAFE_DSOC_SUM_MAX",
    "PARAM_NAMES",
    "REF_POINT",
    "dsoc_repair_target",
    "dsoc_sum_violates_limit",
    "project_dsoc_pair",
]
