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

__all__ = [
    "DEFAULT_BOUNDS",
    "DSOC3_MIN",
    "DSOC_SUM_MAX",
    "FAILURE_PENALTY",
    "IDEAL_POINT",
    "PARAM_NAMES",
    "REF_POINT",
]
