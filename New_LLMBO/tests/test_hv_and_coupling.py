from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataBase.database import ObservationDB
from llm.llm_interface import PhysicsHeuristicFallback, ResponseParser
from llmbo.gp_model import MaternGPModel
from utils.constants import (
    DEFAULT_BOUNDS,
    DSOC_SUM_MAX,
    LLM_SAFE_DSOC_SUM_MAX,
    dsoc_sum_violates_limit,
    project_dsoc_pair,
)


def test_duplicate_pareto_points_are_skipped() -> None:
    db = ObservationDB()

    theta = np.array([4.0, 3.5, 2.5, 0.25, 0.20], dtype=float)
    objectives = np.array([4200.0, 4.2, 0.80], dtype=float)

    db.add_observation(theta=theta, objectives=objectives, feasible=True, source="test")
    db.add_observation(theta=theta, objectives=objectives.copy(), feasible=True, source="test")

    assert db.pareto_size == 1


def test_hypervolume_is_non_decreasing_for_duplicate_and_better_points() -> None:
    db = ObservationDB()

    theta_1 = np.array([4.0, 3.5, 2.5, 0.25, 0.20], dtype=float)
    theta_2 = np.array([5.0, 4.0, 2.8, 0.20, 0.18], dtype=float)

    obj_1 = np.array([4500.0, 4.5, 0.90], dtype=float)
    obj_2 = np.array([3600.0, 4.0, 0.70], dtype=float)

    db.add_observation(theta=theta_1, objectives=obj_1, feasible=True, source="test")
    hv_1 = db.compute_hypervolume()

    db.add_observation(theta=theta_1, objectives=obj_1.copy(), feasible=True, source="test")
    hv_2 = db.compute_hypervolume()

    db.add_observation(theta=theta_2, objectives=obj_2, feasible=True, source="test")
    hv_3 = db.compute_hypervolume()

    assert hv_2 >= hv_1
    assert hv_3 >= hv_2


def test_lambda_is_annealed_and_clamped() -> None:
    gp = MaternGPModel(param_bounds=DEFAULT_BOUNDS)
    grid = np.array([[4.0, 3.5, 2.5, 0.25, 0.20]], dtype=float)
    weights = np.array([1.0], dtype=float)

    with patch.object(gp, "posterior_covariance", return_value=np.array([[1e-8]], dtype=float)):
        coupling = gp.build_preference_coupling(
            grid=grid,
            weights=weights,
            confidence=1.0,
            t=3,
            lambda_max=1.0,
            lambda_min=0.0,
            decay_rate=0.75,
        )

    expected_base = 1.0 / np.sqrt(1e-8)
    expected_annealed = expected_base * (0.75 ** 3)

    assert expected_annealed > 1.0
    assert coupling.lambda_value == 1.0


def test_dsoc_projection_respects_strict_hard_limit() -> None:
    d1, d2 = project_dsoc_pair(0.40, 0.30, dsoc_sum_max=DSOC_SUM_MAX)

    assert d1 + d2 < DSOC_SUM_MAX
    assert not dsoc_sum_violates_limit(d1, d2, dsoc_sum_max=DSOC_SUM_MAX)


def test_llm_parser_repairs_to_soft_safety_margin() -> None:
    parser = ResponseParser(
        DEFAULT_BOUNDS,
        dsoc_sum_max=DSOC_SUM_MAX,
        soft_dsoc_sum_max=LLM_SAFE_DSOC_SUM_MAX,
    )

    repaired = parser.repair_theta(np.array([4.8, 3.6, 2.3, 0.38, 0.29], dtype=float))

    assert repaired[3] + repaired[4] <= LLM_SAFE_DSOC_SUM_MAX + 1e-9
    assert repaired[3] + repaired[4] < DSOC_SUM_MAX


def test_physics_fallback_prior_points_stay_under_soft_margin() -> None:
    fallback = PhysicsHeuristicFallback(
        DEFAULT_BOUNDS,
        dsoc_sum_max=DSOC_SUM_MAX,
        soft_dsoc_sum_max=LLM_SAFE_DSOC_SUM_MAX,
    )

    points = fallback.physics_informed_warmstart(12)

    assert points
    assert all(point[3] + point[4] <= LLM_SAFE_DSOC_SUM_MAX + 1e-9 for point in points)
