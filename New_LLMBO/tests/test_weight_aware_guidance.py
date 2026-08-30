from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataBase.database import ObservationDB
from llm.llm_interface import LLMConfig, LLMInterface, _build_guidance_prompt
from llm.warmstart_prompt import PLACEHOLDER_PATTERN
from llmbo.gp_model import LLMPreferenceCoupling, MaternGPModel
from utils.constants import DEFAULT_BOUNDS


def test_guidance_prompt_exposes_weight_and_hv_context() -> None:
    state = {
        "iteration": 3,
        "max_iterations": 10,
        "w_vec": [0.55, 0.30, 0.15],
        "theta_best": [4.8, 4.0, 2.8, 0.22, 0.18],
        "f_min": 0.182,
        "eta": 0.05,
        "ideal_point": [1900.0, 1.8, 0.35],
        "y_min": [3.10, 1.80, -0.40],
        "y_max": [3.70, 7.00, -0.05],
        "stagnation_count": 1,
        "current_hv": 0.352,
        "hv_delta_last_3": 0.004,
        "pareto_size": 6,
        "scalarization_formula": "f_w = max_i(w_i * gap_i) + 0.05 * sum_i(w_i * gap_i)",
        "top_scalar_protocols": "iter=2 src=bo theta=[4.8, 4.0, 2.8, 0.22, 0.18] scalar=0.182000",
        "similar_weight_guidance_success": "similar_weight_guidance: matches=2, weighted_success=0.75",
        "boundary_failure_stats": "recent_failures=1/10, recent_monotone=0/10, near_safe=2/8, near_hard=0/8",
        "hv_feedback_summary": "current_hv=0.352000, hv_delta_last_3=0.004000, pareto_size=6",
        "previous_guidance": {"mode": "point", "confidence": 0.6},
        "selective_history_summary": "weight=[0.55, 0.3, 0.15]",
        "proposal_summary": {"ready": True, "n_components": 2},
        "uncertainty_hotspots": [],
    }

    prompt = _build_guidance_prompt(state, DEFAULT_BOUNDS, "pareto-context")

    assert "Current optimization target:" in prompt
    assert "Current hypervolume:" in prompt
    assert "HV delta over last 3 iterations:" in prompt
    assert "Top protocols under current weight:" in prompt
    assert "Similar-weight guidance reliability:" in prompt
    assert "IMPORTANT: optimize the current scalarized target f_w" in prompt
    assert not PLACEHOLDER_PATTERN.findall(prompt)


def test_database_llm_summary_helpers_return_structured_summaries() -> None:
    db = ObservationDB()

    db.add_observation(
        theta=np.array([4.0, 3.5, 2.5, 0.20, 0.18], dtype=float),
        objectives=np.array([4200.0, 4.5, 0.80], dtype=float),
        feasible=True,
        source="init",
        iteration=0,
    )
    db.record_iteration_stats(
        extra={
            "w_vec": [0.60, 0.30, 0.10],
            "llm_guidance": {"mode": "point", "confidence": 0.7},
        }
    )

    db.add_observation(
        theta=np.array([4.8, 4.0, 2.8, 0.19, 0.17], dtype=float),
        objectives=np.array([3600.0, 4.1, 0.65], dtype=float),
        feasible=True,
        source="bo",
        iteration=1,
    )
    db.record_iteration_stats(
        extra={
            "w_vec": [0.58, 0.32, 0.10],
            "llm_guidance": {"mode": "region", "confidence": 0.6},
        }
    )

    db.add_observation(
        theta=np.array([3.2, 4.1, 2.4, 0.25, 0.18], dtype=float),
        objectives=np.array([5000.0, 4.8, 0.90], dtype=float),
        feasible=True,
        source="bo",
        iteration=2,
    )
    db.add_observation(
        theta=np.array([4.5, 3.5, 2.2, 0.39, 0.29], dtype=float),
        objectives=np.array([7200.0, 40.0, 5.0], dtype=float),
        feasible=False,
        source="bo",
        iteration=2,
    )

    hv_summary = db.get_hv_feedback_summary(window=2)
    similar_stats = db.get_similar_weight_guidance_stats(
        np.array([0.59, 0.31, 0.10], dtype=float),
        similarity_threshold=0.8,
        fallback_score=0.2,
    )
    boundary_stats = db.get_boundary_failure_stats(
        safe_dsoc_sum_max=0.65,
        hard_dsoc_sum_max=0.70,
        recent_window=10,
    )

    assert hv_summary["current_hv"] > 0.0
    assert "hv_delta_last_2" in hv_summary["summary"]
    assert similar_stats["similar_count"] == 1
    assert similar_stats["success_rate"] == 1.0
    assert "weighted_success=1.00" in similar_stats["summary"]
    assert boundary_stats["recent_failures"] >= 1
    assert boundary_stats["recent_monotone"] >= 1
    assert "near_safe=" in boundary_stats["summary"]


def test_point_coupling_is_localized_by_mask() -> None:
    gp = MaternGPModel(param_bounds=DEFAULT_BOUNDS)
    center = np.array([4.8, 4.0, 2.8, 0.22, 0.18], dtype=float)
    near = center.copy()
    far = np.array([3.0, 2.5, 2.0, 0.36, 0.28], dtype=float)

    coupling = LLMPreferenceCoupling(
        mode="point",
        grid=center[None, :],
        weights=np.array([1.0], dtype=float),
        confidence=0.8,
        lambda_value=0.5,
        posterior_variance=1.0,
        gate=0.8,
        local_center=center.copy(),
        local_sigma=np.array([0.25, 0.25, 0.20, 0.04, 0.04], dtype=float),
    )

    with patch.object(gp, "predict", return_value=(np.array([0.40, 0.40]), np.array([0.10, 0.10]))):
        with patch.object(gp, "posterior_covariance", return_value=np.array([[1.0], [1.0]], dtype=float)):
            with patch.object(gp, "target_standardization", return_value=(0.0, 1.0)):
                mean, std = gp.predict_with_coupling(np.vstack([near, far]), coupling=coupling)

    assert std.shape == (2,)
    assert mean[0] < mean[1]


def test_mock_backend_region_preference_returns_heuristic_point() -> None:
    llm = LLMInterface(
        param_bounds=DEFAULT_BOUNDS,
        config=LLMConfig(backend="mock", n_samples=1, temperature=0.0),
    )
    state = {
        "iteration": 2,
        "top_scalar_points": [
            {
                "theta": [4.8, 4.0, 2.8, 0.22, 0.18],
                "raw_objectives": [3600.0, 4.0, 0.70],
                "scalar_y": 0.12,
            },
            {
                "theta": [4.4, 3.6, 2.6, 0.20, 0.17],
                "raw_objectives": [3900.0, 3.8, 0.62],
                "scalar_y": 0.16,
            },
        ],
        "recent_observations": [],
    }

    pref = llm.query_region_preference(state)

    assert pref.kind == "point"
    assert pref.coordinate_space == "raw"
    assert pref.preference_direction == "promising"
    assert pref.point is not None
    assert pref.confidence >= 0.7
