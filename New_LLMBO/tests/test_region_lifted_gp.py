from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llmbo.optimizer as optimizer_module
from llmbo.acquisition import AcquisitionFunction, AcquisitionResult, expected_improvement
from llmbo.gp_model import MaternGPModel
from llmbo.optimizer import BayesOptimizer
from llmbo.region_lifted_gp import (
    LLMRegionPreference,
    RegionLiftConfig,
    RegionLiftResult,
    evaluate_region_lift_on_pool,
    parse_region_preference_payload,
    sample_region_candidates,
)
from utils.constants import DEFAULT_BOUNDS, DSOC_SUM_MAX


def test_parse_region_preference_accepts_nested_region_bounds_and_array_points() -> None:
    pref = parse_region_preference_payload(
        {
            "kind": "region",
            "region": {
                "lower": [3.0, 2.5, 2.1, 0.15, 0.12],
                "upper": [4.0, 3.2, 2.8, 0.25, 0.20],
            },
            "confidence": 0.8,
        }
    )

    assert pref.kind == "region"
    assert pref.parser_status == "ok"
    assert pref.lb == {"I1": 3.0, "I2": 2.5, "I3": 2.1, "dSOC1": 0.15, "dSOC2": 0.12}
    assert pref.ub == {"I1": 4.0, "I2": 3.2, "I3": 2.8, "dSOC1": 0.25, "dSOC2": 0.20}

    point_pref = parse_region_preference_payload(
        {
            "kind": "point",
            "point": [4.5, 3.5, 2.5, 0.2, 0.2],
            "confidence": 0.75,
        }
    )

    assert point_pref.kind == "point"
    assert point_pref.point == {"I1": 4.5, "I2": 3.5, "I3": 2.5, "dSOC1": 0.2, "dSOC2": 0.2}


def _fit_gp() -> MaternGPModel:
    X = np.array(
        [
            [2.5, 2.4, 2.2, 0.16, 0.14],
            [3.0, 2.8, 2.3, 0.20, 0.16],
            [3.6, 3.1, 2.4, 0.24, 0.18],
            [4.2, 3.5, 2.5, 0.26, 0.20],
            [5.0, 4.0, 2.7, 0.30, 0.20],
            [5.6, 4.6, 2.9, 0.34, 0.24],
        ],
        dtype=float,
    )
    y = np.array([1.1, 0.8, 0.55, 0.70, 0.95, 1.20], dtype=float)
    gp = MaternGPModel(DEFAULT_BOUNDS, normalize_y=True, n_restarts_optimizer=0, random_state=0)
    gp.fit(X, y)
    return gp


def _preference(confidence: float = 0.9) -> LLMRegionPreference:
    payload = {
        "kind": "region",
        "coordinate_space": "raw",
        "preference_direction": "promising",
        "lb": {"I1": 3.2, "I2": 2.9, "I3": 2.3, "dSOC1": 0.20, "dSOC2": 0.16},
        "ub": {"I1": 4.0, "I2": 3.5, "I3": 2.5, "dSOC1": 0.26, "dSOC2": 0.20},
        "confidence": confidence,
        "preference_type": "balanced",
        "reason": "test region",
        "risk_flags": [],
    }
    return parse_region_preference_payload(payload)


def _point_preference(confidence: float = 0.9) -> LLMRegionPreference:
    payload = {
        "kind": "point",
        "coordinate_space": "raw",
        "preference_direction": "promising",
        "point": {"I1": 2.0, "I2": 2.0, "I3": 3.0, "dSOC1": 0.10, "dSOC2": 0.30},
        "confidence": confidence,
        "preference_type": "fast_charge",
        "reason": "test point",
        "risk_flags": [],
    }
    return parse_region_preference_payload(payload)


def test_standardized_posterior_covariance_scaling() -> None:
    gp = _fit_gp()
    X = np.array([[3.5, 3.1, 2.4, 0.23, 0.18], [4.5, 3.8, 2.6, 0.28, 0.19]], dtype=float)

    _, y_std = gp.target_standardization()
    cov_z = gp.posterior_covariance_standardized(X)
    cov_y = gp.posterior_covariance_raw(X)
    _, sigma_z = gp.predict_standardized(X)

    assert np.allclose(cov_z, cov_y / (y_std ** 2))
    assert np.allclose(sigma_z, np.sqrt(np.diag(cov_z)))


def test_predict_returns_raw_mean_and_raw_std() -> None:
    gp = _fit_gp()
    X = np.array([[3.5, 3.1, 2.4, 0.23, 0.18]], dtype=float)

    mean_y, std_y = gp.predict(X)
    mean_z, std_z = gp.predict_standardized(X)
    y_mean, y_std = gp.target_standardization()

    assert np.allclose(mean_y, mean_z * y_std + y_mean)
    assert np.allclose(std_y, std_z * y_std)


def test_target_transform_none_preserves_baseline_gp_behavior() -> None:
    X = np.array(
        [
            [2.5, 2.4, 2.2, 0.16, 0.14],
            [3.0, 2.8, 2.3, 0.20, 0.16],
            [3.6, 3.1, 2.4, 0.24, 0.18],
            [4.2, 3.5, 2.5, 0.26, 0.20],
        ],
        dtype=float,
    )
    y = np.array([1.1, 0.8, 0.55, 0.70], dtype=float)
    probe = np.array([[3.5, 3.1, 2.4, 0.23, 0.18]], dtype=float)
    gp_default = MaternGPModel(DEFAULT_BOUNDS, normalize_y=True, n_restarts_optimizer=0, random_state=0)
    gp_explicit = MaternGPModel(
        DEFAULT_BOUNDS,
        normalize_y=True,
        n_restarts_optimizer=0,
        target_transform_mode="none",
        random_state=0,
    )

    gp_default.fit(X, y)
    gp_explicit.fit(X, y)

    assert np.allclose(gp_default.predict(probe)[0], gp_explicit.predict(probe)[0])
    assert np.allclose(gp_default.predict(probe)[1], gp_explicit.predict(probe)[1])
    assert gp_explicit.training_summary()["target_transform_mode"] == "none"


def test_promising_region_lowers_standardized_mean_and_increases_minimization_ei() -> None:
    gp = _fit_gp()
    X_pool = np.array(
        [
            [3.5, 3.1, 2.4, 0.23, 0.18],
            [5.7, 4.8, 2.9, 0.34, 0.23],
        ],
        dtype=float,
    )
    cfg = RegionLiftConfig(
        enable_region_lifted_gp=True,
        region_lift_min_confidence=0.5,
        region_lift_min_volume=1e-6,
        region_lift_max_plain_ei_gap=10.0,
        region_lift_active_until=10,
    )
    pref = _preference()
    mean_z, sigma_z = gp.predict_standardized(X_pool)
    f_min_z = (0.55 - gp.target_standardization()[0]) / gp.target_standardization()[1]
    ei_plain = expected_improvement(mean_z, sigma_z, f_min_z)

    result = evaluate_region_lift_on_pool(
        gp=gp,
        candidate_pool=X_pool,
        f_min_y=0.55,
        preference=pref,
        existing_X=np.empty((0, 5)),
        bounds=DEFAULT_BOUNDS,
        config=cfg,
        trust=1.0,
        bo_iteration=0,
    )

    assert result.telemetry["max_shift_z"] > 0.0
    assert result.telemetry["sigma_unchanged"] is True
    if result.accepted:
        assert result.telemetry["lifted_ei_at_lift"] >= float(np.max(ei_plain))
    else:
        assert result.fallback_reason == "same_as_plain"


def test_region_lift_guards_fail_open_to_plain_ei() -> None:
    gp = _fit_gp()
    X_pool = np.array(
        [
            [3.5, 3.1, 2.4, 0.23, 0.18],
            [5.7, 4.8, 2.9, 0.34, 0.23],
        ],
        dtype=float,
    )
    cfg = RegionLiftConfig(enable_region_lifted_gp=True)
    bad_pref = parse_region_preference_payload(
        {
            "kind": "region",
            "coordinate_space": "normalized",
            "preference_direction": "promising",
            "lb": {"I1": 0.2, "I2": 0.2, "I3": 0.2, "dSOC1": 0.2, "dSOC2": 0.2},
            "ub": {"I1": 0.3, "I2": 0.3, "I3": 0.3, "dSOC1": 0.3, "dSOC2": 0.3},
            "confidence": 1.0,
        }
    )

    result = evaluate_region_lift_on_pool(
        gp=gp,
        candidate_pool=X_pool,
        f_min_y=0.55,
        preference=bad_pref,
        existing_X=np.empty((0, 5)),
        bounds=DEFAULT_BOUNDS,
        config=cfg,
        trust=1.0,
        bo_iteration=0,
    )

    assert result.accepted is False
    assert result.selected_source == "fallback"
    assert result.fallback_reason == "non_raw_coordinate_space"


def test_region_lift_inactive_anneal_falls_back() -> None:
    gp = _fit_gp()
    X_pool = np.array(
        [
            [3.5, 3.1, 2.4, 0.23, 0.18],
            [5.7, 4.8, 2.9, 0.34, 0.23],
        ],
        dtype=float,
    )
    cfg = RegionLiftConfig(
        enable_region_lifted_gp=True,
        region_lift_min_confidence=0.5,
        region_lift_min_volume=1e-6,
        region_lift_active_until=1,
    )

    result = evaluate_region_lift_on_pool(
        gp=gp,
        candidate_pool=X_pool,
        f_min_y=0.55,
        preference=_preference(),
        existing_X=np.empty((0, 5)),
        bounds=DEFAULT_BOUNDS,
        config=cfg,
        trust=1.0,
        bo_iteration=1,
    )

    assert result.accepted is False
    assert result.fallback_reason == "inactive_anneal"


def test_region_lift_zero_shift_falls_back() -> None:
    gp = _fit_gp()
    X_pool = np.array(
        [
            [3.5, 3.1, 2.4, 0.23, 0.18],
            [5.7, 4.8, 2.9, 0.34, 0.23],
        ],
        dtype=float,
    )
    cfg = RegionLiftConfig(
        enable_region_lifted_gp=True,
        region_lift_min_confidence=0.5,
        region_lift_min_volume=1e-6,
        region_lift_max_shift_std=0.0,
        region_lift_active_until=10,
    )

    result = evaluate_region_lift_on_pool(
        gp=gp,
        candidate_pool=X_pool,
        f_min_y=0.55,
        preference=_preference(),
        existing_X=np.empty((0, 5)),
        bounds=DEFAULT_BOUNDS,
        config=cfg,
        trust=1.0,
        bo_iteration=0,
    )

    assert result.accepted is False
    assert result.fallback_reason == "zero_shift"


def test_region_lift_same_as_plain_is_not_accepted() -> None:
    gp = _fit_gp()
    X_pool = np.array([[3.5, 3.1, 2.4, 0.23, 0.18]], dtype=float)
    cfg = RegionLiftConfig(
        enable_region_lifted_gp=True,
        region_lift_min_confidence=0.5,
        region_lift_min_volume=1e-6,
        region_lift_active_until=10,
        region_lift_max_plain_ei_gap=10.0,
    )

    result = evaluate_region_lift_on_pool(
        gp=gp,
        candidate_pool=X_pool,
        f_min_y=0.55,
        preference=_preference(),
        existing_X=np.empty((0, 5)),
        bounds=DEFAULT_BOUNDS,
        config=cfg,
        trust=1.0,
        bo_iteration=0,
    )

    assert result.accepted is False
    assert result.fallback_reason == "same_as_plain"


def test_degenerate_region_is_repaired_into_valid_candidates() -> None:
    pref = parse_region_preference_payload(
        {
            "kind": "region",
            "coordinate_space": "raw",
            "preference_direction": "promising",
            "lb": {"I1": 2.0, "I2": 2.0, "I3": 2.0, "dSOC1": 0.35, "dSOC2": 0.10},
            "ub": {"I1": 2.0, "I2": 2.0, "I3": 3.0, "dSOC1": 0.40, "dSOC2": 0.10},
            "confidence": 0.85,
        }
    )
    cfg = RegionLiftConfig(
        enable_region_lifted_gp=True,
        region_lift_min_confidence=0.5,
        region_lift_n_anchors=8,
        region_lift_min_volume=1e-6,
        region_lift_min_width=0.03,
    )

    candidates = sample_region_candidates(pref, DEFAULT_BOUNDS, cfg, n_candidates=8)

    assert candidates.shape[0] > 0
    assert candidates.shape[1] == 5


class _AnchorWeightedDummyGP:
    def predict_standardized(self, X: np.ndarray):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] == 3:
            mean = np.array([0.18, 0.70, 1.00], dtype=float)
            std = np.array([0.20, 0.05, 0.05], dtype=float)
        else:
            mean = np.array([0.55, 0.60], dtype=float)
            std = np.array([0.10, 0.10], dtype=float)
        return mean[: X.shape[0]], std[: X.shape[0]]

    def target_standardization(self):
        return 0.0, 1.0

    def posterior_covariance_standardized(self, X_left: np.ndarray, X_right=None):
        left = np.atleast_2d(np.asarray(X_left, dtype=float))
        right = left if X_right is None else np.atleast_2d(np.asarray(X_right, dtype=float))
        if left.shape[0] == 3 and right.shape[0] == 3:
            return np.eye(3, dtype=float)
        return np.array([[0.95, 0.10, 0.05], [0.20, 0.40, 0.10]], dtype=float)[: left.shape[0], : right.shape[0]]


class _CorrelationDummyGP:
    def predict_standardized(self, X: np.ndarray):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] == 3:
            return np.array([0.30, 0.31, 0.95], dtype=float), np.array([0.20, 0.20, 0.05], dtype=float)
        return np.array([0.45, 0.42], dtype=float), np.array([0.10, 0.10], dtype=float)

    def target_standardization(self):
        return 0.0, 1.0

    def posterior_covariance_standardized(self, X_left: np.ndarray, X_right=None):
        left = np.atleast_2d(np.asarray(X_left, dtype=float))
        right = left if X_right is None else np.atleast_2d(np.asarray(X_right, dtype=float))
        if left.shape[0] == 3 and right.shape[0] == 3:
            return np.eye(3, dtype=float)
        return np.array([[0.75, 0.72, 0.05], [0.20, 0.18, 0.05]], dtype=float)[: left.shape[0], : right.shape[0]]


def test_anchor_weighting_uses_gp_signal_inside_region() -> None:
    gp = _AnchorWeightedDummyGP()
    X_pool = np.array(
        [
            [3.20, 3.00, 2.20, 0.18, 0.14],
            [4.80, 4.00, 2.70, 0.26, 0.20],
        ],
        dtype=float,
    )
    cfg = RegionLiftConfig(
        enable_region_lifted_gp=True,
        region_lift_min_confidence=0.5,
        region_lift_n_anchors=3,
        region_lift_min_volume=1e-6,
        region_lift_min_width=0.03,
        region_lift_anchor_weighting="ei_softmax",
        region_lift_anchor_temperature=0.15,
        region_lift_active_until=10,
        region_lift_max_plain_ei_gap=10.0,
    )
    pref = parse_region_preference_payload(
        {
            "kind": "region",
            "coordinate_space": "raw",
            "preference_direction": "promising",
            "lb": {"I1": 3.0, "I2": 2.8, "I3": 2.2, "dSOC1": 0.18, "dSOC2": 0.14},
            "ub": {"I1": 3.8, "I2": 3.4, "I3": 2.5, "dSOC1": 0.24, "dSOC2": 0.18},
            "confidence": 0.9,
        }
    )

    result = evaluate_region_lift_on_pool(
        gp=gp,
        candidate_pool=X_pool,
        f_min_y=0.30,
        preference=pref,
        existing_X=np.empty((0, 5)),
        bounds=DEFAULT_BOUNDS,
        config=cfg,
        trust=1.0,
        bo_iteration=0,
    )

    assert result.telemetry["anchor_weighting_mode"] == "ei_softmax"
    assert result.telemetry["anchor_weight_max"] > (1.0 / 3.0)


def test_normalized_correlation_lift_prefers_more_correlated_candidate() -> None:
    gp = _CorrelationDummyGP()
    X_pool = np.array(
        [
            [3.20, 3.00, 2.20, 0.18, 0.14],
            [4.10, 3.80, 2.60, 0.22, 0.18],
        ],
        dtype=float,
    )
    cfg = RegionLiftConfig(
        enable_region_lifted_gp=True,
        region_lift_min_confidence=0.5,
        region_lift_n_anchors=3,
        region_lift_min_volume=1e-6,
        region_lift_min_width=0.03,
        region_lift_active_until=10,
        region_lift_max_plain_ei_gap=10.0,
        region_lift_require_inside=False,
        region_lift_min_sigma_ratio=0.0,
    )
    pref = parse_region_preference_payload(
        {
            "kind": "region",
            "coordinate_space": "raw",
            "preference_direction": "promising",
            "lb": {"I1": 3.0, "I2": 2.8, "I3": 2.2, "dSOC1": 0.18, "dSOC2": 0.14},
            "ub": {"I1": 3.8, "I2": 3.4, "I3": 2.5, "dSOC1": 0.24, "dSOC2": 0.18},
            "confidence": 0.9,
        }
    )

    result = evaluate_region_lift_on_pool(
        gp=gp,
        candidate_pool=X_pool,
        f_min_y=0.30,
        preference=pref,
        existing_X=np.empty((0, 5)),
        bounds=DEFAULT_BOUNDS,
        config=cfg,
        trust=1.0,
        bo_iteration=0,
    )

    assert result.telemetry["max_corr"] >= result.telemetry["corr_at_plain"]
    assert result.telemetry["region_reliability"] > 0.0


def test_region_candidates_respect_dsoc_interior_margin() -> None:
    pref = parse_region_preference_payload(
        {
            "kind": "region",
            "coordinate_space": "raw",
            "preference_direction": "promising",
            "lb": {"I1": 3.0, "I2": 2.8, "I3": 2.2, "dSOC1": 0.31, "dSOC2": 0.30},
            "ub": {"I1": 4.0, "I2": 3.8, "I3": 2.7, "dSOC1": 0.40, "dSOC2": 0.34},
            "confidence": 0.9,
        }
    )
    cfg = RegionLiftConfig(
        enable_region_lifted_gp=True,
        region_lift_min_confidence=0.5,
        region_lift_n_anchors=8,
        region_lift_min_volume=1e-6,
        region_lift_min_width=0.03,
        region_lift_dsoc_margin=0.02,
    )

    candidates = sample_region_candidates(pref, DEFAULT_BOUNDS, cfg, n_candidates=8)

    assert len(candidates) > 0
    assert np.all(candidates[:, 3] + candidates[:, 4] <= DSOC_SUM_MAX - 0.02 + 1e-9)


def test_region_lift_default_off_in_mainline_presets() -> None:
    mainline = BayesOptimizer(config={"experiment_preset": "warmstart_plain_ei"})
    strict = BayesOptimizer(config={"experiment_preset": "strict_baseline"})

    assert mainline.cfg["enable_region_lifted_gp"] is False
    assert strict.cfg["enable_region_lifted_gp"] is False


def test_point_current_probe_candidates_push_currents_toward_upper_bounds() -> None:
    bo = BayesOptimizer(config={"experiment_preset": "warmstart_region_lifted_gp_force_pool_tuned"})
    pref = _point_preference()

    probes = bo._build_point_current_probe_candidates(pref)

    assert len(probes) == 3
    current_tuples = {tuple(np.round(np.asarray(p)[:3], 6)) for p in probes}
    assert (6.0, 5.0, 3.0) in current_tuples
    assert any(row[0] > 2.0 and row[1] > 2.0 for row in current_tuples)


def test_sample_region_candidates_force_keeps_high_current_point_probes(monkeypatch) -> None:
    bo = BayesOptimizer(config={"experiment_preset": "warmstart_region_lifted_gp_force_pool_tuned"})
    pref = _point_preference()
    bo.gp = object()
    bo.database = SimpleNamespace(size=0, get_all=lambda: [])

    monkeypatch.setattr(
        bo,
        "_rank_region_candidates_with_gp",
        lambda candidates, max_keep: [np.asarray(candidates[0], dtype=float).copy()],
    )

    X = bo._sample_region_candidates_from_preference(preference=pref, t=0)
    rows = {tuple(np.round(row[:3], 6)) for row in np.atleast_2d(X)}

    assert (6.0, 5.0, 3.0) in rows
    assert len(rows) >= 2


def test_region_lift_force_pool_candidates_are_injected_before_af_step_and_shared_with_eval(monkeypatch) -> None:
    bo = BayesOptimizer(
        config={
            "experiment_preset": "warmstart_region_lifted_gp",
            "region_lift_external_influence_mode": "force_pool",
            "max_iterations": 1,
            "checkpoint_every": 99,
        }
    )
    bo.setup()
    observations = [
        (np.array([4.0, 3.5, 2.5, 0.25, 0.20], dtype=float), np.array([4200.0, 4.2, 0.80], dtype=float)),
        (np.array([5.0, 4.0, 2.8, 0.20, 0.18], dtype=float), np.array([3600.0, 4.0, 0.70], dtype=float)),
        (np.array([3.6, 3.1, 2.2, 0.30, 0.20], dtype=float), np.array([5000.0, 3.6, 0.60], dtype=float)),
    ]
    for theta, objectives in observations:
        bo.database.add_observation(theta=theta, objectives=objectives, feasible=True, source="test")
    bo.initialize_acquisition()

    region_candidates = np.array(
        [
            [3.30, 3.00, 2.30, 0.22, 0.16],
            [3.80, 3.30, 2.45, 0.25, 0.18],
        ],
        dtype=float,
    )
    captured: dict = {}

    monkeypatch.setattr(bo.llm, "query_region_preference", lambda state: _preference())
    monkeypatch.setattr(
        bo,
        "_sample_region_candidates_from_preference",
        lambda preference, t: region_candidates.copy(),
    )

    def fake_step(*, X_candidates, X_external_restarts=None, database, t, w_vec, lift, prior):
        X = np.atleast_2d(np.asarray(X_candidates, dtype=float))
        captured["step_candidates"] = X.copy()
        captured["step_external_restarts"] = (
            None if X_external_restarts is None else np.atleast_2d(np.asarray(X_external_restarts, dtype=float)).copy()
        )
        n = X.shape[0]
        return AcquisitionResult(
            selected_thetas=[X[0].copy()],
            selected_indices=[0],
            selected_scores=np.array([1.0], dtype=float),
            all_alpha=np.ones(n, dtype=float),
            all_ei=np.ones(n, dtype=float),
            all_wcharge=np.ones(n, dtype=float),
            all_mean=np.zeros(n, dtype=float),
            all_std=np.ones(n, dtype=float),
            state=bo.af.get_state(),
            debug={},
            all_mean_base=np.zeros(n, dtype=float),
            candidate_pool=X.copy(),
        )

    monkeypatch.setattr(bo.af, "step", fake_step)

    def fake_eval_region(**kwargs):
        captured["lift_pool"] = np.asarray(kwargs["candidate_pool"], dtype=float).copy()
        return RegionLiftResult(
            selected_index=0,
            selected_source="fallback",
            accepted=False,
            fallback_reason="unit_test",
            telemetry={
                "active": True,
                "accepted": False,
                "selected_source": "fallback",
                "fallback_reason": "unit_test",
            },
        )

    monkeypatch.setattr(optimizer_module, "evaluate_region_lift_on_pool", fake_eval_region)
    bo.simulator = SimpleNamespace(
        evaluate=lambda theta: {
            "raw_objectives": np.array([3500.0, 3.9, 0.65], dtype=float),
            "feasible": True,
        }
    )

    bo.run_optimization_loop()

    assert np.allclose(captured["step_candidates"], region_candidates)
    assert captured["step_external_restarts"] is None
    assert np.allclose(captured["lift_pool"], captured["step_candidates"])


def test_region_lift_diagnostic_only_keeps_region_candidates_out_of_af_step(monkeypatch) -> None:
    bo = BayesOptimizer(
        config={
            "experiment_preset": "warmstart_region_lifted_gp",
            "region_lift_external_influence_mode": "diagnostic_only",
            "max_iterations": 1,
            "checkpoint_every": 99,
        }
    )
    bo.setup()
    observations = [
        (np.array([4.0, 3.5, 2.5, 0.25, 0.20], dtype=float), np.array([4200.0, 4.2, 0.80], dtype=float)),
        (np.array([5.0, 4.0, 2.8, 0.20, 0.18], dtype=float), np.array([3600.0, 4.0, 0.70], dtype=float)),
        (np.array([3.6, 3.1, 2.2, 0.30, 0.20], dtype=float), np.array([5000.0, 3.6, 0.60], dtype=float)),
    ]
    for theta, objectives in observations:
        bo.database.add_observation(theta=theta, objectives=objectives, feasible=True, source="test")
    bo.initialize_acquisition()

    region_candidates = np.array(
        [
            [3.30, 3.00, 2.30, 0.22, 0.16],
            [3.80, 3.30, 2.45, 0.25, 0.18],
        ],
        dtype=float,
    )
    base_pool = np.array(
        [
            [3.10, 2.90, 2.20, 0.21, 0.16],
            [3.90, 3.40, 2.45, 0.26, 0.18],
        ],
        dtype=float,
    )
    captured: dict = {}

    monkeypatch.setattr(bo.llm, "query_region_preference", lambda state: _preference())
    monkeypatch.setattr(
        bo,
        "_sample_region_candidates_from_preference",
        lambda preference, t: region_candidates.copy(),
    )

    def fake_step(*, X_candidates, X_external_restarts=None, database, t, w_vec, lift, prior):
        captured["step_candidates"] = None if X_candidates is None else np.atleast_2d(np.asarray(X_candidates, dtype=float)).copy()
        captured["step_external_restarts"] = (
            None if X_external_restarts is None else np.atleast_2d(np.asarray(X_external_restarts, dtype=float)).copy()
        )
        n = base_pool.shape[0]
        return AcquisitionResult(
            selected_thetas=[base_pool[0].copy()],
            selected_indices=[0],
            selected_scores=np.array([1.0], dtype=float),
            all_alpha=np.ones(n, dtype=float),
            all_ei=np.ones(n, dtype=float),
            all_wcharge=np.ones(n, dtype=float),
            all_mean=np.zeros(n, dtype=float),
            all_std=np.ones(n, dtype=float),
            state=bo.af.get_state(),
            debug={},
            all_mean_base=np.zeros(n, dtype=float),
            candidate_pool=base_pool.copy(),
        )

    monkeypatch.setattr(bo.af, "step", fake_step)

    def fake_eval_region(**kwargs):
        captured["lift_pool"] = np.asarray(kwargs["candidate_pool"], dtype=float).copy()
        return RegionLiftResult(
            selected_index=0,
            selected_source="fallback",
            accepted=False,
            fallback_reason="unit_test",
            telemetry={
                "active": True,
                "accepted": False,
                "selected_source": "fallback",
                "fallback_reason": "unit_test",
            },
        )

    monkeypatch.setattr(optimizer_module, "evaluate_region_lift_on_pool", fake_eval_region)
    bo.simulator = SimpleNamespace(
        evaluate=lambda theta: {
            "raw_objectives": np.array([3500.0, 3.9, 0.65], dtype=float),
            "feasible": True,
        }
    )

    bo.run_optimization_loop()

    assert captured["step_candidates"] is None
    assert captured["step_external_restarts"] is None
    assert captured["lift_pool"].shape[0] == 4
    assert np.allclose(captured["lift_pool"][:2], base_pool)


def test_region_lift_force_pool_restart_only_keeps_region_candidates_out_of_raw_pool(monkeypatch) -> None:
    bo = BayesOptimizer(
        config={
            "experiment_preset": "warmstart_region_lifted_gp_force_pool_tuned",
            "max_iterations": 1,
            "checkpoint_every": 99,
        }
    )
    bo.setup()
    observations = [
        (np.array([4.0, 3.5, 2.5, 0.25, 0.20], dtype=float), np.array([4200.0, 4.2, 0.80], dtype=float)),
        (np.array([5.0, 4.0, 2.8, 0.20, 0.18], dtype=float), np.array([3600.0, 4.0, 0.70], dtype=float)),
        (np.array([3.6, 3.1, 2.2, 0.30, 0.20], dtype=float), np.array([5000.0, 3.6, 0.60], dtype=float)),
    ]
    for theta, objectives in observations:
        bo.database.add_observation(theta=theta, objectives=objectives, feasible=True, source="test")
    bo.initialize_acquisition()

    region_candidates = np.array(
        [
            [3.30, 3.00, 2.30, 0.22, 0.16],
            [3.80, 3.30, 2.45, 0.25, 0.18],
        ],
        dtype=float,
    )
    base_pool = np.array(
        [
            [3.10, 2.90, 2.20, 0.21, 0.16],
            [3.90, 3.40, 2.45, 0.26, 0.18],
        ],
        dtype=float,
    )
    captured: dict = {}

    monkeypatch.setattr(bo.llm, "query_region_preference", lambda state: _preference())
    monkeypatch.setattr(
        bo,
        "_sample_region_candidates_from_preference",
        lambda preference, t: region_candidates.copy(),
    )

    def fake_step(*, X_candidates, X_external_restarts=None, database, t, w_vec, lift, prior):
        captured["step_candidates"] = None if X_candidates is None else np.atleast_2d(np.asarray(X_candidates, dtype=float)).copy()
        captured["step_external_restarts"] = (
            None if X_external_restarts is None else np.atleast_2d(np.asarray(X_external_restarts, dtype=float)).copy()
        )
        n = base_pool.shape[0]
        return AcquisitionResult(
            selected_thetas=[base_pool[0].copy()],
            selected_indices=[0],
            selected_scores=np.array([1.0], dtype=float),
            all_alpha=np.ones(n, dtype=float),
            all_ei=np.ones(n, dtype=float),
            all_wcharge=np.ones(n, dtype=float),
            all_mean=np.zeros(n, dtype=float),
            all_std=np.ones(n, dtype=float),
            state=bo.af.get_state(),
            debug={},
            all_mean_base=np.zeros(n, dtype=float),
            candidate_pool=base_pool.copy(),
        )

    monkeypatch.setattr(bo.af, "step", fake_step)

    def fake_eval_region(**kwargs):
        captured["lift_pool"] = np.asarray(kwargs["candidate_pool"], dtype=float).copy()
        return RegionLiftResult(
            selected_index=0,
            selected_source="fallback",
            accepted=False,
            fallback_reason="unit_test",
            telemetry={
                "active": True,
                "accepted": False,
                "selected_source": "fallback",
                "fallback_reason": "unit_test",
            },
        )

    monkeypatch.setattr(optimizer_module, "evaluate_region_lift_on_pool", fake_eval_region)
    bo.simulator = SimpleNamespace(
        evaluate=lambda theta: {
            "raw_objectives": np.array([3500.0, 3.9, 0.65], dtype=float),
            "feasible": True,
        }
    )

    bo.run_optimization_loop()

    assert captured["step_candidates"] is None
    assert np.allclose(captured["step_external_restarts"], region_candidates)
    assert captured["lift_pool"].shape[0] == 4
    assert np.allclose(captured["lift_pool"][:2], base_pool)


def test_region_lift_summary_counts_use_accepted_flag() -> None:
    bo = BayesOptimizer(config={"experiment_preset": "warmstart_region_lifted_gp"})
    bo._region_lift_telemetry = [
        {"accepted": False, "selected_source": "lifted", "fallback_reason": "same_as_plain", "hv_gain_raw": 0.0},
        {"accepted": True, "selected_source": "lifted", "fallback_reason": None, "hv_gain_raw": 0.1},
    ]

    summary = bo._summarize_region_lift_telemetry()

    assert summary["region_lift_attempt_count"] == 2
    assert summary["region_lift_accept_count"] == 1
    assert summary["lift_accept_rate"] == 0.5


def test_external_candidates_do_not_consume_internal_restart_budget(monkeypatch) -> None:
    af = AcquisitionFunction(
        gp=object(),
        param_bounds=DEFAULT_BOUNDS,
        n_restarts_optimizer=2,
        n_random_candidates=0,
        n_external_local_restarts=1,
        random_seed=0,
    )
    theta_best = np.array([4.0, 3.5, 2.5, 0.24, 0.18], dtype=float)
    g1 = np.array([4.1, 3.6, 2.55, 0.24, 0.18], dtype=float)
    g2 = np.array([4.2, 3.7, 2.60, 0.24, 0.18], dtype=float)
    e1 = np.array([3.2, 2.9, 2.2, 0.20, 0.16], dtype=float)
    e2 = np.array([3.4, 3.1, 2.3, 0.22, 0.16], dtype=float)
    optimized_from = []

    monkeypatch.setattr(af, "_sample_gaussian", lambda n, mu, sigma: [g1.copy(), g2.copy()])
    monkeypatch.setattr(af, "_sample_uniform", lambda n: [])

    def fake_optimize(seed: np.ndarray, f_min: float, lift=None) -> np.ndarray:
        optimized_from.append(np.asarray(seed, dtype=float).copy())
        return np.asarray(seed, dtype=float).copy()

    monkeypatch.setattr(af, "_optimize_from_seed", fake_optimize)

    pool = af._build_candidate_pool(theta_best, np.vstack([e1, e2]), f_min=0.2, lift=None)

    assert len(optimized_from) == 3
    assert np.allclose(optimized_from[0], theta_best)
    assert np.allclose(optimized_from[1], g1)
    assert np.allclose(optimized_from[2], e1)
    assert any(np.allclose(row, e2) for row in pool)


def test_region_lift_override_disabled_keeps_plain_selection(monkeypatch) -> None:
    bo = BayesOptimizer(
        config={
            "experiment_preset": "warmstart_region_lifted_gp",
            "region_lift_apply_override": False,
        }
    )
    bo.database = SimpleNamespace(size=0, get_all=lambda: [], get_f_min=lambda: 1.0)
    bo.gp = object()
    pref = _preference()
    X = np.array(
        [
            [3.5, 3.1, 2.4, 0.23, 0.18],
            [3.7, 3.2, 2.45, 0.24, 0.18],
        ],
        dtype=float,
    )
    acq_result = AcquisitionResult(
        selected_thetas=[X[0].copy()],
        selected_indices=[0],
        selected_scores=np.array([1.0], dtype=float),
        all_alpha=np.array([1.0, 0.9], dtype=float),
        all_ei=np.array([1.0, 0.9], dtype=float),
        all_wcharge=np.ones(2, dtype=float),
        all_mean=np.zeros(2, dtype=float),
        all_std=np.ones(2, dtype=float),
        state=SimpleNamespace(),
        debug={},
        all_mean_base=np.zeros(2, dtype=float),
        candidate_pool=X.copy(),
    )

    monkeypatch.setattr(
        optimizer_module,
        "evaluate_region_lift_on_pool",
        lambda **kwargs: RegionLiftResult(
            selected_index=1,
            selected_source="lifted",
            accepted=True,
            fallback_reason=None,
            telemetry={
                "active": True,
                "selected_source": "lifted",
                "fallback_reason": None,
                "plain_candidate_inside_region": True,
            },
        ),
    )

    out = bo._maybe_apply_region_lifted_gp(
        t=0,
        w_vec=np.array([1.0, 0.0, 0.0], dtype=float),
        scalar_y=np.array([1.0], dtype=float),
        ideal_point_raw=np.zeros(3, dtype=float),
        acq_result=acq_result,
        plain_selected_indices=[0],
        plain_selected_scores=np.array([1.0], dtype=float),
        preference=pref,
        diagnostic_region_candidates=np.empty((0, 5), dtype=float),
        region_pool_influenced_acquisition=False,
        region_influence_mode="diagnostic_only",
    )

    assert out.selected_indices == [0]
    assert bo._last_region_lift_summary["accepted"] is False
    assert bo._last_region_lift_summary["selected_source"] == "plain_ei"
    assert bo._last_region_lift_summary["fallback_reason"] == "override_disabled"
    assert bo._last_region_lift_summary["diagnostic_override_candidate_available"] is True


def test_region_lift_override_can_select_diagnostic_region_candidate(monkeypatch) -> None:
    bo = BayesOptimizer(
        config={
            "experiment_preset": "warmstart_region_lifted_gp",
            "region_lift_apply_override": True,
            "region_lift_override_uses_diagnostic_pool": True,
        }
    )
    bo.database = SimpleNamespace(size=0, get_all=lambda: [], get_f_min=lambda: 1.0)
    bo.gp = object()
    pref = _preference()
    X = np.array(
        [
            [3.5, 3.1, 2.4, 0.23, 0.18],
            [3.7, 3.2, 2.45, 0.24, 0.18],
        ],
        dtype=float,
    )
    region_candidate = np.array([[4.2, 3.4, 2.55, 0.22, 0.16]], dtype=float)
    acq_result = AcquisitionResult(
        selected_thetas=[X[0].copy()],
        selected_indices=[0],
        selected_scores=np.array([1.0], dtype=float),
        all_alpha=np.array([1.0, 0.9], dtype=float),
        all_ei=np.array([1.0, 0.9], dtype=float),
        all_wcharge=np.ones(2, dtype=float),
        all_mean=np.zeros(2, dtype=float),
        all_std=np.ones(2, dtype=float),
        state=SimpleNamespace(),
        debug={},
        all_mean_base=np.zeros(2, dtype=float),
        candidate_pool=X.copy(),
    )

    def fake_region_lift(**kwargs):
        pool = kwargs["candidate_pool"]
        assert pool.shape[0] == 3
        assert np.allclose(pool[2], region_candidate[0])
        return RegionLiftResult(
            selected_index=2,
            selected_source="lifted",
            accepted=True,
            fallback_reason=None,
            telemetry={
                "active": True,
                "selected_source": "lifted",
                "fallback_reason": None,
                "lifted_ei_at_lift": 1.25,
                "plain_candidate_inside_region": False,
            },
        )

    monkeypatch.setattr(optimizer_module, "evaluate_region_lift_on_pool", fake_region_lift)

    out = bo._maybe_apply_region_lifted_gp(
        t=0,
        w_vec=np.array([1.0, 0.0, 0.0], dtype=float),
        scalar_y=np.array([1.0], dtype=float),
        ideal_point_raw=np.zeros(3, dtype=float),
        acq_result=acq_result,
        plain_selected_indices=[0],
        plain_selected_scores=np.array([1.0], dtype=float),
        preference=pref,
        diagnostic_region_candidates=region_candidate,
        region_pool_influenced_acquisition=False,
        region_influence_mode="diagnostic_only",
    )

    assert out.selected_indices == [2]
    assert np.allclose(out.selected_thetas[0], region_candidate[0])
    assert np.allclose(out.selected_scores, np.array([1.25]))
    assert bo._last_region_lift_summary["accepted"] is True
    assert bo._last_region_lift_summary["override_uses_diagnostic_pool"] is True
    assert bo._last_region_lift_summary["selection_candidate_pool_size"] == 3


def test_region_lift_inactive_window_skips_region_sampling(monkeypatch) -> None:
    bo = BayesOptimizer(
        config={
            "experiment_preset": "warmstart_region_lifted_gp",
            "region_lift_active_until": 0,
            "max_iterations": 1,
            "checkpoint_every": 99,
        }
    )
    bo.setup()
    observations = [
        (np.array([4.0, 3.5, 2.5, 0.25, 0.20], dtype=float), np.array([4200.0, 4.2, 0.80], dtype=float)),
        (np.array([5.0, 4.0, 2.8, 0.20, 0.18], dtype=float), np.array([3600.0, 4.0, 0.70], dtype=float)),
        (np.array([3.6, 3.1, 2.2, 0.30, 0.20], dtype=float), np.array([5000.0, 3.6, 0.60], dtype=float)),
    ]
    for theta, objectives in observations:
        bo.database.add_observation(theta=theta, objectives=objectives, feasible=True, source="test")
    bo.initialize_acquisition()

    monkeypatch.setattr(
        bo.llm,
        "query_region_preference",
        lambda state: (_ for _ in ()).throw(AssertionError("region query should be skipped")),
    )
    monkeypatch.setattr(
        bo,
        "_sample_region_candidates_from_preference",
        lambda preference, t: (_ for _ in ()).throw(AssertionError("region sampling should be skipped")),
    )
    bo.simulator = SimpleNamespace(
        evaluate=lambda theta: {
            "raw_objectives": np.array([3500.0, 3.9, 0.65], dtype=float),
            "feasible": True,
        }
    )

    bo.run_optimization_loop()

    assert "region" not in bo._last_candidate_source_counts
    assert bo._last_region_lift_summary["fallback_reason"] == "inactive_window_skipped"
    assert bo._last_region_lift_summary["inactive_window_skipped"] is True


def test_guarded_pool_requires_previous_gate_pass() -> None:
    bo = BayesOptimizer(
        config={
            "experiment_preset": "warmstart_region_lifted_gp_guarded_pool",
            "region_lift_guard_min_anchor_consistency": 0.35,
            "region_lift_guard_min_reliability": 0.20,
            "region_lift_guard_max_plain_ei_gap": 0.25,
        }
    )

    assert bo._should_influence_acquisition_with_region(t=0) is False

    bo._last_region_lift_summary = {
        "lambda_t": 0.1,
        "diagnostic_override_candidate_available": True,
        "lift_candidate_inside_region": True,
        "plain_ei_gap": 0.10,
        "anchor_consistency": 0.50,
        "region_reliability": 0.40,
        "corr_at_lift": 0.25,
        "inactive_window_skipped": False,
        "region_influence_gate_passed": True,
    }
    bo._update_region_influence_gate_from_summary()

    assert bo._should_influence_acquisition_with_region(t=1) is True

    bo._last_region_lift_summary = {
        "lambda_t": 0.1,
        "diagnostic_override_candidate_available": True,
        "lift_candidate_inside_region": True,
        "plain_ei_gap": 0.50,
        "anchor_consistency": 0.50,
        "region_reliability": 0.40,
        "corr_at_lift": 0.25,
        "inactive_window_skipped": False,
        "region_influence_gate_passed": False,
    }
    bo._update_region_influence_gate_from_summary()

    assert bo._should_influence_acquisition_with_region(t=1) is False
