from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmbo.acquisition import AcquisitionFunction, AcquisitionPrior
from utils.constants import DEFAULT_BOUNDS


PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]


class DummyDatabase:
    def __init__(self, theta_best: np.ndarray, f_min: float, stagnation_count: int = 0) -> None:
        self._theta_best = np.asarray(theta_best, dtype=float)
        self._f_min = float(f_min)
        self._stagnation_count = int(stagnation_count)

    def get_f_min(self) -> float:
        return self._f_min

    def get_theta_best(self) -> np.ndarray:
        return self._theta_best.copy()

    def has_improved(self) -> bool:
        return False

    def get_stagnation_count(self) -> int:
        return self._stagnation_count


class DummyGP:
    def __init__(self, lookup: dict[tuple[float, ...], tuple[float, float]]) -> None:
        self.lookup = dict(lookup)

    def _query(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X_new, dtype=float))
        mean = []
        std = []
        for row in X:
            key = tuple(np.round(row, 6))
            m, s = self.lookup.get(key, (1.0, 0.01))
            mean.append(float(m))
            std.append(float(s))
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._query(X_new)

    def predict_with_coupling(self, X_new: np.ndarray, coupling=None) -> tuple[np.ndarray, np.ndarray]:
        return self._query(X_new)


def test_acquisition_prior_guidance_bonus_and_risk_penalty() -> None:
    center = np.array([5.0, 4.2, 2.9, 0.22, 0.18], dtype=float)
    far = np.array([2.8, 2.4, 2.1, 0.34, 0.31], dtype=float)
    safe = np.array([4.8, 4.0, 2.8, 0.23, 0.18], dtype=float)
    risky = np.array([5.1, 4.6, 3.0, 0.35, 0.33], dtype=float)

    prior = AcquisitionPrior(
        guidance_alpha=0.4,
        guidance_mode="point",
        guidance_center=center,
        guidance_sigma=np.array([0.3, 0.3, 0.2, 0.05, 0.05], dtype=float),
        safe_dsoc_sum_max=0.65,
        hard_dsoc_sum_max=0.70,
        safe_risk_weight=0.5,
        hard_risk_weight=3.0,
        monotone_risk_weight=0.5,
    )

    bonus = prior.bonus(np.vstack([center, far]))
    risk = prior.risk(np.vstack([safe, risky]))

    assert bonus[0] > bonus[1]
    assert risk[1] > risk[0]


def test_acquisition_step_uses_prior_risk_to_avoid_boundary_candidate() -> None:
    theta_best = np.array([4.6, 3.9, 2.7, 0.22, 0.18], dtype=float)
    safe = np.array([4.8, 4.0, 2.8, 0.23, 0.18], dtype=float)
    risky = np.array([5.0, 4.8, 3.0, 0.36, 0.30], dtype=float)

    lookup = {
        tuple(np.round(theta_best, 6)): (0.380, 0.05),
        tuple(np.round(safe, 6)): (0.350, 0.05),
        tuple(np.round(risky, 6)): (0.340, 0.05),
    }
    gp = DummyGP(lookup)
    af = AcquisitionFunction(
        gp=gp,
        param_bounds=DEFAULT_BOUNDS,
        n_select=1,
        n_restarts_optimizer=0,
        n_random_candidates=0,
        random_seed=0,
    )
    database = DummyDatabase(theta_best=theta_best, f_min=0.36)

    baseline = af.step(
        X_candidates=np.vstack([safe, risky]),
        database=database,
        t=0,
    )
    assert np.allclose(baseline.selected_thetas[0], risky)

    prior = AcquisitionPrior(
        safe_dsoc_sum_max=0.65,
        hard_dsoc_sum_max=0.70,
        safe_risk_weight=4.0,
        hard_risk_weight=3.0,
        monotone_risk_weight=0.2,
    )
    guided = af.step(
        X_candidates=np.vstack([safe, risky]),
        database=database,
        t=0,
        prior=prior,
    )

    assert np.allclose(guided.selected_thetas[0], safe)
    assert guided.all_risk_penalty is not None
    assert guided.all_risk_penalty[guided.selected_indices[0]] <= guided.all_risk_penalty.max()
