from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as scipy_norm

from llmbo.gp_model import GPProtocol
from utils.constants import DSOC_SUM_MAX as CANONICAL_DSOC_SUM_MAX

logger = logging.getLogger(__name__)

PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]
DSOC_SUM_MAX = CANONICAL_DSOC_SUM_MAX


@runtime_checkable
class DatabaseProtocol(Protocol):
    def get_f_min(self) -> float:
        ...

    def get_theta_best(self) -> np.ndarray:
        ...

    def has_improved(self) -> bool:
        ...

    def get_stagnation_count(self) -> int:
        ...


@runtime_checkable
class LLMPriorProtocol(Protocol):
    def get_warmstart_center(self) -> Optional[np.ndarray]:
        ...


@dataclasses.dataclass
class AcquisitionState:
    mu: np.ndarray
    sigma: np.ndarray
    alpha_t: float
    stagnation_count: int
    t: int
    f_min: float
    theta_best: np.ndarray
    grad_psi_at_best: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "alpha_t": float(self.alpha_t),
            "stagnation_count": int(self.stagnation_count),
            "t": int(self.t),
            "f_min": float(self.f_min),
            "theta_best": self.theta_best.tolist(),
            "grad_psi_at_best": self.grad_psi_at_best.tolist(),
        }


@dataclasses.dataclass
class AcquisitionResult:
    selected_thetas: List[np.ndarray]
    selected_indices: List[int]
    selected_scores: np.ndarray
    all_alpha: np.ndarray
    all_ei: np.ndarray
    all_wcharge: np.ndarray
    all_mean: np.ndarray
    all_std: np.ndarray
    state: AcquisitionState
    debug: Dict[str, Any]
    all_mean_base: Optional[np.ndarray] = None
    lift_summary: Optional[Dict[str, Any]] = None


class AcquisitionFunction:
    """Plain EI optimizer with multi-start L-BFGS-B in 5D."""

    def __init__(
        self,
        gp: GPProtocol,
        param_bounds: Dict[str, Tuple[float, float]],
        n_select: int = 1,
        n_restarts_optimizer: int = 16,
        n_random_candidates: int = 128,
        random_seed: Optional[int] = None,
    ) -> None:
        self.gp = gp
        self.param_bounds = param_bounds
        self.n_select = int(n_select)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.n_random_candidates = int(n_random_candidates)
        self._rng = np.random.default_rng(random_seed)

        self._lo = np.array([param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        self._hi = np.array([param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        self._bounds = list(zip(self._lo.tolist(), self._hi.tolist()))

        center = (self._lo + self._hi) / 2.0
        spread = np.maximum((self._hi - self._lo) * 0.15, 1e-3)
        self._state = AcquisitionState(
            mu=center.copy(),
            sigma=spread,
            alpha_t=0.0,
            stagnation_count=0,
            t=0,
            f_min=float("inf"),
            theta_best=center.copy(),
            grad_psi_at_best=np.zeros(len(PARAM_KEYS), dtype=float),
        )

    def initialize(
        self,
        database: DatabaseProtocol,
        llm_prior: Optional[LLMPriorProtocol] = None,
    ) -> None:
        theta_best = np.asarray(database.get_theta_best(), dtype=float).ravel()
        mu = theta_best.copy()
        if llm_prior is not None:
            try:
                center = llm_prior.get_warmstart_center()
            except Exception:
                center = None
            if center is not None:
                center = np.asarray(center, dtype=float).ravel()
                if center.size == len(PARAM_KEYS):
                    mu = self._repair_theta(center)

        self._state = AcquisitionState(
            mu=mu,
            sigma=np.maximum((self._hi - self._lo) * 0.15, 1e-3),
            alpha_t=0.0,
            stagnation_count=int(database.get_stagnation_count()),
            t=0,
            f_min=float(database.get_f_min()),
            theta_best=theta_best.copy(),
            grad_psi_at_best=np.zeros(len(PARAM_KEYS), dtype=float),
        )

    def step(
        self,
        X_candidates: Optional[np.ndarray] = None,
        database: Optional[DatabaseProtocol] = None,
        t: int = 0,
        w_vec: Optional[np.ndarray] = None,
        lift: Optional[Any] = None,
    ) -> AcquisitionResult:
        if database is None:
            raise ValueError("database is required")

        f_min = float(database.get_f_min())
        theta_best = self._repair_theta(database.get_theta_best())
        stagnation_count = int(database.get_stagnation_count())

        sigma_scale = 1.0 + 0.20 * min(stagnation_count, 3)
        self._state = AcquisitionState(
            mu=theta_best.copy(),
            sigma=np.maximum((self._hi - self._lo) * 0.15 * sigma_scale, 1e-3),
            alpha_t=0.0,
            stagnation_count=stagnation_count,
            t=int(t),
            f_min=f_min,
            theta_best=theta_best.copy(),
            grad_psi_at_best=np.zeros(len(PARAM_KEYS), dtype=float),
        )

        candidate_pool = self._build_candidate_pool(theta_best, X_candidates, f_min, lift=lift)
        mean, std = self.gp.predict_with_coupling(candidate_pool, coupling=lift)
        mean_base = self.gp.predict(candidate_pool)[0] if lift is not None else mean.copy()
        ei = expected_improvement(mean, std, f_min)
        wcharge = np.ones_like(ei)
        score = ei * wcharge

        if np.all(score <= 1e-12):
            logger.info("EI surface is flat; falling back to max-uncertainty selection")
            score = std.copy()

        selected_indices = self._select_top_unique(candidate_pool, score, self.n_select)
        selected_thetas = [candidate_pool[i].copy() for i in selected_indices]
        selected_scores = score[selected_indices]

        return AcquisitionResult(
            selected_thetas=selected_thetas,
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            all_alpha=score,
            all_ei=ei,
            all_wcharge=wcharge,
            all_mean=mean,
            all_std=std,
            state=self.get_state(),
            debug={
                "n_pool": int(candidate_pool.shape[0]),
                "best_score": float(np.max(score)) if len(score) else 0.0,
                "best_ei": float(np.max(ei)) if len(ei) else 0.0,
                "stagnation_count": stagnation_count,
                "gp_llm_coupling": lift is not None,
            },
            all_mean_base=mean_base,
            lift_summary=lift.to_dict() if lift is not None else None,
        )

    def get_state(self) -> AcquisitionState:
        return dataclasses.replace(
            self._state,
            mu=self._state.mu.copy(),
            sigma=self._state.sigma.copy(),
            theta_best=self._state.theta_best.copy(),
            grad_psi_at_best=self._state.grad_psi_at_best.copy(),
        )

    def save_state(self) -> Dict[str, Any]:
        return self.get_state().to_dict()

    def _build_candidate_pool(
        self,
        theta_best: np.ndarray,
        X_candidates: Optional[np.ndarray],
        f_min: float,
        lift: Optional[Any] = None,
    ) -> np.ndarray:
        pool: List[np.ndarray] = []

        if X_candidates is not None:
            provided = self._coerce_candidate_array(X_candidates)
            pool.extend(provided)

        pool.append(theta_best.copy())
        pool.extend(self._sample_gaussian(self.n_restarts_optimizer, self._state.mu, self._state.sigma))
        pool.extend(self._sample_uniform(self.n_random_candidates))

        seeds = self._deduplicate(pool)
        optimized: List[np.ndarray] = []
        for seed in seeds[: self.n_restarts_optimizer]:
            optimized.append(self._optimize_from_seed(seed, f_min, lift=lift))

        pool.extend(optimized)
        pool = self._deduplicate(pool)
        if not pool:
            pool = [self._repair_theta(theta_best)]

        return np.vstack(pool)

    def _optimize_from_seed(
        self,
        seed: np.ndarray,
        f_min: float,
        lift: Optional[Any] = None,
    ) -> np.ndarray:
        x0 = self._repair_theta(seed)

        def objective(x: np.ndarray) -> float:
            x = self._repair_theta(x)
            mean, std = self.gp.predict_with_coupling(x[None, :], coupling=lift)
            return -float(expected_improvement(mean, std, f_min)[0])

        try:
            result = minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=self._bounds,
                options={"maxiter": 100},
            )
            if result.success:
                return self._repair_theta(result.x)
        except Exception as exc:
            logger.debug("L-BFGS-B failed from seed %s: %s", np.round(x0, 4), exc)
        return x0

    def _coerce_candidate_array(self, X_candidates: np.ndarray) -> List[np.ndarray]:
        X = np.atleast_2d(np.asarray(X_candidates, dtype=float))
        if X.shape[1] != len(PARAM_KEYS):
            raise ValueError(f"Expected {len(PARAM_KEYS)} candidate dimensions, got {X.shape[1]}")

        if np.all((X >= -1e-9) & (X <= 1.0 + 1e-9)):
            X = self._lo + X * (self._hi - self._lo)

        return [self._repair_theta(row) for row in X]

    def _sample_uniform(self, n: int) -> List[np.ndarray]:
        if n <= 0:
            return []
        X = self._rng.uniform(self._lo, self._hi, size=(n, len(PARAM_KEYS)))
        return [self._repair_theta(row) for row in X]

    def _sample_gaussian(self, n: int, mu: np.ndarray, sigma: np.ndarray) -> List[np.ndarray]:
        if n <= 0:
            return []
        X = mu + sigma * self._rng.standard_normal(size=(n, len(PARAM_KEYS)))
        return [self._repair_theta(row) for row in X]

    def _repair_theta(self, theta: np.ndarray) -> np.ndarray:
        x = np.asarray(theta, dtype=float).ravel().copy()
        x = np.clip(x, self._lo, self._hi)
        if x[3] + x[4] > DSOC_SUM_MAX:
            scale = (DSOC_SUM_MAX * 0.995) / (x[3] + x[4])
            x[3] *= scale
            x[4] *= scale
            x = np.clip(x, self._lo, self._hi)
        return x

    def _deduplicate(self, points: List[np.ndarray]) -> List[np.ndarray]:
        unique: List[np.ndarray] = []
        seen = set()
        for point in points:
            x = self._repair_theta(point)
            key = tuple(np.round(x, 6))
            if key in seen:
                continue
            seen.add(key)
            unique.append(x)
        return unique

    @staticmethod
    def _select_top_unique(X: np.ndarray, score: np.ndarray, n_select: int) -> List[int]:
        order = np.argsort(score)[::-1]
        chosen: List[int] = []
        for idx in order:
            if any(np.linalg.norm(X[idx] - X[j]) < 1e-6 for j in chosen):
                continue
            chosen.append(int(idx))
            if len(chosen) >= n_select:
                break
        if not chosen and len(order):
            chosen.append(int(order[0]))
        return chosen


def expected_improvement(mean: np.ndarray, std: np.ndarray, f_min: float) -> np.ndarray:
    mean = np.asarray(mean, dtype=float).ravel()
    std = np.clip(np.asarray(std, dtype=float).ravel(), 1e-12, None)
    improvement = f_min - mean
    z = improvement / std
    ei = improvement * scipy_norm.cdf(z) + std * scipy_norm.pdf(z)
    ei[std <= 1e-12] = 0.0
    return np.maximum(ei, 0.0)


def build_acquisition_function(
    gp: GPProtocol,
    psi_fn: Any,
    param_bounds: Dict[str, Tuple[float, float]],
    n_select: int = 1,
    n_restarts_optimizer: int = 16,
    n_random_candidates: int = 128,
    random_seed: Optional[int] = None,
    **_: Any,
) -> AcquisitionFunction:
    return AcquisitionFunction(
        gp=gp,
        param_bounds=param_bounds,
        n_select=n_select,
        n_restarts_optimizer=n_restarts_optimizer,
        n_random_candidates=n_random_candidates,
        random_seed=random_seed,
    )
