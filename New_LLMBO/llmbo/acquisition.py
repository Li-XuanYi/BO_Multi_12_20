from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as scipy_norm

from llmbo.gp_model import GPProtocol
from utils.constants import (
    DSOC_SUM_MAX as CANONICAL_DSOC_SUM_MAX,
    dsoc_sum_violates_limit,
    project_dsoc_pair,
)

logger = logging.getLogger(__name__)

PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]
DSOC_SUM_MAX = CANONICAL_DSOC_SUM_MAX


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


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
    all_prior_bonus: Optional[np.ndarray] = None
    all_risk_penalty: Optional[np.ndarray] = None


@dataclasses.dataclass
class AcquisitionPrior:
    proposal_scorer: Optional[Any] = None
    proposal_alpha: float = 0.0
    proposal_anchor: float = 0.0
    proposal_scale: float = 1.0
    guidance_alpha: float = 0.0
    guidance_mode: Optional[str] = None
    guidance_center: Optional[np.ndarray] = None
    guidance_lb: Optional[np.ndarray] = None
    guidance_ub: Optional[np.ndarray] = None
    guidance_sigma: Optional[np.ndarray] = None
    safe_dsoc_sum_max: float = 0.65
    hard_dsoc_sum_max: float = DSOC_SUM_MAX
    safe_risk_weight: float = 0.0
    hard_risk_weight: float = 0.0
    monotone_risk_weight: float = 0.0
    agreement: float = 1.0

    def is_active(self) -> bool:
        return bool(
            self.proposal_alpha > 0.0
            or self.guidance_alpha > 0.0
            or self.safe_risk_weight > 0.0
            or self.hard_risk_weight > 0.0
            or self.monotone_risk_weight > 0.0
        )

    def bonus(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if len(X) == 0:
            return np.zeros(0, dtype=float)
        bonus = np.zeros(X.shape[0], dtype=float)
        if self.proposal_alpha > 0.0 and self.proposal_scorer is not None:
            raw = np.asarray(self.proposal_scorer(X), dtype=float).ravel()
            scale = max(float(self.proposal_scale), 1e-6)
            normalized = _stable_sigmoid((raw - float(self.proposal_anchor)) / scale)
            bonus += float(self.proposal_alpha) * normalized
        if self.guidance_alpha > 0.0:
            bonus += float(self.guidance_alpha) * self._guidance_bonus(X)
        return bonus

    def risk(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if len(X) == 0:
            return np.zeros(0, dtype=float)

        dsoc_sum = X[:, 3] + X[:, 4]
        safe_band = max(float(self.hard_dsoc_sum_max) - float(self.safe_dsoc_sum_max), 1e-6)
        safe_excess = np.maximum(dsoc_sum - float(self.safe_dsoc_sum_max), 0.0) / safe_band
        hard_excess = np.maximum(dsoc_sum - float(self.hard_dsoc_sum_max) + 1e-6, 0.0) / max(0.01, safe_band)
        monotone_violation = np.maximum(X[:, 1] - X[:, 0], 0.0) + np.maximum(X[:, 2] - X[:, 1], 0.0)

        return (
            float(self.safe_risk_weight) * safe_excess
            + float(self.hard_risk_weight) * hard_excess
            + float(self.monotone_risk_weight) * monotone_violation
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.is_active(),
            "proposal_alpha": float(self.proposal_alpha),
            "proposal_anchor": float(self.proposal_anchor),
            "proposal_scale": float(self.proposal_scale),
            "guidance_alpha": float(self.guidance_alpha),
            "guidance_mode": self.guidance_mode,
            "guidance_center": None if self.guidance_center is None else np.asarray(self.guidance_center, dtype=float).tolist(),
            "guidance_lb": None if self.guidance_lb is None else np.asarray(self.guidance_lb, dtype=float).tolist(),
            "guidance_ub": None if self.guidance_ub is None else np.asarray(self.guidance_ub, dtype=float).tolist(),
            "guidance_sigma": None if self.guidance_sigma is None else np.asarray(self.guidance_sigma, dtype=float).tolist(),
            "safe_dsoc_sum_max": float(self.safe_dsoc_sum_max),
            "hard_dsoc_sum_max": float(self.hard_dsoc_sum_max),
            "safe_risk_weight": float(self.safe_risk_weight),
            "hard_risk_weight": float(self.hard_risk_weight),
            "monotone_risk_weight": float(self.monotone_risk_weight),
            "agreement": float(self.agreement),
            "has_proposal_scorer": self.proposal_scorer is not None,
        }

    def _guidance_bonus(self, X: np.ndarray) -> np.ndarray:
        if self.guidance_mode == "region" and self.guidance_lb is not None and self.guidance_ub is not None:
            lb = np.asarray(self.guidance_lb, dtype=float)[None, :]
            ub = np.asarray(self.guidance_ub, dtype=float)[None, :]
            scale = np.maximum(ub - lb, 1e-3)
            under = np.maximum(lb - X, 0.0)
            over = np.maximum(X - ub, 0.0)
            delta = (under + over) / scale
            dist2 = np.sum(delta ** 2, axis=1)
            return np.exp(-0.5 * dist2)
        if self.guidance_mode == "point" and self.guidance_center is not None:
            center = np.asarray(self.guidance_center, dtype=float)[None, :]
            sigma = np.maximum(
                np.asarray(self.guidance_sigma if self.guidance_sigma is not None else np.full(X.shape[1], 0.1), dtype=float),
                1e-3,
            )[None, :]
            diff = (X - center) / sigma
            return np.exp(-0.5 * np.sum(diff ** 2, axis=1))
        return np.zeros(X.shape[0], dtype=float)


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
        prior: Optional[AcquisitionPrior] = None,
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
        prior_bonus = np.zeros_like(ei)
        risk_penalty = np.zeros_like(ei)
        score = ei * wcharge

        if prior is not None and prior.is_active():
            prior_bonus = prior.bonus(candidate_pool)
            risk_penalty = prior.risk(candidate_pool)
            score = self._normalize_feature(np.log1p(ei)) + prior_bonus - risk_penalty

        if np.all(score <= 1e-12):
            logger.info("EI surface is flat; falling back to max-uncertainty selection")
            score = std.copy() - risk_penalty

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
                "n_external_candidates": 0 if X_candidates is None else int(np.atleast_2d(X_candidates).shape[0]),
                "best_score": float(np.max(score)) if len(score) else 0.0,
                "best_ei": float(np.max(ei)) if len(ei) else 0.0,
                "best_prior_bonus": float(np.max(prior_bonus)) if len(prior_bonus) else 0.0,
                "best_risk_penalty": float(np.max(risk_penalty)) if len(risk_penalty) else 0.0,
                "stagnation_count": stagnation_count,
                "gp_llm_coupling": lift is not None,
                "acq_prior_active": prior is not None and prior.is_active(),
                "acq_prior": None if prior is None else prior.to_dict(),
            },
            all_mean_base=mean_base,
            lift_summary=lift.to_dict() if lift is not None else None,
            all_prior_bonus=prior_bonus,
            all_risk_penalty=risk_penalty,
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
        if dsoc_sum_violates_limit(x[3], x[4], dsoc_sum_max=DSOC_SUM_MAX):
            x[3], x[4] = project_dsoc_pair(x[3], x[4], dsoc_sum_max=DSOC_SUM_MAX)
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

    @staticmethod
    def _normalize_feature(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).ravel()
        if len(arr) <= 1:
            return np.zeros_like(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std <= 1e-12:
            span = float(np.max(arr) - np.min(arr))
            if span <= 1e-12:
                return np.zeros_like(arr)
            return (arr - float(np.min(arr))) / span - 0.5
        return np.clip((arr - mean) / std, -3.0, 3.0)


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
