from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
from sklearn.cluster import KMeans

from utils.constants import (
    DSOC_SUM_MAX,
    LLM_SAFE_DSOC_SUM_MAX,
    dsoc_sum_violates_limit,
    project_dsoc_pair,
)

logger = logging.getLogger(__name__)

PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]


@dataclasses.dataclass
class ProposalTrainingRecord:
    theta: np.ndarray
    scalar_y: float
    improvement: float
    feasible: bool
    near_constraint_penalty: float
    monotone_penalty: float
    source: str
    iteration: int
    weight: float


@dataclasses.dataclass
class ProposalBlendConfig:
    n_proposal: int = 24
    proposal_mix_local: float = 0.30
    proposal_mix_global: float = 0.70
    dedup_radius: float = 1e-6


@runtime_checkable
class ProposalSamplerProtocol(Protocol):
    def fit(self, records: List[ProposalTrainingRecord]) -> Dict[str, Any]:
        ...

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        center: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ...

    def score(self, X: np.ndarray) -> np.ndarray:
        ...

    def is_ready(self) -> bool:
        ...

    def summary(self) -> Dict[str, Any]:
        ...


class WeightedGMMSampler:
    """Small-sample weighted GMM-like proposal sampler.

    This implementation avoids relying on sklearn's GaussianMixture sample
    weights support by:
      1. selecting elite points with weighted utility scores,
      2. clustering them with weighted KMeans,
      3. fitting one Gaussian per cluster using weighted empirical moments.
    """

    def __init__(
        self,
        param_bounds: Dict[str, tuple[float, float]],
        *,
        n_components: int = 3,
        min_train_size: int = 8,
        covariance_floor: float = 1e-3,
        elite_fraction: float = 0.35,
        blend: Optional[ProposalBlendConfig] = None,
        safe_dsoc_sum_max: Optional[float] = LLM_SAFE_DSOC_SUM_MAX,
        enforce_monotone_profile: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.param_bounds = param_bounds
        self.n_components = max(int(n_components), 1)
        self.min_train_size = max(int(min_train_size), 2)
        self.covariance_floor = max(float(covariance_floor), 1e-6)
        self.elite_fraction = float(np.clip(elite_fraction, 0.05, 1.0))
        self.blend = blend or ProposalBlendConfig()
        self.safe_dsoc_sum_max = (
            min(float(safe_dsoc_sum_max), float(DSOC_SUM_MAX))
            if safe_dsoc_sum_max is not None else None
        )
        self.enforce_monotone_profile = bool(enforce_monotone_profile)
        self.random_state = random_state

        self._lo = np.array([param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        self._hi = np.array([param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        self._diag_floor = np.maximum((self._hi - self._lo) * self.covariance_floor, 1e-3)
        self._mixture_weights = np.empty((0,), dtype=float)
        self._means = np.empty((0, len(PARAM_KEYS)), dtype=float)
        self._covariances = np.empty((0, len(PARAM_KEYS), len(PARAM_KEYS)), dtype=float)
        self._global_cov = np.diag(self._diag_floor ** 2)
        self._summary: Dict[str, Any] = {"ready": False, "reason": "not_fitted"}
        self._is_fitted = False

    def fit(self, records: List[ProposalTrainingRecord]) -> Dict[str, Any]:
        clean_records = [record for record in records if record.feasible]
        if len(clean_records) < self.min_train_size:
            self._reset_summary(
                reason="insufficient_records",
                n_records=len(clean_records),
            )
            return self.summary()

        X = np.array([np.asarray(record.theta, dtype=float).ravel() for record in clean_records], dtype=float)
        weights = np.array([max(float(record.weight), 0.0) for record in clean_records], dtype=float)
        if np.all(weights <= 0.0):
            scores = np.array([float(record.scalar_y) for record in clean_records], dtype=float)
            ranks = np.argsort(np.argsort(scores))
            weights = np.maximum(len(scores) - ranks, 1).astype(float)

        elite_count = min(len(clean_records), max(2, int(math.ceil(len(clean_records) * self.elite_fraction))))
        elite_order = np.argsort(weights)[::-1][:elite_count]
        X_elite = X[elite_order]
        W_elite = weights[elite_order]
        W_elite = np.maximum(W_elite, 1e-12)

        unique_count = len(np.unique(np.round(X_elite, decimals=6), axis=0))
        n_components = min(self.n_components, max(1, elite_count // 2), max(unique_count, 1))
        if n_components == 1:
            labels = np.zeros(len(X_elite), dtype=int)
        else:
            kmeans = KMeans(
                n_clusters=n_components,
                n_init=10,
                random_state=self.random_state,
            )
            labels = kmeans.fit(X_elite, sample_weight=W_elite).labels_

        global_mean = np.average(X_elite, axis=0, weights=W_elite)
        self._global_cov = self._weighted_covariance(X_elite, W_elite, global_mean)

        mixture_weights: List[float] = []
        means: List[np.ndarray] = []
        covs: List[np.ndarray] = []
        for cluster_id in range(n_components):
            mask = labels == cluster_id
            if not np.any(mask):
                continue
            X_k = X_elite[mask]
            W_k = W_elite[mask]
            cluster_weight = float(np.sum(W_k))
            if cluster_weight <= 0.0:
                continue
            mean_k = np.average(X_k, axis=0, weights=W_k)
            cov_k = self._weighted_covariance(X_k, W_k, mean_k)
            mixture_weights.append(cluster_weight)
            means.append(mean_k)
            covs.append(cov_k)

        if not mixture_weights:
            self._reset_summary(
                reason="empty_clusters",
                n_records=len(clean_records),
            )
            return self.summary()

        self._mixture_weights = np.asarray(mixture_weights, dtype=float)
        self._mixture_weights /= max(float(self._mixture_weights.sum()), 1e-12)
        self._means = np.asarray(means, dtype=float)
        self._covariances = np.asarray(covs, dtype=float)
        self._is_fitted = True
        self._summary = {
            "ready": True,
            "n_records": int(len(clean_records)),
            "elite_count": int(elite_count),
            "n_components": int(len(self._mixture_weights)),
            "mixture_weights": np.round(self._mixture_weights, 6).tolist(),
            "means": np.round(self._means, 6).tolist(),
            "diag_std": np.round(
                np.sqrt(np.clip(np.diagonal(self._covariances, axis1=1, axis2=2), 1e-12, None)),
                6,
            ).tolist(),
        }
        return self.summary()

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        center: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not self._is_fitted or n <= 0:
            return np.empty((0, len(PARAM_KEYS)), dtype=float)

        n = int(n)
        n_local = 0
        if center is not None:
            n_local = int(round(n * float(np.clip(self.blend.proposal_mix_local, 0.0, 1.0))))
        n_global = max(n - n_local, 0)

        samples: List[np.ndarray] = []
        seen = set()

        def try_add(x: np.ndarray) -> None:
            repaired = self._repair_theta(x)
            key = tuple(np.round(repaired, 6))
            if key in seen:
                return
            seen.add(key)
            samples.append(repaired)

        max_attempts = max(4 * n, 16)
        attempts = 0
        while len(samples) < n_global and attempts < max_attempts:
            attempts += 1
            comp_id = int(rng.choice(len(self._mixture_weights), p=self._mixture_weights))
            draw = rng.multivariate_normal(self._means[comp_id], self._covariances[comp_id])
            try_add(draw)

        if center is not None and n_local > 0:
            local_center = self._repair_theta(np.asarray(center, dtype=float).ravel())
            local_cov = self._local_covariance(local_center)
            attempts = 0
            while len(samples) < n_global + n_local and attempts < max_attempts:
                attempts += 1
                draw = rng.multivariate_normal(local_center, local_cov)
                try_add(draw)

        if len(samples) < n:
            component_order = np.argsort(self._mixture_weights)[::-1]
            for comp_id in component_order:
                try_add(self._means[int(comp_id)])
                if len(samples) >= n:
                    break

        return np.vstack(samples[:n]) if samples else np.empty((0, len(PARAM_KEYS)), dtype=float)

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if not self._is_fitted or len(X) == 0:
            return np.zeros(X.shape[0], dtype=float)

        component_logs = []
        for weight, mean, cov in zip(self._mixture_weights, self._means, self._covariances):
            component_logs.append(self._component_logpdf(X, mean, cov) + math.log(max(float(weight), 1e-12)))
        stacked = np.vstack(component_logs)
        max_log = np.max(stacked, axis=0, keepdims=True)
        return (max_log + np.log(np.sum(np.exp(stacked - max_log), axis=0, keepdims=True))).ravel()

    def is_ready(self) -> bool:
        return bool(self._is_fitted)

    def summary(self) -> Dict[str, Any]:
        return dict(self._summary)

    def _reset_summary(self, *, reason: str, n_records: int) -> None:
        self._mixture_weights = np.empty((0,), dtype=float)
        self._means = np.empty((0, len(PARAM_KEYS)), dtype=float)
        self._covariances = np.empty((0, len(PARAM_KEYS), len(PARAM_KEYS)), dtype=float)
        self._is_fitted = False
        self._summary = {
            "ready": False,
            "reason": reason,
            "n_records": int(n_records),
        }

    def _weighted_covariance(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        mean: np.ndarray,
    ) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).ravel()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mean = np.asarray(mean, dtype=float).ravel()
        if len(X) <= 1:
            return np.diag(self._diag_floor ** 2)

        diff = X - mean[np.newaxis, :]
        cov = (diff * weights[:, None]).T @ diff / max(float(weights.sum()), 1e-12)
        cov = 0.5 * (cov + cov.T)
        cov += np.diag(self._diag_floor ** 2)
        diag = np.maximum(np.diag(cov), self._diag_floor ** 2)
        cov[np.diag_indices_from(cov)] = diag
        return cov

    def _local_covariance(self, center: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            return np.diag(self._diag_floor ** 2)
        distances = np.linalg.norm(self._means - center[np.newaxis, :], axis=1)
        nearest = int(np.argmin(distances))
        cov = self._covariances[nearest].copy()
        cov += np.diag((0.5 * self._diag_floor) ** 2)
        return cov

    def _repair_theta(self, theta: np.ndarray) -> np.ndarray:
        x = np.asarray(theta, dtype=float).ravel().copy()
        x = np.clip(x, self._lo, self._hi)
        if self.enforce_monotone_profile:
            x[1] = min(x[1], x[0])
            x[2] = min(x[2], x[1])
            x = np.clip(x, self._lo, self._hi)
            x[1] = min(x[1], x[0])
            x[2] = min(x[2], x[1])
        repair_limit = self.safe_dsoc_sum_max or float(DSOC_SUM_MAX)
        if dsoc_sum_violates_limit(x[3], x[4], dsoc_sum_max=repair_limit):
            x[3], x[4] = project_dsoc_pair(x[3], x[4], dsoc_sum_max=repair_limit)
            x = np.clip(x, self._lo, self._hi)
        return x

    @staticmethod
    def _component_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        mean = np.asarray(mean, dtype=float).ravel()
        cov = np.asarray(cov, dtype=float)
        d = X.shape[1]
        jitter = 1e-9
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(cov + np.eye(d) * jitter)
        diff = (X - mean[np.newaxis, :]).T
        solve = np.linalg.solve(chol, diff)
        quad = np.sum(solve ** 2, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def build_proposal_sampler(
    param_bounds: Dict[str, tuple[float, float]],
    config: Dict[str, Any],
    *,
    random_state: Optional[int] = None,
) -> ProposalSamplerProtocol:
    proposal_type = str(config.get("proposal_type", "weighted_gmm")).lower()
    if proposal_type != "weighted_gmm":
        logger.warning("Unsupported proposal_type=%s; falling back to weighted_gmm", proposal_type)

    blend = ProposalBlendConfig(
        n_proposal=int(config.get("proposal_n_samples", 24)),
        proposal_mix_local=float(config.get("proposal_local_mix", 0.30)),
        proposal_mix_global=1.0 - float(config.get("proposal_local_mix", 0.30)),
        dedup_radius=float(config.get("proposal_dedup_radius", 1e-6)),
    )
    return WeightedGMMSampler(
        param_bounds=param_bounds,
        n_components=int(config.get("proposal_n_components", 3)),
        min_train_size=int(config.get("proposal_min_train_size", 8)),
        covariance_floor=float(config.get("proposal_cov_floor", 1e-3)),
        elite_fraction=float(config.get("proposal_elite_fraction", 0.35)),
        blend=blend,
        safe_dsoc_sum_max=config.get("proposal_safe_dsoc_sum_max", config.get("llm_safe_dsoc_sum_max", LLM_SAFE_DSOC_SUM_MAX)),
        enforce_monotone_profile=bool(config.get("proposal_enforce_monotone_profile", False)),
        random_state=random_state,
    )
