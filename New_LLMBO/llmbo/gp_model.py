from __future__ import annotations

import dataclasses
import logging
import warnings
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

logger = logging.getLogger(__name__)

PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]


@dataclasses.dataclass
class LLMPreferenceCoupling:
    mode: str
    grid: np.ndarray
    weights: np.ndarray
    confidence: float
    lambda_value: float
    posterior_variance: float
    gate: float = 1.0
    align_score: float = 1.0
    history_score: float = 1.0
    hv_score: float = 1.0
    stage_score: float = 1.0
    local_center: Optional[np.ndarray] = None
    local_lb: Optional[np.ndarray] = None
    local_ub: Optional[np.ndarray] = None
    local_sigma: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "grid": np.asarray(self.grid, dtype=float).tolist(),
            "weights": np.asarray(self.weights, dtype=float).tolist(),
            "confidence": float(self.confidence),
            "lambda_value": float(self.lambda_value),
            "posterior_variance": float(self.posterior_variance),
            "gate": float(self.gate),
            "align_score": float(self.align_score),
            "history_score": float(self.history_score),
            "hv_score": float(self.hv_score),
            "stage_score": float(self.stage_score),
            "local_center": None if self.local_center is None else np.asarray(self.local_center, dtype=float).tolist(),
            "local_lb": None if self.local_lb is None else np.asarray(self.local_lb, dtype=float).tolist(),
            "local_ub": None if self.local_ub is None else np.asarray(self.local_ub, dtype=float).tolist(),
            "local_sigma": None if self.local_sigma is None else np.asarray(self.local_sigma, dtype=float).tolist(),
        }

    @property
    def strength(self) -> float:
        return float(self.lambda_value) * float(self.gate)

    @property
    def gram_value(self) -> float:
        return float(self.posterior_variance)


@runtime_checkable
class GPProtocol(Protocol):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w_vec: Optional[np.ndarray] = None,
        t: int = 0,
    ) -> "GPProtocol":
        ...

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def predict_with_coupling(
        self,
        X_new: np.ndarray,
        coupling: Optional["LLMPreferenceCoupling"] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def posterior_covariance(
        self,
        X_left: np.ndarray,
        X_right: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        ...

    def build_preference_coupling(
        self,
        grid: np.ndarray,
        weights: np.ndarray,
        confidence: float,
        mode: str = "region",
        t: int = 0,
        lambda_max: float = 1.0,
        lambda_min: float = 0.0,
        decay_rate: float = 0.75,
        gate: float = 1.0,
        align_score: float = 1.0,
        history_score: float = 1.0,
        hv_score: float = 1.0,
        stage_score: float = 1.0,
        local_center: Optional[np.ndarray] = None,
        local_lb: Optional[np.ndarray] = None,
        local_ub: Optional[np.ndarray] = None,
        local_sigma: Optional[np.ndarray] = None,
    ) -> "LLMPreferenceCoupling":
        ...

    def training_summary(self) -> Dict[str, Any]:
        ...

    def target_standardization(self) -> Tuple[float, float]:
        ...

    def transform_targets(self, y: np.ndarray) -> np.ndarray:
        ...


class PsiFunction:
    """
    Lightweight 5D charging proxy kept only to preserve the old coupling hook.

    The current BO stack does not use physics-informed kernels, but keeping this
    interface makes it easy to reintroduce coupling logic later without another
    large refactor.
    """

    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        self._bounds = param_bounds

    def evaluate(self, theta: np.ndarray) -> float:
        x = self._coerce(theta)
        d3 = max(0.0, 0.8 - x[3] - x[4])
        spans = np.array([x[3], x[4], d3], dtype=float)
        currents = x[:3]
        return float(np.dot(currents, spans))

    def gradient_raw(self, theta: np.ndarray) -> np.ndarray:
        x = self._coerce(theta)
        d3 = max(0.0, 0.8 - x[3] - x[4])
        return np.array(
            [
                x[3],
                x[4],
                d3,
                x[0] - x[2],
                x[1] - x[2],
            ],
            dtype=float,
        )

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        grad = self.gradient_raw(theta)
        norm = np.linalg.norm(grad)
        if norm <= 1e-12:
            return grad
        return grad / norm

    @staticmethod
    def _coerce(theta: np.ndarray) -> np.ndarray:
        x = np.asarray(theta, dtype=float).ravel()
        if x.size != 5:
            raise ValueError(f"Expected 5-D theta, got {x.size}")
        return x


class MaternGPModel:
    """Standard Gaussian process model on normalized 5D inputs."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        kernel_nu: float = 2.5,
        alpha: float = 1e-6,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 5,
        target_transform_mode: str = "none",
        random_state: Optional[int] = None,
    ) -> None:
        self.param_bounds = param_bounds
        self.kernel_nu = float(kernel_nu)
        self.alpha = float(alpha)
        self.normalize_y = bool(normalize_y)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.target_transform_mode = str(target_transform_mode or "none").lower()
        self.random_state = random_state

        self._lo = np.array([param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        self._hi = np.array([param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        self._model: Optional[GaussianProcessRegressor] = None
        self._y_train_mean: float = 0.0
        self._y_train_std: float = 1.0
        self._target_transform_shift: float = 0.0
        self._last_summary: Dict[str, Any] = {
            "n_train": 0,
            "kernel": None,
            "length_scale": None,
            "noise": None,
            "y_best": None,
            "target_transform_mode": self.target_transform_mode,
            "target_transform_shift": 0.0,
            "nll": None,
            "standardized_residual_mean": None,
            "standardized_residual_var": None,
            "sigma_coverage_1sigma": None,
            "sigma_coverage_2sigma": None,
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w_vec: Optional[np.ndarray] = None,
        t: int = 0,
    ) -> "MaternGPModel":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y shape mismatch: {X.shape} vs {y.shape}")
        if X.shape[1] != len(PARAM_KEYS):
            raise ValueError(f"Expected {len(PARAM_KEYS)} features, got {X.shape[1]}")
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 points to fit the GP")

        X_norm = self._normalize_X(X)
        y_model = self._fit_target_transform(y)
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(X.shape[1]),
                length_scale_bounds=(1e-3, 1e3),
                nu=self.kernel_nu,
            )
            + WhiteKernel(
                noise_level=max(self.alpha, 1e-8),
                noise_level_bounds=(1e-12, 1e-1),
            )
        )
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._model.fit(X_norm, y_model)
        self._sync_target_standardization(y_model)

        length_scale = None
        noise = None
        try:
            length_scale = np.asarray(self._model.kernel_.k1.k2.length_scale).tolist()
            noise = float(self._model.kernel_.k2.noise_level)
        except Exception:
            pass
        nll = None
        try:
            nll = float(-self._model.log_marginal_likelihood_value_)
        except Exception:
            nll = None
        residual_summary = self._training_residual_summary(X, y_model)

        self._last_summary = {
            "n_train": int(X.shape[0]),
            "kernel": str(self._model.kernel_),
            "length_scale": length_scale,
            "noise": noise,
            "y_best": float(np.min(y_model)),
            "iteration": int(t),
            "target_transform_mode": self.target_transform_mode,
            "target_transform_shift": float(self._target_transform_shift),
            "nll": nll,
            **residual_summary,
        }
        logger.info(
            "Fitted Matern GP: n=%d y_best=%.6f kernel=%s",
            X.shape[0],
            float(np.min(y_model)),
            self._last_summary["kernel"],
        )
        return self

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        if X_new.shape[1] != len(PARAM_KEYS):
            raise ValueError(f"Expected {len(PARAM_KEYS)} features, got {X_new.shape[1]}")

        mean = self._require_model().predict(self._normalize_X(X_new), return_std=False)
        cov = self.posterior_covariance_raw(X_new)
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)

    def target_standardization(self) -> Tuple[float, float]:
        """Return the fitted target standardization in the GP modeling space."""
        self._require_model()
        return float(self._y_train_mean), float(max(self._y_train_std, 1e-12))

    def transform_targets(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float).ravel()
        mode = self.target_transform_mode
        if mode == "none":
            return values
        if mode == "log1p_shifted":
            return np.log1p(values + float(self._target_transform_shift))
        raise ValueError(f"Unknown target_transform_mode: {self.target_transform_mode}")

    def _fit_target_transform(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float).ravel()
        if self.target_transform_mode == "none":
            self._target_transform_shift = 0.0
            return values
        if self.target_transform_mode == "log1p_shifted":
            self._target_transform_shift = max(0.0, -float(np.min(values)) + 1e-6)
            return np.log1p(values + float(self._target_transform_shift))
        raise ValueError(f"Unknown target_transform_mode: {self.target_transform_mode}")

    def predict_standardized(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict in the GP standardized objective space.

        Region-Lifted GP v1 applies acquisition-time mean shifts only in this
        standardized space. The covariance scaling is cov_z = cov_y / y_std^2.
        """
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        mean_y = self._require_model().predict(self._normalize_X(X_new), return_std=False)
        y_mean, y_std = self.target_standardization()
        mean_z = (np.asarray(mean_y, dtype=float) - y_mean) / y_std
        cov_z = self.posterior_covariance_standardized(X_new)
        std_z = np.sqrt(np.clip(np.diag(cov_z), 1e-12, None))
        return mean_z, std_z

    def posterior_covariance_standardized(
        self,
        X_left: np.ndarray,
        X_right: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Posterior covariance in standardized target space."""
        return self.posterior_covariance(X_left, X_right)

    def posterior_covariance_raw(
        self,
        X_left: np.ndarray,
        X_right: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Posterior covariance in raw scalarized-target space."""
        _, y_std = self.target_standardization()
        return self.posterior_covariance_standardized(X_left, X_right) * (y_std ** 2)

    def predict_with_coupling(
        self,
        X_new: np.ndarray,
        coupling: Optional[LLMPreferenceCoupling] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean, std = self.predict(X_new)
        if coupling is None:
            return mean, std

        sigma_xg_z = self.posterior_covariance_standardized(X_new, coupling.grid)
        base_z = np.asarray(sigma_xg_z @ coupling.weights, dtype=float).ravel()
        mask = self._coupling_local_mask(X_new, coupling)
        shift_z = (
            float(coupling.lambda_value)
            * float(np.clip(coupling.gate, 0.0, 1.0))
            * mask
            * base_z
        )
        _, y_std = self.target_standardization()
        shift_y = shift_z * y_std
        return mean - shift_y, std

    def posterior_covariance(
        self,
        X_left: np.ndarray,
        X_right: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        left = np.atleast_2d(np.asarray(X_left, dtype=float))
        same_query = X_right is None
        right = left if same_query else np.atleast_2d(np.asarray(X_right, dtype=float))

        model = self._require_model()
        left_norm = self._normalize_X(left)
        right_norm = self._normalize_X(right)
        x_train = model.X_train_

        prior_left_right = self._latent_kernel(model, left_norm, right_norm)
        prior_left_train = self._latent_kernel(model, left_norm, x_train)
        prior_right_train = self._latent_kernel(model, right_norm, x_train)

        v_left = solve_triangular(model.L_, prior_left_train.T, lower=True, check_finite=False)
        v_right = solve_triangular(model.L_, prior_right_train.T, lower=True, check_finite=False)
        cov = np.asarray(prior_left_right - v_left.T @ v_right, dtype=float)
        if same_query:
            cov = 0.5 * (cov + cov.T)
            diag = np.clip(np.diag(cov), 1e-12, None)
            cov[np.diag_indices_from(cov)] = diag
        return cov

    def build_preference_coupling(
        self,
        grid: np.ndarray,
        weights: np.ndarray,
        confidence: float,
        mode: str = "region",
        t: int = 0,
        lambda_max: float = 1.0,
        lambda_min: float = 0.0,
        decay_rate: float = 0.75,
        gate: float = 1.0,
        align_score: float = 1.0,
        history_score: float = 1.0,
        hv_score: float = 1.0,
        stage_score: float = 1.0,
        local_center: Optional[np.ndarray] = None,
        local_lb: Optional[np.ndarray] = None,
        local_ub: Optional[np.ndarray] = None,
        local_sigma: Optional[np.ndarray] = None,
    ) -> LLMPreferenceCoupling:
        grid = np.atleast_2d(np.asarray(grid, dtype=float))
        weights = np.asarray(weights, dtype=float).ravel()
        if grid.shape[0] != weights.size:
            raise ValueError(f"grid/weights mismatch: {grid.shape[0]} vs {weights.size}")
        if grid.shape[1] != len(PARAM_KEYS):
            raise ValueError(f"Expected {len(PARAM_KEYS)}-D grid, got {grid.shape[1]}")

        weights = np.clip(weights, 0.0, None)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            raise ValueError("Region lift weights must sum to a positive value")
        weights = weights / weight_sum

        sigma_gg = self.posterior_covariance(grid, grid)
        posterior_variance = float(weights @ sigma_gg @ weights)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        t = max(int(t), 0)
        lambda_max = max(float(lambda_max), 0.0)
        lambda_min = min(max(float(lambda_min), 0.0), lambda_max)
        decay_rate = float(np.clip(decay_rate, 0.0, 1.0))

        base_lambda = confidence / np.sqrt(max(posterior_variance, 1e-12))
        annealed_lambda = base_lambda * (decay_rate ** t)
        lambda_value = float(np.clip(annealed_lambda, lambda_min, lambda_max))

        logger.debug(
            "Lambda coupling: base=%.6f annealed=%.6f clamped=%.6f variance=%.6e t=%d",
            base_lambda,
            annealed_lambda,
            lambda_value,
            posterior_variance,
            t,
        )

        return LLMPreferenceCoupling(
            mode=str(mode),
            grid=grid,
            weights=weights,
            confidence=confidence,
            lambda_value=lambda_value,
            posterior_variance=posterior_variance,
            gate=float(np.clip(gate, 0.0, 1.0)),
            align_score=float(np.clip(align_score, 0.0, 1.0)),
            history_score=float(np.clip(history_score, 0.0, 1.0)),
            hv_score=float(np.clip(hv_score, 0.0, 1.0)),
            stage_score=float(np.clip(stage_score, 0.0, 1.0)),
            local_center=None if local_center is None else np.asarray(local_center, dtype=float).ravel(),
            local_lb=None if local_lb is None else np.asarray(local_lb, dtype=float).ravel(),
            local_ub=None if local_ub is None else np.asarray(local_ub, dtype=float).ravel(),
            local_sigma=None if local_sigma is None else np.asarray(local_sigma, dtype=float).ravel(),
        )

    def predict_lifted(
        self,
        X_new: np.ndarray,
        lift: Optional[LLMPreferenceCoupling] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict_with_coupling(X_new, coupling=lift)

    def kernel_matrix(
        self,
        X_left: np.ndarray,
        X_right: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.posterior_covariance(X_left, X_right)

    def build_region_lift(
        self,
        grid: np.ndarray,
        weights: np.ndarray,
        confidence: float,
        mode: str = "region",
        t: int = 0,
        lambda_max: float = 1.0,
        lambda_min: float = 0.0,
        decay_rate: float = 0.75,
        gate: float = 1.0,
        align_score: float = 1.0,
        history_score: float = 1.0,
        hv_score: float = 1.0,
        stage_score: float = 1.0,
        local_center: Optional[np.ndarray] = None,
        local_lb: Optional[np.ndarray] = None,
        local_ub: Optional[np.ndarray] = None,
        local_sigma: Optional[np.ndarray] = None,
    ) -> LLMPreferenceCoupling:
        return self.build_preference_coupling(
            grid=grid,
            weights=weights,
            confidence=confidence,
            mode=mode,
            t=t,
            lambda_max=lambda_max,
            lambda_min=lambda_min,
            decay_rate=decay_rate,
            gate=gate,
            align_score=align_score,
            history_score=history_score,
            hv_score=hv_score,
            stage_score=stage_score,
            local_center=local_center,
            local_lb=local_lb,
            local_ub=local_ub,
            local_sigma=local_sigma,
        )

    def training_summary(self) -> Dict[str, Any]:
        return dict(self._last_summary)

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self._lo) / (self._hi - self._lo + 1e-12)

    @staticmethod
    def _latent_kernel(
        model: GaussianProcessRegressor,
        X_left: np.ndarray,
        X_right: np.ndarray,
    ) -> np.ndarray:
        kernel = model.kernel_
        if hasattr(kernel, "k1"):
            kernel = kernel.k1
        return np.asarray(kernel(X_left, X_right), dtype=float)

    def _require_model(self) -> GaussianProcessRegressor:
        if self._model is None:
            raise RuntimeError("GP model has not been fitted yet")
        return self._model

    def _sync_target_standardization(self, y: np.ndarray) -> None:
        model = self._require_model()
        mean = getattr(model, "_y_train_mean", 0.0)
        std = getattr(model, "_y_train_std", 1.0)
        try:
            self._y_train_mean = float(np.asarray(mean, dtype=float).ravel()[0])
        except Exception:
            self._y_train_mean = float(np.mean(np.asarray(y, dtype=float)))
        try:
            self._y_train_std = float(np.asarray(std, dtype=float).ravel()[0])
        except Exception:
            self._y_train_std = float(np.std(np.asarray(y, dtype=float)))
        if not np.isfinite(self._y_train_std) or self._y_train_std <= 1e-12:
            self._y_train_std = 1.0

    def _training_residual_summary(self, X: np.ndarray, y_model: np.ndarray) -> Dict[str, Optional[float]]:
        try:
            pred_mean = self._require_model().predict(self._normalize_X(X), return_std=False)
            cov = self.posterior_covariance_raw(X)
            pred_std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            standardized = (
                np.asarray(y_model, dtype=float).ravel() - np.asarray(pred_mean, dtype=float).ravel()
            ) / pred_std
            return {
                "standardized_residual_mean": float(np.mean(standardized)),
                "standardized_residual_var": float(np.var(standardized)),
                "sigma_coverage_1sigma": float(np.mean(np.abs(standardized) <= 1.0)),
                "sigma_coverage_2sigma": float(np.mean(np.abs(standardized) <= 2.0)),
            }
        except Exception:
            return {
                "standardized_residual_mean": None,
                "standardized_residual_var": None,
                "sigma_coverage_1sigma": None,
                "sigma_coverage_2sigma": None,
            }

    @staticmethod
    def _coupling_local_mask(
        X_new: np.ndarray,
        coupling: LLMPreferenceCoupling,
    ) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X_new, dtype=float))

        if coupling.mode == "point" and coupling.local_center is not None:
            center = np.asarray(coupling.local_center, dtype=float)[None, :]
            sigma = np.asarray(
                coupling.local_sigma if coupling.local_sigma is not None else np.full(X.shape[1], 0.1),
                dtype=float,
            )[None, :]
            sigma = np.maximum(sigma, 1e-6)
            diff = (X - center) / sigma
            return np.exp(-0.5 * np.sum(diff ** 2, axis=1))

        if (
            coupling.mode == "region"
            and coupling.local_lb is not None
            and coupling.local_ub is not None
        ):
            lb = np.asarray(coupling.local_lb, dtype=float)[None, :]
            ub = np.asarray(coupling.local_ub, dtype=float)[None, :]
            width = np.maximum((ub - lb) / 2.0, 1e-6)
            below = np.maximum(lb - X, 0.0)
            above = np.maximum(X - ub, 0.0)
            outside = (below + above) / width
            mask = np.exp(-0.5 * np.sum(outside ** 2, axis=1))
            inside = np.all((X >= lb - 1e-12) & (X <= ub + 1e-12), axis=1)
            mask[inside] = 1.0
            return mask

        return np.ones(X.shape[0], dtype=float)


def build_gp_stack(
    param_bounds: Dict[str, Tuple[float, float]],
    gamma_max: Optional[float] = None,
    gamma_min: Optional[float] = None,
    gamma_t_decay: Optional[float] = None,
    kernel_nu: float = 2.5,
    alpha: float = 1e-6,
    normalize_y: bool = True,
    n_restarts_optimizer: int = 5,
    target_transform_mode: str = "none",
    random_state: Optional[int] = None,
    **_: Any,
) -> Tuple[PsiFunction, None, None, MaternGPModel]:
    """
    Compatibility factory.

    The old code expected a four-tuple `(psi_fn, coupling_mgr, gamma_ann, gp_model)`.
    We keep the same return signature even though the current minimal workflow only
    uses the GP model directly.
    """

    psi_fn = PsiFunction(param_bounds)
    gp_model = MaternGPModel(
        param_bounds=param_bounds,
        kernel_nu=kernel_nu,
        alpha=alpha,
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts_optimizer,
        target_transform_mode=target_transform_mode,
        random_state=random_state,
    )
    return psi_fn, None, None, gp_model
