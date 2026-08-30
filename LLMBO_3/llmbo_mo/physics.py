"""Physics-based surrogate components for MO-LLMBO.

Two public classes:

PsiFunction
    Analytical proxy for the three PyBaMM objectives (§3.1).
    Returns (psi_values, gradients) with no simulation.
    Physical parameters θ = [I1, I2, I3, SOC_sw1, SOC_sw2, V_CV, I_cutoff].

    Model equations (CC phase only; CV stage not modelled — Plan A):
        Ψ_time  [min]     = Q·Σ_k ΔSOCk/I_k · 60
        Ψ_temp  [°C rise] = (R_dc·Q·3600/mc_p) · Σ_k I_k·ΔSOCk
        Ψ_aging [Ah]      = A_sei·Q · Σ_k (ΔSOCk/I_k)·exp(α_sei·I_k)

    V_CV (index 5) and I_cutoff (index 6) have zero gradients (Plan A).
    The RBF kernel in GEKModel learns those dimensions purely from data.

GEKModel
    Gradient-Enhanced Kriging with ARD-RBF kernel (§3.2).
    sklearn-compatible: predict(X, return_std=True) → (mean, std).

    Augmented observations (2n×2n system):
        z = [y_tch (n,);  g_obs (n,)]
        y_tch[i]  = F_tch^aug(θ_i; λ_t)          Chebyshev scalarised target
        g_obs[i]  = grad_proj[i, k̂_i]             directional derivative
        k̂_i       = argmax_{k ∈ active_dims} |grad_proj[i, k]|

    Augmented ARD-RBF kernel (k(x,x') = σ_f² exp(-½ Σ_k Δx_k²/l_k²)):
        K_ff[i,j]  = k(x_i, x_j)
        K_f∂[i,j]  = k(x_i, x_j) · (x_i^{k̂_j} − x_j^{k̂_j}) / l_{k̂_j}²
        K_∂f       = −K_f∂ᵀ   (antisymmetric)
        K_∂∂[i,j]  = k(x_i,x_j) · [δ_{k̂_i,k̂_j}/l_{k̂_j}²
                       − Δx_i^{k̂_i}·Δx_i^{k̂_j} / (l_{k̂_i}²·l_{k̂_j}²)]

    Noise: diag([σ_n²·1_n,  σ_g²·1_n])
    MLE over {log σ_f, log l_{0…d-1}, log σ_n, log σ_g} via multi-restart L-BFGS-B.
    Falls back to sklearn GaussianProcessRegressor when n < config.gek_min_n.

Reference:
    Ulaganathan et al. (2016) "Performance study of gradient-enhanced Kriging",
    Advances in Engineering Software 90, 48-60.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.optimize import minimize

from config import MOLLMBOConfig

log = logging.getLogger(__name__)
Float = np.floating[Any]


# ══════════════════════════════════════════════════════════════════════════════
#  PsiFunction
# ══════════════════════════════════════════════════════════════════════════════

class PsiFunction:
    """Analytical proxy objectives with exact parameter gradients (§3.1).

    All methods are static — no instance state.

    θ layout (7 dims, PARAM_NAMES order):
        0 I1       [A]   stage-1 CC current
        1 I2       [A]   stage-2 CC current
        2 I3       [A]   stage-3 CC current
        3 SOC_sw1  [−]   SOC at stage-1 → stage-2 switch
        4 SOC_sw2  [−]   SOC at stage-2 → stage-3 switch
        5 V_CV     [V]   CV voltage       (zero gradient, Plan A)
        6 I_cutoff [A]   CV cutoff current (zero gradient, Plan A)

    Output units match battery_model.EvalResult:
        Ψ_time  [min]
        Ψ_temp  [°C rise above ambient]
        Ψ_aging [Ah]
    """

    @staticmethod
    def compute(
        theta: NDArray,
        config: MOLLMBOConfig,
    ) -> tuple[NDArray, NDArray]:
        """Compute proxy values and analytic gradients for one θ.

        Args:
            theta:  (7,)   parameter vector (no bounds enforced here)
            config: MOLLMBOConfig with physical constants

        Returns:
            psi:   (3,)    [Ψ_time, Ψ_temp, Ψ_aging]
            grad:  (3, 7)  ∂Ψ_k/∂θ_j; columns 5-6 are identically zero
        """
        I1 = float(theta[0])
        I2 = float(theta[1])
        I3 = float(theta[2])
        s1 = float(theta[3])   # SOC_sw1
        s2 = float(theta[4])   # SOC_sw2
        # theta[5] V_CV     — not used (Plan A)
        # theta[6] I_cutoff — not used (Plan A)

        s0   = config.soc_start
        send = config.soc_end
        Q    = config.q_nom     # [Ah]

        # Stage SOC spans
        ds1 = s1 - s0       # ΔSOCk stage 1
        ds2 = s2 - s1       # ΔSOCk stage 2
        ds3 = send - s2     # ΔSOCk stage 3

        psi  = np.zeros(3)
        grad = np.zeros((3, 7))   # rows = objectives, cols = θ dims

        # ── Ψ_time  [min] ────────────────────────────────────────────────────
        # t_k = ΔSOCk · Q / I_k  [h] × 60 → [min]
        psi[0] = (ds1 / I1 + ds2 / I2 + ds3 / I3) * Q * 60.0

        grad[0, 0] = -ds1 * Q * 60.0 / I1**2         # ∂/∂I1
        grad[0, 1] = -ds2 * Q * 60.0 / I2**2         # ∂/∂I2
        grad[0, 2] = -ds3 * Q * 60.0 / I3**2         # ∂/∂I3
        grad[0, 3] =  Q * 60.0 * (1.0/I1 - 1.0/I2)  # ∂/∂SOC_sw1: ds1↑ ds2↓
        grad[0, 4] =  Q * 60.0 * (1.0/I2 - 1.0/I3)  # ∂/∂SOC_sw2: ds2↑ ds3↓
        # dims 5, 6 remain 0

        # ── Ψ_temp  [°C rise] ────────────────────────────────────────────────
        # Lumped adiabatic Joule heating:
        #   E_k = I_k² · R_dc · Δt_k,  Δt_k = ΔSOCk·Q/I_k · 3600 [s]
        #   → E_k = I_k · ΔSOCk · Q · R_dc · 3600  [J]
        #   ΔT = Σ E_k / (m·c_p)
        #   α_th = R_dc · Q · 3600 / mc_p  [K per A·SOC_fraction]
        alpha_th = config.dcir * Q * 3600.0 / config.psi_mc_p

        psi[1] = alpha_th * (I1 * ds1 + I2 * ds2 + I3 * ds3)

        grad[1, 0] = alpha_th * ds1          # ∂/∂I1
        grad[1, 1] = alpha_th * ds2          # ∂/∂I2
        grad[1, 2] = alpha_th * ds3          # ∂/∂I3
        grad[1, 3] = alpha_th * (I1 - I2)   # ∂/∂SOC_sw1: d(I1·ds1)/ds1=I1, d(I2·ds2)/ds1=-I2
        grad[1, 4] = alpha_th * (I2 - I3)   # ∂/∂SOC_sw2: d(I2·ds2)/ds2=I2, d(I3·ds3)/ds2=-I3

        # ── Ψ_aging  [Ah] ────────────────────────────────────────────────────
        # Butler-Volmer SEI proxy:
        #   aging_k = A · Q · (ΔSOCk / I_k) · exp(α · I_k)
        a  = config.psi_sei_alpha
        A  = config.psi_sei_A
        e1 = np.exp(a * I1)
        e2 = np.exp(a * I2)
        e3 = np.exp(a * I3)

        psi[2] = A * Q * (ds1/I1*e1 + ds2/I2*e2 + ds3/I3*e3)

        # ∂/∂I_k [(ΔSOCk/I_k)·exp(α·I_k)] = ΔSOCk·exp(α·I_k)·(α·I_k − 1)/I_k²
        grad[2, 0] = A * Q * ds1 * e1 * (a*I1 - 1.0) / I1**2
        grad[2, 1] = A * Q * ds2 * e2 * (a*I2 - 1.0) / I2**2
        grad[2, 2] = A * Q * ds3 * e3 * (a*I3 - 1.0) / I3**2
        # ∂/∂SOC_sw1: d(ds1/I1·e1)/ds1 = e1/I1,  d(ds2/I2·e2)/ds1 = -e2/I2
        grad[2, 3] = A * Q * (e1/I1 - e2/I2)
        grad[2, 4] = A * Q * (e2/I2 - e3/I3)

        return psi, grad

    @staticmethod
    def batch(
        thetas: NDArray,
        config: MOLLMBOConfig,
    ) -> tuple[NDArray, NDArray]:
        """Vectorised compute over a batch of configurations.

        Args:
            thetas: (n, 7)
            config: MOLLMBOConfig

        Returns:
            psi:   (n, 3)
            grad:  (n, 3, 7)
        """
        results  = [PsiFunction.compute(thetas[i], config) for i in range(len(thetas))]
        psi_all  = np.vstack([r[0] for r in results])         # (n, 3)
        grad_all = np.stack([r[1] for r in results], axis=0)  # (n, 3, 7)
        return psi_all, grad_all


# ══════════════════════════════════════════════════════════════════════════════
#  GEKModel
# ══════════════════════════════════════════════════════════════════════════════

class GEKModel:
    """Gradient-Enhanced Kriging with ARD-RBF kernel (§3.2).

    Public interface matches sklearn.gaussian_process.GaussianProcessRegressor:
        fit(X, y_tch, grad_proj)
        predict(X, return_std=False) → mean  or  (mean, std)

    After fit(), key diagnostic attributes:
        X_train_   (n, d)   training inputs
        k_hat_     (n,)     parameter direction used per observation
        g_obs_     (n,)     derivative observation values
        alpha_     (2n,)    K_aug_noisy^{-1} · z_aug
        L_factor_  (2n,2n)  lower Cholesky of K_aug_noisy
        l2_        (d,)     optimised squared ARD length scales
        sf2_       float    optimised signal variance σ_f²
        log_sn_    float    log function noise
        log_sg_    float    log gradient noise
    """

    def __init__(self, config: MOLLMBOConfig) -> None:
        self.config       = config
        self.active_dims_ = np.array(config.gek_grad_dims, dtype=int)
        # active_dims_ = [0,1,2,3,4] = I1,I2,I3,SOC_sw1,SOC_sw2
        self._is_fitted     = False
        self._using_vanilla = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X:          NDArray,   # (n, d)   training inputs, normalised to [0,1]^d
        y_tch:      NDArray,   # (n,)     F_tch^aug targets
        grad_proj:  NDArray,   # (n, d)   projected gradients in F_tch space
    ) -> GEKModel:
        """Fit GEK to function observations and directional gradient info.

        Args:
            X:          (n, d)  normalised inputs (normalisation done by caller)
            y_tch:      (n,)    Chebyshev scalarised values;
                                y_tch[i] = F_tch^aug(θ_i; λ_t)
            grad_proj:  (n, d)  projected gradients:
                                  grad_proj[i] = λ_t[i*_i]·∇f_{i*_i}(θ_i)/(z^nad−z*+ε)
                                  dims 5,6 (V_CV, I_cutoff) are zero (Plan A)
        Returns:
            self
        """
        X = np.atleast_2d(X)
        n, d = X.shape

        self.X_train_ = X.copy()
        self.d_       = d

        # Fallback to vanilla GP when n is too small for stable MLE
        if n < self.config.gek_min_n:
            log.debug("n=%d < gek_min_n=%d; vanilla sklearn GP", n, self.config.gek_min_n)
            return self._fit_vanilla(X, y_tch)

        # Select most informative derivative direction per observation:
        # k̂_i = argmax_{k ∈ active_dims} |grad_proj[i, k]|
        active_grads = grad_proj[:, self.active_dims_]          # (n, d')
        local_idx    = np.argmax(np.abs(active_grads), axis=1)  # (n,) in [0, d'-1]
        self.k_hat_  = self.active_dims_[local_idx]             # (n,) global dim indices
        self.g_obs_  = grad_proj[np.arange(n), self.k_hat_]    # (n,) derivative values

        z_aug = np.concatenate([y_tch, self.g_obs_])           # (2n,)

        # MLE optimisation for ARD-RBF hyperparameters
        self._mle_optimize(X, z_aug)

        # Final Cholesky factorisation with optimised hyperparameters
        K_aug = self._build_Kaug(X, self.k_hat_, self.l2_, self.sf2_)
        sn2   = float(np.exp(2.0 * self.log_sn_))
        sg2   = float(np.exp(2.0 * self.log_sg_))
        noise = np.concatenate([np.full(n, sn2), np.full(n, sg2)])
        K_noisy = K_aug + np.diag(noise)

        try:
            self.L_factor_ = linalg.cholesky(K_noisy, lower=True)
        except linalg.LinAlgError:
            log.warning("GEK Cholesky failed; adding jitter 1e-6")
            K_noisy += np.eye(2 * n) * 1e-6
            self.L_factor_ = linalg.cholesky(K_noisy, lower=True)

        self.alpha_         = linalg.cho_solve((self.L_factor_, True), z_aug)
        self._is_fitted     = True
        self._using_vanilla = False
        return self

    def predict(
        self,
        X:          NDArray,
        return_std: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Sklearn-compatible posterior prediction.

        Args:
            X:           (m, d) test points
            return_std:  if True, also return posterior std

        Returns:
            mean (m,)  —  or —  (mean (m,), std (m,))
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        X = np.atleast_2d(X)

        # Vanilla GP path
        if self._using_vanilla:
            return self._sklearn_gp.predict(X, return_std=return_std)

        # GEK prediction
        k_star = self._predict_k_star(
            X, self.X_train_, self.k_hat_, self.l2_, self.sf2_
        )                                            # (m, 2n)
        mean = k_star @ self.alpha_                  # (m,)

        if not return_std:
            return mean

        # Predictive variance: k(x*,x*) − k_*ᵀ (K+noise)^{-1} k_*
        # Via Cholesky: v = L^{-1} k_*ᵀ  →  var = sf² − ||v||²
        v   = linalg.solve_triangular(self.L_factor_, k_star.T, lower=True)  # (2n, m)
        var = self.sf2_ - np.sum(v**2, axis=0)      # (m,)
        var = np.maximum(var, 1e-10)
        return mean, np.sqrt(var)

    # ── Kernel construction ────────────────────────────────────────────────────

    def _build_Kaug(
        self,
        X:     NDArray,   # (n, d)
        k_hat: NDArray,   # (n,) int — active param direction per obs
        l2:    NDArray,   # (d,) squared ARD length scales
        sf2:   float,
    ) -> NDArray:
        """Assemble 2n×2n augmented covariance matrix.

        Block structure:
            K_aug = [ K_ff     K_f∂  ]
                    [ K_∂f     K_∂∂  ]

        See module docstring for full analytical expressions.

        Returns:
            (2n, 2n) symmetric positive-semidefinite matrix (pre-noise)
        """
        n = X.shape[0]

        # Pairwise differences: diff[i,j,k] = X[i,k] − X[j,k]
        diff   = X[:, None, :] - X[None, :, :]          # (n, n, d)
        sqdist = np.sum(diff**2 / l2, axis=-1)           # (n, n)
        K_ff   = sf2 * np.exp(-0.5 * sqdist)            # (n, n)

        # ── K_f∂  ──────────────────────────────────────────────────────────
        # K_f∂[i,j] = K_ff[i,j] · (X[i,k̂_j] − X[j,k̂_j]) / l2[k̂_j]
        #
        # diff_kj[i,j] = diff[i, j, k_hat[j]]
        # Advanced index: diff[:, np.arange(n), k_hat] gives shape (n,n)
        # where result[i,j] = diff[i, j, k_hat[j]]  ✓
        diff_kj = diff[:, np.arange(n), k_hat]           # (n, n)
        l2_kj   = l2[k_hat]                              # (n,)
        K_f_partial = K_ff * diff_kj / l2_kj[None, :]   # (n, n)

        # K_∂f[i,j] = ∂k(x_i,x_j)/∂x_i^{k̂_i}
        #           = k(x_i,x_j) · (X[j,k̂_i] − X[i,k̂_i]) / l2[k̂_i]
        #           = −K_f∂[j,i]  →  K_∂f = −K_f∂.T  (antisymmetric)
        K_partial_f = -K_f_partial.T                       # (n, n)

        # ── K_∂∂  ──────────────────────────────────────────────────────────
        # diff_ki[i,j] = X[i,k̂_i] − X[j,k̂_i]
        # X[j, k̂_i] = X[:, k_hat][j, i]  →  X[:, k_hat].T[i, j]
        X_ki       = X[np.arange(n), k_hat]        # (n,)  X[i, k̂_i]
        X_along_ki = X[:, k_hat].T                 # (n,n) entry[i,j] = X[j, k̂_i]
        diff_ki    = X_ki[:, None] - X_along_ki    # (n,n) X[i,k̂_i] − X[j,k̂_i]

        kronecker = (k_hat[:, None] == k_hat[None, :]).astype(float)  # (n,n)
        l2_ki     = l2[k_hat]                      # (n,)

        K_dd = K_ff * (
            kronecker / l2_kj[None, :]
            - diff_ki * diff_kj / (l2_ki[:, None] * l2_kj[None, :])
        )

        # Assemble 2n×2n matrix
        K_aug = np.empty((2 * n, 2 * n))
        K_aug[:n, :n] = K_ff
        K_aug[:n, n:] = K_f_partial
        K_aug[n:, :n] = K_partial_f
        K_aug[n:, n:] = K_dd
        return K_aug

    def _predict_k_star(
        self,
        X_test:  NDArray,   # (m, d)
        X_train: NDArray,   # (n, d)
        k_hat:   NDArray,   # (n,) active param direction per training obs
        l2:      NDArray,
        sf2:     float,
    ) -> NDArray:
        """Cross-covariance between m test function values and 2n train augmented obs.

        Cov(f(x*), f(x_j))              = k(x*, x_j)
        Cov(f(x*), ∂f/∂x^{k̂_j}(x_j))  = k(x*, x_j)·(x*^{k̂_j}−x_j^{k̂_j})/l_{k̂_j}²

        Returns: (m, 2n)
        """
        n = X_train.shape[0]

        diff   = X_test[:, None, :] - X_train[None, :, :]   # (m, n, d)
        sqdist = np.sum(diff**2 / l2, axis=-1)               # (m, n)
        k_vals = sf2 * np.exp(-0.5 * sqdist)                 # (m, n)

        # Derivative block: k(x*,x_j) · (x*^{k̂_j} − x_j^{k̂_j}) / l2[k̂_j]
        diff_kj   = diff[:, np.arange(n), k_hat]             # (m, n)
        l2_kj     = l2[k_hat]                                # (n,)
        k_partial = k_vals * diff_kj / l2_kj[None, :]        # (m, n)

        return np.hstack([k_vals, k_partial])                 # (m, 2n)

    # ── MLE ────────────────────────────────────────────────────────────────────

    def _neg_lml(
        self,
        params: NDArray,   # [log_sf, log_l_0…log_l_{d-1}, log_sn, log_sg]
        X:      NDArray,   # (n, d)
        z_aug:  NDArray,   # (2n,)
        k_hat:  NDArray,   # (n,)
    ) -> float:
        """Negative log-marginal-likelihood for GEK.

        log p = −½ zᵀ K^{-1} z − ½ log|K| − N/2 log 2π
        Evaluated via Cholesky for stability.
        Returns 1e10 on non-PD matrix (infeasible hyperparameter region).
        """
        n, d = X.shape

        log_sf = params[0]
        log_l  = params[1: 1 + d]
        log_sn = params[1 + d]
        log_sg = params[2 + d]

        l2  = np.exp(2.0 * log_l)
        sf2 = np.exp(2.0 * log_sf)
        sn2 = np.exp(2.0 * log_sn)
        sg2 = np.exp(2.0 * log_sg)

        K_aug = self._build_Kaug(X, k_hat, l2, sf2)
        noise   = np.concatenate([np.full(n, sn2), np.full(n, sg2)])
        K_noisy = K_aug + np.diag(noise)

        try:
            L = linalg.cholesky(K_noisy, lower=True)
        except linalg.LinAlgError:
            return 1e10

        alpha_L = linalg.solve_triangular(L, z_aug, lower=True)   # (2n,)
        N       = 2 * n
        lml     = (
            -0.5 * float(alpha_L @ alpha_L)
            - float(np.sum(np.log(np.diag(L))))
            - 0.5 * N * np.log(2.0 * np.pi)
        )
        return -lml   # minimise the negative

    def _mle_optimize(self, X: NDArray, z_aug: NDArray) -> None:
        """Multi-restart L-BFGS-B MLE for {log_sf, log_l[d], log_sn, log_sg}.

        Subsamples to config.max_n_mle observations when n exceeds the cap to
        keep the 2n×2n Cholesky tractable (§10.2, max_n_mle=100 → 200×200).

        Stores optimised params in:
            self.log_sf_, log_l_, log_sn_, log_sg_
            self.l2_     (d,)   = exp(2·log_l_)
            self.sf2_    float  = exp(2·log_sf_)
        """
        d        = X.shape[1]
        n_full   = X.shape[0] // 2   # z_aug has length 2n
        n_params = 1 + d + 2
        bounds   = (
            [(-3.0, 3.0)]           # log_sf
            + [(-3.0, 3.0)] * d     # log_l ARD
            + [(-10.0, 0.0)]        # log_sn
            + [(-10.0, 0.0)]        # log_sg
        )

        # ── Subsample if n exceeds max_n_mle ──────────────────────────────────
        max_n = self.config.max_n_mle
        if n_full > max_n:
            rng_sub = np.random.RandomState(self.config.random_seed + n_full)
            idx     = rng_sub.choice(n_full, size=max_n, replace=False)
            idx_sorted = np.sort(idx)
            X_mle   = X[idx_sorted]
            # z_aug layout: [y_0..y_{n-1}, g_0..g_{n-1}]
            y_sub   = z_aug[:n_full][idx_sorted]
            g_sub   = z_aug[n_full:][idx_sorted]
            z_mle   = np.concatenate([y_sub, g_sub])
            k_hat_mle = self.k_hat_[idx_sorted]
        else:
            X_mle     = X
            z_mle     = z_aug
            k_hat_mle = self.k_hat_

        rng      = np.random.RandomState(self.config.random_seed)
        best_val = np.inf
        best_x   = None

        for _ in range(self.config.n_restarts):
            x0 = np.concatenate([
                rng.uniform(-1.0,  1.0, 1),   # log_sf
                rng.uniform(-1.0,  1.0, d),   # log_l (ARD)
                rng.uniform(-5.0, -2.0, 2),   # log_sn, log_sg
            ])
            res = minimize(
                self._neg_lml,
                x0,
                args=(X_mle, z_mle, k_hat_mle),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-9},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x   = res.x

        if best_x is None:
            log.warning("All MLE restarts failed; using default hyperparameters")
            best_x = np.zeros(n_params)
            best_x[-2:] = -3.0

        self.log_sf_ = float(best_x[0])
        self.log_l_  = best_x[1: 1 + d].copy()
        self.log_sn_ = float(best_x[1 + d])
        self.log_sg_ = float(best_x[2 + d])
        self.l2_     = np.exp(2.0 * self.log_l_)
        self.sf2_    = float(np.exp(2.0 * self.log_sf_))

    # ── Vanilla GP fallback ────────────────────────────────────────────────────

    def _fit_vanilla(self, X: NDArray, y_tch: NDArray) -> GEKModel:
        """Sklearn GaussianProcessRegressor for early iterations (n < gek_min_n).

        Exposes the same predict() interface. Hyperparameters from fitted kernel
        are stored in log_sf_, l2_, etc. for diagnostics.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

        d = X.shape[1]
        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=np.ones(d),
            length_scale_bounds=(1e-2, 1e2),
        )
        self._sklearn_gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.config.gp_alpha,
            n_restarts_optimizer=self.config.n_restarts,
            normalize_y=True,
            random_state=self.config.random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._sklearn_gp.fit(X, y_tch)

        # Expose fitted kernel parameters for downstream diagnostics
        fitted_k     = self._sklearn_gp.kernel_
        self.sf2_    = float(fitted_k.k1.constant_value)
        self.l2_     = fitted_k.k2.length_scale ** 2
        self.log_l_  = np.log(fitted_k.k2.length_scale)
        self.log_sf_ = 0.5 * np.log(max(self.sf2_, 1e-30))
        self.log_sn_ = 0.5 * np.log(self.config.gp_alpha)
        self.log_sg_ = 0.5 * np.log(self.config.gek_sigma_grad**2)

        self._using_vanilla = True
        self._is_fitted     = True
        return self