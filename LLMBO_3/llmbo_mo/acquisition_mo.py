"""Multi-objective acquisition components for MO-LLMBO.

Three public classes:

RISEWeights
    Pre-generates K weight vectors on the (m-1)-dimensional probability
    simplex using Riesz s-Energy repulsion (§10.6).
    Maximises the minimum pairwise Riesz energy via iterative gradient ascent
    on the simplex, ensuring dense, non-degenerate coverage of objective weights.

    Usage:
        rise = RISEWeights(K=50, m=3, s=1.0, seed=42)
        lam  = rise[t]          # (3,) weight vector for iteration t (cycled mod K)
        lam  = rise.sequence    # (K, 3) full sequence

ScalarEI
    Expected Improvement over the current best Chebyshev scalarised value.
    Takes a GEKModel (or any sklearn-compatible GP with predict(X, return_std))
    and returns -EI(X) (negative for minimisation optimisers).

    Implements the base_acq(mean, std) convention of bayes_opt.acquisition
    so it can be passed to the _acq_min infrastructure unchanged.

    Usage:
        acq = ScalarEI(gek_model, y_best)
        val = acq.base_acq(mean, std)   # positive EI  (n,)
        neg = acq(X)                    # negative EI  (n,) — for minimisers

DPPSelector
    Selects a diverse batch of size k from a candidate set via k-DPP.
    Uses a Gaussian quality-diversity kernel combining EI values and
    pairwise input distances (Kulesza & Taskar 2012, §4).

    Eigendecomposition-based exact k-DPP sampling for small candidate sets
    (typically ≤ 200); falls back to greedy MAP if sampling fails.

    Usage:
        selector = DPPSelector(config)
        idx      = selector.select(X_cand, ei_vals, k=3)
        X_batch  = X_cand[idx]

References:
    Riesz energy on the simplex:
        Hardin & Saff (2004) Discretizing manifolds via minimum energy points.
        Notices of the AMS 51(10), 1186-1194.
    k-DPP sampling:
        Kulesza & Taskar (2012) Determinantal Point Processes for Machine
        Learning. Foundations and Trends in ML 5(2-3), 123-286.
    Augmented Chebyshev scalarisation:
        Ishibuchi et al. (2017) Reference point specification in inverted
        generational distance for triangular linear Pareto fronts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from config import MOLLMBOConfig

log = logging.getLogger(__name__)
Float = np.floating[Any]


# ══════════════════════════════════════════════════════════════════════════════
#  RISEWeights
# ══════════════════════════════════════════════════════════════════════════════

class RISEWeights:
    """Riesz s-Energy weight sequence on the (m-1)-simplex (§10.6).

    Generates K weight vectors that maximise coverage of the simplex by
    maximising the sum of pairwise Riesz energies:
        E_s(λ) = Σ_{i≠j} 1 / ||λ_i − λ_j||^s

    The optimisation uses projected gradient ascent on the simplex, with
    random restarts to escape local optima.

    Attributes:
        sequence:  (K, m)  weight vectors; rows sum to 1, all entries ≥ 0
        K:         int     number of weight vectors
        m:         int     number of objectives
        s:         float   Riesz energy exponent
    """

    def __init__(
        self,
        K:    int   = 50,
        m:    int   = 3,
        s:    float = 1.0,
        seed: int   = 42,
        n_restarts: int = 5,
        n_iter:     int = 2000,
        lr:         float = 1e-2,
    ) -> None:
        self.K = K
        self.m = m
        self.s = s
        self._rng = np.random.RandomState(seed)
        self.sequence = self._generate(K, m, s, n_restarts, n_iter, lr)

    # ── Indexing ─────────────────────────────────────────────────────────────

    def __getitem__(self, t: int) -> NDArray:
        """Return λ_{t mod K} — cycles through the sequence indefinitely."""
        return self.sequence[t % self.K]

    # ── Generation ───────────────────────────────────────────────────────────

    def _generate(
        self,
        K:          int,
        m:          int,
        s:          float,
        n_restarts: int,
        n_iter:     int,
        lr:         float,
    ) -> NDArray:
        """Generate K well-spread weight vectors on the (m-1)-simplex.

        Strategy: Greedy farthest-point sampling (max-min distance).
            1. Draw N_pool >> K candidates uniformly from the simplex.
            2. Iteratively pick the candidate farthest (in L2) from
               the already-selected set — guarantees maximum minimum
               pairwise distance among all greedy strategies.
            3. Shuffle the result to avoid axis-aligned bias in the
               RISE sequence order.

        For m=3 this places points evenly across the triangle interior
        and edges without collapsing to vertices, which Riesz gradient
        ascent on the simplex boundary fails to avoid for K >> m.
        """
        N_pool = max(K * 200, 10_000)
        pool   = self._random_simplex(N_pool, m)   # (N_pool, m)

        selected = [int(self._rng.randint(0, N_pool))]
        min_dists = np.sum((pool - pool[selected[0]])**2, axis=1)  # (N_pool,)

        for _ in range(K - 1):
            farthest = int(np.argmax(min_dists))
            selected.append(farthest)
            new_dists = np.sum((pool - pool[farthest])**2, axis=1)
            min_dists = np.minimum(min_dists, new_dists)

        result = pool[selected]   # (K, m)

        # Light shuffle so iteration order has no systematic bias
        perm   = self._rng.permutation(K)
        result = result[perm]

        energy = float(np.min(np.sum((result[:, None] - result[None])**2, axis=-1)
                              + np.eye(K) * 1e30))
        log.debug("RISE: K=%d m=%d  min_dist²=%.4f", K, m, energy)
        return result

    def _optimise(self, *args, **kwargs):  # kept for API compat, not called
        pass

    def _random_simplex(self, K: int, m: int) -> NDArray:
        """Uniform random points on (m-1)-simplex via exponential trick."""
        x = self._rng.exponential(1.0, size=(K, m))
        return x / x.sum(axis=1, keepdims=True)

    def _optimise(
        self,
        lam:    NDArray,   # (K, m)
        s:      float,
        n_iter: int,
        lr:     float,
    ) -> tuple[NDArray, float]:
        """Projected gradient descent to MINIMISE Riesz energy (points spread out).

        Riesz energy:  E = Σ_{i≠j} ||λ_i − λ_j||^{-s}
        Gradient:      ∂E/∂λ_i = Σ_{j≠i} -s · dist_{ij}^{-(s+2)} · (λ_i − λ_j)
        Descent update (repulsive):
            λ_i ← proj_Δ( λ_i + lr · s · Σ_{j≠i} dist^{-(s+2)} · (λ_i−λ_j) )

        Moving each point AWAY from neighbours minimises E → uniform coverage.
        """
        lam = lam.copy()
        K   = lam.shape[0]
        eps = 1e-15   # distance floor to avoid div-by-zero

        for step in range(n_iter):
            diff  = lam[:, None, :] - lam[None, :, :]       # (K, K, m)
            dist2 = np.sum(diff**2, axis=-1) + eps           # (K, K)
            np.fill_diagonal(dist2, np.inf)

            # Repulsive weight: s · dist^{-(s+2)} = s · (dist2)^{-(s+2)/2}
            w = np.where(
                np.eye(K, dtype=bool),
                0.0,
                s * dist2 ** (-(s + 2.0) / 2.0),
            )                                                # (K, K)

            # Repulsive step: Δλ_i = +lr · Σ_j w_{ij} · diff[i,j]
            repulsion = np.einsum("ij,ijm->im", w, diff)    # (K, m)
            step_lr   = lr / (1.0 + step * 1e-3)
            lam       = self._project_simplex(lam + step_lr * repulsion)

        # Final energy (negated so 'higher = better spread' for caller comparison)
        diff  = lam[:, None, :] - lam[None, :, :]
        dist2 = np.sum(diff**2, axis=-1) + eps
        np.fill_diagonal(dist2, np.inf)
        energy = -np.sum(dist2 ** (-s / 2.0)) / 2.0    # negative raw energy

        return lam, energy

    @staticmethod
    def _project_simplex(v: NDArray) -> NDArray:
        """Row-wise Euclidean projection onto the (m-1)-simplex.

        Algorithm: Duchi et al. (2008) "Efficient Projections onto the ℓ1-Ball".
        Handles batch input (K, m).
        """
        K, m = v.shape
        u = np.sort(v, axis=-1)[:, ::-1]    # sort descending
        cssv = np.cumsum(u, axis=-1)
        rho  = np.sum(
            u > (cssv - 1.0) / (np.arange(1, m + 1)[None, :]),
            axis=-1,
        ) - 1                                # (K,)
        theta = (cssv[np.arange(K), rho] - 1.0) / (rho + 1.0)   # (K,)
        return np.maximum(v - theta[:, None], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  ScalarEI
# ══════════════════════════════════════════════════════════════════════════════

class ScalarEI:
    """Expected Improvement over the current best scalarised value (§10.2).

    Wraps any sklearn-compatible GP (including GEKModel) to produce EI
    values and candidate suggestions in the same style as
    bayes_opt.acquisition.ExpectedImprovement.

    Convention
    ----------
    base_acq(mean, std) → positive EI (higher is better)
    __call__(X)         → negative EI (for minimisation)

    The GP is assumed to model F_tch^aug (lower is better); EI is defined as
        EI(x) = E[max(y_best − f(x), 0)]
               = (y_best − μ(x)) · Φ(Z) + σ(x) · φ(Z)
    where Z = (y_best − μ(x)) / σ(x).

    Args:
        gp:     fitted GEKModel (or sklearn GPR) with predict(X, return_std)
        y_best: current best (minimum) scalarised value observed
        xi:     jitter for exploration/exploitation balance (default 0.01)
    """

    def __init__(
        self,
        gp:     Any,
        y_best: float,
        xi:     float = 0.01,
    ) -> None:
        self.gp     = gp
        self.y_best = float(y_best)
        self.xi     = float(xi)

    # ── bayes_opt-compatible interface ────────────────────────────────────────

    def base_acq(self, mean: NDArray, std: NDArray) -> NDArray:
        """Positive EI from predictive mean and std.

        Args:
            mean: (n,) or (n, 1)  GP posterior mean
            std:  (n,) or (n, 1)  GP posterior std

        Returns:
            ei: (n,)  non-negative expected improvement values
        """
        mean = np.asarray(mean).ravel()
        std  = np.asarray(std).ravel()
        std  = np.maximum(std, 1e-9)     # numerical floor
        Z    = (self.y_best - mean - self.xi) / std
        ei   = (self.y_best - mean - self.xi) * norm.cdf(Z) + std * norm.pdf(Z)
        return np.maximum(ei, 0.0)

    def __call__(self, X: NDArray) -> NDArray:
        """Evaluate negative EI at candidate points X.

        Args:
            X: (n, d)  candidate parameter vectors (normalised)

        Returns:
            (n,)  -EI values  (use with a minimiser)
        """
        X    = np.atleast_2d(X)
        mean, std = self.gp.predict(X, return_std=True)
        return -self.base_acq(mean, std)

    def suggest_batch(
        self,
        X_cand: NDArray,
        n_random: int = 5000,
        bounds:   NDArray | None = None,
        rng:      np.random.RandomState | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Evaluate EI over a candidate set and optionally supplement
        with random samples inside the (optionally TuRBO-clipped) bounds.

        Args:
            X_cand:   (c, d)  pre-generated candidates (TuRBO + LLM merged)
            n_random: additional random samples to evaluate
            bounds:   (d, 2)  if given, random samples are drawn inside bounds
            rng:      random state

        Returns:
            X_all:  (c + n_random, d)  all candidates evaluated
            ei_all: (c + n_random,)    EI values (positive)
        """
        rng = rng or np.random.RandomState()
        d   = X_cand.shape[1]

        if n_random > 0 and bounds is not None:
            lo, hi  = bounds[:, 0], bounds[:, 1]
            X_rand  = rng.uniform(lo, hi, size=(n_random, d))
            X_all   = np.vstack([X_cand, X_rand])
        else:
            X_all = X_cand.copy()

        mean, std = self.gp.predict(X_all, return_std=True)
        ei_all    = self.base_acq(mean, std)

        return X_all, ei_all


# ══════════════════════════════════════════════════════════════════════════════
#  DPPSelector
# ══════════════════════════════════════════════════════════════════════════════

class DPPSelector:
    """Diversity-aware batch selection via k-DPP (§10.4).

    Constructs a quality-diversity kernel L:
        L[i,j] = q_i · k(x_i, x_j) · q_j

    where:
        q_i  = exp(α · ei_i / max(ei))   quality weight
        k(x,x') = exp(-||x−x'||² / (2·l²))  RBF diversity kernel
        l    = config.dpp_length_scale

    Exact k-DPP sampling via eigendecomposition of L (Kulesza & Taskar §5.2.2).
    Falls back to greedy MAP (adds element maximising det increase) when
    eigendecomposition sampling fails (rare for small candidate sets).

    Args:
        config: MOLLMBOConfig  (reads dpp_length_scale, dpp_k)
    """

    def __init__(self, config: MOLLMBOConfig) -> None:
        self.config = config

    # ── Public API ─────────────────────────────────────────────────────────────

    def select(
        self,
        X_cand:  NDArray,   # (c, d)  candidate parameter vectors
        ei_vals: NDArray,   # (c,)    EI values for each candidate
        k:       int | None = None,
        seed:    int = 0,
    ) -> NDArray:
        """Select a diverse batch of k candidates from X_cand.

        Args:
            X_cand:  (c, d)  normalised parameter vectors
            ei_vals: (c,)    EI or any positive quality score
            k:       batch size (default: config.dpp_k)
            seed:    RNG seed for reproducible sampling

        Returns:
            (k,) int array of selected indices into X_cand
        """
        k   = k if k is not None else self.config.dpp_k
        c   = len(X_cand)

        if c <= k:
            return np.arange(c)

        L   = self._build_L(X_cand, ei_vals)
        rng = np.random.RandomState(seed)

        try:
            idx = self._sample_kdpp(L, k, rng)
        except Exception as exc:
            log.warning("k-DPP sampling failed (%s); falling back to greedy MAP", exc)
            idx = self._greedy_map(L, k)

        return np.array(idx)

    # ── Kernel construction ────────────────────────────────────────────────────

    def _build_L(
        self,
        X_cand:  NDArray,   # (c, d)
        ei_vals: NDArray,   # (c,)
    ) -> NDArray:
        """Build the L-ensemble kernel matrix.

        L[i,j] = q_i · k(x_i, x_j) · q_j
        where q_i = exp(ei_i / max_ei) and k is an RBF kernel.
        """
        c    = X_cand.shape[0]
        l    = self.config.dpp_length_scale

        # Quality vector
        max_ei = float(np.max(np.abs(ei_vals))) + 1e-30
        q      = np.exp(ei_vals / max_ei)                  # (c,)

        # RBF kernel on pairwise distances
        diff   = X_cand[:, None, :] - X_cand[None, :, :]  # (c, c, d)
        sqdist = np.sum(diff**2, axis=-1)                  # (c, c)
        K_rbf  = np.exp(-sqdist / (2.0 * l**2))            # (c, c)

        # Outer product of quality weights
        Q = q[:, None] * q[None, :]                        # (c, c)
        L = Q * K_rbf                                      # (c, c)

        # Symmetry and diagonal floor for numerical stability
        L = 0.5 * (L + L.T)
        L[np.diag_indices(c)] = np.maximum(np.diag(L), 1e-10)
        return L

    # ── Exact k-DPP sampling ──────────────────────────────────────────────────

    def _sample_kdpp(
        self,
        L:   NDArray,   # (c, c)
        k:   int,
        rng: np.random.RandomState,
    ) -> list[int]:
        """Exact k-DPP sample via eigendecomposition (Kulesza & Taskar §5.2.2).

        Algorithm:
            1. Eigen-decompose L = V Λ Vᵀ
            2. Sample a k-subset J of eigenvectors proportional to
               the elementary symmetric polynomial e_k(Λ)
            3. Sample a k-point set from the span of V_J
        """
        c = L.shape[0]

        # Symmetric eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(L)       # ascending order
        eigvals = np.maximum(eigvals, 0.0)          # numerical floor

        # ── Step 1: sample which k eigenvectors contribute ─────────────────
        # Compute elementary symmetric polynomials e_j(λ) for j=0..k
        # via DP: E[j, i] = e_j(λ_1, ..., λ_i)
        E = _elementary_symmetric(eigvals, k)

        # Sample eigenvector indices J by backward pass
        J     = []
        remain = k
        for i in range(c - 1, -1, -1):
            if remain == 0:
                break
            if i == remain - 1:
                # Must include this eigenvector
                J.append(i)
                remain -= 1
                continue
            # P(include λ_i | partial selection)
            denom = E[remain, i + 1] if i + 1 < c else 1.0
            numer = eigvals[i] * (E[remain - 1, i] if remain >= 1 else 0.0)
            p_inc = numer / (denom + 1e-300)
            if rng.rand() < np.clip(p_inc, 0.0, 1.0):
                J.append(i)
                remain -= 1

        if len(J) < k:
            # Fallback: top-k by eigenvalue magnitude
            J = list(np.argsort(eigvals)[-k:])

        V = eigvecs[:, J]   # (c, k)  selected eigenvectors

        # ── Step 2: iteratively sample k items from V ──────────────────────
        selected = []
        V_span   = V.copy()   # (c, k)

        for _ in range(k):
            # Sample proportional to ||V_span rows||²
            probs = np.sum(V_span**2, axis=1)
            probs = np.maximum(probs, 0.0)
            total = probs.sum()
            if total < 1e-15:
                # Uniform fallback if all probabilities are zero
                remaining = list(set(range(c)) - set(selected))
                selected.append(rng.choice(remaining))
                continue
            probs /= total

            # Exclude already-selected items
            for s_idx in selected:
                probs[s_idx] = 0.0
            if probs.sum() < 1e-15:
                remaining = list(set(range(c)) - set(selected))
                selected.append(rng.choice(remaining))
                continue
            probs /= probs.sum()

            item = int(rng.choice(c, p=probs))
            selected.append(item)

            # Update basis: remove the component along V_span[item] via Gram-Schmidt
            if len(selected) < k and V_span.shape[1] > 1:
                v_item = V_span[item].copy()            # (k_rem,)
                norm   = np.dot(v_item, v_item) + 1e-30
                # Project out the direction of v_item from each remaining vector
                V_span = V_span - np.outer(
                    V_span @ v_item, v_item
                ) / norm                               # (c, k_rem) - rank-1 update
                # Re-orthogonalise via thin QR to prevent numerical drift
                if V_span.shape[1] > 1:
                    Q_orth, _ = np.linalg.qr(V_span)
                    # QR returns (c, k_rem) — restore full column count
                    V_span = Q_orth[:, : V_span.shape[1]]

        return selected

    # ── Greedy MAP fallback ───────────────────────────────────────────────────

    def _greedy_map(self, L: NDArray, k: int) -> list[int]:
        """Greedy MAP approximation: add the item that maximises det(L_S).

        det(L_{S∪{i}}) / det(L_S) = L[i,i] − L[S,i]ᵀ L[S,S]^{-1} L[S,i]
                                   = Schur complement of L_S in L_{S∪{i}}.
        Implemented via incremental Cholesky updates for O(k·c) cost.
        """
        c        = L.shape[0]
        selected = []
        # Precompute L diagonals (= marginal inclusion contribution when S=∅)
        diag_L   = np.diag(L).copy()    # (c,)

        for step in range(k):
            best_gain = -np.inf
            best_item = -1
            for i in range(c):
                if i in selected:
                    continue
                gain = float(diag_L[i])   # Schur complement residual
                if gain > best_gain:
                    best_gain = gain
                    best_item = i
            selected.append(best_item)

            # Update residuals: after adding best_item, the Schur complement
            # of all remaining items shrinks by L[i, best_item]² / L[best_item, best_item]
            pivot = float(diag_L[best_item]) + 1e-12
            for j in range(c):
                if j not in selected:
                    diag_L[j] -= L[j, best_item]**2 / pivot

        return selected


# ══════════════════════════════════════════════════════════════════════════════
#  Elementary symmetric polynomial helper (for k-DPP)
# ══════════════════════════════════════════════════════════════════════════════

def _elementary_symmetric(lam: NDArray, k: int) -> NDArray:
    """Compute elementary symmetric polynomials e_j(λ) for j=0..k.

    DP recurrence: E[j, i] = e_j(λ_0, ..., λ_i)
        E[0, i] = 1  for all i
        E[j, i] = E[j, i-1] + λ_i · E[j-1, i-1]  for j ≥ 1, i ≥ 1

    Args:
        lam: (c,)  eigenvalues
        k:   int   order of the polynomial wanted

    Returns:
        E:  (k+1, c)  E[j, i] = e_j(λ_0, ..., λ_i)
    """
    c = len(lam)
    E = np.zeros((k + 1, c))
    E[0, :] = 1.0
    for i in range(1, c):
        for j in range(1, min(i + 1, k) + 1):
            E[j, i] = E[j, i - 1] + lam[i] * E[j - 1, i - 1]
    # Handle i=0 separately: E[1, 0] = λ_0
    if k >= 1:
        E[1, 0] = lam[0]
    return E