"""Persistent state management for MO-LLMBO.

Three public classes:

Database
    Append-only store for all EvalResult observations.
    Provides normalised / log-transformed objective arrays ready for GEK fitting.
    Tracks per-iteration Chebyshev scalarised values (f_tch_history) for
    convergence diagnostics and ablation analysis.

TuRBOState
    Trust-region state machine (§10.3).
    Manages the axis-aligned hypercube trust region [c−σ, c+σ]^d centred on
    the current best point.  Expands on successive_successes ≥ success_tol and
    contracts on successive_failures ≥ failure_tol.
    Gradient discrepancy check: inflates σ when physics gradient disagrees
    with the GEK posterior gradient.

OptimizerState
    Aggregated optimisation state shared across iterations.
    Holds z* (ideal point), z^nad (nadir point), the current Pareto front,
    and the pre-generated RISE weight sequence.
    Provides Chebyshev scalarisation (augmented, §10.6) and gradient
    projection helpers consumed by the main loop.

Design invariants:
    - All objectives are stored in ORIGINAL units (t_charge [min], T_peak [°C],
      delta_Q_aging [Ah]); log-transform and normalisation happen in
      Database.get_gp_targets() at query time.
    - TuRBOState operates in NORMALISED parameter space [0,1]^d; the
      caller is responsible for scaling θ before passing to update().
    - OptimizerState.z_star / z_nad are updated in TRANSFORMED space
      (log-aging, then normalised) to match the GEK model scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from battery_model import EvalResult
from config import MOLLMBOConfig, PARAM_NAMES
from pareto import (
    non_dominated_sort,
    normalize_objectives,
    log_transform_aging,
    log_transform_objectives,
    compute_hypervolume,
    compute_reference_point,
    crowding_distance,
)

log = logging.getLogger(__name__)
Float = np.floating[Any]


# ══════════════════════════════════════════════════════════════════════════════
#  Database
# ══════════════════════════════════════════════════════════════════════════════

class Database:
    """Append-only observation store with GP-ready target computation.

    Internal storage:
        _results:   list[EvalResult]        all evaluations (valid + invalid)
        _thetas:    list[NDArray (7,)]      parameter vectors
        _objs:      list[NDArray (3,)]      raw objectives (penalty for failures)
        f_tch_history: list[float]          per-iteration best Chebyshev value
        hv_history:    list[float]          per-iteration hypervolume

    Public methods:
        add(result)
        add_batch(results)
        get_valid_thetas()   → (n, 7)
        get_valid_objs()     → (n, 3)   raw
        get_gp_targets(state)→ (n,), (n, 7)  y_tch, grad_proj for GEK
        pareto_front_thetas()→ (k, 7)
        pareto_front_objs()  → (k, 3)  raw
        summary_stats()      → dict
    """

    def __init__(self, config: MOLLMBOConfig) -> None:
        self.config    = config
        self._results:  list[EvalResult] = []
        self._thetas:   list[NDArray]    = []
        self._objs:     list[NDArray]    = []
        self._valid_mask: list[bool]     = []

        # Diagnostic histories (one entry appended per call to record_iteration)
        self.f_tch_history: list[float] = []
        self.hv_history:    list[float] = []

    # ── Insertion ─────────────────────────────────────────────────────────────

    def add(self, result: EvalResult) -> None:
        """Append a single EvalResult.

        Failed / constraint-violated results are stored with physical penalty
        values instead of the sentinel 999, so they never corrupt GP fitting
        or HV reference point calculations if the valid_mask filter is bypassed.
        """
        self._results.append(result)
        self._thetas.append(result.theta.copy())
        if result.constraint_ok:
            self._objs.append(result.objectives.copy())
        else:
            # Physical upper bounds replace the sentinel 999
            self._objs.append(np.array([
                self.config.penalty_t_charge,
                self.config.penalty_T_peak,
                self.config.penalty_delta_Q,
            ], dtype=float))
        self._valid_mask.append(bool(result.constraint_ok))

    def add_batch(self, results: list[EvalResult]) -> None:
        """Append a batch of EvalResults."""
        for r in results:
            self.add(r)

    # ── Raw accessors ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._results)

    @property
    def n_valid(self) -> int:
        return int(sum(self._valid_mask))

    def get_all_thetas(self) -> NDArray:
        """(n, 7) — all evaluations including constraint violations."""
        return np.vstack(self._thetas) if self._thetas else np.empty((0, 7))

    def get_all_objs(self) -> NDArray:
        """(n, 3) — raw objectives; penalty rows for failures."""
        return np.vstack(self._objs) if self._objs else np.empty((0, 3))

    def get_valid_thetas(self) -> NDArray:
        """(n_valid, 7) — only constraint-satisfied evaluations."""
        mask = np.array(self._valid_mask, dtype=bool)
        if not mask.any():
            return np.empty((0, 7))
        return np.vstack(self._thetas)[mask]

    def get_valid_objs(self) -> NDArray:
        """(n_valid, 3) — raw objectives for valid evaluations only."""
        mask = np.array(self._valid_mask, dtype=bool)
        if not mask.any():
            return np.empty((0, 3))
        return np.vstack(self._objs)[mask]

    # ── GP target construction ─────────────────────────────────────────────────

    def get_gp_targets(
        self,
        state:       OptimizerState,
        lambda_t:    NDArray,          # (3,) Chebyshev weight vector
        physics_grads: NDArray | None = None,
        # (n_valid, 3, 7)  ∇Ψ_k(θ_i) from PsiFunction.batch()
        # pass None to get zeros (no GEK gradient info)
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Compute GEK training targets for iteration t.

        Pipeline:
            1. Get valid (θ, f) pairs
            2. log10-transform aging column (if config.log_aging)
            3. Compute normalised ĝ_k using current z* and z^nad
            4. Compute augmented Chebyshev y_tch[i] = F_tch^aug(θ_i; λ_t)
            5. Compute per-point gradient projection grad_proj[i] ∈ R^7

        Args:
            state:          OptimizerState with current z_star, z_nad
            lambda_t:       (3,)  current RISE weight vector
            physics_grads:  (n, 3, 7) or None  analytical gradients from PsiFunction

        Returns:
            X_norm:     (n, 7)   θ normalised to [0,1]^d
            y_tch:      (n,)     F_tch^aug scalarised targets
            grad_proj:  (n, 7)   projected gradient ∂F_tch/∂θ for GEK
        """
        thetas = self.get_valid_thetas()   # (n, 7)
        objs   = self.get_valid_objs()     # (n, 3)
        n      = len(thetas)

        if n == 0:
            return np.empty((0, 7)), np.empty((0,)), np.empty((0, 7))

        # ── Step 1: log-transform time and/or aging ──────────────────────────
        objs_t = log_transform_objectives(
            objs,
            log_time  = self.config.log_time,
            log_aging = self.config.log_aging,
        )

        # ── Step 2: normalise with current z*, z^nad ─────────────────────────
        # ĝ_k[i] = (f_k[i] − z*_k) / (z^nad_k − z*_k + ε)
        eps  = self.config.tch_eps
        denom = state.z_nad - state.z_star + eps      # (3,)
        g_hat = (objs_t - state.z_star) / denom       # (n, 3)

        # ── Step 3: augmented Chebyshev F_tch^aug ────────────────────────────
        rho = self.config.tch_rho
        weighted  = lambda_t * g_hat                  # (n, 3)
        tch_max   = weighted.max(axis=1)              # (n,)  max_k term
        tch_sum   = weighted.sum(axis=1)              # (n,)  Σ term
        y_tch     = tch_max + rho * tch_sum           # (n,)

        # ── Step 4: gradient projection ──────────────────────────────────────
        # For each point i:
        #   i*_i = argmax_k{ λ_t[k] · ĝ_k[i] }
        #   grad_proj[i] = λ_t[i*_i] · ∇f_{i*_i}(θ_i) / denom[i*_i]
        i_star = np.argmax(weighted, axis=1)           # (n,)

        if physics_grads is not None:
            # physics_grads: (n, 3, 7)  → select objective i*_i for each point
            # grad_f[i] = physics_grads[i, i_star[i], :]  shape (7,)
            grad_f_selected = physics_grads[np.arange(n), i_star, :]   # (n, 7)
            # Scale: λ_t[i*] · ∇f_{i*} / denom[i*]
            scale       = lambda_t[i_star] / denom[i_star]              # (n,)
            grad_proj   = grad_f_selected * scale[:, None]              # (n, 7)
        else:
            grad_proj = np.zeros((n, 7))

        # ── Step 5: normalise θ to [0,1]^d ────────────────────────────────────
        bounds  = self.config.bounds_array               # (7, 2)
        lo, hi  = bounds[:, 0], bounds[:, 1]
        X_norm  = (thetas - lo) / (hi - lo + 1e-30)

        # Clip to [0,1] for safety (constraint violations can push θ out of bounds)
        X_norm  = np.clip(X_norm, 0.0, 1.0)

        # Normalise grad_proj to [0,1]^d scale (chain rule for normalisation)
        # ∂y/∂θ_norm = ∂y/∂θ_raw · (hi − lo)
        grad_proj_norm = grad_proj * (hi - lo)           # (n, 7)

        return X_norm, y_tch, grad_proj_norm

    # ── Pareto helpers ────────────────────────────────────────────────────────

    def pareto_front_indices(self) -> list[int]:
        """Indices (into valid observations) of Pareto-optimal points."""
        objs = self.get_valid_objs()
        if len(objs) == 0:
            return []
        try:
            fronts = non_dominated_sort(objs)
            return fronts[0] if fronts else []
        except (IndexError, Exception) as exc:
            log.debug("non_dominated_sort fallback: %s", exc)
            n = len(objs)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                        dominated[i] = True
                        break
            return list(np.where(~dominated)[0])

    def pareto_front_thetas(self) -> NDArray:
        """(k, 7)  parameter vectors on the current Pareto front."""
        idx    = self.pareto_front_indices()
        thetas = self.get_valid_thetas()
        return thetas[idx] if len(idx) > 0 else np.empty((0, 7))

    def pareto_front_objs(self) -> NDArray:
        """(k, 3) raw objectives on the current Pareto front."""
        idx  = self.pareto_front_indices()
        objs = self.get_valid_objs()
        return objs[idx] if len(idx) > 0 else np.empty((0, 3))

    # ── Iteration bookkeeping ─────────────────────────────────────────────────

    def record_iteration(
        self,
        state:    OptimizerState,
        lambda_t: NDArray,
    ) -> None:
        """Append per-iteration diagnostics to f_tch_history and hv_history.

        Called once per BO iteration (after evaluate_batch and state update).
        Stores the best (minimum) Chebyshev value seen so far under λ_t, and
        the current hypervolume of the Pareto front.
        """
        thetas = self.get_valid_thetas()
        if len(thetas) == 0:
            self.f_tch_history.append(np.nan)
            self.hv_history.append(0.0)
            return

        objs_t = log_transform_objectives(self.get_valid_objs(),
                     log_time=self.config.log_time,
                     log_aging=self.config.log_aging)
        eps   = self.config.tch_eps
        denom = state.z_nad - state.z_star + eps
        g_hat = (objs_t - state.z_star) / denom
        weighted = lambda_t * g_hat
        tch_vals = weighted.max(axis=1) + self.config.tch_rho * weighted.sum(axis=1)
        self.f_tch_history.append(float(tch_vals.min()))

        pf_objs = self.pareto_front_objs()
        if len(pf_objs) >= 2:
            pf_objs_t = log_transform_objectives(
                pf_objs,
                log_time  = self.config.log_time,
                log_aging = self.config.log_aging,
            )
            ref = np.array(self.config.hv_ref_fixed, dtype=float)
            dominated = np.all(pf_objs_t < ref, axis=1)
            try:
                hv = compute_hypervolume(pf_objs_t[dominated], ref) if dominated.sum() >= 2 else 0.0
            except Exception:
                hv = 0.0
        else:
            hv = 0.0
        self.hv_history.append(hv)

    # ── Summary stats ─────────────────────────────────────────────────────────

    def summary_stats(self) -> dict:
        """Compact summary for logging and LLM context."""
        n_tot   = len(self._results)
        n_valid = self.n_valid
        pf_objs = self.pareto_front_objs()

        stats: dict = {
            "n_total":  n_tot,
            "n_valid":  n_valid,
            "n_failed": n_tot - n_valid,
            "n_pareto": len(pf_objs),
        }

        if n_valid > 0:
            valid_objs = self.get_valid_objs()
            stats["obj_min"] = valid_objs.min(axis=0).tolist()
            stats["obj_max"] = valid_objs.max(axis=0).tolist()

        if len(pf_objs) >= 2:
            pf_objs_t = log_transform_objectives(
                pf_objs,
                log_time  = self.config.log_time,
                log_aging = self.config.log_aging,
            )
            ref = np.array(self.config.hv_ref_fixed, dtype=float)
            dominated = np.all(pf_objs_t < ref, axis=1)
            try:
                stats["hypervolume"] = compute_hypervolume(pf_objs_t[dominated], ref) if dominated.sum() >= 2 else 0.0
            except Exception:
                stats["hypervolume"] = None

        if self.hv_history:
            stats["hv_history_last5"] = self.hv_history[-5:]

        return stats


# ══════════════════════════════════════════════════════════════════════════════
#  TuRBOState
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TuRBOState:
    """Trust-region state for TuRBO-1 (§10.3).

    The trust region is an axis-aligned hypercube of side-length 2σ (in
    normalised [0,1]^d space) centred on the current best θ.

    State machine:
        After each batch evaluation:
        - If any new point improves the best Chebyshev value:
            successive_successes += 1;  successive_failures = 0
        - Otherwise:
            successive_failures += 1;  successive_successes = 0
        - σ doubles  when successive_successes ≥ success_tol
        - σ halves   when successive_failures  ≥ failure_tol
        - σ clipped to [σ_min, σ_max]

    Gradient inflation (§10.3, β_grad):
        When the discrepancy between the physics proxy gradient and the GEK
        posterior mean gradient exceeds r_thresh (cosine distance), σ² is
        temporarily inflated by β_grad to encourage wider exploration.

    Attributes:
        center:              (d,)  best θ found so far (normalised)
        sigma:               float current trust-region half-side
        sigma_min / sigma_max float absolute bounds on sigma
        successive_successes int
        successive_failures  int
        success_tol / failure_tol int
        best_y:              float best Chebyshev value observed so far
        n_restarts:          int   number of times trust region was restarted
    """

    config:              MOLLMBOConfig
    center:              NDArray = field(default=None)   # set on first update
    sigma:               float   = 0.2
    sigma_min:           float   = field(init=False)
    sigma_max:           float   = field(init=False)
    successive_successes: int    = 0
    successive_failures:  int    = 0
    success_tol:         int    = field(init=False)
    failure_tol:         int    = field(init=False)
    best_y:              float   = np.inf
    n_restarts:          int     = 0

    def __post_init__(self) -> None:
        self.sigma_min   = self.config.sigma_min_frac
        self.sigma_max   = self.config.sigma_max_frac
        self.success_tol = self.config.turbo_success_tol
        self.failure_tol = self.config.turbo_failure_tol

    # ── State update ──────────────────────────────────────────────────────────

    def update(
        self,
        new_centers:  NDArray,   # (batch, d)  newly evaluated θ (normalised)
        new_y_values: NDArray,   # (batch,)    Chebyshev values
    ) -> bool:
        """Update TR state after evaluating a batch.

        Returns:
            improved: True if any new point improved best_y
        """
        best_idx = int(np.argmin(new_y_values))
        y_new    = float(new_y_values[best_idx])
        improved = y_new < self.best_y

        if improved:
            self.best_y             = y_new
            self.center             = new_centers[best_idx].copy()
            self.successive_successes += 1
            self.successive_failures  = 0
        else:
            self.successive_failures  += 1
            self.successive_successes  = 0

        self._adjust_sigma()
        return improved

    def _adjust_sigma(self) -> None:
        """Expand or contract trust region; clip to [sigma_min, sigma_max]."""
        if self.successive_successes >= self.success_tol:
            self.sigma = min(self.sigma * 2.0, self.sigma_max)
            self.successive_successes = 0
            log.debug("TuRBO: expanded σ → %.4f", self.sigma)
        elif self.successive_failures >= self.failure_tol:
            self.sigma = max(self.sigma / 2.0, self.sigma_min)
            self.successive_failures = 0
            log.debug("TuRBO: contracted σ → %.4f", self.sigma)

            if self.sigma <= self.sigma_min + 1e-8:
                self.n_restarts += 1
                self.sigma               = 0.2       # restart to default
                self.successive_failures = 0
                self.successive_successes = 0
                log.info("TuRBO: restarted (σ_min reached), restart #%d", self.n_restarts)

    # ── Gradient inflation ────────────────────────────────────────────────────

    def inflate_sigma_if_disagreement(
        self,
        grad_phys: NDArray,   # (d,)  physics proxy gradient (normalised space)
        grad_gek:  NDArray,   # (d,)  GEK posterior gradient
    ) -> None:
        """Inflate σ when physics and data gradients disagree (§10.3).

        Disagreement is measured as 1 − cosine_similarity.
        Inflates σ² by β_grad when disagreement > r_thresh.
        """
        norm_p = np.linalg.norm(grad_phys) + 1e-30
        norm_g = np.linalg.norm(grad_gek)  + 1e-30
        cos_sim = float(np.dot(grad_phys, grad_gek) / (norm_p * norm_g))
        discrepancy = 1.0 - cos_sim

        if discrepancy > self.config.r_thresh:
            beta = self.config.beta_grad_inflate
            self.sigma = min(
                self.sigma * np.sqrt(1.0 + beta),
                self.sigma_max,
            )
            log.debug(
                "TuRBO: grad discrepancy=%.3f > r_thresh=%.2f; σ → %.4f",
                discrepancy, self.config.r_thresh, self.sigma,
            )

    # ── Trust region clipping ─────────────────────────────────────────────────

    def clip_to_trust_region(
        self,
        X: NDArray,   # (n, d) candidates in normalised space
    ) -> NDArray:
        """Clip candidates to the trust region hypercube.

        Trust region: [center − σ, center + σ]^d  ∩  [0, 1]^d.

        Args:
            X: (n, d)

        Returns:
            (n, d) clipped candidates
        """
        if self.center is None:
            return X
        lo = np.clip(self.center - self.sigma, 0.0, 1.0)
        hi = np.clip(self.center + self.sigma, 0.0, 1.0)
        return np.clip(X, lo, hi)

    def trust_region_bounds(self) -> NDArray | None:
        """(d, 2) bounds for the trust region in normalised space, or None."""
        if self.center is None:
            return None
        lo = np.clip(self.center - self.sigma, 0.0, 1.0)
        hi = np.clip(self.center + self.sigma, 0.0, 1.0)
        return np.column_stack([lo, hi])

    def trust_region_bounds_raw(self, config: MOLLMBOConfig) -> NDArray | None:
        """(d, 2) bounds in ORIGINAL parameter space, or None."""
        norm_bounds = self.trust_region_bounds()
        if norm_bounds is None:
            return None
        bounds = config.bounds_array      # (d, 2)
        lo_raw = bounds[:, 0] + norm_bounds[:, 0] * (bounds[:, 1] - bounds[:, 0])
        hi_raw = bounds[:, 0] + norm_bounds[:, 1] * (bounds[:, 1] - bounds[:, 0])
        return np.column_stack([lo_raw, hi_raw])


# ══════════════════════════════════════════════════════════════════════════════
#  OptimizerState
# ══════════════════════════════════════════════════════════════════════════════

class OptimizerState:
    """Aggregated optimisation state shared across BO iterations (§10.6).

    Manages:
        - RISE weight sequence (pre-generated once in __init__)
        - z_star  (m,): current ideal point in transformed objective space
        - z_nad   (m,): current nadir point in transformed objective space
        - Pareto front cache (indices into Database.get_valid_*)

    All objective vectors stored here are in TRANSFORMED space:
        log10-aged aging column + min-max normalised to [0,1]^m.
    z_star and z_nad operate in the SAME transformed space as the GEK targets.

    Key methods:
        update_from_database(db)        recompute z*, z^nad, Pareto front
        scalarise(objs_t, lambda_t)     → (n,) F_tch^aug values
        get_lambda(t)                   → (3,) weight vector for iteration t
    """

    def __init__(self, config: MOLLMBOConfig) -> None:
        self.config = config

        # RISE sequence — generated once, length = t_max
        from acquisition_mo import RISEWeights
        self._rise = RISEWeights(
            K    = config.t_max,
            m    = config.n_obj,
            s    = config.rise_s,
            seed = config.random_seed,
        )
        log.info(
            "OptimizerState: RISE K=%d m=%d s=%.1f generated",
            config.t_max, config.n_obj, config.rise_s,
        )

        # Ideal and nadir points (transformed space); initialised conservatively
        self.z_star = np.zeros(config.n_obj)
        self.z_nad  = np.ones(config.n_obj)

        # Running bounds used by normalize_objectives
        self._f_min_trans: NDArray | None = None
        self._f_max_trans: NDArray | None = None

        # Pareto front cache (raw objectives, updated by update_from_database)
        self._pareto_objs_raw:   NDArray = np.empty((0, config.n_obj))
        self._pareto_thetas_raw: NDArray = np.empty((0, config.dim))

        self.iteration: int = 0    # current BO iteration index (0-based)

    # ── Weight sequence access ────────────────────────────────────────────────

    def get_lambda(self, t: int | None = None) -> NDArray:
        """Return λ_t for iteration t (default: self.iteration).

        Cycles modulo K so the caller never needs to manage index bounds.
        """
        t = t if t is not None else self.iteration
        return self._rise[t]

    @property
    def rise_sequence(self) -> NDArray:
        """(K, m) full RISE sequence."""
        return self._rise.sequence

    # ── State update ──────────────────────────────────────────────────────────

    def update_from_database(self, db: Database) -> None:
        """Recompute z*, z^nad, and Pareto front from all valid observations.

        Should be called after each evaluate_batch + Database.add_batch.

        Pipeline:
            1. Get valid raw objectives from db
            2. Log-transform aging if config.log_aging
            3. Update running min/max bounds (f_min, f_max)
            4. Normalise → [0,1]^m
            5. z_star = column-wise min of normalised objectives
            6. z_nad  = column-wise max of normalised objectives
            7. Recompute Pareto front on raw (pre-normalisation) objectives
        """
        objs_raw = db.get_valid_objs()    # (n, 3)  raw
        if len(objs_raw) == 0:
            return

        # Log-transform aging column
        objs_t = log_transform_objectives(objs_raw, log_time=self.config.log_time, log_aging=self.config.log_aging)

        # Update running normalisation bounds
        objs_norm, self._f_min_trans, self._f_max_trans = normalize_objectives(
            objs_t,
            f_min = self._f_min_trans,
            f_max = self._f_max_trans,
            eps   = self.config.eps_norm,
        )

        # Ideal and nadir in normalised space
        self.z_star = objs_norm.min(axis=0)   # (3,)
        self.z_nad  = objs_norm.max(axis=0)   # (3,)

        # Ensure z_nad > z_star by at least tch_eps
        gap = self.z_nad - self.z_star
        self.z_nad  = np.where(gap < self.config.tch_eps,
                               self.z_star + self.config.tch_eps,
                               self.z_nad)

        # Pareto front on RAW objectives (for HV reporting and LLM context)
        try:
            fronts = non_dominated_sort(objs_raw)
            pf_idx = fronts[0] if fronts else []
        except (IndexError, Exception):
            n_ = len(objs_raw)
            dom = np.zeros(n_, dtype=bool)
            for i in range(n_):
                for j in range(n_):
                    if i != j and np.all(objs_raw[j] <= objs_raw[i]) and np.any(objs_raw[j] < objs_raw[i]):
                        dom[i] = True; break
            pf_idx = list(np.where(~dom)[0])
        if pf_idx:
            thetas = db.get_valid_thetas()
            self._pareto_objs_raw   = objs_raw[pf_idx]
            self._pareto_thetas_raw = thetas[pf_idx]
        else:
            self._pareto_objs_raw   = np.empty((0, self.config.n_obj))
            self._pareto_thetas_raw = np.empty((0, self.config.dim))

        log.debug(
            "OptimizerState: z*=%s  z^nad=%s  |PF|=%d",
            np.round(self.z_star, 3),
            np.round(self.z_nad, 3),
            len(self._pareto_objs_raw),
        )

    # ── Chebyshev scalarisation ───────────────────────────────────────────────

    def scalarise(
        self,
        objs_raw: NDArray,    # (n, 3) or (3,)  raw objectives
        lambda_t: NDArray,    # (3,)
    ) -> NDArray:
        """Augmented Chebyshev scalarisation (§10.6).

        F_tch^aug(θ; λ) = max_k{λ_k·ĝ_k} + ρ·Σ_k λ_k·ĝ_k
        ĝ_k = (f_k_transformed − z*_k) / (z^nad_k − z*_k + ε)

        Args:
            objs_raw:  (n, 3) or (3,)  raw objectives
            lambda_t:  (3,)

        Returns:
            (n,) or scalar  Chebyshev scalarised values
        """
        scalar_input = objs_raw.ndim == 1
        if scalar_input:
            objs_raw = objs_raw[None, :]   # → (1, 3)

        objs_t = log_transform_objectives(objs_raw, log_time=self.config.log_time, log_aging=self.config.log_aging)

        eps   = self.config.tch_eps
        denom = self.z_nad - self.z_star + eps
        g_hat = (objs_t - self.z_star) / denom   # (n, 3)

        weighted = lambda_t * g_hat               # (n, 3)
        tch = weighted.max(axis=1) + self.config.tch_rho * weighted.sum(axis=1)

        return float(tch[0]) if scalar_input else tch

    def i_star(
        self,
        objs_raw: NDArray,   # (n, 3)
        lambda_t: NDArray,   # (3,)
    ) -> NDArray:
        """Per-point dominant objective index i* = argmax_k{λ_k·ĝ_k}.

        Returns:
            (n,) int  dominant objective index for each observation
        """
        objs_t = log_transform_objectives(objs_raw, log_time=self.config.log_time, log_aging=self.config.log_aging)
        eps    = self.config.tch_eps
        denom  = self.z_nad - self.z_star + eps
        g_hat  = (objs_t - self.z_star) / denom
        return np.argmax(lambda_t * g_hat, axis=1).astype(int)   # (n,)

    # ── Pareto front properties ───────────────────────────────────────────────

    @property
    def pareto_objs(self) -> NDArray:
        """(k, 3) raw objectives on the current Pareto front."""
        return self._pareto_objs_raw

    @property
    def pareto_thetas(self) -> NDArray:
        """(k, 7) parameter vectors on the current Pareto front."""
        return self._pareto_thetas_raw

    def pareto_summary(self) -> dict:
        """Compact Pareto front summary for LLM context and logging."""
        pf = self._pareto_objs_raw
        if len(pf) == 0:
            return {"n_pareto": 0}

        summary = {
            "n_pareto":        len(pf),
            "t_charge_range":  [float(pf[:, 0].min()), float(pf[:, 0].max())],
            "T_peak_range":    [float(pf[:, 1].min()), float(pf[:, 1].max())],
            "aging_range":     [float(pf[:, 2].min()), float(pf[:, 2].max())],
            "z_star":          self.z_star.tolist(),
            "z_nad":           self.z_nad.tolist(),
        }

        # Best-in-class points
        summary["best_t_charge"] = {
            k: float(v) for k, v in zip(
                ["t_charge", "T_peak", "aging"],
                pf[pf[:, 0].argmin()],
            )
        }
        summary["best_T_peak"] = {
            k: float(v) for k, v in zip(
                ["t_charge", "T_peak", "aging"],
                pf[pf[:, 1].argmin()],
            )
        }
        summary["best_aging"] = {
            k: float(v) for k, v in zip(
                ["t_charge", "T_peak", "aging"],
                pf[pf[:, 2].argmin()],
            )
        }
        return summary