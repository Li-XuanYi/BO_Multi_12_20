"""Main optimisation loop for MO-LLMBO.

Algorithm 6 (§10, FrameWork.md):
    Phase 0 – Warm-start (n_init evaluations, LLM-guided sobol/random)
    Phase 1 – BO iterations t = 0 … t_max-1
        For each iteration t:
        (A) Get λ_t = rise_sequence[t]
        (B) Compute GEK training targets  (y_tch, grad_proj)
        (C) Fit / refit GEKModel on all valid data
        (D) TuRBO: build trust-region bounds; generate n_random candidates
        (E) Evaluate EI over trust-region candidates  →  C_acq
        (F) LLM: generate n_cand candidates; parse θ vectors  →  C_llm
        (G) Merge C_acq ∪ C_llm; DPP-select batch_size candidates
        (H) evaluate_batch (PyBaMM, parallel)
        (I) Update Database, OptimizerState, TuRBOState
        (J) Gradient discrepancy check → optional TuRBO σ inflation
        (K) Log / record iteration diagnostics

Public API:
    MOLLMBOptimizer(config)
    .run()         → Database, OptimizerState
    .warm_start()  → list[EvalResult]   (phase 0 only)
    .step(t)       → list[EvalResult]   (single iteration)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from acquisition_mo import DPPSelector, RISEWeights, ScalarEI
from battery_model import EvalResult, evaluate_batch
from config import MOLLMBOConfig, PARAM_NAMES
from database import Database, OptimizerState, TuRBOState
from llm_interface import DatabaseSummarizer, LLMInterface
from physics import GEKModel, PsiFunction

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  MOLLMBOptimizer
# ══════════════════════════════════════════════════════════════════════════════

class MOLLMBOptimizer:
    """Multi-objective LLM-assisted Bayesian Optimiser (§10).

    Usage:
        cfg = MOLLMBOConfig()
        opt = MOLLMBOptimizer(cfg)
        db, state = opt.run()          # full run
        # or step-by-step:
        opt.warm_start()
        for t in range(cfg.t_max):
            opt.step(t)
        db, state = opt.db, opt.state

    After run() the following attributes are populated:
        opt.db        Database  — all observations
        opt.state     OptimizerState — z*, z^nad, Pareto front, RISE seq
        opt.turbo     TuRBOState
        opt.gek       GEKModel  — last-fitted surrogate
        opt.timing    dict      — wall-clock breakdown
    """

    def __init__(self, config: MOLLMBOConfig) -> None:
        self.config = config
        self.rng    = np.random.RandomState(config.random_seed)

        # Core state objects
        self.db     = Database(config)
        self.state  = OptimizerState(config)
        self.turbo  = TuRBOState(config=config)
        self.gek    = GEKModel(config)

        # Auxiliary components
        self.dpp     = DPPSelector(config)
        self.llm     = LLMInterface(config) if config.use_llm else None
        self.summarizer = DatabaseSummarizer(config) if config.use_llm else None

        # Timing and diagnostics
        self.timing: dict[str, float] = {
            "warm_start": 0.0, "gek_fit": 0.0,
            "acq_opt": 0.0, "llm": 0.0, "eval": 0.0,
        }
        self._coupling_matrix: dict | None = None   # LLM touchpoint 1a

    # ── Full run ──────────────────────────────────────────────────────────────

    def run(self) -> tuple[Database, OptimizerState]:
        """Execute complete warm-start + BO loop.

        Returns:
            (db, state)  after all evaluations
        """
        log.info("MO-LLMBO start: tag=%s  n_init=%d  t_max=%d",
                 self.config.ablation_tag(), self.config.n_init, self.config.t_max)

        # ── Phase 0: warm-start ───────────────────────────────────────────
        t0 = time.perf_counter()
        self.warm_start()
        self.timing["warm_start"] = time.perf_counter() - t0
        log.info("Warm-start done: %d valid / %d total (%.1fs)",
                 self.db.n_valid, len(self.db), self.timing["warm_start"])

        # ── Phase 1: BO iterations ────────────────────────────────────────
        for t in range(self.config.t_max):
            t_iter = time.perf_counter()
            results = self.step(t)
            elapsed = time.perf_counter() - t_iter

            pf_size = len(self.state.pareto_objs)
            hv      = self.db.hv_history[-1] if self.db.hv_history else 0.0
            log.info(
                "Iter %3d/%d  λ=%s  new_evals=%d  |PF|=%d  HV=%.4f  (%.1fs)",
                t + 1, self.config.t_max,
                np.round(self.state.get_lambda(t), 2),
                len(results), pf_size, hv, elapsed,
            )

        log.info(
            "MO-LLMBO done: total_evals=%d  valid=%d  |PF|=%d",
            len(self.db), self.db.n_valid, len(self.state.pareto_objs),
        )
        return self.db, self.state

    # ── Phase 0: warm-start ───────────────────────────────────────────────────

    def warm_start(self) -> list[EvalResult]:
        """Generate and evaluate n_init initial candidates (§10.5, Touchpoint 1b).

        Strategy:
            1. If use_llm: ask LLM for coupling matrix (Touchpoint 1a),
               then request n_init warm-start candidates (Touchpoint 1b).
            2. Supplement with Sobol/random samples to reach n_init total.
            3. evaluate_batch → Database.add_batch → OptimizerState.update.
        """
        n_init  = self.config.n_init
        thetas: list[NDArray] = []

        # ── Touchpoint 1a: coupling matrix (LLM, optional) ────────────────
        if self.config.use_llm and self.llm is not None:
            try:
                t0 = time.perf_counter()
                self._coupling_matrix = self.llm.get_coupling_matrices()
                self.timing["llm"] += time.perf_counter() - t0
                log.info("Touchpoint 1a: coupling matrix obtained")
            except Exception as exc:
                log.warning("Touchpoint 1a failed: %s", exc)
                self._coupling_matrix = None

        # ── Touchpoint 1b: LLM warm-start candidates ──────────────────────
        if self.config.use_llm and self.llm is not None:
            try:
                t0 = time.perf_counter()
                summary = self.summarizer.generate_summary(
                    self.db, level="none", lambda_t=None
                )
                llm_cands = self.llm.generate_candidates(
                    summary=summary,
                    n=n_init,
                    lambda_t=None,   # no Chebyshev direction yet
                )
                for cand in llm_cands:
                    th = self._parse_candidate(cand)
                    if th is not None:
                        thetas.append(th)
                self.timing["llm"] += time.perf_counter() - t0
                log.info("Touchpoint 1b: %d LLM warm-start candidates", len(thetas))
            except Exception as exc:
                log.warning("Touchpoint 1b failed: %s", exc)

        # ── Fill remaining with Sobol / random ────────────────────────────
        n_needed = n_init - len(thetas)
        if n_needed > 0:
            thetas.extend(self._random_candidates(n_needed))

        thetas = thetas[:n_init]   # cap at n_init

        # ── Evaluate ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = evaluate_batch(thetas, self.config)
        self.timing["eval"] += time.perf_counter() - t0

        self.db.add_batch(results)
        self.state.update_from_database(self.db)

        # Initialise TuRBO centre at best warm-start point
        self._init_turbo_center()

        log.debug("Warm-start: %d results (%d valid)",
                  len(results), sum(r.constraint_ok for r in results))
        return results

    # ── Single BO iteration ───────────────────────────────────────────────────

    def step(self, t: int) -> list[EvalResult]:
        """Execute one BO iteration (§10, Algorithm 6 inner loop).

        Args:
            t: iteration index (0-based)

        Returns:
            list[EvalResult]  — batch_size new evaluations
        """
        cfg     = self.config
        lambda_t = self.state.get_lambda(t)

        # ── (A) Compute GEK training targets ──────────────────────────────
        t0 = time.perf_counter()
        physics_grads = self._compute_physics_grads()   # (n, 3, 7) or None
        X_norm, y_tch, grad_proj = self.db.get_gp_targets(
            self.state, lambda_t, physics_grads=physics_grads
        )
        log.debug("Iter %d: n_obs=%d  y_tch=[%.3f,%.3f]",
                  t, len(X_norm), y_tch.min() if len(y_tch) else 0,
                              y_tch.max() if len(y_tch) else 0)

        # ── (B) Fit GEKModel (or vanilla GP if n < gek_min_n) ─────────────
        if len(X_norm) >= 2:
            gek_to_fit = self.gek if cfg.use_gek else self._vanilla_gp_wrapper()
            try:
                gek_to_fit.fit(X_norm, y_tch, grad_proj)
                self.gek = gek_to_fit
                gek_ok = True
            except Exception as exc:
                log.warning("GEK fit failed at iter %d: %s", t, exc)
                gek_ok = False
        else:
            gek_ok = False
        self.timing["gek_fit"] += time.perf_counter() - t0

        # ── (C) TuRBO trust-region bounds ─────────────────────────────────
        tr_bounds = self.turbo.trust_region_bounds()    # (d, 2) in [0,1]^d or None
        if tr_bounds is None:
            tr_bounds = np.column_stack([
                np.zeros(cfg.dim), np.ones(cfg.dim)
            ])

        # ── (D) Generate acquisition candidates ───────────────────────────
        t0 = time.perf_counter()
        if gek_ok:
            C_acq = self._generate_acq_candidates(
                y_tch, lambda_t, tr_bounds
            )                                           # (n_acq, d)  normalised
        else:
            # Fallback: random candidates in trust region
            C_acq = self._random_normalised_in_bounds(
                n=cfg.n_cand * 5, bounds=tr_bounds
            )
        self.timing["acq_opt"] += time.perf_counter() - t0

        # ── (E) LLM candidates (Touchpoint 2) ────────────────────────────
        t0 = time.perf_counter()
        C_llm_norm = self._generate_llm_candidates(lambda_t)  # (k, d) normalised
        self.timing["llm"] += time.perf_counter() - t0

        # ── (F) Merge → DPP batch selection ──────────────────────────────
        C_all = np.vstack([C_acq, C_llm_norm]) if len(C_llm_norm) > 0 else C_acq

        if gek_ok:
            mean_all, std_all = self.gek.predict(C_all, return_std=True)
            y_best  = float(y_tch.min()) if len(y_tch) else 0.0
            acq_obj = ScalarEI(self.gek, y_best)
            ei_all  = acq_obj.base_acq(mean_all, std_all)
        else:
            ei_all = self.rng.uniform(0.01, 1.0, len(C_all))

        if cfg.use_dpp:
            sel_idx = self.dpp.select(
                C_all, ei_all, k=cfg.batch_size,
                seed=int(self.rng.randint(0, 2**31))
            )
            C_batch_norm = C_all[sel_idx]
        else:
            # Greedy top-k by EI
            top_idx      = np.argsort(ei_all)[-cfg.batch_size:]
            C_batch_norm = C_all[top_idx]

        # ── (G) Denormalise → original parameter space ────────────────────
        thetas_batch = self._denormalise(C_batch_norm)

        # ── (H) Evaluate batch ────────────────────────────────────────────
        t0 = time.perf_counter()
        results = evaluate_batch(thetas_batch, self.config)
        self.timing["eval"] += time.perf_counter() - t0

        # ── (I) Update Database and OptimizerState ────────────────────────
        self.db.add_batch(results)
        self.state.update_from_database(self.db)
        self.state.iteration = t + 1

        # TuRBO update: compute y_tch for new points
        new_y = self._tch_values(
            np.vstack([r.objectives for r in results]),
            lambda_t,
        )
        new_X_norm = self._normalise(thetas_batch)
        improved   = self.turbo.update(new_X_norm, new_y)

        # ── (J) Gradient discrepancy check ────────────────────────────────
        if cfg.use_turbo and gek_ok and self.turbo.center is not None:
            self._gradient_discrepancy_check(lambda_t)

        # ── (K) Record diagnostics ────────────────────────────────────────
        self.db.record_iteration(self.state, lambda_t)

        log.debug(
            "Iter %d: improved=%s  σ=%.3f  valid_new=%d",
            t, improved, self.turbo.sigma,
            sum(r.constraint_ok for r in results),
        )
        return results

    # ── Acquisition candidate generation ──────────────────────────────────────

    def _generate_acq_candidates(
        self,
        y_tch:    NDArray,    # (n,) current Chebyshev targets
        lambda_t: NDArray,    # (3,)
        tr_bounds: NDArray,   # (d, 2) in [0,1]^d
    ) -> NDArray:
        """Sample random candidates in trust region; return top by EI.

        Generates n_random random points inside tr_bounds, evaluates EI,
        and returns the top n_cand (no L-BFGS-B here — TuRBO uses random
        search + the DPP diversity selection replaces gradient-based acq. max).

        Returns:
            (n_cand, d)  normalised candidates sorted by descending EI
        """
        cfg       = self.config
        n_random  = max(cfg.n_cand * 20, 2000)
        y_best    = float(y_tch.min()) if len(y_tch) > 0 else 0.0

        X_rand   = self._random_normalised_in_bounds(n_random, tr_bounds)
        acq_obj  = ScalarEI(self.gek, y_best, xi=0.01)
        ei_vals  = acq_obj.base_acq(*self.gek.predict(X_rand, return_std=True))

        # Return top n_cand (diversity handled by DPP later)
        top_k    = min(cfg.n_cand * 3, len(X_rand))
        top_idx  = np.argsort(ei_vals)[-top_k:][::-1]
        return X_rand[top_idx]

    def _generate_llm_candidates(self, lambda_t: NDArray) -> NDArray:
        """Touchpoint 2: LLM-proposed candidates → normalised θ vectors.

        Returns:
            (k, d) normalised; empty (0, d) if LLM disabled or failed
        """
        cfg = self.config
        if not cfg.use_llm or self.llm is None:
            return np.empty((0, cfg.dim))

        try:
            summary = self.summarizer.generate_summary(
                self.db, level=cfg.context_level, lambda_t=lambda_t
            )
            raw_cands = self.llm.generate_candidates(
                summary=summary,
                n=cfg.n_cand,
                lambda_t=lambda_t,
            )
            thetas = []
            for cand in raw_cands:
                th = self._parse_candidate(cand)
                if th is not None:
                    thetas.append(th)

            if not thetas:
                return np.empty((0, cfg.dim))

            return self._normalise(np.vstack(thetas))

        except Exception as exc:
            log.warning("LLM Touchpoint 2 failed: %s", exc)
            return np.empty((0, cfg.dim))

    # ── Physics gradient computation ──────────────────────────────────────────

    def _compute_physics_grads(self) -> NDArray | None:
        """Return (n_valid, 3, 7) analytical gradients, or None if GEK disabled."""
        if not self.config.use_gek:
            return None
        thetas = self.db.get_valid_thetas()
        if len(thetas) == 0:
            return None
        try:
            _, grads = PsiFunction.batch(thetas, self.config)
            return grads   # (n, 3, 7)
        except Exception as exc:
            log.warning("PsiFunction.batch failed: %s", exc)
            return None

    # ── TuRBO helpers ─────────────────────────────────────────────────────────

    def _init_turbo_center(self) -> None:
        """Set TuRBO centre to best warm-start point (by Chebyshev value)."""
        thetas = self.db.get_valid_thetas()
        objs   = self.db.get_valid_objs()
        if len(thetas) == 0:
            return

        lam0 = self.state.get_lambda(0)
        y    = self.state.scalarise(objs, lam0)   # (n,)
        best = int(np.argmin(y))

        self.turbo.best_y = float(y[best])
        self.turbo.center = self._normalise(thetas[[best]])[0]
        log.debug("TuRBO init: center=%s  best_y=%.4f",
                  np.round(self.turbo.center, 3), self.turbo.best_y)

    def _gradient_discrepancy_check(self, lambda_t: NDArray) -> None:
        """Inflate TuRBO σ when physics and GEK gradients disagree (§10.3).

        Evaluates both gradients at the current trust-region centre (in
        original space), then delegates to TuRBOState.inflate_sigma_if_disagreement.
        """
        try:
            center_raw = self._denormalise(self.turbo.center[None, :])[0]

            # Physics gradient at center
            _, g_psi = PsiFunction.compute(center_raw, self.config)
            lam = lambda_t
            i_star = int(np.argmax(lam * g_psi.mean(axis=1)))
            grad_phys_raw = lam[i_star] * g_psi[i_star]   # (7,)

            # Normalise gradient to [0,1]^d scale
            ranges = self.config.param_ranges                # (7,)
            grad_phys = grad_phys_raw * ranges

            # GEK posterior gradient: finite difference at center (cheap)
            c0    = self.turbo.center
            eps_fd = 1e-4
            mean0 = float(self.gek.predict(c0[None, :])[0])
            grad_gek = np.zeros(self.config.dim)
            for k in range(self.config.dim):
                ck       = c0.copy(); ck[k] += eps_fd
                ck       = np.clip(ck, 0.0, 1.0)
                mean_k   = float(self.gek.predict(ck[None, :])[0])
                grad_gek[k] = (mean_k - mean0) / eps_fd

            self.turbo.inflate_sigma_if_disagreement(grad_phys, grad_gek)

        except Exception as exc:
            log.debug("Gradient discrepancy check failed: %s", exc)

    # ── Chebyshev scalarisation for new points ────────────────────────────────

    def _tch_values(
        self,
        objs: NDArray,    # (n, 3) raw objectives
        lambda_t: NDArray,
    ) -> NDArray:
        """Compute augmented Chebyshev values for a batch of raw objectives.

        Handles penalty rows (failed/violated sims) by returning a large value.
        """
        n   = len(objs)
        out = np.full(n, 1e6)
        for i in range(n):
            if np.any(objs[i] >= 999.0):
                continue
            out[i] = float(self.state.scalarise(objs[i], lambda_t))
        return out

    # ── Coordinate transformations ────────────────────────────────────────────

    def _normalise(self, thetas: NDArray) -> NDArray:
        """(n, d) raw θ → (n, d) normalised to [0,1]^d."""
        bounds = self.config.bounds_array
        lo, hi = bounds[:, 0], bounds[:, 1]
        return np.clip((thetas - lo) / (hi - lo + 1e-30), 0.0, 1.0)

    def _denormalise(self, X_norm: NDArray) -> NDArray:
        """(n, d) normalised [0,1]^d → (n, d) raw parameter space."""
        bounds = self.config.bounds_array
        lo, hi = bounds[:, 0], bounds[:, 1]
        return lo + X_norm * (hi - lo)

    def _random_candidates(self, n: int) -> list[NDArray]:
        """n random θ vectors sampled uniformly within bounds."""
        bounds = self.config.bounds_array
        lo, hi = bounds[:, 0], bounds[:, 1]
        thetas = lo + self.rng.uniform(size=(n, self.config.dim)) * (hi - lo)
        return [thetas[i] for i in range(n)]

    def _random_normalised_in_bounds(
        self,
        n: int,
        bounds: NDArray,    # (d, 2) in [0,1]^d
    ) -> NDArray:
        """(n, d) random points inside the given [0,1]^d sub-bounds."""
        lo, hi = bounds[:, 0], bounds[:, 1]
        return lo + self.rng.uniform(size=(n, self.config.dim)) * (hi - lo)

    # ── Candidate parsing ─────────────────────────────────────────────────────

    def _parse_candidate(self, cand: dict | list | Any) -> NDArray | None:
        """Convert an LLM-returned candidate to a (7,) θ array.

        Handles three formats from LLMInterface.generate_candidates:
            dict  with keys matching PARAM_NAMES  (primary)
            list  of 7 floats in PARAM_NAMES order
            other → None (discarded)

        Also enforces hard bounds and rejects out-of-range candidates.
        """
        bounds = self.config.bounds_array   # (7, 2)

        try:
            if isinstance(cand, dict):
                theta = np.array([float(cand[k]) for k in PARAM_NAMES])
            elif isinstance(cand, (list, tuple, np.ndarray)):
                theta = np.asarray(cand, dtype=float).ravel()
                if len(theta) != self.config.dim:
                    return None
            else:
                return None
        except (KeyError, ValueError, TypeError):
            return None

        # Clip to bounds (LLM may produce slightly out-of-range values)
        theta = np.clip(theta, bounds[:, 0], bounds[:, 1])
        return theta

    # ── Vanilla GP wrapper (ablation: use_gek=False) ─────────────────────────

    def _vanilla_gp_wrapper(self) -> GEKModel:
        """Return a GEKModel configured to always use the vanilla sklearn GP.

        Used when config.use_gek=False; the GEKModel falls back to sklearn GP
        for n < gek_min_n, so we temporarily set gek_min_n=∞ to force it.
        """
        from copy import deepcopy
        cfg_copy           = deepcopy(self.config)
        cfg_copy.gek_min_n = 10**9   # force vanilla GP always
        return GEKModel(cfg_copy)


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience entry-point
# ══════════════════════════════════════════════════════════════════════════════

def run_optimization(
    config: MOLLMBOConfig | None = None,
    **kwargs: Any,
) -> tuple[Database, OptimizerState]:
    """One-line entry point.

    Args:
        config: MOLLMBOConfig (default: MOLLMBOConfig())
        **kwargs: overrides applied to a fresh MOLLMBOConfig if config is None

    Returns:
        (db, state) after the full run
    """
    if config is None:
        config = MOLLMBOConfig(**kwargs)
    return MOLLMBOptimizer(config).run()