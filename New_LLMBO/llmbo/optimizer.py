from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from DataBase.database import DEFAULT_BOUNDS, ObservationDB
from llm.llm_interface import IterationGuidance, build_llm_interface
from llmbo.acquisition import build_acquisition_function
from llmbo.gp_model import build_gp_stack
from pybamm_simulator import PyBaMMSimulator
from utils.constants import DSOC_SUM_MAX as CANONICAL_DSOC_SUM_MAX, IDEAL_POINT, REF_POINT

logger = logging.getLogger(__name__)

PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]
DSOC_SUM_MAX = CANONICAL_DSOC_SUM_MAX


DEFAULT_CONFIG = {
    "max_iterations": 20,
    "n_warmstart": 10,
    "n_random_init": 3,
    "init_strategy": "manual",
    "init_budget": None,
    "warmstart_ratio": 0.5,
    "fixed_init_points": None,
    "fixed_init_source": "shared_init",
    "n_candidates": 15,
    "n_select": 1,
    "warmstart_batch_size": 10,
    "warmstart_max_attempts": 4,
    "warmstart_hv_log_interval": 5,
    "llm_backend": os.getenv("LLM_BACKEND", "openai"),
    "llm_model": os.getenv("LLM_MODEL", "gpt-4.1-mini"),
    "llm_api_base": os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.nuwaapi.com/v1"),
    "llm_api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
    "llm_n_samples": 3,
    "llm_temperature": 0.7,
    "kernel_nu": 2.5,
    "gp_alpha": 1e-6,
    "gp_normalize_y": True,
    "gp_n_restarts_optimizer": 5,
    "ei_n_restarts": 12,
    "ei_n_random_samples": 96,
    "riesz_n_div": 10,
    "riesz_s": 2.0,
    "riesz_n_iter": 300,
    "riesz_lr": 5e-3,
    "riesz_seed": 42,
    "w_sample_seed": 0,
    "init_seed": 2026,
    "eta": 0.05,
    "enable_iterative_guidance": True,
    "enable_gp_llm_coupling": True,
    "guidance_grid_size": 64,
    "guidance_point_grid_size": 25,
    "guidance_point_local_scale": 0.75,
    "guidance_probe_size": 128,
    "guidance_hotspots": 5,
    "checkpoint_dir": "checkpoints",
    "checkpoint_every": 5,
    "battery_model": "LG INR21700-M50 (Chen2020)",
}


def log_transform_objectives(Y_raw: np.ndarray) -> np.ndarray:
    Y_raw = np.atleast_2d(np.asarray(Y_raw, dtype=float))
    Y_tilde = Y_raw.copy()
    Y_tilde[:, 0] = np.log10(np.maximum(Y_raw[:, 0], 1.0))
    Y_tilde[:, 2] = np.log10(np.maximum(Y_raw[:, 2], 1e-12))
    return Y_tilde


def compute_dynamic_bounds(Y_tilde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Y_tilde = np.atleast_2d(np.asarray(Y_tilde, dtype=float))
    return Y_tilde.min(axis=0), Y_tilde.max(axis=0)


def normalize_objectives(
    Y_tilde: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    denom = np.asarray(y_max, dtype=float) - np.asarray(y_min, dtype=float)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return (np.asarray(Y_tilde, dtype=float) - np.asarray(y_min, dtype=float)) / denom


def compute_tchebycheff(
    Y_bar: np.ndarray,
    w_vec: np.ndarray,
    eta: float = 0.05,
) -> np.ndarray:
    Y_bar = np.atleast_2d(np.asarray(Y_bar, dtype=float))
    w = np.asarray(w_vec, dtype=float).ravel()
    weighted = Y_bar * w[np.newaxis, :]
    return weighted.max(axis=1) + float(eta) * weighted.sum(axis=1)


def compute_tchebycheff_from_raw(
    Y_raw: np.ndarray,
    w_vec: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
    eta: float = 0.05,
) -> np.ndarray:
    Y_tilde = log_transform_objectives(Y_raw)
    Y_bar = normalize_objectives(Y_tilde, y_min, y_max)
    return compute_tchebycheff(Y_bar, w_vec, eta=eta)


def compute_tchebycheff_from_raw_with_ideal(
    Y_raw: np.ndarray,
    w_vec: np.ndarray,
    ideal_point_raw: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
    eta: float = 0.05,
) -> np.ndarray:
    Y_tilde = log_transform_objectives(Y_raw)
    ideal_tilde = log_transform_objectives(np.asarray(ideal_point_raw, dtype=float)[None, :])[0]
    denom = np.asarray(y_max, dtype=float) - np.asarray(y_min, dtype=float)
    denom = np.where(denom < 1e-12, 1.0, denom)
    Y_gap = np.abs(Y_tilde - ideal_tilde[np.newaxis, :]) / denom[np.newaxis, :]
    return compute_tchebycheff(Y_gap, w_vec, eta=eta)


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if np.allclose(v.sum(), 1.0) and np.all(v >= 0.0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = float(cssv[rho] - 1.0) / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def generate_riesz_weight_set(
    n_obj: int = 3,
    n_div: int = 10,
    s: float = 2.0,
    n_iter: int = 300,
    lr: float = 5e-3,
    seed: int = 42,
    eps_min: float = 0.01,
) -> np.ndarray:
    """
    Generate a Riesz-relaxed weight set on the probability simplex.

    This starts from a Das-Dennis grid and applies projected gradient steps on
    the Riesz energy to spread the weights more evenly across the simplex.
    """

    def das_dennis(divisions: int, dimensions: int) -> List[List[int]]:
        if dimensions == 1:
            return [[divisions]]
        points: List[List[int]] = []
        for i in range(divisions + 1):
            for rest in das_dennis(divisions - i, dimensions - 1):
                points.append([i] + rest)
        return points

    W = np.array(das_dennis(n_div, n_obj), dtype=float) / float(n_div)
    W = np.maximum(W, eps_min)
    W = W / W.sum(axis=1, keepdims=True)

    for _ in range(int(n_iter)):
        grad = np.zeros_like(W)
        for i in range(len(W)):
            diff = W[i] - W
            dist2 = np.sum(diff ** 2, axis=1)
            dist2[i] = np.inf
            factor = float(s) / (dist2 ** ((float(s) + 2.0) / 2.0) + 1e-15)
            factor[i] = 0.0
            grad[i] = np.sum(factor[:, None] * diff, axis=0)

        W = W + float(lr) * grad
        for i in range(len(W)):
            W[i] = _project_to_simplex(W[i])
        W = np.maximum(W, eps_min)
        W = W / W.sum(axis=1, keepdims=True)

    return W


class BayesOptimizer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        user_cfg = config or {}
        self.cfg = {**DEFAULT_CONFIG, **user_cfg}
        if "enable_gp_llm_coupling" not in user_cfg and "enable_region_lift" in user_cfg:
            self.cfg["enable_gp_llm_coupling"] = bool(user_cfg["enable_region_lift"])
        if self.cfg.get("n_candidates") and not config.get("ei_n_random_samples") if config else True:
            self.cfg.setdefault("ei_n_random_samples", max(64, int(self.cfg["n_candidates"]) * 8))

        seed = self.cfg.get("w_sample_seed")
        self._rng = np.random.default_rng(seed)
        self._weight_order: List[int] = []

        self.param_bounds = {k: tuple(v) for k, v in DEFAULT_BOUNDS.items()}
        self.simulator: Optional[PyBaMMSimulator] = None
        self.database: Optional[ObservationDB] = None
        self.llm: Any = None
        self.psi_fn: Any = None
        self.gp: Any = None
        self.af: Any = None
        self._weight_set: Optional[np.ndarray] = None
        self._warmstart_hv_trace: List[Dict[str, Any]] = []
        self._hv_eval_trace: List[Dict[str, Any]] = []
        self._y_tilde_min = np.zeros(3, dtype=float)
        self._y_tilde_max = np.ones(3, dtype=float)
        self._previous_guidance: Optional[Dict[str, Any]] = None
        self._last_coupling_summary: Optional[Dict[str, Any]] = None

        Path(self.cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    def setup(self) -> None:
        logger.info("=" * 60)
        logger.info("Setting up 5D GP-LLM-coupled MOBO optimizer")
        logger.info("=" * 60)

        self.simulator = PyBaMMSimulator()
        self.param_bounds = {
            key: tuple(bounds) for key, bounds in getattr(self.simulator, "param_bounds", DEFAULT_BOUNDS).items()
        }

        self.database = ObservationDB(
            param_bounds=self.param_bounds,
            ref_point=REF_POINT.copy(),
            ideal_point=IDEAL_POINT.copy(),
            normalize=True,
        )

        backend = str(self.cfg["llm_backend"]).lower()
        api_key = str(self.cfg.get("llm_api_key") or "")
        if backend != "mock" and not api_key:
            logger.warning("No valid LLM API key found; falling back to mock warmstart backend")
            backend = "mock"
        self.cfg["llm_backend"] = backend

        self.llm = build_llm_interface(
            param_bounds=self.param_bounds,
            backend=backend,
            model=self.cfg["llm_model"],
            api_base=self.cfg["llm_api_base"],
            api_key=api_key,
            n_samples=self.cfg["llm_n_samples"],
            temperature=self.cfg["llm_temperature"],
            battery_model=self.cfg["battery_model"],
        )

        self.psi_fn, _, _, self.gp = build_gp_stack(
            param_bounds=self.param_bounds,
            kernel_nu=self.cfg["kernel_nu"],
            alpha=self.cfg["gp_alpha"],
            normalize_y=self.cfg["gp_normalize_y"],
            n_restarts_optimizer=self.cfg["gp_n_restarts_optimizer"],
            random_state=self.cfg.get("w_sample_seed"),
        )

        self.af = build_acquisition_function(
            gp=self.gp,
            psi_fn=self.psi_fn,
            param_bounds=self.param_bounds,
            n_select=self.cfg["n_select"],
            n_restarts_optimizer=self.cfg["ei_n_restarts"],
            n_random_candidates=self.cfg["ei_n_random_samples"],
            random_seed=self.cfg.get("w_sample_seed"),
        )

        from llmbo.riesz_cache import load_or_generate_riesz

        self._weight_set = load_or_generate_riesz(
            n_obj=3,
            n_div=self.cfg["riesz_n_div"],
            s=self.cfg["riesz_s"],
            n_iter=self.cfg["riesz_n_iter"],
            lr=self.cfg["riesz_lr"],
            seed=self.cfg["riesz_seed"],
        )
        logger.info("Riesz weight set ready: shape=%s", self._weight_set.shape)

    def run_initialization(self) -> None:
        n_warmstart, n_random_init = self._resolve_init_counts()
        logger.info("=" * 60)
        logger.info(
            "Initialization: strategy=%s warmstart=%d random_init=%d",
            self.cfg.get("init_strategy", "manual"),
            n_warmstart,
            n_random_init,
        )
        logger.info("=" * 60)

        scheduled: List[Tuple[str, np.ndarray]] = []
        fixed_init_points = self.cfg.get("fixed_init_points")

        if fixed_init_points is not None:
            fixed = np.atleast_2d(np.asarray(fixed_init_points, dtype=float))
            logger.info("Using %d fixed initialization points from config", fixed.shape[0])
            source = str(self.cfg.get("fixed_init_source", "shared_init"))
            scheduled.extend((source, self._repair_theta(theta)) for theta in fixed)
        else:
            if n_warmstart > 0:
                warmstart_points = self.llm.generate_warmstart_candidates(
                    n=n_warmstart,
                    batch_size=int(self.cfg["warmstart_batch_size"]),
                    max_attempts=int(self.cfg["warmstart_max_attempts"]),
                )
                scheduled.extend(("llm_warmstart", theta) for theta in warmstart_points)

            if n_random_init > 0:
                random_points = self._lhs_candidates(
                    n_random_init,
                    seed=int(self.cfg.get("init_seed", self.cfg.get("w_sample_seed", 2026) or 2026)),
                )
                scheduled.extend(("random_init", theta) for theta in random_points)

        scheduled = self._deduplicate_tagged_points(scheduled)
        hv_trace: List[Dict[str, Any]] = []
        log_interval = max(1, int(self.cfg.get("warmstart_hv_log_interval", 5)))

        for i, (source, theta) in enumerate(scheduled, start=1):
            logger.info("Init [%d/%d] src=%s theta=%s", i, len(scheduled), source, np.round(theta, 4))
            t0 = time.perf_counter()
            result = self.simulator.evaluate(theta)
            elapsed = time.perf_counter() - t0
            self.database.add_from_simulator(
                theta=theta,
                result=result,
                source=source,
                iteration=0,
            )
            self._record_hv_snapshot(
                phase="init",
                iteration=0,
                source=source,
                theta=theta,
                feasible=bool(result["feasible"]),
                elapsed_s=elapsed,
            )
            logger.info(
                "  -> feasible=%s obj=%s (%.1fs)",
                result["feasible"],
                np.round(result["raw_objectives"], 6),
                elapsed,
            )

            if i % log_interval == 0 or i == len(scheduled):
                hv_trace.append(
                    {
                        "n_evaluated": i,
                        "hypervolume": self.database.compute_hypervolume(),
                        "hypervolume_raw": self.database.compute_hypervolume_raw(),
                        "pareto_size": self.database.pareto_size,
                    }
                )

        self._warmstart_hv_trace = hv_trace
        if self.database.n_feasible == 0:
            raise RuntimeError("Initialization produced no feasible observations")

    def _resolve_init_counts(self) -> Tuple[int, int]:
        strategy = str(self.cfg.get("init_strategy", "manual")).lower()
        if strategy == "manual":
            return int(self.cfg.get("n_warmstart", 0)), int(self.cfg.get("n_random_init", 0))

        budget = self.cfg.get("init_budget")
        if budget is None:
            budget = int(self.cfg.get("n_warmstart", 0)) + int(self.cfg.get("n_random_init", 0))
        budget = max(0, int(budget))

        if strategy == "warmstart_only":
            return budget, 0
        if strategy == "random_only":
            return 0, budget
        if strategy == "mixed":
            ratio = float(self.cfg.get("warmstart_ratio", 0.5))
            ratio = min(max(ratio, 0.0), 1.0)
            n_warmstart = int(round(budget * ratio))
            n_warmstart = min(max(n_warmstart, 0), budget)
            return n_warmstart, budget - n_warmstart

        raise ValueError(f"Unsupported init_strategy: {strategy}")

    def initialize_acquisition(self) -> None:
        self._update_dynamic_bounds()
        w_init = np.full(3, 1.0 / 3.0, dtype=float)
        _, Y_raw = self.database.get_train_XY(feasible_only=True, normalize_X=False, normalize_Y=False)
        ideal_point_raw = self._compute_dynamic_ideal_point(Y_raw) if Y_raw.size else None
        self.database.update_tchebycheff_context(
            w_vec=w_init,
            y_min=self._y_tilde_min,
            y_max=self._y_tilde_max,
            ideal_point_raw=ideal_point_raw,
            eta=float(self.cfg["eta"]),
        )
        self.af.initialize(self.database, llm_prior=self.llm)

    def run_optimization_loop(self) -> None:
        logger.info("=" * 60)
        logger.info("Optimization loop: %d iterations", self.cfg["max_iterations"])
        logger.info("=" * 60)

        for t in range(int(self.cfg["max_iterations"])):
            iter_start = time.perf_counter()
            logger.info("--- Iteration %d ---", t)

            if self.database.n_feasible < 2:
                logger.warning("Not enough feasible points for GP; adding bootstrap LHS point")
                theta = self._lhs_candidates(1, seed=1000 + t)[0]
                result = self.simulator.evaluate(theta)
                self.database.add_from_simulator(theta=theta, result=result, source="bootstrap", iteration=t + 1)
                self._record_hv_snapshot(
                    phase="bo",
                    iteration=t + 1,
                    source="bootstrap",
                    theta=theta,
                    feasible=bool(result["feasible"]),
                )
                self.database.record_iteration_stats(extra={"t": t, "w_vec": None, "n_new_evals": 1})
                continue

            w_vec = self._next_weight()
            self._update_dynamic_bounds()
            X_train, Y_raw = self.database.get_train_XY(feasible_only=True, normalize_X=False, normalize_Y=False)
            ideal_point_raw = self._compute_dynamic_ideal_point(Y_raw)
            self.database.update_tchebycheff_context(
                w_vec=w_vec,
                y_min=self._y_tilde_min,
                y_max=self._y_tilde_max,
                ideal_point_raw=ideal_point_raw,
                eta=float(self.cfg["eta"]),
            )
            scalar_y = compute_tchebycheff_from_raw_with_ideal(
                Y_raw=Y_raw,
                w_vec=w_vec,
                ideal_point_raw=ideal_point_raw,
                y_min=self._y_tilde_min,
                y_max=self._y_tilde_max,
                eta=float(self.cfg["eta"]),
            )
            self.gp.fit(X_train, scalar_y, w_vec=w_vec, t=t)

            guidance = None
            coupling = None
            X_candidates = None
            uncertainty_hotspots: List[Dict[str, Any]] = []
            self._previous_guidance = None
            self._last_coupling_summary = None

            if bool(self.cfg.get("enable_iterative_guidance", True)):
                uncertainty_hotspots = self._compute_uncertainty_hotspots(t)
                guidance_state = self._build_guidance_state(
                    t=t,
                    w_vec=w_vec,
                    ideal_point_raw=ideal_point_raw,
                    uncertainty_hotspots=uncertainty_hotspots,
                )
                guidance = self.llm.query_iteration_guidance(guidance_state)

                guidance_candidates = None
                if guidance is not None:
                    self._previous_guidance = guidance.to_dict()
                    guidance_payload_candidates = self._build_gp_llm_coupling_from_guidance(guidance, t)
                    if bool(self.cfg.get("enable_gp_llm_coupling", True)):
                        coupling, guidance_candidates = guidance_payload_candidates
                        self._last_coupling_summary = coupling.to_dict()
                        logger.info(
                            "  Guidance mode=%s confidence=%.3f coupling_lambda=%.6f",
                            guidance.mode,
                            guidance.confidence,
                            float(coupling.lambda_value),
                        )
                    else:
                        _, guidance_candidates = guidance_payload_candidates
                        logger.info(
                            "  Guidance mode=%s confidence=%.3f GP-LLM coupling=disabled",
                            guidance.mode,
                            guidance.confidence,
                        )

                hotspot_candidates = np.array(
                    [hotspot["theta"] for hotspot in uncertainty_hotspots],
                    dtype=float,
                ) if uncertainty_hotspots else np.empty((0, len(PARAM_KEYS)))

                if guidance_candidates is not None:
                    X_candidates = guidance_candidates
                    if hotspot_candidates.size:
                        X_candidates = np.vstack([guidance_candidates, hotspot_candidates])
                elif hotspot_candidates.size:
                    X_candidates = hotspot_candidates

            acq_result = self.af.step(
                X_candidates=X_candidates,
                database=self.database,
                t=t,
                w_vec=w_vec,
                lift=coupling,
            )

            n_new = 0
            guidance_payload = (
                json.dumps(guidance.to_dict(), ensure_ascii=False)
                if guidance is not None else None
            )
            for rank, theta in enumerate(acq_result.selected_thetas):
                logger.info("  Evaluate rank=%d theta=%s", rank, np.round(theta, 4))
                t_eval = time.perf_counter()
                sim_result = self.simulator.evaluate(theta)
                elapsed_eval = time.perf_counter() - t_eval
                self.database.add_from_simulator(
                    theta=theta,
                    result=sim_result,
                    source="bo",
                    iteration=t + 1,
                    acq_value=float(acq_result.selected_scores[rank]),
                    acq_type="EI_gp_llm_coupled" if coupling is not None else "EI",
                    gp_pred={
                        "mean_coupled": float(acq_result.all_mean[acq_result.selected_indices[rank]]),
                        "mean_base": float(acq_result.all_mean_base[acq_result.selected_indices[rank]]),
                        "std": float(acq_result.all_std[acq_result.selected_indices[rank]]),
                        "coupling_lambda": float(coupling.lambda_value) if coupling is not None else 0.0,
                        "coupling_mode": guidance.mode if guidance is not None else None,
                    },
                    llm_rationale=guidance_payload,
                )
                self._record_hv_snapshot(
                    phase="bo",
                    iteration=t + 1,
                    source="bo",
                    theta=theta,
                    feasible=bool(sim_result["feasible"]),
                    elapsed_s=elapsed_eval,
                    acq_value=float(acq_result.selected_scores[rank]),
                )
                logger.info(
                    "    -> feasible=%s obj=%s acq=%.6f (%.1fs)",
                    sim_result["feasible"],
                    np.round(sim_result["raw_objectives"], 6),
                    float(acq_result.selected_scores[rank]),
                    elapsed_eval,
                )
                n_new += 1

            iter_elapsed = time.perf_counter() - iter_start
            self.database.record_iteration_stats(
                extra={
                    "t": t,
                    "w_vec": w_vec.tolist(),
                    "n_new_evals": n_new,
                    "iter_time_s": round(iter_elapsed, 2),
                    "llm_guidance": self._previous_guidance,
                    "gp_llm_coupling": self._last_coupling_summary,
                }
            )
            logger.info(
                "Iteration %d complete: HV=%.6f |PF|=%d n=%d (%.1fs)",
                t,
                self.database.compute_hypervolume(),
                self.database.pareto_size,
                self.database.size,
                iter_elapsed,
            )

            if (t + 1) % int(self.cfg["checkpoint_every"]) == 0:
                self._save_checkpoint(t)

    def run(self) -> ObservationDB:
        self.setup()
        self.run_initialization()
        self.initialize_acquisition()
        self.run_optimization_loop()
        logger.info("Optimization finished: HV=%.6f", self.database.compute_hypervolume())
        return self.database

    def save_results(self, output_dir: str = "results") -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        self.database.save(str(output / "database.json"))
        self.database.save(str(output / "db_final.json"))

        pareto = [
            {
                "theta": obs.theta.tolist(),
                "objectives": obs.objectives.tolist(),
                "source": obs.source,
                "iteration": obs.iteration,
            }
            for obs in self.database.get_pareto_front()
        ]
        with open(output / "pareto_front.json", "w", encoding="utf-8") as f:
            json.dump(pareto, f, indent=2, ensure_ascii=False)

        summary = {
            "n_total": self.database.size,
            "n_feasible": self.database.n_feasible,
            "pareto_size": self.database.pareto_size,
            "hypervolume": self.database.compute_hypervolume(),
            "hypervolume_raw": self.database.compute_hypervolume_raw(),
            "warmstart_trace": self._warmstart_hv_trace,
            "hv_trace": self._hv_eval_trace,
            "last_guidance": self._previous_guidance,
            "last_gp_llm_coupling": self._last_coupling_summary,
            "config": self._jsonable_config(),
        }
        with open(output / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Results saved to %s", output)

    def _save_checkpoint(self, t: int) -> None:
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.database.save(str(ckpt_dir / f"db_t{t:04d}.json"))
        with open(ckpt_dir / f"af_t{t:04d}.json", "w", encoding="utf-8") as f:
            json.dump(self.af.save_state(), f, indent=2)
        with open(ckpt_dir / f"summary_t{t:04d}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "t": t,
                    "n_total": self.database.size,
                    "n_feasible": self.database.n_feasible,
                    "pareto_size": self.database.pareto_size,
                    "hypervolume": self.database.compute_hypervolume(),
                    "last_guidance": self._previous_guidance,
                    "last_gp_llm_coupling": self._last_coupling_summary,
                    "config": self._jsonable_config(),
                },
                f,
                indent=2,
            )

    def _update_dynamic_bounds(self) -> None:
        feasible = self.database.get_feasible()
        if not feasible:
            self._y_tilde_min = np.zeros(3, dtype=float)
            self._y_tilde_max = np.ones(3, dtype=float)
            return

        Y_raw = np.array([obs.objectives for obs in feasible], dtype=float)
        Y_tilde = log_transform_objectives(Y_raw)
        self._y_tilde_min, self._y_tilde_max = compute_dynamic_bounds(Y_tilde)

        global_min = np.array(
            [
                np.log10(self.database.ideal_point[0]),
                self.database.ideal_point[1],
                np.log10(self.database.ideal_point[2]),
            ],
            dtype=float,
        )
        global_max = np.array(
            [
                np.log10(self.database.ref_point[0]),
                self.database.ref_point[1],
                np.log10(self.database.ref_point[2]),
            ],
            dtype=float,
        )
        hist_range = self._y_tilde_max - self._y_tilde_min
        global_range = global_max - global_min

        for i in range(3):
            if hist_range[i] < 0.05 * global_range[i]:
                self._y_tilde_min[i] = global_min[i]
                self._y_tilde_max[i] = global_max[i]

    def _next_weight(self) -> np.ndarray:
        if not self._weight_order:
            order = self._rng.permutation(len(self._weight_set))
            self._weight_order = order.tolist()
        return self._weight_set[self._weight_order.pop()]

    def _compute_dynamic_ideal_point(self, Y_raw: np.ndarray) -> np.ndarray:
        Y_raw = np.atleast_2d(np.asarray(Y_raw, dtype=float))
        if Y_raw.size == 0:
            return np.asarray(self.database.ideal_point, dtype=float).copy()
        return Y_raw.min(axis=0)

    def _sobol_grid(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        n_points: int,
        seed: int,
    ) -> np.ndarray:
        lb = np.asarray(lb, dtype=float).ravel()
        ub = np.asarray(ub, dtype=float).ravel()
        n_points = max(1, int(n_points))
        span = ub - lb
        adj_ub = np.where(span <= 1e-12, lb + 1e-9, ub)
        m = int(np.ceil(np.log2(n_points)))
        sampler = qmc.Sobol(d=len(lb), scramble=True, seed=seed)
        sample = sampler.random_base2(m=m)[:n_points]
        scaled = qmc.scale(sample, lb, adj_ub)
        return np.clip(scaled, lb, ub)

    def _estimate_search_sigma(self) -> np.ndarray:
        lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        stagnation = int(self.database.get_stagnation_count())
        sigma_scale = 1.0 + 0.20 * min(stagnation, 3)
        return np.maximum((hi - lo) * 0.15 * sigma_scale, 1e-3)

    def _compute_uncertainty_hotspots(self, t: int) -> List[Dict[str, Any]]:
        n_probe = int(self.cfg.get("guidance_probe_size", 128))
        top_k = int(self.cfg.get("guidance_hotspots", 5))
        lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        probe = self._sobol_grid(lo, hi, n_probe, seed=int(self.cfg.get("w_sample_seed", 0) or 0) + 1000 + t)
        _, std = self.gp.predict(probe)
        order = np.argsort(std)[::-1][:top_k]
        return [
            {
                "theta": probe[idx].tolist(),
                "std": float(std[idx]),
            }
            for idx in order
        ]

    def _build_guidance_state(
        self,
        *,
        t: int,
        w_vec: np.ndarray,
        ideal_point_raw: np.ndarray,
        uncertainty_hotspots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "iteration": t + 1,
            "max_iterations": int(self.cfg["max_iterations"]),
            "w_vec": np.asarray(w_vec, dtype=float).tolist(),
            "theta_best": self.database.get_theta_best().tolist(),
            "f_min": float(self.database.get_f_min()),
            "mu": self.database.get_theta_best().tolist(),
            "sigma": self._estimate_search_sigma().tolist(),
            "stagnation_count": int(self.database.get_stagnation_count()),
            "database": self.database,
            "uncertainty_hotspots": uncertainty_hotspots,
            "previous_guidance": self._previous_guidance,
            "ideal_point": np.asarray(ideal_point_raw, dtype=float).tolist(),
        }

    def _build_gp_llm_coupling_from_guidance(
        self,
        guidance: IterationGuidance,
        t: int,
    ) -> Tuple[Any, np.ndarray]:
        if guidance.mode == "region":
            grid = self._sobol_grid(
                np.asarray(guidance.lb, dtype=float),
                np.asarray(guidance.ub, dtype=float),
                n_points=int(self.cfg.get("guidance_grid_size", 64)),
                seed=int(self.cfg.get("w_sample_seed", 0) or 0) + 2000 + t,
            )
            weights = np.full(grid.shape[0], 1.0 / max(grid.shape[0], 1), dtype=float)
        else:
            center = self._repair_theta(guidance.representative_point())
            sigma = np.maximum(
                self._estimate_search_sigma() * float(self.cfg.get("guidance_point_local_scale", 0.75)),
                np.array([0.08, 0.08, 0.04, 0.015, 0.015], dtype=float),
            )
            lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
            hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
            lb = np.maximum(lo, center - 2.0 * sigma)
            ub = np.minimum(hi, center + 2.0 * sigma)
            grid = self._sobol_grid(
                lb,
                ub,
                n_points=int(self.cfg.get("guidance_point_grid_size", 25)),
                seed=int(self.cfg.get("w_sample_seed", 0) or 0) + 3000 + t,
            )
            grid = np.vstack([center[None, :], grid])
            grid = np.vstack(self._deduplicate_points([self._repair_theta(row) for row in grid]))
            diff = (grid - center[np.newaxis, :]) / sigma[np.newaxis, :]
            weights = np.exp(-0.5 * np.sum(diff ** 2, axis=1))

        coupling = self.gp.build_preference_coupling(
            grid=grid,
            weights=weights,
            confidence=float(guidance.confidence),
            mode=guidance.mode,
        )
        return coupling, grid

    def _lhs_candidates(self, n: int, seed: int = 0) -> List[np.ndarray]:
        if n <= 0:
            return []
        lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        rng = np.random.default_rng(seed)
        samples = np.zeros((n, len(PARAM_KEYS)), dtype=float)
        intervals = np.linspace(0.0, 1.0, n + 1)
        for dim in range(len(PARAM_KEYS)):
            perm = rng.permutation(n)
            lower = intervals[perm]
            upper = intervals[perm + 1]
            samples[:, dim] = lower + rng.random(n) * (upper - lower)

        candidates = []
        for row in samples:
            theta = lo + row * (hi - lo)
            if theta[3] + theta[4] > DSOC_SUM_MAX:
                scale = (DSOC_SUM_MAX * 0.995) / (theta[3] + theta[4])
                theta[3] *= scale
                theta[4] *= scale
            candidates.append(np.clip(theta, lo, hi))
        return candidates

    @staticmethod
    def _deduplicate_tagged_points(
        tagged_points: List[Tuple[str, np.ndarray]]
    ) -> List[Tuple[str, np.ndarray]]:
        deduped: List[Tuple[str, np.ndarray]] = []
        seen = set()
        for source, theta in tagged_points:
            key = tuple(np.round(np.asarray(theta, dtype=float).ravel(), 6))
            if key in seen:
                continue
            seen.add(key)
            deduped.append((source, np.asarray(theta, dtype=float).ravel()))
        return deduped

    @staticmethod
    def _deduplicate_points(points: List[np.ndarray]) -> List[np.ndarray]:
        deduped: List[np.ndarray] = []
        seen = set()
        for theta in points:
            key = tuple(np.round(np.asarray(theta, dtype=float).ravel(), 6))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(np.asarray(theta, dtype=float).ravel())
        return deduped

    def _repair_theta(self, theta: np.ndarray) -> np.ndarray:
        x = np.asarray(theta, dtype=float).ravel().copy()
        lo = np.array([self.param_bounds[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([self.param_bounds[k][1] for k in PARAM_KEYS], dtype=float)
        x = np.clip(x, lo, hi)
        if x[3] + x[4] > DSOC_SUM_MAX:
            scale = (DSOC_SUM_MAX * 0.995) / (x[3] + x[4])
            x[3] *= scale
            x[4] *= scale
            x = np.clip(x, lo, hi)
        return x

    def _jsonable_config(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in self.cfg.items():
            if "api_key" in key.lower():
                result[key] = "<redacted>" if value else ""
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                result[key] = value
        return result

    def _record_hv_snapshot(
        self,
        *,
        phase: str,
        iteration: int,
        source: str,
        theta: np.ndarray,
        feasible: bool,
        elapsed_s: Optional[float] = None,
        acq_value: Optional[float] = None,
    ) -> None:
        if self.database is None:
            return

        snapshot = {
            "eval_index": self.database.size,
            "phase": phase,
            "iteration": int(iteration),
            "source": source,
            "theta": np.asarray(theta, dtype=float).ravel().tolist(),
            "feasible": bool(feasible),
            "hypervolume": self.database.compute_hypervolume(),
            "hypervolume_raw": self.database.compute_hypervolume_raw(),
            "pareto_size": self.database.pareto_size,
            "n_total": self.database.size,
            "n_feasible": self.database.n_feasible,
        }
        if elapsed_s is not None:
            snapshot["elapsed_s"] = float(elapsed_s)
        if acq_value is not None:
            snapshot["acq_value"] = float(acq_value)
        self._hv_eval_trace.append(snapshot)

    def get_hv_eval_trace(self) -> List[Dict[str, Any]]:
        return list(self._hv_eval_trace)
