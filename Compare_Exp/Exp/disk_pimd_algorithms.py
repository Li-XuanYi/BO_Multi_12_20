"""
disk_pimd_algorithms.py - Python implementation of DISK and PIMD algorithms
============================================================================

DISK (Dynamic Island Single-objective Kriging):
    Wang H, et al. "DISK: A New Kriging Surrogate for Expensive Multi-objective Optimization"
    - Dynamic island decomposition strategy
    - Kriging surrogate model
    - Multi-to-single objective conversion using Tchebycheff weights

PIMD (Pareto-based Infilling with Maximum Diversity):
    Wang H, et al. "A Maximal Diversity Approach to the Multi-objective Optimization"
    - Diversity-based infill strategy
    - Maximizes Pareto front diversity

Usage:
    from Compare_Exp.Exp.disk_pimd_algorithms import DISKOptimizer, PIMDOptimizer

    # DISK
    optimizer = DISKOptimizer(seed=8409, n_evals=50, wmax=60, alpha=5)
    optimizer.run()
    optimizer.save_results("output/disk/")

    # PIMD
    optimizer = PIMDOptimizer(seed=8409, n_evals=50, wmax=15, eta=5)
    optimizer.run()
    optimizer.save_results("output/pimd/")
"""

from __future__ import annotations

import json
import logging
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pybamm_simulator import PyBaMMSimulator
from DataBase.database import ObservationDB, make_observation_db
from utils.constants import DEFAULT_BOUNDS, REF_POINT, DSOC_SUM_MAX

logger = logging.getLogger(__name__)

# Decision variable bounds
_XL = np.array([DEFAULT_BOUNDS["I1"][0], DEFAULT_BOUNDS["I2"][0],
                DEFAULT_BOUNDS["I3"][0], DEFAULT_BOUNDS["dSOC1"][0],
                DEFAULT_BOUNDS["dSOC2"][0]])
_XU = np.array([DEFAULT_BOUNDS["I1"][1], DEFAULT_BOUNDS["I2"][1],
                DEFAULT_BOUNDS["I3"][1], DEFAULT_BOUNDS["dSOC1"][1],
                DEFAULT_BOUNDS["dSOC2"][1]])


class KrigingSurrogate:
    """Simple Kriging (Gaussian Process) surrogate model."""

    def __init__(self, theta: np.ndarray = None, noise: float = 1e-5):
        """
        Initialize Kriging model.

        Args:
            theta: Length scale parameters (n_dim,)
            noise: Noise variance
        """
        self.theta = theta
        self.noise = noise
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.n_samples = 0
        self.n_dim = 0
        self.K_inv: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.L: Optional[np.ndarray] = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Gaussian (RBF) kernel."""
        if self.theta is None:
            # Automatic relevance determination
            if self.X_train is not None:
                theta = np.ones(self.X_train.shape[1])
            else:
                theta = np.ones(X1.shape[1])
        else:
            theta = self.theta

        # Squared exponential kernel
        dist = cdist(X1 / theta, X2 / theta, metric='euclidean')
        return np.exp(-0.5 * dist ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KrigingSurrogate':
        """Fit the Kriging model."""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).ravel()
        self.n_samples = self.X_train.shape[0]
        self.n_dim = self.X_train.shape[1]

        # Normalize targets
        self.y_mean = np.mean(self.y_train)
        self.y_std = np.std(self.y_train) if np.std(self.y_train) > 0 else 1.0
        y_normalized = (self.y_train - self.y_mean) / self.y_std

        # Compute kernel matrix
        K = self._kernel(self.X_train, self.X_train)
        K += self.noise * np.eye(self.n_samples)

        # Cholesky decomposition for stability
        try:
            self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y_normalized))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            K_inv = np.linalg.pinv(K)
            self.alpha = K_inv @ y_normalized
            self.K_inv = K_inv

        return self

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict at new points."""
        X = np.atleast_2d(X)

        if self.X_train is None:
            raise ValueError("Model not fitted yet")

        # Compute kernel between test and train points
        K_star = self._kernel(X, self.X_train)

        # Predict mean
        y_pred = K_star @ self.alpha
        y_pred = y_pred * self.y_std + self.y_mean

        if not return_std:
            return y_pred, None

        # Predict variance
        K_star_star = self._kernel(X, X)

        if self.L is not None:
            v = np.linalg.solve(self.L, K_star.T)
            var = np.diag(K_star_star) - np.sum(v ** 2, axis=0)
        else:
            var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)

        var = np.maximum(var, 0)  # Ensure non-negative
        std = np.sqrt(var) * self.y_std

        return y_pred, std

    def expected_improvement(self, X: np.ndarray, f_min: float, xi: float = 0.01) -> np.ndarray:
        """Compute Expected Improvement acquisition function."""
        mu, sigma = self.predict(X, return_std=True)

        if sigma is None:
            return np.zeros(len(X))

        sigma = np.maximum(sigma, 1e-9)
        imp = f_min - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0

        return ei


class MultiObjectiveProblem:
    """Wrapper for multi-objective optimization problem."""

    def __init__(self, simulator: PyBaMMSimulator, param_set: str = "Chen2020"):
        self.simulator = simulator
        self.param_set = param_set
        self.n_var = 5
        self.n_obj = 3
        self.xl = _XL.copy()
        self.xu = _XU.copy()
        self.eval_count = 0

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Evaluate a single solution."""
        x = np.asarray(x).ravel()

        # Check constraint
        dsoc_sum = x[3] + x[4]
        if dsoc_sum > DSOC_SUM_MAX:
            # Infeasible
            objectives = REF_POINT.copy()
            feasible = False
        else:
            try:
                result = self.simulator.evaluate(x)
                feasible = bool(result.get("feasible", False))
                if feasible:
                    objectives = np.asarray(result["raw_objectives"], dtype=float)
                else:
                    objectives = REF_POINT.copy()
            except Exception as e:
                logger.warning(f"Simulation failed: {e}")
                objectives = REF_POINT.copy()
                feasible = False

        self.eval_count += 1
        return objectives, feasible

    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a batch of solutions."""
        n = X.shape[0]
        F = np.zeros((n, self.n_obj))
        feasible = np.zeros(n, dtype=bool)

        for i in range(n):
            F[i], feasible[i] = self.evaluate(X[i])

        return F, feasible


class BaseMOEAOptimizer(ABC):
    """Base class for multi-objective evolutionary algorithms."""

    ALGORITHM_NAME: str = "base"

    def __init__(
        self,
        seed: int = 0,
        n_evals: int = 50,
        population_size: int = 20,
        param_set: str = "Chen2020",
    ):
        self.seed = seed
        self.n_evals = n_evals
        self.population_size = min(population_size, n_evals)
        self.param_set = param_set

        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)

        self.simulator = PyBaMMSimulator(param_set=param_set)
        self.problem = MultiObjectiveProblem(self.simulator, param_set)
        self.db = make_observation_db()

        # Population
        self.population_X: Optional[np.ndarray] = None
        self.population_F: Optional[np.ndarray] = None
        self.population_feasible: Optional[np.ndarray] = None

        self.hv_trace: List[Dict] = []
        self._t0 = datetime.now()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the algorithm."""
        pass

    @abstractmethod
    def _iterate(self) -> None:
        """Perform one iteration."""
        pass

    def _update_database(self, X: np.ndarray, F: np.ndarray, feasible: np.ndarray, source: str) -> None:
        """Update observation database."""
        for i in range(len(X)):
            self.db.add_from_simulator(
                theta=X[i],
                result={
                    "raw_objectives": F[i],
                    "feasible": bool(feasible[i]),
                    "violation": None,
                },
                source=source,
                iteration=self.problem.eval_count,
            )

    def _record_hv(self, X: np.ndarray, F: np.ndarray, feasible: np.ndarray, phase: str) -> None:
        """Record hypervolume trace.

        For batch updates, we reconstruct prefix HV values over the newly-added
        observations so init and population batches share the same per-eval
        semantics as the single-sample BO traces.
        """
        batch_size = len(X)
        end_idx = self.db.size
        start_idx = max(0, end_idx - batch_size)
        observations = self.db.get_all()
        prefix_db = make_observation_db(param_bounds=self.db.param_bounds, param_set=self.param_set)

        for obs in observations[:start_idx]:
            prefix_db.add_observation(
                theta=obs.theta,
                objectives=obs.objectives,
                feasible=obs.feasible,
                violation=obs.violation,
                source=obs.source,
                iteration=obs.iteration,
                acq_value=obs.acq_value,
                acq_type=obs.acq_type,
                gp_pred=obs.gp_pred,
                llm_rationale=obs.llm_rationale,
                details=obs.details,
            )

        for i, obs in enumerate(observations[start_idx:end_idx]):
            prefix_db.add_observation(
                theta=obs.theta,
                objectives=obs.objectives,
                feasible=obs.feasible,
                violation=obs.violation,
                source=obs.source,
                iteration=obs.iteration,
                acq_value=obs.acq_value,
                acq_type=obs.acq_type,
                gp_pred=obs.gp_pred,
                llm_rationale=obs.llm_rationale,
                details=obs.details,
            )
            display_hv = prefix_db.compute_hypervolume()
            raw_hv = prefix_db.compute_hypervolume_raw()
            canonical_hv = raw_hv / prefix_db.hv_max if prefix_db.hv_max > 1e-12 else 0.0
            self.hv_trace.append({
                "eval_index": start_idx + i + 1,
                "phase": phase,
                "iteration": int(obs.iteration),
                "source": self.ALGORITHM_NAME,
                "theta": obs.theta.tolist(),
                "feasible": bool(obs.feasible),
                "hypervolume": display_hv,
                "display_hv": display_hv,
                "canonical_hv": canonical_hv,
                "hypervolume_raw": raw_hv,
                "pareto_size": prefix_db.pareto_size,
                "n_total": prefix_db.size,
                "n_feasible": prefix_db.n_feasible,
                "elapsed_s": (datetime.now() - self._t0).total_seconds(),
            })

    def run(self) -> ObservationDB:
        """Run the optimization."""
        logger.info(f"Starting {self.ALGORITHM_NAME} with seed {self.seed}")
        logger.info(f"Total evaluations: {self.n_evals}")
        logger.info(f"Population size: {self.population_size}")

        # Initialize
        self._initialize()
        self._update_database(self.population_X, self.population_F, self.population_feasible, f"{self.ALGORITHM_NAME}_init")
        self._record_hv(self.population_X, self.population_F, self.population_feasible, "init")

        # Optimization loop
        while self.problem.eval_count < self.n_evals:
            self._iterate()
            self._update_database(
                self.population_X[-self.population_size:],
                self.population_F[-self.population_size:],
                self.population_feasible[-self.population_size:],
                self.ALGORITHM_NAME,
            )
            self._record_hv(
                self.population_X[-self.population_size:],
                self.population_F[-self.population_size:],
                self.population_feasible[-self.population_size:],
                "opt",
            )

            if self.problem.eval_count % 10 == 0:
                hv = self.db.compute_hypervolume()
                logger.info(f"Eval {self.problem.eval_count}/{self.n_evals}, HV={hv:.4f}")

        logger.info(f"{self.ALGORITHM_NAME} completed: {self.db.size} evals, HV={self.db.compute_hypervolume():.4f}")
        return self.db

    def save_results(self, output_dir: str) -> Dict[str, Any]:
        """Save experiment results."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        display_hv = self.db.compute_hypervolume()
        raw_hv = self.db.compute_hypervolume_raw()
        canonical_hv = raw_hv / self.db.hv_max if self.db.hv_max > 1e-12 else 0.0

        # Build best per objective
        best_per_obj = {}
        from DataBase.database import OBJECTIVE_NAMES
        best = self.db.get_best_per_objective()
        for name in OBJECTIVE_NAMES:
            if name in best:
                obs = best[name]
                idx = list(OBJECTIVE_NAMES).index(name)
                best_per_obj[name] = {
                    "value": float(obs.objectives[idx]),
                    "theta": obs.theta.tolist(),
                }

        summary = {
            "algorithm": self.ALGORITHM_NAME,
            "seed": self.seed,
            "n_evals": self.n_evals,
            "population_size": self.population_size,
            "param_set": self.param_set,
            "n_total": self.db.size,
            "n_feasible": self.db.n_feasible,
            "pareto_size": self.db.pareto_size,
            "hypervolume": display_hv,
            "display_hv": display_hv,
            "canonical_hv": canonical_hv,
            "hypervolume_raw": raw_hv,
            "hv_trace": self.hv_trace,
            "best_per_objective": best_per_obj,
            "timestamp": datetime.now().isoformat(),
        }

        # Save files
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.db.save(str(out / "database.json"))

        # Save Pareto front
        pareto = self.db.get_pareto_front()
        pf_data = [{"theta": o.theta.tolist(), "objectives": o.objectives.tolist()} for o in pareto]
        with open(out / "pareto_front.json", "w") as f:
            json.dump({"points": pf_data}, f, indent=2)

        logger.info(f"Results saved to {out}")
        return summary


class DISKOptimizer(BaseMOEAOptimizer):
    """
    DISK: Dynamic Island Single-objective Kriging

    References:
        Wang H, et al. "DISK: A New Kriging Surrogate for Expensive Multi-objective Optimization"
    """

    ALGORITHM_NAME = "disk"

    def __init__(
        self,
        seed: int = 0,
        n_evals: int = 50,
        population_size: int = 20,
        param_set: str = "Chen2020",
        wmax: int = 60,
        alpha: int = 5,
    ):
        super().__init__(seed, n_evals, population_size, param_set)
        self.wmax = wmax
        self.alpha = alpha

        # DISK-specific
        self.surrogates: List[KrigingSurrogate] = []
        self.weight_vectors: List[np.ndarray] = []
        self.n_islands = 0

    def _initialize(self) -> None:
        """Initialize with random population."""
        # Generate random initial population
        self.population_X = self.rng.uniform(_XL, _XU, (self.population_size, 5))
        self.population_F, self.population_feasible = self.problem.evaluate_batch(self.population_X)

        # Initialize weight vectors (simplified)
        self.n_islands = min(self.wmax, self.population_size)
        self.weight_vectors = self._generate_weights(self.n_islands)

        # Initialize surrogates
        self.surrogates = [KrigingSurrogate() for _ in range(self.n_islands)]

    def _generate_weights(self, n_weights: int) -> List[np.ndarray]:
        """Generate Tchebycheff weight vectors."""
        weights = []
        for i in range(n_weights):
            w = self.rng.dirichlet(np.ones(3))
            weights.append(w)
        return weights

    def _scalarize(self, F: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Tchebycheff scalarization."""
        # Normalize objectives
        f_max = np.max(F, axis=0) if len(F) > 0 else np.ones(3)
        f_min = np.min(F, axis=0) if len(F) > 0 else np.zeros(3)
        f_range = f_max - f_min
        f_range[f_range < 1e-6] = 1.0

        F_norm = (F - f_min) / f_range

        # Tchebycheff with augmentation
        eta = 0.05
        weighted = weight * F_norm
        tch = np.max(weighted, axis=1) + eta * np.sum(weighted, axis=1)
        return tch

    def _iterate(self) -> None:
        """One iteration of DISK."""
        # Update surrogates with current data
        for i, surrogate in enumerate(self.surrogates):
            if len(self.population_X) > 0:
                f_scalar = self._scalarize(self.population_F, self.weight_vectors[i])
                try:
                    surrogate.fit(self.population_X, f_scalar)
                except Exception as e:
                    logger.warning(f"Surrogate fitting failed: {e}")

        # Generate new candidates using EI
        n_candidates = min(self.population_size, self.n_evals - self.problem.eval_count)
        if n_candidates <= 0:
            return

        new_X = []
        for i in range(min(n_candidates, self.n_islands)):
            # Use EI to find promising points
            f_scalar = self._scalarize(self.population_F, self.weight_vectors[i])
            f_min = np.min(f_scalar) if len(f_scalar) > 0 else 0.0

            # Optimize EI
            best_x = self._optimize_ei(self.surrogates[i], f_min)
            new_X.append(best_x)

        # Fill remaining with random if needed
        while len(new_X) < n_candidates:
            new_X.append(self.rng.uniform(_XL, _XU))

        new_X = np.array(new_X)
        new_F, new_feasible = self.problem.evaluate_batch(new_X)

        # Update population
        self.population_X = np.vstack([self.population_X, new_X])
        self.population_F = np.vstack([self.population_F, new_F])
        self.population_feasible = np.concatenate([self.population_feasible, new_feasible])

    def _optimize_ei(self, surrogate: KrigingSurrogate, f_min: float) -> np.ndarray:
        """Optimize Expected Improvement."""
        # Multi-start optimization
        n_starts = 5
        best_x = None
        best_ei = -np.inf

        for _ in range(n_starts):
            x0 = self.rng.uniform(_XL, _XU)

            try:
                result = minimize(
                    lambda x: -surrogate.expected_improvement(x.reshape(1, -1), f_min)[0],
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(_XL, _XU)),
                    options={'maxiter': 100},
                )

                if result.success:
                    ei = -result.fun
                    if ei > best_ei:
                        best_ei = ei
                        best_x = result.x
            except Exception:
                continue

        if best_x is None:
            best_x = self.rng.uniform(_XL, _XU)

        return best_x


class PIMDOptimizer(BaseMOEAOptimizer):
    """
    PIMD: Pareto-based Infilling with Maximum Diversity

    References:
        Wang H, et al. "A Maximal Diversity Approach to the Multi-objective Optimization"
    """

    ALGORITHM_NAME = "pimd"

    def __init__(
        self,
        seed: int = 0,
        n_evals: int = 50,
        population_size: int = 20,
        param_set: str = "Chen2020",
        wmax: int = 15,
        eta: int = 5,
    ):
        super().__init__(seed, n_evals, population_size, param_set)
        self.wmax = wmax
        self.eta = eta

        # PIMD-specific
        self.surrogate: Optional[KrigingSurrogate] = None
        self.weight_vectors: List[np.ndarray] = []

    def _initialize(self) -> None:
        """Initialize with random population."""
        # Generate random initial population
        self.population_X = self.rng.uniform(_XL, _XU, (self.population_size, 5))
        self.population_F, self.population_feasible = self.problem.evaluate_batch(self.population_X)

        # Initialize surrogate
        self.surrogate = KrigingSurrogate()

        # Generate weight vectors
        self.weight_vectors = self._generate_weights(self.wmax)

    def _generate_weights(self, n_weights: int) -> List[np.ndarray]:
        """Generate uniform weight vectors."""
        weights = []
        for i in range(n_weights):
            w = self.rng.dirichlet(np.ones(3))
            weights.append(w)
        return weights

    def _scalarize(self, F: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Tchebycheff scalarization."""
        f_max = np.max(F, axis=0) if len(F) > 0 else np.ones(3)
        f_min = np.min(F, axis=0) if len(F) > 0 else np.zeros(3)
        f_range = f_max - f_min
        f_range[f_range < 1e-6] = 1.0

        F_norm = (F - f_min) / f_range

        eta = 0.05
        weighted = weight * F_norm
        tch = np.max(weighted, axis=1) + eta * np.sum(weighted, axis=1)
        return tch

    def _compute_diversity(self, X: np.ndarray) -> np.ndarray:
        """Compute diversity metric for candidates."""
        if len(self.population_X) == 0:
            return np.ones(len(X))

        # Distance to nearest neighbor in population
        distances = cdist(X, self.population_X)
        min_distances = np.min(distances, axis=1)
        return min_distances

    def _iterate(self) -> None:
        """One iteration of PIMD."""
        n_candidates = min(self.population_size, self.n_evals - self.problem.eval_count)
        if n_candidates <= 0:
            return

        # Build scalarized surrogate for each weight
        all_candidates_X = []
        all_candidates_score = []

        for weight in self.weight_vectors:
            f_scalar = self._scalarize(self.population_F, weight)

            # Fit surrogate
            try:
                self.surrogate.fit(self.population_X, f_scalar)
            except Exception as e:
                logger.warning(f"Surrogate fitting failed: {e}")
                continue

            # Optimize for this weight
            for _ in range(max(1, n_candidates // len(self.weight_vectors))):
                x0 = self.rng.uniform(_XL, _XU)

                try:
                    result = minimize(
                        lambda x: self.surrogate.predict(x.reshape(1, -1), return_std=False)[0],
                        x0,
                        method='L-BFGS-B',
                        bounds=list(zip(_XL, _XU)),
                        options={'maxiter': 100},
                    )

                    if result.success:
                        all_candidates_X.append(result.x)
                        # Score: lower scalarized value is better
                        all_candidates_score.append(result.fun)
                except Exception:
                    continue

        if len(all_candidates_X) == 0:
            # Fallback to random
            new_X = self.rng.uniform(_XL, _XU, (n_candidates, 5))
        else:
            all_candidates_X = np.array(all_candidates_X)
            all_candidates_score = np.array(all_candidates_score)

            # Compute diversity
            diversity = self._compute_diversity(all_candidates_X)

            # PIMD selection: balance between convergence and diversity
            # Normalize scores and diversity
            score_norm = (all_candidates_score - np.min(all_candidates_score))
            score_norm = score_norm / (np.max(score_norm) + 1e-6)

            div_norm = diversity / (np.max(diversity) + 1e-6)

            # Combined score (lower is better)
            # Weight factor: wmax and eta parameters control balance
            combined_score = score_norm - self.eta * div_norm

            # Select top n_candidates
            selected_indices = np.argsort(combined_score)[:n_candidates]
            new_X = all_candidates_X[selected_indices]

        # Evaluate new candidates
        new_F, new_feasible = self.problem.evaluate_batch(new_X)

        # Update population
        self.population_X = np.vstack([self.population_X, new_X])
        self.population_F = np.vstack([self.population_F, new_F])
        self.population_feasible = np.concatenate([self.population_feasible, new_feasible])
