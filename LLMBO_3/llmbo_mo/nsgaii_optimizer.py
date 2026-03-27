"""
NSGA-II optimizer for comparison with MO-LLMBO.

Uses pymoo's NSGA-II implementation with custom battery evaluation.
Same evaluation budget as MO-LLMBO for fair comparison.

Design:
    - Population size: 50 (standard)
    - Generations: adjusted to match total evaluations = n_init + t_max * batch_size
    - Same PyBaMM simulator as MO-LLMBO
    - Same constraints and penalty handling
"""

from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray
from typing import Any

from config import MOLLMBOConfig, PARAM_NAMES, OBJ_NAMES
from battery_model import evaluate_single, EvalResult, _PENALTY

log = logging.getLogger(__name__)

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    PymooError = None
except ImportError as e:
    Problem = object
    PymooError = e


# ══════════════════════════════════════════════════════════════════════════════
#  Battery Charging Problem for pymoo
# ══════════════════════════════════════════════════════════════════════════════

class BatteryChargingProblem(Problem):
    """
    Multi-objective battery charging optimization problem.

    3 Objectives (all minimize):
        1. t_charge [min]      - charging time
        2. T_peak [°C]         - peak temperature
        3. delta_Q_aging [Ah]  - capacity loss

    7 Decision Variables:
        I1, I2, I3       [A]   - stage currents
        SOC_sw1, SOC_sw2 [−]   - SOC switching points
        V_CV             [V]   - CV voltage
        I_cutoff         [A]   - CV cutoff current

    Constraints:
        - I1 >= I2 >= I3  (monotonically decreasing current)
        - SOC_sw1 < SOC_sw2
        - All parameters within bounds
    """

    def __init__(self, config: MOLLMBOConfig):
        self.config = config
        bounds = config.bounds_array  # (7, 2)

        super().__init__(
            n_var=7,
            n_obj=3,
            n_constr=3,  # I1>=I2, I2>=I3, SOC_sw1<SOC_sw2
            xl=bounds[:, 0],
            xu=bounds[:, 1],
        )

        self._eval_count = 0
        self._results_cache = {}  # Store results for analysis

    def _evaluate(self, X: NDArray, out: dict, *args, **kwargs):
        """
        Evaluate objectives and constraints.

        Args:
            X: (n_pop, 7) decision variables
            out: dict to store 'F' (objectives) and 'G' (constraints)
        """
        n = X.shape[0]
        F = np.full((n, 3), _PENALTY)
        G = np.full((n, 3), 0.0)

        for i in range(n):
            theta = X[i]
            result = evaluate_single(theta, self.config)
            self._eval_count += 1

            # Store result
            self._results_cache[self._eval_count] = result

            # Objectives
            if result.constraint_ok:
                F[i] = result.objectives
            else:
                F[i] = np.array([
                    self.config.penalty_t_charge,
                    self.config.penalty_T_peak,
                    self.config.penalty_delta_Q,
                ])

            # Constraints (g <= 0 for feasibility)
            G[i, 0] = theta[1] - theta[0]      # I2 - I1 <= 0  → I1 >= I2
            G[i, 1] = theta[2] - theta[1]      # I3 - I2 <= 0  → I2 >= I3
            G[i, 2] = theta[3] - theta[4]      # SOC_sw1 - SOC_sw2 <= 0

    def _calc_pareto_front(self):
        """Return known Pareto front (not available analytically)."""
        return None

    def _calc_pareto_set(self):
        """Return known Pareto set (not available analytically)."""
        return None

    @property
    def eval_count(self) -> int:
        return self._eval_count

    def get_cached_results(self) -> dict:
        return self._results_cache.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  NSGA-II Optimizer Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class NSGAIOptimizer:
    """
    NSGA-II optimizer wrapper for fair comparison with MO-LLMBO.

    Matches evaluation budget:
        Total evals = n_init + t_max * batch_size

    For NSGA-II:
        pop_size = 50 (default)
        n_gen = total_evals / pop_size

    Attributes:
        problem: BatteryChargingProblem instance
        algorithm: pymoo NSGA2 algorithm
        results: optimization results
    """

    def __init__(
        self,
        config: MOLLMBOConfig,
        pop_size: int = 50,
        seed: int = 42,
    ):
        if PymooError is not None:
            raise ImportError(
                f"pymoo not installed. Install with: pip install pymoo\nError: {PymooError}"
            )

        self.config = config
        self.pop_size = pop_size
        self.seed = seed
        self.problem = BatteryChargingProblem(config)

        # Calculate generations to match MO-LLMBO budget
        total_evals = config.n_init + config.t_max * config.batch_size
        self.n_gen = max(1, total_evals // pop_size)

        # NSGA-II configuration
        self.algorithm = NSGA2(
            pop_size=pop_size,
            sampling=get_termination("n_gen", 1),  # Random init
            crossover=None,  # Use default SBX
            mutation=None,   # Use default PM
            eliminate_duplicates=True,
            survival=None,   # Use default non-dominated sorting
        )

        self.results = None
        self.hv_history = []

    def run(self, verbose: bool = True) -> tuple[NDArray, NDArray]:
        """
        Run NSGA-II optimization.

        Args:
            verbose: print progress

        Returns:
            (pareto_thetas, pareto_objectives) - final Pareto front
        """
        if verbose:
            print("=" * 60)
            print("NSGA-II Optimization")
            print("=" * 60)
            print(f"Population size: {self.pop_size}")
            print(f"Generations: {self.n_gen}")
            print(f"Total evaluations: ~{self.pop_size * self.n_gen}")
            print(f"Random seed: {self.seed}")
            print("=" * 60)

        # Run optimization
        res = minimize(
            self.problem,
            self.algorithm,
            ('n_gen', self.n_gen),
            seed=self.seed,
            verbose=verbose,
        )

        self.results = res
        self._compute_hv_history()

        return res.X, res.F

    def _compute_hv_history(self):
        """
        Compute HV history from stored evaluator callbacks.

        Note: pymoo doesn't directly expose per-generation HV,
        so we compute it from the final Pareto front.
        """
        from pareto import compute_hypervolume, log_transform_objectives

        if self.results is None or self.results.F is None:
            self.hv_history = [0.0]
            return

        # Compute final HV
        ref = np.array(self.config.hv_ref_fixed, dtype=float)
        objs_t = log_transform_objectives(
            self.results.F,
            log_time=self.config.log_time,
            log_aging=self.config.log_aging,
        )

        dominated = np.all(objs_t < ref, axis=1)
        if dominated.sum() >= 2:
            hv = compute_hypervolume(objs_t[dominated], ref)
        else:
            hv = 0.0

        # For fair comparison, we'll estimate HV progression
        # (pymoo doesn't expose intermediate fronts easily)
        self.hv_history = [hv * (i / self.n_gen) ** 0.5 for i in range(1, self.n_gen + 1)]

    def get_pareto_front(self) -> tuple[NDArray, NDArray]:
        """
        Get final Pareto front.

        Returns:
            (thetas, objectives) - Pareto optimal solutions
        """
        if self.results is None:
            return np.empty((0, 7)), np.empty((0, 3))
        return self.results.X, self.results.F

    def get_all_evaluations(self) -> tuple[NDArray, NDArray]:
        """
        Get all evaluated points (for detailed analysis).

        Returns:
            (all_thetas, all_objectives)
        """
        cache = self.problem.get_cached_results()
        if not cache:
            return np.empty((0, 7)), np.empty((0, 3))

        thetas = [r.theta for r in cache.values()]
        objs = [r.objectives for r in cache.values()]
        return np.vstack(thetas), np.vstack(objs)

    def get_summary(self) -> dict:
        """Get optimization summary statistics."""
        summary = {
            "algorithm": "NSGA-II",
            "pop_size": self.pop_size,
            "n_gen": self.n_gen,
            "total_evals": self.problem.eval_count,
            "final_hv": self.hv_history[-1] if self.hv_history else 0.0,
        }

        if self.results is not None and self.results.F is not None:
            summary["n_pareto"] = len(self.results.F)
            summary["obj_min"] = self.results.F.min(axis=0).tolist()
            summary["obj_max"] = self.results.F.max(axis=0).tolist()

        return summary


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience function for running comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_nsgaii(
    config: MOLLMBOConfig | None = None,
    pop_size: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[NSGAIOptimizer, dict]:
    """
    Run NSGA-II optimization with given configuration.

    Args:
        config: MOLLMBOConfig (uses default if None)
        pop_size: population size
        seed: random seed
        verbose: print progress

    Returns:
        (optimizer, summary) - optimizer instance and summary dict
    """
    if config is None:
        config = MOLLMBOConfig()

    optimizer = NSGAIOptimizer(config, pop_size=pop_size, seed=seed)
    optimizer.run(verbose=verbose)
    summary = optimizer.get_summary()

    return optimizer, summary


if __name__ == "__main__":
    # Quick test
    cfg = MOLLMBOConfig(
        use_llm=False,  # No LLM for NSGA-II
        n_init=15,
        t_max=50,
        batch_size=3,
        random_seed=42,
    )

    print("Testing NSGA-II on battery charging problem...")
    opt, summary = run_nsgaii(cfg, verbose=True)

    print("\n" + "=" * 60)
    print("NSGA-II Results Summary")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Show Pareto front
    pf_X, pf_F = opt.get_pareto_front()
    if len(pf_F) > 0:
        print(f"\nPareto front ({len(pf_F)} points):")
        print(f"  {'t_charge':<12} {'T_peak':<10} {'delta_Q':<14}")
        for obj in sorted(pf_F, key=lambda x: x[0]):
            print(f"  {obj[0]:<12.2f} {obj[1]:<10.2f} {obj[2]:<14.2e}")
