"""
nsga2_runner.py — NSGA-II baseline runner for paper comparison
===============================================================
Wraps pymoo's NSGA-II to optimize the same 5D battery charging problem.
Outputs summary.json / database.json / pareto_front.json compatible with
the existing LLMBO experiment pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pybamm_simulator import PyBaMMSimulator
from DataBase.database import ObservationDB, make_observation_db
from utils.constants import DEFAULT_BOUNDS, REF_POINT, DSOC_SUM_MAX

logger = logging.getLogger(__name__)

# Bounds as ordered arrays
_XL = np.array([DEFAULT_BOUNDS["I1"][0], DEFAULT_BOUNDS["I2"][0],
                 DEFAULT_BOUNDS["I3"][0], DEFAULT_BOUNDS["dSOC1"][0],
                 DEFAULT_BOUNDS["dSOC2"][0]])
_XU = np.array([DEFAULT_BOUNDS["I1"][1], DEFAULT_BOUNDS["I2"][1],
                 DEFAULT_BOUNDS["I3"][1], DEFAULT_BOUNDS["dSOC1"][1],
                 DEFAULT_BOUNDS["dSOC2"][1]])


class BatteryChargingProblem(Problem):
    """pymoo Problem wrapping PyBaMMSimulator."""

    def __init__(self, simulator: PyBaMMSimulator):
        super().__init__(
            n_var=5, n_obj=3, n_ieq_constr=1,
            xl=_XL.copy(), xu=_XU.copy(),
        )
        self.simulator = simulator

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        F = np.zeros((n, 3))
        G = np.zeros((n, 1))
        for i in range(n):
            theta = X[i]
            res = self.simulator.evaluate(theta)
            if res["feasible"]:
                F[i] = res["raw_objectives"]
            else:
                F[i] = REF_POINT.copy()
            # Constraint: dSOC1 + dSOC2 <= DSOC_SUM_MAX
            G[i, 0] = (theta[3] + theta[4]) - DSOC_SUM_MAX
        out["F"] = F
        out["G"] = G


class HVTraceCallback(Callback):
    """Records HV after each evaluation batch into the ObservationDB."""

    def __init__(self, db: ObservationDB, simulator: PyBaMMSimulator):
        super().__init__()
        self.db = db
        self.simulator = simulator
        self._eval_count = 0
        self._t0 = time.time()
        self.hv_trace: List[Dict] = []

    def _record(self, X, F, G, generation: int):
        for i in range(X.shape[0]):
            theta = X[i].copy()
            feasible = bool(G[i, 0] <= 0) and bool(F[i, 0] < REF_POINT[0])
            objectives = F[i].copy()

            source = "nsga2_init" if generation == 0 else "nsga2"
            self.db.add_from_simulator(
                theta=theta,
                result={"raw_objectives": objectives, "feasible": feasible,
                         "violation": None},
                source=source,
                iteration=generation,
            )
            self._eval_count += 1

            display_hv = self.db.compute_hypervolume()
            raw_hv = self.db.compute_hypervolume_raw()
            canonical_hv = raw_hv / self.db.hv_max if self.db.hv_max > 1e-12 else 0.0
            self.hv_trace.append({
                "eval_index": self._eval_count,
                "phase": "init" if generation == 0 else "ga",
                "iteration": generation,
                "source": source,
                "theta": theta.tolist(),
                "feasible": feasible,
                "hypervolume": display_hv,
                "display_hv": display_hv,
                "canonical_hv": canonical_hv,
                "hypervolume_raw": raw_hv,
                "pareto_size": self.db.pareto_size,
                "n_total": self.db.size,
                "n_feasible": self.db.n_feasible,
                "elapsed_s": round(time.time() - self._t0, 2),
            })

    def notify(self, algorithm, **kwargs):
        gen = algorithm.n_gen - 1  # gen 0 = initial population
        pop = algorithm.pop
        if pop is None:
            return
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")
        if X is None or F is None:
            return
        if G is None:
            G = np.zeros((X.shape[0], 1))
        self._record(X, F, G, generation=gen)


class NSGA2Runner:
    """Run NSGA-II on the battery charging problem with output compatible to LLMBO experiments."""

    def __init__(self, seed: int = 0, n_evals: int = 56, pop_size: int = 20):
        self.seed = seed
        self.n_evals = n_evals
        self.pop_size = min(pop_size, n_evals)
        self.simulator = PyBaMMSimulator()
        self.db = make_observation_db()
        self.hv_trace: List[Dict] = []
        self._result_summary: Optional[Dict] = None

    def run(self) -> ObservationDB:
        rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)

        problem = BatteryChargingProblem(self.simulator)
        callback = HVTraceCallback(self.db, self.simulator)

        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=LHS(),
            crossover=SBX(prob=0.9, eta=20),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        termination = get_termination("n_eval", self.n_evals)

        logger.info("NSGA-II: seed=%d, pop=%d, max_evals=%d", self.seed, self.pop_size, self.n_evals)

        pymoo_minimize(
            problem, algorithm, termination,
            seed=self.seed,
            callback=callback,
            verbose=False,
        )

        self.hv_trace = callback.hv_trace
        logger.info("NSGA-II done: %d evals, %d feasible, %d Pareto, HV=%.4f",
                     self.db.size, self.db.n_feasible, self.db.pareto_size,
                     self.db.compute_hypervolume())
        return self.db

    def save_results(self, output_dir: str) -> Dict:
        os.makedirs(output_dir, exist_ok=True)

        display_hv = self.db.compute_hypervolume()
        raw_hv = self.db.compute_hypervolume_raw()
        canonical_hv = raw_hv / self.db.hv_max if self.db.hv_max > 1e-12 else 0.0

        summary = {
            "algorithm": "nsga2",
            "seed": self.seed,
            "n_evals": self.n_evals,
            "pop_size": self.pop_size,
            "n_total": self.db.size,
            "n_feasible": self.db.n_feasible,
            "pareto_size": self.db.pareto_size,
            "hypervolume": display_hv,
            "display_hv": display_hv,
            "canonical_hv": canonical_hv,
            "hypervolume_raw": raw_hv,
            "hv_trace": self.hv_trace,
            "best_per_objective": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Best per objective
        best = self.db.get_best_per_objective()
        from DataBase.database import OBJECTIVE_NAMES
        for name in OBJECTIVE_NAMES:
            if name in best:
                o = best[name]
                idx = list(OBJECTIVE_NAMES).index(name)
                summary["best_per_objective"][name] = {
                    "value": float(o.objectives[idx]),
                    "theta": o.theta.tolist() if hasattr(o.theta, 'tolist') else list(o.theta),
                }

        # Save summary.json
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save database.json
        self.db.save(os.path.join(output_dir, "database.json"))

        # Save pareto_front.json
        pareto = self.db.get_pareto_front()
        pf_data = []
        for o in pareto:
            pf_data.append({
                "theta": o.theta.tolist() if hasattr(o.theta, 'tolist') else list(o.theta),
                "objectives": o.objectives.tolist() if hasattr(o.objectives, 'tolist') else list(o.objectives),
            })
        with open(os.path.join(output_dir, "pareto_front.json"), "w") as f:
            json.dump(pf_data, f, indent=2)

        self._result_summary = summary
        logger.info("Results saved to %s", output_dir)
        return summary
