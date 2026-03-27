"""PyBaMM simulation wrapper for LG INR21700-M50.

Single evaluation:  evaluate_single(theta, config)  → EvalResult
Batch evaluation:   evaluate_batch(thetas, config)   → list[EvalResult]

θ layout (matches PARAM_NAMES in config.py):
    [I1, I2, I3, SOC_sw1, SOC_sw2, V_CV, I_cutoff]  shape (7,)

Experiment structure (PyBaMM v25 convention):
    cycle 0 : pre-discharge to SOC_start  (constant, not part of θ)
    cycle 1 : rest 1 min
    cycle 2 : CC stage 1 at I1
    cycle 3 : CC stage 2 at I2
    cycle 4 : CC stage 3 at I3
    cycle 5 : CV hold at V_CV until I_cutoff

In PyBaMM v25 each Experiment step maps to one entry in sol.cycles.
Charging phase = sol.cycles[CHARGE_CYCLE_START:]  (CHARGE_CYCLE_START = 2)
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# Index of the first charging cycle in sol.cycles (0=discharge, 1=rest)
_CHARGE_CYCLE_START = 2

# Penalty returned on simulation failure or hard-constraint violation
_PENALTY = 999.0


@dataclass
class EvalResult:
    """Output of a single PyBaMM evaluation.

    Objectives (all to be minimised):
        t_charge       : total charging time [min]
        T_peak         : max volume-averaged temperature during charging [°C]
        delta_Q_aging  : SEI-driven capacity loss during charging [Ah]

    Constraint flags (hard):
        V_violated     : terminal voltage exceeded 4.2 V at any point
        T_violated     : surface temperature exceeded T_max [°C]
        plating_risk   : min anode surface potential ≤ 0 V (Li plating)
        constraint_ok  : all hard constraints satisfied

    Meta:
        theta          : (7,) parameter array
        monotone_ok    : I1 ≥ I2 ≥ I3  (soft, enforced via LLM prompt)
        failed         : simulation raised an exception
        error_msg      : exception string if failed
    """

    theta: NDArray
    t_charge: float = _PENALTY
    T_peak: float = _PENALTY
    delta_Q_aging: float = _PENALTY
    V_violated: bool = False
    T_violated: bool = False
    plating_risk: bool = False
    monotone_ok: bool = True
    failed: bool = False
    error_msg: str = ""

    @property
    def constraint_ok(self) -> bool:
        return not (self.failed or self.V_violated or self.T_violated or self.plating_risk)

    @property
    def objectives(self) -> NDArray:
        """shape (3,) = [t_charge, T_peak, delta_Q_aging]"""
        return np.array([self.t_charge, self.T_peak, self.delta_Q_aging])


# ── Experiment construction ────────────────────────────────────────────────────

def _build_experiment(theta: NDArray, q_nom: float, soc_start: float, soc_end: float):
    """Construct a PyBaMM Experiment for a 3-stage CC-CV protocol.

    Stage durations are computed from SOC deltas and stage currents:
        t_k = (SOC_sw_k - SOC_sw_{k-1}) * Q_nom / I_k  [min]

    The pre-discharge step brings the cell from full charge to SOC_start.
    """
    import pybamm  # deferred: not imported at module level for subprocess safety

    I1, I2, I3, soc_sw1, soc_sw2, V_CV, I_cutoff = theta

    # Stage durations [min]; clipped to avoid zero-length steps
    t1 = max((soc_sw1 - soc_start) * q_nom / I1 * 60.0, 0.1)
    t2 = max((soc_sw2 - soc_sw1)   * q_nom / I2 * 60.0, 0.1)
    t3 = max((soc_end  - soc_sw2)   * q_nom / I3 * 60.0, 0.1)

    # Pre-discharge from SOC=1.0 to soc_start at C/5 rate
    t_dis = (1.0 - soc_start) * q_nom / (q_nom / 5.0) * 60.0  # min

    return pybamm.Experiment([
        f"Discharge at C/5 for {t_dis:.1f} minutes or until 2.5 V",
        "Rest for 1 minutes",
        f"Charge at {I1:.4f} A for {t1:.3f} minutes or until {V_CV:.4f} V",
        f"Charge at {I2:.4f} A for {t2:.3f} minutes or until {V_CV:.4f} V",
        f"Charge at {I3:.4f} A for {t3:.3f} minutes or until {V_CV:.4f} V",
        f"Hold at {V_CV:.4f} V until {I_cutoff:.4f} A",
    ])


# ── Single evaluation ──────────────────────────────────────────────────────────

def evaluate_single(theta: NDArray, config: Any) -> EvalResult:
    """Run one PyBaMM SPMe simulation and extract the three objectives.

    Args:
        theta:  (7,)  [I1, I2, I3, SOC_sw1, SOC_sw2, V_CV, I_cutoff]
        config: MOLLMBOConfig

    Returns:
        EvalResult  (penalty values + failed=True on exception)
    """
    import pybamm  # deferred for subprocess safety

    theta = np.asarray(theta, dtype=float)
    I1, I2, I3, soc_sw1, soc_sw2, V_CV, I_cutoff = theta
    monotone_ok = bool(I1 >= I2 >= I3)

    result = EvalResult(theta=theta, monotone_ok=monotone_ok)

    try:
        params = pybamm.ParameterValues("Chen2020")
        params["Ambient temperature [K]"] = config.t_ambient + 273.15

        model = pybamm.lithium_ion.SPMe(
            options={"SEI": "solvent-diffusion limited", "thermal": "lumped"}
        )
        experiment = _build_experiment(
            theta, config.q_nom, config.soc_start, config.soc_end
        )

        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = sim.solve(calc_esoh=False)

        # Verify the charging phase completed (≥4 cycles expected)
        if len(sol.cycles) < _CHARGE_CYCLE_START + 1:
            result.failed = True
            result.error_msg = f"Simulation stopped early: only {len(sol.cycles)} cycles"
            return result

        charge_cycles = sol.cycles[_CHARGE_CYCLE_START:]

        # ── Objective 1: charging time ─────────────────────────────────────
        t_start = charge_cycles[0]["Time [s]"].entries[0]
        t_end   = charge_cycles[-1]["Time [s]"].entries[-1]
        result.t_charge = (t_end - t_start) / 60.0  # → min

        # ── Objective 2: peak temperature ──────────────────────────────────
        result.T_peak = max(
            cyc["Volume-averaged cell temperature [K]"].entries.max()
            for cyc in charge_cycles
        ) - 273.15  # → °C

        # ── Objective 3: SEI aging (capacity loss) ─────────────────────────
        Q_loss_start = charge_cycles[0]["Loss of capacity to negative SEI [A.h]"].entries[0]
        Q_loss_end   = charge_cycles[-1]["Loss of capacity to negative SEI [A.h]"].entries[-1]
        result.delta_Q_aging = float(Q_loss_end - Q_loss_start)

        # ── Hard constraint checks ─────────────────────────────────────────
        V_max_seen = max(
            cyc["Terminal voltage [V]"].entries.max() for cyc in charge_cycles
        )
        result.V_violated = bool(V_max_seen > config.V_max)

        T_limit = config.T_max_conservative
        result.T_violated = bool(result.T_peak > T_limit)

        U_neg_min = min(
            cyc["X-averaged negative electrode surface potential difference [V]"].entries.min()
            for cyc in charge_cycles
        )
        result.plating_risk = bool(U_neg_min <= 0.0)

    except Exception as exc:
        result.failed = True
        result.error_msg = str(exc)
        log.warning("Simulation failed for θ=%s: %s", theta, exc)

    return result


# ── Batch evaluation (parallel) ────────────────────────────────────────────────

def evaluate_batch(
    thetas: list[NDArray],
    config: Any,
    workers: int | None = None,
) -> list[EvalResult]:
    """Evaluate a batch of θ vectors in parallel via ProcessPoolExecutor.

    Wall-clock time ≈ max(sim_i) instead of Σ sim_i.

    Args:
        thetas:  list of (7,) arrays
        config:  MOLLMBOConfig
        workers: number of parallel workers (default: config.n_workers)
    """
    n_workers = workers if workers is not None else config.n_workers
    n_workers = min(n_workers, len(thetas))

    results: list[EvalResult] = [None] * len(thetas)  # type: ignore[list-item]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(evaluate_single, t, config): i
            for i, t in enumerate(thetas)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                log.error("Worker exception for θ[%d]: %s", idx, exc)
                results[idx] = EvalResult(
                    theta=thetas[idx], failed=True, error_msg=str(exc)
                )

    return results


# ── Convenience helpers ────────────────────────────────────────────────────────

def results_to_objectives(results: list[EvalResult]) -> NDArray:
    """Stack objectives from a list of EvalResults.

    Returns:
        (n, 3)  [t_charge, T_peak, delta_Q_aging] — penalty rows for failures
    """
    return np.vstack([r.objectives for r in results])


def results_to_thetas(results: list[EvalResult]) -> NDArray:
    """Stack θ arrays from a list of EvalResults.  Returns (n, 7)."""
    return np.vstack([r.theta for r in results])


def filter_valid(results: list[EvalResult]) -> list[EvalResult]:
    """Return only results where constraint_ok is True."""
    return [r for r in results if r.constraint_ok]