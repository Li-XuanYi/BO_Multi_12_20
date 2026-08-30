"""Pareto-front utilities for multi-objective optimization.

All objectives are assumed to be minimized throughout.
Public API:
    non_dominated_sort(objectives)          -> list[list[int]]
    compute_hypervolume(front, ref_point)   -> float
    compute_reference_point(objectives)     -> NDArray
    crowding_distance(front_objectives)     -> NDArray
    normalize_objectives(objectives, ...)   -> (NDArray, NDArray, NDArray)
    log_transform_aging(objectives)         -> NDArray  (in-place clone)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Float = np.floating[Any]


# ── Non-dominated sort ────────────────────────────────────────────────────────

def non_dominated_sort(objectives: NDArray[Float]) -> list[list[int]]:
    """NSGA-II fast non-dominated sort.

    Args:
        objectives: (n, m)  all objectives minimized
    Returns:
        Ordered list of fronts; front[0] = Pareto-optimal indices.
    """
    n = len(objectives)
    dom_count = np.zeros(n, dtype=int)      # #solutions dominating i
    dom_set: list[list[int]] = [[] for _ in range(n)]  # solutions i dominates

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(objectives[i], objectives[j]):
                dom_set[i].append(j)
                dom_count[j] += 1
            elif _dominates(objectives[j], objectives[i]):
                dom_set[j].append(i)
                dom_count[i] += 1

    fronts: list[list[int]] = [[i for i in range(n) if dom_count[i] == 0]]
    k = 0
    while k < len(fronts) and fronts[k]:
        nxt: list[int] = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        k += 1
        if nxt:
            fronts.append(nxt)

    return fronts


def _dominates(a: NDArray[Float], b: NDArray[Float]) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


# ── Hypervolume ───────────────────────────────────────────────────────────────

def compute_hypervolume(
    pareto_objectives: NDArray[Float],
    reference_point: NDArray[Float],
) -> float:
    """Hypervolume indicator (HV).

    Tries pygmo → pymoo → exact 2-D fallback.

    Args:
        pareto_objectives: (k, m)
        reference_point:   (m,)  must strictly dominate all Pareto points
    """
    try:
        import pygmo as pg
        return pg.hypervolume(pareto_objectives).compute(reference_point)
    except ImportError:
        pass
    try:
        from pymoo.indicators.hv import HV
        return float(HV(ref_point=reference_point)(pareto_objectives))
    except ImportError:
        pass
    if pareto_objectives.shape[1] == 2:
        return _hv_2d(pareto_objectives, reference_point)
    raise ImportError(
        "Install pygmo or pymoo for HV computation with m > 2 objectives."
    )


def _hv_2d(front: NDArray[Float], ref: NDArray[Float]) -> float:
    """Exact 2-D hypervolume (sweep-line)."""
    pts = front[np.argsort(front[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for pt in pts:
        hv += (ref[0] - pt[0]) * (prev_y - pt[1])
        prev_y = pt[1]
    return max(hv, 0.0)


def compute_reference_point(
    objectives: NDArray[Float],
    scale: float = 1.1,
) -> NDArray[Float]:
    """Reference point = scale × column-wise max over all evaluated points.

    Args:
        objectives: (n, m)
        scale:      §1 specifies 1.1; updated dynamically each iteration
    Returns:
        (m,)
    """
    return scale * objectives.max(axis=0)


# ── Crowding distance ─────────────────────────────────────────────────────────

def crowding_distance(front_objectives: NDArray[Float]) -> NDArray[Float]:
    """NSGA-II crowding distance.

    Args:
        front_objectives: (k, m)
    Returns:
        (k,)  boundary solutions → inf
    """
    k, m = front_objectives.shape
    if k <= 2:
        return np.full(k, np.inf)

    dist = np.zeros(k)
    for j in range(m):
        idx = np.argsort(front_objectives[:, j])
        obj_range = front_objectives[idx[-1], j] - front_objectives[idx[0], j] + 1e-8
        dist[idx[0]] = dist[idx[-1]] = np.inf
        for i in range(1, k - 1):
            dist[idx[i]] += (
                front_objectives[idx[i + 1], j] - front_objectives[idx[i - 1], j]
            ) / obj_range
    return dist


# ── Normalisation (§1.2) ──────────────────────────────────────────────────────

def normalize_objectives(
    objectives: NDArray[Float],
    f_min: NDArray[Float] | None = None,
    f_max: NDArray[Float] | None = None,
    eps: float = 1e-6,
) -> tuple[NDArray[Float], NDArray[Float], NDArray[Float]]:
    """Min-max normalisation with dynamic bound update.

    Args:
        objectives: (n, m)
        f_min, f_max: running bounds; None → computed from objectives
        eps: numerical stability constant ε
    Returns:
        (normalised, f_min, f_max)  bounds updated to cover current data
    """
    f_min = objectives.min(axis=0) if f_min is None else np.minimum(f_min, objectives.min(axis=0))
    f_max = objectives.max(axis=0) if f_max is None else np.maximum(f_max, objectives.max(axis=0))
    normalised = (objectives - f_min) / (f_max - f_min + eps)
    return normalised, f_min, f_max


def log_transform_objectives(
    objectives: NDArray[Float],
    time_col:   int  = 0,
    aging_col:  int  = 2,
    log_time:   bool = True,
    log_aging:  bool = True,
) -> NDArray[Float]:
    """Apply log10 to time and/or aging columns before normalisation (§1.2).

    Args:
        objectives: (n, 3)  [t_charge, T_peak, delta_Q_aging]
        time_col:   column index of t_charge  (default 0)
        aging_col:  column index of delta_Q_aging (default 2)
        log_time:   apply log10 to t_charge
        log_aging:  apply log10 to delta_Q_aging
    Returns:
        (n, 3) with selected columns log10-transformed; input unchanged
    """
    out = objectives.copy()
    if log_time:
        out[:, time_col]  = np.log10(np.maximum(out[:, time_col],  1e-10))
    if log_aging:
        out[:, aging_col] = np.log10(np.maximum(out[:, aging_col], 1e-10))
    return out


def log_transform_aging(objectives: NDArray[Float], aging_col: int = 2) -> NDArray[Float]:
    """Backward-compatible alias: log10 on aging column only."""
    return log_transform_objectives(
        objectives, log_time=False, log_aging=True, aging_col=aging_col
    )