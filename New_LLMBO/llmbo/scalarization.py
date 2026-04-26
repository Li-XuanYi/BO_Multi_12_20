from __future__ import annotations

from typing import Tuple

import numpy as np


def log_transform_objectives(Y_raw: np.ndarray) -> np.ndarray:
    """Transform objectives to the HV/scalarization space.

    Time and aging use log10; temperature remains in raw Kelvin-rise units.
    """
    Y_raw = np.atleast_2d(np.asarray(Y_raw, dtype=float))
    Y_tilde = Y_raw.copy()
    Y_tilde[:, 0] = np.log10(np.maximum(Y_raw[:, 0], 1.0))
    Y_tilde[:, 2] = np.log10(np.maximum(Y_raw[:, 2], 1e-12))
    return Y_tilde


def compute_dynamic_bounds(Y_tilde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Y_tilde = np.atleast_2d(np.asarray(Y_tilde, dtype=float))
    return Y_tilde.min(axis=0), Y_tilde.max(axis=0)


def compute_log_space_global_bounds(
    ideal_point_raw: np.ndarray,
    ref_point_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    global_min = log_transform_objectives(np.asarray(ideal_point_raw, dtype=float)[None, :])[0]
    global_max = log_transform_objectives(np.asarray(ref_point_raw, dtype=float)[None, :])[0]
    return global_min, global_max


def apply_min_range_floor(
    y_min: np.ndarray,
    y_max: np.ndarray,
    ideal_point_raw: np.ndarray,
    ref_point_raw: np.ndarray,
    min_fraction: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Avoid near-zero dynamic ranges by falling back to global bounds per objective."""
    y_min = np.asarray(y_min, dtype=float).copy()
    y_max = np.asarray(y_max, dtype=float).copy()
    global_min, global_max = compute_log_space_global_bounds(ideal_point_raw, ref_point_raw)
    hist_range = y_max - y_min
    global_range = global_max - global_min
    threshold = float(min_fraction) * global_range
    mask = hist_range < threshold
    y_min[mask] = global_min[mask]
    y_max[mask] = global_max[mask]
    return y_min, y_max


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


def canonical_hv_from_raw(hv_raw: float, hv_max: float) -> float:
    denom = float(hv_max) if float(hv_max) > 1e-12 else 1.0
    return float(hv_raw) / denom
