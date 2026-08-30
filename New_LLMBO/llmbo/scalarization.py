from __future__ import annotations

from typing import Tuple

import numpy as np

OBJECTIVE_PREPROCESS_MODES = ("minmax", "zscore", "none")


def canonicalize_objective_preprocess_mode(mode: str | None) -> str:
    normalized = str(mode or "minmax").lower().replace("-", "_")
    if normalized in {"min_max", "minmax"}:
        return "minmax"
    if normalized in {"z", "zscore", "z_score", "standard", "standardize", "standardized"}:
        return "zscore"
    if normalized in {"none", "raw", "identity", "no"}:
        return "none"
    raise ValueError(
        f"Unknown objective_preprocess_mode: {mode}. "
        f"Expected one of {OBJECTIVE_PREPROCESS_MODES}."
    )


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


def compute_objective_preprocess_context(
    Y_tilde: np.ndarray,
    ideal_point_raw: np.ndarray,
    ref_point_raw: np.ndarray,
    preprocess_mode: str = "minmax",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return location/upper context for log-space objective preprocessing.

    The returned pair is intentionally compatible with the historical
    ``y_min``/``y_max`` contract: downstream code uses ``y_max - y_min`` as
    the scale. For min-max this is literal bounds; for z-score it is
    mean/mean+std; for none it is zeros/ones.
    """
    Y_tilde = np.atleast_2d(np.asarray(Y_tilde, dtype=float))
    mode = canonicalize_objective_preprocess_mode(preprocess_mode)

    if mode == "minmax":
        y_min, y_max = compute_dynamic_bounds(Y_tilde)
        return apply_min_range_floor(
            y_min,
            y_max,
            ideal_point_raw,
            ref_point_raw,
            min_fraction=0.05,
        )

    if mode == "zscore":
        center = np.mean(Y_tilde, axis=0)
        scale = np.std(Y_tilde, axis=0)
        global_min, global_max = compute_log_space_global_bounds(ideal_point_raw, ref_point_raw)
        fallback_scale = np.maximum(global_max - global_min, 1.0)
        bad_scale = (~np.isfinite(scale)) | (scale <= 1e-12)
        scale = np.where(bad_scale, fallback_scale, scale)
        return center, center + scale

    zeros = np.zeros(Y_tilde.shape[1], dtype=float)
    ones = np.ones(Y_tilde.shape[1], dtype=float)
    return zeros, ones


def normalize_objectives(
    Y_tilde: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    denom = np.asarray(y_max, dtype=float) - np.asarray(y_min, dtype=float)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return (np.asarray(Y_tilde, dtype=float) - np.asarray(y_min, dtype=float)) / denom


def normalize_objectives_raw(
    Y_raw: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> np.ndarray:
    denom = np.asarray(y_max, dtype=float) - np.asarray(y_min, dtype=float)
    denom = np.where(denom < 1e-12, 1.0, denom)
    return (np.asarray(Y_raw, dtype=float) - np.asarray(y_min, dtype=float)) / denom


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
    preprocess_mode: str = "minmax",
) -> np.ndarray:
    Y_tilde = log_transform_objectives(Y_raw)
    mode = canonicalize_objective_preprocess_mode(preprocess_mode)
    if mode == "none":
        Y_bar = Y_tilde
    else:
        Y_bar = normalize_objectives(Y_tilde, y_min, y_max)
    return compute_tchebycheff(Y_bar, w_vec, eta=eta)


def compute_tchebycheff_from_raw_with_ideal(
    Y_raw: np.ndarray,
    w_vec: np.ndarray,
    ideal_point_raw: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
    eta: float = 0.05,
    preprocess_mode: str = "minmax",
) -> np.ndarray:
    Y_tilde = log_transform_objectives(Y_raw)
    ideal_tilde = log_transform_objectives(np.asarray(ideal_point_raw, dtype=float)[None, :])[0]
    mode = canonicalize_objective_preprocess_mode(preprocess_mode)
    if mode == "none":
        Y_gap = np.abs(Y_tilde - ideal_tilde[np.newaxis, :])
    else:
        denom = np.asarray(y_max, dtype=float) - np.asarray(y_min, dtype=float)
        denom = np.where(denom < 1e-12, 1.0, denom)
        Y_gap = np.abs(Y_tilde - ideal_tilde[np.newaxis, :]) / denom[np.newaxis, :]
    return compute_tchebycheff(Y_gap, w_vec, eta=eta)


def prepare_parego_reference_weights(
    w_vec: np.ndarray,
    eps_min: float = 1e-6,
    invert: bool = True,
) -> np.ndarray:
    w = np.asarray(w_vec, dtype=float).ravel()
    w = np.maximum(w, float(eps_min))
    if invert:
        return 1.0 / w
    return w


def compute_parego_reference_from_raw(
    Y_raw: np.ndarray,
    w_vec: np.ndarray,
    eta: float = 0.05,
    eps_min: float = 1e-6,
    invert_weights: bool = True,
) -> np.ndarray:
    """Classic ParEGO scalarization on raw objectives with per-iteration min-max scaling."""
    Y_raw = np.atleast_2d(np.asarray(Y_raw, dtype=float))
    y_min = Y_raw.min(axis=0)
    y_max = Y_raw.max(axis=0)
    Y_bar = normalize_objectives_raw(Y_raw, y_min, y_max)
    w_eff = prepare_parego_reference_weights(w_vec, eps_min=eps_min, invert=invert_weights)
    return compute_tchebycheff(Y_bar, w_eff, eta=eta)


def canonical_hv_from_raw(hv_raw: float, hv_max: float) -> float:
    denom = float(hv_max) if float(hv_max) > 1e-12 else 1.0
    return float(hv_raw) / denom
