from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from llmbo.acquisition import expected_improvement
from utils.constants import DSOC_SUM_MAX

PARAM_KEYS = ["I1", "I2", "I3", "dSOC1", "dSOC2"]


@dataclasses.dataclass
class LLMRegionPreference:
    kind: str = "none"
    coordinate_space: str = "raw"
    preference_direction: str = "promising"
    point: Optional[Dict[str, float]] = None
    lb: Optional[Dict[str, float]] = None
    ub: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    preference_type: str = "unspecified"
    reason: str = ""
    risk_flags: List[str] = dataclasses.field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None
    raw_response_hash: Optional[str] = None
    raw_text_preview: str = ""
    llm_call_diagnostics: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    parser_status: str = "ok"

    @classmethod
    def none(cls, reason: str = "none") -> "LLMRegionPreference":
        return cls(kind="none", confidence=0.0, parser_status=reason)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RegionLiftConfig:
    enable_region_lifted_gp: bool = False
    region_lift_external_influence_mode: str = "diagnostic_only"
    region_lift_lambda_max: float = 0.25
    region_lift_min_confidence: float = 0.60
    region_lift_n_anchors: int = 32
    region_lift_max_shift_std: float = 0.25
    region_lift_active_until: int = 12
    region_lift_anneal: str = "linear_decay"
    region_lift_max_plain_ei_gap: float = 0.25
    region_lift_log_ei_eps: float = 1e-12
    region_lift_kernel_jitter: float = 1e-6
    region_lift_min_norm_sq: float = 1e-12
    region_lift_min_volume: float = 1e-5
    region_lift_max_volume: float = 0.25
    region_lift_min_width: float = 0.03
    region_lift_max_width: float = 0.80
    region_lift_close_distance: float = 0.05
    region_lift_max_close_fraction: float = 0.5
    region_lift_min_feasible_anchor_ratio: float = 0.6
    region_lift_near_region_tol: float = 0.05
    region_lift_trust_init: float = 0.5
    region_lift_trust_beta: float = 0.2
    region_lift_anchor_weighting: str = "ei_softmax"
    region_lift_anchor_temperature: float = 0.35
    region_lift_require_inside: bool = True
    region_lift_min_sigma_ratio: float = 0.85
    region_lift_dsoc_margin: float = 0.02
    region_lift_guard_min_anchor_consistency: float = 0.35
    region_lift_guard_min_reliability: float = 0.20
    region_lift_guard_max_plain_ei_gap: float = 0.25
    region_lift_guard_require_inside: bool = True
    region_lift_guard_require_positive_corr: bool = True

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RegionLiftConfig":
        values = {}
        for field in dataclasses.fields(cls):
            if field.name in cfg:
                values[field.name] = cfg[field.name]
        return cls(**values)


@dataclasses.dataclass
class RegionLiftResult:
    selected_index: int
    selected_source: str
    accepted: bool
    fallback_reason: Optional[str]
    telemetry: Dict[str, Any]


def _hash_payload(payload: Any) -> str:
    try:
        text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    except Exception:
        text = str(payload)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _coerce_param_dict(value: Any) -> Optional[Dict[str, float]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        out: Dict[str, float] = {}
        for key in PARAM_KEYS:
            if key not in value:
                return None
            raw = value[key]
            if raw is None:
                return None
            try:
                out[key] = float(raw)
            except Exception:
                return None
        return out
    if isinstance(value, (list, tuple)) and len(value) == len(PARAM_KEYS):
        out = {}
        for key, raw in zip(PARAM_KEYS, value):
            if raw is None:
                return None
            try:
                out[key] = float(raw)
            except Exception:
                return None
        return out
    return None


def _extract_region_bounds(payload: Mapping[str, Any]) -> Tuple[Any, Any]:
    lb = payload.get("lb") or payload.get("lower") or payload.get("lower_bounds")
    ub = payload.get("ub") or payload.get("upper") or payload.get("upper_bounds")
    region = payload.get("region") or payload.get("bounds") or payload.get("box")
    if (lb is not None or ub is not None) or region is None:
        return lb, ub
    if isinstance(region, Mapping):
        return (
            region.get("lb") or region.get("lower") or region.get("lower_bounds"),
            region.get("ub") or region.get("upper") or region.get("upper_bounds"),
        )
    if isinstance(region, (list, tuple)) and len(region) == 2:
        return region[0], region[1]
    return lb, ub


def parse_region_preference_payload(payload: Any) -> LLMRegionPreference:
    if not isinstance(payload, Mapping):
        return LLMRegionPreference.none("invalid_json")
    kind = str(payload.get("kind", payload.get("type", payload.get("mode", "none")))).lower()
    if kind in {"box", "bounds"}:
        kind = "region"
    if kind not in {"point", "region", "none"}:
        return LLMRegionPreference.none("invalid_kind")
    raw_hash = _hash_payload(payload)
    lb_raw, ub_raw = _extract_region_bounds(payload)
    pref = LLMRegionPreference(
        kind=kind,
        coordinate_space=str(payload.get("coordinate_space", "raw")).lower(),
        preference_direction=str(payload.get("preference_direction", "promising")).lower(),
        point=_coerce_param_dict(payload.get("point") or payload.get("theta") or payload.get("x")),
        lb=_coerce_param_dict(lb_raw),
        ub=_coerce_param_dict(ub_raw),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        preference_type=str(payload.get("preference_type", "unspecified")),
        reason=str(payload.get("reason", payload.get("rationale", ""))),
        risk_flags=[str(x) for x in payload.get("risk_flags", [])] if isinstance(payload.get("risk_flags", []), list) else [],
        raw_response=dict(payload),
        raw_response_hash=raw_hash,
        parser_status="ok",
    )
    if pref.kind == "point" and pref.point is None:
        pref.kind = "none"
        pref.parser_status = "invalid_point"
    if pref.kind == "region" and (pref.lb is None or pref.ub is None):
        pref.kind = "none"
        pref.parser_status = "invalid_region_bounds"
    return pref


def _bounds_arrays(bounds: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.array([bounds[k][0] for k in PARAM_KEYS], dtype=float)
    hi = np.array([bounds[k][1] for k in PARAM_KEYS], dtype=float)
    return lo, hi


def _dict_to_array(values: Mapping[str, float]) -> np.ndarray:
    return np.array([float(values[k]) for k in PARAM_KEYS], dtype=float)


def _normalize(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=float) - lo) / (hi - lo + 1e-12)


def _zscore_feature(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if len(arr) <= 1:
        return np.zeros_like(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 1e-12:
        return np.zeros_like(arr)
    return np.clip((arr - mean) / std, -3.0, 3.0)


def _inside_region(X: np.ndarray, lb: np.ndarray, ub: np.ndarray, tol: float = 0.0) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.all((X >= lb - tol) & (X <= ub + tol), axis=1)


def _deterministic_feasible(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    in_bounds = np.all((X >= lo - 1e-12) & (X <= hi + 1e-12), axis=1)
    dsoc_ok = (X[:, 3] + X[:, 4]) < float(DSOC_SUM_MAX)
    return in_bounds & dsoc_ok


def _sobol_box(lb: np.ndarray, ub: np.ndarray, n: int) -> np.ndarray:
    n = max(int(n), 1)
    m = int(math.ceil(math.log2(n)))
    sampler = qmc.Sobol(d=len(PARAM_KEYS), scramble=False)
    unit = sampler.random_base2(m=m)[:n]
    return lb + unit * (ub - lb)


def _project_center_to_feasible(center: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(center, dtype=float).ravel(), lo, hi)
    dsoc_limit = float(DSOC_SUM_MAX) - 1e-6
    if x[3] + x[4] < dsoc_limit:
        return x

    excess = (x[3] + x[4]) - dsoc_limit
    room = np.maximum(x[3:5] - lo[3:5], 0.0)
    room_sum = float(np.sum(room))
    if room_sum <= 1e-12:
        x[3] = min(x[3], dsoc_limit - lo[4])
        x[4] = min(x[4], dsoc_limit - x[3])
        return np.clip(x, lo, hi)

    reduction = excess * (room / room_sum)
    x[3:5] = x[3:5] - reduction
    x = np.clip(x, lo, hi)
    if x[3] + x[4] >= dsoc_limit:
        scale = dsoc_limit / max(x[3] + x[4], 1e-12)
        x[3:5] *= scale
        x = np.clip(x, lo, hi)
    return x


def _fit_box_around_center(
    center: np.ndarray,
    width: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=float).ravel()
    width = np.asarray(width, dtype=float).ravel()
    span = hi - lo
    width = np.clip(width, 1e-9, span)

    lb = center - 0.5 * width
    ub = center + 0.5 * width

    shift_up = np.maximum(lo - lb, 0.0)
    lb = lb + shift_up
    ub = ub + shift_up

    shift_down = np.maximum(ub - hi, 0.0)
    lb = lb - shift_down
    ub = ub - shift_down

    lb = np.clip(lb, lo, hi - width)
    ub = lb + width
    ub = np.minimum(ub, hi)
    lb = np.maximum(ub - width, lo)
    return lb, ub


def _apply_dsoc_margin(
    lb: np.ndarray,
    ub: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    config: RegionLiftConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    lb = np.asarray(lb, dtype=float).copy()
    ub = np.asarray(ub, dtype=float).copy()
    margin = max(float(config.region_lift_dsoc_margin), 0.0)
    if margin <= 0.0:
        return lb, ub

    safe_limit = float(DSOC_SUM_MAX) - margin
    if ub[3] + ub[4] <= safe_limit + 1e-12:
        return lb, ub

    excess = (ub[3] + ub[4]) - safe_limit
    widths = np.maximum(ub[3:5] - lb[3:5], 1e-9)
    shares = widths / max(float(np.sum(widths)), 1e-12)

    lb[3:5] = lb[3:5] - excess * shares
    ub[3:5] = ub[3:5] - excess * shares
    lb = np.clip(lb, lo, hi)
    ub = np.clip(ub, lo, hi)

    if ub[3] + ub[4] > safe_limit + 1e-12:
        residual = (ub[3] + ub[4]) - safe_limit
        ub[3:5] = np.maximum(lb[3:5] + 1e-9, ub[3:5] - residual * shares)
        ub = np.clip(ub, lo, hi)
        lb = np.minimum(lb, ub - 1e-9)
    return lb, ub


def _repair_region_box(
    lb: np.ndarray,
    ub: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    config: RegionLiftConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    span = hi - lo
    min_width_norm = np.full(len(PARAM_KEYS), float(config.region_lift_min_width), dtype=float)
    max_width_norm = np.full(len(PARAM_KEYS), float(config.region_lift_max_width), dtype=float)
    width_norm = (np.asarray(ub, dtype=float) - np.asarray(lb, dtype=float)) / (span + 1e-12)
    width_norm = np.clip(width_norm, min_width_norm, max_width_norm)

    min_volume = max(float(config.region_lift_min_volume), 1e-12)
    max_volume = max(float(config.region_lift_max_volume), min_volume)

    volume = float(np.prod(width_norm))
    if volume < min_volume:
        growth = (min_volume / max(volume, 1e-12)) ** (1.0 / float(len(PARAM_KEYS)))
        width_norm = np.minimum(width_norm * growth, max_width_norm)
        for _ in range(len(PARAM_KEYS) * 2):
            volume = float(np.prod(width_norm))
            if volume >= min_volume - 1e-12:
                break
            headroom = max_width_norm - width_norm
            active = headroom > 1e-12
            if not np.any(active):
                break
            growth = (min_volume / max(volume, 1e-12)) ** (1.0 / float(np.sum(active)))
            updated = width_norm.copy()
            updated[active] = np.minimum(width_norm[active] * growth, max_width_norm[active])
            if np.allclose(updated, width_norm):
                break
            width_norm = updated
    elif volume > max_volume:
        shrink = (max_volume / max(volume, 1e-12)) ** (1.0 / float(len(PARAM_KEYS)))
        width_norm = np.maximum(width_norm * shrink, min_width_norm)

    center = 0.5 * (np.asarray(lb, dtype=float) + np.asarray(ub, dtype=float))
    center = _project_center_to_feasible(center, lo, hi)
    width = np.clip(width_norm * span, 1e-9, span)
    repaired_lb, repaired_ub = _fit_box_around_center(center, width, lo, hi)
    repaired_lb, repaired_ub = _apply_dsoc_margin(repaired_lb, repaired_ub, lo, hi, config)
    return repaired_lb, repaired_ub


def _gp_anchor_weights(
    gp: Any,
    anchors: np.ndarray,
    f_min_z: float,
    existing_X: np.ndarray,
    bounds: Mapping[str, Tuple[float, float]],
    config: RegionLiftConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mode = str(config.region_lift_anchor_weighting).lower()
    uniform = np.full(len(anchors), 1.0 / max(len(anchors), 1), dtype=float)
    if len(anchors) == 0:
        return uniform, {"anchor_weighting_mode": "empty"}
    if mode == "uniform":
        return uniform, {
            "anchor_weighting_mode": "uniform",
            "anchor_weight_max": float(np.max(uniform)),
            "anchor_weight_entropy": float(-np.sum(uniform * np.log(uniform + 1e-12))),
        }

    try:
        mean_a, sigma_a = gp.predict_standardized(anchors)
        ei_a = expected_improvement(mean_a, sigma_a, f_min_z)
    except Exception as exc:
        return uniform, {"anchor_weighting_mode": f"uniform_after_gp_error:{type(exc).__name__}"}

    signal_ei = np.asarray(ei_a, dtype=float).ravel()
    if len(signal_ei) != len(anchors):
        return uniform, {"anchor_weighting_mode": "uniform_after_shape_mismatch"}
    lo, hi = _bounds_arrays(bounds)
    anchors_norm = _normalize(anchors, lo, hi)
    existing = np.atleast_2d(np.asarray(existing_X, dtype=float)) if np.asarray(existing_X).size else np.empty((0, len(PARAM_KEYS)))
    if existing.size:
        existing_norm = _normalize(existing, lo, hi)
        novelty = np.min(np.linalg.norm(anchors_norm[:, None, :] - existing_norm[None, :, :], axis=2), axis=1)
    else:
        novelty = np.ones(len(anchors), dtype=float)

    signal = (
        0.60 * _zscore_feature(np.log(np.maximum(signal_ei, 1e-12)))
        + 0.25 * _zscore_feature(np.asarray(sigma_a, dtype=float))
        + 0.15 * _zscore_feature(novelty)
    )
    if not np.any(np.isfinite(signal)):
        return uniform, {"anchor_weighting_mode": "uniform_after_nonfinite_signal"}

    temperature = max(float(config.region_lift_anchor_temperature), 1e-6)
    logits = signal / temperature
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
        return uniform, {"anchor_weighting_mode": "uniform_after_bad_softmax"}
    weights = weights / weight_sum
    entropy = float(-np.sum(weights * np.log(weights + 1e-12)))
    entropy_norm = entropy / max(math.log(len(weights) + 1e-12), 1e-12)
    signal_spread = float(np.std(signal))
    anchor_consistency = float(np.clip(0.55 * (1.0 - entropy_norm) + 0.45 * np.tanh(max(signal_spread, 0.0)), 0.0, 1.0))
    return weights, {
        "anchor_weighting_mode": mode,
        "anchor_best_ei": float(np.max(ei_a)),
        "anchor_mean_ei": float(np.mean(ei_a)),
        "anchor_weight_max": float(np.max(weights)),
        "anchor_weight_entropy": entropy,
        "anchor_consistency": anchor_consistency,
        "anchor_novelty_mean": float(np.mean(novelty)),
    }


def _preference_bounds(
    preference: LLMRegionPreference,
    bounds: Mapping[str, Tuple[float, float]],
    config: RegionLiftConfig,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    lo, hi = _bounds_arrays(bounds)
    if preference.kind == "region":
        lb = _dict_to_array(preference.lb or {})
        ub = _dict_to_array(preference.ub or {})
    elif preference.kind == "point":
        point = _dict_to_array(preference.point or {})
        rel_width = max(
            float(config.region_lift_min_width),
            float(config.region_lift_min_volume) ** (1.0 / float(len(PARAM_KEYS))),
        )
        rel_width = min(rel_width, float(config.region_lift_max_width))
        width = np.maximum((hi - lo) * rel_width, 1e-9)
        lb = point - 0.5 * width
        ub = point + 0.5 * width
    else:
        return None, None, "no_preference"

    raw_lb = np.minimum(lb, ub)
    raw_ub = np.maximum(lb, ub)
    raw_lb = np.clip(raw_lb, lo, hi)
    raw_ub = np.clip(raw_ub, lo, hi)
    lb, ub = _repair_region_box(raw_lb, raw_ub, lo, hi, config)
    if np.any(ub <= lb + 1e-12):
        return None, None, "empty_region"
    return lb, ub, ""


def sample_region_candidates(
    preference: LLMRegionPreference,
    bounds: Mapping[str, Tuple[float, float]],
    config: RegionLiftConfig,
    *,
    n_candidates: Optional[int] = None,
) -> np.ndarray:
    fallback = _validate_preference(preference, config)
    if fallback:
        return np.empty((0, len(PARAM_KEYS)), dtype=float)

    lb, ub, bounds_error = _preference_bounds(preference, bounds, config)
    if lb is None or ub is None or bounds_error:
        return np.empty((0, len(PARAM_KEYS)), dtype=float)

    lo, hi = _bounds_arrays(bounds)
    span = np.maximum(hi - lo, 1e-12)
    width_norm = (ub - lb) / span
    relative_volume = float(np.prod(width_norm))
    tol = 1e-12
    if relative_volume < float(config.region_lift_min_volume) - tol or relative_volume > float(config.region_lift_max_volume) + tol:
        return np.empty((0, len(PARAM_KEYS)), dtype=float)
    if np.any(width_norm < float(config.region_lift_min_width) - tol) or np.any(width_norm > float(config.region_lift_max_width) + tol):
        return np.empty((0, len(PARAM_KEYS)), dtype=float)

    total = max(int(n_candidates or config.region_lift_n_anchors), 1)
    candidates = _sobol_box(lb, ub, total)
    feasible_mask = _deterministic_feasible(candidates, lo, hi)
    candidates = candidates[feasible_mask]
    if len(candidates) == 0:
        center = _project_center_to_feasible(0.5 * (lb + ub), lo, hi)
        if _deterministic_feasible(center[None, :], lo, hi)[0]:
            return center[None, :]
        return np.empty((0, len(PARAM_KEYS)), dtype=float)

    unique: List[np.ndarray] = []
    seen = set()
    for row in candidates:
        key = tuple(np.round(np.asarray(row, dtype=float), 12))
        if key in seen:
            continue
        seen.add(key)
        unique.append(np.asarray(row, dtype=float))
    return np.vstack(unique)


def evaluate_region_lift_on_pool(
    *,
    gp: Any,
    candidate_pool: np.ndarray,
    f_min_y: float,
    preference: LLMRegionPreference,
    existing_X: np.ndarray,
    bounds: Mapping[str, Tuple[float, float]],
    config: RegionLiftConfig,
    trust: float,
    bo_iteration: int,
    plain_index_override: Optional[int] = None,
) -> RegionLiftResult:
    X_pool = np.atleast_2d(np.asarray(candidate_pool, dtype=float))
    plain_index = 0
    telemetry: Dict[str, Any] = {
        "active": True,
        "llm_raw_response_hash": preference.raw_response_hash,
        "parser_status": preference.parser_status,
        "selected_source": "fallback",
        "fallback_reason": None,
        "trust_before": float(trust),
        "trust_after": float(trust),
        "trust_update_reason": "pending",
    }
    fallback_selected_source = "fallback"

    def _fallback(reason: str) -> RegionLiftResult:
        telemetry["selected_source"] = fallback_selected_source
        telemetry["fallback_reason"] = reason
        return RegionLiftResult(plain_index, fallback_selected_source, False, reason, telemetry)

    try:
        mean_z, sigma_z = gp.predict_standardized(X_pool)
        y_mean, y_std = gp.target_standardization()
        if hasattr(gp, "transform_targets"):
            f_min_model = float(
                np.asarray(gp.transform_targets(np.array([float(f_min_y)], dtype=float)), dtype=float).ravel()[0]
            )
        else:
            f_min_model = float(f_min_y)
        f_min_z = (float(f_min_model) - y_mean) / y_std
    except Exception as exc:
        return _fallback(f"standardization_unavailable:{type(exc).__name__}")

    ei_plain = expected_improvement(mean_z, sigma_z, f_min_z)
    plain_index = int(np.argmax(ei_plain))
    if plain_index_override is not None:
        override_idx = int(plain_index_override)
        if 0 <= override_idx < len(X_pool):
            plain_index = override_idx
    eps = float(config.region_lift_log_ei_eps)
    plain_log = np.log(np.maximum(ei_plain, eps))
    telemetry.update(
        {
            "plain_idx": int(plain_index),
            "plain_log_ei_surrogate_at_plain": float(plain_log[plain_index]),
            "region_lift_log_ei_eps": eps,
        }
    )

    fallback = _validate_preference(preference, config)
    if fallback:
        return _fallback(fallback)

    lb, ub, bounds_error = _preference_bounds(preference, bounds, config)
    if lb is None or ub is None:
        return _fallback(bounds_error)

    lo, hi = _bounds_arrays(bounds)
    span = np.maximum(hi - lo, 1e-12)
    width_norm = (ub - lb) / span
    relative_volume = float(np.prod(width_norm))
    telemetry.update(
        {
            "region_raw_lb": lb.tolist(),
            "region_raw_ub": ub.tolist(),
            "region_normalized_lb": _normalize(lb, lo, hi).tolist(),
            "region_normalized_ub": _normalize(ub, lo, hi).tolist(),
            "relative_volume": relative_volume,
            "per_dim_widths": width_norm.tolist(),
            "plain_candidate_inside_region": bool(_inside_region(X_pool[plain_index], lb, ub, tol=0.0)[0]),
        }
    )
    tol = 1e-12
    if relative_volume < float(config.region_lift_min_volume) - tol or relative_volume > float(config.region_lift_max_volume) + tol:
        return _fallback("bad_region_volume")
    if np.any(width_norm < float(config.region_lift_min_width) - tol) or np.any(width_norm > float(config.region_lift_max_width) + tol):
        return _fallback("bad_region_width")

    anchors = _sobol_box(lb, ub, int(config.region_lift_n_anchors))
    existing = np.atleast_2d(np.asarray(existing_X, dtype=float)) if np.asarray(existing_X).size else np.empty((0, len(PARAM_KEYS)))
    anchors_norm = _normalize(anchors, lo, hi)
    if existing.size:
        existing_norm = _normalize(existing, lo, hi)
        dists = np.linalg.norm(anchors_norm[:, None, :] - existing_norm[None, :, :], axis=2)
        min_dists = dists.min(axis=1)
    else:
        min_dists = np.full(anchors.shape[0], np.inf, dtype=float)
    close_fraction = float(np.mean(min_dists <= float(config.region_lift_close_distance)))
    feasible_mask = _deterministic_feasible(anchors, lo, hi)
    feasible_ratio = float(np.mean(feasible_mask))
    telemetry.update(
        {
            "anchor_min_dist_to_existing": float(np.min(min_dists)) if len(min_dists) else None,
            "anchor_close_fraction": close_fraction,
            "feasible_anchor_ratio": feasible_ratio,
        }
    )
    if close_fraction > float(config.region_lift_max_close_fraction):
        return _fallback("too_close_to_existing")
    if feasible_ratio < float(config.region_lift_min_feasible_anchor_ratio):
        return _fallback("low_feasible_anchor_ratio")

    anchors = anchors[feasible_mask]
    if len(anchors) == 0:
        return _fallback("no_feasible_anchors")
    weights, weight_telemetry = _gp_anchor_weights(
        gp=gp,
        anchors=anchors,
        f_min_z=float(f_min_z),
        existing_X=existing,
        bounds=bounds,
        config=config,
    )
    telemetry.update(weight_telemetry)

    try:
        K_gg = gp.posterior_covariance_standardized(anchors, anchors)
        K_xg = gp.posterior_covariance_standardized(X_pool, anchors)
    except Exception as exc:
        return _fallback(f"kernel_unavailable:{type(exc).__name__}")
    jitter = float(config.region_lift_kernel_jitter)
    K_gg = np.asarray(K_gg, dtype=float) + jitter * np.eye(len(anchors))
    K_xg = np.asarray(K_xg, dtype=float)
    norm_sq = float(weights @ K_gg @ weights)
    anneal = _anneal_factor(int(bo_iteration), config)
    anchor_consistency = float(np.clip(weight_telemetry.get("anchor_consistency", 0.0), 0.0, 1.0))
    reliability = (
        float(np.clip(preference.confidence, 0.0, 1.0))
        * float(np.clip(trust, 0.0, 1.0))
        * anchor_consistency
    )
    lambda_t = anneal * float(config.region_lift_lambda_max)
    telemetry.update(
        {
            "lambda_t": float(lambda_t),
            "kernel_norm_sq": float(norm_sq),
            "anneal_t": float(anneal),
            "region_reliability": float(reliability),
            "positive_kernel_fraction": float(np.mean(K_xg > 0.0)) if K_xg.size else 0.0,
        }
    )
    if anneal <= 0.0:
        return _fallback("inactive_anneal")
    if lambda_t <= 0.0 or reliability <= 0.0:
        return _fallback("zero_lambda")

    var_x = np.clip(np.asarray(sigma_z, dtype=float) ** 2, 1e-12, None)
    corr = np.asarray(K_xg @ weights, dtype=float).ravel() / np.sqrt(
        np.maximum(var_x * max(norm_sq, float(config.region_lift_min_norm_sq)), 1e-12)
    )
    corr = np.clip(corr, -1.0, 1.0)
    lift_score = lambda_t * reliability * np.maximum(corr, 0.0)
    shift_z = np.clip(lift_score, 0.0, float(config.region_lift_max_shift_std))
    telemetry.update(
        {
            "corr_at_plain": float(corr[plain_index]) if len(corr) else 0.0,
            "max_corr": float(np.max(corr)) if len(corr) else 0.0,
            "max_shift_z": float(np.max(shift_z)) if len(shift_z) else 0.0,
            "mean_shift_z": float(np.mean(shift_z)) if len(shift_z) else 0.0,
        }
    )
    if float(np.max(shift_z)) <= eps:
        return _fallback("zero_shift")

    mean_lifted_z = mean_z - shift_z
    ei_lifted = expected_improvement(mean_lifted_z, sigma_z, f_min_z)
    lift_index = int(np.argmax(ei_lifted))
    if lift_index == plain_index:
        telemetry["lift_idx"] = int(lift_index)
        telemetry["sigma_unchanged"] = True
        return _fallback("same_as_plain")

    plain_log_at_lift = float(plain_log[lift_index])
    gap = float(plain_log[plain_index] - plain_log_at_lift)
    telemetry.update(
        {
            "lift_idx": int(lift_index),
            "plain_log_ei_surrogate_at_lift": plain_log_at_lift,
            "lifted_ei_at_lift": float(ei_lifted[lift_index]),
            "plain_ei_at_plain": float(ei_plain[plain_index]),
            "plain_ei_at_lift": float(ei_plain[lift_index]),
            "plain_ei_gap": gap,
            "corr_at_lift": float(corr[lift_index]),
            "lift_candidate_inside_region": bool(_inside_region(X_pool[lift_index], lb, ub, tol=float(config.region_lift_near_region_tol))[0]),
            "sigma_unchanged": True,
            "lift_sigma_z": float(sigma_z[lift_index]),
            "plain_sigma_z": float(sigma_z[plain_index]),
        }
    )

    if gap > float(config.region_lift_max_plain_ei_gap):
        return _fallback("plain_ei_gap")
    if bool(config.region_lift_require_inside) and not bool(telemetry["lift_candidate_inside_region"]):
        return _fallback("outside_region")
    sigma_ratio = float(sigma_z[lift_index] / max(float(sigma_z[plain_index]), 1e-12))
    telemetry["sigma_ratio_vs_plain"] = sigma_ratio
    if sigma_ratio < float(config.region_lift_min_sigma_ratio):
        return _fallback("low_sigma_ratio")

    telemetry["selected_source"] = "lifted"
    telemetry["fallback_reason"] = None
    return RegionLiftResult(lift_index, "lifted", True, None, telemetry)


def _validate_preference(preference: LLMRegionPreference, config: RegionLiftConfig) -> Optional[str]:
    if preference.kind == "none":
        return preference.parser_status or "no_preference"
    if preference.parser_status != "ok":
        return preference.parser_status
    if preference.coordinate_space != "raw":
        return "non_raw_coordinate_space"
    if preference.preference_direction != "promising":
        return "non_promising_direction"
    if float(preference.confidence) < float(config.region_lift_min_confidence):
        return "low_confidence"
    if preference.kind not in {"point", "region"}:
        return "invalid_kind"
    return None


def _anneal_factor(bo_iteration: int, config: RegionLiftConfig) -> float:
    active_until = max(int(config.region_lift_active_until), 0)
    if active_until <= 0:
        return 0.0
    if str(config.region_lift_anneal).lower() == "none":
        return 1.0
    t = max(int(bo_iteration), 0)
    return float(np.clip(1.0 - (t / float(active_until)), 0.0, 1.0))
