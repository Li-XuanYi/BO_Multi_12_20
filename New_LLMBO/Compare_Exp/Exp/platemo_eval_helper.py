"""
Python-side evaluator used by the PlatEMO MATLAB bridge.

MATLAB calls ``evaluate_json`` for each expensive evaluation.  Keeping this
logic in Python lets DISK/PIMD use the same PyBaMM simulator, bounds, penalty
conventions, and trace format as the native Compare_Exp runners.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from pybamm_simulator import PyBaMMSimulator
from utils.constants import DSOC_SUM_MAX, FAILURE_PENALTY, REF_POINT

_SIMULATOR: Optional[PyBaMMSimulator] = None
_START_TIME = time.time()
_EVAL_COUNT = 0


def _get_simulator() -> PyBaMMSimulator:
    global _SIMULATOR
    if _SIMULATOR is None:
        _SIMULATOR = PyBaMMSimulator()
    return _SIMULATOR


def _as_float_array(theta: Iterable[Any]) -> np.ndarray:
    return np.asarray([float(x) for x in list(theta)], dtype=float).ravel()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate(theta: Iterable[Any], trace_path: str = "") -> Dict[str, Any]:
    """Evaluate one 5D battery charging design and optionally append JSONL."""
    global _EVAL_COUNT

    x = _as_float_array(theta)
    if x.size != 5:
        raise ValueError(f"Expected 5 decision variables, got {x.size}: {x!r}")

    dsoc_violation = float(x[3] + x[4] - DSOC_SUM_MAX)
    if dsoc_violation > 0.0:
        feasible = False
        objectives = FAILURE_PENALTY.copy()
        violation = f"dSOC1+dSOC2 exceeds {DSOC_SUM_MAX:.6g}"
        details: Dict[str, Any] = {"skipped_simulator": True}
    else:
        try:
            result = _get_simulator().evaluate(x)
            feasible = bool(result.get("feasible", False))
            objectives = np.asarray(result.get("raw_objectives", REF_POINT), dtype=float).ravel()
            if objectives.size != 3:
                objectives = REF_POINT.copy()
                feasible = False
            violation = result.get("violation")
            details = _json_safe(result.get("details", {}))
        except Exception as exc:  # MATLAB should receive a penalty, not lose the whole run.
            feasible = False
            objectives = FAILURE_PENALTY.copy()
            violation = f"simulator_exception: {type(exc).__name__}: {exc}"
            details = {"exception": repr(exc)}
        if not feasible:
            objectives = np.asarray(objectives if objectives.size == 3 else FAILURE_PENALTY, dtype=float)

    constraint = 0.0 if feasible else max(1.0, dsoc_violation)
    _EVAL_COUNT += 1
    record = {
        "eval_index": int(_EVAL_COUNT),
        "theta": x.tolist(),
        "objectives": objectives.tolist(),
        "feasible": feasible,
        "constraint": float(constraint),
        "dsoc_sum": float(x[3] + x[4]),
        "violation": violation,
        "details": details,
        "elapsed_s": round(time.time() - _START_TIME, 6),
        "timestamp": datetime.now().isoformat(),
    }
    _append_jsonl(trace_path, record)
    return record


def evaluate_json(theta: Iterable[Any], trace_path: str = "") -> str:
    """MATLAB-friendly wrapper returning a JSON string."""
    return json.dumps(evaluate(theta, trace_path), ensure_ascii=False)
