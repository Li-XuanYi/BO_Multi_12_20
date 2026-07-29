"""Preview Ecker2015 convergence with a bounded, benchmark-calibrated NHV.

This script does not overwrite the archived canonical-HV figures or summaries.
It applies one common reporting box to both methods:

    ideal = [2500 s, 0 K, 0.004 %]
    ref   = [7000 s, 35 K, 0.009 %]

Time and aging are log10-transformed. Every transformed objective is normalized
and clipped to [0, 1], and hypervolume is evaluated against [1, 1, 1].
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from pymoo.config import Config
from pymoo.indicators.hv import HV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmbo.scalarization import log_transform_objectives


ARCHIVE_ROOT = (
    PROJECT_ROOT
    / "Compare_Exp"
    / "experiment_records"
    / "Ecker2015_HV05-12"
)
IMAGE_ROOT = PROJECT_ROOT / "Compare_Exp" / "images" / "Ecker2015_HV05-12"
CURVE_ROOT = ARCHIVE_ROOT / "curve_data"

IDEAL_RAW = np.array([2500.0, 0.0, 0.004], dtype=float)
REFERENCE_RAW = np.array([7000.0, 35.0, 0.009], dtype=float)
REFERENCE_NORMALIZED = np.ones(3, dtype=float)
Config.warnings["not_compiled"] = False
HV_CALCULATOR = HV(ref_point=REFERENCE_NORMALIZED)
MAX_EVALS = 56
SEEDS = (8409, 8410, 8411, 8412, 8413)

METHODS = {
    "ParEGO": {
        "pattern": "raw_sources/ParEGO/seed{seed}/parego_matlab_reference_Ecker2015/database.json",
        "color": "#2878B5",
        "marker": "s",
    },
    "LLMBO-MO": {
        "pattern": "raw_sources/LLMBO-MO/seed{seed}/minmax/database.json",
        "color": "#D62728",
        "marker": "o",
    },
}


def _normalized_objectives(points_raw: np.ndarray) -> np.ndarray:
    points_tilde = log_transform_objectives(points_raw)
    ideal_tilde = log_transform_objectives(IDEAL_RAW[None, :])[0]
    reference_tilde = log_transform_objectives(REFERENCE_RAW[None, :])[0]
    denominator = reference_tilde - ideal_tilde
    return np.clip((points_tilde - ideal_tilde) / denominator, 0.0, 1.0)


def _nhv(points_raw: List[np.ndarray]) -> float:
    if not points_raw:
        return 0.0
    points_normalized = _normalized_objectives(np.asarray(points_raw, dtype=float))
    return float(HV_CALCULATOR.do(points_normalized))


def _load_trace(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    feasible_points: List[np.ndarray] = []
    trace: List[float] = []
    for observation in payload["observations"][:MAX_EVALS]:
        if bool(observation.get("feasible", False)):
            feasible_points.append(np.asarray(observation["objectives"], dtype=float))
        trace.append(_nhv(feasible_points))
    if len(trace) != MAX_EVALS:
        raise ValueError(f"Expected {MAX_EVALS} observations in {path}, got {len(trace)}")
    return np.asarray(trace, dtype=float)


def _collect() -> Dict[str, Dict[str, np.ndarray]]:
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for method, spec in METHODS.items():
        traces = []
        for seed in SEEDS:
            path = ARCHIVE_ROOT / spec["pattern"].format(seed=seed)
            if not path.exists():
                raise FileNotFoundError(path)
            traces.append(_load_trace(path))
        stack = np.vstack(traces)
        result[method] = {
            "traces": stack,
            "mean": stack.mean(axis=0),
            "std": stack.std(axis=0, ddof=0),
        }
    return result


def _write_outputs(result: Dict[str, Dict[str, np.ndarray]]) -> None:
    CURVE_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = CURVE_ROOT / "ecker2015_nhv_calibrated_preview.csv"
    json_path = CURVE_ROOT / "ecker2015_nhv_calibrated_preview.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "evaluation",
                "parego_mean_nhv",
                "parego_population_std",
                "llmbo_mo_mean_nhv",
                "llmbo_mo_population_std",
            ]
        )
        for index in range(MAX_EVALS):
            writer.writerow(
                [
                    index + 1,
                    result["ParEGO"]["mean"][index],
                    result["ParEGO"]["std"][index],
                    result["LLMBO-MO"]["mean"][index],
                    result["LLMBO-MO"]["std"][index],
                ]
            )

    final_values = {
        method: {
            "per_seed": {
                str(seed): float(result[method]["traces"][seed_index, -1])
                for seed_index, seed in enumerate(SEEDS)
            },
            "mean": float(result[method]["mean"][-1]),
            "population_std": float(result[method]["std"][-1]),
        }
        for method in METHODS
    }
    payload = {
        "status": "calibrated_preview_not_adopted_as_paper_metric",
        "metric": {
            "name": "normalized hypervolume",
            "ideal_raw": IDEAL_RAW.tolist(),
            "reference_raw": REFERENCE_RAW.tolist(),
            "reference_normalized": REFERENCE_NORMALIZED.tolist(),
            "log10_objectives": ["charging_time", "capacity_fade"],
            "clipping": [0.0, 1.0],
            "std": "population (ddof=0)",
        },
        "seeds": list(SEEDS),
        "max_evaluations": MAX_EVALS,
        "final": final_values,
        "delta_mean": (
            final_values["LLMBO-MO"]["mean"] - final_values["ParEGO"]["mean"]
        ),
        "relative_gain_percent": (
            final_values["LLMBO-MO"]["mean"] / final_values["ParEGO"]["mean"] - 1.0
        )
        * 100.0,
        "paired_wins": sum(
            final_values["LLMBO-MO"]["per_seed"][str(seed)]
            > final_values["ParEGO"]["per_seed"][str(seed)]
            for seed in SEEDS
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot(result: Dict[str, Dict[str, np.ndarray]]) -> None:
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    evaluations = np.arange(1, MAX_EVALS + 1)
    marker_indices = np.array([0, 7, 14, 21, 28, 35, 42, 49])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 15,
            "axes.labelsize": 19,
            "legend.fontsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    for method, spec in METHODS.items():
        mean = result[method]["mean"]
        std = result[method]["std"]
        color = spec["color"]
        ax.fill_between(
            evaluations,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            evaluations,
            mean,
            color=color,
            linewidth=3.2,
            marker=spec["marker"],
            markevery=marker_indices,
            markersize=8,
            label=method,
        )
        ax.annotate(
            f"{mean[-1]:.4f}",
            xy=(MAX_EVALS, mean[-1]),
            xytext=(-8, 10 if method == "LLMBO-MO" else -22),
            textcoords="offset points",
            ha="right",
            color=color,
            fontweight="bold",
        )

    ax.set_xlim(0, MAX_EVALS)
    ax.set_ylim(0.05, 0.60)
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Normalized hypervolume")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#666666")
    fig.tight_layout()

    png_path = IMAGE_ROOT / "ecker2015_nhv_calibrated_preview.png"
    pdf_path = IMAGE_ROOT / "ecker2015_nhv_calibrated_preview.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    result = _collect()
    _write_outputs(result)
    _plot(result)
    for method in METHODS:
        print(
            f"{method}: "
            f"{result[method]['mean'][-1]:.10f} +/- "
            f"{result[method]['std'][-1]:.10f}"
        )


if __name__ == "__main__":
    main()
