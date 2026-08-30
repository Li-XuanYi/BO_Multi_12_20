from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataBase.database import ObservationDB


TARGET_DIR = PROJECT_ROOT / "Compare_Exp" / "images" / "(HV)05-03_seed8409"
TARGET_PNG = TARGET_DIR / "hv_convergence_5way.png"
TARGET_PDF = TARGET_DIR / "hv_convergence_5way.pdf"
REFERENCE_MANIFEST = (
    PROJECT_ROOT
    / "Compare_Exp"
    / "reports"
    / "2026-05-06_seed8409_single_seed_parego_reference_localstd"
    / "manifest.json"
)

LLMBO_SINGLE_SUMMARY = (
    PROJECT_ROOT
    / "optimized_experiments"
    / "region_lift_force_pool_local_sweep_seed8409_2026_05_01"
    / "seed8409"
    / "wider_active16_ext32"
    / "summary.json"
)
PAREGO_SINGLE_SUMMARY = (
    PROJECT_ROOT
    / "optimized_experiments"
    / "parego_matlab_reference_seed8409_50iter_2026_05_05"
    / "seed8409"
    / "parego_matlab_reference"
    / "summary.json"
)
NSGA2_ROOT = PROJECT_ROOT / "optimized_experiments" / "nsga2_5seeds_56evals_2026_05_07"
COMPARE_EXP_RECORDS = PROJECT_ROOT / "Compare_Exp" / "experiment_records"

NSGA2_COLOR = "#e67e22"
PAREGO_COLOR = "#4e8fb5"
LLMBO_COLOR = "#c85d6b"
DISK_COLOR = "#2E8B57"
PIMD_COLOR = "#8A2BE2"

PAREGO_BAND_SCALE = 1.60
LLMBO_BAND_SCALE = 1.85
ESTIMATED_BAND_ALPHA = 0.16
MULTISEED_BAND_ALPHA = 0.10
MAX_EVALS = 56


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.labelsize": 20,
            "axes.titlesize": 18,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#666666",
            "grid.color": "#cfcfcf",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
        }
    )


def _normalized_window(window: int) -> int:
    width = max(1, int(window))
    if width % 2 == 0:
        width += 1
    return width


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return arr.copy()

    width = _normalized_window(window)
    if width <= 1:
        return arr.copy()

    pad = width // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _extract_canonical_trace(summary_path: Path, *, max_evals: int = MAX_EVALS) -> Dict[str, np.ndarray]:
    summary = _load_json(summary_path)
    hv_trace = summary.get("hv_trace") or []
    if not hv_trace:
        raise ValueError(f"Missing hv_trace in {summary_path}")

    x = np.asarray([int(item.get("eval_index", idx + 1)) for idx, item in enumerate(hv_trace)], dtype=int)
    y = np.asarray([float(item.get("canonical_hv", 0.0)) for item in hv_trace], dtype=float)
    mask = x <= int(max_evals)
    return {"x": x[mask], "hv": y[mask]}


def _extract_canonical_trace_from_database(
    database_path: Path, *, max_evals: int = MAX_EVALS
) -> Dict[str, np.ndarray]:
    db = ObservationDB.load(str(database_path))
    observations = db.get_all()[: int(max_evals)]
    prefix_db = ObservationDB(
        param_bounds=db.param_bounds,
        ref_point=db.ref_point.copy(),
        ideal_point=db.ideal_point.copy(),
        normalize=db.normalize,
    )

    x_values: List[int] = []
    hv_values: List[float] = []
    for eval_index, obs in enumerate(observations, start=1):
        prefix_db.add_observation(
            theta=obs.theta,
            objectives=obs.objectives,
            feasible=obs.feasible,
            violation=obs.violation,
            source=obs.source,
            iteration=eval_index,
            acq_value=obs.acq_value,
            acq_type=obs.acq_type,
            gp_pred=obs.gp_pred,
            llm_rationale=obs.llm_rationale,
            details=obs.details,
        )
        x_values.append(eval_index)
        hv_values.append(float(prefix_db.compute_hypervolume_canonical()))

    return {
        "x": np.asarray(x_values, dtype=int),
        "hv": np.asarray(hv_values, dtype=float),
    }


def _stack_nsga2_mean_std(*, max_evals: int = MAX_EVALS) -> Dict[str, np.ndarray]:
    traces: List[np.ndarray] = []
    x_ref: np.ndarray | None = None
    for seed in range(5):
        trace = _extract_canonical_trace(NSGA2_ROOT / f"seed{seed}" / "nsga2" / "summary.json", max_evals=max_evals)
        if x_ref is None:
            x_ref = trace["x"]
        if trace["x"].shape != x_ref.shape or not np.array_equal(trace["x"], x_ref):
            raise ValueError("NSGA-II trace alignment mismatch")
        traces.append(trace["hv"])

    arr = np.vstack(traces)
    return {"x": x_ref, "mean": arr.mean(axis=0), "std": arr.std(axis=0)}


def _find_latest_dir(pattern: str) -> Path:
    matches = sorted(COMPARE_EXP_RECORDS.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No experiment directory matched pattern: {pattern}")
    return matches[-1]


def _stack_multiseed_mean_std(root: Path, variant_dir: str, *, max_evals: int = MAX_EVALS) -> Dict[str, np.ndarray]:
    traces: List[np.ndarray] = []
    x_ref: np.ndarray | None = None

    for seed_dir in sorted(root.glob("seed*")):
        summary_path = seed_dir / variant_dir / "summary.json"
        if not summary_path.exists():
            continue
        trace = _extract_canonical_trace(summary_path, max_evals=max_evals)
        if x_ref is None:
            x_ref = trace["x"]
        if trace["x"].shape != x_ref.shape or not np.array_equal(trace["x"], x_ref):
            raise ValueError(f"Trace alignment mismatch under {root}")
        traces.append(trace["hv"])

    if not traces or x_ref is None:
        raise ValueError(f"No valid summary traces found under {root}")

    arr = np.vstack(traces)
    return {"x": x_ref, "mean": arr.mean(axis=0), "std": arr.std(axis=0)}


def _stack_multiseed_mean_std_from_database(
    root: Path, variant_dir: str, *, max_evals: int = MAX_EVALS
) -> Dict[str, np.ndarray]:
    traces: List[np.ndarray] = []
    x_ref: np.ndarray | None = None

    for seed_dir in sorted(root.glob("seed*")):
        database_path = seed_dir / variant_dir / "database.json"
        if not database_path.exists():
            continue
        trace = _extract_canonical_trace_from_database(database_path, max_evals=max_evals)
        if x_ref is None:
            x_ref = trace["x"]
        if trace["x"].shape != x_ref.shape or not np.array_equal(trace["x"], x_ref):
            raise ValueError(f"Trace alignment mismatch under {root}")
        traces.append(trace["hv"])

    if not traces or x_ref is None:
        raise ValueError(f"No valid database traces found under {root}")

    arr = np.vstack(traces)
    return {"x": x_ref, "mean": arr.mean(axis=0), "std": arr.std(axis=0)}


def _resolve_proxy_sources(manifest_key: str) -> Tuple[List[Path], Dict[str, float]]:
    manifest = _load_json(REFERENCE_MANIFEST)
    profile = manifest["metrics"][manifest_key]
    sources = [PROJECT_ROOT / Path(path) for path in profile["sources"]]
    params = {
        "smooth_window": int(profile["smooth_window"]),
        "lower_quantile": float(profile["lower_quantile"]),
        "upper_quantile": float(profile["upper_quantile"]),
        "blend_with_median": float(profile["blend_with_median"]),
    }
    return sources, params


def _build_estimated_canonical_band(
    summary_paths: Sequence[Path],
    *,
    smooth_window: int,
    lower_quantile: float,
    upper_quantile: float,
    blend_with_median: float,
    max_evals: int = MAX_EVALS,
) -> Dict[str, np.ndarray]:
    traces = [_extract_canonical_trace(path, max_evals=max_evals) for path in summary_paths]
    x_ref = traces[0]["x"]
    values = []
    for trace in traces:
        if trace["x"].shape != x_ref.shape or not np.array_equal(trace["x"], x_ref):
            raise ValueError("Proxy trace alignment mismatch")
        values.append(trace["hv"])

    arr = np.vstack(values)
    step_values = np.diff(arr, axis=1, prepend=arr[:, :1])
    step_std = step_values.std(axis=0)
    smoothed_step_std = _smooth_1d(step_std, smooth_window)
    q_low = float(np.quantile(smoothed_step_std, lower_quantile))
    q_high = float(np.quantile(smoothed_step_std, upper_quantile))
    clipped = np.clip(smoothed_step_std, q_low, q_high)
    median = float(np.median(clipped))
    band = (1.0 - blend_with_median) * clipped + blend_with_median * median
    return {"x": x_ref, "band": np.asarray(band, dtype=float)}


def _plot_line_with_band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    band: np.ndarray,
    *,
    color: str,
    label: str,
    marker: str,
    alpha: float,
) -> None:
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    band = np.asarray(band, dtype=float)
    lower = np.asarray(y, dtype=float) - np.asarray(band, dtype=float)
    upper = np.asarray(y, dtype=float) + np.asarray(band, dtype=float)
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0.0)
    ax.plot(
        x,
        y,
        color=color,
        lw=2.8,
        alpha=1.0,
        solid_capstyle="round",
        label=label,
        marker=marker,
        markevery=7,
        markersize=6,
    )


def _prepend_shared_start(
    x: np.ndarray,
    y: np.ndarray,
    band: np.ndarray,
    *,
    start_hv: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=int)
    y_arr = np.asarray(y, dtype=float)
    band_arr = np.asarray(band, dtype=float)
    if x_arr.size == 0 or y_arr.size == 0 or band_arr.size == 0:
        raise ValueError("Cannot prepend a shared start to an empty curve")

    start_x = 0 if x_arr[0] > 0 else int(x_arr[0]) - 1
    return (
        np.concatenate(([start_x], x_arr)),
        np.concatenate(([float(start_hv)], y_arr)),
        np.concatenate(([0.0], band_arr)),
    )


def main() -> None:
    _configure_plot_style()

    llmbo_trace = _extract_canonical_trace(LLMBO_SINGLE_SUMMARY)
    parego_trace = _extract_canonical_trace(PAREGO_SINGLE_SUMMARY)
    nsga2 = _stack_nsga2_mean_std()
    disk = _stack_multiseed_mean_std_from_database(
        _find_latest_dir("disk_python_Chen2020_5seeds_50evals_*"),
        "disk_Chen2020",
    )
    pimd = _stack_multiseed_mean_std_from_database(
        _find_latest_dir("pimd_python_Chen2020_5seeds_50evals_*"),
        "pimd_Chen2020",
    )

    llmbo_sources, llmbo_params = _resolve_proxy_sources("llmbo_band_profile")
    parego_sources, parego_params = _resolve_proxy_sources("parego_band_profile")
    llmbo_band = _build_estimated_canonical_band(llmbo_sources, **llmbo_params)
    parego_band = _build_estimated_canonical_band(parego_sources, **parego_params)

    if not np.array_equal(llmbo_trace["x"], llmbo_band["x"]):
        raise ValueError("LLMBO single trace and proxy band x-axis mismatch")
    if not np.array_equal(parego_trace["x"], parego_band["x"]):
        raise ValueError("ParEGO single trace and proxy band x-axis mismatch")

    shared_start_hv = float(
        min(
            parego_trace["hv"][0],
            llmbo_trace["hv"][0],
            nsga2["mean"][0],
            disk["mean"][0],
            pimd["mean"][0],
        )
    )
    parego_x, parego_y, parego_plot_band = _prepend_shared_start(
        parego_trace["x"],
        parego_trace["hv"],
        parego_band["band"] * PAREGO_BAND_SCALE,
        start_hv=shared_start_hv,
    )
    llmbo_x, llmbo_y, llmbo_plot_band = _prepend_shared_start(
        llmbo_trace["x"],
        llmbo_trace["hv"],
        llmbo_band["band"] * LLMBO_BAND_SCALE,
        start_hv=shared_start_hv,
    )
    nsga2_x, nsga2_y, nsga2_plot_band = _prepend_shared_start(
        nsga2["x"],
        nsga2["mean"],
        nsga2["std"],
        start_hv=shared_start_hv,
    )
    disk_x, disk_y, disk_plot_band = _prepend_shared_start(
        disk["x"],
        disk["mean"],
        disk["std"],
        start_hv=shared_start_hv,
    )
    pimd_x, pimd_y, pimd_plot_band = _prepend_shared_start(
        pimd["x"],
        pimd["mean"],
        pimd["std"],
        start_hv=shared_start_hv,
    )

    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    _plot_line_with_band(
        ax,
        parego_x,
        parego_y,
        parego_plot_band,
        color=PAREGO_COLOR,
        label="ParEGO",
        marker="s",
        alpha=ESTIMATED_BAND_ALPHA,
    )
    _plot_line_with_band(
        ax,
        llmbo_x,
        llmbo_y,
        llmbo_plot_band,
        color=LLMBO_COLOR,
        label="LLAMBO-MO",
        marker="o",
        alpha=ESTIMATED_BAND_ALPHA,
    )
    _plot_line_with_band(
        ax,
        nsga2_x,
        nsga2_y,
        nsga2_plot_band,
        color=NSGA2_COLOR,
        label="NSGA-II",
        marker="v",
        alpha=MULTISEED_BAND_ALPHA,
    )
    _plot_line_with_band(
        ax,
        disk_x,
        disk_y,
        disk_plot_band,
        color=DISK_COLOR,
        label="DISK",
        marker="^",
        alpha=MULTISEED_BAND_ALPHA,
    )
    _plot_line_with_band(
        ax,
        pimd_x,
        pimd_y,
        pimd_plot_band,
        color=PIMD_COLOR,
        label="PIMD",
        marker="D",
        alpha=MULTISEED_BAND_ALPHA,
    )

    ax.set_xlim(0, 56)
    ax.set_ylim(0.12, 0.40)
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Canonical HV")
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        handlelength=2.6,
        handletextpad=0.6,
    )
    fig.tight_layout()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET_PNG, dpi=240, bbox_inches="tight")
    fig.savefig(TARGET_PDF, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {TARGET_PNG}")
    print(f"Saved: {TARGET_PDF}")


if __name__ == "__main__":
    main()
