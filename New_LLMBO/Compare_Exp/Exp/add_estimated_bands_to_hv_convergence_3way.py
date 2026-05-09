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


TARGET_PNG = PROJECT_ROOT / "Compare_Exp" / "images" / "(HV)05-03_seed8409" / "hv_convergence_3way.png"
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

NSGA2_COLOR = "#e67e22"
PAREGO_COLOR = "#4e8fb5"
LLMBO_COLOR = "#c85d6b"

PAREGO_BAND_SCALE = 1.60
LLMBO_BAND_SCALE = 1.85
ESTIMATED_BAND_ALPHA = 0.16


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


def _extract_canonical_trace(summary_path: Path, *, max_evals: int = 56) -> Dict[str, np.ndarray]:
    summary = _load_json(summary_path)
    hv_trace = summary.get("hv_trace") or []
    if not hv_trace:
        raise ValueError(f"Missing hv_trace in {summary_path}")

    x = np.asarray([int(item.get("eval_index", idx + 1)) for idx, item in enumerate(hv_trace)], dtype=int)
    y = np.asarray([float(item.get("canonical_hv", 0.0)) for item in hv_trace], dtype=float)
    mask = x <= int(max_evals)
    return {"x": x[mask], "hv": y[mask]}


def _stack_nsga2_mean_std(*, max_evals: int = 56) -> Dict[str, np.ndarray]:
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
    return {
        "x": x_ref,
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
    }


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
    max_evals: int = 56,
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


def main() -> None:
    _configure_plot_style()

    llmbo_trace = _extract_canonical_trace(LLMBO_SINGLE_SUMMARY)
    parego_trace = _extract_canonical_trace(PAREGO_SINGLE_SUMMARY)
    nsga2 = _stack_nsga2_mean_std()

    llmbo_sources, llmbo_params = _resolve_proxy_sources("llmbo_band_profile")
    parego_sources, parego_params = _resolve_proxy_sources("parego_band_profile")

    llmbo_band = _build_estimated_canonical_band(llmbo_sources, **llmbo_params)
    parego_band = _build_estimated_canonical_band(parego_sources, **parego_params)

    if not np.array_equal(llmbo_trace["x"], llmbo_band["x"]):
        raise ValueError("LLMBO single trace and proxy band x-axis mismatch")
    if not np.array_equal(parego_trace["x"], parego_band["x"]):
        raise ValueError("ParEGO single trace and proxy band x-axis mismatch")

    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    _plot_line_with_band(
        ax,
        parego_trace["x"],
        parego_trace["hv"],
        parego_band["band"] * PAREGO_BAND_SCALE,
        color=PAREGO_COLOR,
        label="ParEGO",
        marker="s",
        alpha=ESTIMATED_BAND_ALPHA,
    )
    _plot_line_with_band(
        ax,
        llmbo_trace["x"],
        llmbo_trace["hv"],
        llmbo_band["band"] * LLMBO_BAND_SCALE,
        color=LLMBO_COLOR,
        label="LLAMBO-MO",
        marker="o",
        alpha=ESTIMATED_BAND_ALPHA,
    )
    _plot_line_with_band(
        ax,
        nsga2["x"],
        nsga2["mean"],
        nsga2["std"],
        color=NSGA2_COLOR,
        label="NSGA-II",
        marker="v",
        alpha=0.10,
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

    TARGET_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Updated: {TARGET_PNG}")


if __name__ == "__main__":
    main()
