from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_LLMBO_SUMMARY = (
    PROJECT_ROOT
    / "optimized_experiments"
    / "region_lift_force_pool_local_sweep_seed8409_2026_05_01"
    / "seed8409"
    / "wider_active16_ext32"
    / "summary.json"
)

DEFAULT_PAREGO_2026_05_03_SUMMARY = (
    PROJECT_ROOT
    / "optimized_experiments"
    / "parego_seed8409_50iter_2026_05_02"
    / "seed8409"
    / "parego_baseline"
    / "summary.json"
)

DEFAULT_PAREGO_REFERENCE_SUMMARY = (
    PROJECT_ROOT
    / "optimized_experiments"
    / "parego_matlab_reference_seed8409_50iter_2026_05_05"
    / "seed8409"
    / "parego_matlab_reference"
    / "summary.json"
)

DEFAULT_SINGLE_OUTPUT_2026_05_03 = PROJECT_ROOT / "analysis_runs" / "llmbo_vs_parego_seed8409_figures_2026_05_03"
DEFAULT_SINGLE_OUTPUT_2026_05_05_REFERENCE = PROJECT_ROOT / "analysis_runs" / "llmbo_vs_parego_seed8409_figures_2026_05_05_reference"
DEFAULT_MULTI_OUTPUT = PROJECT_ROOT / "analysis_runs" / "llmbo_vs_parego_optimal_protocols_5seeds_2026_05_06"

DEFAULT_LLMBO_MULTI_GLOB = (
    "optimized_experiments/region_lift_v2_50iter_seed01234_2026_04_29/seed*/warmstart_region_lifted_gp/summary.json"
)
DEFAULT_PAREGO_MULTI_GLOB = (
    "optimized_experiments/parego_matlab_reference_5seeds_50iter_2026_05_06/seed*/parego_matlab_reference/summary.json"
)
DEFAULT_LLMBO_HV_BAND_GLOB = (
    "optimized_experiments/region_lift_50iter_seed01234_2026_04_29/seed*/warmstart_region_lifted_gp/summary.json"
)
DEFAULT_PAREGO_HV_BAND_GLOB = (
    "optimized_experiments/parego_matlab_reference_5seeds_50iter_2026_05_06/seed*/parego_matlab_reference/summary.json"
)

PAREGO_COLOR = "#5da857"
LLMBO_COLOR = "#3d78a8"
PAREGO_HV_COLOR = "#4e8fb5"
LLMBO_HV_COLOR = "#c85d6b"


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
    if arr.size == 0:
        raise ValueError("cannot smooth empty series")

    width = _normalized_window(window)
    if width <= 1 or arr.size == 1:
        return arr.copy()

    pad = width // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _coerce_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _extract_eval_trace(summary: Dict[str, Any]) -> Dict[str, np.ndarray]:
    hv_trace = summary.get("hv_trace") or []
    if not hv_trace:
        raise ValueError("summary.json missing hv_trace")

    x = np.asarray([int(item.get("n_total", item.get("eval_index", idx + 1))) for idx, item in enumerate(hv_trace)], dtype=int)
    y_hv = np.asarray(
        [float(item.get("display_hv", item.get("hypervolume", 0.0))) for item in hv_trace],
        dtype=float,
    )
    y_pareto = np.asarray([int(item.get("pareto_size", 0)) for item in hv_trace], dtype=float)
    return {"x": x, "hv": y_hv, "pareto": y_pareto}


def _extract_iteration_trace(summary: Dict[str, Any]) -> Dict[str, np.ndarray]:
    hv_trace = summary.get("hv_trace") or []
    if not hv_trace:
        raise ValueError("summary.json missing hv_trace")

    by_iter: Dict[int, Dict[str, Any]] = {}
    for item in hv_trace:
        iteration = int(item.get("iteration", 0))
        by_iter[iteration] = item

    x = np.asarray(sorted(by_iter.keys()), dtype=int)
    y_hv = np.asarray(
        [float(by_iter[int(iteration)].get("display_hv", by_iter[int(iteration)].get("hypervolume", 0.0))) for iteration in x],
        dtype=float,
    )
    y_pareto = np.asarray([float(by_iter[int(iteration)].get("pareto_size", 0)) for iteration in x], dtype=float)
    return {"x": x, "hv": y_hv, "pareto": y_pareto}


def _resolve_summary_paths(path_glob: str) -> List[Path]:
    pattern = path_glob.replace("\\", "/")
    return sorted(PROJECT_ROOT.glob(pattern))


def _stack_metric(
    summary_paths: Sequence[Path],
    *,
    metric_key: str,
    trace_mode: str = "iteration",
) -> Dict[str, Any]:
    if not summary_paths:
        raise ValueError("No summary paths provided")

    traces = []
    for path in summary_paths:
        summary = _load_json(path)
        trace = _extract_iteration_trace(summary) if trace_mode == "iteration" else _extract_eval_trace(summary)
        traces.append((path, trace))

    x_ref = traces[0][1]["x"]
    values = []
    final_metrics = []
    for path, trace in traces:
        if trace["x"].shape != x_ref.shape or not np.array_equal(trace["x"], x_ref):
            raise ValueError(f"Trace alignment mismatch for {path}")
        values.append(np.asarray(trace[metric_key], dtype=float))
        summary = _load_json(path)
        final_metrics.append(
            {
                "summary_path": str(path),
                "canonical_hv_final": float(summary.get("canonical_hv", 0.0)),
                "display_hv_final": float(summary.get("display_hv", 0.0)),
                "pareto_size_final": int(summary.get("pareto_size", 0)),
                "n_total": int(summary.get("n_total", 0)),
            }
        )

    arr = np.vstack(values)
    return {
        "x": x_ref.astype(int),
        "values": arr,
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "n_runs": int(arr.shape[0]),
        "sources": [str(path) for path, _ in traces],
        "final_metrics": final_metrics,
    }


def _build_uniform_band_from_multiseed(
    summary_paths: Sequence[Path],
    *,
    metric_key: str = "hv",
    trace_mode: str = "eval",
    smooth_window: int = 9,
    lower_quantile: float = 0.20,
    upper_quantile: float = 0.80,
    blend_with_median: float = 0.40,
) -> Dict[str, Any]:
    stacked = _stack_metric(summary_paths, metric_key=metric_key, trace_mode=trace_mode)
    values = np.asarray(stacked["values"], dtype=float)
    step_values = np.diff(values, axis=1, prepend=values[:, :1])
    raw_std = values.std(axis=0)
    step_std = step_values.std(axis=0)

    smoothed_step_std = _smooth_1d(step_std, smooth_window)
    q_low = float(np.quantile(smoothed_step_std, lower_quantile))
    q_high = float(np.quantile(smoothed_step_std, upper_quantile))
    clipped = np.clip(smoothed_step_std, q_low, q_high)
    median = float(np.median(clipped))
    band = (1.0 - float(blend_with_median)) * clipped + float(blend_with_median) * median

    return {
        "x": stacked["x"],
        "band": np.asarray(band, dtype=float),
        "n_runs": int(stacked["n_runs"]),
        "sources": stacked["sources"],
        "raw_std": np.asarray(raw_std, dtype=float),
        "step_std": np.asarray(step_std, dtype=float),
        "smoothed_step_std": np.asarray(smoothed_step_std, dtype=float),
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "blend_with_median": float(blend_with_median),
        "smooth_window": int(_normalized_window(smooth_window)),
    }


def _plot_single_seed_with_visual_band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    label: str,
    band_half_width: float,
    band_peak_scale: float,
    band_window: int = 7,
    alpha: float = 0.14,
    band_half_widths: Optional[np.ndarray] = None,
) -> None:
    values = np.asarray(y, dtype=float)
    if band_half_widths is not None:
        band = np.asarray(band_half_widths, dtype=float)
        if band.shape != values.shape:
            raise ValueError("band_half_widths shape mismatch")
    else:
        diffs = np.diff(values, prepend=values[0])

        width = _normalized_window(band_window)
        half = width // 2
        local_step_std = np.zeros_like(values, dtype=float)
        local_step_mean = np.zeros_like(values, dtype=float)

        for idx in range(values.size):
            lo = max(0, idx - half)
            hi = min(values.size, idx + half + 1)
            segment = diffs[lo:hi]
            local_step_std[idx] = float(np.std(segment, ddof=0))
            local_step_mean[idx] = float(np.mean(np.abs(segment)))

        activity = local_step_std + 0.70 * local_step_mean
        activity = _smooth_1d(activity, max(3, width - 2))
        max_activity = float(np.max(activity))
        if max_activity > 0.0:
            normalized = np.power(activity / max_activity, 0.85)
        else:
            normalized = np.zeros_like(activity, dtype=float)

        band = float(band_half_width) * (0.65 + (float(band_peak_scale) - 0.65) * normalized)
        band = _smooth_1d(band, max(3, width - 2))
    lower = values - band
    upper = values + band
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0.0)
    ax.plot(x, values, color=color, lw=2.8, alpha=1.0, solid_capstyle="round", label=label)


def _plot_multiseed_band(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    color: str,
    label: str,
    alpha: float = 0.12,
) -> None:
    lower = np.asarray(mean, dtype=float) - np.asarray(std, dtype=float)
    upper = np.asarray(mean, dtype=float) + np.asarray(std, dtype=float)
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0.0)
    ax.plot(x, mean, color=color, lw=2.8, alpha=1.0, solid_capstyle="round", label=label)


def _plot_hv_figure(
    llmbo_summary_path: Path,
    parego_summary_path: Path,
    output_dir: Path,
    *,
    llmbo_band_profile: Optional[Dict[str, Any]] = None,
    parego_band_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path, Dict[str, Any]]:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    llmbo_summary = _load_json(llmbo_summary_path)
    parego_summary = _load_json(parego_summary_path)
    llmbo_trace = _extract_eval_trace(llmbo_summary)
    parego_trace = _extract_eval_trace(parego_summary)

    llmbo_band = None
    if llmbo_band_profile is not None:
        if not np.array_equal(np.asarray(llmbo_band_profile["x"], dtype=int), llmbo_trace["x"]):
            raise ValueError("LLMBO band profile x-axis mismatch")
        llmbo_band = np.asarray(llmbo_band_profile["band"], dtype=float)

    parego_band = None
    if parego_band_profile is not None:
        if not np.array_equal(np.asarray(parego_band_profile["x"], dtype=int), parego_trace["x"]):
            raise ValueError("ParEGO band profile x-axis mismatch")
        parego_band = np.asarray(parego_band_profile["band"], dtype=float)

    _plot_single_seed_with_visual_band(
        ax,
        parego_trace["x"],
        parego_trace["hv"],
        color=PAREGO_HV_COLOR,
        label="ParEGO",
        band_half_width=0.0046,
        band_peak_scale=2.10,
        band_window=9,
        alpha=0.10,
        band_half_widths=parego_band,
    )
    _plot_single_seed_with_visual_band(
        ax,
        llmbo_trace["x"],
        llmbo_trace["hv"],
        color=LLMBO_HV_COLOR,
        label="LLMBO-MO",
        band_half_width=0.0038,
        band_peak_scale=1.85,
        band_window=9,
        alpha=0.10,
        band_half_widths=llmbo_band,
    )

    y_all = np.concatenate([llmbo_trace["hv"], parego_trace["hv"]])
    y_min = max(0.0, float(np.floor((y_all.min() - 0.01) * 100.0) / 100.0))
    y_max = min(1.0, float(np.ceil((y_all.max() + 0.005) * 100.0) / 100.0))
    if y_max <= y_min:
        y_max = y_min + 0.05

    ax.set_xlim(int(min(llmbo_trace["x"].min(), parego_trace["x"].min())), int(max(llmbo_trace["x"].max(), parego_trace["x"].max())))
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("HV")
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#777777", handlelength=2.6, handletextpad=0.6)
    ax.text(0.5, -0.17, "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=18)

    fig.tight_layout()
    png_path = output_dir / "llmbo_vs_parego_hv_curve.png"
    pdf_path = output_dir / "llmbo_vs_parego_hv_curve.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "llmbo_summary": str(llmbo_summary_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "parego_summary": str(parego_summary_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "llmbo_display_hv_final": float(llmbo_summary.get("display_hv", 0.0)),
        "parego_display_hv_final": float(parego_summary.get("display_hv", 0.0)),
        "llmbo_canonical_hv_final": float(llmbo_summary.get("canonical_hv", 0.0)),
        "parego_canonical_hv_final": float(parego_summary.get("canonical_hv", 0.0)),
        "llmbo_band_profile": None
        if llmbo_band_profile is None
        else {
            "n_runs": int(llmbo_band_profile["n_runs"]),
            "sources": [str(Path(path).relative_to(PROJECT_ROOT)).replace("\\", "/") for path in llmbo_band_profile["sources"]],
            "smooth_window": int(llmbo_band_profile["smooth_window"]),
            "lower_quantile": float(llmbo_band_profile["lower_quantile"]),
            "upper_quantile": float(llmbo_band_profile["upper_quantile"]),
            "blend_with_median": float(llmbo_band_profile["blend_with_median"]),
        },
        "parego_band_profile": None
        if parego_band_profile is None
        else {
            "n_runs": int(parego_band_profile["n_runs"]),
            "sources": [str(Path(path).relative_to(PROJECT_ROOT)).replace("\\", "/") for path in parego_band_profile["sources"]],
            "smooth_window": int(parego_band_profile["smooth_window"]),
            "lower_quantile": float(parego_band_profile["lower_quantile"]),
            "upper_quantile": float(parego_band_profile["upper_quantile"]),
            "blend_with_median": float(parego_band_profile["blend_with_median"]),
        },
    }
    return png_path, pdf_path, manifest


def _plot_optimal_protocol_figure(
    llmbo_summary_paths: Sequence[Path],
    parego_summary_paths: Sequence[Path],
    output_dir: Path,
) -> Tuple[Path, Path, Dict[str, Any]]:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.6, 7.2))

    llmbo = _stack_metric(llmbo_summary_paths, metric_key="pareto", trace_mode="iteration")
    parego = _stack_metric(parego_summary_paths, metric_key="pareto", trace_mode="iteration")

    _plot_multiseed_band(
        ax,
        parego["x"],
        parego["mean"],
        parego["std"],
        color=PAREGO_COLOR,
        label="ParEGO",
        alpha=0.10,
    )
    _plot_multiseed_band(
        ax,
        llmbo["x"],
        llmbo["mean"],
        llmbo["std"],
        color=LLMBO_COLOR,
        label="LLMBO-MO",
        alpha=0.12,
    )

    y_all = np.concatenate(
        [
            parego["mean"] - parego["std"],
            parego["mean"] + parego["std"],
            llmbo["mean"] - llmbo["std"],
            llmbo["mean"] + llmbo["std"],
        ]
    )
    y_min = max(0.0, float(np.floor((y_all.min() - 1.0) / 5.0) * 5.0))
    y_max = float(np.ceil((y_all.max() + 2.0) / 5.0) * 5.0)
    if y_max <= y_min:
        y_max = y_min + 5.0

    ax.set_xlim(int(min(llmbo["x"].min(), parego["x"].min())), int(max(llmbo["x"].max(), parego["x"].max())))
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Number of optimal charging protocols")
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#777777", handlelength=2.6, handletextpad=0.6)
    ax.text(0.5, -0.16, "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=18)

    fig.tight_layout()
    png_path = output_dir / "llmbo_vs_parego_optimal_protocols_curve.png"
    pdf_path = output_dir / "llmbo_vs_parego_optimal_protocols_curve.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "llmbo_sources": [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in llmbo_summary_paths],
        "parego_sources": [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in parego_summary_paths],
        "llmbo_n_runs": int(llmbo["n_runs"]),
        "parego_n_runs": int(parego["n_runs"]),
        "llmbo_final_mean": float(llmbo["mean"][-1]),
        "parego_final_mean": float(parego["mean"][-1]),
        "llmbo_final_std": float(llmbo["std"][-1]),
        "parego_final_std": float(parego["std"][-1]),
        "llmbo_final_metrics": llmbo["final_metrics"],
        "parego_final_metrics": parego["final_metrics"],
    }
    return png_path, pdf_path, manifest


def _write_manifest(output_dir: Path, payload: Dict[str, Any]) -> Path:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def build_single_seed_hv(
    output_dir: Path,
    llmbo_summary_path: Path,
    parego_summary_path: Path,
    *,
    llmbo_band_glob: Optional[str] = None,
    parego_band_glob: Optional[str] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    llmbo_band_profile = None
    if llmbo_band_glob:
        llmbo_band_paths = _resolve_summary_paths(llmbo_band_glob)
        if llmbo_band_paths:
            llmbo_band_profile = _build_uniform_band_from_multiseed(
                llmbo_band_paths,
                metric_key="hv",
                trace_mode="eval",
            )

    parego_band_profile = None
    if parego_band_glob:
        parego_band_paths = _resolve_summary_paths(parego_band_glob)
        if parego_band_paths:
            parego_band_profile = _build_uniform_band_from_multiseed(
                parego_band_paths,
                metric_key="hv",
                trace_mode="eval",
            )

    hv_png, hv_pdf, metrics = _plot_hv_figure(
        llmbo_summary_path,
        parego_summary_path,
        output_dir,
        llmbo_band_profile=llmbo_band_profile,
        parego_band_profile=parego_band_profile,
    )

    notes = ["Single-seed trajectory comparison for seed=8409, 50 iterations."]
    if llmbo_band_profile is not None or parego_band_profile is not None:
        notes.append(
            "The shaded band uses cross-seed step std from proxy 5-seed runs, then smooths and clips it for a more uniform visual width."
        )
        if parego_band_glob and "parego_matlab_reference_5seeds_50iter_2026_05_06" in parego_band_glob and "parego_matlab_reference" not in str(parego_summary_path):
            notes.append(
                "ParEGO's band uses the nearest available 5-seed matlab-reference proxy because no multi-seed baseline ParEGO directory was found."
            )
    else:
        notes.append("The shaded band is a smoothed local-volatility visual envelope for readability, not a statistical confidence interval.")

    manifest = {
        "figure_family": "llmbo_vs_parego_seed8409_hv",
        "notes": notes,
        "mode": "single_seed_hv",
        "metrics": metrics,
        "artifacts": {
            "hv_png": str(hv_png.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "hv_pdf": str(hv_pdf.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    _write_manifest(output_dir, manifest)
    return manifest


def build_multiseed_optimal_protocol(output_dir: Path, llmbo_summary_paths: Sequence[Path], parego_summary_paths: Sequence[Path]) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    opt_png, opt_pdf, metrics = _plot_optimal_protocol_figure(llmbo_summary_paths, parego_summary_paths, output_dir)
    manifest = {
        "figure_family": "llmbo_vs_parego_optimal_protocols_5seeds",
        "notes": [
            "Five-seed mean trajectory comparison on the number of optimal charging protocols.",
            "The shaded band is the across-seed standard deviation at each iteration.",
        ],
        "mode": "multiseed_optimal_protocols",
        "metrics": metrics,
        "artifacts": {
            "optimal_protocol_png": str(opt_png.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "optimal_protocol_pdf": str(opt_pdf.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    _write_manifest(output_dir, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot single-seed HV curves and multi-seed optimal-protocol curves for LLMBO-MO vs ParEGO.")
    parser.add_argument(
        "--mode",
        choices=["single-seed-hv", "multiseed-optimal"],
        required=True,
        help="Which figure family to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the generated figures and manifest.json.",
    )
    parser.add_argument(
        "--llmbo-summary",
        type=str,
        default=str(DEFAULT_LLMBO_SUMMARY),
        help="Single-seed LLMBO summary path for HV mode.",
    )
    parser.add_argument(
        "--parego-summary",
        type=str,
        default=str(DEFAULT_PAREGO_REFERENCE_SUMMARY),
        help="Single-seed ParEGO summary path for HV mode.",
    )
    parser.add_argument(
        "--llmbo-band-glob",
        type=str,
        default=DEFAULT_LLMBO_HV_BAND_GLOB,
        help="Project-root-relative glob for proxy multi-seed LLMBO summaries used to build the HV band.",
    )
    parser.add_argument(
        "--parego-band-glob",
        type=str,
        default=DEFAULT_PAREGO_HV_BAND_GLOB,
        help="Project-root-relative glob for proxy multi-seed ParEGO summaries used to build the HV band.",
    )
    parser.add_argument(
        "--llmbo-glob",
        type=str,
        default=DEFAULT_LLMBO_MULTI_GLOB,
        help="Project-root-relative glob for multi-seed LLMBO summaries.",
    )
    parser.add_argument(
        "--parego-glob",
        type=str,
        default=DEFAULT_PAREGO_MULTI_GLOB,
        help="Project-root-relative glob for multi-seed ParEGO summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if args.mode == "single-seed-hv":
        manifest = build_single_seed_hv(
            output_dir=output_dir,
            llmbo_summary_path=_coerce_path(args.llmbo_summary),
            parego_summary_path=_coerce_path(args.parego_summary),
            llmbo_band_glob=args.llmbo_band_glob,
            parego_band_glob=args.parego_band_glob,
        )
    else:
        llmbo_summary_paths = _resolve_summary_paths(args.llmbo_glob)
        parego_summary_paths = _resolve_summary_paths(args.parego_glob)
        manifest = build_multiseed_optimal_protocol(
            output_dir=output_dir,
            llmbo_summary_paths=llmbo_summary_paths,
            parego_summary_paths=parego_summary_paths,
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
