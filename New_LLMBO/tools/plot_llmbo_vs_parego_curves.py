from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LLMBO_SUMMARY = (
    PROJECT_ROOT
    / "optimized_experiments"
    / "region_lift_force_pool_local_sweep_seed8409_2026_05_01"
    / "seed8409"
    / "wider_active16_ext32"
    / "summary.json"
)

PAREGO_SUMMARY = (
    PROJECT_ROOT
    / "optimized_experiments"
    / "parego_seed8409_50iter_2026_05_02"
    / "seed8409"
    / "parego_baseline"
    / "summary.json"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_runs" / "llmbo_vs_parego_seed8409_figures_2026_05_03"


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


def _extract_trace(summary: Dict[str, Any]) -> Dict[str, np.ndarray]:
    hv_trace = summary.get("hv_trace") or []
    if not hv_trace:
        raise ValueError("summary.json missing hv_trace")

    x = np.asarray([int(item.get("n_total", item.get("eval_index", idx + 1))) for idx, item in enumerate(hv_trace)], dtype=int)
    y_hv = np.asarray(
        [float(item.get("display_hv", item.get("hypervolume", 0.0))) for item in hv_trace],
        dtype=float,
    )
    y_pareto = np.asarray([int(item.get("pareto_size", 0)) for item in hv_trace], dtype=int)
    return {"x": x, "hv": y_hv, "pareto": y_pareto}


def _plot_with_band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    label: str,
    band_half_width: float,
) -> None:
    # Single-seed comparison: this band is a visual envelope, not a statistical CI.
    lower = np.asarray(y, dtype=float) - float(band_half_width)
    upper = np.asarray(y, dtype=float) + float(band_half_width)
    ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0.0)
    ax.plot(x, y, color=color, lw=2.6, alpha=1.0, solid_capstyle="round", label=label)


def _plot_hv_figure(
    llmbo: Dict[str, np.ndarray],
    parego: Dict[str, np.ndarray],
    output_dir: Path,
) -> Tuple[Path, Path]:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    _plot_with_band(ax, parego["x"], parego["hv"], color="#4e8fb5", label="ParEGO", band_half_width=0.0045)
    _plot_with_band(ax, llmbo["x"], llmbo["hv"], color="#c85d6b", label="LLMBO-MO", band_half_width=0.0035)

    y_all = np.concatenate([llmbo["hv"], parego["hv"]])
    y_min = max(0.0, float(np.floor((y_all.min() - 0.01) * 100.0) / 100.0))
    y_max = min(1.0, float(np.ceil((y_all.max() + 0.005) * 100.0) / 100.0))
    if y_max <= y_min:
        y_max = y_min + 0.05

    ax.set_xlim(int(min(llmbo["x"].min(), parego["x"].min())), int(max(llmbo["x"].max(), parego["x"].max())))
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("HV")
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#777777")
    ax.text(0.5, -0.17, "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=18)

    fig.tight_layout()
    png_path = output_dir / "llmbo_vs_parego_hv_curve.png"
    pdf_path = output_dir / "llmbo_vs_parego_hv_curve.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _plot_pareto_figure(
    llmbo: Dict[str, np.ndarray],
    parego: Dict[str, np.ndarray],
    output_dir: Path,
) -> Tuple[Path, Path]:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    _plot_with_band(ax, parego["x"], parego["pareto"], color="#6aae5c", label="ParEGO", band_half_width=3.0)
    _plot_with_band(ax, llmbo["x"], llmbo["pareto"], color="#3f78a0", label="LLMBO-MO", band_half_width=3.0)

    y_all = np.concatenate([llmbo["pareto"], parego["pareto"]]).astype(float)
    y_min = max(0.0, float(np.floor((y_all.min() - 1.0) / 5.0) * 5.0))
    y_max = float(np.ceil((y_all.max() + 2.0) / 5.0) * 5.0)
    if y_max <= y_min:
        y_max = y_min + 5.0

    ax.set_xlim(int(min(llmbo["x"].min(), parego["x"].min())), int(max(llmbo["x"].max(), parego["x"].max())))
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Number of optimal charging protocols")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#777777")
    ax.text(0.5, -0.17, "(b)", transform=ax.transAxes, ha="center", va="top", fontsize=18)

    fig.tight_layout()
    png_path = output_dir / "llmbo_vs_parego_pareto_curve.png"
    pdf_path = output_dir / "llmbo_vs_parego_pareto_curve.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def build_figures(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    llmbo_summary = _load_json(LLMBO_SUMMARY)
    parego_summary = _load_json(PAREGO_SUMMARY)

    llmbo_trace = _extract_trace(llmbo_summary)
    parego_trace = _extract_trace(parego_summary)

    hv_png, hv_pdf = _plot_hv_figure(llmbo_trace, parego_trace, output_dir)
    pareto_png, pareto_pdf = _plot_pareto_figure(llmbo_trace, parego_trace, output_dir)

    manifest = {
        "figure_family": "llmbo_vs_parego_seed8409",
        "notes": [
            "Single-seed trajectory comparison for seed=8409, 50 iterations.",
            "HV axis uses display_hv from hv_trace for paper-style [0,1] plotting.",
            "A light fill_between visual band is used for style; it is not a statistical confidence interval.",
        ],
        "sources": {
            "llmbo_summary": str(LLMBO_SUMMARY.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "parego_summary": str(PAREGO_SUMMARY.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "metrics": {
            "llmbo_mo": {
                "display_hv_final": float(llmbo_summary.get("display_hv", 0.0)),
                "canonical_hv_final": float(llmbo_summary.get("canonical_hv", 0.0)),
                "pareto_size_final": int(llmbo_summary.get("pareto_size", 0)),
                "n_total": int(llmbo_summary.get("n_total", 0)),
            },
            "parego": {
                "display_hv_final": float(parego_summary.get("display_hv", 0.0)),
                "canonical_hv_final": float(parego_summary.get("canonical_hv", 0.0)),
                "pareto_size_final": int(parego_summary.get("pareto_size", 0)),
                "n_total": int(parego_summary.get("n_total", 0)),
            },
        },
        "artifacts": {
            "hv_png": str(hv_png.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "hv_pdf": str(hv_pdf.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "pareto_png": str(pareto_png.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "pareto_pdf": str(pareto_pdf.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LLMBO-MO vs ParEGO curves in a paper-like style.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated PNG/PDF figures and manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_figures(args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
