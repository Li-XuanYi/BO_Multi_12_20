"""
plot_comparison.py — 三方对比图 (NSGA-II / ParEGO / LLAMBO-MO)
================================================================
沿用 plot_llmbo_vs_parego_curves.py 的绘图风格。

数据源:
  - NSGA-II:  5-seed mean±std (nsga2_5seeds_56evals_2026_05_07)
  - ParEGO:   单 seed=8409 (parego_matlab_reference_seed8409_50iter_2026_05_05)
  - LLAMBO-MO: 单 seed=8409 (region_lift_force_pool_local_sweep / wider_active16_ext32)

生成 4 张图 (PDF + PNG):
  1. HV 收敛曲线 (eval-indexed, canonical_hv)
  2. 最优协议数量 vs evaluations
  3. Pareto 前沿 3D 散点
  4. Pareto 前沿 2D 投影

Usage:
    pixi run python baselines/plot_comparison.py
    pixi run python baselines/plot_comparison.py --output figs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 颜色 ──
NSGA2_COLOR = "#e67e22"
PAREGO_COLOR = "#4e8fb5"
LLMBO_COLOR = "#c85d6b"

LABELS = {"nsga2": "NSGA-II", "parego": "ParEGO", "llmbo": "LLAMBO-MO"}
MARKERS = {"nsga2": "v", "parego": "s", "llmbo": "o"}
COLOR_MAP = {"nsga2": NSGA2_COLOR, "parego": PAREGO_COLOR, "llmbo": LLMBO_COLOR}

# ── 数据路径 ──
# NSGA-II: 5-seed
NSGA2_ROOT = PROJECT_ROOT / "optimized_experiments" / "nsga2_5seeds_56evals_2026_05_07"

# ParEGO: seed=8409
PAREGO_SINGLE_DIR = (
    PROJECT_ROOT / "optimized_experiments"
    / "parego_matlab_reference_seed8409_50iter_2026_05_05"
    / "seed8409" / "parego_matlab_reference"
)

# LLAMBO-MO: seed=8409, wider_active16_ext32
LLMBO_SINGLE_DIR = (
    PROJECT_ROOT / "optimized_experiments"
    / "region_lift_force_pool_local_sweep_seed8409_2026_05_01"
    / "seed8409" / "wider_active16_ext32"
)


# ═══════════════════════════════════════════════════════════════
#  样式 (沿用 plot_llmbo_vs_parego_curves.py)
# ═══════════════════════════════════════════════════════════════
def _configure_plot_style() -> None:
    plt.rcParams.update({
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
    })


def _normalized_window(window: int) -> int:
    width = max(1, int(window))
    if width % 2 == 0:
        width += 1
    return width


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    width = _normalized_window(window)
    if width <= 1 or arr.size == 1:
        return arr.copy()
    pad = width // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(padded, kernel, mode="valid")


# ═══════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════
def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_eval_trace(summary: Dict) -> Dict[str, np.ndarray]:
    hv_trace = summary.get("hv_trace") or []
    if not hv_trace:
        return {"x": np.array([]), "hv": np.array([]), "pareto": np.array([])}
    x = np.array([int(t.get("eval_index", i + 1)) for i, t in enumerate(hv_trace)])
    hv = np.array([float(t.get("canonical_hv", 0)) for t in hv_trace])
    pareto = np.array([int(t.get("pareto_size", 0)) for t in hv_trace])
    return {"x": x, "hv": hv, "pareto": pareto}


def _load_single_trace(data_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    summary_path = data_dir / "summary.json"
    if not summary_path.exists():
        return None
    return _extract_eval_trace(_load_json(summary_path))


def _load_single_pareto(data_dir: Path) -> Optional[np.ndarray]:
    pf_path = data_dir / "pareto_front.json"
    if not pf_path.exists():
        return None
    pf = json.loads(pf_path.read_text())
    if not pf:
        return None
    return np.array([p["objectives"] for p in pf])


MAX_EVALS = 56  # 统一评估预算


def _load_multiseed_stacked(
    root: Path, variant: str, *, metric_key: str = "hv", n_seeds: int = 5,
    max_evals: int = MAX_EVALS,
) -> Optional[Dict[str, Any]]:
    """Load multi-seed data, truncate to max_evals, return mean ± std."""
    traces = []
    for s in range(n_seeds):
        d = root / f"seed{s}" / variant
        trace = _load_single_trace(d)
        if trace and len(trace["x"]) > 0:
            # Truncate to max_evals
            mask = trace["x"] <= max_evals
            traces.append({
                "x": trace["x"][mask],
                "hv": trace["hv"][mask],
                "pareto": trace["pareto"][mask],
            })
    if not traces:
        return None
    min_len = min(len(t["x"]) for t in traces)
    x_ref = traces[0]["x"][:min_len]
    arr = np.vstack([t[metric_key][:min_len] for t in traces])
    return {
        "x": x_ref, "mean": arr.mean(axis=0), "std": arr.std(axis=0),
        "n_runs": int(arr.shape[0]),
    }


def _load_nsga2_stacked(metric_key: str = "hv") -> Optional[Dict[str, Any]]:
    return _load_multiseed_stacked(NSGA2_ROOT, "nsga2", metric_key=metric_key)


# ── 统一 5-seed mean±std band (沿用原脚本 _plot_multiseed_band) ──
def _plot_band(
    ax: plt.Axes, x: np.ndarray, mean: np.ndarray, std: np.ndarray, *,
    color: str, label: str, marker: str = "o", markevery: int = 7,
) -> None:
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0.0)
    ax.plot(x, mean, color=color, lw=2.8, alpha=1.0, solid_capstyle="round",
            label=label, marker=marker, markevery=markevery, markersize=6)


# ═══════════════════════════════════════════════════════════════
#  Figure 1: HV 收敛曲线
# ═══════════════════════════════════════════════════════════════
def _plot_single_trace(
    ax: plt.Axes, trace: Dict[str, np.ndarray], *,
    color: str, label: str, marker: str = "o", markevery: int = 7,
    smooth_window: int = 3,
) -> None:
    x, hv = trace["x"], trace["hv"]
    hv_smooth = _smooth_1d(hv, smooth_window) if len(hv) > smooth_window else hv
    ax.plot(x, hv_smooth, color=color, lw=2.8, alpha=1.0, solid_capstyle="round",
            label=label, marker=marker, markevery=markevery, markersize=6)


def _plot_single_pareto_trace(
    ax: plt.Axes, trace: Dict[str, np.ndarray], *,
    color: str, label: str, marker: str = "o", markevery: int = 7,
    smooth_window: int = 3,
) -> None:
    x, vals = trace["x"], trace["pareto"]
    vals_smooth = _smooth_1d(vals, smooth_window) if len(vals) > smooth_window else vals
    ax.plot(x, vals_smooth, color=color, lw=2.8, alpha=1.0, solid_capstyle="round",
            label=label, marker=marker, markevery=markevery, markersize=6)


def plot_hv_convergence(output_dir: Path):
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    all_y = []

    # ParEGO (seed8409)
    parego_trace = _load_single_trace(PAREGO_SINGLE_DIR)
    if parego_trace and len(parego_trace["x"]) > 0:
        _plot_single_trace(ax, parego_trace, color=PAREGO_COLOR, label="ParEGO", marker="s")
        all_y.append(parego_trace["hv"])

    # LLAMBO-MO (seed8409)
    llmbo_trace = _load_single_trace(LLMBO_SINGLE_DIR)
    if llmbo_trace and len(llmbo_trace["x"]) > 0:
        _plot_single_trace(ax, llmbo_trace, color=LLMBO_COLOR, label="LLAMBO-MO", marker="o")
        all_y.append(llmbo_trace["hv"])

    # NSGA-II (5-seed mean±std)
    nsga2 = _load_nsga2_stacked("hv")
    if nsga2:
        _plot_band(ax, nsga2["x"], nsga2["mean"], nsga2["std"],
                   color=NSGA2_COLOR, label="NSGA-II", marker="v")
        all_y.extend([nsga2["mean"] - nsga2["std"], nsga2["mean"] + nsga2["std"]])

    if all_y:
        y_all = np.concatenate(all_y)
        y_min = max(0.0, float(np.floor((y_all.min() - 0.01) * 50.0) / 50.0))
        y_max = min(1.0, float(np.ceil((y_all.max() + 0.005) * 50.0) / 50.0))
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Canonical HV")
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="#777777", handlelength=2.6, handletextpad=0.6)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = output_dir / f"hv_convergence_3way.{ext}"
        fig.savefig(p, dpi=240, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Figure 2: 最优协议数量 vs evaluations
# ═══════════════════════════════════════════════════════════════
def plot_optimal_protocols(output_dir: Path):
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    all_y = []

    # ParEGO (seed8409)
    parego_trace = _load_single_trace(PAREGO_SINGLE_DIR)
    if parego_trace and len(parego_trace["x"]) > 0:
        _plot_single_pareto_trace(ax, parego_trace, color=PAREGO_COLOR, label="ParEGO", marker="s")
        all_y.append(parego_trace["pareto"])

    # LLAMBO-MO (seed8409)
    llmbo_trace = _load_single_trace(LLMBO_SINGLE_DIR)
    if llmbo_trace and len(llmbo_trace["x"]) > 0:
        _plot_single_pareto_trace(ax, llmbo_trace, color=LLMBO_COLOR, label="LLAMBO-MO", marker="o")
        all_y.append(llmbo_trace["pareto"])

    # NSGA-II (5-seed mean±std)
    nsga2 = _load_nsga2_stacked("pareto")
    if nsga2:
        _plot_band(ax, nsga2["x"], nsga2["mean"], nsga2["std"],
                   color=NSGA2_COLOR, label="NSGA-II", marker="v")
        all_y.extend([nsga2["mean"] - nsga2["std"], nsga2["mean"] + nsga2["std"]])

    if all_y:
        y_all = np.concatenate([np.asarray(y) for y in all_y])
        y_min = max(0.0, float(np.floor((y_all.min() - 1) / 5) * 5))
        y_max = float(np.ceil((y_all.max() + 2) / 5) * 5)
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Number of optimal charging protocols")
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.legend(loc="upper left", frameon=True, fancybox=False,
              edgecolor="#777777", handlelength=2.6, handletextpad=0.6)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = output_dir / f"optimal_protocols_3way.{ext}"
        fig.savefig(p, dpi=240, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Figure 3: Pareto 前沿 3D
# ═══════════════════════════════════════════════════════════════
def plot_pareto_3d(output_dir: Path):
    _configure_plot_style()
    fig = plt.figure(figsize=(8.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    data_sources = [
        (NSGA2_COLOR, "NSGA-II", "v", NSGA2_ROOT / "seed4" / "nsga2"),
        (PAREGO_COLOR, "ParEGO", "s", PAREGO_SINGLE_DIR),
        (LLMBO_COLOR, "LLAMBO-MO", "o", LLMBO_SINGLE_DIR),
    ]
    for color, label, marker, d in data_sources:
        objs = _load_single_pareto(d)
        if objs is None:
            continue
        ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2] * 100,
                   c=color, label=label, marker=marker, s=40, alpha=0.80,
                   edgecolors="k", linewidths=0.3)

    ax.set_xlabel("Charging Time [s]", fontsize=12, labelpad=8)
    ax.set_ylabel("Peak Temp Rise [K]", fontsize=12, labelpad=8)
    ax.set_zlabel("Capacity Fade [%]", fontsize=12, labelpad=8)
    ax.legend(fontsize=12, loc="upper right")
    ax.view_init(elev=25, azim=135)
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = output_dir / f"pareto_front_3d_3way.{ext}"
        fig.savefig(p, dpi=240, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Figure 4: Pareto 前沿 2D 投影
# ═══════════════════════════════════════════════════════════════
def plot_pareto_2d(output_dir: Path):
    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5))

    data_sources = [
        (NSGA2_COLOR, "NSGA-II", "v", NSGA2_ROOT / "seed4" / "nsga2"),
        (PAREGO_COLOR, "ParEGO", "s", PAREGO_SINGLE_DIR),
        (LLMBO_COLOR, "LLAMBO-MO", "o", LLMBO_SINGLE_DIR),
    ]

    panels = [
        (0, 2, "Charging Time [s]", "Capacity Fade [%]", lambda y: y * 100, "(a)"),
        (0, 1, "Charging Time [s]", "Peak Temp Rise [K]", lambda y: y, "(b)"),
    ]

    for ax, (xi, yi, xlabel, ylabel, transform, sublabel) in zip(axes, panels):
        for color, label, marker, d in data_sources:
            objs = _load_single_pareto(d)
            if objs is None:
                continue
            ax.scatter(objs[:, xi], transform(objs[:, yi]),
                       c=color, label=label, marker=marker, s=40, alpha=0.80,
                       edgecolors="k", linewidths=0.3)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.grid(True)
        ax.legend(fontsize=12, loc="upper right", frameon=True,
                  fancybox=False, edgecolor="#777777")
        ax.text(0.5, -0.18, sublabel, transform=ax.transAxes,
                ha="center", va="top", fontsize=18)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = output_dir / f"pareto_front_2d_3way.{ext}"
        fig.savefig(p, dpi=240, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Three-way comparison: NSGA-II vs ParEGO vs LLAMBO-MO")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "figs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Generating plots → {args.output}/")

    plot_hv_convergence(args.output)
    plot_optimal_protocols(args.output)
    plot_pareto_3d(args.output)
    plot_pareto_2d(args.output)

    print("Done.")


if __name__ == "__main__":
    main()
