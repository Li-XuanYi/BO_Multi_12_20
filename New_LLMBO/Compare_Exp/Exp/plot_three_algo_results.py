"""
plot_three_algo_results.py — 三算法对比结果绘图脚本
=========================================================
从 run_three_algo_comparison.py 生成的实验结果中读取数据，绘制：
1. HV 收敛曲线 (eval-indexed, canonical_hv)
2. 最优协议数量 vs evaluations
3. Pareto 前沿 3D 散点
4. Pareto 前沿 2D 投影

Usage:
    python Compare_Exp/Exp/plot_three_algo_results.py --exp-dir Compare_Exp/experiment_records/three_algo_comparison_5seeds_56evals_YYYY_MM_DD
    python Compare_Exp/Exp/plot_three_algo_results.py --exp-dir <path> --output figs/
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════
# 样式配置
# ═══════════════════════════════════════════════════════════════════════════

NSGA2_COLOR = "#e67e22"
PAREGO_COLOR = "#4e8fb5"
LLMBO_COLOR = "#c85d6b"

LABELS = {"nsga2": "NSGA-II", "parego": "ParEGO", "llmbo": "LLAMBO-MO"}
MARKERS = {"nsga2": "v", "parego": "s", "llmbo": "o"}
COLOR_MAP = {"nsga2": NSGA2_COLOR, "parego": PAREGO_COLOR, "llmbo": LLMBO_COLOR}


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


# ═══════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════

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


def _load_multiseed_stacked(
    exp_root: Path,
    algorithm: str,
    seeds: List[int],
    *,
    metric_key: str = "hv",
    max_evals: int = 56,
) -> Optional[Dict[str, Any]]:
    """Load multi-seed data, truncate to max_evals, return mean ± std."""
    traces = []
    for seed in seeds:
        if algorithm == "nsga2":
            d = exp_root / f"seed{seed}" / "nsga2"
        elif algorithm == "parego":
            d = exp_root / f"seed{seed}" / "parego_matlab_reference"
        elif algorithm == "llmbo":
            d = exp_root / f"seed{seed}" / "llmbo_mo"
        else:
            continue

        trace = _load_single_trace(d)
        if trace and len(trace["x"]) > 0:
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
        "x": x_ref,
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "n_runs": int(arr.shape[0]),
    }


def _plot_band(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    color: str,
    label: str,
    marker: str = "o",
    markevery: int = 7,
) -> None:
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0.0)
    ax.plot(x, mean, color=color, lw=2.8, alpha=1.0, solid_capstyle="round",
            label=label, marker=marker, markevery=markevery, markersize=6)


# ═══════════════════════════════════════════════════════════════════════════
# 图表生成
# ═══════════════════════════════════════════════════════════════════════════

def plot_hv_convergence(exp_root: Path, seeds: List[int], output_dir: Path):
    """Figure 1: HV 收敛曲线"""
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    all_y = []

    # ParEGO
    parego = _load_multiseed_stacked(exp_root, "parego", seeds, metric_key="hv")
    if parego:
        _plot_band(ax, parego["x"], parego["mean"], parego["std"],
                   color=PAREGO_COLOR, label="ParEGO", marker="s")
        all_y.extend([parego["mean"] - parego["std"], parego["mean"] + parego["std"]])

    # LLAMBO-MO
    llmbo = _load_multiseed_stacked(exp_root, "llmbo", seeds, metric_key="hv")
    if llmbo:
        _plot_band(ax, llmbo["x"], llmbo["mean"], llmbo["std"],
                   color=LLMBO_COLOR, label="LLAMBO-MO", marker="o")
        all_y.extend([llmbo["mean"] - llmbo["std"], llmbo["mean"] + llmbo["std"]])

    # NSGA-II
    nsga2 = _load_multiseed_stacked(exp_root, "nsga2", seeds, metric_key="hv")
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


def plot_optimal_protocols(exp_root: Path, seeds: List[int], output_dir: Path):
    """Figure 2: 最优协议数量 vs evaluations"""
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    all_y = []

    parego = _load_multiseed_stacked(exp_root, "parego", seeds, metric_key="pareto")
    if parego:
        _plot_band(ax, parego["x"], parego["mean"], parego["std"],
                   color=PAREGO_COLOR, label="ParEGO", marker="s")
        all_y.extend([parego["mean"] - parego["std"], parego["mean"] + parego["std"]])

    llmbo = _load_multiseed_stacked(exp_root, "llmbo", seeds, metric_key="pareto")
    if llmbo:
        _plot_band(ax, llmbo["x"], llmbo["mean"], llmbo["std"],
                   color=LLMBO_COLOR, label="LLAMBO-MO", marker="o")
        all_y.extend([llmbo["mean"] - llmbo["std"], llmbo["mean"] + llmbo["std"]])

    nsga2 = _load_multiseed_stacked(exp_root, "nsga2", seeds, metric_key="pareto")
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


def plot_pareto_3d(exp_root: Path, seeds: List[int], output_dir: Path):
    """Figure 3: Pareto 前沿 3D 散点"""
    _configure_plot_style()
    fig = plt.figure(figsize=(8.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    last_seed = seeds[-1]

    data_sources = [
        (NSGA2_COLOR, "NSGA-II", "v", exp_root / f"seed{last_seed}" / "nsga2"),
        (PAREGO_COLOR, "ParEGO", "s", exp_root / f"seed{last_seed}" / "parego_matlab_reference"),
        (LLMBO_COLOR, "LLAMBO-MO", "o", exp_root / f"seed{last_seed}" / "llmbo_mo"),
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


def plot_pareto_2d(exp_root: Path, seeds: List[int], output_dir: Path):
    """Figure 4: Pareto 前沿 2D 投影"""
    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5))

    last_seed = seeds[-1]

    data_sources = [
        (NSGA2_COLOR, "NSGA-II", "v", exp_root / f"seed{last_seed}" / "nsga2"),
        (PAREGO_COLOR, "ParEGO", "s", exp_root / f"seed{last_seed}" / "parego_matlab_reference"),
        (LLMBO_COLOR, "LLAMBO-MO", "o", exp_root / f"seed{last_seed}" / "llmbo_mo"),
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


# ═══════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot three-algorithm comparison results"
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Experiment directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <exp-dir>/figures)"
    )
    args = parser.parse_args()

    if not args.exp_dir.exists():
        print(f"Error: Experiment directory not found: {args.exp_dir}")
        sys.exit(1)

    report_path = args.exp_dir / "comparison_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        seeds = report.get("config", {}).get("seeds", [389, 822, 2323, 4097, 4304])
    else:
        seeds = []
        for d in args.exp_dir.iterdir():
            if d.is_dir() and d.name.startswith("seed"):
                try:
                    seed = int(d.name[4:])
                    seeds.append(seed)
                except ValueError:
                    pass
        seeds = sorted(seeds) if seeds else [389, 822, 2323, 4097, 4304]

    # 默认输出到 Compare_Exp/images/<experiment_name>/
    if args.output is None:
        exp_name = args.exp_dir.name
        args.output = PROJECT_ROOT / "Compare_Exp" / "images" / exp_name
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {args.exp_dir}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {args.output}")
    print("\nGenerating plots...")

    plot_hv_convergence(args.exp_dir, seeds, args.output)
    plot_optimal_protocols(args.exp_dir, seeds, args.output)
    plot_pareto_3d(args.exp_dir, seeds, args.output)
    plot_pareto_2d(args.exp_dir, seeds, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
