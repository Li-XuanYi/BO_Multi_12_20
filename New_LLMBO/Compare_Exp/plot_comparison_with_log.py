"""
plot_comparison_with_log.py — 多算法对比图（带Log处理说明）
============================================================
绘制四条曲线对比：LLMBO-MO vs ParEGO vs NSGA-II vs GA
带阴影带（mean ± std），展示50轮迭代结果

参考图例格式：
- LLMBO-MO (log10 time & aging)
- ParEGO (log10 time & aging)
- NSGA-II
- GA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 颜色定义
COLORS = {
    "llmbo": "#c85d6b",      # 红色系
    "parego": "#4e8fb5",     # 蓝色系
    "nsga2": "#e67e22",      # 橙色系
    "ga": "#2ecc71",         # 绿色系
}

LABELS = {
    "llmbo": "LLMBO-MO (log$_{10}$ time \& aging)",
    "parego": "ParEGO (log$_{10}$ time \& aging)",
    "nsga2": "NSGA-II",
    "ga": "GA",
}

MARKERS = {
    "llmbo": "o",
    "parego": "s",
    "nsga2": "v",
    "ga": "^",
}


def _configure_plot_style() -> None:
    """配置绘图样式（论文风格）"""
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


def _load_summary(data_dir: Path) -> Optional[Dict]:
    """加载summary.json文件"""
    summary_path = data_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_hv_trace(summary: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """提取HV轨迹（eval_index, canonical_hv）"""
    hv_trace = summary.get("hv_trace", [])
    if not hv_trace:
        return np.array([]), np.array([])

    x = np.array([t.get("eval_index", i + 1) for i, t in enumerate(hv_trace)])
    hv = np.array([t.get("canonical_hv", 0.0) for t in hv_trace])
    return x, hv


def _load_multiseed_data(
    root: Path,
    variant: str,
    param_set: str,
    n_seeds: int = 5,
    max_evals: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    加载多seed数据，返回mean和std

    期望目录结构：
    root/seed0/variant_param_set/summary.json
    root/seed1/variant_param_set/summary.json
    ...
    """
    traces = []
    for s in range(n_seeds):
        seed_dir = root / f"seed{s}" / f"{variant}_{param_set}"
        summary = _load_summary(seed_dir)
        if summary:
            x, hv = _extract_hv_trace(summary)
            if len(x) > 0:
                # 截断到max_evals
                mask = x <= max_evals
                traces.append({
                    "x": x[mask],
                    "hv": hv[mask],
                })

    if not traces:
        return None

    # 对齐长度（取最小）
    min_len = min(len(t["x"]) for t in traces)
    x_ref = traces[0]["x"][:min_len]

    # 堆叠计算mean和std
    hv_matrix = np.vstack([t["hv"][:min_len] for t in traces])

    return {
        "x": x_ref,
        "mean": hv_matrix.mean(axis=0),
        "std": hv_matrix.std(axis=0),
        "n_runs": len(traces),
    }


def plot_hv_convergence_with_shaded(
    data_sources: List[Tuple[str, str, str, Path, str]],
    output_path: Path,
    max_evals: int = 50,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    绘制HV收敛曲线对比（带阴影带）

    Args:
        data_sources: List of (label_key, variant, param_set, root_dir, display_name)
        output_path: 输出文件路径
        max_evals: 最大评估次数
        ylim: y轴范围
    """
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    all_y_values = []

    for label_key, variant, param_set, root, display_name in data_sources:
        data = _load_multiseed_data(root, variant, param_set, n_seeds=5, max_evals=max_evals)

        if data is None:
            print(f"Warning: No data found for {display_name}")
            continue

        x = data["x"]
        mean = data["mean"]
        std = data["std"]
        color = COLORS.get(label_key, "#333333")
        marker = MARKERS.get(label_key, "o")

        # 绘制阴影带（mean ± std）
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.15,
            linewidth=0.0,
        )

        # 绘制mean曲线
        ax.plot(
            x,
            mean,
            color=color,
            lw=2.5,
            label=display_name,
            marker=marker,
            markevery=5,
            markersize=6,
        )

        all_y_values.extend([(mean - std).min(), (mean + std).max()])

    # 设置坐标轴
    ax.set_xlabel("Number of evaluations", fontsize=18)
    ax.set_ylabel("Hypervolume", fontsize=18)
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # 设置y轴范围
    if ylim:
        ax.set_ylim(ylim)
    elif all_y_values:
        y_min = max(0.0, min(all_y_values) - 0.02)
        y_max = min(1.0, max(all_y_values) + 0.02)
        ax.set_ylim(y_min, y_max)

    # 图例
    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        handlelength=2.5,
        handletextpad=0.6,
    )

    fig.tight_layout()

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):
        p = output_path.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot HV convergence comparison with log-transform indication"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Compare_Exp/images/hv_comparison_50iter.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=50,
        help="Maximum number of evaluations to plot",
    )

    args = parser.parse_args()

    # 配置数据源
    # 格式: (label_key, variant, param_set, root_dir, display_name)
    data_sources = [
        # LLMBO-MO Chen2020
        (
            "llmbo",
            "warmstart_plain_ei",
            "Chen2020",
            Path("optimized_experiments/llmbo_chen2020_5seeds_50iter"),
            "LLMBO-MO (log$_{10}$ time \& aging)",
        ),
        # ParEGO Chen2020
        (
            "parego",
            "parego_matlab_reference",
            "Chen2020",
            Path("optimized_experiments/parego_matlab_reference_Chen2020_5seeds_50evals_2026_05_09"),
            "ParEGO (log$_{10}$ time \& aging)",
        ),
        # ParEGO Ecker2015
        (
            "parego",
            "parego_matlab_reference",
            "Ecker2015",
            Path("optimized_experiments/parego_matlab_reference_Ecker2015_5seeds_50evals_2026_05_09"),
            "ParEGO-Ecker (log$_{10}$ time \& aging)",
        ),
        # ParEGO ORegan2022
        (
            "parego",
            "parego_matlab_reference",
            "ORegan2022",
            Path("optimized_experiments/parego_matlab_reference_ORegan2022_5seeds_50evals_2026_05_09"),
            "ParEGO-ORegan (log$_{10}$ time \& aging)",
        ),
    ]

    # 只使用存在的目录
    available_sources = [
        src for src in data_sources
        if src[3].exists() or (src[3].parent / f"{src[3].name}").exists()
    ]

    if not available_sources:
        print("Error: No experiment data found!")
        print("Please run experiments first or update the paths.")
        sys.exit(1)

    print(f"Found {len(available_sources)} data sources:")
    for src in available_sources:
        print(f"  - {src[4]}: {src[3]}")

    plot_hv_convergence_with_shaded(
        available_sources,
        args.output,
        max_evals=args.max_evals,
    )

    print(f"\nPlot saved to: {args.output}")


if __name__ == "__main__":
    main()
