"""
plot_parego_ecker_oregan.py — ParEGO在Ecker2015和ORegan2022上的HV表现对比
=============================================================================
绘制两张图：
1. ParEGO + Ecker2015 的HV收敛曲线（蓝色实线 + 阴影带）
2. ParEGO + ORegan2022 的HV收敛曲线（蓝色实线 + 阴影带）

Usage:
    python Compare_Exp/plot_parego_ecker_oregan.py
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

# 颜色定义 - ParEGO使用蓝色
PAREGO_COLOR = "#4e8fb5"


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

    目录结构：
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


def plot_single_parego(
    data: Dict[str, Any],
    param_set: str,
    output_path: Path,
    max_evals: int = 50,
) -> None:
    """
    绘制单张ParEGO HV收敛曲线图（蓝色实线 + 阴影带）

    Args:
        data: 包含x, mean, std的数据字典
        param_set: 参数集名称（用于图例）
        output_path: 输出文件路径
        max_evals: 最大评估次数
    """
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    x = data["x"]
    mean = data["mean"]
    std = data["std"]

    # 绘制阴影带（mean ± std）
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=PAREGO_COLOR,
        alpha=0.15,
        linewidth=0.0,
    )

    # 绘制蓝色实线
    ax.plot(
        x,
        mean,
        color=PAREGO_COLOR,
        lw=2.5,
        label=f"ParEGO ({param_set})",
        marker="s",
        markevery=5,
        markersize=6,
        linestyle="-",  # 实线
    )

    # 设置坐标轴
    ax.set_xlabel("Number of evaluations", fontsize=18)
    ax.set_ylabel("Hypervolume", fontsize=18)
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # 设置y轴范围
    y_min = max(0.0, (mean - std).min() - 0.02)
    y_max = min(1.0, (mean + std).max() + 0.02)
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
        description="Plot ParEGO performance on Ecker2015 and ORegan2022"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Compare_Exp/images"),
        help="Output directory for images",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=50,
        help="Maximum number of evaluations to plot",
    )

    args = parser.parse_args()

    # 定义两个实验的数据源
    experiments = [
        {
            "name": "Ecker2015",
            "variant": "parego_matlab_reference",
            "param_set": "Ecker2015",
            "root": Path("optimized_experiments/parego_matlab_reference_Ecker2015_5seeds_56evals_2026_05_09"),
            "output": args.output_dir / "parego_ecker2015_hv_50iter",
        },
        {
            "name": "ORegan2022",
            "variant": "parego_matlab_reference",
            "param_set": "ORegan2022",
            "root": Path("optimized_experiments/parego_matlab_reference_ORegan2022_5seeds_56evals_2026_05_09"),
            "output": args.output_dir / "parego_oregan2022_hv_50iter",
        },
    ]

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Processing: {exp['name']}")
        print(f"{'='*60}")

        # 检查数据是否存在
        if not exp["root"].exists():
            print(f"Warning: Data directory not found: {exp['root']}")
            print(f"Skipping {exp['name']}...")
            continue

        # 加载多seed数据
        data = _load_multiseed_data(
            exp["root"],
            exp["variant"],
            exp["param_set"],
            n_seeds=5,
            max_evals=args.max_evals,
        )

        if data is None:
            print(f"Warning: No data found for {exp['name']}")
            continue

        print(f"Loaded {data['n_runs']} runs")
        print(f"Final HV: {data['mean'][-1]:.4f} ± {data['std'][-1]:.4f}")

        # 绘制图表
        plot_single_parego(
            data,
            exp["name"],
            exp["output"],
            max_evals=args.max_evals,
        )

    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
