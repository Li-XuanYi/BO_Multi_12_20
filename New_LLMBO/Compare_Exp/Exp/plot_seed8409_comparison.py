"""
plot_seed8409_comparison.py — Seed 8409 LLMBO-MO vs ParEGO 对比图
====================================================================
展示seed=8409时LLMBO-MO (wider_active16_ext32) 与 ParEGO (matlab_reference) 的对比
- LLMBO-MO: canonical HV = 0.3848
- ParEGO: canonical HV = 0.3523
- 优势差值: 0.0325

Usage:
    python Compare_Exp/Exp/plot_seed8409_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 颜色配置
PAREGO_COLOR = "#4e8fb5"
LLMBO_COLOR = "#c85d6b"


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


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_hv_trace(summary_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Extract HV trace from summary.json"""
    if not summary_path.exists():
        return None

    summary = _load_json(summary_path)
    hv_trace = summary.get("hv_trace", [])
    if not hv_trace:
        return None

    x = np.array([int(t.get("eval_index", i + 1)) for i, t in enumerate(hv_trace)])
    hv = np.array([float(t.get("canonical_hv", 0)) for t in hv_trace])

    return {"x": x, "hv": hv}


def plot_hv_convergence_comparison():
    """Plot HV convergence comparison for seed 8409"""
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    # 数据路径
    exp_dir = PROJECT_ROOT / "Compare_Exp" / "experiment_records" / "seed8409_llmbo_vs_parego_50iter"

    llmbo_dir = exp_dir / "seed8409" / "llmbo_mo"
    parego_dir = exp_dir / "seed8409" / "parego_matlab_reference"

    # 提取HV轨迹
    llmbo_trace = _extract_hv_trace(llmbo_dir / "summary.json")
    parego_trace = _extract_hv_trace(parego_dir / "summary.json")

    all_y = []

    # 绘制ParEGO
    if parego_trace:
        ax.plot(parego_trace["x"], parego_trace["hv"],
                color=PAREGO_COLOR, lw=2.8, alpha=1.0, solid_capstyle="round",
                label="ParEGO (matlab_reference)", marker="s", markevery=7, markersize=6)
        all_y.extend(parego_trace["hv"])

        # 标注最终值
        final_x = parego_trace["x"][-1]
        final_hv = parego_trace["hv"][-1]
        ax.annotate(f'{final_hv:.4f}',
                    xy=(final_x, final_hv),
                    xytext=(final_x - 8, final_hv - 0.015),
                    fontsize=12,
                    color=PAREGO_COLOR,
                    fontweight='bold')

    # 绘制LLMBO-MO
    if llmbo_trace:
        ax.plot(llmbo_trace["x"], llmbo_trace["hv"],
                color=LLMBO_COLOR, lw=2.8, alpha=1.0, solid_capstyle="round",
                label="LLMBO-MO (wider_active16_ext32)", marker="o", markevery=7, markersize=6)
        all_y.extend(llmbo_trace["hv"])

        # 标注最终值
        final_x = llmbo_trace["x"][-1]
        final_hv = llmbo_trace["hv"][-1]
        ax.annotate(f'{final_hv:.4f}',
                    xy=(final_x, final_hv),
                    xytext=(final_x - 8, final_hv + 0.01),
                    fontsize=12,
                    color=LLMBO_COLOR,
                    fontweight='bold')

    # 设置Y轴范围
    if all_y:
        y_all = np.array(all_y)
        y_min = max(0.0, float(np.floor((y_all.min() - 0.02) * 50.0) / 50.0))
        y_max = min(1.0, float(np.ceil((y_all.max() + 0.02) * 50.0) / 50.0))
        ax.set_ylim(y_min, y_max)

    # 添加优势区域标注
    if llmbo_trace and parego_trace:
        llmbo_final = llmbo_trace["hv"][-1]
        parego_final = parego_trace["hv"][-1]
        diff = llmbo_final - parego_final

        # 在图表下方添加差值说明
        ax.text(0.5, -0.15, f'LLMBO-MO advantage: +{diff:.4f} ({diff/llmbo_final*100:.1f}%)',
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=13,
                color='#2c3e50',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Canonical Hypervolume")
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="#777777", handlelength=2.6, handletextpad=0.6)

    # 添加标题
    ax.set_title("Seed 8409: LLMBO-MO vs ParEGO (50 iterations)", fontsize=16, pad=15)

    fig.tight_layout()

    # 保存到 Compare_Exp/images
    output_dir = PROJECT_ROOT / "Compare_Exp" / "images" / "seed8409_llmbo_vs_parego"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):
        p = output_dir / f"hv_convergence_seed8409_comparison.{ext}"
        fig.savefig(p, dpi=240, bbox_inches="tight")
        print(f"  Saved: {p}")

    plt.close(fig)

    # 打印对比信息
    print("\n" + "=" * 60)
    print("Seed 8409 Comparison Results")
    print("=" * 60)
    if llmbo_trace and parego_trace:
        llmbo_final = llmbo_trace["hv"][-1]
        parego_final = parego_trace["hv"][-1]
        print(f"  LLMBO-MO (wider_active16_ext32):  {llmbo_final:.6f}")
        print(f"  ParEGO (matlab_reference):        {parego_final:.6f}")
        print(f"  Difference:                       +{llmbo_final - parego_final:.6f}")
        print(f"  LLMBO-MO advantage:               {(llmbo_final - parego_final)/parego_final*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    plot_hv_convergence_comparison()
