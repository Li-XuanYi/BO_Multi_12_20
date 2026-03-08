"""
三维 Pareto 前沿可视化（Batch 6）

用法:
  python plots/pareto_3d.py --results results/V0/seed_42.json
  python plots/pareto_3d.py --results results/V0/seed_42.json --save figs/pareto.png
  python plots/pareto_3d.py --results results/V0/seed_42.json --compare results/V6/seed_42.json
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _load_pareto(path: str):
    """从结果 JSON 加载 Pareto 前沿点（[[time, temp, aging], ...]）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pf = data.get("pareto_front", [])
    if pf:
        return np.array(pf, dtype=float)

    # 兼容格式：从 database 重新提取
    entries = data.get("database", [])
    valid = [e for e in entries if e.get("valid", False)]
    if not valid:
        return np.empty((0, 3))

    objs = np.array([[e["time"], e["temp"], e["aging"]] for e in valid])
    # 非支配排序
    n = len(objs)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = (
            np.all(objs <= objs[i], axis=1) &
            np.any(objs < objs[i], axis=1) &
            (np.arange(n) != i)
        )
        if np.any(dominated):
            mask[i] = False
    return objs[mask]


def _normalize(pts: np.ndarray, ideal: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """将目标归一化到 [0, 1]（理想点→0，参考点→1）。"""
    rng = ref - ideal
    rng[rng < 1e-12] = 1.0
    return (pts - ideal) / rng


def plot_pareto_3d(
    results_path: str,
    compare_path: str = None,
    save_path: str = None,
    figsize: tuple = (8, 6),
    ideal: np.ndarray = None,
    ref: np.ndarray = None,
):
    """
    绘制 3D Pareto 前沿。

    参数
    ----
    results_path : str
        主方法结果路径（必须）
    compare_path : str | None
        对比方法路径（可选）
    save_path : str | None
        图片保存路径；None 则显示
    ideal / ref : np.ndarray
        理想点与参考点（用于归一化显示），默认使用 FrameWork 值
    """
    if ideal is None:
        ideal = np.array([1200.0, 298.15, 1e-6])
    if ref is None:
        ref = np.array([7200.0, 323.15, 0.008])

    pf_main = _load_pareto(results_path)
    if len(pf_main) == 0:
        print(f"[WARN] {results_path} 中未找到 Pareto 点")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # 主方法
    if len(pf_main) > 0:
        norm = _normalize(pf_main, ideal, ref)
        sc = ax.scatter(
            norm[:, 0], norm[:, 1], norm[:, 2],
            c=norm[:, 2],            # 按 aging 着色
            cmap="RdYlGn_r",
            s=60, alpha=0.85,
            edgecolors="k", linewidths=0.4,
            label=os.path.basename(os.path.dirname(results_path)) or "LLMBO-MO",
            zorder=3,
        )
        cb = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cb.set_label("Aging (norm.)", fontsize=9)

    # 对比方法（灰色半透明）
    if compare_path:
        pf_cmp = _load_pareto(compare_path)
        if len(pf_cmp) > 0:
            norm_cmp = _normalize(pf_cmp, ideal, ref)
            ax.scatter(
                norm_cmp[:, 0], norm_cmp[:, 1], norm_cmp[:, 2],
                c="#BDBDBD", s=35, alpha=0.5,
                edgecolors="gray", linewidths=0.3,
                label=os.path.basename(os.path.dirname(compare_path)),
                zorder=2,
            )

    ax.set_xlabel("Charge time (norm.)", fontsize=10, labelpad=8)
    ax.set_ylabel("Peak temp (norm.)", fontsize=10, labelpad=8)
    ax.set_zlabel("Aging (norm.)", fontsize=10, labelpad=8)
    ax.set_title("Pareto Front (3-Objective)", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_zlim(0, 1.05)
    ax.legend(fontsize=9, loc="upper left")
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[保存] {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="3D Pareto 前沿可视化")
    parser.add_argument("--results", required=True, help="主方法结果 JSON 路径")
    parser.add_argument("--compare", default=None, help="对比方法结果 JSON 路径（可选）")
    parser.add_argument("--save", default=None, help="图片保存路径 (e.g., figs/pareto.png)")
    args = parser.parse_args()

    plot_pareto_3d(
        results_path=args.results,
        compare_path=args.compare,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
