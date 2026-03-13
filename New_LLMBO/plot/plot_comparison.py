"""
New_LLMBO/plot_comparison.py
==============================
可视化脚本：HV 收敛曲线对比图

功能：
1. HV 收敛曲线（均值 ± 95% CI）
2. Pareto front 对比图
3. 支持多方法对比
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 设置 matplotlib 后端（无 GUI 环境）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiment_configs import METHOD_STYLE, DEFAULT_BUDGET


def _load_results(base_dir: str, method: str) -> List[Dict]:
    """加载单个方法的所有种子结果"""
    method_dir = os.path.join(base_dir, method)
    if not os.path.exists(method_dir):
        return []

    results = []
    for seed_file in sorted(Path(method_dir).glob("seed_*.json")):
        try:
            with open(seed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            print(f"[警告] 无法加载 {seed_file}: {e}")

    return results


def _load_all_results(base_dir: str) -> Dict[str, List[Dict]]:
    """加载所有方法的结果"""
    all_results = {}
    base = Path(base_dir)

    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        results = _load_results(base_dir, method)
        if results:
            all_results[method] = results

    return all_results


def _pad_hv_curves(curves: List[List[float]]) -> Tuple[np.ndarray, int]:
    """对齐 HV 曲线长度（用最后一个值填充）"""
    if not curves or not curves[0]:
        return np.array([]), 0

    lengths = [len(c) for c in curves]
    max_len = max(lengths)

    padded = []
    for c in curves:
        if len(c) >= max_len:
            padded.append(c[:max_len])
        else:
            padded.append(c + [c[-1]] * (max_len - len(c)))

    return np.array(padded), max_len


def plot_hv_convergence(
    base_dir: str,
    save_path: Optional[str] = None,
    methods: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "HV Convergence Comparison",
    ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
):
    """
    绘制 HV 收敛曲线对比图（均值 ± 95% CI）
    """
    all_results = _load_all_results(base_dir)

    if not all_results:
        print(f"[错误] 未找到结果目录：{base_dir}")
        return

    if methods:
        all_results = {m: all_results.get(m, []) for m in methods}

    fig, ax = plt.subplots(figsize=figsize)

    for method, seed_results in sorted(all_results.items()):
        # 提取 HV 曲线
        hv_curves = [r.get("hv_history", []) for r in seed_results]
        hv_curves = [hv for hv in hv_curves if hv]  # 过滤空曲线

        if not hv_curves:
            continue

        padded, n = _pad_hv_curves(hv_curves)
        x = np.arange(1, n + 1)
        mean = np.mean(padded, axis=0)

        # 95% CI
        n_seeds = len(padded)
        if n_seeds > 1:
            sem = np.std(padded, axis=0, ddof=1) / np.sqrt(n_seeds)
            ci = 1.96 * sem  # 95% CI
        else:
            ci = np.zeros(n)

        # 获取样式
        style = METHOD_STYLE.get(method, {"color": "#999999", "ls": "--", "lw": 1.5})
        label = style.get("label", method)

        # 绘制曲线
        ax.plot(
            x, mean,
            color=style.get("color", "#999999"),
            linestyle=style.get("ls", "-"),
            linewidth=style.get("lw", 1.5),
            label=label,
            zorder=3
        )

        # 绘制置信带
        ax.fill_between(
            x, mean - ci, mean + ci,
            color=style.get("color", "#999999"),
            alpha=0.15,
            zorder=2
        )

    ax.set_xlabel("Number of Evaluations", fontsize=12)
    ax.set_ylabel("Hypervolume (normalized)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)

    ax.legend(fontsize=9, ncol=2, loc="lower right", framealpha=0.9)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[保存] {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_pareto_front_comparison(
    base_dir: str,
    save_path: Optional[str] = None,
    methods: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Pareto Front Comparison",
    dpi: int = 300,
    objective_names: Tuple[str, str, str] = ("Charging Time (s)", "Peak Temp (K)", "Aging"),
):
    """
    绘制 Pareto front 对比图（3D 或 2D 投影）
    """
    all_results = _load_all_results(base_dir)

    if not all_results:
        print(f"[错误] 未找到结果目录：{base_dir}")
        return

    if methods:
        all_results = {m: all_results.get(m, []) for m in methods}

    fig = plt.figure(figsize=figsize)

    # 3D Pareto Front
    ax = fig.add_subplot(111, projection='3d')

    for method, seed_results in sorted(all_results.items()):
        style = METHOD_STYLE.get(method, {"color": "#999999", "marker": "o", "s": 30})
        label = style.get("label", method)

        # 聚合所有种子的可行解
        all_objs = []
        for r in seed_results:
            for ev in r.get("evaluations", []):
                if ev.get("feasible", False):
                    objs = [ev.get("time", 0), ev.get("temp", 0), ev.get("aging", 0)]
                    all_objs.append(objs)

        if not all_objs:
            continue

        objs_array = np.array(all_objs)

        # 简单非支配过滤
        nd_mask = _get_nondominated_mask(objs_array)
        pf_objs = objs_array[nd_mask]

        if len(pf_objs) == 0:
            continue

        ax.scatter(
            pf_objs[:, 0], pf_objs[:, 1], pf_objs[:, 2],
            c=style.get("color", "#999999"),
            marker=style.get("marker", "o"),
            s=style.get("s", 30),
            label=label,
            alpha=0.7
        )

    ax.set_xlabel(objective_names[0], fontsize=10)
    ax.set_ylabel(objective_names[1], fontsize=10)
    ax.set_zlabel(objective_names[2], fontsize=10)
    ax.set_title(title, fontsize=12)

    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[保存] {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _get_nondominated_mask(objectives: np.ndarray) -> np.ndarray:
    """获取非支配解的掩码"""
    n = len(objectives)
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            # j 支配 i
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                mask[i] = False
                break

    return mask


def plot_box_comparison(
    base_dir: str,
    save_path: Optional[str] = None,
    methods: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Final HV Distribution",
    ylabel: str = "Final Hypervolume",
    dpi: int = 300,
):
    """
    绘制最终 HV 的箱线图对比
    """
    all_results = _load_all_results(base_dir)

    if not all_results:
        print(f"[错误] 未找到结果目录：{base_dir}")
        return

    if methods:
        all_results = {m: all_results.get(m, []) for m in methods}

    fig, ax = plt.subplots(figsize=figsize)

    # 准备数据
    data_to_plot = []
    labels = []

    for method in sorted(all_results.keys()):
        seed_results = all_results[method]
        final_hvs = []

        for r in seed_results:
            hv = r.get("hv_history", [])
            if hv:
                final_hvs.append(hv[-1])

        if final_hvs:
            data_to_plot.append(final_hvs)
            style = METHOD_STYLE.get(method, {})
            labels.append(style.get("label", method))

    if not data_to_plot:
        print("[警告] 没有数据可绘制")
        return

    # 绘制箱线图
    bp = ax.boxplot(
        data_to_plot,
        labels=labels,
        patch_artist=True,
        notch=True,
        whis=1.5
    )

    # 填充颜色
    colors = [METHOD_STYLE.get(m, {}).get("color", "#999999") for m in sorted(all_results.keys()) if m in all_results]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis='y', linestyle="--", alpha=0.3)
    plt.xticks(rotation=15)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"[保存] {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ============================================================
# CLI 入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="可视化对比工具")
    parser.add_argument(
        "--results-dir",
        default="./results",
        help="结果目录"
    )
    parser.add_argument(
        "--type",
        choices=["hv", "pareto", "box"],
        default="hv",
        help="图表类型"
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="指定方法"
    )
    parser.add_argument(
        "--save",
        default=None,
        help="保存路径"
    )
    parser.add_argument(
        "--title",
        default=None,
        help="图标题"
    )

    args = parser.parse_args()

    if args.type == "hv":
        plot_hv_convergence(
            base_dir=args.results_dir,
            save_path=args.save,
            methods=args.methods,
            title=args.title or "HV Convergence Comparison"
        )
    elif args.type == "pareto":
        plot_pareto_front_comparison(
            base_dir=args.results_dir,
            save_path=args.save,
            methods=args.methods,
            title=args.title or "Pareto Front Comparison"
        )
    elif args.type == "box":
        plot_box_comparison(
            base_dir=args.results_dir,
            save_path=args.save,
            methods=args.methods,
            title=args.title or "Final HV Distribution"
        )


if __name__ == "__main__":
    main()
