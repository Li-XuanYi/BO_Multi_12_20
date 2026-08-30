"""
超体积收敛曲线对比图（Batch 6）

用法:
  python plots/hv_comparison.py --results_dir results/
  python plots/hv_comparison.py --results_dir results/ --save figs/hv.png
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ---------- 颜色与线型映射 ----------
METHOD_STYLE = {
    "V0_Full":              dict(color="#2196F3", ls="-",  lw=2.2, label="LLMBO-MO (Full)"),
    "V1_NoWarmStart":       dict(color="#4CAF50", ls="--", lw=1.7, label="w/o WarmStart"),
    "V2_NoPhysicsKernel":   dict(color="#FF9800", ls="--", lw=1.7, label="w/o PhysicsKernel"),
    "V3_NoGenAcq":          dict(color="#9C27B0", ls="--", lw=1.7, label="w/o GenAcq"),
    "V4_NoAdaptiveW":       dict(color="#00BCD4", ls=":",  lw=1.7, label="w/o AdaptiveW"),
    "V5_NoHVFeedback":      dict(color="#FF5722", ls=":",  lw=1.7, label="w/o HVFeedback"),
    "V6_VanillaBO":         dict(color="#795548", ls=":",  lw=1.7, label="VanillaBO"),
    "RandomSearch":         dict(color="#9E9E9E", ls="-",  lw=1.4, label="Random"),
    "SobolGP":              dict(color="#607D8B", ls="--", lw=1.4, label="Sobol+GP"),
    "ParEGO":               dict(color="#F44336", ls="-",  lw=1.7, label="ParEGO"),
    "NSGA2":                dict(color="#8BC34A", ls="-.",  lw=1.7, label="NSGA-II"),
    "MOEAD":                dict(color="#FFC107", ls="-.",  lw=1.7, label="MOEA/D"),
}
DEFAULT_STYLE = dict(color="#BDBDBD", ls=":", lw=1.2)


def _load_hv_curves(base_dir: str):
    """读取 results/ 下所有 seed_*.json 文件，返回 {method: [hv_curve, ...]} 。"""
    base = Path(base_dir)
    curves = {}
    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for seed_file in sorted(method_dir.glob("seed_*.json")):
            try:
                with open(seed_file, encoding="utf-8") as f:
                    data = json.load(f)
                hv = data.get("hv_history", [])
                if hv:
                    curves.setdefault(method, []).append(hv)
            except Exception:
                continue
    return curves


def _pad_or_trim(curves, target_len=None):
    """对齐所有曲线长度（用最后一个值填充）。"""
    if not curves:
        return curves, 0
    lengths = [len(c) for c in curves]
    if target_len is None:
        target_len = max(lengths)
    padded = []
    for c in curves:
        if len(c) >= target_len:
            padded.append(c[:target_len])
        else:
            padded.append(c + [c[-1]] * (target_len - len(c)))
    return padded, target_len


def plot_hv_comparison(
    results_dir: str,
    save_path: str = None,
    methods: list = None,
    figsize: tuple = (9, 5),
    title: str = "Hypervolume Convergence",
):
    """
    绘制各方法的超体积收敛曲线（均值 ± 95% CI）。

    参数
    ----
    results_dir : str
        实验结果根目录（包含 method/seed_*.json）
    save_path : str | None
        保存路径；None 则仅显示
    methods : list | None
        指定显示方法；None 则全部
    """
    all_curves = _load_hv_curves(results_dir)
    if not all_curves:
        print(f"[ERROR] 未找到任何结果文件：{results_dir}")
        sys.exit(1)

    if methods:
        all_curves = {m: v for m, v in all_curves.items() if m in methods}

    fig, ax = plt.subplots(figsize=figsize)

    for method, seed_curves in sorted(all_curves.items()):
        padded, n = _pad_or_trim(seed_curves)
        arr = np.array(padded, dtype=float)        # (n_seeds, n_iter)
        x = np.arange(1, n + 1)
        mean = arr.mean(axis=0)
        if len(arr) > 1:
            sem = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
            ci = 1.96 * sem                        # 95% CI
        else:
            ci = np.zeros(n)

        style = {**DEFAULT_STYLE, **METHOD_STYLE.get(method, {}), "label": method}
        ax.plot(x, mean, **{k: v for k, v in style.items() if k != "label"},
                label=style["label"])
        ax.fill_between(x, mean - ci, mean + ci,
                        color=style["color"], alpha=0.12)

    ax.set_xlabel("Evaluation count", fontsize=12)
    ax.set_ylabel("Hypervolume (normalized)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[保存] {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="超体积收敛曲线对比")
    parser.add_argument("--results_dir", default="results",
                        help="实验结果根目录 (default: results/)")
    parser.add_argument("--save", default=None,
                        help="图片保存路径 (e.g., figs/hv.png)")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="指定方法名列表（空则显示全部）")
    parser.add_argument("--title", default="Hypervolume Convergence",
                        help="图标题")
    args = parser.parse_args()

    plot_hv_comparison(
        results_dir=args.results_dir,
        save_path=args.save,
        methods=args.methods,
        title=args.title,
    )


if __name__ == "__main__":
    main()
