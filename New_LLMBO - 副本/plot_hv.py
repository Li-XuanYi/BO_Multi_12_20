"""
plot_hv.py — HV 收敛曲线绘图工具
==================================
输入: Excel 文件，格式要求见下方说明
输出: HV 收敛曲线图 (PNG/PDF)

Excel 格式说明
--------------
可以有多个 Sheet，每个 Sheet 对应一个子图 (subplot)。
若只有一个 Sheet 则只输出一张图。

每个 Sheet 的列结构:
  第 1 列: "Evaluation" — 评估次数 (x 轴)
  后续列: 每列为一次独立实验 (run) 的 HV 值
          列名格式: "<方法名>_run<编号>"
          例如: "EIMO_run1", "EIMO_run2", "ParEGO_run1", "ParEGO_run3"

脚本会自动:
  - 按方法名分组 (根据 "_run" 前缀)
  - 计算每个方法的均值和标准差
  - 绘制均值曲线 + 阴影置信带 (±1 std)
  - 不同方法用不同颜色和图例区分

用法:
  python plot_hv.py input.xlsx [output.png]
  python plot_hv.py input.xlsx output.pdf
"""

import sys
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ── 样式配置（模仿论文图风格） ──────────────────────────────────────────

# 方法名 → 颜色映射（可扩展）
METHOD_COLORS = {
    "EIMO":     "#C75B7A",   # 红/粉
    "LLM-MOBO": "#C75B7A",
    "ParEGO":   "#4A90C4",   # 蓝
    "BO":       "#4A90C4",
    "NSGA-II":  "#6AAF6A",   # 绿
    "qNEHVI":   "#E8A838",   # 橙
    "Random":   "#888888",   # 灰
}

# 默认颜色循环（未在映射中的方法）
DEFAULT_CYCLE = ["#C75B7A", "#4A90C4", "#6AAF6A", "#E8A838",
                 "#9B59B6", "#1ABC9C", "#E74C3C", "#34495E"]


def get_color(method_name: str, idx: int) -> str:
    """根据方法名返回颜色，未知方法按索引循环。"""
    for key, color in METHOD_COLORS.items():
        if key.lower() in method_name.lower():
            return color
    return DEFAULT_CYCLE[idx % len(DEFAULT_CYCLE)]


def parse_sheet(df: pd.DataFrame):
    """
    解析一个 Sheet 的数据。

    Returns
    -------
    evaluations : np.ndarray (N,)
    method_data : dict  {method_name: np.ndarray (N, n_runs)}
    """
    # 第一列是 Evaluation
    eval_col = df.columns[0]
    evaluations = df[eval_col].values.astype(float)

    # 其余列按方法名分组
    method_runs = defaultdict(list)
    for col in df.columns[1:]:
        col_str = str(col)
        # 尝试匹配 "<method>_run<N>" 或 "<method>_<N>"
        match = re.match(r"^(.+?)(?:_run|_r|_)\d+$", col_str, re.IGNORECASE)
        if match:
            method_name = match.group(1).strip()
        else:
            # 无 _run 后缀，整列名作为方法名
            method_name = col_str.strip()
        method_runs[method_name].append(df[col].values.astype(float))

    method_data = {}
    for name, runs in method_runs.items():
        method_data[name] = np.column_stack(runs)  # (N, n_runs)

    return evaluations, method_data


def plot_hv_convergence(
    excel_path: str,
    output_path: str = "hv_convergence.png",
    figsize_per_subplot: tuple = (4.5, 3.5),
    dpi: int = 300,
    alpha_band: float = 0.20,
    linewidth: float = 1.8,
    ylabel: str = "HV",
    xlabel: str = "Number of evaluations",
    subplot_labels: list = None,  # ["(a)", "(b)", "(c)"]
    ylim: tuple = None,  # e.g. (0.8, 0.9)
):
    """
    主绘图函数。

    Parameters
    ----------
    excel_path : str       输入 Excel 路径
    output_path : str      输出图片路径 (.png / .pdf)
    figsize_per_subplot : tuple  每个子图尺寸
    dpi : int              输出分辨率
    alpha_band : float     置信带透明度
    linewidth : float      曲线粗细
    ylabel / xlabel : str  坐标轴标签
    subplot_labels : list  子图标签列表，如 ["(a)", "(b)", "(c)"]
    ylim : tuple           y 轴范围（None 则自动）
    """
    # 读取所有 Sheet
    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names
    n_sheets = len(sheet_names)

    # 子图布局
    fig_w = figsize_per_subplot[0] * n_sheets
    fig_h = figsize_per_subplot[1]
    fig, axes = plt.subplots(1, n_sheets, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    if subplot_labels is None and n_sheets > 1:
        subplot_labels = [f"({chr(97+i)})" for i in range(n_sheets)]

    for sheet_idx, sheet_name in enumerate(sheet_names):
        ax = axes[sheet_idx]
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        evaluations, method_data = parse_sheet(df)

        for m_idx, (method_name, data_matrix) in enumerate(method_data.items()):
            color = get_color(method_name, m_idx)
            mean = np.nanmean(data_matrix, axis=1)
            std  = np.nanstd(data_matrix, axis=1)
            n_runs = data_matrix.shape[1]

            ax.plot(evaluations, mean, color=color, linewidth=linewidth,
                    label=method_name, zorder=3)
            ax.fill_between(evaluations, mean - std, mean + std,
                            color=color, alpha=alpha_band, zorder=2)

        ax.set_xlabel(xlabel, fontsize=11)
        if sheet_idx == 0:
            ax.set_ylabel(ylabel, fontsize=11)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.tick_params(labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(False)

        # 子图标签
        if subplot_labels and sheet_idx < len(subplot_labels):
            ax.set_xlabel(f"{xlabel}\n{subplot_labels[sheet_idx]}", fontsize=11)

        # 图例只在第一个子图显示
        if sheet_idx == 0:
            ax.legend(fontsize=9, loc="lower right", framealpha=0.9,
                      edgecolor="none")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_hv] 已保存: {output_path}")
    return output_path


# ── CLI 入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python plot_hv.py <input.xlsx> [output.png]")
        print("      输出格式由扩展名决定 (.png / .pdf / .svg)")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "hv_convergence.png"

    plot_hv_convergence(in_path, out_path)
