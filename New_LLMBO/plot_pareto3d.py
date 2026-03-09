"""
plot_pareto3d.py — 三目标 Pareto 前沿 3D 散点图绘图工具
========================================================
输入: Excel 文件
输出: 3D 散点图 (PNG/PDF)

Excel 格式说明
--------------
列结构（列名不区分大小写，支持中英文）:
  - "Charging Time" 或 "充电时间" 或 "Time"        : 充电时间 (s)
  - "Temperature" 或 "温度" 或 "Temp"               : 峰值温度 (K)
  - "Aging" 或 "老化程度" 或 "Lithium-ion loss"     : 老化指标 (C)
  - "Method" 或 "方法" 或 "Group"（可选）           : 分组列（如 LLM-MOBO / ParEGO）
  - "Label"（可选）                                  : 标注文字（如 "A", "B"）

用法:
  python plot_pareto3d.py input.xlsx [output.png]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ── 列名匹配 ────────────────────────────────────────────────────────────

COLUMN_ALIASES = {
    "time":  ["charging time", "充电时间", "time", "charge_time", "t_charge"],
    "temp":  ["temperature", "温度", "temp", "peak_temp", "peak temperature"],
    "aging": ["aging", "老化程度", "lithium-ion loss", "aging degree",
              "li_loss", "degradation", "老化"],
    "group": ["method", "方法", "group", "soh", "类别", "category"],
    "label": ["label", "标注", "标签", "annotation"],
}


def find_column(df: pd.DataFrame, key: str) -> str:
    aliases = COLUMN_ALIASES.get(key, [key])
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in df_cols_lower:
            return df_cols_lower[alias.lower()]
    raise KeyError(f"找不到 '{key}' 对应的列。期望: {aliases}，实际: {list(df.columns)}")


# ── 方法/分组颜色 ───────────────────────────────────────────────────────

METHOD_COLORS = {
    "LLM-MOBO":  "#C75B7A",
    "EIMO":      "#C75B7A",
    "ParEGO":    "#4A90C4",
    "BO":        "#4A90C4",
    "NSGA-II":   "#6AAF6A",
    "qNEHVI":    "#E8A838",
    "Random":    "#888888",
}
DEFAULT_CYCLE = ["#9B7ED8", "#7EC8E3", "#E8A07A", "#A8D86E",
                 "#E74C3C", "#3498DB", "#F39C12", "#1ABC9C"]


def get_method_color(name: str, idx: int) -> str:
    for key, color in METHOD_COLORS.items():
        if key.lower() in str(name).lower():
            return color
    return DEFAULT_CYCLE[idx % len(DEFAULT_CYCLE)]


def plot_pareto3d(
    excel_path: str,
    output_path: str = "pareto3d.png",
    figsize: tuple = (10, 7.5),
    dpi: int = 300,
    scatter_size: int = 20,
    scatter_alpha: float = 0.60,
    star_size: int = 200,
    elev: float = 20,
    azim: float = -135,       # 朝右
    xlabel: str = "Charging Time/s",
    ylabel: str = "Temperature/K",
    zlabel: str = "Lithium-ion loss/C",
):
    df = pd.read_excel(excel_path)

    time_col  = find_column(df, "time")
    temp_col  = find_column(df, "temp")
    aging_col = find_column(df, "aging")

    try:
        group_col = find_column(df, "group")
        has_group = True
    except KeyError:
        has_group = False

    try:
        label_col = find_column(df, "label")
        has_label = True
    except KeyError:
        has_label = False

    # ── 绘图 ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if has_group:
        groups = df[group_col].dropna().unique()
        for g_idx, g_val in enumerate(groups):
            mask = df[group_col] == g_val
            color = get_method_color(g_val, g_idx)
            ax.scatter(
                df.loc[mask, time_col],
                df.loc[mask, temp_col],
                df.loc[mask, aging_col],
                c=color, s=scatter_size, alpha=scatter_alpha,
                edgecolors="none", label=str(g_val), depthshade=True,
            )
    else:
        ax.scatter(
            df[time_col], df[temp_col], df[aging_col],
            c="#C75B7A", s=scatter_size, alpha=scatter_alpha,
            edgecolors="none", label="LLM-MOBO", depthshade=True,
        )

    # 标注红星
    if has_label:
        labeled = df.dropna(subset=[label_col])
        for _, row in labeled.iterrows():
            ax.scatter(
                [row[time_col]], [row[temp_col]], [row[aging_col]],
                marker="*", s=star_size, c="#C0392B", edgecolors="#8B0000",
                linewidths=0.5, zorder=10,
            )
            z_range = df[aging_col].max() - df[aging_col].min()
            z_offset = z_range * 0.03
            ax.text(
                row[time_col], row[temp_col], row[aging_col] + z_offset,
                str(row[label_col]),
                fontsize=10, fontweight="bold", color="#8B0000",
                ha="center", va="bottom",
            )

    ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=11, labelpad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=8)

    ax.legend(fontsize=10, loc="upper right", framealpha=0.85,
              edgecolor="none", markerscale=1.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_pareto3d] 已保存: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python plot_pareto3d.py <input.xlsx> [output.png]")
        sys.exit(1)
    in_path  = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "pareto3d.png"
    plot_pareto3d(in_path, out_path)
