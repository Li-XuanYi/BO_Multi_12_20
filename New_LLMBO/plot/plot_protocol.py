"""
plot_protocol.py — LLAMBO-MO 充电协议可视化
============================================
输入: 一组或多组充电协议参数 (I₁, SOC₁, I₂)
输出: 4 子图 (V-t, T-t, I-t, Q_loss-t)

使用方法:
---------
# 单协议
python plot_protocol.py --protocols 5.0,0.35,2.5

# 多协议对比 (最多 6 条)
python plot_protocol.py --protocols 5.0,0.35,2.5  7.0,0.20,4.0

# 指定标签
python plot_protocol.py --protocols 5.0,0.35,2.5  7.0,0.20,4.0 \
                        --labels "Protocol A" "Protocol B"

# 指定输出文件名
python plot_protocol.py --protocols 5.0,0.35,2.5 --output charging_result.png

# 分别保存 4 张独立图片 (而非 1 张 2×2 合图)
python plot_protocol.py --protocols 5.0,0.35,2.5 --separate
"""

try:
    import pybamm
    PYBAMM_AVAILABLE = True
except ImportError:
    pybamm = None
    PYBAMM_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

# ── 中文字体 & 样式 ─────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size":        12,
    "axes.labelsize":   13,
    "axes.titlesize":   14,
    "legend.fontsize":  11,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "lines.linewidth":  1.8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# 尝试加载中文字体 (SimHei / WenQuanYi)，失败则用英文
for _font in ["SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

logger = logging.getLogger(__name__)

# ── 配色方案 (最多 6 条协议) ─────────────────────────────────
COLORS = [
    "#2ca02c",   # 绿 (Protocol A)
    "#9467bd",   # 紫 (Protocol B)
    "#1f77b4",   # 蓝
    "#d62728",   # 红
    "#ff7f0e",   # 橙
    "#17becf",   # 青
]

FARADAY = 96485.33212  # C/mol


# =============================================================
#  核心: 运行仿真并提取时间序列
# =============================================================
def simulate_and_extract(
    I1: float,
    SOC1: float,
    I2: float,
    battery_config: Optional[Dict] = None,
    constraints: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    运行 PyBaMM 仿真, 返回完整时间序列数据

    Returns
    -------
    dict with keys:
        time_s    : np.ndarray   时间 [s]
        voltage   : np.ndarray   端电压 [V]
        current   : np.ndarray   充电电流 [A]
        temp      : np.ndarray   电池温度 [K]
        li_loss_C : np.ndarray   锂损失 [C] (库仑)
    或 None (仿真失败)
    """
    if not PYBAMM_AVAILABLE:
        raise ImportError("PyBaMM 未安装, 请先: pip install pybamm")

    # ── 默认配置 ──
    batt = {
        "param_set":        "Chen2020",
        "nominal_capacity": 5.0,
        "init_voltage":     3.0,
        "init_temp":        298.15,
        "ambient_temp":     298.15,
        "soc_end":          0.80,
    }
    if battery_config:
        batt.update(battery_config)

    cons = {"voltage_max": 4.3, "temp_max": 328.15}
    if constraints:
        cons.update(constraints)

    Q_nom_Ah = batt["nominal_capacity"]
    V_max    = cons["voltage_max"]

    # ── 计算 SOC0 ──
    param_tmp = pybamm.ParameterValues(batt["param_set"])
    try:
        param_tmp.set_initial_stoichiometries(f"{batt['init_voltage']} V")
        c_n_max  = param_tmp["Maximum concentration in negative electrode [mol.m-3]"]
        c_n_init = param_tmp["Initial concentration in negative electrode [mol.m-3]"]
        x_n_init = c_n_init / c_n_max
        x_n_full = 0.9014   # Chen2020 满充
        x_n_empty = 0.028
        soc0 = float(np.clip((x_n_init - x_n_empty) / (x_n_full - x_n_empty), 0, 1))
    except Exception:
        soc0 = 0.04

    # ── 参数校验 ──
    if SOC1 <= soc0 or SOC1 >= batt["soc_end"]:
        logger.error(f"SOC1={SOC1} 不在 ({soc0:.3f}, {batt['soc_end']}) 范围内")
        return None

    # ── 阶段时间估算 ──
    t1_safe = (SOC1 - soc0) * Q_nom_Ah * 3600.0 / I1 * 1.10
    t2_safe = (batt["soc_end"] - SOC1) * Q_nom_Ah * 3600.0 / I2 * 1.10

    # ── 构建模型 ──
    model = pybamm.lithium_ion.SPMe(
        options={
            "thermal":         "lumped",
            "SEI":             "reaction limited",
            "lithium plating": "irreversible",
        }
    )

    param = pybamm.ParameterValues(batt["param_set"])
    param["Upper voltage cut-off [V]"]              = V_max + 0.1
    param["SEI growth activation energy [J.mol-1]"] = 37500.0
    param["Initial temperature [K]"]                = batt["init_temp"]
    param["Ambient temperature [K]"]                = batt["ambient_temp"]

    # ── O'Kane2022 析锂参数 ──
    plating_params = {
        "Exchange-current density for plating [A.m-2]": 0.001,
        "Lithium plating open-circuit potential [V]":   0.0,
        "Dead lithium decay constant [s-1]":            3.33e-7,
        "Lithium plating transfer coefficient":         0.65,
        "Initial dead lithium concentration [mol.m-3]": 0.0,
        "Initial plated lithium concentration [mol.m-3]": 0.0,
        "Lithium metal partial molar volume [m3.mol-1]":  1.3e-5,
        "Typical plated lithium concentration [mol.m-3]": 1000.0,
    }
    param.update(plating_params, check_already_exists=False)

    try:
        param.set_initial_stoichiometries(f"{batt['init_voltage']} V")
    except Exception as e:
        logger.warning(f"set_initial_stoichiometries 失败: {e}")

    # ── Experiment ──
    experiment = pybamm.Experiment([
        f"Charge at {I1:.4f} A for {t1_safe:.1f} seconds or until {V_max} V",
        f"Charge at {I2:.4f} A for {t2_safe:.1f} seconds or until {V_max} V",
    ])

    # ── 求解 ──
    try:
        try:
            solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
        except Exception:
            solver = pybamm.ScipySolver(atol=1e-6, rtol=1e-6)

        sim = pybamm.Simulation(
            model, experiment=experiment,
            parameter_values=param, solver=solver,
        )
        sol = sim.solve()
        if sol is None:
            logger.error("求解返回 None")
            return None
    except Exception as e:
        logger.error(f"求解失败: {e}")
        return None

    # ── 提取时间序列 ──
    def _to_1d(arr):
        """将可能的多维数组压缩为 1D (空间维度取均值)"""
        arr = np.asarray(arr)
        while arr.ndim > 1:
            arr = np.mean(arr, axis=0)
        return arr

    time_s  = _to_1d(sol["Time [s]"].entries)
    voltage = _to_1d(sol["Voltage [V]"].entries)
    current = _to_1d(sol["Current [A]"].entries)

    # 温度
    temp = None
    for var in ["Cell temperature [K]",
                "X-averaged cell temperature [K]",
                "Volume-averaged cell temperature [K]"]:
        try:
            temp = _to_1d(sol[var].entries)
            break
        except KeyError:
            continue
    if temp is None:
        logger.warning("未找到温度变量, 使用恒温假设")
        temp = np.full_like(time_s, batt["init_temp"])

    # 锂损失 (mol → C)
    li_loss_mol = np.zeros_like(time_s)
    for var in [
        "Loss of lithium to SEI [mol]",
        "Loss of lithium to negative SEI [mol]",
        "Loss of lithium to negative SEI on cracks [mol]",
        "Loss of lithium to lithium plating [mol]",
        "Loss of lithium to negative lithium plating [mol]",
    ]:
        try:
            entries = _to_1d(sol[var].entries)
            li_loss_mol = li_loss_mol + entries
        except KeyError:
            continue

    li_loss_C = li_loss_mol * FARADAY  # mol → C (库仑)

    return {
        "time_s":    np.asarray(time_s),
        "voltage":   np.asarray(voltage),
        "current":   np.abs(np.asarray(current)),   # 充电电流取正
        "temp":      np.asarray(temp),
        "li_loss_C": np.asarray(li_loss_C),
    }


# =============================================================
#  绘图: 2×2 合图 (类似参考图)
# =============================================================
def plot_combined(
    data_list:  List[Dict],
    labels:     List[str],
    output:     str = "charging_protocol.png",
    v_max:      float = 4.3,
    t_max:      float = 328.15,
    show:       bool = False,
) -> str:
    """
    绘制 2×2 合图

    Parameters
    ----------
    data_list : list of dict   每条协议的时间序列
    labels    : list of str    对应标签
    output    : str            输出路径
    v_max     : float          电压约束 (画虚线)
    t_max     : float          温度约束 (画虚线)
    show      : bool           是否弹窗显示

    Returns
    -------
    str  输出文件路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    ax_v, ax_t = axes[0]   # 上: 电压, 温度
    ax_i, ax_q = axes[1]   # 下: 电流, 锂损失

    for idx, (data, label) in enumerate(zip(data_list, labels)):
        c = COLORS[idx % len(COLORS)]
        t = data["time_s"]

        ax_v.plot(t, data["voltage"],   color=c, label=label)
        ax_t.plot(t, data["temp"],      color=c, label=label)
        ax_i.plot(t, data["current"],   color=c, label=label)
        ax_q.plot(t, data["li_loss_C"], color=c, label=label)

    # ── 约束虚线 ──
    ax_v.axhline(v_max, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_t.axhline(t_max, color="red", linestyle="--", linewidth=1.0, alpha=0.7)

    # ── 标签 ──
    ax_v.set_xlabel("Time/s");  ax_v.set_ylabel("Voltage/V")
    ax_t.set_xlabel("Time/s");  ax_t.set_ylabel("Temperature/K")
    ax_i.set_xlabel("Time/s");  ax_i.set_ylabel("Input Current/A")
    ax_q.set_xlabel("Time/s");  ax_q.set_ylabel("Lithium-ion loss/C")

    # ── 子图编号 ──
    for ax, tag in zip([ax_v, ax_t, ax_i, ax_q], ["(a)", "(b)", "(c)", "(d)"]):
        ax.set_title(tag, fontsize=12, loc="center", pad=6)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.8)

    fig.savefig(output)
    logger.info(f"合图已保存: {output}")
    if show:
        plt.show()
    plt.close(fig)
    return output


# =============================================================
#  绘图: 4 张独立图片
# =============================================================
def plot_separate(
    data_list:  List[Dict],
    labels:     List[str],
    prefix:     str = "charging",
    v_max:      float = 4.3,
    t_max:      float = 328.15,
    show:       bool = False,
) -> List[str]:
    """
    分别保存 4 张独立图片

    Returns
    -------
    list of str  输出文件路径列表
    """
    configs = [
        ("voltage",   "Voltage/V",           "voltage",   v_max, f"{prefix}_voltage.png"),
        ("temp",      "Temperature/K",       "temp",      t_max, f"{prefix}_temperature.png"),
        ("current",   "Input Current/A",     "current",   None,  f"{prefix}_current.png"),
        ("li_loss_C", "Lithium-ion loss/C",  "li_loss_C", None,  f"{prefix}_li_loss.png"),
    ]

    tags = ["(a)", "(b)", "(c)", "(d)"]
    paths = []

    for (key, ylabel, data_key, constraint, fname), tag in zip(configs, tags):
        fig, ax = plt.subplots(figsize=(5.5, 4))

        for idx, (data, label) in enumerate(zip(data_list, labels)):
            c = COLORS[idx % len(COLORS)]
            ax.plot(data["time_s"], data[data_key], color=c, label=label)

        if constraint is not None:
            ax.axhline(constraint, color="red", linestyle="--",
                       linewidth=1.0, alpha=0.7)

        ax.set_xlabel("Time/s")
        ax.set_ylabel(ylabel)
        ax.set_title(tag, fontsize=12, loc="center", pad=6)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.8)

        fig.savefig(fname)
        paths.append(fname)
        logger.info(f"已保存: {fname}")
        if show:
            plt.show()
        plt.close(fig)

    return paths


# =============================================================
#  便捷接口: 一键绘图
# =============================================================
def plot_protocols(
    protocols:  List[Tuple[float, float, float]],
    labels:     Optional[List[str]] = None,
    output:     str = "charging_protocol.png",
    separate:   bool = False,
    show:       bool = False,
    battery_config: Optional[Dict] = None,
    constraints:    Optional[Dict] = None,
):
    """
    一键接口: 传入协议参数, 自动仿真 + 绘图

    Parameters
    ----------
    protocols : list of (I1, SOC1, I2) tuples
    labels    : list of str, 可选标签 (默认 Protocol A/B/C...)
    output    : str, 合图输出路径
    separate  : bool, True → 4 张独立图
    show      : bool, True → plt.show()

    Returns
    -------
    str | list of str   输出文件路径

    Example
    -------
    >>> plot_protocols(
    ...     protocols=[(5.0, 0.35, 2.5), (7.0, 0.20, 4.0)],
    ...     labels=["Protocol A", "Protocol B"],
    ... )
    """
    n = len(protocols)
    if labels is None:
        alpha = "ABCDEFGHIJ"
        labels = [f"Protocol {alpha[i]}" for i in range(n)]

    cons = constraints or {"voltage_max": 4.3, "temp_max": 328.15}

    data_list = []
    for i, (I1, SOC1, I2) in enumerate(protocols):
        print(f"[{i+1}/{n}] 仿真 {labels[i]}: I1={I1}A, SOC1={SOC1}, I2={I2}A ...")
        data = simulate_and_extract(I1, SOC1, I2, battery_config, cons)
        if data is None:
            print(f"  ✗ 仿真失败, 跳过")
            continue
        data_list.append(data)
        obj_time = data["time_s"][-1]
        obj_temp = np.max(data["temp"])
        obj_loss = data["li_loss_C"][-1]
        print(f"  ✓ 时间={obj_time:.0f}s  峰值温度={obj_temp:.2f}K  锂损失={obj_loss:.2f}C")

    if not data_list:
        print("所有协议仿真失败, 无法绘图")
        return None

    # 过滤掉失败的标签
    valid_labels = labels[:len(data_list)]

    v_max = cons.get("voltage_max", 4.3)
    t_max = cons.get("temp_max", 328.15)

    if separate:
        prefix = os.path.splitext(output)[0]
        return plot_separate(data_list, valid_labels, prefix, v_max, t_max, show)
    else:
        return plot_combined(data_list, valid_labels, output, v_max, t_max, show)


# =============================================================
#  CLI
# =============================================================
def main():
    parser = argparse.ArgumentParser(
        description="LLAMBO-MO 充电协议可视化",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--protocols", nargs="+", required=True,
        help="协议参数, 格式: I1,SOC1,I2  (可指定多组)\n"
             "示例: --protocols 5.0,0.35,2.5  7.0,0.20,4.0",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="协议标签 (默认 Protocol A/B/C...)",
    )
    parser.add_argument(
        "--output", default="charging_protocol.png",
        help="输出文件名 (默认: charging_protocol.png)",
    )
    parser.add_argument(
        "--separate", action="store_true",
        help="分别保存 4 张独立图片",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="弹窗显示图片",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    # 解析协议参数
    protocols = []
    for p_str in args.protocols:
        parts = p_str.split(",")
        if len(parts) != 3:
            parser.error(f"协议格式错误: '{p_str}', 需要 I1,SOC1,I2")
        I1, SOC1, I2 = float(parts[0]), float(parts[1]), float(parts[2])
        protocols.append((I1, SOC1, I2))

    result = plot_protocols(
        protocols=protocols,
        labels=args.labels,
        output=args.output,
        separate=args.separate,
        show=args.show,
    )

    if result is None:
        exit(1)

    if isinstance(result, list):
        print(f"\n已保存 {len(result)} 张图片:")
        for p in result:
            print(f"  {p}")
    else:
        print(f"\n已保存: {result}")


if __name__ == "__main__":
    main()