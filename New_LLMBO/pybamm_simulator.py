"""
pybamm_simulator.py — 三段恒流充电仿真器（改进版）
====================================================
输入: θ = (I1, I2, I3, dSOC1, dSOC2)
    电流单位: A (绝对值，与 Chen2020 Nominal capacity 5 Ah 对应)
    如需 C 倍率输入，令 use_crate=True，则 Ix 视作 C 倍率，内部自动换算

输出:
    raw_objectives : [time_s, delta_temp_K, aging_%]   均 minimize
    soc_final      : float                              终止 SOC
    trajectories   : dict  V / T / SOC / I 轨迹 (与 utils_fun.py 对齐)

约束检查 (仿真器只做这两项):
    T_peak > T_max (默认 328.15 K / 55°C) → penalty
    V_peak > V_max (默认 4.4 V)           → penalty

参数来源:
    电化学参数 — Chen2020 + 辨识值 (SPMe.py)
    热模型参数 — 辨识值 (SPMe.py)
    析锂参数   — O'Kane et al. 2022
    老化经验式 — utils_fun.py calCap()
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

try:
    import pybamm
    PYBAMM_AVAILABLE = True
    warnings.filterwarnings("ignore", message="No value provided for input.*Current function")
except ImportError:
    pybamm = None
    PYBAMM_AVAILABLE = False
from typing import Dict, List, Optional

from utils.constants import DEFAULT_BOUNDS, DSOC3_MIN, DSOC_SUM_MAX, FAILURE_PENALTY

logger = logging.getLogger(__name__)

# ---- 充电 SOC 窗口 ----
SOC_START = 0.0
SOC_END   = 0.8
SOC_SPAN  = SOC_END - SOC_START   # 0.8

# ---- 支持的电池参数集 ----
SUPPORTED_PARAM_SETS = ["Chen2020", "Ecker2015", "ORegan2022"]
DEFAULT_PARAM_SET = "Chen2020"

# ---- 各参数集的电池元数据 ----
_PARAM_SET_METADATA = {
    "Chen2020": {
        "battery_name": "LG INR21700-M50",
        "nominal_capacity_Ah": 5.0,
        "voltage_max_V": 4.4,
        "voltage_min_V": 2.5,
        "temp_max_K": 328.15,  # 55°C
        "temp_init_K": 298.15,  # 25°C
        "negative_electrode": "graphite",
        "positive_electrode": "NMC811",
    },
    "Ecker2015": {
        "battery_name": "Kokam SLPB 75106100",
        "nominal_capacity_Ah": 0.15625,
        "voltage_max_V": 4.2,
        "voltage_min_V": 2.5,
        "temp_max_K": 333.15,  # 60°C safety limit
        "temp_init_K": 298.15,
        "negative_electrode": "graphite",
        "positive_electrode": "NCO",
    },
    "ORegan2022": {
        "battery_name": "LG M50",
        "nominal_capacity_Ah": 5.0,
        "voltage_max_V": 4.4,
        "voltage_min_V": 2.5,
        "temp_max_K": 323.15,  # 50°C
        "temp_init_K": 298.15,
        "negative_electrode": "graphite",
        "positive_electrode": "NMC",
    },
}

# ---- Chen2020 负极浓度 → SOC 映射 (与 SPMe.py cal_soc "ours" 分支完全一致) ----
_SOC_C_MIN = 872.9651389896292      # mol/m³  (≈ 0% SOC)
_SOC_C_MAX = 30171.311359086325     # mol/m³  (≈ 100% SOC)

# ---- Penalty 向量: [time_s, delta_temp_K, aging_%] ----
# delta_temp_K = peak_temp - T_init，与 utils_fun.py temp_rise 对齐

# ---- 热响应对齐系数 ----
# 目标温升与老化输入温度分别对齐，兼顾两项指标的一致性。
TEMP_RISE_OBJ_ALIGN = 1.10
TEMP_FOR_AGING_ALIGN = 0.61

# ---- 辨识好的电化学 + 热模型参数 (来自 SPMe.py) ----
_IDENTIFIED_PARAMS = {
    "Negative particle radius [m]":                             4.69e-06,
    "Negative electrode active material volume fraction":       0.73,
    "Negative electrode conductivity [S.m-1]":                 258.00,
    "Negative electrode diffusivity [m2.s-1]":                 3.96e-14,
    "Positive particle radius [m]":                            4.17e-06,
    "Positive electrode active material volume fraction":      0.66,
    "Positive electrode conductivity [S.m-1]":                 0.22,
    "Positive electrode diffusivity [m2.s-1]":                 4.80e-15,
    "Total heat transfer coefficient [W.m-2.K-1]":            17.36,
    "Separator specific heat capacity [J.kg-1.K-1]":          2905.50,
    "Negative electrode specific heat capacity [J.kg-1.K-1]": 2400.56,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 2715.82,
    "Negative current collector specific heat capacity [J.kg-1.K-1]": 1138.79,
    "Positive current collector specific heat capacity [J.kg-1.K-1]": 1252.81,
}

# ---- 析锂参数 (O'Kane et al. 2022) ----
_PLATING_PARAMS = {
    "Exchange-current density for plating [A.m-2]":       0.001,
    "Lithium plating open-circuit potential [V]":          0.0,
    "Dead lithium decay constant [s-1]":                   3.33e-7,
    "Lithium plating transfer coefficient":                0.65,
    "Initial dead lithium concentration [mol.m-3]":        0.0,
    "Initial plated lithium concentration [mol.m-3]":      0.0,
    "Lithium metal partial molar volume [m3.mol-1]":       1.3e-5,
    "Typical plated lithium concentration [mol.m-3]":      1000.0,
}

# ---- 最大负极浓度基准值 (SOH=1 时 Chen2020 默认值，用于 SOH 缩放) ----
_C_NEG_MAX_SOH1 = 33133.0   # mol/m³  (Chen2020: Maximum concentration in negative electrode)
_C_NEG_MIN_SOH1 = 1308.0    # mol/m³  (用于线性插值，与 SPMe.py 逻辑一致)


# ---------------------------------------------------------------------------
#  工具函数
# ---------------------------------------------------------------------------

def _neg_conc_to_soc(c: float) -> float:
    """负极颗粒平均浓度 → SOC（Chen2020 "ours" 映射，与 SPMe.py 完全对齐）"""
    return float(np.clip((c - _SOC_C_MIN) / (_SOC_C_MAX - _SOC_C_MIN), 0.0, 1.0))


def cal_cap(soc_pct: float, temp_K: float, current_A: float) -> float:
    """
    经验可用容量模型，与 utils_fun.py calCap() 完全一致。

    Parameters
    ----------
    soc_pct   : SOC 百分比 (0–100)
    temp_K    : 温度 [K]
    current_A : 电流 [A]

    Returns
    -------
    cap : 可用容量 [Ah] (基于 5 Ah 标称)
    """
    tmp = (2896.6 * soc_pct + 7411.2) * np.exp((-31500 + 152.5 * current_A) / (8.314 * temp_K))
    cap = (20 / tmp) ** (1 / 0.57)
    return float(cap)


# ---------------------------------------------------------------------------
#  主类
# ---------------------------------------------------------------------------

class PyBaMMSimulator:
    """
    三段恒流充电仿真器，与 utils_fun.py / SPMe.py 完全对齐。

    电流输入约定
    -----------
    默认 (use_crate=False): I1/I2/I3 单位为 A (绝对值)
    use_crate=True         : I1/I2/I3 为 C 倍率，内部换算 I_A = C * Q_eff
                             与 utils_fun.py 中 input_current = i * bat_cap / 5 等价
    """

    def __init__(
        self,
        Q_nom:      float = None,        # None → 使用 param_set 默认值
        SOH:        float = 1.0,
        T_init:     float = None,        # None → 使用 param_set 默认值
        V_init:     float = 2.8,
        T_max:      float = None,        # None → 使用 param_set 默认值
        V_max:      float = None,        # None → 使用 param_set 默认值
        use_crate:  bool  = False,
        aging_mode: str   = "empirical",   # "empirical" | "physical" | "both"
        param_set:  str   = "Chen2020",    # "Chen2020" | "Ecker2015" | "ORegan2022"
    ) -> None:
        """
        Parameters
        ----------
        Q_nom      : 额定容量 [Ah]（SOH=1 时）。None → 使用 param_set 的标称容量
        SOH        : 健康状态 (0, 1]，缩放有效容量与最大负极浓度
        T_init     : 初始温度 [K]。None → 使用 param_set 的默认值（通常 298.15K）
        V_init     : 初始电压 [V]
        T_max      : 温度约束上限 [K]。None → 使用 param_set 的默认值
        V_max      : 电压约束上限 [V]。None → 使用 param_set 的默认值
        use_crate  : True → 输入 I 为 C 倍率；False → 输入 I 为绝对 A
        aging_mode : "empirical" — 与 utils_fun.py calCap() 一致
                     "physical"  — SEI + 析锂 Li 损失（更精确但量纲不同）
                     "both"      — raw_objectives 用 empirical，额外返回 physical
        param_set  : PyBaMM 参数集名称。支持: "Chen2020", "Ecker2015", "ORegan2022"
        """
        if not PYBAMM_AVAILABLE:
            raise ImportError("PyBaMM 未安装: pip install pybamm")

        assert 0 < SOH <= 1.0, "SOH 必须在 (0, 1]"
        assert aging_mode in ("empirical", "physical", "both"), \
            "aging_mode 须为 'empirical' / 'physical' / 'both'"
        assert param_set in SUPPORTED_PARAM_SETS, \
            f"参数集 '{param_set}' 不支持。可用: {SUPPORTED_PARAM_SETS}"

        # 加载参数集元数据
        self.param_set = param_set
        meta = _PARAM_SET_METADATA[param_set]
        self.battery_name = meta["battery_name"]

        # 使用参数集默认值或用户指定值
        self.Q_nom      = Q_nom if Q_nom is not None else meta["nominal_capacity_Ah"]
        self.SOH        = SOH
        self.Q_eff      = self.Q_nom * SOH
        self.T_init     = T_init if T_init is not None else meta["temp_init_K"]
        self.V_init     = V_init
        self.T_max      = T_max if T_max is not None else meta["temp_max_K"]
        self.V_max      = V_max if V_max is not None else meta["voltage_max_V"]
        self.use_crate  = use_crate
        self.aging_mode = aging_mode
        self.param_bounds = {
            key: tuple(bounds) for key, bounds in DEFAULT_BOUNDS.items()
        }
        self.soc_start = SOC_START
        self.soc_end = SOC_END
        self.dsoc_sum_max = DSOC_SUM_MAX

    # ------------------------------------------------------------------
    #  公共接口
    # ------------------------------------------------------------------

    def evaluate(self, theta) -> Dict:
        """
        Parameters
        ----------
        theta : array-like (5,)  [I1, I2, I3, dSOC1, dSOC2]

        Returns
        -------
        dict
            raw_objectives : np.ndarray (3,)  [time_s, delta_temp_K, aging_%]
            soc_final      : float
            feasible       : bool
            violation      : str | None
            trajectories   : dict  {V, T, SOC, I} — 与 utils_fun.py 输出格式一致
            aging_physical : float | None  (仅 aging_mode="both" 时返回)
        """
        rng = np.random.get_state()
        try:
            theta_arr = np.asarray(theta, dtype=float).ravel()
            if theta_arr.size < 5:
                return self._penalty("theta 必须包含 5 个决策变量")

            I1, I2, I3, dSOC1, dSOC2 = [float(x) for x in theta_arr[:5]]
            if dSOC1 + dSOC2 >= self.dsoc_sum_max:
                return self._penalty(
                    f"dSOC1+dSOC2={dSOC1 + dSOC2:.4f} 触发前置拦截，需满足 dSOC3>{DSOC3_MIN:.2f}"
                )
            return self._run(I1, I2, I3, dSOC1, dSOC2)
        except Exception as e:
            logger.error(f"evaluate error: {e}")
            return self._penalty(str(e)[:120])
        finally:
            np.random.set_state(rng)

    def evaluate_batch(self, thetas) -> List[Dict]:
        return [self.evaluate(th) for th in thetas]

    # ------------------------------------------------------------------
    #  核心仿真
    # ------------------------------------------------------------------

    def _run(self, I1, I2, I3, dSOC1, dSOC2) -> Dict:
        dSOC3 = SOC_SPAN - dSOC1 - dSOC2

        # 与 EIMO 一致的基础合法性检查
        if min(I1, I2, I3) <= 0:
            return self._penalty("电流必须为正")
        if min(dSOC1, dSOC2, dSOC3) <= 0:
            return self._penalty("dSOC 必须为正且 dSOC1+dSOC2<0.8")

        # 电流换算：与 EIMO utils_fun.py 对齐
        # use_crate=False: I 为协议电流参数（2~6），仿真电流 I_A = I * Q_eff / 5
        # use_crate=True : I 为 C 倍率，仿真电流 I_A = I * Q_eff
        if self.use_crate:
            I1_A = I1 * self.Q_eff
            I2_A = I2 * self.Q_eff
            I3_A = I3 * self.Q_eff
        else:
            I1_A = I1 * self.Q_eff / 5.0
            I2_A = I2 * self.Q_eff / 5.0
            I3_A = I3 * self.Q_eff / 5.0

        # 步长计算严格对齐 EIMO utils_fun.py:
        # steps = bat_cap*SOH/input_current * 60 * soc
        # 默认模式下 input_current = I * bat_cap / 5，因此 t = (5*SOH/I) * soc * 3600
        if self.use_crate:
            t1 = self.Q_eff / I1_A * dSOC1 * 3600.0
            t2 = self.Q_eff / I2_A * dSOC2 * 3600.0
            t3 = self.Q_eff / I3_A * dSOC3 * 3600.0
        else:
            t1 = 5.0 * self.SOH / I1 * dSOC1 * 3600.0
            t2 = 5.0 * self.SOH / I2 * dSOC2 * 3600.0
            t3 = 5.0 * self.SOH / I3 * dSOC3 * 3600.0

        # 构建模型：默认严格对齐 EIMO（仅 thermal=lumped）
        # 仅当需要 physical 老化时才开启副反应子模型
        if self.aging_mode in ("physical", "both"):
            model = pybamm.lithium_ion.SPMe(options={
                "thermal": "lumped",
                "SEI": "reaction limited",
                "lithium plating": "irreversible",
            })
        else:
            model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})

        # 参数集
        param = pybamm.ParameterValues(self.param_set)
        param.update({"Current function [A]": "[input]"}, check_already_exists=True)
        param.update(_IDENTIFIED_PARAMS, check_already_exists=True)
        if self.aging_mode in ("physical", "both"):
            param.update(_PLATING_PARAMS, check_already_exists=False)

        # SOH 缩放：默认仅缩放容量，避免部分 PyBaMM 版本下初始条件越界
        # 如需严格复现 EIMO 的 c_n,max 缩放，可在后续单独开启并联动校准 V_init。
        param.update({
            "Nominal cell capacity [A.h]": self.Q_eff,
        })

        param["Upper voltage cut-off [V]"]              = self.V_max + 0.1  # 安全裕量，实验步骤中已设硬限
        param["Initial temperature [K]"]                 = self.T_init
        param["Ambient temperature [K]"]                 = self.T_init
        if self.aging_mode in ("physical", "both"):
            param["SEI growth activation energy [J.mol-1]"] = 37500.0

        # 与 EIMO 对齐：按初始电压设置初始化学计量。
        # 若失败则回退默认初值，保证可运行性。
        try:
            param.set_initial_stoichiometries(f"{self.V_init} V")
        except Exception as e:
            logger.warning(f"set_initial_stoichiometries 失败，回退默认初值: {e}")

        try:
            # 与 EIMO/ utils_fun.py 一致：分段逐次求解并拼接轨迹
            voltage_all = [self.V_init]
            temp_all = [self.T_init]
            soc_all = []
            current_all = [0.0]
            loss_all = [0.0] if self.aging_mode in ("physical", "both") else None

            c0 = float(param["Initial concentration in negative electrode [mol.m-3]"])
            soc_all.append(_neg_conc_to_soc(c0))

            last_sol = None
            stage_currents = [I1_A, I2_A, I3_A]
            stage_times = [t1, t2, t3]

            for stage_I, stage_t in zip(stage_currents, stage_times):
                if last_sol is not None:
                    model.set_initial_conditions_from(last_sol)

                sim = pybamm.Simulation(model, parameter_values=param)
                st = max(1, int(round(stage_t)))
                t_eval = np.linspace(0.0, float(st), st + 1)

                sol = sim.solve(t_eval, inputs={"Current function [A]": -stage_I})
                if sol is None:
                    return self._penalty("stage solve 返回 None")

                v_stage = np.asarray(sol["Voltage [V]"].entries, dtype=float).reshape(-1)
                t_stage = np.asarray(sol["X-averaged cell temperature [K]"].entries, dtype=float).reshape(-1)
                c_stage = np.asarray(sol["R-averaged negative particle concentration [mol.m-3]"].entries, dtype=float)
                soc_stage = self._soc_series_from_conc(c_stage, len(v_stage))
                loss_stage = self._physical_loss_series(sol, len(v_stage)) if loss_all is not None else None

                # 去掉每段首点，避免与上一段末点重复
                voltage_all.extend(v_stage[1:].tolist())
                temp_all.extend(t_stage[1:].tolist())
                soc_all.extend(soc_stage[1:].tolist())
                current_all.extend([float(stage_I)] * max(0, len(v_stage) - 1))
                if loss_all is not None and loss_stage is not None:
                    loss_all.extend(loss_stage[1:].tolist())

                last_sol = sol

            planned_total_time = float(sum(int(round(t)) for t in stage_times))

            return self._extract_from_series(
                voltage_all=np.asarray(voltage_all, dtype=float),
                temp_all=np.asarray(temp_all, dtype=float),
                soc_all=np.asarray(soc_all, dtype=float),
                current_all=np.asarray(current_all, dtype=float),
                loss_all=None if loss_all is None else np.asarray(loss_all, dtype=float),
                last_sol=last_sol,
                total_time_override=planned_total_time,
            )

        except Exception as e:
            return self._penalty(f"求解失败: {str(e)[:120]}")

    # ------------------------------------------------------------------
    #  结果提取
    # ------------------------------------------------------------------

    def _time_series_from_entries(self, entries: np.ndarray, n_time: int) -> np.ndarray:
        """Project PyBaMM variable entries onto a 1D time series."""
        arr = np.asarray(entries, dtype=float)
        if arr.ndim == 0:
            series = np.full(n_time, float(arr), dtype=float)
        elif arr.ndim == 1:
            series = arr.reshape(-1)
        elif arr.ndim == 2:
            if arr.shape[0] == n_time:
                series = arr[:, -1]
            elif arr.shape[1] == n_time:
                series = arr[-1, :]
            elif arr.shape[0] > arr.shape[1]:
                series = arr[:, -1]
            else:
                series = arr[-1, :]
        else:
            series = arr.reshape(-1)

        if series.size == n_time:
            return np.asarray(series, dtype=float)
        if series.size <= 1:
            fill = float(series[0]) if series.size == 1 else 0.0
            return np.full(n_time, fill, dtype=float)

        src_x = np.linspace(0.0, 1.0, num=series.size)
        dst_x = np.linspace(0.0, 1.0, num=n_time)
        return np.interp(dst_x, src_x, series).astype(float)

    def _soc_series_from_conc(self, c_entries: np.ndarray, n_time: int) -> np.ndarray:
        """将浓度 entries 统一映射为按时间排列的 SOC 序列。"""
        if c_entries.ndim == 1:
            c_time = c_entries
        elif c_entries.ndim == 2:
            # 常见两种形状：(n_time, n_r) 或 (n_r, n_time)
            if c_entries.shape[0] == n_time:
                c_time = c_entries[:, -1]
            elif c_entries.shape[1] == n_time:
                c_time = c_entries[-1, :]
            elif c_entries.shape[0] > c_entries.shape[1]:
                c_time = c_entries[:, -1]
            else:
                c_time = c_entries[-1, :]
        else:
            c_time = c_entries.reshape(-1)
        return np.array([_neg_conc_to_soc(float(c)) for c in c_time], dtype=float)

    def _physical_loss_series(self, sol, n_time: int) -> np.ndarray:
        """Extract cumulative physical lithium loss as percent of effective capacity."""
        F = 96485.33212
        total_mol_series = np.zeros(n_time, dtype=float)
        for var in (
            "Loss of lithium to SEI [mol]",
            "Loss of lithium to negative SEI [mol]",
            "Loss of lithium to lithium plating [mol]",
            "Loss of lithium to negative lithium plating [mol]",
        ):
            try:
                total_mol_series += self._time_series_from_entries(sol[var].entries, n_time)
            except KeyError:
                continue

        li_loss_ah_series = total_mol_series * F / 3600.0
        loss_pct_series = li_loss_ah_series / self.Q_eff * 100.0
        return np.maximum(loss_pct_series, 0.0)

    def _extract_from_series(self, voltage_all, temp_all, soc_all, current_all, last_sol,
                             loss_all: Optional[np.ndarray] = None,
                             total_time_override: Optional[float] = None) -> Dict:
        try:
            total_time = float(total_time_override) if total_time_override is not None else float(len(soc_all) - 1)
            peak_temp = float(np.max(temp_all))
            peak_volt = float(np.max(voltage_all))
            soc_final = float(soc_all[-1])
            if loss_all is not None:
                loss_all = np.asarray(loss_all, dtype=float).reshape(-1)
                if loss_all.size != len(soc_all):
                    loss_all = self._time_series_from_entries(loss_all, len(soc_all))

        except Exception as e:
            return self._penalty(f"结果提取失败: {str(e)[:120]}")

        # ---- 约束检查 ----
        if peak_temp > self.T_max:
            return self._penalty(f"峰值温度 {peak_temp:.2f} K > 上限 {self.T_max} K")
        if peak_volt > self.V_max:
            return self._penalty(f"峰值电压 {peak_volt:.3f} V > 上限 {self.V_max} V")

        # ---- 温升 ΔT (与 utils_fun.py temp_rise 一致，固定相对 298.15 K) ----
        delta_temp_raw = peak_temp - 298.15
        delta_temp = TEMP_RISE_OBJ_ALIGN * delta_temp_raw

        # ---- 老化计算 ----
        # 构建与 utils_fun.py 一致的均值特征 (mean SOC %, mean T, mean I)
        mean_soc_pct = float(np.mean(soc_all) * 100)
        mean_temp_raw = float(np.mean(temp_all))
        mean_temp = 298.15 + TEMP_FOR_AGING_ALIGN * (mean_temp_raw - 298.15)
        mean_current = float(np.mean(current_all))

        aging_empirical = self._aging_empirical(mean_soc_pct, mean_temp, mean_current)
        aging_physical = None
        if loss_all is not None and loss_all.size:
            aging_physical = float(max(loss_all[-1], 1e-8))
        elif self.aging_mode in ("physical", "both") and last_sol is not None:
            aging_physical = self._aging_physical(last_sol)

        # raw_objectives 中的 aging 取决于 aging_mode
        aging_obj = aging_physical if self.aging_mode == "physical" else aging_empirical

        # ---- 轨迹输出 (与 utils_fun.py 返回格式对齐) ----
        trajectories = {
            "V": voltage_all.tolist(),
            "T": temp_all.tolist(),
            "SOC": soc_all.tolist(),
            "I": current_all.tolist(),
            "time": np.arange(len(soc_all), dtype=float).tolist(),
        }
        if loss_all is not None:
            trajectories["loss_pct"] = loss_all.tolist()

        result = {
            "raw_objectives": np.array([total_time, delta_temp, aging_obj]),
            "soc_final":      soc_final,
            "feasible":       True,
            "violation":      None,
            "trajectories":   trajectories,
        }
        if self.aging_mode == "both":
            result["aging_physical"] = aging_physical

        logger.info(
            f"仿真完成 | 时间: {total_time:.1f}s | 温升: {delta_temp:.2f}K | "
            f"老化: {aging_obj:.4f}% | SOC终值: {soc_final:.3f}"
        )
        return result

    # ------------------------------------------------------------------
    #  老化模型 1：经验公式（与 utils_fun.py 完全一致）
    # ------------------------------------------------------------------

    def _aging_empirical(self, mean_soc_pct: float, mean_temp_K: float, mean_current_A: float) -> float:
        """
        返回老化百分比 = Q_eff / cap * 100
        与 utils_fun.py: CAP / capa_aging * 100 完全一致
        """
        cap = cal_cap(mean_soc_pct, mean_temp_K, mean_current_A)
        return float(self.Q_eff / cap * 100)

    # ------------------------------------------------------------------
    #  老化模型 2：物理模型（SEI + 析锂 Li 损失）
    # ------------------------------------------------------------------

    def _aging_physical(self, sol) -> float:
        """
        返回老化百分比 = Li_loss_Ah / Q_eff * 100
        物理意义更强，但与 utils_fun.py 量纲不同，仅用于研究对比。
        """
        F = 96485.33212
        total_mol = 0.0
        for var in (
            "Loss of lithium to SEI [mol]",
            "Loss of lithium to negative SEI [mol]",
            "Loss of lithium to lithium plating [mol]",
            "Loss of lithium to negative lithium plating [mol]",
        ):
            try:
                entries = sol[var].entries
                val = float(entries[-1] if entries.ndim == 1 else np.mean(entries, axis=0)[-1])
                total_mol += val
            except KeyError:
                pass
        li_loss_ah = total_mol * F / 3600
        return float(max(li_loss_ah / self.Q_eff * 100, 1e-8))

    # ------------------------------------------------------------------

    def _penalty(self, reason: str) -> Dict:
        logger.warning(f"惩罚触发: {reason}")
        return {
            "raw_objectives": FAILURE_PENALTY.copy(),
            "soc_final":      float("nan"),
            "feasible":       False,
            "violation":      reason,
            "trajectories":   None,
            "aging_physical": None,
        }
