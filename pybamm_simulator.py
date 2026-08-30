"""
pybamm_simulator.py — 三段恒流充电仿真器 (基于 SPMe.py 参数)
====================================================================
输入：θ = (I1, I2, I3, dSOC1, dSOC2)
    电流单位：A (绝对值，与 Chen2020 Nominal capacity 5 Ah 对应)
    如需 C 倍率输入，令 use_crate=True，则 Ix 视作 C 倍率，内部自动换算

输出:
    raw_objectives : [time_s, delta_temp_K, aging_%]   均 minimize
    soc_final      : float                              终止 SOC
    trajectories   : dict  V / T / SOC / I 轨迹 (与 utils_fun.py 对齐)

约束检查 (仿真器只做这两项):
    T_peak > T_max (默认 328.15 K / 55°C) → penalty
    V_peak > V_max (默认 4.4 V)           → penalty

参数来源:
    电化学参数 — Chen2020 + 辨识值 (SPM.py)
    热模型参数 — 辨识值 (SPM.py)
    老化经验式 — SPM.py cal_dQloss_pct()
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

from utils.constants import DEFAULT_BOUNDS, DSOC_SUM_MAX, FAILURE_PENALTY

logger = logging.getLogger(__name__)

# =========================================================
# 全局警告屏蔽 (与 SPM.py 一致)
# =========================================================
warnings.filterwarnings("ignore", message=".*No value provided for input.*")
warnings.filterwarnings("ignore", message=".*Signature b.*numpy\.longdouble.*")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
logging.getLogger("pybamm").setLevel(logging.ERROR)
logging.getLogger("pybamm.solvers").setLevel(logging.ERROR)
logging.getLogger("pybamm.solvers.base_solver").setLevel(logging.ERROR)
logging.disable(logging.WARNING)


# =========================================================
# SOC calculation (与 SPM.py 完全一致)
# =========================================================
def _neg_conc_to_soc(c: float) -> float:
    """负极颗粒平均浓度 → SOC（与 SPM.py cal_soc 完全一致）"""
    c_min = 872.9651389896292
    c_max = 30171.311359086325
    return float(np.clip((c - c_min) / (c_max - c_min), 0.0, 1.0))


# =========================================================
# 老化模型参数 (与 SPM.py 完全一致)
# Qloss_pct = (alpha * SOC + beta) * exp((-Ea + eta_a * Ic)/(R*T)) * Ah^z
# =========================================================
AGING_ALPHA = 1.1520
AGING_BETA = 0.4259
AGING_EA = 6002.3          # J/mol
AGING_ETA_A = 91.0281
AGING_Z = 0.5838
RGAS = 8.314


def _aging_severity(soc: float, temp_K: float, current_A: float) -> float:
    """
    老化严重程度因子 s(SOC, Ic, T)，与 SPM.py aging_severity 完全一致。

    Parameters
    ----------
    soc : float
        SOC fraction, 0~1 (若传入>1 会自动转换为小数)
    temp_K : float
        温度 [K]
    current_A : float
        电流 [A]

    Returns
    -------
    float
        严重程度因子 s
    """
    soc_frac = float(soc)
    if soc_frac > 1.0:
        soc_frac = soc_frac / 100.0
    soc_frac = np.clip(soc_frac, 0.0, 1.0)

    # 假设 Q_nom=5Ah（Chen2020 默认值）
    Q_nom = 5.0
    Ic = abs(float(current_A)) / Q_nom
    temp_K = float(temp_K)

    s = (AGING_ALPHA * soc_frac + AGING_BETA) * np.exp(
        (-AGING_EA + AGING_ETA_A * Ic) / (RGAS * temp_K)
    )

    return float(s)


def _cal_dQloss_pct(
    soc_avg: float,
    temp_K: float,
    current_A: float,
    duration_s: float,
    aging_Ah_old: float,
) -> tuple[float, float]:
    """
    增量容量损失百分比，与 SPM.py cal_dQloss_pct 完全一致。

    Parameters
    ----------
    soc_avg : float
        平均 SOC
    temp_K : float
        温度 [K]
    current_A : float
        电流 [A]
    duration_s : float
        持续时间 [s]
    aging_Ah_old : float
        之前的累计 Ah 通过量

    Returns
    -------
    tuple[float, float]
        (dQloss_pct, aging_Ah_new)
    """
    dAh = abs(float(current_A)) * float(duration_s) / 3600.0
    s = _aging_severity(soc_avg, temp_K, current_A)

    Ah_new = aging_Ah_old + dAh
    dQloss_pct = s * (Ah_new ** AGING_Z - aging_Ah_old ** AGING_Z)

    return float(dQloss_pct), Ah_new


# =========================================================
# 充电 SOC 窗口
# =========================================================
SOC_START = 0.0
SOC_END = 0.8
SOC_SPAN = SOC_END - SOC_START  # 0.8

# =========================================================
# Penalty 向量
# =========================================================


# =========================================================
# 主类
# =========================================================
class PyBaMMSimulator:
    """
    三段恒流充电仿真器，基于 SPM.py 参数实现。

    电流输入约定
    -----------
    默认 (use_crate=False): I1/I2/I3 单位为 A (绝对值)
    use_crate=True         : I1/I2/I3 为 C 倍率，内部换算 I_A = C * Q_eff

    参数集
    ------
    仅支持 "Chen2020"，使用 SPM.py 的辨识参数。
    """

    def __init__(
        self,
        Q_nom: float = None,  # None → 使用默认值 5.0 Ah
        SOH: float = 1.0,
        T_init: float = None,  # None → 使用默认值 298.15K
        V_init: float = 2.8,
        T_max: float = None,  # None → 使用默认值 328.15K
        V_max: float = None,  # None → 使用默认值 4.4V
        use_crate: bool = False,
        aging_weight: float = 10.0,
    ) -> None:
        """
        Parameters
        ----------
        Q_nom : float
            额定容量 [Ah]（SOH=1 时）。None → 使用 Chen2020 默认值 5.0 Ah
        SOH : float
            健康状态 (0, 1]，缩放有效容量与最大负极浓度
        T_init : float
            初始温度 [K]。None → 使用默认值 298.15K
        V_init : float
            初始电压 [V]
        T_max : float
            温度约束上限 [K]。None → 使用默认值 328.15K
        V_max : float
            电压约束上限 [V]。None → 使用默认值 4.4V
        use_crate : bool
            True → 输入 I 为 C 倍率；False → 输入 I 为绝对 A
        aging_weight : float
            老化奖励权重，用于 scaling dQloss_pct
        """
        if not PYBAMM_AVAILABLE:
            raise ImportError("PyBaMM 未安装：pip install pybamm")

        assert 0 < SOH <= 1.0, "SOH 必须在 (0, 1]"

        # 加载参数集元数据
        self.battery_name = "LG INR21700-M50"

        # 使用默认值或用户指定值
        self.Q_nom = Q_nom if Q_nom is not None else 5.0
        self.SOH = SOH
        self.Q_eff = self.Q_nom * SOH
        self.T_init = T_init if T_init is not None else 298.15
        self.V_init = V_init
        self.T_max = T_max if T_max is not None else 328.15
        self.V_max = V_max if V_max is not None else 4.4
        self.use_crate = use_crate
        self.aging_weight = aging_weight

        self.param_bounds = {
            key: tuple(bounds) for key, bounds in DEFAULT_BOUNDS.items()
        }
        self.soc_start = SOC_START
        self.soc_end = SOC_END
        self.dsoc_sum_max = DSOC_SUM_MAX

        # 初始化仿真状态
        self._sol = None
        self._step_cnt = 0

    # ------------------------------------------------------------------
    # 公共接口
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
            trajectories   : dict  {V, T, SOC, I} — 与 SPM.py 输出格式一致
        """
        rng = np.random.get_state()
        try:
            theta_arr = np.asarray(theta, dtype=float).ravel()
            if theta_arr.size < 5:
                return self._penalty("theta 必须包含 5 个决策变量")

            from utils.constants import DSOC3_MIN

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
    # 核心仿真
    # ------------------------------------------------------------------

    def _run(self, I1, I2, I3, dSOC1, dSOC2) -> Dict:
        """
        执行三段恒流充电仿真，严格对齐 SPM.py 参数和逻辑。
        """
        from utils.constants import DSOC3_MIN

        dSOC3 = SOC_SPAN - dSOC1 - dSOC2

        # 基础合法性检查（与 SPM.py 一致）
        if min(I1, I2, I3) <= 0:
            return self._penalty("电流必须为正")
        if min(dSOC1, dSOC2, dSOC3) <= 0:
            return self._penalty("dSOC 必须为正且 dSOC1+dSOC2<0.8")

        # 电流换算：与 SPM.py action_mode="current" 对齐
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

        # 步长计算严格对齐 SPM.py:
        # t = (5*SOH/I) * soc * 3600
        t1 = 5.0 * self.SOH / I1 * dSOC1 * 3600.0
        t2 = 5.0 * self.SOH / I2 * dSOC2 * 3600.0
        t3 = 5.0 * self.SOH / I3 * dSOC3 * 3600.0

        # 构建模型：SPMe with thermal=lumped
        model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
        param = pybamm.ParameterValues("Chen2020")

        # === 更新 SPM.py 的辨识参数 ===
        param.update(
            {
                "Current function [A]": "[input]",
                "Upper voltage cut-off [V]": 4.4,

                # Identified electrochemical parameters (SPM.py)
                'Negative particle radius [m]': 4.69e-06,
                'Negative electrode active material volume fraction': 0.73,
                'Negative electrode conductivity [S.m-1]': 258.00,
                'Negative electrode diffusivity [m2.s-1]': 3.96e-14,
                'Positive particle radius [m]': 4.17e-06,
                'Positive electrode active material volume fraction': 0.66,
                'Positive electrode conductivity [S.m-1]': 0.22,
                'Positive electrode diffusivity [m2.s-1]': 4.80e-15,

                # Identified thermal parameters (SPM.py)
                'Total heat transfer coefficient [W.m-2.K-1]': 17.36,
                'Separator specific heat capacity [J.kg-1.K-1]': 2905.50,
                'Negative electrode specific heat capacity [J.kg-1.K-1]': 2400.56,
                "Positive electrode specific heat capacity [J.kg-1.K-1]": 2715.82,
                'Negative current collector specific heat capacity [J.kg-1.K-1]': 1138.79,
                'Positive current collector specific heat capacity [J.kg-1.K-1]': 1252.81,
            }
        )

        # SOH-dependent available lithium / nominal capacity setting (SPM.py)
        param.update({
            'Maximum concentration in negative electrode [mol.m-3]': self.SOH * (33133 - 1308) + 1308
        })
        param.update({
            'Nominal cell capacity [A.h]': 5.0 * self.SOH
        })

        # Initialize from voltage (SPM.py)
        param["Initial temperature [K]"] = self.T_init
        param["Ambient temperature [K]"] = self.T_init

        try:
            param.set_initial_stoichiometries(f"{self.V_init} V")
        except Exception as e:
            logger.warning(f"set_initial_stoichiometries 失败，回退默认初值：{e}")

        try:
            # 分段逐次求解并拼接轨迹
            voltage_all = [self.V_init]
            temp_all = [self.T_init]
            soc_all = []
            current_all = [0.0]

            c0 = float(param["Initial concentration in negative electrode [mol.m-3]"])
            soc_all.append(_neg_conc_to_soc(c0))

            last_sol = None
            stage_currents = [I1_A, I2_A, I3_A]
            stage_times = [t1, t2, t3]

            # 老化跟踪变量
            aging_Ah = 0.0
            aging_dQloss_total = 0.0

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
                # R-averaged concentration is (n_r, n_t) shape - take last time step
                c_entries = np.asarray(sol["R-averaged negative particle concentration [mol.m-3]"].entries, dtype=float)
                if c_entries.ndim == 2:
                    c_time = c_entries[:, -1]  # Last time step across all radial points
                else:
                    c_time = c_entries.reshape(-1)
                soc_stage = np.array([_neg_conc_to_soc(float(c)) for c in c_time], dtype=float)

                # 去掉每段首点，避免与上一段末点重复
                voltage_all.extend(v_stage[1:].tolist())
                temp_all.extend(t_stage[1:].tolist())
                soc_all.extend(soc_stage[1:].tolist())
                current_all.extend([float(stage_I)] * max(0, len(v_stage) - 1))

                # 老化计算（对齐 SPM.py cal_dQloss_pct）
                # 使用当前阶段开始和结束的 SOC 平均值
                if len(soc_stage) > 1:
                    soc_start = soc_stage[0]
                    soc_end = soc_stage[-1]
                    soc_avg = 0.5 * (soc_start + soc_end)
                else:
                    soc_avg = soc_stage[0] if len(soc_stage) > 0 else soc_all[-1] if soc_all else 0.5

                dQloss, aging_Ah = _cal_dQloss_pct(
                    soc_avg=soc_avg,
                    temp_K=float(t_stage[-1]),
                    current_A=float(stage_I),
                    duration_s=float(st),
                    aging_Ah_old=float(aging_Ah),
                )
                aging_dQloss_total += dQloss

                last_sol = sol

            planned_total_time = float(sum(int(round(t)) for t in stage_times))

            return self._extract_from_series(
                voltage_all=np.asarray(voltage_all, dtype=float),
                temp_all=np.asarray(temp_all, dtype=float),
                soc_all=np.asarray(soc_all, dtype=float),
                current_all=np.asarray(current_all, dtype=float),
                last_sol=last_sol,
                total_time_override=planned_total_time,
                aging_dQloss_total=aging_dQloss_total,
            )

        except Exception as e:
            return self._penalty(f"求解失败：{str(e)[:120]}")

    # ------------------------------------------------------------------
    # 结果提取
    # ------------------------------------------------------------------

    def _extract_from_series(
        self,
        voltage_all,
        temp_all,
        soc_all,
        current_all,
        last_sol,
        total_time_override: Optional[float] = None,
        aging_dQloss_total: float = 0.0,
    ) -> Dict:
        """
        从轨迹中提取最终结果，对齐 SPM.py 的输出格式。
        """
        try:
            total_time = float(total_time_override) if total_time_override is not None else float(len(soc_all) - 1)
            peak_temp = float(np.max(temp_all))
            peak_volt = float(np.max(voltage_all))
            soc_final = float(soc_all[-1])

        except Exception as e:
            return self._penalty(f"结果提取失败：{str(e)[:120]}")

        # === 约束检查（与 SPM.py 一致）===
        if peak_temp > self.T_max:
            return self._penalty(f"峰值温度 {peak_temp:.2f} K > 上限 {self.T_max} K")
        if peak_volt > self.V_max:
            return self._penalty(f"峰值电压 {peak_volt:.3f} V > 上限 {self.V_max} V")

        # === 温升 ΔT（与 SPM.py reward 一致）===
        # SPM.py: r_temp = -abs(self.temp - self.sett['ambient temperature']) / 30.0
        delta_temp_raw = peak_temp - self.T_init

        # === 老化计算（对齐 SPM.py aging_weight 逻辑）===
        # SPM.py: r_aging = max(1.0 - aging_weight * dQloss_pct_raw, 0.0)
        aging_pct = self.aging_weight * aging_dQloss_total

        # === 轨迹输出（与 SPM.py traj 格式对齐）===
        trajectories = {
            "V": voltage_all.tolist(),
            "T": temp_all.tolist(),
            "SOC": soc_all.tolist(),
            "I": current_all.tolist(),
            "time": np.arange(len(soc_all), dtype=float).tolist(),
        }

        result = {
            "raw_objectives": np.array([total_time, delta_temp_raw, aging_pct]),
            "soc_final": soc_final,
            "feasible": True,
            "violation": None,
            "trajectories": trajectories,
        }

        logger.info(
            f"仿真完成 | 时间：{total_time:.1f}s | 温升：{delta_temp_raw:.2f}K | "
            f"老化：{aging_pct:.4f}% | SOC 终值：{soc_final:.3f}"
        )
        return result

    # ------------------------------------------------------------------
    # 惩罚处理
    # ------------------------------------------------------------------

    def _penalty(self, reason: str) -> Dict:
        logger.warning(f"惩罚触发：{reason}")
        return {
            "raw_objectives": FAILURE_PENALTY.copy(),
            "soc_final": float("nan"),
            "feasible": False,
            "violation": reason,
            "trajectories": None,
        }
