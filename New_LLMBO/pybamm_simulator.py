"""
pybamm_simulator.py — LLAMBO-MO 充电协议仿真器
==============================================
决策变量: θ = (I₁, SOC₁, I₂)
  I₁   ∈ [3, 7]  A       第一阶段恒流电流
  SOC₁ ∈ [0.10, 0.70]    阶段切换 SOC
  I₂   ∈ [1, 5]  A       第二阶段恒流电流

充电协议: CC₁ → CC₂ (两阶段恒流)
  阶段 1: 以 I₁ 充电, SOC0 → SOC₁
  阶段 2: 以 I₂ 充电, SOC₁ → SOC_end (0.80)

输出 (三目标, 均 minimize):
  f₁ = 充电时间 [s]
  f₂ = 峰值温度 [K]
  f₃ = 老化程度 [%]  (SEI + 锂沉积 锂损失)

约束:
  V ≤ 4.3 V
  T ≤ 328.15 K  (55 °C)
  违规 → 返回固定惩罚值: (7200 s, 338 K, 0.5 %)
"""

try:
    import pybamm
    PYBAMM_AVAILABLE = True
except ImportError:
    pybamm = None
    PYBAMM_AVAILABLE = False

import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================
#  默认配置
# ============================================================
DEFAULT_PARAM_BOUNDS = {
    "I1":   (3.0, 7.0),      # A
    "SOC1": (0.10, 0.70),    # 无量纲
    "I2":   (1.0, 5.0),      # A
}

DEFAULT_BATTERY_CONFIG = {
    "param_set":        "Chen2020",
    "nominal_capacity": 5.0,       # Ah
    "init_voltage":     3.0,       # V  (≈SOC 0.035)
    "init_temp":        298.15,    # K  (25 °C)
    "ambient_temp":     298.15,    # K
    "soc_end":          0.80,      # 目标 SOC
}

DEFAULT_CONSTRAINTS = {
    "voltage_max": 4.3,     # V
    "temp_max":    328.15,  # K  (55 °C)
}

DEFAULT_PENALTY = {
    "time":  7200.0,  # s   (2 h — 极慢)
    "temp":  338.0,   # K   (65 °C)
    "aging": 0.5,     # %   (单周期惩罚)
}

SEI_ACTIVATION_ENERGY = 37500.0  # J/mol


# ============================================================
#  仿真器
# ============================================================
class PyBaMMSimulator:
    """
    LLAMBO-MO 充电协议仿真器

    核心接口:
        evaluate(theta) → dict
            theta = [I₁, SOC₁, I₂]
            返回:
              raw_objectives : np.array([time_s, temp_K, aging_pct])
              feasible       : bool
              violation      : str | None
              details        : dict | None
    """

    def __init__(
        self,
        battery_config: Optional[Dict] = None,
        constraints:    Optional[Dict] = None,
        penalty:        Optional[Dict] = None,
    ):
        if not PYBAMM_AVAILABLE:
            raise ImportError(
                "PyBaMM is not installed. Please install it using: "
                "pip install pybamm"
            )

        self.battery     = {**DEFAULT_BATTERY_CONFIG, **(battery_config or {})}
        self.constraints = {**DEFAULT_CONSTRAINTS,    **(constraints or {})}
        self.penalty     = {**DEFAULT_PENALTY,        **(penalty or {})}

        self.Q_nom_Ah = self.battery["nominal_capacity"]
        self.Q_nom_C  = self.Q_nom_Ah * 3600.0

        # 从初始电压精确计算 SOC0
        self.soc0 = self._compute_soc0()

        logger.info(
            f"PyBaMMSimulator 初始化完毕  "
            f"V0={self.battery['init_voltage']}V → SOC0≈{self.soc0:.4f}  "
            f"SOC_end={self.battery['soc_end']}"
        )

    # ----------------------------------------------------------
    #  公共接口
    # ----------------------------------------------------------
    def evaluate(self, theta) -> Dict:
        """
        评估单条充电协议

        参数
        ----
        theta : array-like, shape (3,)
            [I₁(A), SOC₁, I₂(A)]

        返回
        ----
        dict
            raw_objectives : np.ndarray, shape (3,)
            feasible       : bool
            violation      : str | None
            details        : dict | None
        """
        # PyBaMM 内部可能调用 np.random.seed(), 破坏全局随机状态
        rng_state = np.random.get_state()
        try:
            theta = np.asarray(theta, dtype=float)
            I1, SOC1, I2 = float(theta[0]), float(theta[1]), float(theta[2])
            return self._run(I1, SOC1, I2)
        except Exception as e:
            logger.error(f"evaluate 意外异常: {e}")
            return self._make_penalty(f"unexpected: {str(e)[:200]}")
        finally:
            np.random.set_state(rng_state)

    def evaluate_batch(self, thetas) -> list:
        """批量评估 (顺序执行)"""
        return [self.evaluate(th) for th in thetas]

    # ----------------------------------------------------------
    #  核心仿真
    # ----------------------------------------------------------
    def _run(self, I1: float, SOC1: float, I2: float) -> Dict:

        # ---- 1. 参数验证 ----
        if I1 <= 0 or I2 <= 0:
            return self._make_penalty("invalid: current <= 0")
        if SOC1 <= self.soc0:
            return self._make_penalty(
                f"invalid: SOC1={SOC1:.3f} <= SOC0={self.soc0:.3f}"
            )
        if SOC1 >= self.battery["soc_end"]:
            return self._make_penalty(
                f"invalid: SOC1={SOC1:.3f} >= SOC_end={self.battery['soc_end']}"
            )

        # ---- 2. 近似阶段时间 ----
        # t = ΔSOC × Q_nom(Ah) × 3600 / I(A)
        t1 = (SOC1 - self.soc0) * self.Q_nom_Ah * 3600.0 / I1
        t2 = (self.battery["soc_end"] - SOC1) * self.Q_nom_Ah * 3600.0 / I2

        # 10 % 安全余量 (实际靠电压截止保护)
        t1_safe = t1 * 1.10
        t2_safe = t2 * 1.10

        V_max = self.constraints["voltage_max"]

        # ---- 3. 构建 SPMe 模型 ----
        # 注意: Chen2020 是圆柱电池(LG M50), "x-lumped" 要求软包几何,
        # 所以用 "lumped" (0D 集总热模型, 通过散热系数耦合)
        model = pybamm.lithium_ion.SPMe(
            options={
                "thermal":           "lumped",
                "SEI":               "reaction limited",
                "lithium plating":   "irreversible",
            }
        )

        # ---- 4. 参数值 ----
        param = pybamm.ParameterValues(self.battery["param_set"])
        # 硬保护比约束高 0.1 V, 让 PyBaMM 不会提前终止
        param["Upper voltage cut-off [V]"]             = V_max + 0.1
        param["SEI growth activation energy [J.mol-1]"] = SEI_ACTIVATION_ENERGY
        param["Initial temperature [K]"]                = self.battery["init_temp"]
        param["Ambient temperature [K]"]                = self.battery["ambient_temp"]

        # ---- 4b. 补丁: Chen2020 缺失的析锂参数 (来源: O'Kane 2022) ----
        # Chen2020 只标定了基础 SEI, 未包含锂沉积动力学参数
        # 从 O'Kane et al. (2022) 借用典型值, 保持科研严谨性
        plating_params = {
            # 锂沉积交换电流密度 [A/m²]
            # O'Kane2022 典型值, 控制析锂动力学速率
            "Exchange-current density for plating [A.m-2]": 0.001,
            # 锂沉积开路电位 [V vs Li/Li+], 标准定义 = 0
            "Lithium plating open-circuit potential [V]": 0.0,
            # 死锂衰减常数 [1/s], 不可逆析锂的锂钝化速率
            "Dead lithium decay constant [s-1]": 3.33e-7,
            # 锂沉积传递系数, 控制 Butler-Volmer 不对称性
            "Lithium plating transfer coefficient": 0.65,
            # 初始死锂浓度 [mol/m³], 新电池 = 0
            "Initial dead lithium concentration [mol.m-3]": 0.0,
            # 初始析出锂浓度 [mol/m³], 新电池 = 0
            # PyBaMM 24.1 irreversible plating 子模型必填
            "Initial plated lithium concentration [mol.m-3]": 0.0,
            # 锂金属摩尔体积 [m³/mol]
            # Li 密度 ~534 kg/m³, 摩尔质量 6.941e-3 kg/mol → 13.0e-6 m³/mol
            "Lithium metal partial molar volume [m3.mol-1]": 1.3e-5,
            # ── PyBaMM 24.1 新增必填项 ──────────────────────────────────
            # 典型析锂参考浓度 [mol/m³], 用于 irreversible plating 子模型的
            # 非量纲化 (c_Li_typ)。
            # 来源: O'Kane et al. (2022) Table S1, c_Li_typ = 1000 mol/m³
            # 物理意义: 析出锂层的参考浓度尺度, 影响无量纲析锂电流表达式,
            #           不影响有量纲物理量的量级。
            "Typical plated lithium concentration [mol.m-3]": 1000.0,
        }
        param.update(plating_params, check_already_exists=False)
        logger.info(
            "已注入 O'Kane2022 析锂参数: "
            f"j0_plating={plating_params['Exchange-current density for plating [A.m-2]']} A/m²  "
            f"c_Li_typ={plating_params['Typical plated lithium concentration [mol.m-3]']} mol/m³"
        )

        try:
            param.set_initial_stoichiometries(
                f"{self.battery['init_voltage']} V"
            )
        except Exception as e:
            logger.warning(f"set_initial_stoichiometries 失败: {e}")

        # ---- 5. Experiment ----
        experiment = pybamm.Experiment([
            (
                f"Charge at {I1:.4f} A for {t1_safe:.1f} seconds "
                f"or until {V_max} V"
            ),
            (
                f"Charge at {I2:.4f} A for {t2_safe:.1f} seconds "
                f"or until {V_max} V"
            ),
        ])

        # ---- 6. 求解 ----
        try:
            # 使用 IDAKLUSolver 替代默认的 CasadiSolver
            try:
                solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
            except Exception:
                # IDAKLU 不可用，回退到 ScipySolver
                solver = pybamm.ScipySolver(atol=1e-6, rtol=1e-6)

            sim = pybamm.Simulation(
                model,
                experiment=experiment,
                parameter_values=param,
                solver=solver,
            )
            sol = sim.solve()
            if sol is None:
                return self._make_penalty("solve_failed: solution is None")
        except Exception as e:
            return self._make_penalty(f"solve_error: {str(e)[:200]}")

        # ---- 7. 提取 & 约束检查 ----
        return self._extract(sol)

    # ----------------------------------------------------------
    #  结果提取
    # ----------------------------------------------------------
    def _extract(self, sol) -> Dict:
        """从 PyBaMM Solution 提取三目标 + 约束检查"""

        try:
            # 充电时间
            time_entries = sol["Time [s]"].entries
            total_time   = float(time_entries[-1])

            # 峰值温度 (lumped 模型用 "Cell temperature [K]",
            # x-lumped 用 "X-averaged cell temperature [K]")
            for _temp_var in (
                "Cell temperature [K]",
                "X-averaged cell temperature [K]",
                "Volume-averaged cell temperature [K]",
            ):
                try:
                    temp_entries = sol[_temp_var].entries
                    break
                except KeyError:
                    continue
            else:
                raise KeyError("无法找到温度变量")
            peak_temp = float(np.max(temp_entries))

            # 峰值电压 (约束检查用)
            voltage_entries = sol["Voltage [V]"].entries
            peak_voltage    = float(np.max(voltage_entries))

            # 老化
            aging = self._extract_aging(sol)

        except Exception as e:
            return self._make_penalty(f"extract_error: {str(e)[:200]}")

        # ---- 约束 ----
        V_max = self.constraints["voltage_max"]
        T_max = self.constraints["temp_max"]

        if peak_voltage > V_max:
            return self._make_penalty(
                f"voltage={peak_voltage:.3f}V > {V_max}V"
            )
        if peak_temp > T_max:
            return self._make_penalty(
                f"temp={peak_temp:.2f}K > {T_max}K"
            )

        # ---- 成功 ----
        raw = np.array([total_time, peak_temp, max(aging, 1e-8)])

        return {
            "raw_objectives": raw,
            "feasible":       True,
            "violation":      None,
            "details": {
                "time":         total_time,
                "temp":         peak_temp,
                "aging":        max(aging, 1e-8),
                "peak_voltage": peak_voltage,
                "total_steps":  len(time_entries),
            },
        }

    # ----------------------------------------------------------
    #  老化提取
    # ----------------------------------------------------------
    def _extract_aging(self, sol) -> float:
        """
        从 SEI + 锂沉积提取容量损失百分比

        aging(%) = (Q_loss / Q_nom) × 100
        Q_loss   = Σ(li_loss_mol) × F / 3600
        """
        F = 96485.33212  # C/mol
        total_loss_mol = 0.0

        # SEI 锂损失 (兼容多版本变量名)
        for name in [
            "Loss of lithium to SEI [mol]",
            "Loss of lithium to negative SEI [mol]",
            "Loss of lithium to negative SEI on cracks [mol]",
        ]:
            val = self._safe_final(sol, name)
            if val is not None:
                total_loss_mol += val

        # 锂沉积损失
        for name in [
            "Loss of lithium to lithium plating [mol]",
            "Loss of lithium to negative lithium plating [mol]",
        ]:
            val = self._safe_final(sol, name)
            if val is not None:
                total_loss_mol += val

        if total_loss_mol <= 0:
            return 1e-8

        Q_loss_Ah = total_loss_mol * F / 3600.0
        return float(max((Q_loss_Ah / self.Q_nom_Ah) * 100.0, 1e-8))

    @staticmethod
    def _safe_final(sol, var_name: str) -> Optional[float]:
        """安全读取变量的最终时刻值"""
        try:
            entries = sol[var_name].entries
            if entries.ndim > 1:
                entries = np.mean(entries, axis=0)  # 空间平均
            return float(entries[-1])
        except (KeyError, IndexError):
            return None

    # ----------------------------------------------------------
    #  SOC0 计算
    # ----------------------------------------------------------
    def _compute_soc0(self) -> float:
        """
        从 init_voltage 精确计算 SOC0

        原理: 调用 set_initial_stoichiometries 后读取
        负极初始浓度, 然后映射到 SOC
        """
        try:
            param = pybamm.ParameterValues(self.battery["param_set"])
            c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]

            # 读取默认满充浓度 (100 % SOC)
            c_n_full = param[
                "Initial concentration in negative electrode [mol.m-3]"
            ]
            x_n_full = c_n_full / c_n_max  # ≈ 0.9014 for Chen2020

            # 设置到 init_voltage
            param.set_initial_stoichiometries(
                f"{self.battery['init_voltage']} V"
            )
            c_n_init = param[
                "Initial concentration in negative electrode [mol.m-3]"
            ]
            x_n_init = c_n_init / c_n_max

            # Chen2020 空电化学计量比 (≈0% SOC)
            # 从 OCP 曲线推算: 在 2.5 V 下 x_n ≈ 0.028
            x_n_empty = 0.028

            soc0 = (x_n_init - x_n_empty) / (x_n_full - x_n_empty)
            soc0 = float(np.clip(soc0, 0.0, 1.0))

            logger.info(
                f"SOC0 计算: x_n_init={x_n_init:.4f}, "
                f"x_n_full={x_n_full:.4f}, x_n_empty={x_n_empty:.3f} "
                f"→ SOC0={soc0:.4f}"
            )
            return soc0

        except Exception as e:
            logger.warning(f"SOC0 计算失败 ({e}), 使用近似值 0.04")
            return 0.04

    # ----------------------------------------------------------
    #  惩罚结果
    # ----------------------------------------------------------
    def _make_penalty(self, violation: str) -> Dict:
        """违规/失败 → 返回固定惩罚值 + 标记"""
        logger.warning(f"惩罚: {violation}")
        raw = np.array([
            self.penalty["time"],
            self.penalty["temp"],
            self.penalty["aging"],
        ])
        return {
            "raw_objectives": raw,
            "feasible":       False,
            "violation":      violation,
            "details":        None,
        }

    # ----------------------------------------------------------
    #  属性
    # ----------------------------------------------------------
    @property
    def param_bounds(self) -> Dict:
        """决策变量边界 (SOC1 下界跟随 SOC0 动态调整)"""
        soc1_lo = max(self.soc0 + 0.05, DEFAULT_PARAM_BOUNDS["SOC1"][0])
        return {
            "I1":   DEFAULT_PARAM_BOUNDS["I1"],
            "SOC1": (soc1_lo, DEFAULT_PARAM_BOUNDS["SOC1"][1]),
            "I2":   DEFAULT_PARAM_BOUNDS["I2"],
        }

    @property
    def soc0_value(self) -> float:
        """供其他模块使用的 SOC0 值"""
        return self.soc0


# ============================================================
#  命令行测试
# ============================================================
if __name__ == "__main__":
    import time as _time

    if not PYBAMM_AVAILABLE:
        print("=" * 60)
        print("错误：PyBaMM 未安装")
        print("=" * 60)
        print("\n请使用以下命令安装 PyBaMM:")
        print("  pip install pybamm")
        print("\n或者使用 conda:")
        print("  conda install -c conda-forge pybamm")
        print("\n注意：PyBaMM 需要 Python 3.9 或更高版本")
        exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("LLAMBO-MO PyBaMM 仿真器测试")
    print("=" * 60)

    print(f"\nPyBaMM version: {pybamm.__version__}")
    sim = PyBaMMSimulator()
    print(f"SOC0 = {sim.soc0:.4f}")
    print(f"参数边界: {sim.param_bounds}")

    tests = [
        ("保守 (低电流)", [3.5, 0.40, 2.0]),
        ("平衡",         [5.0, 0.35, 2.5]),
        ("激进 (高电流)", [7.0, 0.20, 4.0]),
        ("极端 (可能违规)", [7.0, 0.15, 5.0]),
    ]

    results = []
    for name, theta in tests:
        print(f"\n--- {name}: I1={theta[0]}A, SOC1={theta[1]}, I2={theta[2]}A ---")
        t0 = _time.time()
        res = sim.evaluate(theta)
        dt = _time.time() - t0

        results.append((name, theta, res))

        if res["feasible"]:
            obj = res["raw_objectives"]
            d   = res["details"]
            print(f"  ✓ 充电时间: {obj[0]:.0f}s ({obj[0]/60:.1f}min)")
            print(f"    峰值温度: {obj[1]:.2f}K ({obj[1]-273.15:.1f}°C)")
            print(f"    老化程度: {obj[2]:.6f}%")
            print(f"    峰值电压: {d['peak_voltage']:.3f}V")
        else:
            print(f"  ✗ 违规: {res['violation']}")
            obj = res["raw_objectives"]
            print(f"    惩罚值: t={obj[0]}s, T={obj[1]}K, aging={obj[2]}%")

        print(f"  耗时: {dt:.2f}s")

    # ---- 物理一致性 ----
    print("\n" + "=" * 60)
    print("物理一致性检查")
    print("=" * 60)

    feasible = [(n, t, r) for n, t, r in results if r["feasible"]]
    if len(feasible) >= 2:
        # 电流越大 → 时间越短
        sorted_by_i1 = sorted(feasible, key=lambda x: x[1][0])
        t_low  = sorted_by_i1[0][2]["raw_objectives"][0]
        t_high = sorted_by_i1[-1][2]["raw_objectives"][0]
        print(f"  时间: I1={sorted_by_i1[0][1][0]}A → {t_low:.0f}s, "
              f"I1={sorted_by_i1[-1][1][0]}A → {t_high:.0f}s  "
              f"{'✓' if t_high < t_low else '⚠'}")

        # 电流越大 → 温度越高
        temp_low  = sorted_by_i1[0][2]["raw_objectives"][1]
        temp_high = sorted_by_i1[-1][2]["raw_objectives"][1]
        print(f"  温度: I1={sorted_by_i1[0][1][0]}A → {temp_low:.2f}K, "
              f"I1={sorted_by_i1[-1][1][0]}A → {temp_high:.2f}K  "
              f"{'✓' if temp_high > temp_low else '⚠'}")

        # 电流越大 → 老化越高
        aging_low  = sorted_by_i1[0][2]["raw_objectives"][2]
        aging_high = sorted_by_i1[-1][2]["raw_objectives"][2]
        print(f"  老化: I1={sorted_by_i1[0][1][0]}A → {aging_low:.6f}%, "
              f"I1={sorted_by_i1[-1][1][0]}A → {aging_high:.6f}%  "
              f"{'✓' if aging_high > aging_low else '⚠'}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)