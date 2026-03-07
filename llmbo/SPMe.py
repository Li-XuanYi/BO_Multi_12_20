"""
PyBaMM 仿真封装器 v2（分步仿真版）
输入：充电策略 [I1, SOC1, I2]
输出：三目标 [Time(s), Peak_Temp(K), log_Aging]

充电协议：CC1 → CC2，带不可逆安全衰减
- CC1: 以 I1 恒流充电，直到 SOC 达到 SOC1
- CC2: 以 I2 恒流充电，直到 SOC 达到 soc_target (0.8)
- 安全覆盖（不可逆）：当 V ≥ 4.2V 或 T ≥ 50°C 时触发，
  I_actual = min(I_base, I_trigger × exp(-λ(t - t_trigger)))
  λ = 0.001，触发后不恢复

终止条件：SOC ≥ 0.8 | 电流 < cutoff | 超时 7200s

老化模型：PyBaMM 内置 SEI (solvent-diffusion limited)
老化输出：log10(capacity_loss_pct)，便于优化器区分
"""

import pybamm
import numpy as np
import warnings
from typing import Dict

try:
    from ..config import BATTERY_CONFIG
except ImportError:
    try:
        from config import BATTERY_CONFIG
    except ImportError:
        BATTERY_CONFIG = {
            'param_set': 'Chen2020',
            'init_voltage': 3.0,
            'init_temp': 298.15,
            'sample_time': 90.0,
            'voltage_max': 4.2,
            'temp_max': 318.15,
            'soc_target': 0.80,
        }


class BatterySimulator:
    """
    电池仿真器 v2（分步仿真 + 安全衰减）

    与 v1 的主要区别：
    1. 输入从 [I1, T1, I2, V_switch] 变为 [I1, SOC1, I2]
    2. 使用 sim.step() 分步仿真替代 Experiment API
    3. 电压/温度安全约束改为不可逆指数衰减（非硬切断）
    4. Phase 判断基于仿真状态而非电流值猜测
    5. SOC 从参数集计算而非硬编码
    6. 老化输出增加 log 变换
    """

    def __init__(
        self,
        param_set: str = None,
        init_voltage: float = None,
        init_temp: float = None,
        soc_target: float = None,
        # 安全衰减参数
        decay_lambda: float = 0.001,
        voltage_decay_trigger: float = 4.2,
        temp_decay_trigger: float = 323.15,  # 50°C
        # 仿真控制
        dt: float = 5.0,
        max_time: float = 7200.0,
        cutoff_current: float = 0.1,
        # SEI 参数
        sei_activation_energy: float = 37500.0,
    ):
        """
        参数:
            param_set: PyBaMM 参数集名称
            init_voltage: 初始电压 [V]（用于设定初始 stoichiometry）
            init_temp: 初始/环境温度 [K]
            soc_target: 目标 SOC（默认 0.8）
            decay_lambda: 衰减系数 λ [1/s]
            voltage_decay_trigger: 电压衰减触发阈值 [V]
            temp_decay_trigger: 温度衰减触发阈值 [K]
            dt: 仿真步长 [s]
            max_time: 最大仿真时间 [s]（兜底）
            cutoff_current: 最小电流阈值 [A]（低于此值视为无法继续充电）
            sei_activation_energy: SEI 生长活化能 [J/mol]
        """
        self.param_set = param_set or BATTERY_CONFIG['param_set']
        self.init_voltage = init_voltage if init_voltage is not None else BATTERY_CONFIG['init_voltage']
        self.init_temp = init_temp if init_temp is not None else BATTERY_CONFIG['init_temp']
        self.soc_target = soc_target if soc_target is not None else BATTERY_CONFIG['soc_target']

        self.decay_lambda = decay_lambda
        self.voltage_decay_trigger = voltage_decay_trigger
        self.temp_decay_trigger = temp_decay_trigger
        self.dt = dt
        self.max_time = max_time
        self.cutoff_current = cutoff_current
        self.sei_activation_energy = sei_activation_energy

        # 从参数集读取的值（在第一次 simulate 时填充）
        self.nominal_capacity = None  # Ah

    # ================================================================
    # 公开接口
    # ================================================================

    def simulate(self, current1: float, soc1: float, current2: float) -> Dict:
        """
        执行 CC1→CC2（带安全衰减）充电协议

        参数:
            current1: CC1 阶段电流 [A]，范围 (0, 6)
            soc1: CC1→CC2 切换 SOC，范围 (0, soc_target)
            current2: CC2 阶段电流 [A]，范围 (0, 6)

        返回:
            结果字典
        """
        # 保护 numpy 随机状态（PyBaMM 会调用 np.random.seed）
        rng_state = np.random.get_state()
        try:
            return self._simulate_inner(current1, soc1, current2)
        finally:
            np.random.set_state(rng_state)

    # ================================================================
    # 内部实现
    # ================================================================

    def _simulate_inner(self, current1: float, soc1: float, current2: float) -> Dict:
        """仿真核心逻辑"""

        # -------- 1. 参数验证 --------
        current1, soc1, current2 = float(current1), float(soc1), float(current2)

        if current1 <= 0 or current2 <= 0:
            return self._make_failure_result("invalid_params: current must be > 0")
        if not (0 < soc1 < self.soc_target):
            return self._make_failure_result(
                f"invalid_params: soc1={soc1} must be in (0, {self.soc_target})"
            )

        # -------- 2. 构建模型与仿真器 --------
        try:
            sim, param = self._build_simulation()
        except Exception as e:
            return self._make_failure_result(f"build_error: {str(e)[:200]}")

        # -------- 3. 分步仿真循环 --------
        # 状态变量
        soc = 0.0  # Chen2020 中 3.0V ≈ 0% SOC
        t = 0.0
        decay_triggered = False
        t_trigger = 0.0
        I_at_trigger = 0.0

        # Profile 记录
        time_rec = []
        voltage_rec = []
        temp_rec = []
        current_rec = []
        soc_rec = []
        phase_rec = []

        termination_reason = "soc_reached"

        while t < self.max_time:
            # 3a. 确定当前阶段和基础电流
            if soc < soc1:
                I_base = current1
                phase = 'CC1'
            else:
                I_base = current2
                phase = 'CC2'

            # 3b. 应用衰减（如已触发）
            if decay_triggered:
                I_decayed = I_at_trigger * np.exp(
                    -self.decay_lambda * (t - t_trigger)
                )
                I_actual = min(I_base, I_decayed)
                # 标记阶段是否受衰减限制
                if I_decayed < I_base:
                    phase = phase + '_DECAY'
            else:
                I_actual = I_base

            # 3c. 电流过小 → 无法继续充电
            if I_actual < self.cutoff_current:
                termination_reason = "current_cutoff"
                break

            # 3d. 执行一个时间步
            try:
                sim.step(
                    self.dt,
                    inputs={"Current function [A]": -I_actual},  # 负值=充电
                    npts=2,
                )
            except Exception as e:
                # Solver 失败（例如电压超上限）：记录已有数据并退出
                termination_reason = f"solver_error: {str(e)[:200]}"
                break

            t += self.dt

            # 3e. 从解中提取瞬时值
            sol = sim.solution
            V = float(sol["Voltage [V]"].entries[-1])
            T = float(sol["X-averaged cell temperature [K]"].entries[-1])

            # 3f. 库仑计数更新 SOC
            soc += I_actual * self.dt / (self.nominal_capacity * 3600.0)

            # 3g. 记录
            time_rec.append(t)
            voltage_rec.append(V)
            temp_rec.append(T)
            current_rec.append(I_actual)
            soc_rec.append(soc)
            phase_rec.append(phase)

            # 3h. 检查衰减触发（基于本步结果，不可逆）
            if not decay_triggered:
                if V >= self.voltage_decay_trigger or T >= self.temp_decay_trigger:
                    decay_triggered = True
                    t_trigger = t
                    I_at_trigger = I_actual

            # 3i. SOC 达标 → 终止
            if soc >= self.soc_target:
                termination_reason = "soc_reached"
                break
        else:
            # while 正常结束（超时）
            termination_reason = "timeout"

        # -------- 4. 提取结果 --------
        if not time_rec:
            return self._make_failure_result("no_steps_completed")

        total_time = time_rec[-1]
        peak_temp = max(temp_rec)
        final_soc = soc_rec[-1]

        # 老化（SEI）
        aging_pct = self._extract_aging(sim.solution)
        log_aging = np.log10(max(aging_pct, 1e-10))

        # 约束检查
        violation = None
        soc_ok = final_soc >= (self.soc_target - 0.01)  # 容许 1% 误差

        if not soc_ok:
            violation = f"soc={final_soc:.3f} < target {self.soc_target}"
        if termination_reason == "timeout":
            violation = f"timeout: {self.max_time}s"
        if termination_reason.startswith("solver_error"):
            violation = termination_reason

        valid = violation is None

        # -------- 5. 降采样 Profile --------
        MAX_PTS = 500
        n_pts = len(time_rec)
        if n_pts > MAX_PTS:
            idx = np.linspace(0, n_pts - 1, MAX_PTS, dtype=int)
            time_prof = [time_rec[i] for i in idx]
            voltage_prof = [voltage_rec[i] for i in idx]
            temp_prof = [temp_rec[i] for i in idx]
            current_prof = [current_rec[i] for i in idx]
            soc_prof = [soc_rec[i] for i in idx]
            phase_prof = [phase_rec[i] for i in idx]
        else:
            time_prof = time_rec
            voltage_prof = voltage_rec
            temp_prof = temp_rec
            current_prof = current_rec
            soc_prof = soc_rec
            phase_prof = phase_rec

        return {
            # 三目标
            'time': total_time,
            'temp': peak_temp,
            'aging': aging_pct,
            'log_aging': log_aging,
            # 状态
            'final_soc': final_soc,
            'valid': valid,
            'violation': violation,
            'termination': termination_reason,
            # 衰减信息
            'decay_triggered': decay_triggered,
            'decay_time': t_trigger if decay_triggered else None,
            # Profile
            'total_steps': n_pts,
            'time_profile': time_prof,
            'voltage_profile': voltage_prof,
            'temp_profile': temp_prof,
            'current_profile': current_prof,
            'soc_profile': soc_prof,
            'phase_profile': phase_prof,
        }

    # ================================================================
    # 模型构建
    # ================================================================

    def _build_simulation(self):
        """构建 PyBaMM 模型和仿真器"""

        model = pybamm.lithium_ion.SPMe(
            options={
                "thermal": "lumped",
                "SEI": "solvent-diffusion limited",
            }
        )

        param = pybamm.ParameterValues(self.param_set)

        # 关键：将电流设为可动态输入的参数
        param["Current function [A]"] = pybamm.InputParameter(
            "Current function [A]"
        )

        # 安全上限（solver 级别，防止数值发散）
        param["Upper voltage cut-off [V]"] = 4.4

        # SEI 参数
        param["SEI growth activation energy [J.mol-1]"] = self.sei_activation_energy

        # 热参数
        param["Initial temperature [K]"] = self.init_temp
        param["Ambient temperature [K]"] = self.init_temp

        # 初始 stoichiometry（基于初始电压）
        param.set_initial_stoichiometries(f"{self.init_voltage} V")

        # 从参数集读取标称容量（避免硬编码）
        self.nominal_capacity = float(param["Nominal cell capacity [A.h]"])

        # Solver
        try:
            solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
        except Exception:
            solver = pybamm.ScipySolver(atol=1e-6, rtol=1e-6)

        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

        return sim, param

    # ================================================================
    # 老化提取
    # ================================================================

    def _extract_aging(self, sol) -> float:
        """
        从 PyBaMM 解中提取 SEI 容量损失百分比

        尝试多个变量名以兼容不同 PyBaMM 版本
        """
        sei_loss_mol = 0.0
        found = False

        # 按优先级尝试不同变量名
        candidate_vars = [
            "Loss of lithium to negative SEI [mol]",
            "Loss of lithium to SEI [mol]",
            "Loss of capacity to negative SEI [A.h]",  # 某些版本直接给 Ah
        ]

        for var_name in candidate_vars:
            try:
                entries = sol[var_name].entries
                if len(entries.shape) > 1:
                    entries = entries[0, :]
                val = float(entries[-1])

                if "A.h" in var_name:
                    # 已经是 Ah，直接转百分比
                    aging_pct = (abs(val) / self.nominal_capacity) * 100.0
                    return max(aging_pct, 1e-10)
                else:
                    sei_loss_mol = abs(val)
                    found = True
                    break
            except KeyError:
                continue

        if not found:
            warnings.warn(
                "SEI loss variable not found in solution. "
                "Aging will be set to 1e-10. "
                f"Available variables: check sol.all_models[0].variables.keys()"
            )
            return 1e-10

        # mol → Ah → 百分比
        F = 96485.33212  # 法拉第常数 [C/mol]
        Q_loss_Ah = sei_loss_mol * F / 3600.0
        aging_pct = (Q_loss_Ah / self.nominal_capacity) * 100.0

        return max(aging_pct, 1e-10)

    # ================================================================
    # 工具方法
    # ================================================================

    def _make_failure_result(self, violation_msg: str) -> Dict:
        """仿真失败时的标准返回"""
        return {
            'time': self.max_time,  # 惩罚值：最大时间
            'temp': self.init_temp,
            'aging': 1e-10,
            'log_aging': -10.0,
            'final_soc': 0.0,
            'valid': False,
            'violation': violation_msg,
            'termination': 'failure',
            'decay_triggered': False,
            'decay_time': None,
            'total_steps': 0,
            'time_profile': [],
            'voltage_profile': [],
            'temp_profile': [],
            'current_profile': [],
            'soc_profile': [],
            'phase_profile': [],
        }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("BatterySimulator v2 - 分步仿真 + 安全衰减")
    print("=" * 60)

    print(f"\nPyBaMM version: {pybamm.__version__}")

    sim_engine = BatterySimulator()
    print(f"参数集: {sim_engine.param_set}")
    print(f"初始电压: {sim_engine.init_voltage}V")
    print(f"初始温度: {sim_engine.init_temp}K ({sim_engine.init_temp - 273.15:.1f}°C)")
    print(f"目标SOC: {sim_engine.soc_target}")
    print(f"衰减触发: V≥{sim_engine.voltage_decay_trigger}V 或 T≥{sim_engine.temp_decay_trigger - 273.15:.0f}°C")
    print(f"衰减系数λ: {sim_engine.decay_lambda}")
    print(f"步长dt: {sim_engine.dt}s")

    def print_result(name, r, elapsed):
        print(f"\n[{name}]")
        print(f"  仿真耗时: {elapsed:.2f}s")
        print(f"  充电时间: {r['time']:.0f}s ({r['time']/60:.1f}min)")
        print(f"  峰值温度: {r['temp']:.2f}K ({r['temp']-273.15:.1f}°C)")
        print(f"  老化: {r['aging']:.8f}%  log={r['log_aging']:.4f}")
        print(f"  最终SOC: {r['final_soc']:.4f}")
        print(f"  有效: {r['valid']}")
        print(f"  终止原因: {r['termination']}")
        print(f"  衰减触发: {r['decay_triggered']}", end="")
        if r['decay_triggered']:
            print(f" @ t={r['decay_time']:.0f}s")
        else:
            print()
        print(f"  步数: {r['total_steps']}")
        if r['violation']:
            print(f"  违约: {r['violation']}")

    # 测试1: 温和策略 — 不太可能触发衰减
    print("\n" + "-" * 60)
    t0 = time.time()
    r1 = sim_engine.simulate(current1=8.0, soc1=0.55, current2=5)
    print_result("温和策略 I1=2A, SOC1=0.4, I2=1.5A", r1, time.time() - t0)

    # 测试2: 中等策略
    print("\n" + "-" * 60)
    t0 = time.time()
    r2 = sim_engine.simulate(current1=4.0, soc1=0.5, current2=2.5)
    print_result("中等策略 I1=4A, SOC1=0.5, I2=2.5A", r2, time.time() - t0)

    # 测试3: 激进策略 — 可能触发衰减
    print("\n" + "-" * 60)
    t0 = time.time()
    r3 = sim_engine.simulate(current1=5.5, soc1=0.3, current2=4.0)
    print_result("激进策略 I1=5.5A, SOC1=0.3, I2=4A", r3, time.time() - t0)

    # 测试4: 极端策略 — 大概率触发衰减
    print("\n" + "-" * 60)
    t0 = time.time()
    r4 = sim_engine.simulate(current1=6.0, soc1=0.6, current2=5.5)
    print_result("极端策略 I1=6A, SOC1=0.6, I2=5.5A", r4, time.time() - t0)

    # 测试5: 反直觉策略 — I1 < I2（先慢后快）
    print("\n" + "-" * 60)
    t0 = time.time()
    r5 = sim_engine.simulate(current1=1.5, soc1=0.2, current2=4.5)
    print_result("先慢后快 I1=1.5A, SOC1=0.2, I2=4.5A", r5, time.time() - t0)

    # 物理一致性检查
    print("\n" + "=" * 60)
    print("物理一致性检查")
    print("=" * 60)

    results = [
        ("温和", r1, 2.0, 1.5),
        ("中等", r2, 4.0, 2.5),
        ("激进", r3, 5.5, 4.0),
        ("极端", r4, 6.0, 5.5),
    ]

    print("\n[温度 vs 电流] 电流越大，温度应越高：")
    for name, r, i1, i2 in results:
        print(f"  {name} (I1={i1},I2={i2}): "
              f"T_peak={r['temp']-273.15:.1f}°C, "
              f"decay={'Y' if r['decay_triggered'] else 'N'}")

    print("\n[时间 vs 电流] 电流越大，时间应越短（衰减除外）：")
    for name, r, i1, i2 in results:
        print(f"  {name}: {r['time']/60:.1f}min, valid={r['valid']}")

    print("\n[老化 vs 电流] 电流越大，老化应越多：")
    for name, r, i1, i2 in results:
        print(f"  {name}: aging={r['aging']:.8f}%, log={r['log_aging']:.4f}")

    # Phase profile 检查
    print("\n[Phase profile 样例（中等策略前20步）]:")
    if r2['phase_profile']:
        for i in range(min(20, len(r2['phase_profile']))):
            p = r2['phase_profile'][i]
            t_val = r2['time_profile'][i]
            soc_val = r2['soc_profile'][i]
            I_val = r2['current_profile'][i]
            print(f"  t={t_val:6.0f}s  SOC={soc_val:.4f}  I={I_val:.3f}A  phase={p}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)