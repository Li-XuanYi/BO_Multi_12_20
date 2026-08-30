"""
PyBaMM 仿真封装器（重构版）
输入：充电策略 [I1, T1(s), I2, V_switch]
输出：三目标 [Time(s), Temp(K), Aging(%)]

充电协议：CC1 → CC2 → CV
- 阶段1 (CC1): 以 I1 恒流充电，持续 T1 秒（若电压先触碰 V_switch 则提前进入 CV）
- 阶段2 (CC2): 以 I2 恒流充电（若电压触碰 V_switch 则进入 CV；若 SOC 达标则终止）
- 阶段3 (CV):  保持 V_switch 恒压充电，直到电流 < C/20 或 SOC 达标

老化模型：PyBaMM 内置 SEI 子模型（solvent-diffusion limited），E_a = 37500 J/mol
"""

import pybamm
import numpy as np
import warnings
from typing import Dict, List
try:
    from ..config import BATTERY_CONFIG
except ImportError:
    try:
        from config import BATTERY_CONFIG
    except ImportError:
        # 最终后备：与 config.py 严格一致
        BATTERY_CONFIG = {
            'param_set': 'Chen2020',
            'init_voltage': 3.0,
            'init_temp': 298.15,
            'sample_time': 90.0,
            'voltage_max': 4.2,
            'temp_max': 318.15,
            'soc_target': 0.80
        }



class BatterySimulator:
    """
    电池仿真器（重构版）
    
    功能：
    1. 执行 CC-CC-CV 充电协议
    2. 使用 PyBaMM 原生 SEI 老化模型
    3. 计算三目标：Time, Temp, Aging
    """
    
    def __init__(
        self,
        param_set: str = None,
        init_voltage: float = None,
        init_temp: float = None,
        sample_time: float = None,
        voltage_max: float = None,
        temp_max: float = None,
        soc_target: float = None,
    ):
        """
        初始化仿真器（参数默认从config读取）
        """
        self.param_set = param_set if param_set is not None else BATTERY_CONFIG['param_set']
        self.init_voltage = init_voltage if init_voltage is not None else BATTERY_CONFIG['init_voltage']
        self.init_temp = init_temp if init_temp is not None else BATTERY_CONFIG['init_temp']
        self.sample_time = sample_time if sample_time is not None else BATTERY_CONFIG['sample_time']
        self.voltage_max = voltage_max if voltage_max is not None else BATTERY_CONFIG['voltage_max']
        self.temp_max = temp_max if temp_max is not None else BATTERY_CONFIG['temp_max']
        self.soc_target = soc_target if soc_target is not None else BATTERY_CONFIG['soc_target']
        
        # 新增属性
        self.cv_cutoff_current = 0.25  # C/20 = 5.0/20 = 0.25A
        self.sei_activation_energy = 37500.0  # J/mol
        self.nominal_capacity = 5.0  # Ah
    
    def simulate(
        self,
        current1: float,
        time1: float,
        current2: float,
        v_switch: float
    ) -> Dict:
        """
        执行 CC-CC-CV 充电协议（Experiment API版）
        
        参数：
            current1: 第一阶段电流 [A]
            time1: 第一阶段最大持续时间 [s]（统一为秒）
            current2: 第二阶段电流 [A]
            v_switch: CC-to-CV 转换电压 [V]
        
        返回：
            结果字典
        """
        # ========== 保护numpy随机状态 ==========
        # PyBaMM内部会调用np.random.seed()，破坏全局随机状态
        # 必须在仿真前后保存/恢复，否则后续随机采样全部重复
        rng_state = np.random.get_state()
        # ==========================================

        try:
            return self._simulate_inner(current1, time1, current2, v_switch)
        finally:
            # ========== 恢复随机状态（无论仿真成功或失败）==========
            np.random.set_state(rng_state)

    def _simulate_inner(
        self,
        current1: float,
        time1: float,
        current2: float,
        v_switch: float
    ) -> Dict:
        """仿真内部实现（被simulate()包装以保护随机状态）"""
        # 步骤1: 参数预处理与安全化
        current1 = float(current1)
        time1 = float(time1)
        current2 = float(current2)
        v_switch = float(v_switch)
        
        # 基本有效性检查
        if current1 <= 0 or current2 <= 0 or time1 <= 0 or v_switch <= 0:
            return self._make_failure_result("invalid_params: negative or zero values")
        
        # 步骤2: 构建PyBaMM Experiment对象
        time1_seconds = time1  # time1 已经是秒（统一单位）
        
        # PyBaMM 24.1支持容量或电压作为终止条件，用容量表示SOC
        # SOC 80% ≈ 0.8 * 5Ah = 4Ah
        target_capacity = self.soc_target * self.nominal_capacity
        
        experiment = pybamm.Experiment(
            [
                f"Charge at {current1} A for {time1_seconds} seconds or until {v_switch} V",
                f"Charge at {current2} A until {v_switch} V",
                f"Hold at {v_switch} V until {self.cv_cutoff_current} A",
            ],
            termination=f"{target_capacity:.2f} Ah capacity",
        )
        
        # 步骤3: 创建模型和参数
        model = pybamm.lithium_ion.SPMe(
            options={
                "thermal": "lumped",
                "SEI": "solvent-diffusion limited"
            }
        )
        
        param = pybamm.ParameterValues(self.param_set)
        param["Upper voltage cut-off [V]"] = 4.4
        param["SEI growth activation energy [J.mol-1]"] = self.sei_activation_energy
        param["Initial temperature [K]"] = self.init_temp
        param["Ambient temperature [K]"] = self.init_temp
        param.set_initial_stoichiometries(f"{self.init_voltage} V")
        
        # 步骤4: 创建Simulation并求解
        try:
            # 使用IDAKLUSolver替代默认的CasadiSolver(需要cvodes.dll)
            try:
                solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
            except Exception:
                # IDAKLU不可用，回退到ScipySolver
                solver = pybamm.ScipySolver(atol=1e-6, rtol=1e-6)
            
            simulation = pybamm.Simulation(
                model,
                experiment=experiment,
                parameter_values=param,
                solver=solver,
            )
            
            sol = simulation.solve()
            
            if sol is None:
                return self._make_failure_result("solve_failed: solution is None")
                
        except Exception as e:
            return self._make_failure_result(f"simulation_error: {str(e)[:200]}")
        
        # 步骤5: 从解中提取目标值和约束信息
        try:
            # 总充电时间
            total_time = float(sol["Time [s]"].entries[-1])
            
            # 温度曲线和峰值温度
            temp_entries = sol["X-averaged cell temperature [K]"].entries
            peak_temp = float(np.max(temp_entries))
            
            # 电压曲线
            voltage_entries = sol["Voltage [V]"].entries
            
            # 电流曲线（取绝对值，PyBaMM中充电电流为负）
            current_entries = np.abs(sol["Current [A]"].entries)
            
            # SOC曲线（从负极浓度计算）
            c_neg_var = sol["X-averaged negative particle concentration [mol.m-3]"]
            c_neg_entries = c_neg_var.entries
            
            # 如果是多维（空间+时间），取第一个空间点或平均
            if len(c_neg_entries.shape) > 1:
                c_neg_entries = c_neg_entries[0, :]  # 取第一个空间点
            
            soc_entries = np.array([self._calculate_soc(c, param) for c in c_neg_entries])
            final_soc = float(soc_entries[-1])
            
            # SEI锂损失（累积值，PyBaMM 24.1分离正负极）
            try:
                # 尝试负极SEI损失
                sei_loss_neg_var = sol["Loss of lithium to negative SEI [mol]"]
                sei_loss_neg = sei_loss_neg_var.entries
                
                # 处理多维情况
                if len(sei_loss_neg.shape) > 1:
                    sei_loss_neg = sei_loss_neg[0, :]
                
                neg_loss = float(sei_loss_neg[-1])
            except KeyError:
                neg_loss = 0.0
            
            try:
                # 尝试正极SEI损失
                sei_loss_pos_var = sol["Loss of lithium to positive SEI [mol]"]
                sei_loss_pos = sei_loss_pos_var.entries
                
                if len(sei_loss_pos.shape) > 1:
                    sei_loss_pos = sei_loss_pos[0, :]
                
                pos_loss = float(sei_loss_pos[-1])
            except KeyError:
                pos_loss = 0.0
            
            total_sei_loss = neg_loss + pos_loss
            
            aging = self._calculate_aging_from_sei(total_sei_loss)
            
        except Exception as e:
            return self._make_failure_result(f"extraction_error: {str(e)[:200]}")
        
        # 步骤6: 约束检查
        violation = None
        
        if peak_temp > self.temp_max:
            violation = f"temp={peak_temp:.2f}K > {self.temp_max}K"
        
        peak_voltage = float(np.max(voltage_entries))
        if violation is None and peak_voltage > 4.4:
            violation = f"voltage={peak_voltage:.3f}V > 4.4V"
        
        # 步骤7: 提取阶段信息
        cv_time = 0.0
        if hasattr(sol, 'sub_solutions') and sol.sub_solutions is not None and len(sol.sub_solutions) >= 3:
            try:
                cv_sol = sol.sub_solutions[2]
                cv_time = float(cv_sol["Time [s]"].entries[-1] - cv_sol["Time [s]"].entries[0])
            except:
                pass
        
        if cv_time == 0.0:
            # 回退：估算CV时间
            cv_mask = voltage_entries >= (v_switch - 0.01)
            cv_time = float(np.sum(cv_mask) / max(len(voltage_entries), 1) * total_time)
        
        # 步骤8: 构建返回字典（降采样profile）
        MAX_PROFILE_POINTS = 500
        if len(temp_entries) > MAX_PROFILE_POINTS:
            indices = np.linspace(0, len(temp_entries) - 1, MAX_PROFILE_POINTS, dtype=int)
            temp_profile = temp_entries[indices].tolist()
            voltage_profile = voltage_entries[indices].tolist()
            current_profile = current_entries[indices].tolist()
            soc_profile = soc_entries[indices].tolist()
        else:
            temp_profile = temp_entries.tolist()
            voltage_profile = voltage_entries.tolist()
            current_profile = current_entries.tolist()
            soc_profile = soc_entries.tolist()
        
        # 构建phase_profile
        phase_profile = []
        for i in range(len(current_profile)):
            i_abs = current_profile[i]
            v = voltage_profile[i] if i < len(voltage_profile) else 0
            if abs(i_abs - current1) < 0.1:
                phase_profile.append('CC1')
            elif abs(i_abs - current2) < 0.1:
                phase_profile.append('CC2')
            else:
                phase_profile.append('CV')
        
        return {
            'time': total_time,
            'temp': peak_temp,
            'aging': aging,
            'final_soc': final_soc,
            'valid': violation is None,
            'violation': violation,
            'cv_time': cv_time,
            'total_steps': len(temp_entries),
            'current_profile': current_profile,
            'temp_profile': temp_profile,
            'voltage_profile': voltage_profile,
            'soc_profile': soc_profile,
            'phase_profile': phase_profile,
        }
    
    def _make_failure_result(self, violation_msg: str) -> Dict:
        """构建仿真失败时的标准返回字典"""
        return {
            'time': 0.0,
            'temp': self.temp_max,
            'aging': 1e-6,
            'final_soc': 0.0,
            'valid': False,
            'violation': violation_msg,
            'cv_time': 0.0,
            'total_steps': 0,
            'current_profile': [],
            'temp_profile': [],
            'voltage_profile': [],
            'soc_profile': [],
            'phase_profile': [],
        }
    
    def _calculate_soc(self, c_neg: float, param) -> float:
        """基于负极浓度计算SOC（与SPM.py一致的经验映射）"""
        c_min = 873.0
        c_max = 30171.3
        soc = (c_neg - c_min) / (c_max - c_min)
        return float(np.clip(soc, 0.0, 1.0))
    
    def simulate_3d(
        self,
        I1: float,
        SOC1: float,
        I2: float
    ) -> Dict:
        """
        执行两阶段 CC 充电协议（3D 决策空间 - FrameWork.md §0）
        
        输入：theta = (I1, SOC1, I2)
        - I1: 第一阶段电流 [A]
        - SOC1: 第一阶段结束时的 SOC（切换点）
        - I2: 第二阶段电流 [A]
        
        输出：三目标 (time, temp, aging)
        
        充电协议：CC1 → CC2 → CV
        - 阶段 1 (CC1): 以 I1 恒流充电，直到 SOC 达到 SOC1
        - 阶段 2 (CC2): 以 I2 恒流充电，直到 SOC 达到 0.8
        - 阶段 3 (CV): 恒压充电直到电流 < C/20
        
        持续时间计算：
        t1 = (SOC1 - SOC0) * Q_NOM / I1
        t2 = (SOC_END - SOC1) * Q_NOM / I2
        """
        from config import SOC0, SOC_END, Q_NOM
        
        # 计算持续时间（秒）
        t1 = (SOC1 - SOC0) * Q_NOM / I1
        t2 = (SOC_END - SOC1) * Q_NOM / I2
        
        # 调用原有的 4D simulate 方法
        # 注意：这里将 3D 参数转换为 4D 接口
        # time1 = t1, v_switch = 4.2（上限）
        return self.simulate(
            current1=I1,
            time1=t1,
            current2=I2,
            v_switch=4.2
        )

    def _calculate_aging_from_sei(self, cumulative_sei_loss_mol: float) -> float:
        """
        从 PyBaMM SEI 模型输出计算容量衰减百分比
        
        Qa = (Q_loss / Q_nom) × 100%
        Q_loss = sei_loss_mol × F
        """
        if cumulative_sei_loss_mol <= 0 or np.isnan(cumulative_sei_loss_mol):
            return 1e-6
        
        F = 96485.33212  # 法拉第常数 [C/mol]
        Q_loss_Ah = cumulative_sei_loss_mol * F / 3600.0
        Q_nom = self.nominal_capacity
        
        aging_pct = (Q_loss_Ah / Q_nom) * 100.0
        
        return float(max(aging_pct, 1e-6))


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("测试 BatterySimulator (Experiment API 版)")
    print("=" * 60)
    
    # 测试0: PyBaMM基础检查
    print("\n[测试0] PyBaMM版本检查")
    print(f"  PyBaMM version: {pybamm.__version__}")
    
    sim = BatterySimulator()
    print(f"  参数集: {sim.param_set}")
    print(f"  初始电压: {sim.init_voltage}V")
    print(f"  初始温度: {sim.init_temp}K ({sim.init_temp - 273.15:.1f}°C)")
    print(f"  目标SOC: {sim.soc_target * 100:.0f}%")
    
    # 测试1: 保守策略
    print("\n[测试1] 保守策略 (低电流, 长时间)")
    t0 = time.time()
    r1 = sim.simulate(current1=3.5, time1=15.0, current2=2.0, v_switch=4.1)
    elapsed = time.time() - t0
    
    print(f"  仿真时间: {elapsed:.2f}s")
    print(f"  充电时间: {r1['time']:.0f}s ({r1['time']/60:.1f}min)")
    print(f"  峰值温度: {r1['temp']:.2f}K ({r1['temp']-273.15:.1f}°C)")
    print(f"  老化: {r1['aging']:.6f}%")
    print(f"  最终SOC: {r1['final_soc']:.3f}")
    print(f"  CV时间: {r1['cv_time']:.0f}s")
    print(f"  步数: {r1['total_steps']}")
    print(f"  有效: {r1['valid']}, 违约: {r1['violation']}")
    
    # 测试2: 激进策略
    print("\n[测试2] 激进策略 (高电流, 短时间)")
    t0 = time.time()
    r2 = sim.simulate(current1=5.5, time1=8.0, current2=3.5, v_switch=4.15)
    elapsed = time.time() - t0
    
    print(f"  仿真时间: {elapsed:.2f}s")
    print(f"  充电时间: {r2['time']:.0f}s ({r2['time']/60:.1f}min)")
    print(f"  峰值温度: {r2['temp']:.2f}K ({r2['temp']-273.15:.1f}°C)")
    print(f"  老化: {r2['aging']:.6f}%")
    print(f"  最终SOC: {r2['final_soc']:.3f}")
    print(f"  有效: {r2['valid']}, 违约: {r2['violation']}")
    
    # 测试3: 极端低电流长时间
    print("\n[测试3] 极端低电流长时间")
    t0 = time.time()
    r3 = sim.simulate(current1=1.0, time1=30.0, current2=0.5, v_switch=4.05)
    elapsed = time.time() - t0
    
    print(f"  仿真时间: {elapsed:.2f}s")
    print(f"  充电时间: {r3['time']:.0f}s ({r3['time']/60:.1f}min)")
    print(f"  峰值温度: {r3['temp']:.2f}K ({r3['temp']-273.15:.1f}°C)")
    print(f"  有效: {r3['valid']}")
    
    # 测试4: 极端高电流短时间
    print("\n[测试4] 极端高电流短时间")
    t0 = time.time()
    r4 = sim.simulate(current1=8.0, time1=3.0, current2=6.0, v_switch=4.18)
    elapsed = time.time() - t0
    
    print(f"  仿真时间: {elapsed:.2f}s")
    print(f"  充电时间: {r4['time']:.0f}s ({r4['time']/60:.1f}min)")
    print(f"  峰值温度: {r4['temp']:.2f}K ({r4['temp']-273.15:.1f}°C)")
    print(f"  有效: {r4['valid']}, 违约: {r4['violation']}")
    
    # 物理一致性检查
    print("\n" + "=" * 60)
    print("物理一致性检查")
    print("=" * 60)
    
    print(f"\n[老化对比]")
    print(f"  保守策略: {r1['aging']:.6f}%")
    print(f"  激进策略: {r2['aging']:.6f}%")
    print(f"  比值: {r2['aging']/max(r1['aging'],1e-10):.2f}x")
    if r2['aging'] > r1['aging']:
        print("  ✓ 高电流导致更高老化（温度-SEI耦合生效）")
    else:
        print("  ⚠ 老化耦合可能未按预期工作")
    
    print(f"\n[温度对比]")
    print(f"  低电流 (1A): {r3['temp']-273.15:.1f}°C")
    print(f"  中电流 (3.5A): {r1['temp']-273.15:.1f}°C")
    print(f"  高电流 (8A): {r4['temp']-273.15:.1f}°C")
    if r3['temp'] < r1['temp'] < r4['temp']:
        print("  ✓ 电流越大温度越高（热模型生效）")
    else:
        print("  ⚠ 温度-电流关系异常")
    
    print(f"\n[时间对比]")
    print(f"  激进策略: {r2['time']/60:.1f}min")
    print(f"  保守策略: {r1['time']/60:.1f}min")
    if r2['time'] < r1['time']:
        print("  ✓ 高电流充电更快（符合预期）")
    
    print(f"\n[Profile长度检查]")
    print(f"  保守策略 profile长度: {len(r1['temp_profile'])}")
    print(f"  激进策略 profile长度: {len(r2['temp_profile'])}")
    if all(len(r1[k]) == len(r1['temp_profile']) for k in ['current_profile', 'voltage_profile', 'soc_profile']):
        print("  ✓ 所有profile长度一致")
    else:
        print("  ⚠ Profile长度不匹配")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

