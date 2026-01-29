"""
PyBaMM 仿真封装器
输入：充电策略 [I1, switch_soc, I2]
输出：三目标 [Time, Temp, SOH]
"""

import pybamm
import numpy as np
import warnings
from typing import Dict, Tuple, List
from config import BATTERY_CONFIG, AGING_CONFIG


class BatterySimulator:
    """
    电池仿真器
    
    功能：
    1. 执行两阶段恒流充电
    2. 计算三个目标：Time, Temp, Aging
    3. 检查物理约束
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
        aging_config: Dict = None
    ):
        """
        初始化仿真器（参数默认从config读取）
        
        参数：
            param_set: PyBaMM参数集名称
            init_voltage: 初始电压 [V]
            init_temp: 初始温度 [K]
            sample_time: 每步时长 [s]
            voltage_max: 电压约束上限
            temp_max: 温度约束上限
            soc_target: 充电目标SOC
            aging_config: 老化计算参数
        """
        # 从config读取默认值
        self.param_set = param_set if param_set is not None else BATTERY_CONFIG['param_set']
        self.init_voltage = init_voltage if init_voltage is not None else BATTERY_CONFIG['init_voltage']
        self.init_temp = init_temp if init_temp is not None else BATTERY_CONFIG['init_temp']
        self.sample_time = sample_time if sample_time is not None else BATTERY_CONFIG['sample_time']
        self.voltage_max = voltage_max if voltage_max is not None else BATTERY_CONFIG['voltage_max']
        self.temp_max = temp_max if temp_max is not None else BATTERY_CONFIG['temp_max']
        self.soc_target = soc_target if soc_target is not None else BATTERY_CONFIG['soc_target']
        
        # 老化参数（使用新的key名称）
        if aging_config is None:
            aging_config = AGING_CONFIG
        self.aging_config = aging_config
        
        # 初始化PyBaMM模型
        self._setup_model()
    
    def _setup_model(self):
        """设置PyBaMM模型"""
        # 使用SPMe模型（包含热效应）
        options = {"thermal": "lumped"}
        self.model = pybamm.lithium_ion.SPMe(options=options)
        
        # 加载参数
        self.param = pybamm.ParameterValues(self.param_set)
        self.param["Upper voltage cut-off [V]"] = 4.4  # 允许略超
        
        # 设置初始SOC（通过电压）
        self.param.set_initial_stoichiometries(f"{self.init_voltage} V")
        
        # 使用输入电流
        self.param["Current function [A]"] = "[input]"
    
    def simulate(
        self, 
        current1: float, 
        switch_soc: float, 
        current2: float
    ) -> Dict:
        """
        执行两阶段充电仿真（修复：基于SOC阈值切换）
        
        参数：
            current1: 第一阶段电流 [A]
            switch_soc: SOC切换阈值 [0-1]
            current2: 第二阶段电流 [A]
        
        返回：
            {
                'time': 充电步数,
                'temp': 峰值温度 [K],
                'aging': 老化代理指标,
                'valid': 是否满足约束,
                'violation': 约束违反信息,
                'current_profile': 电流序列,
                'temp_profile': 温度序列,
                'voltage_profile': 电压序列,
                'soc_profile': SOC序列
            }
        """
        # 重置模型（每次仿真独立）
        model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
        param = pybamm.ParameterValues(self.param_set)
        param["Upper voltage cut-off [V]"] = 4.4
        param.set_initial_stoichiometries(f"{self.init_voltage} V")
        param["Current function [A]"] = "[input]"
        
        # 初始化记录
        current_profile = []
        temp_profile = []
        voltage_profile = []
        soc_profile = []
        
        # 初始状态
        voltage = self.init_voltage
        temp = self.init_temp
        soc = self._calculate_soc(
            param["Initial concentration in negative electrode [mol.m-3]"]
        )
        
        step = 0
        max_steps = 500  # 防止无限循环
        violation = None
        sol = None
        
        # 充电循环
        while step < max_steps:
            # 选择电流（修复：基于SOC阈值切换）
            if soc < switch_soc:
                current = current1
            else:
                current = current2
            
            # 电压补偿（避免过充）
            if voltage >= 4.0:
                current = current * np.exp(-0.9 * (voltage - 4.0))
            
            # 执行一步仿真
            try:
                # 连续求解
                if sol is not None:
                    model.set_initial_conditions_from(sol)
                
                simulation = pybamm.Simulation(model, parameter_values=param)
                t_eval = np.linspace(0, self.sample_time, 2)
                sol = simulation.solve(t_eval, inputs={"Current function [A]": -current})
                
                # 提取状态
                voltage = sol["Voltage [V]"].entries[-1]
                temp = sol["X-averaged cell temperature [K]"].entries[-1]
                c_neg = sol["R-averaged negative particle concentration [mol.m-3]"].entries[-1][-1]
                soc = self._calculate_soc(c_neg)
                
                # 记录
                current_profile.append(current)
                temp_profile.append(temp)
                voltage_profile.append(voltage)
                soc_profile.append(soc)
                
                # 检查约束
                if voltage > self.voltage_max:
                    violation = f'voltage={voltage:.3f}V > {self.voltage_max}V'
                    break
                
                if temp > self.temp_max:
                    violation = f'temp={temp:.2f}K > {self.temp_max}K'
                    break
                
                # 检查充电完成
                if soc >= self.soc_target:
                    break
                
                step += 1
                
            except Exception as e:
                warnings.warn(f"Simulation failed at step {step}: {e}")
                violation = f'simulation_error: {str(e)[:50]}'
                break
        
        # 计算老化代理指标
        aging = self._calculate_aging(
            np.array(current_profile),
            np.array(temp_profile)
        )
        
        # 返回结果
        return {
            'time': step + 1,
            'temp': max(temp_profile) if temp_profile else self.temp_max,
            'aging': aging,
            'valid': violation is None,
            'violation': violation,
            'current_profile': current_profile,
            'temp_profile': temp_profile,
            'voltage_profile': voltage_profile,
            'soc_profile': soc_profile
        }
    
    def _calculate_soc(self, c_neg: float) -> float:
        """
        计算SOC
        
        参数：
            c_neg: 负极浓度 [mol/m^3]
        
        返回：
            SOC [0-1]
        """
        # 基于负极浓度计算SOC（参考BO_demo.py）
        c_min = 873.0
        c_max = 30171.3
        return (c_neg - c_min) / (c_max - c_min)
    
    def _calculate_aging(
        self, 
        current_profile: np.ndarray, 
        temp_profile: np.ndarray
    ) -> float:
        """
        计算容量衰减百分比(SEI生长模型)
        
        返回:
            容量衰减百分比 [%]，保证 >= 1e-6
        """
        # ========== 修复：避免返回0 ==========
        if len(current_profile) == 0 or len(temp_profile) == 0:
            return 1e-6
        
        if np.any(np.isnan(current_profile)) or np.any(np.isnan(temp_profile)):
            return 1e-6
        
        if np.any(np.isinf(current_profile)) or np.any(np.isinf(temp_profile)):
            return 1e-6
        # ====================================
        
        # SEI生长模型参数（从config读取）
        k_sei = self.aging_config.get('k_sei', 5e-5)
        E_a = self.aging_config.get('E_a', 30000)
        R = self.aging_config.get('R', 8.314)
        T_ref = self.aging_config.get('T_ref', 298.15)
        
        # 电流积分(充电量) [A·h]
        charge_throughput = np.sum(np.abs(current_profile)) * self.sample_time / 3600
        
        # 温度加速因子(归一化到T_ref)
        # ========== 修复：避免除零 ==========
        temp_profile_safe = np.clip(temp_profile, 273.0, 400.0)
        temp_factor = np.mean(np.exp(-E_a/R * (1/temp_profile_safe - 1/T_ref)))
        
        if np.isnan(temp_factor) or np.isinf(temp_factor):
            temp_factor = 1.0
        # ===================================
        
        # 容量衰减 = k × 充电量 × 温度因子
        capacity_loss = k_sei * charge_throughput * temp_factor
        
        # ========== 修复：保证 >= 1e-6 ==========
        capacity_loss = max(capacity_loss, 1e-6)
        
        if np.isnan(capacity_loss) or np.isinf(capacity_loss):
            return 1e-6
        # ======================================
        
        return capacity_loss  # 单位: %

# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("测试 BatterySimulator...")
    
    # 初始化
    sim = BatterySimulator()
    
    # 测试充电策略
    result = sim.simulate(current1=5.0, switch_soc=0.5, current2=3.0)
    
    print(f"\n充电结果:")
    print(f"  时间: {result['time']} 步")
    print(f"  峰值温度: {result['temp']:.2f} K ({result['temp']-273.15:.1f}°C)")
    print(f"  老化指标: {result['aging']:.6f}")
    print(f"  是否有效: {result['valid']}")
    if not result['valid']:
        print(f"  约束违反: {result['violation']}")
    
    print("\n测试完成！")