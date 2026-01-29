"""
LLM-MOBO 主程序
基于sklearn GP手写BO循环
"""

import numpy as np
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple



# 导入配置
from config import (
    BATTERY_CONFIG, PARAM_BOUNDS, AGING_CONFIG,
    MOBO_CONFIG, BO_CONFIG, LLM_CONFIG, DATA_CONFIG
)

# 导入模块
from battery_env.wrapper import BatterySimulator
from models.gp_model import MOGPModel
from acquisition.scalarization import TchebycheffScalarizer
from acquisition.optimization import AcquisitionOptimizer
from components.warmstart import LLMWarmStart
from components.coupling_inference import LLMCouplingInference
from components.llm_weighting import LLAMBOWeighting
from utils.visualization import (
    plot_pareto_front_3d,
    plot_optimization_history,
    save_results_json
)
from utils.transforms import DataTransformer


class LLMMOBO:
    """
    LLM增强的多目标贝叶斯优化器
    
    核心流程：
    1. LLM Warm Start
    2. 独立GP建模（3个GP）
    3. 狄利克雷采样权重
    4. Tchebycheff标量化
    5. LLM增强EI
    6. 仿真评估
    7. gamma自适应
    """
    
    def __init__(
        self,
        llm_api_key: str = "sk-Evfy9FZGKZ31bpgdNsDSFfkWMopRE6EN4V4r801oRaIi8jm7",
        n_warmstart: int = 5,
        n_random_init: int = 10,
        n_iterations: int = 50,
        gamma_init: float = 0.5,
        verbose: bool = True,
        # ========== 新增：消融实验控制 ==========
        use_coupling: bool = True,
        use_warmstart: bool = True,
        use_llm_acq: bool = True
        # ======================================
    ):
        """
        初始化优化器
        
        参数：
            llm_api_key: LLM API密钥
            n_warmstart: LLM热启动样本数
            n_random_init: 随机初始化样本数
            n_iterations: BO迭代次数
            gamma_init: 初始耦合强度
            verbose: 详细输出
            use_coupling: 启用物理耦合核
            use_warmstart: 启用LLM热启动
            use_llm_acq: 启用LLM空间加权
        """
        self.verbose = verbose
        self.n_warmstart = n_warmstart
        self.n_random_init = n_random_init
        self.n_iterations = n_iterations
        self.gamma = gamma_init
        
        # ========== 新增：保存消融标志 ==========
        self.use_coupling = use_coupling
        self.use_warmstart = use_warmstart
        self.use_llm_acq = use_llm_acq
        # ======================================
        
        # 初始化仿真器
        self.simulator = BatterySimulator(
            param_set=BATTERY_CONFIG['param_set'],
            init_voltage=BATTERY_CONFIG['init_voltage'],
            init_temp=BATTERY_CONFIG['init_temp'],
            sample_time=BATTERY_CONFIG['sample_time'],
            voltage_max=BATTERY_CONFIG['voltage_max'],
            temp_max=BATTERY_CONFIG['temp_max'],
            soc_target=BATTERY_CONFIG['soc_target'],
            aging_config=AGING_CONFIG
        )
        
        # 初始化LLM客户端（如果有API key）
        self.llm_enabled = llm_api_key is not None
        if self.llm_enabled:
            self.llm_warmstart = LLMWarmStart(
                api_key=llm_api_key,
                # 其他参数从config读取
                verbose=self.verbose
            )
            self.llm_coupling = LLMCouplingInference(
                api_key=llm_api_key,
                # 其他参数从config读取
                verbose=self.verbose
            )
        else:
            self.llm_warmstart = None
            self.llm_coupling = None
        
        # ========== 新增：初始化LLM权重模块 ==========
        if self.llm_enabled and self.use_llm_acq:
            self.llm_weighting = LLAMBOWeighting(
                param_bounds=PARAM_BOUNDS,
                llm_api_key=llm_api_key,
                base_url=LLM_CONFIG['base_url'],
                model=LLM_CONFIG['model'],
                verbose=self.verbose
            )
        else:
            self.llm_weighting = None
        # ==============================================
        
        # 数据变换器（处理aging的Log变换）
        self.data_transformer = DataTransformer(
            enable_log_aging=True,
            verbose=self.verbose
        )
        
        # 数据库（保存所有评估历史）
        self.database = []
        
        # 多目标GP管理器
        self.mogp = MOGPModel(
            use_coupling=self.use_coupling,
            gamma_init=gamma_init,
            verbose=self.verbose
        )
        
        # ========== 先初始化理想点和参考点（必须在创建scalarizer之前） ==========
        # 参考点和理想点（初始化为Log空间）
        if AGING_CONFIG.get('enable_log', True):
            # Log空间的边界
            ideal_aging = np.log10(1e-6 + 1e-10)      # ≈ -6.0
            reference_aging = np.log10(0.1 + 1e-10)   # ≈ -1.0
        else:
            ideal_aging = 0.0
            reference_aging = 0.1
        
        self.ideal_point = np.array([
            MOBO_CONFIG['ideal_point']['time'],
            MOBO_CONFIG['ideal_point']['temp'],
            ideal_aging
        ])
        self.reference_point = np.array([
            MOBO_CONFIG['reference_point']['time'],
            MOBO_CONFIG['reference_point']['temp'],
            reference_aging
        ])
        
        if self.verbose:
            print(f"  理想点: Time={self.ideal_point[0]}, Temp={self.ideal_point[1]:.1f}K, Aging={self.ideal_point[2]:.6f}")
            print(f"  参考点: Time={self.reference_point[0]}, Temp={self.reference_point[1]:.1f}K, Aging={self.reference_point[2]:.6f}")
        # ========================================================================
        
        # Tchebycheff标量化器
        self.scalarizer = TchebycheffScalarizer(
            ideal_point=self.ideal_point,
            reference_point=self.reference_point,
            eta=MOBO_CONFIG['eta']
        )
        
        # 采集函数优化器（修复：启用LLM weighting）
        self.acq_optimizer = AcquisitionOptimizer(
            param_bounds=PARAM_BOUNDS,
            scalarizer=self.scalarizer,
            enable_llm_weighting=(self.llm_enabled and self.use_llm_acq),  # ← 修改
            verbose=self.verbose
        )
        
        if self.verbose:
            print("="*70)
            print("LLM-MOBO 初始化完成")
            print("="*70)
            print(f"仿真器: PyBaMM SPMe ({BATTERY_CONFIG['param_set']})")
            print(f"LLM增强: {'启用' if self.llm_enabled else '禁用'}")
            print(f"热启动样本: {n_warmstart}")
            print(f"随机初始化: {n_random_init}")
            print(f"BO迭代次数: {n_iterations}")
            print(f"初始gamma: {gamma_init}")
            print("="*70)
    
    async def optimize(self):
        """
        主优化循环（异步）
        """
        if self.verbose:
            print("\n开始优化...")
        
        # ===== 阶段1: 初始化 =====
        if self.verbose:
            print(f"\n[阶段1] 初始化 ({self.n_warmstart + self.n_random_init} 个样本)")
        
        # 1.1 LLM Warm Start
        if self.llm_enabled and self.n_warmstart > 0:
            warmstart_samples = await self._llm_warmstart()
            for sample in warmstart_samples:
                self._evaluate_and_store(sample)
        
        # 1.2 随机初始化
        for _ in range(self.n_random_init):
            random_sample = self._random_sample()
            self._evaluate_and_store(random_sample)
        
        if self.verbose:
            print(f"  初始化完成: {len(self.database)} 个有效样本")
        
        # ===== 阶段1.5: LLM推理耦合矩阵（修复：添加此步骤）=====
        if self.llm_enabled and LLM_CONFIG['enable_wij_inference']:
            if self.verbose:
                print(f"\n[阶段1.5] LLM推理参数耦合矩阵")
            
            try:
                wij_matrix = await self.llm_coupling.infer_coupling_matrix(
                    param_names=['current1', 'switch_soc', 'current2'],
                    current_data=self.database[:15]
                )
                
                if wij_matrix is not None:
                    # 注入到GP模型（使用新方法）
                    self.mogp.set_llm_coupling_matrix(wij_matrix)
                    if self.verbose:
                        print(f"  W_LLM已注入到GP模型")
                else:
                    if self.verbose:
                        print(f"  W_LLM推理失败，GP将仅使用W_data")
                
            except Exception as e:
                if self.verbose:
                    print(f"  LLM推理失败: {e}，GP将仅使用W_data")
        
        # ===== 阶段2: BO循环 =====
        for iteration in range(self.n_iterations):
            if self.verbose:
                print(f"\n[迭代 {iteration+1}/{self.n_iterations}]")
            
            # ========== 新增：每5轮更新LLM焦点 ==========
            if self.llm_weighting is not None and iteration % 5 == 0 and iteration > 0:
                if self.verbose:
                    print(f"  [LLM焦点更新]")
                await self.llm_weighting.update_focus_from_llm(
                    database=self.database,
                    gp_list=self.mogp.get_gp_list()
                )
            # ===========================================
            
            # 2.1 采样新权重（狄利克雷分布）
            weights = self._sample_weights()
            if self.verbose:
                print(f"  权重: time={weights[0]:.2f}, temp={weights[1]:.2f}, aging={weights[2]:.2f}")
            
            # 2.2 拟合3个独立GP
            self._fit_gps()
            
            # 2.3 优化采集函数（找到下一个查询点）
            next_point = self._optimize_acquisition(weights)
            if self.verbose:
                print(f"  下一个点: I1={next_point['current1']:.2f}, SOC_sw={next_point['switch_soc']:.2f}, I2={next_point['current2']:.2f}")
            
            # 2.4 评估新点
            self._evaluate_and_store(next_point)
            
            # 2.5 更新gamma（自适应耦合强度）
            self._update_gamma(iteration)
            
            # 2.6 保存中间结果
            if (iteration + 1) % DATA_CONFIG['save_interval'] == 0:
                self._save_checkpoint(iteration + 1)
        
        # ===== 阶段3: 结果分析 =====
        if self.verbose:
            print("\n[阶段3] 优化完成，分析结果...")
        
        results = self._analyze_results()
        
        # ========== 新增：计算HV历史 ==========
        from utils.visualization import compute_hv_history
        
        # 参考点（使用原始空间）
        ref_point = np.array([
            MOBO_CONFIG['reference_point']['time'],
            MOBO_CONFIG['reference_point']['temp'],
            MOBO_CONFIG['reference_point']['aging']
        ])
        
        hv_history = compute_hv_history(self.database, ref_point)
        results['hv_history'] = hv_history
        
        if self.verbose:
            print(f"  最终Hypervolume: {hv_history[-1]:.4f}")
        # ====================================
        
        self._save_final_results(results)
        
        return results
    
    async def _llm_warmstart(self) -> List[Dict]:
        """LLM热启动（生成初始样本）"""
        # ========== 新增：检查标志 ==========
        if not self.use_warmstart:
            if self.verbose:
                print("  [LLM WarmStart] 已禁用，使用随机采样")
            return [self._random_sample() for _ in range(self.n_warmstart)]
        # ==================================
        
        if self.llm_warmstart is None:
            print("  [LLM Warm Start] 未启用，使用随机采样")
            return [self._random_sample() for _ in range(self.n_warmstart)]
        
        try:
            strategies = await self.llm_warmstart.generate_strategies(
                n_strategies=self.n_warmstart,
                param_bounds=PARAM_BOUNDS
            )
            return strategies
        except Exception as e:
            print(f"  [LLM Warm Start] 失败: {e}，回退到随机采样")
            return [self._random_sample() for _ in range(self.n_warmstart)]
    
    def _random_sample(self) -> Dict:
        """随机采样一个充电策略"""
        return {
            'current1': np.random.uniform(*PARAM_BOUNDS['current1']),
            'switch_soc': np.random.uniform(*PARAM_BOUNDS['switch_soc']),
            'current2': np.random.uniform(*PARAM_BOUNDS['current2'])
        }
    
    def _evaluate_and_store(self, params: Dict):
        """
        评估充电策略并存储到数据库
        
        关键处理：
        1. 提取并保留rationale（LLAMBO特性）
        2. 只传递物理参数给仿真器（避免TypeError）
        3. 完整记录包含所有信息（用于后续分析）
        
        参数：
            params: 策略字典，可能包含：
                - current1, switch_soc, current2（必需）
                - rationale（可选，来自LLM）
                - target_zone（可选，来自WarmStart）
        """
        # 提取非物理参数（用于记录但不传给仿真器）
        rationale = params.pop('rationale', None)
        target_zone = params.pop('target_zone', None)
        
        # 仅提取仿真器需要的3个物理参数
        sim_params = {
            'current1': params['current1'],
            'switch_soc': params['switch_soc'],
            'current2': params['current2']
        }
        
        # 执行仿真
        result = self.simulator.simulate(**sim_params)
        
        # 构建完整记录
        record = {
            'params': sim_params.copy(),  # 物理参数
            'time': result['time'],
            'temp': result['temp'],
            'aging': result['aging'],
            'valid': result['valid'],
            'violation': result['violation']
        }
        
        # 添加元信息（如果有）
        if rationale is not None:
            record['rationale'] = rationale
        if target_zone is not None:
            record['target_zone'] = target_zone
        
        # 存储到数据库
        self.database.append(record)
        
        # 详细日志输出
        if self.verbose:
            if result['valid']:
                # 基础信息（修复1：6位小数）
                log_msg = (f"    评估: time={result['time']}, "
                          f"temp={result['temp']:.1f}K, "
                          f"aging={result['aging']:.6f}")
                
                # LLM解释（如果有）
                if rationale is not None:
                    rationale_short = (rationale[:50] + '...' 
                                     if len(rationale) > 50 
                                     else rationale)
                    log_msg += f"\n          策略: {rationale_short}"
                
                # 目标区域（如果有）
                if target_zone is not None:
                    log_msg += f" [{target_zone}]"
                
                print(log_msg)
            else:
                print(f"    约束违反: {result['violation']}")
    
    def _sample_weights(self) -> np.ndarray:
        """从狄利克雷分布采样权重"""
        alpha = np.array(MOBO_CONFIG['dirichlet_alpha'])
        return np.random.dirichlet(alpha)
    
    def _llm_weight_function(self, X: np.ndarray) -> float:
        """
        LLM空间加权（LLAMBO公式9）
        
        调用LLAMBOWeighting计算权重：
        W_LLM(θ) = ∏ [1/√(2πσ²) exp(-(θ-μ)²/(2σ²))]
        
        参数：
            X: 候选点 [current1, switch_soc, current2]
        
        返回：
            weight: [0, 1]
        """
        if self.llm_weighting is None:
            return 1.0  # 无LLM，均匀权重
        
        # 处理批量输入
        if X.ndim == 2:
            return np.array([self.llm_weighting.compute_weight(x) for x in X])
        else:
            return self.llm_weighting.compute_weight(X)
    
    def _fit_gps(self):
        """拟合3个独立GP（修复：使用数据变换）"""
        # 应用数据变换并保存到类属性
        self.transformed_db = self.data_transformer.fit_transform_database(self.database)
        
        # 拟合GP（使用变换后的数据）
        self.mogp.fit(self.transformed_db)
        
        # 更新耦合矩阵（基于GP梯度）
        if len(self.database) >= 10:
            self.mogp.update_coupling_matrix(method='gradient')
        
        # 更新标量化器的边界（基于变换后的数据）
        bounds = self.data_transformer.get_transformed_bounds()
        self.scalarizer.ideal_point = bounds['ideal']
        self.scalarizer.reference_point = bounds['reference']
        self.scalarizer.range = bounds['reference'] - bounds['ideal']
        self.scalarizer.range = np.where(self.scalarizer.range > 1e-10, self.scalarizer.range, 1.0)
    
    def _optimize_acquisition(self, weights: np.ndarray) -> Dict:
        """优化采集函数（加权EI + Tchebycheff）"""
        # 使用采集函数优化器
        gp_list = self.mogp.get_gp_list()
        
        # 修复：传入变换后的数据库（Log空间）
        database_to_use = getattr(self, 'transformed_db', self.database)
        
        next_point = self.acq_optimizer.optimize(
            gp_list=gp_list,
            weights=weights,
            database=database_to_use,
            llm_weight_func=lambda X: self._llm_weight_function(X)
        )
        
        return next_point
    
    def _tchebycheff(self, objectives: np.ndarray, weights: np.ndarray) -> float:
        """Tchebycheff标量化"""
        # 归一化
        normalized = (objectives - self.ideal_point) / (self.reference_point - self.ideal_point)
        
        # Tchebycheff
        max_term = np.max(weights * normalized)
        sum_term = MOBO_CONFIG['eta'] * np.sum(weights * normalized)
        
        return max_term + sum_term
    
    def _update_gamma(self, iteration: int):
        """更新耦合强度gamma"""
        if iteration < 2:
            return
        
        # 计算最近两轮的改进率
        valid_data = [r for r in self.database if r['valid']]
        if len(valid_data) < 2:
            return
        
        # 提取最近两个标量化值（使用当前权重）
        recent_scalars = []
        for r in valid_data[-2:]:
            objectives = np.array([r['time'], r['temp'], r['aging']])
            # 使用均匀权重计算（简化）
            weights_uniform = np.array([1/3, 1/3, 1/3])
            scalar = self.scalarizer.scalarize(objectives, weights_uniform)
            recent_scalars.append(scalar)
        
        # 计算改进率
        if recent_scalars[0] > 1e-10:
            improvement_rate = (recent_scalars[0] - recent_scalars[1]) / recent_scalars[0]
        else:
            improvement_rate = 0.0
        
        # 更新gamma
        self.mogp.update_gamma(improvement_rate)
    
    def _save_checkpoint(self, iteration: int):
        """保存检查点"""
        if self.verbose:
            print(f"  保存检查点 (迭代{iteration})")
        
        # TODO: 实现中间结果保存
    
    def _analyze_results(self) -> Dict:
        """分析优化结果"""
        valid_data = [r for r in self.database if r['valid']]
        
        # 提取帕累托前沿（简化版）
        pareto_front = self._extract_pareto_front(valid_data)
        
        return {
            'database': self.database,
            'pareto_front': pareto_front,
            'n_evaluations': len(self.database),
            'n_valid': len(valid_data)
        }
    
    def _extract_pareto_front(self, data: List[Dict]) -> List[Dict]:
        """提取帕累托前沿（简化实现）"""
        if len(data) == 0:
            return []
        
        # 提取目标值
        objectives = np.array([[r['time'], r['temp'], r['aging']] for r in data])
        
        # 简单的非支配排序
        pareto_mask = np.ones(len(data), dtype=bool)
        
        for i in range(len(data)):
            for j in range(len(data)):
                if i != j:
                    # 检查j是否支配i
                    if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                        pareto_mask[i] = False
                        break
        
        pareto_indices = np.where(pareto_mask)[0]
        pareto_front = [data[i] for i in pareto_indices]
        
        return pareto_front
    
    def _save_final_results(self, results: Dict):
        """保存最终结果"""
        import os
        os.makedirs(DATA_CONFIG['save_dir'], exist_ok=True)
        
        # 保存JSON
        save_path_json = os.path.join(DATA_CONFIG['save_dir'], 'results.json')
        save_results_json(
            results=results,
            save_path=save_path_json
        )
        
        # 绘制图表
        save_path_history = os.path.join(DATA_CONFIG['save_dir'], 'optimization_history.png')
        save_path_pareto = os.path.join(DATA_CONFIG['save_dir'], 'pareto_front.png')
        
        plot_optimization_history(
            database=results['database'],
            save_path=save_path_history,
            show=False
        )
        
        plot_pareto_front_3d(
            database=results['database'],
            save_path=save_path_pareto,
            show=False
        )
        
        if self.verbose:
            print(f"\n最终结果:")
            print(f"  总评估次数: {results['n_evaluations']}")
            print(f"  有效样本: {results['n_valid']}")
            print(f"  帕累托前沿: {len(results['pareto_front'])} 个解")
            print(f"\n结果已保存至: {DATA_CONFIG['save_dir']}")


# ============================================================
# 主入口
# ============================================================
async def main():
    # 设置LLM API key（从环境变量或直接设置）
    import os
    llm_api_key = "sk-Evfy9FZGKZ31bpgdNsDSFfkWMopRE6EN4V4r801oRaIi8jm7"
    
    # 初始化优化器
    optimizer = LLMMOBO(
        llm_api_key=llm_api_key,
        n_warmstart=5,
        n_random_init=10,
        n_iterations=10,  
        verbose=True
    )
    
    # 运行优化
    results = await optimizer.optimize()
    
    print("\n优化完成！")
    return results


if __name__ == "__main__":
    asyncio.run(main())