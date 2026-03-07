"""
LLM-MOBO 主程序（第三轮重构：闭环集成层）
新增：
1. ExperimentDatabase（三层数据库）- 导入自database.py
2. compute_hypervolume（HV反馈）- 导入自database.py
3. LLM敏感度排序 + ARD长度尺度
4. 生成式采集函数
5. HV驱动的γ自适应
"""

import numpy as np
import asyncio
import os
from typing import Dict, List

# 导入配置
from config import (
    BATTERY_CONFIG, PARAM_BOUNDS,
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

# 导入独立的数据库模块
from database import ExperimentDatabase, compute_hypervolume, compute_hypervolume_normalized


# ============================================================
# 主类：LLMMOBO (第三轮重构)
# ============================================================
class LLMMOBO:
    """
    LLM增强的多目标贝叶斯优化器（第三轮：闭环集成）
    
    新增功能：
    1. ExperimentDatabase三层数据库
    2. Hypervolume反馈闭环
    3. LLM敏感度排序 + ARD长度尺度
    4. 生成式采集函数
    5. HV驱动的γ自适应
    """
    
    def __init__(
        self,
        llm_api_key: str = None,
        n_warmstart: int = 5,
        n_random_init: int = 10,
        n_iterations: int = 50,
        gamma_init: float = 0.5,
        verbose: bool = True,
        use_coupling: bool = True,
        use_warmstart: bool = True,
        use_llm_acq: bool = True,
        use_llm_weighting: bool = True,
        gamma_adaptive: bool = True,
        db_path: str = ':memory:'
    ):
        """
        初始化优化器
        
        参数:
            llm_api_key: LLM API密钥
            n_warmstart: LLM热启动样本数
            n_random_init: 随机初始化样本数
            n_iterations: BO迭代次数
            gamma_init: 初始耦合强度
            verbose: 详细输出
            use_coupling: 启用物理耦合核
            use_warmstart: 启用LLM热启动
            use_llm_acq: 启用LLM空间加权
            db_path: 数据库路径
        """
        self.verbose = verbose
        self.n_warmstart = n_warmstart
        self.n_random_init = n_random_init
        self.n_iterations = n_iterations
        self.gamma = gamma_init
        
        # 消融标志
        self.use_coupling = use_coupling
        self.use_warmstart = use_warmstart
        self.use_llm_acq = use_llm_acq
        self.use_llm_weighting = use_llm_weighting
        self.gamma_adaptive = gamma_adaptive
        
        # ========== 新增：三层数据库 ==========
        self.db = ExperimentDatabase(db_path)
        # ====================================
        
        # ========== 新增：HV跟踪（归一化到[0,1]）==========
        self.hv_history = []
        self.reference_point_original = np.array([
            MOBO_CONFIG['reference_point']['time'],
            MOBO_CONFIG['reference_point']['temp'],
            MOBO_CONFIG['reference_point']['aging']
        ])
        self.ideal_point_original = np.array([
            MOBO_CONFIG['ideal_point']['time'],
            MOBO_CONFIG['ideal_point']['temp'],
            MOBO_CONFIG['ideal_point']['aging']
        ])
        # =================================
        
        # 初始化仿真器
        self.simulator = BatterySimulator(
            param_set=BATTERY_CONFIG['param_set'],
            init_voltage=BATTERY_CONFIG['init_voltage'],
            init_temp=BATTERY_CONFIG['init_temp'],
            sample_time=BATTERY_CONFIG['sample_time'],
            voltage_max=BATTERY_CONFIG['voltage_max'],
            temp_max=BATTERY_CONFIG['temp_max'],
            soc_target=BATTERY_CONFIG['soc_target']
        )
        
        # 初始化LLM客户端
        self.llm_enabled = llm_api_key is not None
        if self.llm_enabled:
            self.llm_warmstart = LLMWarmStart(
                api_key=llm_api_key,
                verbose=self.verbose
            )
            self.llm_coupling = LLMCouplingInference(
                api_key=llm_api_key,
                verbose=self.verbose
            )
            
            if self.use_llm_acq and self.use_llm_weighting:
                self.llm_weighting = LLAMBOWeighting(
                    param_bounds=PARAM_BOUNDS,
                    llm_api_key=llm_api_key,
                    base_url=LLM_CONFIG['base_url'],
                    model=LLM_CONFIG['model'],
                    verbose=self.verbose
                )
            else:
                self.llm_weighting = None
        else:
            self.llm_warmstart = None
            self.llm_coupling = None
            self.llm_weighting = None
        
        # 数据变换器
        self.data_transformer = DataTransformer(
            enable_log_aging=True,
            verbose=self.verbose
        )
        
        # 多目标GP管理器
        self.mogp = MOGPModel(
            use_coupling=self.use_coupling,
            gamma_init=gamma_init,
            verbose=self.verbose
        )
        
        # 理想点和参考点(Log空间)
        if DATA_CONFIG.get('enable_log_aging', True):
            ideal_aging = np.log10(1e-6 + 1e-10)
            reference_aging = np.log10(0.1 + 1e-10)
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
        
        # Tchebycheff标量化器
        self.scalarizer = TchebycheffScalarizer(
            ideal_point=self.ideal_point,
            reference_point=self.reference_point,
            eta=MOBO_CONFIG['eta']
        )
        
        # 采集函数优化器
        self.acq_optimizer = AcquisitionOptimizer(
            param_bounds=PARAM_BOUNDS,
            scalarizer=self.scalarizer,
            enable_llm_weighting=(self.llm_enabled and self.use_llm_acq),
            llm_api_key=llm_api_key if (self.llm_enabled and self.use_llm_acq) else None,
            base_url=LLM_CONFIG.get('base_url'),
            model=LLM_CONFIG.get('model'),
            verbose=self.verbose
        )
        
        if self.verbose:
            print("="*70)
            print("LLM-MOBO 初始化完成（第三轮：闭环集成）")
            print("="*70)
            print(f"仿真器: PyBaMM SPMe ({BATTERY_CONFIG['param_set']})")
            print(f"LLM增强: {'启用' if self.llm_enabled else '禁用'}")
            print(f"热启动样本: {n_warmstart}")
            print(f"随机初始化: {n_random_init}")
            print(f"BO迭代次数: {n_iterations}")
            print(f"初始gamma: {gamma_init}")
            print(f"数据库: {'内存' if db_path == ':memory:' else db_path}")
            print("="*70)
    
    async def optimize(self):
        """
        主优化循环（第三轮重构：12步闭环）
        
        流程：
        A. 采样权重
        B. 数据变换
        C. LLM敏感度排序 + ARD长度尺度
        D. LLM耦合矩阵
        E. GP训练(Pilot→Gradient→Merge→Main)
        F. 更新标量化器
        G. LLM焦点更新
        H. 生成式采集优化
        I. 仿真评估
        J. HV反馈 + γ自适应
        K. 状态记录
        L. 周期性检查点
        """
        if self.verbose:
            print("\n开始优化...")
        
        # ===== 阶段1: 热启动 + 随机初始化 =====
        if self.verbose:
            print(f"\n[阶段1] 初始化 ({self.n_warmstart + self.n_random_init} 个样本)")
        
        # 1.1 LLM Warm Start
        if self.llm_enabled and self.use_warmstart and self.n_warmstart > 0:
            warmstart_samples = await self._llm_warmstart()
            for sample in warmstart_samples:
                self._evaluate_and_store(sample)
        
        # 1.2 随机初始化（确保不使用固定种子）
        np.random.seed(int.from_bytes(os.urandom(4), 'big'))  # 使用真正不可预测的种子
        for _ in range(self.n_random_init):
            random_sample = self._random_sample()
            self._evaluate_and_store(random_sample)
        
        if self.verbose:
            n_valid = len(self.db.get_valid_experiments())
            print(f"  初始化完成: {n_valid} 个有效样本")
        
        # ===== 阶段2: BO主循环（12步闭环）=====
        for iteration in range(self.n_iterations):
            if self.verbose:
                print(f"\n[迭代 {iteration+1}/{self.n_iterations}]")
            
            # ========== A. 采样权重 ==========
            weights = self._sample_weights()
            if self.verbose:
                print(f"  [A] 权重: time={weights[0]:.2f}, temp={weights[1]:.2f}, aging={weights[2]:.2f}")
            
            # ========== B. 数据变换 ==========
            legacy_db = self.db.to_legacy_format()
            transformed_db = self.data_transformer.fit_transform_database(legacy_db)
            if self.verbose:
                print(f"  [B] 数据变换完成 (Log aging)")
            
            # ========== C. LLM敏感度排序 + ARD长度尺度 ==========
            if self.llm_weighting is not None and self.llm_enabled and self.use_llm_weighting:
                try:
                    ranking = await self.llm_weighting.infer_sensitivity_ranking(
                        database=legacy_db,
                        iteration=iteration + 1
                    )
                    
                    # 获取LLM驱动的长度尺度并注入GP
                    llm_length_scales = self.llm_weighting.ard_length_scales
                    if llm_length_scales is not None:
                        self.mogp.set_ard_length_scales(llm_length_scales)
                    
                    if self.verbose:
                        print(f"  [C] 敏感度排序: {dict(zip(self.llm_weighting.param_names, ranking))}")
                        print(f"      ARD长度尺度: {llm_length_scales}")
                except Exception as e:
                    if self.verbose:
                        print(f"  [C] 敏感度排序失败: {e}，使用上一轮结果")
            else:
                if self.verbose:
                    print(f"  [C] LLM未启用，跳过敏感度排序")
            
            # ========== D. LLM耦合矩阵推断 ==========
            W_llm = None
            if self.llm_enabled and self.use_coupling and self.llm_coupling is not None:
                if self.verbose:
                    print(f"  [D] LLM推理耦合矩阵")
                try:
                    # 提取最近的观测数据供LLM分析
                    coupling_data = legacy_db[-15:]  # 最近15条记录
                    W_llm = await self.llm_coupling.infer_coupling_matrix(
                        param_names=list(PARAM_BOUNDS.keys()),
                        current_data=coupling_data
                    )
                    if W_llm is not None:
                        self.mogp.set_llm_coupling_matrix(W_llm)
                        if self.verbose:
                            print(f"      W_llm已更新 (对角线外最大值: {np.max(W_llm - np.eye(4)):.3f})")
                except Exception as e:
                    if self.verbose:
                        print(f"      LLM推理失败: {e}，将仅使用W_data")
            else:
                if self.verbose:
                    print(f"  [D] 跳过LLM耦合推断")
            
            # ========== E. GP训练（Pilot → 梯度耦合 → 融合W_llm → Main GP）==========
            # E.1 训练Pilot GP（标准ARD Matern，用于梯度估计）
            gp_pilot = self.mogp.train_pilot_gp(transformed_db)
            
            # E.2 从Pilot GP梯度估计数据驱动耦合矩阵
            from config import get_algorithm_param
            n_samples_grad = min(10, len([r for r in transformed_db if r['valid']]))
            W_data = self.mogp.estimate_coupling_from_gradients(
                gp_pilot, transformed_db, n_samples=n_samples_grad
            )
            
            # E.3 融合W_data和W_llm（如果有）
            W_llm_stored = getattr(self.mogp, 'W_llm', None)
            W_final = self.mogp.merge_coupling_matrices(
                W_data, W_llm_stored,
                alpha=get_algorithm_param('composite_kernel', 'coupling_matrix_alpha', 0.5)
            )
            
            # E.4 训练Main GP（使用融合后的耦合矩阵）
            self.mogp.train_main_gp(transformed_db, coupling_matrix=W_final)
            
            if self.verbose:
                print(f"  [E] GP训练完成 (W_final融合: data×{get_algorithm_param('composite_kernel', 'coupling_matrix_alpha', 0.5):.0%} + LLM×{1-get_algorithm_param('composite_kernel', 'coupling_matrix_alpha', 0.5):.0%})")
            
            # ========== F. 更新标量化器 ==========
            bounds = self.data_transformer.get_transformed_bounds()
            self.scalarizer.ideal_point = bounds['ideal']
            self.scalarizer.reference_point = bounds['reference']
            self.scalarizer.range = bounds['reference'] - bounds['ideal']
            self.scalarizer.range = np.where(self.scalarizer.range > 1e-10, self.scalarizer.range, 1.0)
            if self.verbose:
                print(f"  [F] 标量化器更新")
            
            # ========== G. LLM焦点更新 ==========
            if self.llm_weighting is not None and self.use_llm_weighting and iteration % 5 == 0 and iteration > 0:
                if self.verbose:
                    print(f"  [G] LLM焦点更新")
                await self.llm_weighting.update_focus_from_llm(
                    database=legacy_db,
                    gp_list=self.mogp.get_gp_list()
                )
            
            # ========== H. 生成式采集优化 ==========
            gp_list = self.mogp.get_gp_list()
            next_point = await self.acq_optimizer.optimize(
                gp_list=gp_list,
                weights=weights,
                database=transformed_db,
                llm_weight_func=lambda X: self._llm_weight_function(X),
                iteration=iteration + 1,
                total_iterations=self.n_iterations,
                scalarizer=self.scalarizer,       # 传入 Step F 更新的标量化器（空间一致）
                legacy_db=legacy_db               # 传入原始空间数据（供 LLM prompt 使用）
            )
            if self.verbose:
                print(f"  [H] 下一个点: I1={next_point['current1']:.2f}, T1={next_point['time1']:.2f}, I2={next_point['current2']:.2f}, V_sw={next_point['v_switch']:.2f}")
            
            # ========== I. 仿真评估 ==========
            self._evaluate_and_store(next_point)
            
            # ========== J. HV反馈 + γ自适应 ==========
            hv_current = self._compute_and_update_hv()
            if self.gamma_adaptive:
                self._adaptive_gamma_by_hv(hv_current)
            if self.verbose:
                print(f"  [J] HV={hv_current:.4f}, γ={self.gamma:.4f}")
            
            # ========== K. 状态记录 ==========
            self.db.add_state(iteration, {
                'weights': weights,
                'gamma': self.gamma,
                'hypervolume': hv_current,
                'n_pareto': len(self.db.get_pareto_front()),
                'llm_focus_mu': getattr(self.llm_weighting, 'mu', {}),
                'llm_focus_sigma': getattr(self.llm_weighting, 'sigma', {})
            })
            
            # ========== L. 周期性检查点 ==========
            if (iteration + 1) % DATA_CONFIG.get('save_interval', 10) == 0:
                self._save_checkpoint(iteration + 1)
        
        # ===== 阶段3: 结果分析 =====
        if self.verbose:
            print("\n[阶段3] 优化完成，分析结果...")
        
        results = self._analyze_results()
        self._save_final_results(results)
        
        return results
    
    async def _llm_warmstart(self) -> List[Dict]:
        """LLM热启动"""
        if not self.use_warmstart or self.llm_warmstart is None:
            if self.verbose:
                print("  [LLM WarmStart] 已禁用，使用随机采样")
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
        """随机采样一个充电策略(4D)"""
        sample = {
            'current1': np.random.uniform(*PARAM_BOUNDS['current1']),
            'time1': np.random.uniform(*PARAM_BOUNDS['time1']),
            'current2': np.random.uniform(*PARAM_BOUNDS['current2']),
            'v_switch': np.random.uniform(*PARAM_BOUNDS['v_switch'])
        }
        
        # 显式边界检查（防御性编程）
        for key in sample:
            sample[key] = np.clip(sample[key], PARAM_BOUNDS[key][0], PARAM_BOUNDS[key][1])
        
        if self.verbose:
            print(f"    [DEBUG] 随机参数: I1={sample['current1']:.4f}, T1={sample['time1']:.4f}, I2={sample['current2']:.4f}, Vsw={sample['v_switch']:.4f}")
        return sample
    
    def _evaluate_and_store(self, params: Dict):
        """
        评估充电策略并存储到数据库
        
        参数:
            params: {'current1': ..., 'time1': ..., 'current2': ..., 'v_switch': ...,
                     'rationale': ...(可选), 'scenario': ...(可选)}
        """
        # 提取元数据
        meta = {
            'rationale': params.get('rationale', ''),
            'scenario': params.get('scenario', '')
        }
        
        # 提取仿真参数（剔除元数据字段）
        sim_params = {k: v for k, v in params.items()
                      if k in ('current1', 'time1', 'current2', 'v_switch')}
        
        # 执行仿真
        result = self.simulator.simulate(**sim_params)
        
        # 存储到数据库
        self.db.add_experiment(params, result, meta)
        
        # 日志输出
        if self.verbose:
            if result['valid']:
                log_msg = (f"    评估: time={result['time']}, "
                          f"temp={result['temp']:.1f}K, "
                          f"aging={result['aging']:.6f}")
                
                if meta['rationale']:
                    rationale_short = (meta['rationale'][:50] + '...' 
                                     if len(meta['rationale']) > 50 
                                     else meta['rationale'])
                    log_msg += f"\n          策略: {rationale_short}"
                
                if meta['scenario']:
                    log_msg += f" [{meta['scenario']}]"
                
                print(log_msg)
            else:
                print(f"    约束违反: {result['violation']}")
    
    def _sample_weights(self) -> np.ndarray:
        """从狄利克雷分布采样权重"""
        alpha = np.array(MOBO_CONFIG['dirichlet_alpha'])
        return np.random.dirichlet(alpha)
    
    def _llm_weight_function(self, X: np.ndarray) -> float:
        """LLM空间加权"""
        if self.llm_weighting is None:
            return 1.0
        
        if X.ndim == 2:
            return np.array([self.llm_weighting.compute_weight(x) for x in X])
        else:
            return self.llm_weighting.compute_weight(X)
    
    def _compute_and_update_hv(self) -> float:
        """
        计算当前归一化Hypervolume并更新历史
        
        使用 compute_hypervolume_normalized 将 HV 归一化到 [0, 1]，
        避免原始空间混合单位（秒×K×百分比）导致 γ 自适应失效。
        
        返回:
            hv_current: 当前归一化HV值 ∈ [0, 1]
        """
        pareto_front_records = self.db.get_pareto_front()
        
        if len(pareto_front_records) == 0:
            hv_current = 0.0
        else:
            pareto_objectives = np.array([
                [r['time'], r['temp'], r['aging']]
                for r in pareto_front_records
            ])
            hv_current = compute_hypervolume_normalized(
                pareto_objectives,
                self.reference_point_original,
                self.ideal_point_original
            )
        
        self.hv_history.append(hv_current)
        
        return hv_current
    
    def _adaptive_gamma_by_hv(self, hv_current: float):
        """
        基于HV增量自适应更新γ（PE-GenBO §3.6）
        
        公式: γ_{t+1} = clip(γ_t × (1 + ρ × ΔHV / (|HV_t| + ε)), γ_min, γ_max)
        
        物理直觉：
        - ΔHV > 0（Pareto前沿改善）→ 物理先验在帮忙 → 增大γ，更信任LLM
        - ΔHV ≤ 0（前沿停滞）→ 物理先验可能有幻觉 → 减小γ，更信任数据
        
        参数:
            hv_current: 当前HV值
        """
        if len(self.hv_history) < 2:
            return
        
        hv_prev = self.hv_history[-2]
        delta_hv = hv_current - hv_prev
        
        # 从config读取超参数（不硬编码）
        rho = BO_CONFIG.get('gamma_update_rate', 0.1)
        gamma_min = BO_CONFIG.get('gamma_min', 0.1)
        gamma_max = BO_CONFIG.get('gamma_max', 2.0)
        epsilon = 1e-10  # 防止除零
        
        # 框架公式：γ_{t+1} = γ_t × (1 + ρ × ΔHV / (|HV_t| + ε))
        gamma_new = self.gamma * (1.0 + rho * delta_hv / (abs(hv_current) + epsilon))
        
        # 裁剪到合理范围
        self.gamma = float(np.clip(gamma_new, gamma_min, gamma_max))
        
        # 同步到GP模型
        self.mogp.gamma = self.gamma
    
    def _save_checkpoint(self, iteration: int):
        """保存检查点"""
        if self.verbose:
            print(f"  [L] 保存检查点 (迭代{iteration})")
        
        import os
        os.makedirs(DATA_CONFIG['save_dir'], exist_ok=True)
        
        checkpoint_path = os.path.join(
            DATA_CONFIG['save_dir'],
            f'checkpoint_iter{iteration}.db'
        )
        self.db.save(checkpoint_path)
    
    def _analyze_results(self) -> Dict:
        """分析优化结果"""
        legacy_db = self.db.to_legacy_format()
        pareto_front = self.db.get_pareto_front()
        
        return {
            'database': legacy_db,
            'pareto_front': pareto_front,
            'n_evaluations': len(legacy_db),
            'n_valid': len(self.db.get_valid_experiments()),
            'hv_history': self.hv_history
        }
    
    def _save_final_results(self, results: Dict):
        """保存最终结果"""
        import os
        os.makedirs(DATA_CONFIG['save_dir'], exist_ok=True)
        
        # 保存数据库
        db_path = os.path.join(DATA_CONFIG['save_dir'], 'experiments.db')
        self.db.save(db_path)
        
        # 保存JSON
        save_path_json = os.path.join(DATA_CONFIG['save_dir'], 'results.json')
        save_results_json(results=results, save_path=save_path_json)
        
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
            print(f"  最终HV: {results['hv_history'][-1]:.4f}")
            print(f"\n结果已保存至: {DATA_CONFIG['save_dir']}")
        
        # 关闭数据库
        self.db.close()


# ============================================================
# 主入口
# ============================================================
async def main():
    import os
    llm_api_key = os.getenv('LLM_API_KEY', None)
    
    # 如果环境变量未设置，使用默认key
    if llm_api_key is None:
        llm_api_key = "sk-Sq1zyC8PLM8gafI2fpAccWpzBAzZvuNOPU6ZC9aWA6C883IK"
    
    # 初始化优化器（第三轮）
    optimizer = LLMMOBO(
        llm_api_key=llm_api_key,
        n_warmstart=5,
        n_random_init=10,
        n_iterations=10,
        verbose=True,
        db_path=':memory:'
    )
    
    # 运行优化
    results = await optimizer.optimize()
    
    print("\n优化完成！")
    return results


if __name__ == "__main__":
    asyncio.run(main())
