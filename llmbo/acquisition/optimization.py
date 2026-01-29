"""
采集函数优化模块
实现MC-EI（蒙特卡洛期望改进）+ LLM权重
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from scipy.optimize import minimize
from scipy.stats import norm

from acquisition.scalarization import TchebycheffScalarizer
from config import MOBO_CONFIG, get_algorithm_param


class AcquisitionOptimizer:
    """
    采集函数优化器
    
    核心方法：
    1. MC-EI计算（蒙特卡洛期望改进）
    2. 两阶段优化（随机海选 + 局部精炼）
    3. LLM权重集成（EI-LLM）
    """
    
    def __init__(
        self,
        param_bounds: Dict,
        scalarizer=None,
        n_candidates: int = None,
        n_mc_samples: int = None,
        enable_llm_weighting: bool = False,
        verbose: bool = False
    ):
        """
        初始化采集优化器（参数从config读取）
        
        参数：
            param_bounds: 参数边界
            scalarizer: Tchebycheff标量化器（可选，会在optimize中动态更新）
            n_candidates: 候选点数（阶段1）
            n_mc_samples: MC采样数
            enable_llm_weighting: 是否启用LLM权重
            verbose: 详细输出
        """
        self.param_bounds = param_bounds
        
        # 从config读取默认值
        self.n_candidates = n_candidates if n_candidates is not None else get_algorithm_param('acquisition', 'n_candidates', 2000)
        self.n_mc_samples = n_mc_samples if n_mc_samples is not None else get_algorithm_param('acquisition', 'n_mc_samples', 128)
        self.enable_llm_weighting = enable_llm_weighting
        self.verbose = verbose
        
        # 转换为列表形式（用于scipy）
        self.bounds_list = [
            param_bounds['current1'],
            param_bounds['switch_soc'],
            param_bounds['current2']
        ]
        
        # 初始化标量化器（会在optimize()中动态更新）
        self.scalarizer = scalarizer
    
    def optimize(
        self,
        gp_list: List,
        weights: np.ndarray,
        database: List[Dict],
        llm_weight_func: Optional[Callable] = None
    ) -> Dict:
        """
        优化采集函数，找到下一个查询点
        
        流程：
        1. 初始化标量化器
        2. 随机海选n_candidates个点
        3. 计算MC-EI（可选：应用LLM权重）
        4. 选择top-5，局部优化
        5. 返回最优点
        
        参数：
            gp_list: GP模型列表 [gp_time, gp_temp, gp_aging]
            weights: Dirichlet权重 (3,)
            database: 评估历史
            llm_weight_func: LLM权重函数（可选）
        
        返回：
            best_params: {'current1': ..., 'switch_soc': ..., 'current2': ...}
        """
        # 从database推导理想点和参考点
        valid_data = [r for r in database if r['valid']]
        
        if len(valid_data) == 0:
            # 无有效点，随机采样
            return {
                'current1': np.random.uniform(*self.param_bounds['current1']),
                'switch_soc': np.random.uniform(*self.param_bounds['switch_soc']),
                'current2': np.random.uniform(*self.param_bounds['current2'])
            }
        
        # 提取目标值
        times = np.array([r['time'] for r in valid_data])
        temps = np.array([r['temp'] for r in valid_data])
        agings = np.array([r['aging'] for r in valid_data])
        
        # 动态理想点和参考点
        ideal_point = np.array([
            np.min(times),
            np.min(temps),
            np.min(agings)
        ])
        
        reference_point = np.array([
            MOBO_CONFIG['reference_point']['time'],
            MOBO_CONFIG['reference_point']['temp'],
            MOBO_CONFIG['reference_point']['aging']
        ])
        
        # 初始化标量化器
        self.scalarizer = TchebycheffScalarizer(
            ideal_point=ideal_point,
            reference_point=reference_point,
            eta=MOBO_CONFIG['eta']
        )
        
        # 计算当前最优（用于EI）
        current_best = self._compute_current_best(valid_data, weights)
        
        if self.verbose:
            print(f"    当前最优标量值: {current_best:.4f}")
        
        # ===== 阶段1: 随机海选 =====
        candidates = []
        ei_values = []
        
        for _ in range(self.n_candidates):
            x = np.array([
                np.random.uniform(*self.param_bounds['current1']),
                np.random.uniform(*self.param_bounds['switch_soc']),
                np.random.uniform(*self.param_bounds['current2'])
            ])
            
            # 计算MC-EI
            ei = self._compute_mc_ei(
                x, gp_list, weights, current_best, llm_weight_func
            )
            
            candidates.append(x)
            ei_values.append(ei)
        
        # 选择top-5
        ei_values = np.array(ei_values)
        top_indices = np.argsort(ei_values)[-5:][::-1]
        
        if self.verbose:
            print(f"    阶段1完成: {self.n_candidates}个候选点")
            print(f"    Top-5 EI: {[ei_values[i] for i in top_indices]}")
        
        # ===== 阶段2: 局部优化 =====
        best_ei = -np.inf
        best_x = None
        
        for idx in top_indices:
            x0 = candidates[idx]
            
            # 定义目标函数（最大化EI = 最小化负EI）
            def objective(x):
                ei = self._compute_mc_ei(
                    x, gp_list, weights, current_best, llm_weight_func
                )
                return -ei  # 最小化负EI
            
            # L-BFGS-B优化
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=self.bounds_list,
                    options={
                        'maxiter': get_algorithm_param('acquisition', 'local_maxiter', 20),
                        'ftol': get_algorithm_param('acquisition', 'local_ftol', 1e-6)
                    }
                )
                
                if -result.fun > best_ei:
                    best_ei = -result.fun
                    best_x = result.x
            
            except Exception as e:
                if self.verbose:
                    print(f"    [警告] 局部优化失败: {e}")
                continue
        
        # 回退
        if best_x is None:
            best_idx = np.argmax(ei_values)
            best_x = candidates[best_idx]
            best_ei = ei_values[best_idx]
        
        if self.verbose:
            print(f"    阶段2完成: 最优EI={best_ei:.6f}")
        
        return {
            'current1': float(best_x[0]),
            'switch_soc': float(best_x[1]),
            'current2': float(best_x[2])
        }
    
    def _compute_current_best(
        self,
        valid_data: List[Dict],
        weights: np.ndarray
    ) -> float:
        """
        计算当前最优标量化值
        
        参数：
            valid_data: 有效数据点
            weights: Dirichlet权重
        
        返回：
            current_best: 当前最优标量值（越小越好）
        """
        scalars = []
        
        for r in valid_data:
            objectives = np.array([r['time'], r['temp'], r['aging']])
            scalar = self.scalarizer.scalarize(objectives, weights, normalize=True)
            scalars.append(scalar)
        
        return np.min(scalars)
    
    def _compute_mc_ei(
        self,
        x: np.ndarray,
        gp_list: List,
        weights: np.ndarray,
        current_best: float,
        llm_weight_func: Optional[Callable] = None
    ) -> float:
        """
        计算蒙特卡洛期望改进（MC-EI）
        
        核心修正：
        - Tchebycheff是非线性的，必须先采样再标量化
        - 不能直接算E[Tcheb(f)]
        
        公式：
            EI(x) = E[max(0, f_best - Tcheb(f_sample))]
            其中 f_sample ~ GP(x)
        
        参数：
            x: 候选点 (3,)
            gp_list: GP列表
            weights: Dirichlet权重
            current_best: 当前最优标量值
            llm_weight_func: LLM权重函数（可选）
        
        返回：
            ei: 期望改进值
        """
        x = x.reshape(1, -1)
        
        # GP预测（均值和标准差）
        mu_list = []
        std_list = []
        
        for gp in gp_list:
            mu, std = gp.predict(x, return_std=True)
            mu_list.append(mu[0])
            std_list.append(std[0])
        
        mu_list = np.array(mu_list)
        std_list = np.array(std_list)
        
        # 避免除零
        std_list = np.maximum(std_list, 1e-10)
        
        # 蒙特卡洛采样
        samples = []
        
        for _ in range(self.n_mc_samples):
            # 从GP后验采样
            sample = np.random.normal(mu_list, std_list)
            samples.append(sample)
        
        samples = np.array(samples)  # (n_mc_samples, 3)
        
        # 对每个样本计算Tchebycheff标量化
        scalars = self.scalarizer.scalarize_batch(
            samples, weights, normalize=True
        )
        
        # 计算改进量
        improvements = np.maximum(0, current_best - scalars)
        
        # 期望改进
        ei = np.mean(improvements)
        
        # 应用LLM权重（如果提供）
        if llm_weight_func is not None:
            try:
                llm_weight = llm_weight_func(x[0])
                ei *= llm_weight
            except Exception as e:
                if self.verbose:
                    print(f"    [警告] LLM权重计算失败: {e}")
        
        return ei


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
    
    print("测试 AcquisitionOptimizer...")
    
    # 创建虚拟GP模型
    np.random.seed(42)
    X_train = np.random.rand(15, 3) * np.array([3.0, 0.4, 3.0]) + np.array([3.0, 0.3, 1.0])
    y_train_time = np.random.randint(30, 100, size=15)
    y_train_temp = np.random.uniform(300, 310, size=15)
    y_train_aging = np.random.uniform(0.001, 0.05, size=15)
    
    kernel = C(1.0) * Matern(nu=2.5)
    
    gp_time = GaussianProcessRegressor(kernel=kernel)
    gp_time.fit(X_train, y_train_time)
    
    gp_temp = GaussianProcessRegressor(kernel=kernel)
    gp_temp.fit(X_train, y_train_temp)
    
    gp_aging = GaussianProcessRegressor(kernel=kernel)
    gp_aging.fit(X_train, y_train_aging)
    
    gp_list = [gp_time, gp_temp, gp_aging]
    
    # 创建虚拟database
    fake_database = []
    for i in range(15):
        fake_database.append({
            'params': {
                'current1': X_train[i, 0],
                'switch_soc': X_train[i, 1],
                'current2': X_train[i, 2]
            },
            'time': y_train_time[i],
            'temp': y_train_temp[i],
            'aging': y_train_aging[i],
            'valid': True
        })
    
    # 初始化优化器
    param_bounds = {
        'current1': (3.0, 6.0),
        'switch_soc': (0.3, 0.7),
        'current2': (1.0, 4.0)
    }
    
    optimizer = AcquisitionOptimizer(
        param_bounds=param_bounds,
        n_candidates=500,  # 减少以加速测试
        n_mc_samples=64,   # 减少以加速测试
        verbose=True
    )
    
    # 测试优化
    weights = np.array([0.4, 0.35, 0.25])
    
    next_params = optimizer.optimize(
        gp_list=gp_list,
        weights=weights,
        database=fake_database,
        llm_weight_func=None
    )
    
    print(f"\n下一个查询点:")
    print(f"  current1: {next_params['current1']:.2f}")
    print(f"  switch_soc: {next_params['switch_soc']:.3f}")
    print(f"  current2: {next_params['current2']:.2f}")
    
    print("\n测试完成！")
