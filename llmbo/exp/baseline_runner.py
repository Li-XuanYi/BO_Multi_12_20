"""
Baseline算法运行器

实现：
1. Random Search（纯随机采样）
2. Standard-BO（sklearn GP + Matern核 + EI采集）

设计原则：
- 与LLMMOBO共用BatterySimulator（保证公平性）
- 统一评估预算
- 相同的参数边界
- 相同的随机种子控制
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm
from scipy.optimize import minimize
import warnings

# 抑制sklearn警告
warnings.filterwarnings('ignore', category=UserWarning)


class BaselineOptimizer:
    """
    基准算法优化器
    
    统一接口：
    - 相同的仿真器
    - 相同的评估预算
    - 相同的参数边界
    """
    
    def __init__(
        self,
        simulator,
        param_bounds: Dict[str, Tuple[float, float]],
        n_iterations: int,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        初始化基准优化器
        
        参数：
            simulator: BatterySimulator实例
            param_bounds: 参数边界 {'current1': (3, 6), ...}
            n_iterations: 评估次数预算
            seed: 随机种子
            verbose: 详细输出
        """
        self.simulator = simulator
        self.param_bounds = param_bounds
        self.n_iterations = n_iterations
        self.seed = seed
        self.verbose = verbose
        
        # 数据库
        self.database = []
        
        # 转换为列表形式（用于scipy）
        self.bounds_list = [
            param_bounds['current1'],
            param_bounds['switch_soc'],
            param_bounds['current2']
        ]
        
        # 设置随机种子
        np.random.seed(seed)
    
    def _evaluate(self, params: Dict) -> Dict:
        """
        通用评估接口
        
        参数：
            params: {'current1': ..., 'switch_soc': ..., 'current2': ...}
        
        返回：
            record: 评估记录
        """
        # 提取物理参数
        sim_params = {
            'current1': float(params['current1']),
            'switch_soc': float(params['switch_soc']),
            'current2': float(params['current2'])
        }
        
        # 仿真
        result = self.simulator.simulate(**sim_params)
        
        # 构建记录
        record = {
            'params': sim_params,
            'time': result['time'],
            'temp': result['temp'],
            'aging': result['aging'],
            'valid': result['valid'],
            'violation': result['violation']
        }
        
        # 存储
        self.database.append(record)
        
        # 日志
        if self.verbose and len(self.database) % 5 == 0:
            valid_count = sum(1 for r in self.database if r['valid'])
            print(f"  [Baseline] 已评估 {len(self.database)}/{self.n_iterations}，"
                  f"有效点: {valid_count}")
        
        return record
    
    def run_random(self) -> List[Dict]:
        """
        运行Random Search（纯随机采样）
        
        返回：
            database: 评估历史
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"运行算法: Random Search")
            print(f"评估预算: {self.n_iterations}")
            print(f"随机种子: {self.seed}")
            print(f"{'='*60}")
        
        self.database = []
        
        for i in range(self.n_iterations):
            # 随机采样
            params = {
                'current1': np.random.uniform(*self.param_bounds['current1']),
                'switch_soc': np.random.uniform(*self.param_bounds['switch_soc']),
                'current2': np.random.uniform(*self.param_bounds['current2'])
            }
            
            # 评估
            self._evaluate(params)
        
        # 总结
        if self.verbose:
            valid_count = sum(1 for r in self.database if r['valid'])
            print(f"\n[Random] 完成")
            print(f"  有效点: {valid_count}/{len(self.database)}")
            print(f"  违反约束: {len(self.database) - valid_count}")
        
        return self.database
    
    def run_standard_bo(
        self,
        n_random_init: int = 10,
        acq_samples: int = 1000
    ) -> List[Dict]:
        """
        运行Standard-BO（无LLM，标准Matern核）
        
        配置：
        - 无LLM热启动（随机初始化）
        - 标准Matern核（无物理耦合）
        - EI采集函数
        - 两阶段优化（随机海选 + 局部精炼）
        
        参数：
            n_random_init: 随机初始化点数
            acq_samples: 采集优化候选数
        
        返回：
            database: 评估历史
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"运行算法: Standard-BO")
            print(f"评估预算: {self.n_iterations}")
            print(f"随机初始化: {n_random_init}")
            print(f"随机种子: {self.seed}")
            print(f"{'='*60}")
        
        self.database = []
        
        # ===== 阶段1: 随机初始化 =====
        if self.verbose:
            print(f"\n[阶段1] 随机初始化 ({n_random_init}点)...")
        
        for i in range(n_random_init):
            params = {
                'current1': np.random.uniform(*self.param_bounds['current1']),
                'switch_soc': np.random.uniform(*self.param_bounds['switch_soc']),
                'current2': np.random.uniform(*self.param_bounds['current2'])
            }
            self._evaluate(params)
        
        # ===== 阶段2: BO循环 =====
        if self.verbose:
            print(f"\n[阶段2] BO循环 ({self.n_iterations - n_random_init}轮)...")
        
        for iteration in range(self.n_iterations - n_random_init):
            if self.verbose and iteration % 5 == 0:
                print(f"\n  [BO轮次 {iteration+1}/{self.n_iterations - n_random_init}]")
            
            # 1. 拟合GP
            gp = self._fit_gp()
            
            # 2. 优化采集函数（EI）
            next_params = self._optimize_ei(gp, acq_samples)
            
            # 3. 评估
            self._evaluate(next_params)
        
        # 总结
        if self.verbose:
            valid_count = sum(1 for r in self.database if r['valid'])
            print(f"\n[Standard-BO] 完成")
            print(f"  有效点: {valid_count}/{len(self.database)}")
            print(f"  违反约束: {len(self.database) - valid_count}")
        
        return self.database
    
    def _fit_gp(self) -> GaussianProcessRegressor:
        """
        拟合单目标GP（使用加权标量化）
        
        策略：
        - 随机采样狄利克雷权重
        - 使用Tchebycheff标量化为单目标
        - 标准Matern核（nu=2.5）
        
        返回：
            gp: 训练好的GP
        """
        # 提取有效数据
        valid_data = [r for r in self.database if r['valid']]
        
        if len(valid_data) < 3:
            raise ValueError(f"有效数据不足：{len(valid_data)} < 3")
        
        # 构建训练数据
        X_train = np.array([[
            r['params']['current1'],
            r['params']['switch_soc'],
            r['params']['current2']
        ] for r in valid_data])
        
        # 随机采样权重（模拟多目标）
        weights = np.random.dirichlet([1.0, 1.0, 1.0])
        
        # Tchebycheff标量化
        y_train = []
        for r in valid_data:
            objectives = np.array([r['time'], r['temp'], r['aging']])
            
            # 简化标量化（线性加权）
            scalar = np.dot(weights, objectives)
            y_train.append(scalar)
        
        y_train = np.array(y_train)
        
        # 标准Matern核
        kernel = C(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        
        # 训练GP
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=True,
            random_state=None  # 允许随机性
        )
        
        gp.fit(X_train, y_train)
        
        return gp
    
    def _optimize_ei(
        self,
        gp: GaussianProcessRegressor,
        n_candidates: int
    ) -> Dict:
        """
        优化Expected Improvement采集函数
        
        两阶段策略：
        1. 随机海选n_candidates个点
        2. 选择top-5，用scipy局部优化
        
        参数：
            gp: 训练好的GP
            n_candidates: 候选点数
        
        返回：
            best_params: 最优参数
        """
        # 计算当前最优
        valid_data = [r for r in self.database if r['valid']]
        if len(valid_data) == 0:
            # 无有效点，随机采样
            return {
                'current1': np.random.uniform(*self.param_bounds['current1']),
                'switch_soc': np.random.uniform(*self.param_bounds['switch_soc']),
                'current2': np.random.uniform(*self.param_bounds['current2'])
            }
        
        # 简化：取time最小值作为current_best
        times = [r['time'] for r in valid_data]
        current_best = np.min(times)
        
        # ===== 阶段1: 随机海选 =====
        candidates = []
        ei_values = []
        
        for _ in range(n_candidates):
            x = np.array([
                np.random.uniform(*self.param_bounds['current1']),
                np.random.uniform(*self.param_bounds['switch_soc']),
                np.random.uniform(*self.param_bounds['current2'])
            ])
            
            # 计算EI
            ei = self._compute_ei(gp, x, current_best)
            
            candidates.append(x)
            ei_values.append(ei)
        
        # 选择top-5
        ei_values = np.array(ei_values)
        top_indices = np.argsort(ei_values)[-5:][::-1]
        
        # ===== 阶段2: 局部优化 =====
        best_ei = -np.inf
        best_x = None
        
        for idx in top_indices:
            x0 = candidates[idx]
            
            # 定义目标函数（最大化EI = 最小化负EI）
            def objective(x):
                return -self._compute_ei(gp, x, current_best)
            
            # L-BFGS-B优化
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds_list,
                options={'maxiter': 20}
            )
            
            if -result.fun > best_ei:
                best_ei = -result.fun
                best_x = result.x
        
        # 回退
        if best_x is None:
            best_idx = np.argmax(ei_values)
            best_x = candidates[best_idx]
        
        return {
            'current1': float(best_x[0]),
            'switch_soc': float(best_x[1]),
            'current2': float(best_x[2])
        }
    
    def _compute_ei(
        self,
        gp: GaussianProcessRegressor,
        x: np.ndarray,
        current_best: float,
        xi: float = 0.01
    ) -> float:
        """
        计算Expected Improvement
        
        公式：EI = (improvement) * Φ(Z) + σ * φ(Z)
        其中：Z = (current_best - μ - xi) / σ
        
        参数：
            gp: GP模型
            x: 候选点
            current_best: 当前最优值
            xi: 探索参数
        
        返回：
            ei: EI值
        """
        x = x.reshape(1, -1)
        
        # GP预测
        mu, std = gp.predict(x, return_std=True)
        mu = mu[0]
        std = std[0]
        
        # 避免除零
        if std < 1e-10:
            return 0.0
        
        # 计算EI
        improvement = current_best - mu - xi
        Z = improvement / std
        
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        
        return max(0.0, ei)
    
    def get_results(self) -> Dict:
        """
        获取结果摘要
        
        返回：
            results: {
                'database': 完整历史,
                'pareto_front': Pareto前沿点,
                'n_evaluations': 评估次数,
                'n_valid': 有效点数
            }
        """
        valid_data = [r for r in self.database if r['valid']]
        
        # 提取Pareto前沿（简化：所有有效点）
        pareto_front = valid_data
        
        return {
            'database': self.database,
            'pareto_front': pareto_front,
            'n_evaluations': len(self.database),
            'n_valid': len(valid_data)
        }


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 添加父目录到路径
    sys.path.append(str(Path(__file__).parent.parent))
    
    from battery_env.wrapper import BatterySimulator
    from config import BATTERY_CONFIG, AGING_CONFIG, PARAM_BOUNDS
    
    print("测试 BaselineOptimizer...")
    
    # 初始化仿真器
    simulator = BatterySimulator(
        param_set=BATTERY_CONFIG['param_set'],
        init_voltage=BATTERY_CONFIG['init_voltage'],
        init_temp=BATTERY_CONFIG['init_temp'],
        sample_time=BATTERY_CONFIG['sample_time'],
        voltage_max=BATTERY_CONFIG['voltage_max'],
        temp_max=BATTERY_CONFIG['temp_max'],
        soc_target=BATTERY_CONFIG['soc_target'],
        aging_config=AGING_CONFIG
    )
    
    # 测试Random Search
    print("\n" + "="*60)
    print("测试1: Random Search (10次评估)")
    print("="*60)
    
    optimizer_random = BaselineOptimizer(
        simulator=simulator,
        param_bounds=PARAM_BOUNDS,
        n_iterations=10,
        seed=42,
        verbose=True
    )
    
    results_random = optimizer_random.run_random()
    print(f"\nRandom Search完成，有效点: {len([r for r in results_random if r['valid']])}")
    
    # 测试Standard-BO
    print("\n" + "="*60)
    print("测试2: Standard-BO (15次评估)")
    print("="*60)
    
    optimizer_bo = BaselineOptimizer(
        simulator=simulator,
        param_bounds=PARAM_BOUNDS,
        n_iterations=15,
        seed=42,
        verbose=True
    )
    
    results_bo = optimizer_bo.run_standard_bo(n_random_init=5)
    print(f"\nStandard-BO完成，有效点: {len([r for r in results_bo if r['valid']])}")
    
    print("\n测试完成！")
