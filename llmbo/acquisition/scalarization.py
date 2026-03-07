"""
Tchebycheff标量化模块
实现增强切比雪夫分解（Augmented Tchebycheff）
"""

import numpy as np
from typing import Dict, Tuple


class TchebycheffScalarizer:
    """
    增强切比雪夫标量化器
    
    公式：g(f, λ) = max_i {λ_i * |f_i - z_i*|} + η * Σ_i λ_i * |f_i - z_i*|
    
    其中：
    - f: 目标向量
    - λ: 权重向量
    - z*: 理想点
    - η: 增强系数
    """
    
    def __init__(
        self,
        ideal_point: np.ndarray,
        reference_point: np.ndarray,
        eta: float = 0.05
    ):
        """
        初始化标量化器
        
        参数：
            ideal_point: 理想点 z* (最小化问题的最小值)
            reference_point: 参考点（最坏情况，用于归一化）
            eta: 增强系数
        """
        self.ideal_point = np.array(ideal_point)
        self.reference_point = np.array(reference_point)
        self.eta = eta
        
        # 计算归一化范围
        self.range = self.reference_point - self.ideal_point
        
        # 避免除零
        self.range = np.where(self.range > 1e-10, self.range, 1.0)
    
    def scalarize(
        self,
        objectives: np.ndarray,
        weights: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        标量化目标向量（支持批量）
        
        参数：
            objectives: 目标值
                - 单样本: (3,) → 返回标量
                - 批量: (n_samples, 3) → 返回 (n_samples,)
            weights: 权重 λ
                - (3,) 或 (n_samples, 3)
            normalize: 是否归一化
        
        返回：
            scalar: 标量化值（越小越好）
                - 单样本: float
                - 批量: (n_samples,)
        """
        # 处理输入维度
        is_single_sample = objectives.ndim == 1
        if is_single_sample:
            objectives = objectives.reshape(1, -1)  # (1, 3)
        
        # 确保weights是2D
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)  # (1, 3)
        
        # 归一化（映射到 [0, 1]）
        if normalize:
            # Broadcasting: (n_samples, 3) - (3,) → (n_samples, 3)
            normalized = (objectives - self.ideal_point) / self.range
        else:
            normalized = objectives - self.ideal_point
        
        # Tchebycheff标量化（向量化）
        # weighted: (n_samples, 3)
        weighted = weights * np.abs(normalized)
        
        # max项：捕捉最差维度
        # (n_samples, 3) → (n_samples,)
        max_term = np.max(weighted, axis=1)
        
        # sum项：平衡所有维度
        sum_term = self.eta * np.sum(weighted, axis=1)
        
        # 标量化值
        scalar = max_term + sum_term
        
        # 如果输入是单样本，返回标量
        if is_single_sample:
            return float(scalar[0])
        else:
            return scalar
    
    def scalarize_batch(
        self,
        objectives_batch: np.ndarray,
        weights: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        批量标量化（向量化版本）
        
        参数：
            objectives_batch: (n_samples, n_objectives)
            weights: (n_objectives,) 或 (n_samples, n_objectives)
            normalize: 是否归一化
        
        返回：
            scalars: (n_samples,)
        """
        # 输入验证
        assert objectives_batch.ndim == 2, \
            f"objectives_batch must be 2D, got shape {objectives_batch.shape}"
        assert objectives_batch.shape[1] == len(self.ideal_point), \
            f"objectives dimension mismatch: {objectives_batch.shape[1]} != {len(self.ideal_point)}"
        
        # 直接调用向量化的scalarize
        return self.scalarize(objectives_batch, weights, normalize)
    
    def update_bounds(self, database: list):
        """
        动态更新理想点和参考点（基于历史数据）
        
        参数：
            database: 评估历史
        """
        valid_data = [r for r in database if r['valid']]
        
        if len(valid_data) < 2:
            return
        
        # 提取目标值
        times = np.array([r['time'] for r in valid_data])
        temps = np.array([r['temp'] for r in valid_data])
        agings = np.array([r['aging'] for r in valid_data])
        
        # 更新理想点（取最小值）
        new_ideal = np.array([
            np.min(times),
            np.min(temps),
            np.min(agings)
        ])
        
        # 更新参考点（取95分位数，避免极端值）
        new_ref = np.array([
            np.percentile(times, 95),
            np.percentile(temps, 95),
            np.percentile(agings, 95)
        ])
        
        # 平滑更新（避免剧烈波动）
        alpha = 0.3
        self.ideal_point = alpha * new_ideal + (1 - alpha) * self.ideal_point
        self.reference_point = alpha * new_ref + (1 - alpha) * self.reference_point
        
        # 重新计算范围
        self.range = self.reference_point - self.ideal_point
        self.range = np.where(self.range > 1e-10, self.range, 1.0)


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("测试 TchebycheffScalarizer...")
    
    # 初始化
    ideal_point = np.array([10, 298.15, 0.0])
    reference_point = np.array([300, 309.0, 0.1])
    
    scalarizer = TchebycheffScalarizer(
        ideal_point=ideal_point,
        reference_point=reference_point,
        eta=0.05
    )
    
    # 测试单个目标
    objectives = np.array([50, 305.0, 0.02])
    weights = np.array([0.4, 0.35, 0.25])
    
    scalar = scalarizer.scalarize(objectives, weights)
    
    print(f"\n理想点: {ideal_point}")
    print(f"参考点: {reference_point}")
    print(f"\n目标: {objectives}")
    print(f"权重: {weights}")
    print(f"标量化值: {scalar:.4f}")
    
    # 测试批量标量化
    print("\n测试批量标量化...")
    objectives_batch = np.array([
        [50, 305.0, 0.02],
        [100, 307.0, 0.05],
        [20, 300.0, 0.01]
    ])
    
    scalars = scalarizer.scalarize_batch(objectives_batch, weights)
    print(f"批量标量化结果: {scalars}")
    print(f"最优解索引: {np.argmin(scalars)}")
    
    # 测试动态更新
    print("\n测试动态更新...")
    database = [
        {'time': 50, 'temp': 305, 'aging': 0.02, 'valid': True},
        {'time': 80, 'temp': 307, 'aging': 0.04, 'valid': True},
        {'time': 30, 'temp': 302, 'aging': 0.01, 'valid': True},
    ]
    
    print(f"更新前理想点: {scalarizer.ideal_point}")
    scalarizer.update_bounds(database)
    print(f"更新后理想点: {scalarizer.ideal_point}")
    
    print("\n测试完成！")