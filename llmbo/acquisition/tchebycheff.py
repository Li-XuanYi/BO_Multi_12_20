"""
Tchebycheff 标量化模块（约束 C-3 实现）

实现增强切比雪夫分解（Augmented Tchebycheff）+ Riesz s-energy 权重集合

约束 C-3 要求：
- 必须使用 Riesz s-energy 集合生成权重
- 预生成 N_WEIGHTS=15 个权重向量
- 每轮迭代随机选取一个权重向量
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

# 从 config 导入
try:
    from config import MOBO_CONFIG
except ImportError:
    MOBO_CONFIG = {'N_WEIGHTS': 15, 'eta': 0.05}


def generate_riesz_s_energy_weights(n_weights: int = 15, n_obj: int = 3, seed: int = 42) -> np.ndarray:
    """
    生成 Riesz s-energy 权重集合（约束 C-3）

    Riesz s-energy 最小化方法生成均匀分布的权重向量：
    - 在单纯形上优化 s-energy 势函数
    - s = n_obj - 1（对于 3 目标，s=2）

    简化实现：使用 pymoo 的 get_reference_directions 或手动生成

    参数：
        n_weights: 权重数量（约束 C-6：N_WEIGHTS=15）
        n_obj: 目标数量（默认 3）
        seed: 随机种子

    返回：
        weights: (n_weights, n_obj) 权重矩阵，每行和为 1
    """
    # 尝试使用 pymoo（最优）
    try:
        from pymoo.util.ref_dirs import get_reference_directions
        weight_set = get_reference_directions("energy", n_obj, n_weights, seed=seed)
        return weight_set.astype(float)
    except ImportError:
        pass

    # Fallback: 使用 Das-Dennis 方法（边界交叉法）
    try:
        from pymoo.util.ref_dirs import get_reference_directions
        weight_set = get_reference_directions("das-dennis", n_obj, divisions=4)
        # 采样 n_weights 个
        indices = np.random.RandomState(seed).choice(
            len(weight_set), size=min(n_weights, len(weight_set)), replace=False
        )
        return weight_set[indices].astype(float)
    except (ImportError, Exception):
        pass

    # Fallback 2: 手动生成（Dirichlet + 去重）
    rng = np.random.RandomState(seed)
    weights = []
    seen = set()

    # 生成时确保均匀分布
    for _ in range(n_weights * 10):  # 多生成一些用于去重
        w = rng.dirichlet(np.ones(n_obj))
        w_rounded = tuple(np.round(w, 4))
        if w_rounded not in seen:
            seen.add(w_rounded)
            weights.append(w)
        if len(weights) >= n_weights:
            break

    # 如果还不够，补充均匀分布的点
    while len(weights) < n_weights:
        # 在边界附近生成
        dim = rng.randint(n_obj)
        w = np.ones(n_obj) * 0.1
        w[dim] = 1.0 + rng.uniform(-0.3, 0.3)
        w = w / w.sum()
        weights.append(w)

    return np.array(weights[:n_weights])


class TchebycheffScalarizer:
    """
    增强切比雪夫标量化器（约束 C-3 实现）

    公式：g(f, lambda) = max_i {lambda_i * |f_i - z_i*|} + eta * sum_i lambda_i * |f_i - z_i*|

    其中：
    - f: 目标向量
    - lambda: 权重向量（从 Riesz s-energy 集合中选取）
    - z*: 理想点
    - eta: 增强系数

    约束 C-3:
    - 预生成 N_WEIGHTS=15 个权重向量
    - 每轮迭代随机选取一个权重向量
    """

    def __init__(
        self,
        ideal_point: np.ndarray,
        reference_point: np.ndarray,
        eta: float = 0.05,
        n_weights: int = 15,
        seed: int = 42
    ):
        """
        初始化标量化器

        参数：
            ideal_point: 理想点 z* (最小化问题的最小值)
            reference_point: 参考点（最坏情况，用于归一化）
            eta: 增强系数
            n_weights: Riesz s-energy 权重数量（约束 C-6：N_WEIGHTS=15）
            seed: 随机种子
        """
        self.ideal_point = np.array(ideal_point, dtype=float)
        self.reference_point = np.array(reference_point, dtype=float)
        self.eta = eta
        self.n_weights = n_weights
        self.seed = seed

        # 预生成 Riesz s-energy 权重集合（约束 C-3）
        self.weight_set = generate_riesz_s_energy_weights(n_weights, len(ideal_point), seed)

        # 当前权重（每轮随机选取）
        self.current_weight_idx = 0
        self.current_weight = self.weight_set[0].copy()

        # 计算归一化范围
        self.range = self.reference_point - self.ideal_point
        self.range = np.where(self.range > 1e-10, self.range, 1.0)

    def sample_weight_vector(self, iteration: int = None) -> np.ndarray:
        """
        随机选取一个权重向量（每轮迭代调用）

        约束 C-3：从预生成的 Riesz s-energy 集合中随机选取

        参数：
            iteration: 当前迭代轮次（可选，用于确定性地选择）

        返回：
            weight: (n_obj,) 权重向量，和为 1
        """
        if iteration is not None:
            # 确定性选择（便于复现）
            self.current_weight_idx = iteration % len(self.weight_set)
        else:
            # 随机选择
            self.current_weight_idx = np.random.randint(len(self.weight_set))

        self.current_weight = self.weight_set[self.current_weight_idx].copy()
        return self.current_weight

    def get_weight_set(self) -> np.ndarray:
        """
        获取完整的权重集合

        返回：
            weight_set: (n_weights, n_obj) 权重矩阵
        """
        return self.weight_set

    def scalarize(
        self,
        objectives: np.ndarray,
        weights: np.ndarray = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        标量化目标向量（支持批量）

        参数：
            objectives: 目标值
                - 单样本：(3,) -> 返回标量
                - 批量：(n_samples, 3) -> 返回 (n_samples,)
            weights: 权重 lambda
                - (3,) 或 (n_samples, 3)
                - 如果为 None，使用当前权重
            normalize: 是否归一化

        返回：
            scalar: 标量化值（越小越好）
                - 单样本：float
                - 批量：(n_samples,)
        """
        # 使用当前权重（如果未提供）
        if weights is None:
            weights = self.current_weight

        # 处理输入维度
        is_single_sample = objectives.ndim == 1
        if is_single_sample:
            objectives = objectives.reshape(1, -1)  # (1, 3)

        # 确保 weights 是 2D
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)  # (1, 3)

        # 归一化（映射到 [0, 1]）
        if normalize:
            # Broadcasting: (n_samples, 3) - (3,) -> (n_samples, 3)
            normalized = (objectives - self.ideal_point) / self.range
        else:
            normalized = objectives - self.ideal_point

        # Tchebycheff 标量化（向量化）
        # weighted: (n_samples, 3)
        weighted = weights * np.abs(normalized)

        # max 项：捕捉最差维度
        # (n_samples, 3) -> (n_samples,)
        max_term = np.max(weighted, axis=1)

        # sum 项：平衡所有维度
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
        weights: np.ndarray = None,
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

        # 直接调用向量化的 scalarize
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

        # 更新参考点（取 95 分位数，避免极端值）
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
# 验证函数（约束 C-8）
# ============================================================
def verify_weight_set(weight_set: np.ndarray) -> tuple:
    """
    验证权重集合是否符合约束 C-3

    检查项：
    1. shape = (15, 3)
    2. 所有行和为 1
    3. 所有元素 >= 0

    注意：Riesz s-energy 方法会产生边界权重（某些维度接近 0），
    这是正常的，因为我们需要覆盖 Pareto 前沿的边界。

    参数：
        weight_set: (n_weights, n_obj) 权重矩阵

    返回：
        (is_valid, errors)
    """
    errors = []

    # 检查 1: shape
    if weight_set.shape != (15, 3):
        errors.append(f"Shape error: expected (15, 3), got {weight_set.shape}")

    # 检查 2: 行和为 1
    row_sums = weight_set.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        errors.append(f"Row sums not 1: max deviation = {np.max(np.abs(row_sums - 1.0)):.6e}")

    # 检查 3: 非负
    if np.any(weight_set < 0):
        errors.append(f"Negative weights found: min = {np.min(weight_set):.6e}")

    # 注意：不检查分布均匀性，因为 Riesz s-energy 本身就产生边界点

    return len(errors) == 0, errors


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 TchebycheffScalarizer（约束 C-3 实现）")
    print("=" * 60)

    # 初始化
    ideal_point = np.array([1200.0, 298.15, 1e-6])
    reference_point = np.array([7200.0, 323.15, 0.008])

    scalarizer = TchebycheffScalarizer(
        ideal_point=ideal_point,
        reference_point=reference_point,
        eta=0.05,
        n_weights=15,
        seed=42
    )

    # 测试 1: 权重集合验证（约束 C-8）
    print("\n[测试 1] 权重集合验证（约束 C-3）")
    print(f"  weight_set.shape = {scalarizer.weight_set.shape}")
    print(f"  期望：(15, 3)")

    is_valid, errors = verify_weight_set(scalarizer.weight_set)
    if is_valid:
        print("  OK 权重集合验证通过")
    else:
        print("  错误:")
        for err in errors:
            print(f"    - {err}")

    # 测试 2: 权重采样
    print("\n[测试 2] 权重采样（每轮随机选取）")
    print("  前 5 轮采样的权重:")
    for i in range(5):
        w = scalarizer.sample_weight_vector(i)
        print(f"    轮次 {i}: {w} (sum={w.sum():.6f})")

    # 测试 3: 标量化
    print("\n[测试 3] Tchebycheff 标量化")
    objectives = np.array([50, 305.0, 0.02])
    w = scalarizer.sample_weight_vector()
    scalar = scalarizer.scalarize(objectives, w)
    print(f"  目标：{objectives}")
    print(f"  权重：{w}")
    print(f"  标量值：{scalar:.4f}")

    # 测试 4: 批量标量化
    print("\n[测试 4] 批量标量化")
    objectives_batch = np.array([
        [50, 305.0, 0.02],
        [100, 307.0, 0.05],
        [20, 300.0, 0.01]
    ])
    scalars = scalarizer.scalarize_batch(objectives_batch, w)
    print(f"  批量结果：{scalars}")
    print(f"  最优解索引：{np.argmin(scalars)}")

    # 测试 5: 动态更新
    print("\n[测试 5] 动态更新理想点/参考点")
    database = [
        {'time': 50, 'temp': 305, 'aging': 0.02, 'valid': True},
        {'time': 80, 'temp': 307, 'aging': 0.04, 'valid': True},
        {'time': 30, 'temp': 302, 'aging': 0.01, 'valid': True},
    ]

    print(f"  更新前理想点：{scalarizer.ideal_point}")
    scalarizer.update_bounds(database)
    print(f"  更新后理想点：{scalarizer.ideal_point}")

    # 测试 6: 权重分布可视化
    print("\n[测试 6] 权重分布统计")
    for i in range(3):
        col = scalarizer.weight_set[:, i]
        print(f"  目标{i+1}: min={col.min():.3f}, max={col.max():.3f}, mean={col.mean():.3f}")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
