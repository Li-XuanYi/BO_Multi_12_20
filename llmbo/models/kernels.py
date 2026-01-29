"""
复合核函数模块
实现论文公式(5)(6)：k = k_RBF + γ·k_coupling
"""

import numpy as np
from sklearn.gaussian_process.kernels import (
    Kernel, StationaryKernelMixin, NormalizedKernelMixin,
    Matern, ConstantKernel as C, Hyperparameter
)

# 导入config
from config import get_algorithm_param
from scipy.spatial.distance import cdist


class CouplingKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """
    参数耦合核函数（公式6）
    
    k_coupling(θ, θ') = Σ_ij w_ij · (θ_i - θ'_i) · (θ_j - θ'_j)
    
    物理意义：捕捉参数间的交互效应
    """
    
    def __init__(self, coupling_matrix: np.ndarray = None, coupling_matrix_bounds=(1e-5, 1e5)):
        """
        初始化耦合核
        
        参数：
            coupling_matrix: 耦合权重矩阵 W (n_params, n_params)
            coupling_matrix_bounds: W的优化边界
        """
        if coupling_matrix is None:
            # 默认：单位矩阵（无耦合）
            coupling_matrix = np.eye(3)
        
        self.coupling_matrix = coupling_matrix
        self.coupling_matrix_bounds = coupling_matrix_bounds
    
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        计算核矩阵
        
        参数：
            X: (n_samples_X, n_features)
            Y: (n_samples_Y, n_features) or None
            eval_gradient: 是否计算梯度
        
        返回：
            K: 核矩阵
            K_gradient: 梯度（如果需要）
        """
        if Y is None:
            Y = X
        
        # 计算差值矩阵
        # diff[i, j] = X[i] - Y[j]
        # 形状: (n_samples_X, n_samples_Y, n_features)
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        
        # 计算加权平方差
        # k(x, y) = (x - y)^T W (x - y)
        # 形状: (n_samples_X, n_samples_Y)
        K = np.zeros((X.shape[0], Y.shape[0]))
        
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                delta = diff[i, j]  # (n_features,)
                K[i, j] = delta @ self.coupling_matrix @ delta
        
        # 高斯化：exp(-k)
        K = np.exp(-K)
        
        if eval_gradient:
            # 简化：不计算梯度（sklearn的GP优化器会自动处理）
            return K, np.empty((X.shape[0], Y.shape[0], 0))
        else:
            return K
    
    def diag(self, X):
        """返回对角线元素（自相关）"""
        return np.ones(X.shape[0])
    
    def is_stationary(self):
        """是否平稳核"""
        return True
    
    def __repr__(self):
        return f"CouplingKernel(shape={self.coupling_matrix.shape})"


class CompositeKernel(Kernel):
    """
    复合核函数（公式5）
    
    k(θ, θ') = k_RBF(θ, θ') + γ · k_coupling(θ, θ')
    
    组合：
    - k_RBF: 标准Matern核（捕捉平滑性）
    - k_coupling: 耦合核（捕捉参数交互）
    - γ: 耦合强度（动态调整）
    """
    
    def __init__(
        self,
        base_kernel: Kernel = None,
        coupling_matrix: np.ndarray = None,
        gamma: float = None,
        gamma_bounds: tuple = None
    ):
        """
        初始化复合核（参数从config读取）
        
        参数：
            base_kernel: 基础核（Matern/RBF）
            coupling_matrix: 耦合矩阵 W
            gamma: 耦合强度
            gamma_bounds: gamma的调整范围
        """
        # 从config读取默认值
        if gamma is None:
            gamma = get_algorithm_param('composite_kernel', 'gamma_init', 0.5)
        if gamma_bounds is None:
            gamma_bounds = get_algorithm_param('composite_kernel', 'gamma_bounds', (0.1, 2.0))
        
        # ========== P0修复：保存所有参数（sklearn要求） ==========
        # sklearn的clone()会调用get_params()，需要所有__init__参数都有对应属性
        self.coupling_matrix = coupling_matrix  # ← 必须保存！
        self.gamma = gamma
        self.gamma_bounds = gamma_bounds
        # =======================================================
        
        # 默认基础核：Matern(nu=2.5)
        if base_kernel is None:
            base_kernel = C(1.0) * Matern(nu=2.5, length_scale=1.0)
        
        self.base_kernel = base_kernel
        
        # 创建耦合核（使用保存的coupling_matrix）
        self.coupling_kernel = CouplingKernel(coupling_matrix=self.coupling_matrix)
    
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        计算复合核矩阵
        
        K = K_base + γ * K_coupling
        """
        if Y is None:
            Y = X
        
        # 计算两个核矩阵
        K_base = self.base_kernel(X, Y, eval_gradient=False)
        K_coupling = self.coupling_kernel(X, Y, eval_gradient=False)
        
        # 线性组合
        K = K_base + self.gamma * K_coupling
        
        if eval_gradient:
            # 简化：返回空梯度
            return K, np.empty((X.shape[0], Y.shape[0], 0))
        else:
            return K
    
    def diag(self, X):
        """对角线元素"""
        return self.base_kernel.diag(X) + self.gamma * self.coupling_kernel.diag(X)
    
    def is_stationary(self):
        """是否平稳"""
        return self.base_kernel.is_stationary()
    
    def update_gamma(self, new_gamma: float):
        """动态更新耦合强度"""
        self.gamma = np.clip(new_gamma, *self.gamma_bounds)
    
    def update_coupling_matrix(self, new_matrix: np.ndarray):
        """更新耦合矩阵"""
        # 同时更新两处
        self.coupling_matrix = new_matrix  # ← 更新自身属性
        self.coupling_kernel.coupling_matrix = new_matrix  # 更新子核
    
    def __repr__(self):
        return (f"CompositeKernel(\n"
                f"  base={self.base_kernel},\n"
                f"  gamma={self.gamma:.3f},\n"
                f"  coupling_shape={self.coupling_kernel.coupling_matrix.shape}\n"
                f")")


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("测试 CompositeKernel...")
    
    # 创建测试数据
    np.random.seed(42)
    X = np.random.rand(5, 3) * 5 + 1
    
    # 创建耦合矩阵（示例：I1和I2强耦合）
    W = np.array([
        [1.0, 0.5, 0.2],  # I1 与 I1, I2, t1 的耦合
        [0.5, 1.0, 0.3],  # I2 与 I1, I2, t1 的耦合
        [0.2, 0.3, 1.0]   # t1 与 I1, I2, t1 的耦合
    ])
    
    # 创建复合核
    kernel = CompositeKernel(
        coupling_matrix=W,
        gamma=0.5
    )
    
    # 计算核矩阵
    K = kernel(X)
    
    print(f"\n输入 X:\n{X}")
    print(f"\n耦合矩阵 W:\n{W}")
    print(f"\n核矩阵 K:\n{K}")
    print(f"\nK的形状: {K.shape}")
    print(f"K的对角线（应接近常数）: {np.diag(K)}")
    
    # 测试gamma更新
    print("\n测试gamma更新...")
    print(f"原始gamma: {kernel.gamma}")
    kernel.update_gamma(1.5)
    print(f"更新后gamma: {kernel.gamma}")
    
    K_new = kernel(X)
    print(f"新核矩阵对角线: {np.diag(K_new)}")
    
    print("\n测试完成！")