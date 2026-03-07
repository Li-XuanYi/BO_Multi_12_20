"""
复合核函数模块（第一轮重构）
实现物理耦合核 k_phys = Σ_{ij} W_ij · φ_i · φ_j
支持任意维度 + ARD 长度尺度 + PSD 保证
"""

import numpy as np
from sklearn.gaussian_process.kernels import (
    Kernel, StationaryKernelMixin, NormalizedKernelMixin,
    Matern, ConstantKernel as C, Hyperparameter
)

# 导入config
from config import get_algorithm_param


# ============================================================
# 工具函数：PSD 保证
# ============================================================
def ensure_psd(W_raw, eps=1e-6):
    """
    确保矩阵半正定（通过特征值修正）
    
    算法：
    1. 对称化：W_sym = (W_raw + W_raw.T) / 2
    2. 特征分解：eigenvalues, Q = eigh(W_sym)
    3. 特征值裁剪：eigenvalues_clipped = max(eigenvalues, eps)
    4. 重构：W_psd = Q @ diag(eigenvalues_clipped) @ Q.T
    5. 再次对称化（消除浮点误差）
    
    参数：
        W_raw: np.ndarray, 形状 (d, d)，任意方阵
        eps: float, 最小特征值下限
    
    返回：
        W_psd: np.ndarray, 形状 (d, d)，保证半正定
    """
    # 1. 对称化
    W_sym = (W_raw + W_raw.T) / 2
    
    # 2. 特征分解（eigh 用于对称矩阵，数值更稳定）
    eigenvalues, Q = np.linalg.eigh(W_sym)
    
    # 3. 特征值裁剪
    eigenvalues_clipped = np.maximum(eigenvalues, eps)
    
    # 4. 重构
    W_psd = Q @ np.diag(eigenvalues_clipped) @ Q.T
    
    # 5. 再次对称化（消除浮点误差）
    W_psd = (W_psd + W_psd.T) / 2
    
    return W_psd


# ============================================================
# 核函数 1：物理耦合核
# ============================================================
class CouplingKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """
    物理耦合核函数（重构版）
    
    公式：k_phys(θ, θ') = Σ_{i,j} W_ij · φ_i(θ, θ') · φ_j(θ, θ')
    
    其中：φ_j(θ, θ') = exp(-(θ_j - θ'_j)² / (2 * ℓ̃_j²))
    
    物理意义：
    - W_ij: 参数 i 和 j 的耦合强度
    - φ_j: 单维度的 RBF 核
    - ℓ̃_j: φ_j 内部的长度尺度（独立于 ARD 核的 ℓ_j）
    
    支持任意维度 d（从 coupling_matrix 推断）
    """
    
    def __init__(
        self,
        coupling_matrix: np.ndarray = None,
        phi_length_scales: np.ndarray = None,
        coupling_matrix_bounds: str = "fixed"
    ):
        """
        初始化耦合核
        
        参数：
            coupling_matrix: np.ndarray, 形状 (d, d), 默认 None（首次调用时初始化）
            phi_length_scales: np.ndarray, 形状 (d,), 默认 None（首次调用时初始化为全1）
            coupling_matrix_bounds: str, "fixed"（W 由 LLM 推断，不参与 sklearn 优化）
        """
        self.coupling_matrix = coupling_matrix
        self.phi_length_scales = phi_length_scales
        self.coupling_matrix_bounds = coupling_matrix_bounds
    
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        计算核矩阵
        
        参数：
            X: (n_X, d)
            Y: (n_Y, d) or None
            eval_gradient: bool
        
        返回：
            K: (n_X, n_Y)
            K_gradient: (n_X, n_Y, 0)（耦合核不参与超参数优化）
        """
        if Y is None:
            Y = X
        
        n_X, d = X.shape
        n_Y = Y.shape[0]
        
        # 延迟初始化（第一次调用时根据输入维度初始化）
        if self.coupling_matrix is None:
            self.coupling_matrix = np.eye(d)
        if self.phi_length_scales is None:
            self.phi_length_scales = np.ones(d)
        
        # 计算所有 φ_j 矩阵，堆成 (d, n_X, n_Y)
        phi_tensor = np.zeros((d, n_X, n_Y))
        for j in range(d):
            # diff_j: (n_X, n_Y)
            diff_j = X[:, j:j+1] - Y[:, j:j+1].T
            # φ_j = exp(-diff_j² / (2 * ℓ̃_j²))
            phi_tensor[j] = np.exp(-diff_j**2 / (2 * self.phi_length_scales[j]**2))
        
        # 使用 einsum 高效计算：K_phys = Σ_{ij} W_ij * phi_i * phi_j
        # 等价于双重 for 循环，但速度快 10-100 倍
        K_phys = np.einsum('ij,iab,jab->ab', self.coupling_matrix, phi_tensor, phi_tensor)
        
        if eval_gradient:
            # 耦合核不参与超参数优化（固定参数）
            return K_phys, np.empty((n_X, n_Y, 0))
        else:
            return K_phys
    
    def diag(self, X):
        """
        对角线元素（自相关）
        
        当 X=Y 时，每个 φ_j 的对角线都是 1（因为 diff=0）
        所以 diag = sum of all W_ij
        """
        n_X = X.shape[0]
        return np.full(n_X, np.sum(self.coupling_matrix))
    
    def is_stationary(self):
        """是否平稳核"""
        return True  # φ_j 是平稳核，W 是常数，所以整体平稳
    
    def get_params(self, deep=True):
        """
        sklearn 兼容性：返回参数字典
        
        注意：这些参数不参与优化，但 sklearn 的 clone() 需要它们
        """
        return {
            'coupling_matrix': self.coupling_matrix,
            'phi_length_scales': self.phi_length_scales,
            'coupling_matrix_bounds': self.coupling_matrix_bounds
        }
    
    def update_coupling_matrix(self, new_W):
        """
        更新耦合矩阵（自动 PSD 修正）
        
        参数：
            new_W: np.ndarray, 形状 (d, d)
        """
        eps = get_algorithm_param('composite_kernel', 'eps_psd', 1e-6)
        self.coupling_matrix = ensure_psd(new_W, eps=eps)
    
    def update_phi_length_scales(self, new_scales):
        """
        更新 φ_j 的长度尺度
        
        参数：
            new_scales: np.ndarray, 形状 (d,)
        """
        d = len(new_scales)
        assert d == self.coupling_matrix.shape[0], "Length scales dimension mismatch"
        
        # 确保所有值 > 0（clip 到最小值 1e-4）
        self.phi_length_scales = np.maximum(new_scales, 1e-4)
    
    def __repr__(self):
        if self.coupling_matrix is not None:
            d = self.coupling_matrix.shape[0]
            return f"CouplingKernel(dim={d})"
        else:
            return "CouplingKernel(uninitialized)"


# ============================================================
# 核函数 2：复合核
# ============================================================
class CompositeKernel(Kernel):
    """
    复合核函数（重构版）
    
    公式：k(θ, θ') = k_base(θ, θ') + γ · k_phys(θ, θ')
    
    组合：
    - k_base: ARD Matern 核（捕捉平滑性 + 各向异性）
    - k_phys: 物理耦合核（捕捉参数交互）
    - γ: 耦合强度（动态调整）
    
    支持：
    - 4D 决策空间（current1, time1, current2, v_switch）
    - LLM 驱动的长度尺度更新
    - PSD 保证
    """
    
    def __init__(
        self,
        base_kernel: Kernel = None,
        coupling_kernel: CouplingKernel = None,
        coupling_matrix: np.ndarray = None,
        gamma: float = None,
        gamma_bounds: tuple = None,
        n_dims: int = 4,
        ard_length_scales: np.ndarray = None
    ):
        """
        初始化复合核
        
        参数：
            base_kernel: ARD Matern 核（如果为 None 则自动构造）
            coupling_kernel: 耦合核（如果为 None 则从 coupling_matrix 构造）
            coupling_matrix: (d, d) 耦合矩阵（用于构造 coupling_kernel）
            gamma: 耦合强度
            gamma_bounds: gamma 的调整范围
            n_dims: 决策空间维度（默认 4）
            ard_length_scales: LLM 驱动的 ARD 长度尺度（保证 clone 后保留）
        """
        # 从 config 读取默认值
        if gamma is None:
            gamma = get_algorithm_param('composite_kernel', 'gamma_init', 0.5)
        if gamma_bounds is None:
            gamma_bounds = get_algorithm_param('composite_kernel', 'gamma_bounds', (0.1, 2.0))
        
        # 保存所有参数（sklearn clone 需要）
        self.gamma = gamma
        self.gamma_bounds = gamma_bounds
        self._n_dims = n_dims  # 内部属性，避免与 sklearn 冲突
        self.coupling_matrix = coupling_matrix  # ← 保存原始矩阵
        self.ard_length_scales = ard_length_scales  # ← clone 时保留
        
        # 构造基础核：ARD Matern
        if base_kernel is None:
            # 使用 LLM 驱动的长度尺度（如果有）
            init_ls = np.array(ard_length_scales) if ard_length_scales is not None \
                      else np.ones(self._n_dims)
            base_kernel = (
                C(1.0, constant_value_bounds=(1e-3, 1e3)) *
                Matern(
                    nu=2.5,
                    length_scale=init_ls,  # ← ARD 模式 + LLM 初始化
                    length_scale_bounds=(1e-2, 1e2)
                )
            )
        
        self.base_kernel = base_kernel
        
        # 构造耦合核
        if coupling_kernel is None:
            coupling_kernel = CouplingKernel(coupling_matrix=coupling_matrix)
        
        self.coupling_kernel = coupling_kernel
    
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        计算复合核矩阵
        
        K = K_base + γ * K_phys
        """
        if Y is None:
            Y = X
        
        if eval_gradient:
            # sklearn 梯度计算要求 Y 必须是 None
            # 基础核有梯度（ARD 超参数），耦合核无梯度（固定参数）
            K_base, K_base_grad = self.base_kernel(X, None, eval_gradient=True)
            K_phys = self.coupling_kernel(X, None, eval_gradient=False)
            K = K_base + self.gamma * K_phys
            return K, K_base_grad  # 梯度只来自 base_kernel
        else:
            K_base = self.base_kernel(X, Y, eval_gradient=False)
            K_phys = self.coupling_kernel(X, Y, eval_gradient=False)
            K = K_base + self.gamma * K_phys
            return K
    
    def diag(self, X):
        """对角线元素"""
        return self.base_kernel.diag(X) + self.gamma * self.coupling_kernel.diag(X)
    
    def is_stationary(self):
        """是否平稳"""
        return self.base_kernel.is_stationary()
    
    def get_params(self, deep=True):
        """
        sklearn 兼容性：返回参数字典
        
        包含 ard_length_scales 以确保 clone() 后长度尺度保留
        """
        params = {
            'coupling_matrix': self.coupling_matrix,
            'gamma': self.gamma,
            'gamma_bounds': self.gamma_bounds,
            'n_dims': self._n_dims,
            'ard_length_scales': self.ard_length_scales
        }
        if deep:
            return params
        return params
    
    def set_params(self, **params):
        """
        sklearn 兼容性：设置参数
        """
        valid_params = ['coupling_matrix', 'gamma', 'gamma_bounds', 'n_dims', 'ard_length_scales']
        for key, value in params.items():
            if key in valid_params:
                if key == 'n_dims':
                    self._n_dims = value
                else:
                    setattr(self, key, value)
        return self
    
    def set_ard_length_scales(self, length_scales):
        """
        设置 ARD 长度尺度（LLM 驱动）
        
        同时保存到 self.ard_length_scales，确保 clone() 后保留。
        
        参数：
            length_scales: np.ndarray, 形状 (d,)，ℓ_t = [ℓ_{t,1}, ..., ℓ_{t,d}]
        """
        length_scales = np.asarray(length_scales, dtype=float)
        assert len(length_scales) == self._n_dims, "Length scales dimension mismatch"
        assert np.all(length_scales > 0), "All length scales must be > 0"
        
        # 保存到实例属性（clone 时通过 get_params 传递）
        self.ard_length_scales = length_scales.copy()
        
        # 访问 base_kernel 内部的 Matern 核
        # 结构：ConstantKernel * Matern，Matern 是 k2
        if hasattr(self.base_kernel, 'k2'):
            matern_kernel = self.base_kernel.k2
            matern_kernel.length_scale = length_scales
        else:
            # 回退：假设 base_kernel 本身就是 Matern
            self.base_kernel.length_scale = length_scales
    
    def update_gamma(self, new_gamma: float):
        """动态更新耦合强度"""
        self.gamma = np.clip(new_gamma, *self.gamma_bounds)
    
    def update_coupling_matrix(self, new_matrix: np.ndarray):
        """
        更新耦合矩阵（自动 PSD 修正）
        
        参数：
            new_matrix: np.ndarray, 形状 (d, d)
        """
        # 调用耦合核的 update 方法（内部会调用 ensure_psd）
        self.coupling_kernel.update_coupling_matrix(new_matrix)
        # 同步更新自身属性
        self.coupling_matrix = self.coupling_kernel.coupling_matrix
    
    def __repr__(self):
        return (f"CompositeKernel(\n"
                f"  n_dims={self._n_dims},\n"
                f"  gamma={self.gamma:.3f},\n"
                f"  base_kernel={self.base_kernel.__class__.__name__}\n"
                f")")


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 ensure_psd 函数")
    print("=" * 60)
    
    # 测试 1: 不正定矩阵
    W_bad = np.array([
        [1, 2, 0, 0],
        [2, 1, 0, 0],
        [0, 0, 1, 0.5],
        [0, 0, 0.5, 1]
    ])
    
    print("\n原始矩阵（不正定）:")
    print(W_bad)
    print(f"特征值: {np.linalg.eigvalsh(W_bad)}")
    
    W_psd = ensure_psd(W_bad)
    print("\n修正后矩阵:")
    print(W_psd)
    print(f"特征值: {np.linalg.eigvalsh(W_psd)}")
    print(f"是否对称: {np.allclose(W_psd, W_psd.T)}")
    
    # 测试 2: 本身正定的矩阵应几乎不变
    W_good = np.eye(4)
    W_good_psd = ensure_psd(W_good)
    print(f"\n单位矩阵修正误差: {np.max(np.abs(W_good - W_good_psd)):.2e}")
    
    print("\n" + "=" * 60)
    print("测试 CouplingKernel（4D）")
    print("=" * 60)
    
    # 创建 4×4 PSD 耦合矩阵
    W = 0.8 * np.eye(4) + 0.2 * np.ones((4, 4))
    W = ensure_psd(W)
    
    phi_scales = np.ones(4)
    coupling_kernel = CouplingKernel(coupling_matrix=W, phi_length_scales=phi_scales)
    
    # 测试数据
    np.random.seed(42)
    X = np.random.rand(10, 4)
    
    K = coupling_kernel(X)
    
    print(f"核矩阵形状: {K.shape}")
    print(f"对称性误差: {np.max(np.abs(K - K.T)):.2e}")
    print(f"对角线元素（应为 {np.sum(W):.3f}）: {np.diag(K)[:3]}")
    print(f"最小特征值: {np.min(np.linalg.eigvalsh(K)):.2e}")
    
    print("\n" + "=" * 60)
    print("测试 CompositeKernel（4D + ARD）")
    print("=" * 60)
    
    composite_kernel = CompositeKernel(
        coupling_matrix=W,
        gamma=0.5,
        n_dims=4
    )
    
    K_composite = composite_kernel(X)
    
    print(f"复合核矩阵形状: {K_composite.shape}")
    print(f"对称性误差: {np.max(np.abs(K_composite - K_composite.T)):.2e}")
    print(f"最小特征值: {np.min(np.linalg.eigvalsh(K_composite)):.2e}")
    
    # 测试 ARD 长度尺度设置
    print("\n测试 ARD 长度尺度更新:")
    new_scales = np.array([0.5, 1.0, 1.5, 2.0])
    composite_kernel.set_ard_length_scales(new_scales)
    K_new = composite_kernel(X)
    print(f"更新后核矩阵形状: {K_new.shape}")
    print(f"核矩阵是否变化: {not np.allclose(K_composite, K_new)}")
    
    # 测试 gamma 更新
    print("\n测试 gamma 更新:")
    composite_kernel.update_gamma(1.5)
    K_gamma = composite_kernel(X)
    print(f"gamma 更新后核矩阵形状: {K_gamma.shape}")
    print(f"核矩阵是否变化: {not np.allclose(K_new, K_gamma)}")
    
    # 测试 sklearn 兼容性
    print("\n测试 sklearn clone 兼容性:")
    from sklearn.base import clone
    try:
        kernel_clone = clone(composite_kernel)
        K_clone = kernel_clone(X)
        print(f"✓ clone 成功，核矩阵形状: {K_clone.shape}")
    except Exception as e:
        print(f"✗ clone 失败: {e}")
    
    print("\n" + "=" * 60)
    print("kernels.py 测试完成！")
    print("=" * 60)
