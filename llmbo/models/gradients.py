"""
GP梯度计算模块
计算 ∂U/∂θ (GP预测均值对参数的梯度)
"""

import numpy as np
from typing import Dict, List
from sklearn.gaussian_process import GaussianProcessRegressor


class GPGradientComputer:
    """
    GP梯度计算器
    
    方法：数值梯度（有限差分）
    优点：简单、稳定、适用于任何GP核
    """
    
    def __init__(self, epsilon: float = 1e-4, verbose: bool = False):
        """
        初始化梯度计算器
        
        参数：
            epsilon: 有限差分步长
            verbose: 详细输出
        """
        self.epsilon = epsilon
        self.verbose = verbose
    
    def compute_gradient(
        self,
        gp: GaussianProcessRegressor,
        X: np.ndarray,
        param_names: List[str] = None
    ) -> np.ndarray:
        """
        计算GP预测均值对输入的梯度
        
        参数：
            gp: 训练好的GP模型
            X: 输入点 (1, n_params)
            param_names: 参数名称列表（用于调试）
        
        返回：
            gradient: (n_params,) 梯度向量
        """
        X = X.reshape(1, -1)
        n_params = X.shape[1]
        gradient = np.zeros(n_params)
        
        # 预测中心点
        mu_0 = gp.predict(X)[0]
        
        # 对每个参数计算偏导
        for i in range(n_params):
            # 前向差分
            X_plus = X.copy()
            X_plus[0, i] += self.epsilon
            
            mu_plus = gp.predict(X_plus)[0]
            
            # 计算梯度
            gradient[i] = (mu_plus - mu_0) / self.epsilon
        
        return gradient
    
    def compute_jacobian(
        self,
        gp_list: List[GaussianProcessRegressor],
        X: np.ndarray
    ) -> np.ndarray:
        """
        计算雅可比矩阵（所有目标对所有参数的梯度）
        
        J[i, j] = ∂f_i / ∂θ_j
        
        参数：
            gp_list: GP列表 [gp_time, gp_temp, gp_aging]
            X: 输入点 (1, n_params)
        
        返回：
            J: (n_objectives, n_params) 雅可比矩阵
        """
        n_objectives = len(gp_list)
        n_params = X.shape[1] if len(X.shape) > 1 else len(X)
        
        X = X.reshape(1, -1)
        J = np.zeros((n_objectives, n_params))
        
        for i, gp in enumerate(gp_list):
            J[i, :] = self.compute_gradient(gp, X)
        
        return J
    
    def estimate_coupling_matrix(
        self,
        gp_list: List[GaussianProcessRegressor],
        X_samples: np.ndarray,
        method: str = 'outer_product'
    ) -> np.ndarray:
        """
        从GP梯度估计参数耦合矩阵
        
        方法：计算梯度的平均外积
        W[i,j] ≈ E[∂f/∂θ_i · ∂f/∂θ_j]
        
        参数：
            gp_list: GP列表
            X_samples: 采样点 (n_samples, n_params)
            method: 'outer_product' 或 'correlation'
        
        返回：
            W: (n_params, n_params) 耦合矩阵
        """
        n_params = X_samples.shape[1]
        W = np.zeros((n_params, n_params))
        
        # 对每个采样点计算梯度
        for X in X_samples:
            X = X.reshape(1, -1)
            
            # 计算雅可比矩阵
            J = self.compute_jacobian(gp_list, X)
            
            # ========== P2修复：梯度加权求和 ==========
            # Temp是硬约束（>313K失败），应占主导地位
            weights = np.array([0.3, 0.5, 0.2])  # [Time, Temp, Aging]
            grad_total = np.average(J, axis=0, weights=weights)
            
            if self.verbose and X is X_samples[0]:  # 仅第一个样本打印
                print(f"    [梯度加权] Time:0.3, Temp:0.5, Aging:0.2")
            # ==========================================
            
            # 外积：grad ⊗ grad
            W += np.outer(grad_total, grad_total)
        
        # 平均化
        W /= len(X_samples)
        
        # 归一化到 [0, 1]
        W_max = np.max(np.abs(W))
        if W_max > 1e-10:
            W = W / W_max
        
        # 对称化
        W = (W + W.T) / 2
        
        return W


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
    
    print("测试 GPGradientComputer...")
    
    # 创建虚拟数据
    np.random.seed(42)
    X_train = np.random.rand(10, 3) * 5 + 1  # 10个样本，3个参数
    y_train = np.sum(X_train**2, axis=1)  # 简单的平方和函数
    
    # 训练GP
    kernel = C(1.0) * Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gp.fit(X_train, y_train)
    
    # 计算梯度
    computer = GPGradientComputer(epsilon=1e-4)
    X_test = np.array([[3.0, 15.0, 2.0]])
    gradient = computer.compute_gradient(gp, X_test)
    
    print(f"\n测试点: {X_test[0]}")
    print(f"梯度: {gradient}")
    print(f"理论梯度（2*X）: {2 * X_test[0]}")
    
    # 测试雅可比矩阵
    gp_list = [gp, gp, gp]  # 3个相同的GP（仅测试）
    J = computer.compute_jacobian(gp_list, X_test)
    print(f"\n雅可比矩阵:\n{J}")
    
    # 测试耦合矩阵
    X_samples = np.random.rand(5, 3) * 5 + 1
    W = computer.estimate_coupling_matrix(gp_list, X_samples)
    print(f"\n耦合矩阵:\n{W}")
    
    print("\n测试完成！")