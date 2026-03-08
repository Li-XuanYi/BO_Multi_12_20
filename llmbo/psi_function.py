"""
焦耳热代理函数模块 (FrameWork.md §2 Eq.6)

实现物理代理函数 Psi(theta) 及其Analytical gradient：
- Psi(theta) = R1_bar * I1^2 * t1 + R2_bar * I2^2 * t2
- 其中 t1 = (SOC1 - SOC0) * Q_NOM / I1
- 其中 t2 = (SOC_END - SOC1) * Q_NOM / I2

Analytical gradient (Eqs.8-10)：
- dPsi/dI1 = R1_bar * (SOC1 - SOC0) * Q_NOM
- dPsi/dSOC1 = Q_NOM * (I1 * R1_bar - I2 * R2_bar)
- dPsi/dI2 = R2_bar * (SOC_END - SOC1) * Q_NOM

约束 C-1 要求：
- PSI_R1 = PSI_R2 = 0.01 Ohm
- Q_NOM = 18000 C (5Ah × 3600)
- SOC0 = 0.1, SOC_END = 0.8
"""

import numpy as np
from typing import Union

# 从 config 导入常量
try:
    from config import SOC0, SOC_END, Q_NOM, PSI_R1, PSI_R2
except ImportError:
    # 后备默认值（约束 C-6）
    SOC0 = 0.1
    SOC_END = 0.8
    Q_NOM = 18000  # C = 5Ah × 3600
    PSI_R1 = 0.01  # Ohm
    PSI_R2 = 0.01  # Ohm


class PsiFunction:
    """
    焦耳热代理函数 Psi(theta) - FrameWork.md §2 Eq.6

    输入：theta = (I1, SOC1, I2) - 3D 决策空间
    输出：Psi(theta) - 标量焦耳热 [J]
    """

    # 物理常量
    SOC0 = SOC0        # 初始 SOC
    SOC_END = SOC_END  # 充电终止 SOC
    Q_NOM = Q_NOM      # 标称容量 [C]
    R1_BAR = PSI_R1    # 第一阶段等效电阻 [Ohm]
    R2_BAR = PSI_R2    # 第二阶段等效电阻 [Ohm]

    @staticmethod
    def evaluate(theta: Union[np.ndarray, list]) -> float:
        """
        计算焦耳热代理函数 Psi(theta)

        FrameWork.md §2 Eq.6:
        Psi(theta) = R1_bar * I1^2 * t1 + R2_bar * I2^2 * t2

        其中：
        t1 = (SOC1 - SOC0) * Q_NOM / I1
        t2 = (SOC_END - SOC1) * Q_NOM / I2

        参数：
            theta: 决策变量 (I1, SOC1, I2)
                - I1: 第一阶段电流 [A]
                - SOC1: 切换 SOC
                - I2: 第二阶段电流 [A]

        返回：
            psi: 焦耳热代理值 [J]
        """
        theta = np.asarray(theta, dtype=float)
        I1, SOC1, I2 = theta

        # 计算持续时间
        t1 = (SOC1 - PsiFunction.SOC0) * PsiFunction.Q_NOM / I1
        t2 = (PsiFunction.SOC_END - SOC1) * PsiFunction.Q_NOM / I2

        # 计算焦耳热
        psi = PsiFunction.R1_BAR * I1**2 * t1 + PsiFunction.R2_BAR * I2**2 * t2

        return float(psi)

    @staticmethod
    def gradient(theta: Union[np.ndarray, list]) -> np.ndarray:
        """
        计算Analytical gradient dPsi/dtheta - FrameWork.md Eqs.8-10

        推导：
        Psi = R1 * I1^2 * t1 + R2 * I2^2 * t2
            = R1 * I1^2 * (SOC1 - SOC0) * Q / I1 + R2 * I2^2 * (SOC_END - SOC1) * Q / I2
            = R1 * I1 * (SOC1 - SOC0) * Q + R2 * I2 * (SOC_END - SOC1) * Q

        dPsi/dI1 = R1 * (SOC1 - SOC0) * Q
        dPsi/dSOC1 = R1 * I1 * Q - R2 * I2 * Q = Q * (R1 * I1 - R2 * I2)
        dPsi/dI2 = R2 * (SOC_END - SOC1) * Q

        参数：
            theta: 决策变量 (I1, SOC1, I2)

        返回：
            grad: 梯度向量 (dPsi/dI1, dPsi/dSOC1, dPsi/dI2)
        """
        theta = np.asarray(theta, dtype=float)
        I1, SOC1, I2 = theta

        Q = PsiFunction.Q_NOM
        R1 = PsiFunction.R1_BAR
        R2 = PsiFunction.R2_BAR
        SOC0 = PsiFunction.SOC0
        SOC_END = PsiFunction.SOC_END

        # Analytical gradient (Eqs.8-10)
        dI1 = R1 * (SOC1 - SOC0) * Q
        dSOC1 = Q * (R1 * I1 - R2 * I2)
        dI2 = R2 * (SOC_END - SOC1) * Q

        return np.array([dI1, dSOC1, dI2])

    @staticmethod
    def hessian(theta: Union[np.ndarray, list]) -> np.ndarray:
        """
        计算 Hessian 矩阵 d^2Psi/dtheta^2

        由于 Psi 是关于 theta 的线性函数（在简化后）：
        Psi = R1 * I1 * (SOC1 - SOC0) * Q + R2 * I2 * (SOC_END - SOC1) * Q

        二阶导数：
        d^2Psi/dI1^2 = 0
        d^2Psi/dI1/dSOC1 = R1 * Q
        d^2Psi/dI1/dI2 = 0
        d^2Psi/dSOC1^2 = 0
        d^2Psi/dSOC1/dI2 = -R2 * Q
        d^2Psi/dI2^2 = 0

        参数：
            theta: 决策变量 (I1, SOC1, I2)

        返回：
            H: Hessian 矩阵 (3, 3)
        """
        Q = PsiFunction.Q_NOM
        R1 = PsiFunction.R1_BAR
        R2 = PsiFunction.R2_BAR

        H = np.array([
            [0,      R1 * Q,  0      ],
            [R1 * Q, 0,       -R2 * Q],
            [0,      -R2 * Q, 0      ]
        ])

        return H

    @staticmethod
    def compute_durations(theta: Union[np.ndarray, list]) -> tuple:
        """
        计算充电持续时间

        参数：
            theta: 决策变量 (I1, SOC1, I2)

        返回：
            (t1, t2): 持续时间 [s]
        """
        theta = np.asarray(theta, dtype=float)
        I1, SOC1, I2 = theta

        t1 = (SOC1 - PsiFunction.SOC0) * PsiFunction.Q_NOM / I1
        t2 = (PsiFunction.SOC_END - SOC1) * PsiFunction.Q_NOM / I2

        return t1, t2


# ============================================================
# 验证函数
# ============================================================
def verify_gradient_psi(theta_test: np.ndarray = None, delta: float = 1e-5, tol: float = 1e-4) -> float:
    """
    验证Analytical gradient与数值梯度的误差 < tol

    数值梯度（中心差分）：
    dPsi/dtheta_i ≈ (Psi(theta + delta*e_i) - Psi(theta - delta*e_i)) / (2 * delta)

    参数：
        theta_test: Test point (3,)
        delta: 差分项
        tol: tolerance（约束 C-8 要求 < 1e-4）

    返回：
        max_error: 最大绝对误差

    抛出：
        AssertionError: 如果误差 >= tol
    """
    if theta_test is None:
        theta_test = np.array([4.0, 0.4, 3.0])  # 默认Test point

    # Analytical gradient
    analytical = PsiFunction.gradient(theta_test)

    # 数值梯度（中心差分）
    numerical = np.zeros(3)
    for i in range(3):
        theta_plus = theta_test.copy()
        theta_minus = theta_test.copy()
        theta_plus[i] += delta
        theta_minus[i] -= delta
        numerical[i] = (PsiFunction.evaluate(theta_plus) - PsiFunction.evaluate(theta_minus)) / (2 * delta)

    # 计算最大绝对误差
    max_error = float(np.max(np.abs(analytical - numerical)))

    # 验证
    assert max_error < tol, f"Gradient verification失败：最大误差 {max_error:.6e} >= tolerance {tol}"

    return max_error


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 PsiFunction 模块")
    print("=" * 60)

    # Test 1: Basic evaluation
    print("\n[测试 1] Psi 函数评估")
    theta1 = np.array([4.0, 0.4, 3.0])  # I1=4A, SOC1=0.4, I2=3A
    psi1 = PsiFunction.evaluate(theta1)
    t1, t2 = PsiFunction.compute_durations(theta1)

    print(f"  theta = (I1=4A, SOC1=0.4, I2=3A)")
    print(f"  t1 = {t1:.1f}s = {t1/60:.1f}min")
    print(f"  t2 = {t2:.1f}s = {t2/60:.1f}min")
    print(f"  Psi = {psi1:.2f}J")

    # 测试 2: Gradient verification
    print("\n[测试 2] Gradient verification（Constraint C-8 error < 1e-4）")
    theta_test = np.array([4.0, 0.4, 3.0])
    analytical_grad = PsiFunction.gradient(theta_test)
    print(f"  Test point：theta = {theta_test}")
    print(f"  Analytical gradient：{analytical_grad}")

    error = verify_gradient_psi(theta_test)
    print(f"  OK Gradient OK: max error = {error:.6e}")

    # 测试 3: Boundary test
    print("\n[测试 3] Boundary test")
    theta_low = np.array([0.01, 0.1, 0.01])  # 下边界
    theta_high = np.array([7.99, 0.7, 7.99])  # 上边界

    psi_low = PsiFunction.evaluate(theta_low)
    psi_high = PsiFunction.evaluate(theta_high)

    print(f"  下边界 Psi = {psi_low:.2f}J")
    print(f"  上边界 Psi = {psi_high:.2f}J")

    # Test 4: Hessian
    print("\n[测试 4] Hessian 矩阵")
    H = PsiFunction.hessian(theta_test)
    print(f"  H = \n{H}")
    print(f"  H 的秩 = {np.linalg.matrix_rank(H)}")
    print(f"  H 的行列式 = {np.linalg.det(H):.2e}")

    # Test 5: Physical consistency
    print("\n[测试 5] Physical consistency check")
    print(f"  Low current -> Long time -> High Psi?")

    theta_low_I = np.array([1.0, 0.4, 1.0])
    theta_high_I = np.array([7.0, 0.4, 7.0])

    psi_low_I = PsiFunction.evaluate(theta_low_I)
    psi_high_I = PsiFunction.evaluate(theta_high_I)

    t_low, _ = PsiFunction.compute_durations(theta_low_I)
    t_high, _ = PsiFunction.compute_durations(theta_high_I)

    print(f"    低电流 (1A, 1A): t1={t_low:.0f}s, Psi={psi_low_I:.2f}J")
    print(f"    高电流 (7A, 7A): t1={t_high:.0f}s, Psi={psi_high_I:.2f}J")

    if psi_low_I > psi_high_I:
        print("    OK Low current with long time leads to higher Joule heat（Matches physical intuition）")
    else:
        print("    WARNING Note: High current has I^2 factor but time is much shorter")

    print("\n" + "=" * 60)
    print("All tests completed！")
    print("=" * 60)
