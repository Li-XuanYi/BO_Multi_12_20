"""
物理加权采集函数模块（FrameWork.md §5 实现）

实现 EI × W_charge 采集函数（约束 C-4）：
- alpha(theta) = EI(theta) × W_charge(theta)
- W_charge 是三维高斯搜索权重（Eq.17）
- mu 动态漂移（Eqs.18-19）
- sigma 灵敏度引导（Eq.20）

约束 C-6 参数：
- N_cand = 15（LLM 生成候选数）
- N_select = 3（选择评估数）
- alpha_max = 0.7, alpha_min = 0.05
- t_decay_alpha = 60
- kappa = 0.20
"""

import numpy as np
from scipy.stats import norm
from typing import List, Dict, Optional, Callable, Tuple

# 从 config 导入
try:
    from config import get_algorithm_param, get_llm_param
except ImportError:
    def get_algorithm_param(module, param, default=None):
        return default
    def get_llm_param(module, param, default=None):
        return default


class PhysicsWeightedAcquisition:
    """
    物理加权采集函数（FrameWork.md §5）

    alpha(theta) = EI(theta) × W_charge(theta)

    其中 W_charge 是三维高斯搜索权重：
    W_charge(theta) = prod_i [ exp(-0.5 * ((theta_i - mu_i) / sigma_i)^2) / (sqrt(2*pi) * sigma_i) ]

    mu 动态漂移（Eqs.18-19）：
    mu_{t+1} = alpha_t * mu_t + (1 - alpha_t) * theta_best

    sigma 灵敏度引导（Eq.20）：
    sigma_i = kappa * max_j(|dPsi/dtheta_j|) / |dPsi/dtheta_i|
    """

    def __init__(
        self,
        alpha_max: float = 0.7,
        alpha_min: float = 0.05,
        t_decay_alpha: int = 60,
        kappa: float = 0.20,
        n_cand: int = 15,
        n_select: int = 3,
        verbose: bool = False
    ):
        """
        初始化物理加权采集函数

        参数：
            alpha_max: mu 漂移最大惯性（约束 C-6: 0.7）
            alpha_min: mu 漂移最小惯性（约束 C-6: 0.05）
            t_decay_alpha: mu 漂移衰减时间（约束 C-6: 60）
            kappa: sigma 引导系数（约束 C-6: 0.20）
            n_cand: LLM 生成候选数（约束 C-6: 15）
            n_select: 选择评估数（约束 C-6: 3）
            verbose: 详细输出
        """
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.t_decay_alpha = t_decay_alpha
        self.kappa = kappa
        self.n_cand = n_cand
        self.n_select = n_select
        self.verbose = verbose

        # 搜索中心 mu (3D)
        self.mu = None
        # 搜索宽度 sigma (3D)
        self.sigma = None
        # 当前 alpha_t
        self.alpha_t = alpha_max

    def compute_W_charge(
        self,
        theta: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> float:
        """
        计算三维高斯搜索权重 W_charge - FrameWork.md Eq.17

        W_charge(theta) = prod_i [ exp(-0.5 * ((theta_i - mu_i) / sigma_i)^2) / (sqrt(2*pi) * sigma_i) ]

        参数：
            theta: 候选点 (3,)
            mu: 搜索中心 (3,)
            sigma: 搜索宽度 (3,)

        返回：
            w_charge: 搜索权重（归一化高斯密度）
        """
        theta = np.asarray(theta, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        # 避免除零
        sigma = np.maximum(sigma, 1e-10)

        # 标准化距离
        z = (theta - mu) / sigma

        # 高斯密度（每个维度）
        # N(mu, sigma^2) 的密度 = exp(-0.5*z^2) / (sqrt(2*pi) * sigma)
        gaussian_density = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * sigma)

        # 乘积（三维）
        w_charge = float(np.prod(gaussian_density))

        return w_charge

    def compute_W_charge_log(
        self,
        theta: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> float:
        """
        计算 log(W_charge)（数值稳定性更好）

        log(W_charge) = sum_i [ -0.5 * z_i^2 - log(sqrt(2*pi) * sigma_i) ]

        参数：
            theta: 候选点 (3,)
            mu: 搜索中心 (3,)
            sigma: 搜索宽度 (3,)

        返回：
            log_w_charge: log 搜索权重
        """
        theta = np.asarray(theta, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        sigma = np.maximum(sigma, 1e-10)

        z = (theta - mu) / sigma

        # log 密度
        log_density = -0.5 * z**2 - np.log(np.sqrt(2 * np.pi) * sigma)

        # 求和
        return float(np.sum(log_density))

    def update_mu(
        self,
        theta_best: np.ndarray,
        iteration: int
    ):
        """
        搜索中心 mu 动态漂移 - FrameWork.md Eqs.18-19

        alpha_t = alpha_max * exp(-t / t_decay_alpha) + alpha_min
        mu_{t+1} = alpha_t * mu_t + (1 - alpha_t) * theta_best

        参数：
            theta_best: 当前最优解 (3,)
            iteration: 当前迭代轮次 t
        """
        # 计算 alpha_t（退火公式）
        self.alpha_t = self.alpha_max * np.exp(-iteration / self.t_decay_alpha) + self.alpha_min

        if self.mu is None:
            # 初始化为 theta_best
            self.mu = theta_best.copy()
        else:
            # 漂移更新
            self.mu = self.alpha_t * self.mu + (1 - self.alpha_t) * theta_best

        if self.verbose:
            print(f"    [mu 更新] alpha_t={self.alpha_t:.3f}, mu={self.mu}")

    def update_sigma(self, grad_psi_at_best: np.ndarray, eps_sigma: float = 1e-3):
        """
        搜索宽度 sigma 灵敏度引导 - FrameWork.md Eq.20

        sigma_i = kappa * max_j(|dPsi/dtheta_j|) / (|dPsi/dtheta_i| + eps_sigma)

        参数：
            grad_psi_at_best: Psi 在 theta_best 处的梯度 (3,)
            eps_sigma: 防止除零的小量
        """
        grad_magnitudes = np.abs(np.asarray(grad_psi_at_best, dtype=float))

        # max_j(|dPsi/dtheta_j|)
        c = self.kappa * np.max(grad_magnitudes)

        # sigma_i = c / (|grad_i| + eps)
        self.sigma = c / (grad_magnitudes + eps_sigma)

        if self.verbose:
            print(f"    [sigma 更新] |grad|={grad_magnitudes}, sigma={self.sigma}")

    def compute_EI(
        self,
        gp_mean: float,
        gp_std: float,
        f_min: float
    ) -> float:
        """
        计算期望改进 EI - FrameWork.md §5

        EI(theta) = (f_min - mu) * Phi(z) + sigma * phi(z)
        其中 z = (f_min - mu) / sigma

        参数：
            gp_mean: GP 均值预测
            gp_std: GP 标准差预测
            f_min: 当前最优目标值

        返回：
            ei: 期望改进值
        """
        if gp_std < 1e-10:
            gp_std = 1e-10

        # z = (f_min - mu) / sigma
        z = (f_min - gp_mean) / gp_std

        # EI 公式
        ei = (f_min - gp_mean) * norm.cdf(z) + gp_std * norm.pdf(z)

        return max(ei, 0.0)

    def compute_acquisition(
        self,
        theta: np.ndarray,
        gp_mean: float,
        gp_std: float,
        f_min: float
    ) -> float:
        """
        计算物理加权采集函数 alpha(theta) - FrameWork.md Eq.14

        alpha(theta) = EI(theta) × W_charge(theta)

        参数：
            theta: 候选点 (3,)
            gp_mean: GP 均值预测
            gp_std: GP 标准差预测
            f_min: 当前最优目标值

        返回：
            alpha: 采集值（越大越好）
        """
        theta = np.asarray(theta, dtype=float)

        # 确保 mu 和 sigma 已初始化
        if self.mu is None:
            self.mu = theta.copy()
        if self.sigma is None:
            self.sigma = np.ones(3) * 0.1

        # 计算 EI
        ei = self.compute_EI(gp_mean, gp_std, f_min)

        # 计算 W_charge（使用当前的 self.mu 和 self.sigma）
        w_charge = self.compute_W_charge(theta, self.mu, self.sigma)

        # 物理加权采集
        alpha = ei * w_charge

        return alpha

    def select_candidates(
        self,
        candidates: List[np.ndarray],
        acq_values: np.ndarray
    ) -> List[np.ndarray]:
        """
        选择 top N_select 个候选

        参数：
            candidates: 候选点列表 [(3,), ...]
            acq_values: 采集值数组 (n_cand,)

        返回：
            selected: 选中的候选点列表
        """
        n_select = min(self.n_select, len(candidates))
        top_indices = np.argsort(acq_values)[-n_select:][::-1]

        selected = [candidates[i] for i in top_indices]
        return selected

    def initialize_mu_sigma(
        self,
        database: List[Dict],
        grad_psi: Optional[np.ndarray] = None
    ):
        """
        初始化 mu 和 sigma（从数据库或梯度）

        参数：
            database: 评估历史
            grad_psi: Psi 梯度（可选）
        """
        valid_data = [r for r in database if r['valid']]

        if len(valid_data) == 0:
            # 无数据时使用参数空间中点
            self.mu = np.array([4.0, 0.4, 4.0])  # (I1, SOC1, I2) 中点
            self.sigma = np.array([1.0, 0.1, 1.0])
            return

        # mu: 当前最优点的中心
        times = np.array([r['time'] for r in valid_data])
        best_idx = np.argmin(times)
        best = valid_data[best_idx]['params']
        self.mu = np.array([best['I1'], best['SOC1'], best['I2']])

        # sigma: 默认值
        if grad_psi is not None:
            self.update_sigma(grad_psi)
        else:
            self.sigma = np.array([0.5, 0.05, 0.5])

    # ── main.py 兼容桥接方法 ─────────────────────────────────────

    def update_search_params(
        self,
        theta_best: np.ndarray,
        grad_psi: np.ndarray,
        iteration: int,
        f_min_improved: bool = True
    ):
        """
        统一更新 mu / sigma / stagnation — main.py 调用入口。

        等价于 update_mu + update_sigma + stagnation 自适应。
        """
        self.update_mu(np.asarray(theta_best, dtype=float), iteration)
        self.update_sigma(np.asarray(grad_psi, dtype=float))

        # 停滞时扩展 sigma（FrameWork § 5.3 Eq.22, rho=0.1）
        if not f_min_improved:
            if self.sigma is not None:
                self.sigma = self.sigma * (1.0 + 0.1)

    def score(
        self,
        theta: np.ndarray,
        gp_list: List,
        f_min: float,
        weights: np.ndarray
    ) -> float:
        """
        计算 alpha(θ) = EI(θ) × W_charge(θ) — main.py 调用入口。

        从 gp_list 的 3 个 GP 中提取预测，用 Tchebycheff 组合后计算 EI。

        参数：
            theta    : 候选点 (3,)
            gp_list  : [gp_time, gp_temp, gp_aging]
            f_min    : 当前最优 Tchebycheff 标量值
            weights  : Tchebycheff 权重 (3,)

        返回：
            alpha : 采集值（越大越好）
        """
        theta = np.asarray(theta, dtype=float)
        weights = np.asarray(weights, dtype=float)
        X = theta.reshape(1, -1)

        # 确保 mu/sigma 已初始化
        if self.mu is None:
            self.mu = theta.copy()
        if self.sigma is None:
            self.sigma = np.array([0.5, 0.05, 0.5])

        # 从 GP 列表获取各目标预测
        means = np.zeros(3)
        stds = np.zeros(3)
        for i, gp in enumerate(gp_list):
            if gp is None:
                continue
            try:
                mu_i, sigma_i = gp.predict(X, return_std=True)
                means[i] = float(mu_i[0])
                stds[i] = max(float(sigma_i[0]), 1e-10)
            except Exception:
                pass

        # 用 Tchebycheff 组合均值和方差（eta=0.05 对应 FrameWork Eq.1）
        eta = 0.05
        w = weights / (weights.sum() + 1e-12)
        scal_mean = float(np.max(w * means) + eta * np.dot(w, means))
        scal_std = float(np.max(w * stds) + eta * np.dot(w, stds))

        # EI × W_charge
        ei = self.compute_EI(scal_mean, scal_std, f_min)
        w_charge = self.compute_W_charge(theta, self.mu, self.sigma)

        return float(ei * w_charge)


# ============================================================
# 采集函数优化器（FrameWork.md §5 + 约束 C-4, C-6）
# ============================================================
class AcquisitionOptimizer:
    """
    采集函数优化器（物理加权 EI）

    约束 C-4: alpha(theta) = EI(theta) × W_charge(theta)
    约束 C-6: N_cand=15, N_select=3
    """

    def __init__(
        self,
        param_bounds: Dict,
        n_cand: int = 15,
        n_select: int = 3,
        verbose: bool = False
    ):
        """
        初始化采集优化器

        参数：
            param_bounds: 参数边界 {I1, SOC1, I2}
            n_cand: LLM 生成候选数（约束 C-6: 15）
            n_select: 选择评估数（约束 C-6: 3）
            verbose: 详细输出
        """
        self.param_bounds = param_bounds
        self.n_cand = n_cand
        self.n_select = n_select
        self.verbose = verbose

        self.param_keys = ['I1', 'SOC1', 'I2']
        self.bounds_list = [
            param_bounds['I1'],
            param_bounds['SOC1'],
            param_bounds['I2']
        ]

        # 物理加权采集函数
        self.pe_acq = PhysicsWeightedAcquisition(
            n_cand=n_cand,
            n_select=n_select,
            verbose=verbose
        )

    def optimize(
        self,
        gp_list: List,
        weights: np.ndarray,
        database: List[Dict],
        f_min: float,
        iteration: int = 0,
        llm_candidates: Optional[List[Dict]] = None,
        grad_psi: Optional[np.ndarray] = None
    ) -> Dict:
        """
        采集优化（FrameWork.md §5）

        流程：
        1. 更新 mu（Eqs.18-19）和 sigma（Eq.20）
        2. 生成/获取候选点（LLM 或随机）
        3. 计算 alpha = EI × W_charge
        4. 选择 top N_select

        参数：
            gp_list: GP 模型列表 [gp_time, gp_temp, gp_aging]
            weights: Tchebycheff 权重 (3,)
            database: 评估历史
            f_min: 当前最优标量目标
            iteration: 当前迭代轮次
            llm_candidates: LLM 生成的候选（可选）
            grad_psi: Psi 梯度（用于 sigma 更新）

        返回：
            best_params: {I1, SOC1, I2}
        """
        # ===== Step 1: 更新 mu 和 sigma =====
        # 从数据库中找到当前最优点
        valid_data = [r for r in database if r['valid']]
        if len(valid_data) > 0:
            scalars = []
            for r in valid_data:
                obj = np.array([r['time'], r['temp'], r['aging']])
                scalar = np.dot(obj, weights)  # 简单加权和
                scalars.append(scalar)
            best_idx = np.argmin(scalars)
            theta_best = np.array([
                valid_data[best_idx]['params']['I1'],
                valid_data[best_idx]['params']['SOC1'],
                valid_data[best_idx]['params']['I2']
            ])
            self.pe_acq.update_mu(theta_best, iteration)

            if grad_psi is not None:
                self.pe_acq.update_sigma(grad_psi)
            else:
                # 默认 sigma
                self.pe_acq.sigma = np.array([0.5, 0.05, 0.5])

        # ===== Step 2: 获取候选点 =====
        if llm_candidates is not None and len(llm_candidates) > 0:
            # 使用 LLM 候选
            candidates = []
            for c in llm_candidates[:self.n_cand]:
                candidates.append(np.array([c['I1'], c['SOC1'], c['I2']]))
        else:
            # 随机生成候选
            candidates = self._generate_random_candidates(self.n_cand)

        # ===== Step 3: 计算采集值 alpha = EI × W_charge =====
        acq_values = []
        for theta in candidates:
            # GP 预测（标量化均值）
            mu_list = []
            std_list = []
            for gp in gp_list:
                m, s = gp.predict(theta.reshape(1, -1), return_std=True)
                mu_list.append(m[0])
                std_list.append(s[0])

            # 加权标量化均值
            gp_mean = float(np.dot(np.array(mu_list), weights))
            gp_std = float(np.sqrt(np.dot(np.array(std_list)**2, weights)))

            # 计算 alpha
            alpha = self.pe_acq.compute_acquisition(theta, gp_mean, gp_std, f_min)
            acq_values.append(alpha)

        acq_values = np.array(acq_values)

        # ===== Step 4: 选择 top N_select =====
        selected = self.pe_acq.select_candidates(candidates, acq_values)

        if self.verbose:
            print(f"  [Acquisition] alpha 范围：[{np.min(acq_values):.4f}, {np.max(acq_values):.4f}]")
            print(f"  选中 {len(selected)} 个候选")

        # 返回最优（第一个）
        best = selected[0]
        return {
            'I1': float(best[0]),
            'SOC1': float(best[1]),
            'I2': float(best[2])
        }

    def _generate_random_candidates(self, n: int) -> List[np.ndarray]:
        """
        生成随机候选点

        参数：
            n: 数量

        返回：
            candidates: [(3,), ...]
        """
        candidates = []
        for _ in range(n):
            c = np.array([
                np.random.uniform(*self.param_bounds['I1']),
                np.random.uniform(*self.param_bounds['SOC1']),
                np.random.uniform(*self.param_bounds['I2'])
            ])
            candidates.append(c)
        return candidates


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 PhysicsWeightedAcquisition（FrameWork.md §5）")
    print("=" * 60)

    # ===== 测试 1: W_charge 计算 =====
    print("\n[测试 1] W_charge 计算（Eq.17）")

    pe = PhysicsWeightedAcquisition(verbose=True)
    theta = np.array([4.0, 0.4, 3.0])
    mu = np.array([4.0, 0.4, 3.0])
    sigma = np.array([0.5, 0.05, 0.5])

    w = pe.compute_W_charge(theta, mu, sigma)
    w_log = pe.compute_W_charge_log(theta, mu, sigma)

    print(f"  theta = {theta}")
    print(f"  mu = {mu}")
    print(f"  sigma = {sigma}")
    print(f"  W_charge = {w:.6f}")
    print(f"  log(W_charge) = {w_log:.4f}")

    # ===== 测试 2: mu 漂移（Eqs.18-19） =====
    print("\n[测试 2] mu 动态漂移（Eqs.18-19）")

    pe2 = PhysicsWeightedAcquisition(alpha_max=0.7, alpha_min=0.05, t_decay_alpha=60)

    theta_best = np.array([4.5, 0.5, 3.5])
    print(f"  theta_best = {theta_best}")
    print(f"  mu 演化:")

    for t in [0, 10, 30, 60, 100]:
        pe2.mu = np.array([4.0, 0.4, 3.0])  # 重置
        pe2.update_mu(theta_best, t)
        print(f"    t={t}: alpha_t={pe2.alpha_t:.4f}, mu={pe2.mu}")

    # ===== 测试 3: sigma 灵敏度引导（Eq.20） =====
    print("\n[测试 3] sigma 灵敏度引导（Eq.20）")

    pe3 = PhysicsWeightedAcquisition(kappa=0.20)

    # Psi 梯度：dPsi/dI1, dPsi/dSOC1, dPsi/dI2
    grad_psi = np.array([54.0, 180.0, 72.0])

    pe3.update_sigma(grad_psi)
    print(f"  grad_Psi = {grad_psi}")
    print(f"  sigma = {pe3.sigma}")

    # 验证：梯度大的维度 sigma 小
    print(f"  max(|grad|) = {np.max(np.abs(grad_psi))}")
    print(f"  kappa * max = {0.20 * np.max(np.abs(grad_psi))}")

    # ===== 测试 4: EI 计算 =====
    print("\n[测试 4] EI 计算")

    f_min = 1000.0
    for gp_mean, gp_std in [(900, 100), (1000, 100), (1100, 100), (1000, 50)]:
        ei = pe.compute_EI(gp_mean, gp_std, f_min)
        print(f"  mu={gp_mean}, sigma={gp_std} -> EI={ei:.4f}")

    # ===== 测试 5: alpha = EI × W_charge =====
    print("\n[测试 5] alpha = EI × W_charge（Eq.14）")

    theta = np.array([4.0, 0.4, 3.0])
    mu = np.array([4.0, 0.4, 3.0])
    sigma = np.array([0.5, 0.05, 0.5])
    gp_mean, gp_std = 900, 100
    f_min = 1000

    pe.mu, pe.sigma = mu, sigma
    alpha = pe.compute_acquisition(theta, gp_mean, gp_std, f_min)
    ei = pe.compute_EI(gp_mean, gp_std, f_min)
    w = pe.compute_W_charge(theta, mu, sigma)

    print(f"  EI = {ei:.4f}")
    print(f"  W_charge = {w:.6f}")
    print(f"  alpha = EI × W_charge = {alpha:.4f}")

    # ===== 测试 6: 候选选择 =====
    print("\n[测试 6] 候选选择（N_select=3）")

    pe4 = PhysicsWeightedAcquisition(n_cand=15, n_select=3)

    candidates = [np.random.rand(3) * 10 for _ in range(15)]
    acq_values = np.random.rand(15)

    selected = pe4.select_candidates(candidates, acq_values)
    print(f"  候选数：{len(candidates)}")
    print(f"  选中数：{len(selected)}")
    print(f"  Top 3 采集值：{sorted(acq_values, reverse=True)[:3]}")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
