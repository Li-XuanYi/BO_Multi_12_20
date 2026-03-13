"""
components/gp_model.py
======================
Physics-Informed Composite GP Kernel — LLMBO-MO Framework §2 & §3

公式对应：
  §2  (5)–(7)  : Ψ(θ) = I₁²·R̄₁·t₁ + I₂²·R̄₂·t₂   (ohmic-heat proxy)
  §2  (8)–(10) : ∇Ψ(θ) 解析式（解析 + 中心差分 δ=1e-5 备用）
  §3  (11)     : k^(t)(θ,θ') = RBF + γ·∇Ψ(θ)ᵀ W^(t) ∇Ψ(θ')
  §3  (11')    : W^(t) = Σ wᵢ Wᵢ / Σ wᵢ
  §3  (12)     : f̂(θ) = μ̂ + cᵀ C⁻¹ (F^tch − 1μ̂)
  §3  (13)     : s²(θ) = σ̂²(1 − cᵀ C⁻¹c + (1−1ᵀC⁻¹c)²/(1ᵀC⁻¹1))

模块职责（单一原则）：
  本模块 **只负责核函数和 GP 预测**，不计算 EI / W_charge / α。
  EI 和 Acquisition function 全部在 components/acquisition.py 实现。

对外接口（被其他模块调用）：
─────────────────────────────────────────────────────────────────────
  GPProtocol          ← typing.Protocol，acquisition.py 依赖此接口
                         PhysicsGPModel 自动满足，也可换成标准 GP 做消融

  CouplingMatrixManager.set_llm_matrices(W_time, W_temp, W_aging)
                       ← Touchpoint 1a：初始化时由 LLM 接口层调用一次

  CouplingMatrixManager.get_W(w_vec) → np.ndarray (3,3)
                       ← 每迭代由主循环调用，按 Tchebycheff 权重合成 W^(t)

  GammaAnnealer.gamma(t) → float
                       ← 每迭代调用，返回退火后的 γ

  PhysicsGPModel.fit(X, F_tch, w_vec, t) → self
                       ← 训练（含 MLE 学习 l，缓存 Cholesky 分解）

  PhysicsGPModel.predict(X_new) → (mean: ndarray, std: ndarray)
                       ← 供 acquisition.py 的 EICalculator 调用

  PhysicsGPModel.training_summary() → dict
                       ← 供日志、DatabaseSummarizer 显示超参数状态

  PsiFunction.gradient_raw(theta) → np.ndarray
                       ← 供 acquisition.py 的 SearchSigmaTracker 计算 σⱼ（Eq.20）
                          返回原始量纲（不归一化）

  PsiFunction.gradient(theta) → np.ndarray
                       ← 供核函数内部使用，返回归一化单位向量

  build_gp_stack(param_bounds, ...) → (psi_fn, coupling_mgr, gamma_ann, gp_model)
                       ← 工厂函数，main.py 初始化时调用
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# GPProtocol：acquisition.py / optimizer.py 依赖的最小 GP 接口
# ───────────────────────────────────────────────────────────────────────────

@runtime_checkable
class GPProtocol(Protocol):
    """
    GP 代理模型的最小接口约定。

    PhysicsGPModel 自动满足此协议；消融实验时可换为 sklearn GP 或其他
    标准 GP 库，只需实现这两个方法即可插入 AcquisitionFunction 框架。

    方法说明
    --------
    fit(X, F_tch, w_vec, t)
        训练 GP。X: (n,3)，F_tch: (n,) 标量化目标，w_vec: (3,) 权重，t: 迭代编号。

    predict(X_new) -> (mean, std)
        预测后验均值 (m,) 和标准差 (m,)，均为 np.ndarray。
        std >= 0 保证（AcquisitionFunction 内部仍做 clip 防御）。

    training_summary() -> dict
        返回当前 GP 超参数快照，供 DatabaseSummarizer / 日志使用。
        必须包含键: "n_train", "l", "gamma", "mu_hat", "sig2_hat"。
    """

    def fit(
        self,
        X:     np.ndarray,
        F_tch: np.ndarray,
        w_vec: np.ndarray,
        t:     int,
    ) -> "GPProtocol": ...

    def predict(
        self,
        X_new: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    def training_summary(self) -> Dict[str, Any]: ...

# ─── 物理常数（LG M50, Chen2020） ──────────────────────────────────────────
Q_NOM_DEFAULT: float = 18000.0   # 标称容量 [C]  (5 Ah × 3600)
SOC0_DEFAULT:  float = 0.1       # 初始 SOC
SOC_END_DEFAULT: float = 0.8     # 目标 SOC

# 标称内阻 [Ω]，常数近似；若有 SOC 相关模型可替换 _R_bar_stage
R_INT_DEFAULT: float = 0.015

# ───────────────────────────────────────────────────────────────────────────
# §2  Ψ(θ) 和 ∇Ψ(θ)
# ───────────────────────────────────────────────────────────────────────────

class PsiFunction:
    """
    物理代理势函数 Ψ(θ)：累计欧姆热（Eq.5–7）。

    Ψ(θ) = I₁² · R̄₁(SOC₁) · t₁ + I₂² · R̄₂(SOC₁) · t₂   (Eq.6)

    t₁ = (SOC₁ − SOC₀) · Q_nom / I₁                        (Eq.7a)
    t₂ = (SOC_end − SOC₁) · Q_nom / I₂                     (Eq.7b)

    化简（消去 I²/I = I）：
      Ψ(θ) = Q_nom · [I₁ · R̄₁ · (SOC₁ − SOC₀) + I₂ · R̄₂ · (SOC_end − SOC₁)]

    梯度（Eq.8–10）：
      ∂Ψ/∂I₁   = R̄₁ · (SOC₁ − SOC₀) · Q_nom
      ∂Ψ/∂SOC₁ = Q_nom · [I₁(R̄₁ + (SOC₁−SOC₀)·dR̄₁/dSOC₁)
                          − I₂(R̄₂ − (SOC_end−SOC₁)·dR̄₂/dSOC₁)]
      ∂Ψ/∂I₂   = R̄₂ · (SOC_end − SOC₁) · Q_nom

    若使用常数 R（dR̄/dSOC = 0），中间项简化为 Q_nom·(I₁−I₂)·R。
    """

    def __init__(
        self,
        Q_nom:   float = Q_NOM_DEFAULT,
        SOC0:    float = SOC0_DEFAULT,
        SOC_end: float = SOC_END_DEFAULT,
        R_int:   float = R_INT_DEFAULT,
        use_soc_dependent_R: bool = False,
        fd_delta: float = 1e-5,
        normalize_grad: bool = True,
    ):
        """
        Parameters
        ----------
        normalize_grad : bool, default True
            若为 True，gradient() 返回单位向量 ĝ = ∇Ψ / ‖∇Ψ‖。
            使物理耦合项 γ·ĝᵀWĝ ∈ [0, γ·λ_max(W)]（约 O(γ)=O(0.3)），
            与 RBF 项 ∈ [0,1] 量级匹配；避免原始梯度（~675 J/A）淹没 RBF。
            原始梯度通过 gradient_raw() 获取，用于 acquisition σⱼ 计算（Eq.20）。
        """
        self.Q_nom   = Q_nom
        self.SOC0    = SOC0
        self.SOC_end = SOC_end
        self.R_int   = R_int
        self.use_soc_dependent_R = use_soc_dependent_R
        self.fd_delta = fd_delta
        self.normalize_grad = normalize_grad

    # ── 阶段平均内阻 R̄(SOC_avg) ─────────────────────────────────────────
    def _R_bar(self, soc_avg: float) -> float:
        """
        内阻模型：常数 or 线性 SOC 依赖。

        常数模式（默认）：R̄ = R_int
        SOC 相关模式：R̄(SOC) = R_int · (1 + 0.3·SOC)
          — 充电末段 SOC 升高时内阻微增，模拟真实电池行为。
        """
        if not self.use_soc_dependent_R:
            return self.R_int
        return self.R_int * (1.0 + 0.3 * float(np.clip(soc_avg, 0.0, 1.0)))

    def _dR_bar_dsoc(self, soc_avg: float) -> float:
        """dR̄/dSOC（用于 Eq.9 精确解析梯度）。"""
        if not self.use_soc_dependent_R:
            return 0.0
        return self.R_int * 0.3

    # ── 势函数 Ψ(θ) — Eq.6 ───────────────────────────────────────────────
    def evaluate(self, theta: np.ndarray) -> float:
        """
        Ψ(θ)，θ = [I1, SOC1, I2]。

        Returns
        -------
        float — 量纲：[Ω·C²/A] = [Ω·A·s] = [J]（物理上是焦耳热代理）
        """
        I1, SOC1, I2 = self._unpack(theta)
        soc1_avg = (self.SOC0 + SOC1) / 2.0
        soc2_avg = (SOC1 + self.SOC_end) / 2.0
        R1 = self._R_bar(soc1_avg)
        R2 = self._R_bar(soc2_avg)
        psi = self.Q_nom * (
            I1 * R1 * (SOC1 - self.SOC0) +
            I2 * R2 * (self.SOC_end - SOC1)
        )
        return float(psi)

    # ── 梯度 ∇Ψ(θ) — Eq.8-10 ────────────────────────────────────────────
    def gradient_raw(self, theta: np.ndarray) -> np.ndarray:
        """
        原始 ∇Ψ(θ) ∈ ℝ³（量纲：J/A 或 J/unit）。
        供 acquisition.py 计算 σⱼ（Eq.20-21）使用，反映真实敏感度比例。
        """
        I1, SOC1, I2 = self._unpack(theta)
        soc1_avg = (self.SOC0 + SOC1) / 2.0
        soc2_avg = (SOC1 + self.SOC_end) / 2.0
        R1  = self._R_bar(soc1_avg)
        R2  = self._R_bar(soc2_avg)
        dR1 = self._dR_bar_dsoc(soc1_avg) * 0.5
        dR2 = self._dR_bar_dsoc(soc2_avg) * 0.5

        dPsi_dI1 = self.Q_nom * R1 * (SOC1 - self.SOC0)
        dPsi_dSOC1 = self.Q_nom * (
            I1 * (dR1 * (SOC1 - self.SOC0) + R1) -
            I2 * (R2 - dR2 * (self.SOC_end - SOC1))
        )
        dPsi_dI2 = self.Q_nom * R2 * (self.SOC_end - SOC1)
        return np.array([dPsi_dI1, dPsi_dSOC1, dPsi_dI2], dtype=float)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        ∇Ψ(θ)，供 GP 核函数使用。

        若 normalize_grad=True（默认），返回单位向量 ĝ = ∇Ψ / ‖∇Ψ‖，
        使耦合项 γ·ĝᵀWĝ 与 RBF 项量级匹配（均为 O(1)）。

        若 normalize_grad=False，返回原始梯度（量纲 J/A 级别）。
        """
        g = self.gradient_raw(theta)
        if self.normalize_grad:
            norm = np.linalg.norm(g)
            if norm > 1e-12:
                return g / norm
            return g
        return g

    def gradient_fd(self, theta: np.ndarray) -> np.ndarray:
        """中心差分备用（δ=1e-5，§2.2），返回原始量纲（不归一化）。"""
        grad = np.zeros(3)
        for i in range(3):
            t_p = theta.copy(); t_p[i] += self.fd_delta
            t_m = theta.copy(); t_m[i] -= self.fd_delta
            grad[i] = (self.evaluate(t_p) - self.evaluate(t_m)) / (2.0 * self.fd_delta)
        return grad

    @staticmethod
    def _unpack(theta: np.ndarray) -> Tuple[float, float, float]:
        arr = np.asarray(theta, dtype=float).ravel()
        if arr.size != 3:
            raise ValueError(f"theta 期望 3 维，收到 {arr.size} 维")
        return float(arr[0]), float(arr[1]), float(arr[2])


# ───────────────────────────────────────────────────────────────────────────
# §3.1  耦合矩阵管理（Touchpoint 1a 接口 + 式 11'）
# ───────────────────────────────────────────────────────────────────────────

class CouplingMatrixManager:
    """
    管理 LLM 提供的三个目标耦合矩阵，并在每次迭代动态合成 W^(t)。

    对外接口
    --------
    set_llm_matrices(W_time, W_temp, W_aging)
        由 Touchpoint 1a 调用：接收 LLM 输出的 3×3 矩阵。
        自动投影为 PSD（若不满足）。

    get_W(w_vec) → np.ndarray  (3×3)
        由主循环在每次迭代调用：按 Tchebycheff 权重合成 W^(t)。
    """

    _EIG_FLOOR = 1e-8   # 特征值裁剪下界（防止数值负值）

    def __init__(self):
        # 默认退回单位矩阵（LLM 未提供时使用）
        self._W_time  = np.eye(3)
        self._W_temp  = np.eye(3)
        self._W_aging = np.eye(3)
        self._llm_provided = False

    # ── Touchpoint 1a 入口 ────────────────────────────────────────────────
    def set_llm_matrices(
        self,
        W_time:  np.ndarray,
        W_temp:  np.ndarray,
        W_aging: np.ndarray,
    ) -> None:
        """
        接收并验证 LLM 生成的三个耦合矩阵（每次初始化调用一次）。

        Parameters
        ----------
        W_time, W_temp, W_aging : array-like (3, 3)
            LLM 输出的目标耦合矩阵；若非 PSD 则自动投影。
        """
        self._W_time  = self._validate_and_project(np.array(W_time,  dtype=float), "W_time")
        self._W_temp  = self._validate_and_project(np.array(W_temp,  dtype=float), "W_temp")
        self._W_aging = self._validate_and_project(np.array(W_aging, dtype=float), "W_aging")

        # FIX: trace 归一化 + 与物理默认混合（防 LLM 先验偏差过大）
        blend = 0.7   # LLM 权重 0.7，物理默认 0.3
        W_phys_time  = np.diag([0.8, 0.2, 0.7])   # 物理默认：I1/I2 主导充电时间
        W_phys_temp  = np.diag([1.0, 0.3, 0.4])   # 物理默认：I1 主导峰值温度
        W_phys_aging = np.diag([0.3, 0.6, 0.9])   # 物理默认：I2@高 SOC 主导老化
        for attr, W_phys in [
            ("_W_time",  W_phys_time),
            ("_W_temp",  W_phys_temp),
            ("_W_aging", W_phys_aging),
        ]:
            W_llm = getattr(self, attr)
            # trace 归一化到 trace=3（与 3×3 单位矩阵基准一致）
            tr = np.trace(W_llm)
            if tr > 1e-8:
                W_llm = W_llm * (3.0 / tr)
            # 与物理默认混合
            W_blended = blend * W_llm + (1.0 - blend) * W_phys
            setattr(self, attr, self._validate_and_project(W_blended, attr))
        logger.info(
            "CouplingMatrixManager: LLM 矩阵已 trace 归一化 + %.0f%%/%.0f%% 混合",
            blend * 100, (1 - blend) * 100,
        )

        self._llm_provided = True
        logger.info("CouplingMatrixManager: LLM 耦合矩阵已设置并验证")

    def is_llm_provided(self) -> bool:
        return self._llm_provided

    # ── 每迭代调用：合成 W^(t)  — Eq.11' ────────────────────────────────
    def get_W(self, w_vec: np.ndarray) -> np.ndarray:
        """
        W^(t) = (w₁·W_time + w₂·W_temp + w₃·W_aging) / (w₁+w₂+w₃)  (Eq.11')

        Parameters
        ----------
        w_vec : (3,) Tchebycheff 权重向量（非负）

        Returns
        -------
        W : (3, 3) PSD 矩阵
        """
        w = np.asarray(w_vec, dtype=float).ravel()
        if w.size != 3:
            raise ValueError(f"w_vec 期望 3 维，收到 {w.size} 维")
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total < 1e-12:
            return np.eye(3)   # 安全退回
        alpha = w / total
        W = alpha[0] * self._W_time + alpha[1] * self._W_temp + alpha[2] * self._W_aging
        # 因 PSD 的凸组合仍为 PSD（定理证明），理论上无需再投影
        # 但为对抗浮点误差，做一次轻量 clip
        return self._clip_psd(W)

    # ── PSD 投影：特征值分解 + 正特征值截取 — §3.2 ──────────────────────
    def _validate_and_project(self, W: np.ndarray, name: str) -> np.ndarray:
        if W.shape != (3, 3):
            raise ValueError(f"{name} 必须是 (3,3) 矩阵，收到 {W.shape}")
        W_sym = (W + W.T) / 2.0   # 强制对称
        if self._is_psd(W_sym):
            return W_sym
        logger.warning(
            "%s 不是 PSD（最小特征值 %.4g），执行特征值投影", name, np.linalg.eigvalsh(W_sym).min()
        )
        return self._project_psd(W_sym)

    @classmethod
    def _is_psd(cls, W: np.ndarray, tol: float = 0.0) -> bool:
        return bool(np.linalg.eigvalsh(W).min() >= tol)

    @classmethod
    def _project_psd(cls, W: np.ndarray) -> np.ndarray:
        """W_PSD = Σ_{λᵢ>0} λᵢ vᵢ vᵢᵀ  （§3.2 特征值投影）"""
        evals, evecs = np.linalg.eigh(W)
        evals_clipped = np.maximum(evals, cls._EIG_FLOOR)
        return (evecs * evals_clipped) @ evecs.T

    @classmethod
    def _clip_psd(cls, W: np.ndarray) -> np.ndarray:
        """轻量版：仅裁剪负特征值（用于浮点保护）。"""
        evals, evecs = np.linalg.eigh(W)
        if evals.min() >= 0.0:
            return W
        evals = np.maximum(evals, 0.0)
        return (evecs * evals) @ evecs.T


# ───────────────────────────────────────────────────────────────────────────
# §3.1  γ 退火调度（前期高、后期趋 γ_min）
# ───────────────────────────────────────────────────────────────────────────

class GammaAnnealer:
    """
    耦合强度 γ 的迭代退火。

    γ(t) = (γ_max − γ_min) · exp(−t / t_decay) + γ_min

    设计逻辑
    --------
    - 早期（t≈0）：γ≈γ_max，物理梯度项主导，LLM 提供的 W 矩阵影响大
    - 晚期（t→∞）：γ→γ_min，数据积累后 RBF 主导，减少先验偏差

    参数默认值参照 §10 表：γ₀=0.1 作为 γ_min 基线；γ_max 设为 3×γ₀=0.3。
    """

    def __init__(
        self,
        gamma_max:  float = 0.3,
        gamma_min:  float = 0.05,
        t_decay:    float = 20.0,
    ):
        self.gamma_max = float(gamma_max)
        self.gamma_min = float(gamma_min)
        self.t_decay   = float(t_decay)

    def gamma(self, t: int) -> float:
        """返回第 t 次迭代的 γ 值（t 从 0 开始）。"""
        return (self.gamma_max - self.gamma_min) * np.exp(-t / self.t_decay) + self.gamma_min

    def __repr__(self) -> str:
        return (f"GammaAnnealer(γ_max={self.gamma_max}, γ_min={self.gamma_min}, "
                f"t_decay={self.t_decay})")


# ───────────────────────────────────────────────────────────────────────────
# §3.1  复合核函数  — Eq.11
# ───────────────────────────────────────────────────────────────────────────

class PhysicsCompositeKernel:
    """
    k^(t)(θ,θ') = RBF(θ̃,θ̃') + γ · ∇Ψ(θ)ᵀ W^(t) ∇Ψ(θ')   (Eq.11)

    RBF(θ̃,θ̃') = exp(−‖θ̃−θ̃'‖²/(2l²))
    θ̃  = min-max 归一化到 [0,1]（按 param_bounds）

    Parameters
    ----------
    psi_fn    : PsiFunction
    param_bounds : dict  {"I1": (lo, hi), "SOC1": (lo, hi), "I2": (lo, hi)}
    l         : float, RBF 长度尺度（由 MLE 学习）
    gamma     : float, 当前迭代的耦合强度（由 GammaAnnealer 提供）
    W         : np.ndarray (3,3), 当前 W^(t)（由 CouplingMatrixManager.get_W 提供）
    """

    def __init__(
        self,
        psi_fn: PsiFunction,
        param_bounds: Dict[str, Tuple[float, float]],
    ):
        self.psi_fn = psi_fn
        self._bounds = param_bounds
        self._lo = np.array([param_bounds["I1"][0],
                              param_bounds["SOC1"][0],
                              param_bounds["I2"][0]], dtype=float)
        self._hi = np.array([param_bounds["I1"][1],
                              param_bounds["SOC1"][1],
                              param_bounds["I2"][1]], dtype=float)
        self._range = np.where(self._hi - self._lo > 1e-12,
                                self._hi - self._lo, 1.0)

    # ── 归一化 ────────────────────────────────────────────────────────────
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """θ̃ = (θ − lo) / range，结果 ∈ [0,1]"""
        return (np.atleast_2d(X) - self._lo) / self._range

    # ── 单对核值 ──────────────────────────────────────────────────────────
    def k(
        self,
        theta:  np.ndarray,
        theta_p: np.ndarray,
        l:     float,
        gamma: float,
        W:     np.ndarray,
    ) -> float:
        """标量核值 k(θ, θ')。"""
        # RBF 项
        t_n  = self.normalize(theta.reshape(1, -1))[0]
        tp_n = self.normalize(theta_p.reshape(1, -1))[0]
        rbf = np.exp(-np.sum((t_n - tp_n) ** 2) / (2.0 * l ** 2))

        # 物理耦合项
        gp  = self.psi_fn.gradient(theta)
        gpp = self.psi_fn.gradient(theta_p)
        coupling = float(gp @ W @ gpp)

        return float(rbf + gamma * coupling)

    # ── 核矩阵（向量化） ──────────────────────────────────────────────────
    def kernel_matrix(
        self,
        X:  np.ndarray,   # (n, 3)
        Xp: np.ndarray,   # (m, 3)
        l:     float,
        gamma: float,
        W:     np.ndarray,
    ) -> np.ndarray:
        """
        K ∈ ℝ^{n×m}，其中 K_ij = k(X[i], Xp[j])。

        向量化 RBF + 逐行物理梯度（梯度计算通常不是瓶颈）。
        """
        X  = np.atleast_2d(X)
        Xp = np.atleast_2d(Xp)
        n, m = X.shape[0], Xp.shape[0]

        # ── RBF（全向量化） ───────────────────────────────────────────────
        X_n  = self.normalize(X)    # (n,3)
        Xp_n = self.normalize(Xp)   # (m,3)
        # ‖xᵢ−xⱼ'‖² = ‖xᵢ‖²+‖xⱼ'‖²−2xᵢ·xⱼ'
        sq_n  = np.sum(X_n  ** 2, axis=1, keepdims=True)   # (n,1)
        sq_m  = np.sum(Xp_n ** 2, axis=1, keepdims=True).T # (1,m)
        D2    = sq_n + sq_m - 2.0 * (X_n @ Xp_n.T)
        D2    = np.maximum(D2, 0.0)   # 浮点保护
        K_rbf = np.exp(-D2 / (2.0 * l ** 2))               # (n,m)

        # ── 物理耦合（逐行计算梯度） ──────────────────────────────────────
        if gamma < 1e-12:
            return K_rbf

        G  = np.stack([self.psi_fn.gradient(X[i])  for i in range(n)])   # (n,3)
        Gp = np.stack([self.psi_fn.gradient(Xp[j]) for j in range(m)])   # (m,3)
        # G W Gp^T：(n,3)·(3,3)·(3,m) = (n,m)
        K_phys = (G @ W) @ Gp.T                                           # (n,m)

        return K_rbf + gamma * K_phys


# ───────────────────────────────────────────────────────────────────────────
# §3.3  GP 预测（Universal Kriging，Eq.12-13）
# ───────────────────────────────────────────────────────────────────────────

class PhysicsGPModel:
    """
    带物理复合核的高斯过程回归。

    预测公式（§3.3，Universal Kriging 常数均值）：
      μ̂      = (1ᵀ C⁻¹ 1)⁻¹ (1ᵀ C⁻¹ F)               估计全局均值
      σ̂²     = (F−1μ̂)ᵀ C⁻¹ (F−1μ̂) / n               估计方差
      f̂(θ)  = μ̂ + cᵀ C⁻¹ (F − 1μ̂)                  Eq.12
      s²(θ) = σ̂²[1 − cᵀC⁻¹c + (1−1ᵀC⁻¹c)²/(1ᵀC⁻¹1)] Eq.13

    对外接口
    --------
    fit(X, F_tch, w_vec, t)     — 训练（含 MLE 学习 l）
    predict(X_new) → (μ, σ)     — 预测均值和标准差
    predict_ei(X_new, f_min)    — 直接计算 EI（供 acquisition.py 调用）
    update_coupling(w_vec, t)   — 迭代更新 W^(t) 和 γ(t)
    """

    _NUGGET    = 1e-6    # 对角 jitter（数值稳定）
    _OBS_NOISE = 1e-4    # 观测噪声（防止 C 病态；可在 fit 中覆盖）
    _L_BOUND   = (0.1, 5.0)  # 长度尺度 MLE 搜索范围（避免 l→0 导致方差归零）

    def __init__(
        self,
        psi_fn:       PsiFunction,
        coupling_mgr: CouplingMatrixManager,
        gamma_ann:    GammaAnnealer,
        param_bounds: Dict[str, Tuple[float, float]],
    ):
        self.kernel = PhysicsCompositeKernel(psi_fn, param_bounds)
        self.coupling_mgr = coupling_mgr
        self.gamma_ann    = gamma_ann

        # 训练后填充
        self._X:    Optional[np.ndarray] = None   # (n, 3)
        self._F:    Optional[np.ndarray] = None   # (n,)  F^tch
        self._W:    Optional[np.ndarray] = None   # 当前 W^(t)
        self._l:    float = 1.0
        self._gamma: float = gamma_ann.gamma(0)
        self._mu_hat:   float = 0.0
        self._sig2_hat: float = 1.0

        # Cholesky 因子（训练后缓存）
        self._C_chol: Optional[tuple] = None      # cho_factor 结果
        self._C_inv_F_centered: Optional[np.ndarray] = None  # C⁻¹(F−1μ̂)
        self._C_inv_1: Optional[np.ndarray] = None           # C⁻¹ 1
        self._1T_Cinv_1: float = 1.0

    # ── 更新当前迭代的 W 和 γ ─────────────────────────────────────────────
    def update_coupling(self, w_vec: np.ndarray, t: int) -> None:
        """
        在每次迭代开始时调用，更新 W^(t) 和 γ(t)。
        w_vec 来自 Tchebycheff 权重采样器。
        """
        self._W     = self.coupling_mgr.get_W(w_vec)
        self._gamma = self.gamma_ann.gamma(t)

    # ── 训练：MLE 学习 l，闭合式求 μ̂、σ̂² ──────────────────────────────
    def fit(
        self,
        X:      np.ndarray,   # (n, 3)
        F_tch:  np.ndarray,   # (n,)  标量化目标（Eq.1 输出）
        w_vec:  np.ndarray,   # (3,)  当前 Tchebycheff 权重
        t:      int,          # 当前迭代编号
    ) -> "PhysicsGPModel":
        """
        训练 GP。

        步骤：
          1. update_coupling → W^(t), γ(t)
          2. 用 minimize_scalar 对 l 做 MLE（对数似然）
          3. 闭合式求 μ̂, σ̂², C⁻¹ 各向量（Cholesky 缓存）
        """
        self._X = np.atleast_2d(X).copy()
        self._F = np.asarray(F_tch, dtype=float).ravel().copy()
        n = self._X.shape[0]

        self.update_coupling(w_vec, t)

        # ── MLE for l ─────────────────────────────────────────────────────
        def neg_log_likelihood(log_l: float) -> float:
            l = np.exp(log_l)
            C = self._build_C(l)
            try:
                cf = cho_factor(C)
            except np.linalg.LinAlgError:
                return 1e10
            # μ̂ 闭合式
            Cinv_1 = cho_solve(cf, np.ones(n))
            Cinv_F = cho_solve(cf, self._F)
            mu  = np.dot(np.ones(n), Cinv_F) / np.dot(np.ones(n), Cinv_1)
            r   = self._F - mu
            Cinv_r = cho_solve(cf, r)
            sig2 = float(r @ Cinv_r) / n
            if sig2 < 1e-12:
                sig2 = 1e-12
            # 对数似然（去掉常数项）
            sign, logdet = np.linalg.slogdet(C)
            if sign <= 0:
                return 1e10
            nll = 0.5 * (n * np.log(sig2) + logdet)
            return float(nll)

        res = minimize_scalar(
            neg_log_likelihood,
            bounds=(np.log(self._L_BOUND[0]), np.log(self._L_BOUND[1])),
            method="bounded",
        )
        self._l = float(np.exp(res.x))

        # ── 缓存 Cholesky 及求解结果 ────────────────────────────────────
        C = self._build_C(self._l)
        self._C_chol = cho_factor(C)

        ones = np.ones(n)
        Cinv_1 = cho_solve(self._C_chol, ones)           # C⁻¹ 1
        Cinv_F = cho_solve(self._C_chol, self._F)        # C⁻¹ F

        self._1T_Cinv_1 = float(ones @ Cinv_1)
        self._mu_hat    = float(ones @ Cinv_F) / self._1T_Cinv_1

        r = self._F - self._mu_hat
        Cinv_r = cho_solve(self._C_chol, r)
        self._sig2_hat = float(r @ Cinv_r) / n
        if self._sig2_hat < 1e-12:
            self._sig2_hat = 1e-12

        self._C_inv_F_centered = Cinv_r    # C⁻¹(F−1μ̂)
        self._C_inv_1          = Cinv_1

        logger.debug("GP.fit: n=%d  l=%.4f  γ=%.4f  μ̂=%.4f  σ̂²=%.6f",
                     n, self._l, self._gamma, self._mu_hat, self._sig2_hat)
        return self

    # ── 预测均值和标准差 — Eq.12-13 ──────────────────────────────────────
    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        mean : (m,)  f̂(θ_new)，Eq.12
        std  : (m,)  s(θ_new) = √s²，Eq.13（非负）
        """
        self._check_fitted()
        X_new = np.atleast_2d(X_new)
        m = X_new.shape[0]

        # c_i = k(θ_new, θ^(i))，形状 (m, n)
        C_cross = self.kernel.kernel_matrix(
            X_new, self._X, self._l, self._gamma, self._W
        )   # (m, n)

        # k(θ_new, θ_new) 对角（自相关，n=m 时用向量）
        k_self = np.array([
            self.kernel.k(X_new[i], X_new[i], self._l, self._gamma, self._W)
            for i in range(m)
        ])   # (m,)

        # Eq.12: f̂(θ) = μ̂ + cᵀ C⁻¹ (F−1μ̂)
        mean = self._mu_hat + C_cross @ self._C_inv_F_centered   # (m,)

        # Eq.13: s²(θ) = σ̂²(1 − cᵀC⁻¹c + (1−1ᵀC⁻¹c)²/(1ᵀC⁻¹1))
        #   cᵀ C⁻¹ c：对每个预测点，c ∈ ℝⁿ，结果为标量
        Cinv_c = cho_solve(self._C_chol, C_cross.T)   # (n, m)
        cT_Cinv_c = np.einsum("mn,nm->m", C_cross, Cinv_c)   # (m,)

        #   1ᵀ C⁻¹ c = C_inv_1 · c^T
        oneT_Cinv_c = self._C_inv_1 @ C_cross.T   # (m,)

        bracket = (
            1.0 - cT_Cinv_c +
            (1.0 - oneT_Cinv_c) ** 2 / self._1T_Cinv_1
        )
        var = self._sig2_hat * np.maximum(bracket, 0.0)
        std = np.sqrt(var)

        return mean, std

    # ── 超参数快照（供 DatabaseSummarizer / 日志）─────────────────────────
    def training_summary(self) -> Dict[str, Any]:
        """
        返回当前训练状态的字典快照。

        供 DatabaseSummarizer 在构造 <sensitivity> 块时使用，
        也方便主循环打印调试信息。

        Keys
        ----
        n_train     : int    训练样本数
        l           : float  MLE 长度尺度
        gamma       : float  当前迭代耦合强度 γ
        mu_hat      : float  GP 常数均值估计 μ̂
        sig2_hat    : float  GP 方差估计 σ̂²
        llm_W_active: bool   LLM 耦合矩阵是否已提供
        fitted      : bool   GP 是否已训练
        """
        fitted = self._X is not None
        return {
            "n_train":      int(self._X.shape[0]) if fitted else 0,
            "l":            float(self._l),
            "gamma":        float(self._gamma),
            "mu_hat":       float(self._mu_hat),
            "sig2_hat":     float(self._sig2_hat),
            "llm_W_active": self.coupling_mgr.is_llm_provided(),
            "fitted":       fitted,
        }

    # ── 内部：构建核矩阵 + 噪声 ──────────────────────────────────────────
    def _build_C(self, l: float) -> np.ndarray:
        """训练核矩阵 C ∈ ℝ^{n×n}，加 obs_noise+nugget 保正定。"""
        n = self._X.shape[0]
        C = self.kernel.kernel_matrix(
            self._X, self._X, l, self._gamma, self._W
        )
        C += (self._OBS_NOISE + self._NUGGET) * np.eye(n)
        return C

    def _check_fitted(self) -> None:
        if self._X is None:
            raise RuntimeError("PhysicsGPModel 尚未训练，请先调用 fit()")


# ───────────────────────────────────────────────────────────────────────────
# 工厂函数：一步构建完整 GP 组件栈
# ───────────────────────────────────────────────────────────────────────────

def build_gp_stack(
    param_bounds:   Dict[str, Tuple[float, float]],
    Q_nom:          float = Q_NOM_DEFAULT,
    SOC0:           float = SOC0_DEFAULT,
    SOC_end:        float = SOC_END_DEFAULT,
    R_int:          float = R_INT_DEFAULT,
    use_soc_R:      bool  = False,
    gamma_max:      float = 0.3,
    gamma_min:      float = 0.05,
    gamma_t_decay:  float = 20.0,
) -> Tuple[PsiFunction, CouplingMatrixManager, GammaAnnealer, PhysicsGPModel]:
    """
    一步构建 (psi_fn, coupling_mgr, gamma_ann, gp_model)。

    典型用法（main.py / optimizer.py 初始化阶段）::

        psi, coupling, gamma_ann, gp = build_gp_stack(PARAM_BOUNDS)

        # Touchpoint 1a：LLM 接口层提供耦合矩阵
        coupling.set_llm_matrices(W_time, W_temp, W_aging)

        # 每次迭代（由 optimizer.py 调用）：
        gp.fit(X_train, F_tch, w_vec, t=iter_idx)
        mean, std = gp.predict(X_candidates)  # 供 AcquisitionFunction 调用

        # acquisition.py 需要原始梯度（不归一化）：
        grad_psi = psi.gradient_raw(theta_best)  # → SearchSigmaTracker

        # 日志 / DatabaseSummarizer：
        info = gp.training_summary()

    Returns
    -------
    psi_fn, coupling_mgr, gamma_ann, gp_model
    """
    psi_fn       = PsiFunction(Q_nom, SOC0, SOC_end, R_int, use_soc_R)
    coupling_mgr = CouplingMatrixManager()
    gamma_ann    = GammaAnnealer(gamma_max, gamma_min, gamma_t_decay)
    gp_model     = PhysicsGPModel(psi_fn, coupling_mgr, gamma_ann, param_bounds)
    return psi_fn, coupling_mgr, gamma_ann, gp_model


# ───────────────────────────────────────────────────────────────────────────
# 快速自测（python components/gp_model.py）
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                        format="%(levelname)s %(name)s: %(message)s")

    BOUNDS = {"I1": (3.0, 7.0), "SOC1": (0.1, 0.7), "I2": (1.0, 5.0)}

    print("=" * 60)
    print("1. PsiFunction 自测")
    print("=" * 60)
    psi = PsiFunction(normalize_grad=True)
    theta0 = np.array([5.0, 0.4, 2.5])
    print(f"  Ψ(θ₀)         = {psi.evaluate(theta0):.4f} J(代理)")
    g_raw  = psi.gradient_raw(theta0)
    g_norm = psi.gradient(theta0)
    g_fd   = psi.gradient_fd(theta0)
    print(f"  ∇Ψ 原始        = {g_raw.round(2)}   量级: {np.linalg.norm(g_raw):.1f}")
    print(f"  ∇Ψ 归一化      = {g_norm.round(6)}  ‖ĝ‖={np.linalg.norm(g_norm):.6f}")
    print(f"  ∇Ψ 差分(原始)  = {g_fd.round(2)}")
    err = np.max(np.abs(g_raw - g_fd))
    print(f"  解析 vs 差分误差 = {err:.2e}  (期望 < 1e-5)")

    print("\n" + "=" * 60)
    print("2. CouplingMatrixManager 自测")
    print("=" * 60)
    cm = CouplingMatrixManager()
    # 故意传入非 PSD 矩阵（有负特征值）
    W_bad = np.array([[1.0, 2.0, 0.0],
                      [2.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
    print(f"  W_bad 最小特征值: {np.linalg.eigvalsh(W_bad).min():.4f}  (应<0)")
    W_t = np.eye(3) * 0.5
    W_te = np.eye(3) * 0.3
    cm.set_llm_matrices(W_bad, W_t, W_te)
    w_vec = np.array([0.6, 0.3, 0.1])
    W_cur = cm.get_W(w_vec)
    min_eig = np.linalg.eigvalsh(W_cur).min()
    print(f"  W^(t) 最小特征值: {min_eig:.6f}  (期望 ≥ 0)")

    print("\n" + "=" * 60)
    print("3. GammaAnnealer 自测")
    print("=" * 60)
    ga = GammaAnnealer(gamma_max=0.3, gamma_min=0.05, t_decay=20.0)
    for t in [0, 5, 10, 20, 50]:
        print(f"  γ(t={t:2d}) = {ga.gamma(t):.4f}")

    print("\n" + "=" * 60)
    print("4. PhysicsGPModel fit + predict 自测")
    print("=" * 60)
    psi_fn, coupling_mgr, gamma_ann, gp = build_gp_stack(BOUNDS)
    # 单位矩阵（LLM 未提供时默认）
    # coupling_mgr.set_llm_matrices(...)  ← 实际使用由 Touchpoint 1a 调用

    rng = np.random.default_rng(42)
    n_train = 12
    X_tr = rng.uniform([3.0, 0.1, 1.0], [7.0, 0.7, 5.0], size=(n_train, 3))
    F_tr = rng.uniform(0.0, 1.0, size=n_train)
    w_v  = np.array([0.5, 0.3, 0.2])

    gp.fit(X_tr, F_tr, w_v, t=0)
    summary = gp.training_summary()
    print(f"  l 学习值   = {summary['l']:.4f}")
    print(f"  γ          = {summary['gamma']:.4f}")
    print(f"  μ̂          = {summary['mu_hat']:.4f}")
    print(f"  σ̂²         = {summary['sig2_hat']:.6f}")
    print(f"  LLM W 激活  = {summary['llm_W_active']}")

    X_te = rng.uniform([3.0, 0.1, 1.0], [7.0, 0.7, 5.0], size=(5, 3))
    mean, std = gp.predict(X_te)
    print(f"  预测均值   = {mean.round(4)}")
    print(f"  预测标准差 = {std.round(4)}")
    assert std.min() >= 0.0, "标准差含负值!"
    assert std.max() > 1e-6, "标准差全为 0，核矩阵可能数值异常!"
    print("  ✓ predict() 通过，std > 0")

    # GPProtocol 合规性检查
    assert isinstance(gp, GPProtocol), "PhysicsGPModel 不满足 GPProtocol!"
    print("  ✓ GPProtocol isinstance 检查通过")

    # 验证 gradient_raw() 供 acquisition.py
    g_best = psi_fn.gradient_raw(X_tr[np.argmin(F_tr)])
    print(f"\n  θ_best 处原始梯度 ∇Ψ = {g_best.round(2)}")
    print(f"  （供 acquisition SearchSigmaTracker Eq.20-21）")

    print("\n✓ gp_model 全部自测通过")
