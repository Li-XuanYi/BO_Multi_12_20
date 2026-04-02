"""
components/acquisition.py
==========================
Acquisition Function — LLMBO-MO Framework §5 完整实现

公式对应：
  §5.1  Eq.14  : α(θ)     = EI(θ) · W_charge(θ)
  §5.2  Eq.15  : EI(θ)    = (f_min − f̂(θ))·Φ(z) + s(θ)·φ(z)
  §5.2  Eq.16  : z        = (f_min − f̂(θ)) / s(θ)
  §5.3  Eq.17  : W_charge = Π_j N(θⱼ; μⱼ, σⱼ²)
  §5.3  Eq.18  : μⱼ^(t+1) = α_t · μⱼ^(t) + (1−α_t) · θⱼ^best
  §5.3  Eq.19  : α_t      = α_max·exp(−t/t_decay_α) + α_min
  §5.3  Eq.20  : σⱼ^(t)   = c / (|∂Ψ/∂θⱼ|_{θ_best} + ε_σ)
  §5.3  Eq.21  : c        = κ · max_j |∂Ψ/∂θⱼ|
  §5.3  Eq.22  : σⱼ 停滞扩展: σⱼ^(t+1) = σⱼ^(t) · (1 + ρ·1[stagnation])
  §5.4         : 按 α 降序选 top-N_select 候选

模块职责（单一原则）：
  ① 提供与外部模块的 Protocol 接口定义（GP/Database/LLM）
  ② 实现 μ/σ 动态追踪器（与迭代状态解耦，可单独测试）
  ③ 纯函数式 EI 和 W_charge 计算（内部无状态，便于消融替换）
  ④ AcquisitionFunction 门面类：聚合所有组件，供 optimizer.py 调用

对外接口（被 optimizer.py 使用）：
─────────────────────────────────────────────────────────────────────
  [Protocols]
  GPProtocol          ← 导入自 gp_model，re-export
  DatabaseProtocol    ← database.py 需实现：get_f_min / get_theta_best /
                         has_improved / get_stagnation_count
  LLMPriorProtocol    ← llm_interface.py 需实现：get_warmstart_center
                         （可选，仅用于 μ 的预热初始化）

  [主入口]
  AcquisitionFunction.initialize(theta_best_init)
      ← 在 warmstart 评估完成后、首次迭代前调用（Algorithm 步骤 5）
  AcquisitionFunction.step(X_candidates, database, t, w_vec) → AcquisitionResult
      ← 每迭代由 optimizer.py 调用（Algorithm 步骤 26-29）
  AcquisitionFunction.get_state() → AcquisitionState
      ← DatabaseSummarizer 读取 μ/σ/stagnation 状态

  [工厂函数]
  build_acquisition_function(gp, psi_fn, param_bounds, ...) → AcquisitionFunction
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy.stats import norm as scipy_norm

# 从 gp_model 导入并 re-export（让 optimizer.py 只需 import acquisition）
from llmbo.gp_model import GPProtocol, PsiFunction

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# §A  跨模块 Protocol 接口定义
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class DatabaseProtocol(Protocol):
    """
    Database（D_M）最小接口约定。

    database.py 的 Database 类必须实现这四个方法；
    消融实验或测试时可用任何满足此 Protocol 的对象替换。

    设计说明
    --------
    - get_f_min / get_theta_best 返回当前 D_M 中最优的 Tchebycheff 标量值和对应 θ
    - has_improved 用于停滞检测（Eq.22）：若上次迭代后 f_min 未下降则返回 False
    - get_stagnation_count 返回连续未改进迭代次数（用于日志和调试）
    """

    def get_f_min(self) -> float:
        """返回 D_M 中最好的 Tchebycheff 标量目标值（最小化）。"""
        ...

    def get_theta_best(self) -> np.ndarray:
        """返回对应 f_min 的决策向量 θ ∈ ℝ³。"""
        ...

    def has_improved(self) -> bool:
        """
        如果上次迭代后 f_min 下降则返回 True，否则返回 False。
        供 SearchSigmaTracker 判断停滞（Eq.22）。
        """
        ...

    def get_stagnation_count(self) -> int:
        """返回连续未改进的迭代次数（0 = 刚刚改进过）。"""
        ...


@runtime_checkable
class LLMPriorProtocol(Protocol):
    """
    LLM 初始先验接口（Touchpoint 1b / warm_start 阶段可选）。

    llm_interface.py 的 LLMInterface 类实现此接口；
    若不使用 LLM 先验初始化 μ，则传入 None 让 SearchMuTracker
    用 θ_best（D_M 最优点）初始化。

    典型场景
    --------
    - 若 LLM warm_start 提供了中心点建议（如平均协议参数），可通过此接口传入。
    - 若 warm_start 仅提供离散点（无建议中心），此方法返回 None，
      SearchMuTracker 退而使用 θ_best 初始化。
    """

    def get_warmstart_center(self) -> Optional[np.ndarray]:
        """
        返回 LLM 建议的搜索中心 μ_init ∈ ℝ³，或 None。

        返回 None 时 SearchMuTracker.initialize() 以 θ_best 作为初始 μ。
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# §B  AcquisitionState 数据类（对应 §8.2 OptimizerState 的 acquisition 切片）
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class AcquisitionState:
    """
    Acquisition 函数的可序列化状态快照。

    由 AcquisitionFunction.get_state() 返回，供：
      - DatabaseSummarizer 构造 <sensitivity> / data_card 块
      - optimizer.py 在检查点保存 / 恢复
      - 日志和可视化

    Fields
    ------
    mu           : (3,)   当前搜索中心（Eq.18 维护）
    sigma        : (3,)   当前搜索范围（Eq.20 维护）
    alpha_t      : float  当前 LLM 信任度 α_t（Eq.19）
    stagnation_count : int  连续未改进迭代次数
    t            : int    当前迭代编号
    f_min        : float  当前最优 Tchebycheff 值
    theta_best   : (3,)   当前最优 θ
    grad_psi_at_best : (3,)  ∇Ψ(θ_best) 原始量（供 DatabaseSummarizer §7.2）
    """
    mu:               np.ndarray
    sigma:            np.ndarray
    alpha_t:          float
    stagnation_count: int
    t:                int
    f_min:            float
    theta_best:       np.ndarray
    grad_psi_at_best: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 友好字典（用于检查点）。"""
        return {
            "mu":               self.mu.tolist(),
            "sigma":            self.sigma.tolist(),
            "alpha_t":          float(self.alpha_t),
            "stagnation_count": int(self.stagnation_count),
            "t":                int(self.t),
            "f_min":            float(self.f_min),
            "theta_best":       self.theta_best.tolist(),
            "grad_psi_at_best": self.grad_psi_at_best.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AcquisitionState":
        """从检查点字典恢复。"""
        return cls(
            mu=np.array(d["mu"]),
            sigma=np.array(d["sigma"]),
            alpha_t=float(d["alpha_t"]),
            stagnation_count=int(d["stagnation_count"]),
            t=int(d["t"]),
            f_min=float(d["f_min"]),
            theta_best=np.array(d["theta_best"]),
            grad_psi_at_best=np.array(d["grad_psi_at_best"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# §C  AcquisitionResult 数据类（step() 返回值）
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class AcquisitionResult:
    """
    AcquisitionFunction.step() 的完整返回结果。

    optimizer.py 只需用到 selected_thetas 进行 PyBaMM 评估；
    其余字段供 DatabaseSummarizer、日志、可视化使用。

    Fields
    ------
    selected_thetas  : List[np.ndarray]  top-N_select 候选点，已按 α 降序排列
    selected_indices : List[int]         在原始 X_candidates 中的行索引
    selected_scores  : np.ndarray        对应的 α 分值 (N_select,)

    all_alpha    : (m,)  全部候选点的 α = EI × W_charge
    all_ei       : (m,)  全部候选点的 EI 值
    all_wcharge  : (m,)  全部候选点的 W_charge 值
    all_mean     : (m,)  GP 后验均值
    all_std      : (m,)  GP 后验标准差

    state        : AcquisitionState   更新后的 μ/σ 状态快照
    debug        : dict               原始数值供调试（f_min, stagnated, etc.）
    """
    # 选出的候选点
    selected_thetas:  List[np.ndarray]
    selected_indices: List[int]
    selected_scores:  np.ndarray

    # 全部候选点的得分分解
    all_alpha:   np.ndarray
    all_ei:      np.ndarray
    all_wcharge: np.ndarray
    all_mean:    np.ndarray
    all_std:     np.ndarray

    # 状态与调试
    state: AcquisitionState
    debug: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════
# §D  SearchMuTracker — μ 动态漂移（Eq.18-19）
# ═══════════════════════════════════════════════════════════════════════════

class SearchMuTracker:
    """
    搜索中心 μ 的动态漂移追踪器。

    规则（Framework §5.3）：
      α_t = α_max · exp(−t / t_decay_α) + α_min            Eq.19
      μⱼ^(t+1) = α_t · μⱼ^(t) + (1−α_t) · θⱼ^best,(t)   Eq.18

    物理含义：
      - 早期（t≈0, α_t≈0.75）：μ 偏向 LLM 先验/初始化位置，缓慢漂移
      - 晚期（t→∞,  α_t≈0.05）：μ 快速跟踪 θ_best，数据主导

    对外接口
    --------
    initialize(theta_init, llm_prior)
        → 在 warmstart 评估后调用（Algorithm 步骤 5）

    update(theta_best, t)
        → 每迭代调用（Algorithm 步骤 27）

    get_mu() → np.ndarray (3,)
    get_alpha(t) → float
    state_dict() → dict   （供 AcquisitionState 快照）
    """

    def __init__(
        self,
        param_bounds:  Dict[str, Tuple[float, float]],
        alpha_max:     float = 0.7,
        alpha_min:     float = 0.05,
        t_decay_alpha: float = 60.0,
    ):
        """
        Parameters
        ----------
        param_bounds  : {"I1": (lo, hi), "SOC1": (lo, hi), "I2": (lo, hi)}
        alpha_max     : 初始 LLM 信任度（Eq.19 参数，默认 0.7，§10）
        alpha_min     : 最小 LLM 信任度（Eq.19 参数，默认 0.05，§10）
        t_decay_alpha : 衰减时间尺度（Eq.19 参数，默认 60，§10）
        """
        self._bounds = param_bounds
        self._lo  = np.array([param_bounds["I1"][0],
                               param_bounds["SOC1"][0],
                               param_bounds["I2"][0]], dtype=float)
        self._hi  = np.array([param_bounds["I1"][1],
                               param_bounds["SOC1"][1],
                               param_bounds["I2"][1]], dtype=float)
        self.alpha_max     = float(alpha_max)
        self.alpha_min     = float(alpha_min)
        self.t_decay_alpha = float(t_decay_alpha)

        # μ 内部状态（在 initialize() 之前用参数空间中心）
        self._mu: np.ndarray = (self._lo + self._hi) / 2.0
        self._initialized: bool = False

    # ── 初始化（Algorithm 步骤 5） ────────────────────────────────────────
    def initialize(
        self,
        theta_best: np.ndarray,
        llm_prior:  Optional[LLMPriorProtocol] = None,
    ) -> None:
        """
        在 warmstart 评估完成后、第一次迭代前调用。

        优先级：
          1. llm_prior.get_warmstart_center()（若非 None 且接口返回非 None）
          2. theta_best（D_M 中最优点，Algorithm 步骤 5 明确要求）

        llm_prior 提供的是 LLM 对"好的充电协议"的先验期望中心，
        若 LLM 无此先验（只提供离散点），则退回 theta_best。
        """
        mu_init: Optional[np.ndarray] = None

        if llm_prior is not None:
            try:
                mu_candidate = llm_prior.get_warmstart_center()
                if mu_candidate is not None:
                    mu_candidate = np.asarray(mu_candidate, dtype=float).ravel()
                    if mu_candidate.size == 3 and self._is_within_bounds(mu_candidate):
                        mu_init = mu_candidate
                        logger.info("SearchMuTracker: μ 由 LLM prior 初始化: %s", mu_init)
                    else:
                        logger.warning(
                            "SearchMuTracker: LLM prior 提供的 μ 越界或维度错误，退回 θ_best"
                        )
            except Exception as exc:
                logger.warning("SearchMuTracker: LLM prior 调用失败 (%s)，退回 θ_best", exc)

        if mu_init is None:
            mu_init = np.asarray(theta_best, dtype=float).ravel().copy()
            logger.info("SearchMuTracker: μ 由 θ_best 初始化: %s", mu_init)

        self._mu = np.clip(mu_init, self._lo, self._hi)
        self._initialized = True

    # ── 每迭代更新（Algorithm 步骤 27，Eq.18-19） ─────────────────────────
    def update(self, theta_best: np.ndarray, t: int) -> None:
        """
        μⱼ^(t+1) = α_t · μⱼ^(t) + (1−α_t) · θⱼ^best  (Eq.18)

        Parameters
        ----------
        theta_best : (3,) 当前 D_M 最优 θ（由 Database.get_theta_best() 提供）
        t          : 当前迭代编号（从 0 开始）
        """
        if not self._initialized:
            logger.warning("SearchMuTracker.update() 在 initialize() 前调用，使用当前 μ 继续")
        theta_best = np.asarray(theta_best, dtype=float).ravel()
        alpha = self.get_alpha(t)
        self._mu = alpha * self._mu + (1.0 - alpha) * theta_best
        self._mu = np.clip(self._mu, self._lo, self._hi)   # 保证在参数空间内
        logger.debug("SearchMuTracker: t=%d α_t=%.4f μ=%s", t, alpha, self._mu.round(4))

    # ── 读取接口 ──────────────────────────────────────────────────────────
    def get_mu(self) -> np.ndarray:
        """返回当前搜索中心 μ ∈ ℝ³（已裁剪到参数边界）。"""
        return self._mu.copy()

    def get_alpha(self, t: int) -> float:
        """α_t = α_max·exp(−t/t_decay_α) + α_min  (Eq.19)"""
        return float(
            self.alpha_max * np.exp(-t / self.t_decay_alpha) + self.alpha_min
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mu":           self._mu.tolist(),
            "alpha_max":    self.alpha_max,
            "alpha_min":    self.alpha_min,
            "t_decay_alpha": self.t_decay_alpha,
            "initialized":  self._initialized,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self._mu = np.array(d["mu"])
        self._initialized = bool(d.get("initialized", True))

    def _is_within_bounds(self, theta: np.ndarray) -> bool:
        return bool(np.all(theta >= self._lo) and np.all(theta <= self._hi))


# ═══════════════════════════════════════════════════════════════════════════
# §E  SearchSigmaTracker — σ 敏感度引导 + 停滞扩展（Eq.20-22）
# ═══════════════════════════════════════════════════════════════════════════

class SearchSigmaTracker:
    """
    搜索范围 σ 的物理敏感度引导追踪器。

    规则（Framework §5.3）：
      c       = κ · max_j |∂Ψ/∂θⱼ|_{θ_best}               Eq.21
      σⱼ^(t) = c / (|∂Ψ/∂θⱼ|_{θ_best} + ε_σ)             Eq.20
      若停滞：σⱼ^(t+1) = σⱼ^(t) · (1 + ρ)                  Eq.22

    物理含义（Framework §5.3 注释）：
      - |∂Ψ/∂θⱼ| 大 → Ψ 对 θⱼ 敏感 → σⱼ 小 → 细粒度搜索
      - |∂Ψ/∂θⱼ| 小 → Ψ 对 θⱼ 不敏感 → σⱼ 大 → 粗粒度搜索
      - 停滞时 σ 扩张 10%，鼓励探索新区域

    注意：
      σⱼ 的上界被裁剪到各参数范围的 0.5 倍（防止搜索范围超出合理物理区间）。

    对外接口
    --------
    initialize(theta_best)        → 在 warmstart 后初始化
    update(theta_best, stagnated) → 每迭代调用（先按梯度重算，再判停滞）
    get_sigma() → np.ndarray (3,)
    state_dict() → dict
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        psi_fn:       PsiFunction,
        kappa:        float = 0.20,
        eps_sigma:    float = 0.001,
        rho:          float = 0.1,
    ):
        """
        Parameters
        ----------
        param_bounds : {"I1":..., "SOC1":..., "I2":...}
        psi_fn       : PsiFunction — gradient_raw() 供 Eq.20 使用
        kappa        : κ ≈ 0.20（Eq.21，§10）
        eps_sigma    : ε_σ = 0.001（Eq.20 数值稳定项，§5.3）
        rho          : ρ = 0.1（Eq.22 停滞扩张率，§5.3）
        """
        self._bounds    = param_bounds
        self._psi_fn    = psi_fn
        self.kappa      = float(kappa)
        self.eps_sigma  = float(eps_sigma)
        self.rho        = float(rho)

        # 参数范围（用于裁剪 σ 上界）
        self._lo  = np.array([param_bounds["I1"][0],
                               param_bounds["SOC1"][0],
                               param_bounds["I2"][0]], dtype=float)
        self._hi  = np.array([param_bounds["I1"][1],
                               param_bounds["SOC1"][1],
                               param_bounds["I2"][1]], dtype=float)
        self._range = self._hi - self._lo
        self._sigma_max = 0.5 * self._range   # σ 上界（物理约束）

        # σ 内部状态
        self._sigma: np.ndarray = 0.2 * self._range   # 默认初始值
        self._last_grad: np.ndarray = np.ones(3)       # 上次 ∇Ψ 原始值（日志用）

    # ── 初始化 ────────────────────────────────────────────────────────────
    def initialize(self, theta_best: np.ndarray) -> None:
        """
        在 warmstart 评估后、第一次迭代前调用（不判停滞）。

        Parameters
        ----------
        theta_best : (3,) D_M 中最优 θ
        """
        self._sigma = self._compute_sigma_from_grad(theta_best)
        logger.info(
            "SearchSigmaTracker 初始化: σ=%s  (θ_best=%s)",
            self._sigma.round(4), np.asarray(theta_best).round(4)
        )

    # ── 每迭代更新（Algorithm 步骤 27，Eq.20-22） ─────────────────────────
    def update(self, theta_best: np.ndarray, stagnated: bool) -> None:
        """
        先按当前 θ_best 重算 σ（Eq.20-21），再判停滞扩张（Eq.22）。

        更新顺序：
          1. 用新 θ_best 计算 ∇Ψ（gradient_raw，原始量纲）
          2. 按 Eq.20-21 重算 σⱼ
          3. 若 stagnated=True，σⱼ × (1 + ρ)（Eq.22）
          4. 裁剪到 [ε_σ·10, σ_max]

        Parameters
        ----------
        theta_best : (3,) 当前 D_M 最优 θ
        stagnated  : bool 若上迭代 f_min 未改进则为 True
        """
        sigma_new = self._compute_sigma_from_grad(theta_best)

        # Eq.22：停滞扩张
        if stagnated:
            sigma_new = sigma_new * (1.0 + self.rho)
            logger.debug(
                "SearchSigmaTracker: 停滞扩张 ρ=%.2f → σ=%s",
                self.rho, sigma_new.round(4)
            )

        self._sigma = sigma_new
        logger.debug(
            "SearchSigmaTracker: stagnated=%s  σ=%s",
            stagnated, self._sigma.round(4)
        )

    # ── 读取接口 ──────────────────────────────────────────────────────────
    def get_sigma(self) -> np.ndarray:
        """返回当前搜索范围 σ ∈ ℝ³（已裁剪到物理合理范围）。"""
        return self._sigma.copy()

    def get_last_grad(self) -> np.ndarray:
        """返回上次计算时的 ∇Ψ 原始值（供 DatabaseSummarizer §7.2 <sensitivity> 块）。"""
        return self._last_grad.copy()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "sigma":      self._sigma.tolist(),
            "last_grad":  self._last_grad.tolist(),
            "kappa":      self.kappa,
            "eps_sigma":  self.eps_sigma,
            "rho":        self.rho,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self._sigma = np.array(d["sigma"])
        self._last_grad = np.array(d.get("last_grad", np.ones(3)))

    # ── 内部：Eq.20-21 核心计算 ──────────────────────────────────────────
    def _compute_sigma_from_grad(self, theta_best: np.ndarray) -> np.ndarray:
        """
        σⱼ = c / (|∂Ψ/∂θⱼ| + ε_σ)   Eq.20
        c   = κ · max_j |∂Ψ/∂θⱼ|      Eq.21

        使用 gradient_raw()（原始量纲）确保各维度敏感度比例真实反映
        物理意义（不归一化）。
        """
        grad = self._psi_fn.gradient_raw(np.asarray(theta_best, dtype=float))
        self._last_grad = grad.copy()
        abs_grad = np.abs(grad)

        # Eq.21
        c = self.kappa * abs_grad.max()

        # Eq.20
        sigma_raw = c / (abs_grad + self.eps_sigma)

        # 下界设为参数范围的 5%，避免高敏感维度探索坍塌
        sigma_lower = self._range * 0.05
        sigma_clipped = np.clip(sigma_raw, sigma_lower, self._sigma_max)
        return sigma_clipped


# ═══════════════════════════════════════════════════════════════════════════
# §F  EICalculator — 期望改进（Eq.15-16）
# ═══════════════════════════════════════════════════════════════════════════

class EICalculator:
    """
    期望改进（Expected Improvement）计算器。

    EI(θ) = (f_min − f̂(θ) − ξ) · Φ(z) + s(θ) · φ(z)   Eq.15
    z     = (f_min − f̂(θ) − ξ) / s(θ)                   Eq.16

    通过 GPProtocol 与 GP 解耦，方便消融测试时替换 GP。

    Parameters
    ----------
    gp : GPProtocol
        任何满足 GPProtocol 的 GP 模型（PhysicsGPModel 或标准 GP）。
    xi : float, default 0.0
        勘探奖励（exploration bonus）。xi > 0 鼓励更多探索；
        Framework §5.2 未使用，保留接口备用（设为 0 即为原始公式）。
    """

    def __init__(self, gp: GPProtocol, xi: float = 0.0):
        if not isinstance(gp, GPProtocol):
            raise TypeError(
                f"gp 必须满足 GPProtocol，收到 {type(gp).__name__}"
            )
        self._gp = gp
        self.xi  = float(xi)

    # ── 主计算接口 ────────────────────────────────────────────────────────
    def compute(
        self,
        X_candidates: np.ndarray,
        f_min:        float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量计算 EI。

        Parameters
        ----------
        X_candidates : (m, 3)  LLM 生成的候选点
        f_min        : float   D_M 中最好的 Tchebycheff 值（最小化问题）

        Returns
        -------
        ei   : (m,)  EI 值（非负）
        mean : (m,)  GP 后验均值 f̂(θ)
        std  : (m,)  GP 后验标准差 s(θ)（非负）
        """
        X_candidates = np.atleast_2d(X_candidates)
        mean, std = self._gp.predict(X_candidates)

        # 数值保护：std 确保非负
        std = np.maximum(std, 0.0)

        improvement = f_min - mean - self.xi       # f_min − f̂(θ) − ξ

        # z = improvement / std（std=0 时设 z=0 避免 NaN）
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(std > 1e-12, improvement / std, 0.0)

        # EI = improvement·Φ(z) + std·φ(z)
        ei = improvement * scipy_norm.cdf(z) + std * scipy_norm.pdf(z)
        ei = np.maximum(ei, 0.0)   # 数值保护

        logger.debug(
            "EICalculator: m=%d  f_min=%.6f  EI in [%.3e, %.3e]  "
            "mean in [%.4f, %.4f]  std in [%.4f, %.4f]",
            X_candidates.shape[0], f_min,
            ei.min(), ei.max(),
            mean.min(), mean.max(),
            std.min(), std.max(),
        )
        return ei, mean, std

    @property
    def gp(self) -> GPProtocol:
        """提供对底层 GP 的只读访问（供 AcquisitionFunction 获取 training_summary）。"""
        return self._gp


# ═══════════════════════════════════════════════════════════════════════════
# §G  WChargeCalculator — 物理加权（Eq.17）
# ═══════════════════════════════════════════════════════════════════════════

class WChargeCalculator:
    """
    物理基加权函数 W_charge(θ)。

    W_charge(θ) = Π_{j=1}^{3} N(θⱼ; μⱼ, σⱼ²)   Eq.17

    其中 N(·; μ, σ²) = (1/√(2πσ²)) exp(−(θ−μ)²/(2σ²)) 为高斯 PDF。

    计算策略
    --------
    - 内部在 log 空间计算（防止多个小 PDF 值连乘导致数值下溢）：
        log_W = Σⱼ [−½log(2π) − log(σⱼ) − (θⱼ−μⱼ)²/(2σⱼ²)]
    - 返回实际值 W = exp(log_W)（已做 log_softmax 归一化防溢出）
    - 若所有候选点 log_W 都很小（极端情况），返回均匀权重

    物理含义
    --------
    μⱼ 是当前认为最好的充电参数区域中心（由 SearchMuTracker 维护），
    σⱼ 是基于物理敏感度的搜索半径（由 SearchSigmaTracker 维护）。
    W_charge 越大表示候选点越接近当前物理最优区域。
    """

    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        self._bounds = param_bounds
        self._lo = np.array([param_bounds["I1"][0],
                              param_bounds["SOC1"][0],
                              param_bounds["I2"][0]], dtype=float)
        self._hi = np.array([param_bounds["I1"][1],
                              param_bounds["SOC1"][1],
                              param_bounds["I2"][1]], dtype=float)

    # ── 主计算接口 ────────────────────────────────────────────────────────
    def compute(
        self,
        X_candidates: np.ndarray,   # (m, 3)
        mu:           np.ndarray,   # (3,)  SearchMuTracker.get_mu()
        sigma:        np.ndarray,   # (3,)  SearchSigmaTracker.get_sigma()
    ) -> np.ndarray:
        """
        批量计算 W_charge(θ)。

        Parameters
        ----------
        X_candidates : (m, 3)
        mu           : (3,)   当前搜索中心
        sigma        : (3,)   当前搜索范围（均为正数）

        Returns
        -------
        wcharge : (m,)  W_charge 值（非负，已做数值稳定处理）
        """
        X_candidates = np.atleast_2d(X_candidates)
        mu    = np.asarray(mu, dtype=float).ravel()
        sigma = np.maximum(np.asarray(sigma, dtype=float).ravel(), 1e-8)

        # log W_charge = Σⱼ log N(θⱼ; μⱼ, σⱼ²)
        #              = Σⱼ [-½log(2π) - log(σⱼ) - (θⱼ-μⱼ)²/(2σⱼ²)]
        diff    = X_candidates - mu[np.newaxis, :]   # (m, 3)
        log_w   = -0.5 * np.log(2.0 * np.pi) * 3    # 常数项（对排序无影响但保留完整性）
        log_w   = log_w - np.sum(np.log(sigma))      # -Σ log(σⱼ)
        log_w   = log_w - 0.5 * np.sum(
            (diff / sigma[np.newaxis, :]) ** 2, axis=1
        )   # (m,)  -Σ (θⱼ-μⱼ)²/(2σⱼ²)

        # 数值稳定：shift by max → 转回 exp 不下溢
        log_w_shifted = log_w - log_w.max()
        wcharge = np.exp(log_w_shifted)

        # 极端情况保护：若所有值都趋向 0（通常不会发生），返回均匀权重
        total = wcharge.sum()
        if total < 1e-300:
            logger.warning(
                "WChargeCalculator: 所有候选点 W_charge ≈ 0，退回均匀权重"
            )
            return np.ones(X_candidates.shape[0]) / X_candidates.shape[0]

        logger.debug(
            "WChargeCalculator: m=%d  W_charge in [%.3e, %.3e]  "
            "μ=%s  σ=%s",
            X_candidates.shape[0], wcharge.min(), wcharge.max(),
            mu.round(3), sigma.round(4),
        )
        return wcharge

    def compute_log(
        self,
        X_candidates: np.ndarray,
        mu:           np.ndarray,
        sigma:        np.ndarray,
    ) -> np.ndarray:
        """
        返回 log W_charge（未 shift 版本），供需要 log 空间运算的场景。

        Returns
        -------
        log_wcharge : (m,)  未归一化的 log W_charge
        """
        X_candidates = np.atleast_2d(X_candidates)
        mu    = np.asarray(mu, dtype=float).ravel()
        sigma = np.maximum(np.asarray(sigma, dtype=float).ravel(), 1e-8)
        diff  = X_candidates - mu[np.newaxis, :]
        log_w = (
            -0.5 * np.log(2.0 * np.pi) * 3
            - np.sum(np.log(sigma))
            - 0.5 * np.sum((diff / sigma[np.newaxis, :]) ** 2, axis=1)
        )
        return log_w


# ═══════════════════════════════════════════════════════════════════════════
# §H  AcquisitionScorer — α = EI × W_charge + top-k 选择（Eq.14，§5.4）
# ═══════════════════════════════════════════════════════════════════════════

class AcquisitionScorer:
    """
    综合评分器：α(θ) = EI(θ) · W_charge(θ)  (Eq.14)。

    将 EICalculator 和 WChargeCalculator 的输出相乘，
    并执行 §5.4 的 top-N_select 选择。

    设计说明
    --------
    - EI 和 W_charge 量级差异较大时，直接相乘可能导致某一项完全主导排序。
      本实现提供 `log_mode=True` 选项：在 log 空间相加，等效于原始乘法，
      但数值更稳定（特别是候选点数量多时）。
    - 默认 log_mode=False（与 Framework Eq.14 一致）。
    """

    def __init__(
        self,
        ei_calc:      EICalculator,
        wcharge_calc: WChargeCalculator,
        n_select:     int = 3,
        log_mode:     bool = False,
    ):
        """
        Parameters
        ----------
        ei_calc      : EICalculator
        wcharge_calc : WChargeCalculator
        n_select     : N_select（§10 默认 3）
        log_mode     : 若 True，用 log(EI) + log(W_charge) 替代乘法（数值更稳定）
        """
        self._ei_calc      = ei_calc
        self._wcharge_calc = wcharge_calc
        self.n_select      = int(n_select)
        self.log_mode      = bool(log_mode)

    # ── 评分 ──────────────────────────────────────────────────────────────
    def score(
        self,
        X_candidates: np.ndarray,   # (m, 3)
        f_min:        float,
        mu:           np.ndarray,   # (3,)
        sigma:        np.ndarray,   # (3,)
        t:            int = 0,      # Fix 2: 当前迭代
        T:            int = 20,     # Fix 2: 总迭代数
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fix 2: 基于排名的平衡采集函数，解决 W_charge 动态范围主导问题。

        EI 动态范围 ~10⁴×，W_charge 动态范围 ~10²⁶×，直接相乘等价于只看 W_charge。
        改为：rank_normalize 后加权融合，λ 随迭代退火（早期 0.6 → 晚期 0.3）。

        Returns
        -------
        alpha   : (m,)  综合得分（rank-based 融合）
        ei      : (m,)  EI 分量（原始值）
        wcharge : (m,)  W_charge 分量（原始值）
        mean    : (m,)  GP 后验均值
        std     : (m,)  GP 后验标准差
        """
        ei, mean, std = self._ei_calc.compute(X_candidates, f_min)
        wcharge       = self._wcharge_calc.compute(X_candidates, mu, sigma)

        # EI 退化保护：若 GP 尚未收敛（EI 全部接近 0），fallback 到 GP-UCB
        if ei.max() < 1e-10:
            alpha = mean + 2.0 * std   # GP-UCB, kappa=2
            logger.debug("AcquisitionScorer: EI 退化，fallback 到 GP-UCB")
        else:
            # Fix 2: rank-normalize 两者到 [0,1] 均匀分布
            ei_rank = self._rank_normalize(ei)
            wc_rank = self._rank_normalize(wcharge)

            # 退火：t=0 时 λ=0.6（W_charge 权重适中），t=T 时 λ=0.3（更重视 EI）
            lambda_t = 0.3 * np.exp(-3.0 * t / max(T, 1)) + 0.3
            alpha = (1.0 - lambda_t) * ei_rank + lambda_t * wc_rank

        logger.debug(
            "AcquisitionScorer: m=%d  α in [%.3e, %.3e]  "
            "EI in [%.3e, %.3e]  W in [%.3e, %.3e]  t=%d  T=%d",
            X_candidates.shape[0],
            alpha.min(), alpha.max(),
            ei.min(), ei.max(),
            wcharge.min(), wcharge.max(),
            t, T,
        )
        return alpha, ei, wcharge, mean, std

    @staticmethod
    def _rank_normalize(x: np.ndarray) -> np.ndarray:
        """将数组映射到 [0,1] 均匀分布（基于排名），消除量纲差异。"""
        ranks = np.argsort(np.argsort(x))   # 双重 argsort 得到 0-based 排名
        return ranks / (len(ranks) - 1 + 1e-12)

    # ── top-k 选择（§5.4） ────────────────────────────────────────────────
    def select_top_k(
        self,
        X_candidates: np.ndarray,
        alpha:        np.ndarray,
        k:            Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
        """
        按 α 降序选 top-k 候选点（§5.4：默认 k = N_select = 3）。

        处理特殊情况：
          - 若候选点数量 < k，返回全部（按 α 排序）
          - 若所有 α 相等（极端情况），随机选 k 个

        Returns
        -------
        selected_thetas  : List[np.ndarray]  选出的 θ 列表
        selected_indices : List[int]          在 X_candidates 中的行索引
        selected_scores  : np.ndarray (k,)    对应的 α 值
        """
        k = k if k is not None else self.n_select
        m = X_candidates.shape[0]
        k_actual = min(k, m)

        # 按 α 降序排列
        sorted_idx = np.argsort(alpha)[::-1]
        top_idx    = sorted_idx[:k_actual]

        selected_thetas  = [X_candidates[i].copy() for i in top_idx]
        selected_indices = top_idx.tolist()
        selected_scores  = alpha[top_idx]

        logger.info(
            "AcquisitionScorer: 从 %d 个候选中选出 top-%d  α=%s",
            m, k_actual, selected_scores.round(6)
        )
        return selected_thetas, selected_indices, selected_scores


# ═══════════════════════════════════════════════════════════════════════════
# §I  AcquisitionFunction 门面类（主入口）
# ═══════════════════════════════════════════════════════════════════════════

class AcquisitionFunction:
    """
    Acquisition Function 门面类。

    聚合 SearchMuTracker, SearchSigmaTracker, EICalculator,
    WChargeCalculator, AcquisitionScorer，供 optimizer.py 使用。

    对外接口（optimizer.py 调用顺序）：
    ─────────────────────────────────────────────────────────────────
    1. 初始化阶段（warm_start 评估完成后）：
       af.initialize(database, llm_prior=None)

    2. 每迭代（Algorithm 步骤 26-29）：
       result = af.step(X_candidates, database, t, w_vec)
       → result.selected_thetas  供 PyBaMM 评估

    3. 状态查询（供 DatabaseSummarizer §7.2）：
       state = af.get_state()
       state.grad_psi_at_best  → <sensitivity> 块
       state.mu, state.sigma   → data_card 块

    4. 检查点保存 / 恢复：
       af.save_state() → dict
       af.load_state(d)
    ─────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        gp:           GPProtocol,
        psi_fn:       PsiFunction,
        param_bounds: Dict[str, Tuple[float, float]],
        n_select:     int   = 3,
        # μ tracker 参数
        alpha_max:     float = 0.7,
        alpha_min:     float = 0.05,
        t_decay_alpha: float = 60.0,
        # σ tracker 参数
        kappa:         float = 0.20,
        eps_sigma:     float = 0.001,
        rho:           float = 0.1,
        # EI 参数
        xi:            float = 0.0,
        # 评分参数
        log_mode:      bool  = False,
        # Fix 2: 总迭代数，供退火公式使用
        max_iterations: int  = 20,
    ):
        """
        Parameters
        ----------
        gp           : GPProtocol  已训练的 GP 代理（每迭代 fit 后传入或共享引用）
        psi_fn       : PsiFunction  物理代理函数（gradient_raw() 供 σ 计算）
        param_bounds : {"I1":..., "SOC1":..., "I2":...}
        n_select     : N_select（§10 默认 3）
        alpha_max/min/t_decay_alpha : μ 漂移参数（Eq.19，§10）
        kappa/eps_sigma/rho         : σ 敏感度参数（Eq.20-22，§10）
        xi           : EI 勘探奖励（默认 0，保持与 §5.2 一致）
        log_mode     : True → α 在 log 空间计算（数值更稳定，但偏离 Eq.14 字面）
        """
        self._gp      = gp
        self._psi_fn  = psi_fn
        self._bounds  = param_bounds
        self._n_select = n_select
        self._max_iterations = max_iterations  # Fix 2

        # 子组件实例化
        self._mu_tracker    = SearchMuTracker(
            param_bounds, alpha_max, alpha_min, t_decay_alpha
        )
        self._sigma_tracker = SearchSigmaTracker(
            param_bounds, psi_fn, kappa, eps_sigma, rho
        )
        self._ei_calc       = EICalculator(gp, xi)
        self._wcharge_calc  = WChargeCalculator(param_bounds)
        self._scorer        = AcquisitionScorer(
            self._ei_calc, self._wcharge_calc, n_select, log_mode
        )

        # 运行时状态
        self._current_t:          int   = 0
        self._current_f_min:      float = float("inf")
        self._current_theta_best: Optional[np.ndarray] = None
        self._initialized:        bool  = False

        # ── 归一化边界（供 step() 中 W_charge 坐标对齐）────────────────────
        self._lo = np.array([param_bounds["I1"][0],
                             param_bounds["SOC1"][0],
                             param_bounds["I2"][0]], dtype=float)
        self._hi = np.array([param_bounds["I1"][1],
                             param_bounds["SOC1"][1],
                             param_bounds["I2"][1]], dtype=float)

    # ── 初始化（Algorithm 步骤 5） ────────────────────────────────────────
    def initialize(
        self,
        database:  DatabaseProtocol,
        llm_prior: Optional[LLMPriorProtocol] = None,
    ) -> None:
        """
        在 warmstart 评估完成后、第一次迭代前调用。

        从 Database 获取 θ_best（Algorithm 步骤 5 "μ ← θ_best from D_M"），
        初始化 μ 和 σ。可选传入 LLM 先验中心覆盖 μ 初始化。

        Parameters
        ----------
        database  : DatabaseProtocol  已包含 warm-start 评估结果的 D_M
        llm_prior : LLMPriorProtocol | None  LLM 提供的先验搜索中心（可选）
        """
        theta_best = np.asarray(database.get_theta_best(), dtype=float)
        f_min      = float(database.get_f_min())

        self._mu_tracker.initialize(theta_best, llm_prior)
        self._sigma_tracker.initialize(theta_best)

        self._current_f_min      = f_min
        self._current_theta_best = theta_best.copy()
        self._initialized        = True

        logger.info(
            "AcquisitionFunction 初始化完成: θ_best=%s  f_min=%.6f",
            theta_best.round(4), f_min
        )

    # ── 每迭代主入口（Algorithm 步骤 26-29） ──────────────────────────────
    def step(
        self,
        X_candidates: np.ndarray,        # (m, 3)  LLM 生成的候选点（Touchpoint 2）
        database:     DatabaseProtocol,  # D_M（提供 f_min / θ_best / stagnation）
        t:            int,               # 当前迭代编号（从 0 开始）
        w_vec:        Optional[np.ndarray] = None,   # 当前 Tchebycheff 权重（供日志）
    ) -> AcquisitionResult:
        """
        执行 Algorithm §6 步骤 26-29 的全部逻辑。

        流程：
          26a. 从 Database 读取 f_min, θ_best, stagnation 信息
          26b. 更新 μ（Eq.18-19）
          26c. 更新 σ（Eq.20-22，含停滞判断）
          27.  批量计算 α = EI × W_charge（Eq.14）
          28.  按 α 降序选 top-N_select 候选点

        Parameters
        ----------
        X_candidates : (m, 3)  来自 LLM.generate() 的候选点
        database     : DatabaseProtocol  当前 D_M
        t            : 迭代编号
        w_vec        : (3,) Tchebycheff 权重（仅供日志记录，不影响计算）

        Returns
        -------
        AcquisitionResult — 包含 selected_thetas 和全部调试信息
        """
        if not self._initialized:
            raise RuntimeError(
                "AcquisitionFunction 尚未初始化，请先调用 initialize()"
            )

        X_candidates = np.atleast_2d(X_candidates)
        self._current_t = t

        # ── 步骤 26a：从 Database 读取当前最优状态 ─────────────────────
        f_min      = float(database.get_f_min())
        theta_best = np.asarray(database.get_theta_best(), dtype=float)
        stagnated  = not database.has_improved()
        stagnation_count = database.get_stagnation_count()

        logger.info(
            "AcquisitionFunction.step: t=%d  f_min=%.6f  stagnated=%s  "
            "stagnation_count=%d  n_cand=%d  w=%s",
            t, f_min, stagnated, stagnation_count,
            X_candidates.shape[0],
            w_vec.round(3) if w_vec is not None else "N/A",
        )

        # ── 步骤 26b：更新 μ（Eq.18-19） ──────────────────────────────
        self._mu_tracker.update(theta_best, t)
        mu    = self._mu_tracker.get_mu()
        alpha_t = self._mu_tracker.get_alpha(t)

        # ── 步骤 26c：更新 σ（Eq.20-22） ──────────────────────────────
        self._sigma_tracker.update(theta_best, stagnated)
        sigma = self._sigma_tracker.get_sigma()
        grad_psi = self._sigma_tracker.get_last_grad()   # ∇Ψ 原始值（供摘要）

        # ── 步骤 27：α = EI × W_charge ─────────────────────────────────
        # FIX: X_candidates 是 [0,1]³ 归一化空间，mu/sigma 是物理空间。
        # 将 mu/sigma 映射到同一 [0,1]³ 空间，使 W_charge 距离计算有意义。
        _range = self._hi - self._lo + 1e-12
        mu_norm    = (mu    - self._lo) / _range   # [0,1]³
        sigma_norm = sigma / _range                # 无量纲
        alpha, ei, wcharge, mean, std = self._scorer.score(
            X_candidates, f_min, mu_norm, sigma_norm,
            t=t, T=self._max_iterations,  # Fix 2: 传入迭代信息供退火
        )

        # ── 步骤 28：选 top-N_select ─────────────────────────────────────
        sel_thetas, sel_idx, sel_scores = self._scorer.select_top_k(
            X_candidates, alpha
        )

        # ── 更新内部状态缓存 ─────────────────────────────────────────────
        self._current_f_min      = f_min
        self._current_theta_best = theta_best.copy()

        # ── 构建状态快照（供 DatabaseSummarizer） ─────────────────────
        state = AcquisitionState(
            mu=mu.copy(),
            sigma=sigma.copy(),
            alpha_t=alpha_t,
            stagnation_count=stagnation_count,
            t=t,
            f_min=f_min,
            theta_best=theta_best.copy(),
            grad_psi_at_best=grad_psi.copy(),
        )

        # ── 调试信息 ──────────────────────────────────────────────────
        debug = {
            "t":               t,
            "f_min":           f_min,
            "stagnated":       stagnated,
            "stagnation_count": stagnation_count,
            "alpha_t":         alpha_t,
            "mu":              mu.tolist(),
            "sigma":           sigma.tolist(),
            "grad_psi":        grad_psi.tolist(),
            "n_candidates":    X_candidates.shape[0],
            "n_selected":      len(sel_thetas),
            "w_vec":           w_vec.tolist() if w_vec is not None else None,
            "gp_summary":      self._gp.training_summary(),
        }

        return AcquisitionResult(
            selected_thetas=sel_thetas,
            selected_indices=sel_idx,
            selected_scores=sel_scores,
            all_alpha=alpha,
            all_ei=ei,
            all_wcharge=wcharge,
            all_mean=mean,
            all_std=std,
            state=state,
            debug=debug,
        )

    # ── 状态查询（DatabaseSummarizer 和日志） ──────────────────────────
    def get_state(self) -> AcquisitionState:
        """
        返回当前 AcquisitionState 快照。

        DatabaseSummarizer 用此快照构造 §7.2 的 <sensitivity> 和 data_card：
          state.grad_psi_at_best → |∂Ψ/∂I₁|, |∂Ψ/∂SOC₁|, |∂Ψ/∂I₂|
          state.mu, state.sigma  → 搜索中心和范围
        """
        if not self._initialized or self._current_theta_best is None:
            raise RuntimeError("AcquisitionFunction 尚未初始化，无状态可查询")
        grad_psi = self._sigma_tracker.get_last_grad()
        return AcquisitionState(
            mu=self._mu_tracker.get_mu(),
            sigma=self._sigma_tracker.get_sigma(),
            alpha_t=self._mu_tracker.get_alpha(self._current_t),
            stagnation_count=0,   # 仅快照，精确值需从 database 获取
            t=self._current_t,
            f_min=self._current_f_min,
            theta_best=self._current_theta_best.copy(),
            grad_psi_at_best=grad_psi.copy(),
        )

    # ── 检查点保存 / 恢复 ──────────────────────────────────────────────
    def save_state(self) -> Dict[str, Any]:
        """序列化 acquisition 状态（用于断点续算）。"""
        return {
            "mu_tracker":    self._mu_tracker.state_dict(),
            "sigma_tracker": self._sigma_tracker.state_dict(),
            "current_t":     self._current_t,
            "current_f_min": self._current_f_min,
            "theta_best":    (self._current_theta_best.tolist()
                              if self._current_theta_best is not None else None),
            "initialized":   self._initialized,
        }

    def load_state(self, d: Dict[str, Any]) -> None:
        """从检查点字典恢复 acquisition 状态。"""
        self._mu_tracker.load_state_dict(d["mu_tracker"])
        self._sigma_tracker.load_state_dict(d["sigma_tracker"])
        self._current_t          = int(d["current_t"])
        self._current_f_min      = float(d["current_f_min"])
        self._current_theta_best = (np.array(d["theta_best"])
                                    if d["theta_best"] is not None else None)
        self._initialized        = bool(d["initialized"])
        logger.info("AcquisitionFunction 从检查点恢复: t=%d", self._current_t)

    # ── 属性访问（供 optimizer.py 直接访问子组件） ─────────────────────
    @property
    def mu_tracker(self) -> SearchMuTracker:
        return self._mu_tracker

    @property
    def sigma_tracker(self) -> SearchSigmaTracker:
        return self._sigma_tracker

    @property
    def ei_calculator(self) -> EICalculator:
        return self._ei_calc

    @property
    def wcharge_calculator(self) -> WChargeCalculator:
        return self._wcharge_calc


# ═══════════════════════════════════════════════════════════════════════════
# §J  工厂函数
# ═══════════════════════════════════════════════════════════════════════════

def build_acquisition_function(
    gp:           GPProtocol,
    psi_fn:       PsiFunction,
    param_bounds: Dict[str, Tuple[float, float]],
    n_select:     int   = 3,
    alpha_max:    float = 0.7,
    alpha_min:    float = 0.05,
    t_decay_alpha:float = 60.0,
    kappa:        float = 0.20,
    eps_sigma:    float = 0.001,
    rho:          float = 0.1,
    xi:           float = 0.0,
    log_mode:     bool  = False,
    max_iterations: int = 20,   # Fix 2: 透传给 AcquisitionFunction
) -> AcquisitionFunction:
    """
    工厂函数：一步构建 AcquisitionFunction。

    典型用法（optimizer.py 初始化阶段）::

        from llmbo.gp_model import build_gp_stack
        from llmbo.acquisition import build_acquisition_function

        psi, coupling, gamma_ann, gp = build_gp_stack(PARAM_BOUNDS)
        af = build_acquisition_function(gp, psi, PARAM_BOUNDS)

        # warmstart 评估完成后（Algorithm 步骤 5）：
        af.initialize(database, llm_prior=llm_interface)

        # 每次迭代（步骤 26-29）：
        gp.fit(X_train, F_tch, w_vec, t=t)          # 先训练 GP
        result = af.step(X_candidates, database, t, w_vec)
        for theta in result.selected_thetas:
            # → PyBaMM 评估

    Parameters
    ----------
    gp, psi_fn, param_bounds : 同 AcquisitionFunction.__init__
    其余参数见 §10 超参数表

    Returns
    -------
    AcquisitionFunction 实例（未初始化，需调用 initialize()）
    """
    return AcquisitionFunction(
        gp=gp,
        psi_fn=psi_fn,
        param_bounds=param_bounds,
        n_select=n_select,
        alpha_max=alpha_max,
        alpha_min=alpha_min,
        t_decay_alpha=t_decay_alpha,
        kappa=kappa,
        eps_sigma=eps_sigma,
        rho=rho,
        xi=xi,
        log_mode=log_mode,
        max_iterations=max_iterations,  # Fix 2
    )


# ═══════════════════════════════════════════════════════════════════════════
# §K  自测（python components/acquisition.py）
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(levelname)s %(name)s: %(message)s"
    )

    BOUNDS = {"I1": (3.0, 7.0), "SOC1": (0.1, 0.7), "I2": (1.0, 5.0)}
    rng = np.random.default_rng(0)

    # ── Mock 对象（代替真实 GP / Database / LLM） ─────────────────────────
    from llmbo.gp_model import build_gp_stack

    class MockDatabase:
        """满足 DatabaseProtocol 的最简 Mock。"""
        def __init__(self, theta_best, f_min, improved=True, stagnation_count=0):
            self._theta_best = np.array(theta_best)
            self._f_min = float(f_min)
            self._improved = improved
            self._stagnation_count = stagnation_count
        def get_f_min(self) -> float:        return self._f_min
        def get_theta_best(self) -> np.ndarray: return self._theta_best.copy()
        def has_improved(self) -> bool:       return self._improved
        def get_stagnation_count(self) -> int: return self._stagnation_count

    class MockLLMPrior:
        """满足 LLMPriorProtocol 的最简 Mock。"""
        def __init__(self, center):
            self._center = np.array(center)
        def get_warmstart_center(self) -> Optional[np.ndarray]:
            return self._center.copy()

    print("=" * 60)
    print("1. DatabaseProtocol / LLMPriorProtocol isinstance 检查")
    print("=" * 60)
    db = MockDatabase([5.0, 0.4, 2.5], f_min=0.45)
    llm_prior = MockLLMPrior([4.5, 0.5, 2.0])
    assert isinstance(db, DatabaseProtocol),       "MockDatabase 不满足 DatabaseProtocol!"
    assert isinstance(llm_prior, LLMPriorProtocol), "MockLLMPrior 不满足 LLMPriorProtocol!"
    print("  ✓ Protocol isinstance 检查通过")

    print("\n" + "=" * 60)
    print("2. SearchMuTracker 自测（Eq.18-19）")
    print("=" * 60)
    mu_tracker = SearchMuTracker(BOUNDS, alpha_max=0.7, alpha_min=0.05, t_decay_alpha=60.0)
    mu_tracker.initialize(np.array([5.0, 0.4, 2.5]), llm_prior=llm_prior)
    print(f"  μ 初始化（LLM prior）= {mu_tracker.get_mu().round(4)}")

    for t in [0, 5, 20, 50]:
        alpha_t = mu_tracker.get_alpha(t)
        print(f"  α(t={t:2d}) = {alpha_t:.4f}")

    theta_new_best = np.array([4.0, 0.55, 1.8])
    mu_tracker.update(theta_new_best, t=1)
    print(f"  μ 更新后（t=1, θ_best={theta_new_best}）= {mu_tracker.get_mu().round(4)}")
    assert mu_tracker.get_alpha(0) > mu_tracker.get_alpha(50), "α 应随 t 递减!"
    print("  ✓ α(0) > α(50) 递减性验证通过")

    print("\n" + "=" * 60)
    print("3. SearchSigmaTracker 自测（Eq.20-22）")
    print("=" * 60)
    psi_fn, coupling_mgr, gamma_ann, gp_model = build_gp_stack(BOUNDS)

    sigma_tracker = SearchSigmaTracker(BOUNDS, psi_fn, kappa=0.20, eps_sigma=0.001, rho=0.1)
    theta_best = np.array([5.0, 0.4, 2.5])
    sigma_tracker.initialize(theta_best)
    sigma0 = sigma_tracker.get_sigma().copy()
    print(f"  σ 初始化 = {sigma0.round(4)}")
    print(f"  ∇Ψ 原始 = {sigma_tracker.get_last_grad().round(2)}")

    # 停滞扩张测试（Eq.22）
    sigma_tracker.update(theta_best, stagnated=True)
    sigma1 = sigma_tracker.get_sigma().copy()
    expected_ratio = 1.1
    actual_ratio = (sigma1 / sigma0).mean()
    print(f"  停滞扩张后 σ = {sigma1.round(4)}")
    print(f"  扩张比例（期望 {expected_ratio:.1f}）= {actual_ratio:.4f}")
    assert abs(actual_ratio - expected_ratio) < 1e-4, f"停滞扩张比例错误: {actual_ratio}"
    print("  ✓ 停滞扩张 Eq.22 验证通过")

    print("\n" + "=" * 60)
    print("4. EICalculator 自测（Eq.15-16）")
    print("=" * 60)
    # 用真实 GP 训练
    n_train = 15
    X_tr = rng.uniform([3.0, 0.1, 1.0], [7.0, 0.7, 5.0], size=(n_train, 3))
    F_tr = rng.uniform(0.0, 1.0, size=n_train)
    w_v  = np.array([0.5, 0.3, 0.2])
    gp_model.fit(X_tr, F_tr, w_v, t=0)

    ei_calc = EICalculator(gp_model, xi=0.0)
    X_cand  = rng.uniform([3.0, 0.1, 1.0], [7.0, 0.7, 5.0], size=(15, 3))
    f_min   = float(F_tr.min())
    ei, mean, std = ei_calc.compute(X_cand, f_min)
    print(f"  EI  in [{ei.min():.3e}, {ei.max():.3e}]")
    print(f"  mean in [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  std  in [{std.min():.4f}, {std.max():.4f}]")
    assert (ei >= 0.0).all(), "EI 含负值!"
    assert (std >= 0.0).all(), "std 含负值!"
    print("  ✓ EI 非负，std 非负")

    print("\n" + "=" * 60)
    print("5. WChargeCalculator 自测（Eq.17）")
    print("=" * 60)
    wcharge_calc = WChargeCalculator(BOUNDS)
    mu_w    = np.array([5.0, 0.4, 2.5])
    sigma_w = np.array([1.0, 0.15, 0.8])
    wcharge = wcharge_calc.compute(X_cand, mu_w, sigma_w)
    print(f"  W_charge in [{wcharge.min():.3e}, {wcharge.max():.3e}]")
    assert (wcharge >= 0.0).all(), "W_charge 含负值!"
    # 验证：候选点最接近 μ 的应该得分最高
    dists = np.linalg.norm((X_cand - mu_w) / sigma_w, axis=1)
    nearest_idx  = int(dists.argmin())
    highest_w_idx = int(wcharge.argmax())
    assert nearest_idx == highest_w_idx, (
        f"最近 μ 的点({nearest_idx})不是 W_charge 最高的({highest_w_idx})!"
    )
    print(f"  ✓ 最近 μ 的候选点（idx={nearest_idx}）得分最高，W_charge 正确")

    print("\n" + "=" * 60)
    print("6. AcquisitionFunction 完整流程自测（Algorithm 步骤 5 + 26-29）")
    print("=" * 60)
    af = build_acquisition_function(gp_model, psi_fn, BOUNDS, n_select=3)

    # 步骤 5：初始化
    af.initialize(db, llm_prior=llm_prior)
    print(f"  初始化后 μ = {af.mu_tracker.get_mu().round(4)}")
    print(f"  初始化后 σ = {af.sigma_tracker.get_sigma().round(4)}")

    # 模拟 3 次迭代
    for iter_t in range(3):
        # 模拟 GP 重新训练（新数据）
        X_tr_new = rng.uniform([3.0, 0.1, 1.0], [7.0, 0.7, 5.0], size=(n_train + iter_t*3, 3))
        F_tr_new = rng.uniform(0.0, 1.0, size=n_train + iter_t*3)
        gp_model.fit(X_tr_new, F_tr_new, w_v, t=iter_t)

        # 模拟 LLM 生成 15 个候选点
        X_cand_iter = rng.uniform([3.0, 0.1, 1.0], [7.0, 0.7, 5.0], size=(15, 3))

        # 模拟停滞（iter 1 停滞）
        mock_db = MockDatabase(
            theta_best=[4.5 - iter_t*0.2, 0.45, 2.2],
            f_min=0.45 - iter_t*0.05,
            improved=(iter_t != 1),
            stagnation_count=1 if iter_t == 1 else 0,
        )

        result = af.step(X_cand_iter, mock_db, t=iter_t, w_vec=w_v)

        print(f"\n  迭代 t={iter_t}:")
        print(f"    μ = {result.state.mu.round(4)}")
        print(f"    σ = {result.state.sigma.round(4)}")
        print(f"    α_t = {result.state.alpha_t:.4f}")
        print(f"    stagnated = {not mock_db.has_improved()}")
        print(f"    top-3 scores α = {result.selected_scores.round(6)}")
        print(f"    top-3 θ:")
        for i, theta in enumerate(result.selected_thetas):
            print(f"      [{i}] I1={theta[0]:.3f} SOC1={theta[1]:.3f} I2={theta[2]:.3f}")

        assert len(result.selected_thetas) == 3, "选出的候选点数量不是 3!"
        assert result.all_alpha.shape == (15,), "α shape 错误!"
        assert (result.all_ei >= 0.0).all(), "EI 含负值!"
        assert (result.all_wcharge >= 0.0).all(), "W_charge 含负值!"

    # 检查点保存 / 恢复
    print("\n  检查点保存 / 恢复测试:")
    ckpt = af.save_state()
    af2  = build_acquisition_function(gp_model, psi_fn, BOUNDS, n_select=3)
    af2.load_state(ckpt)
    assert af2._current_t == af._current_t, "恢复后 t 不一致!"
    print(f"  ✓ 检查点恢复成功: t={af2._current_t}")

    print("\n✓ acquisition.py 全部自测通过")
