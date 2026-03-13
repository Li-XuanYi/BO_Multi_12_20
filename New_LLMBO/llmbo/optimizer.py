"""
optimizer.py — LLAMBO-MO 主优化器
===================================
编排所有子模块，实现完整 Bayesian 优化主循环。

调用顺序（严格按照系统框图 §1-§4）：
  §1  初始化：构建组件 → LLM Touchpoint 1a（耦合矩阵）→ Touchpoint 1b（warm-start）
  §2  Warm-start 评估：PyBaMM 评估所有候选 → 填充 ObservationDB
  §3  采集函数初始化：af.initialize(database, llm_prior=llm)
  §4  主循环（t=0..T）：
        从预生成集合 W 随机选 w_vec →
        update_tchebycheff_context → gp.fit →
        LLM Touchpoint 2 → af.step → PyBaMM 评估 → 记录统计 → 保存检查点

Tchebycheff 公式（按规格文档）：
  f̃₁ = t_charge,  f̃₂ = T_peak,  f̃₃ = log₁₀(ΔQ_aging)        (Eq.2a)
  f̄ᵢ = (f̃ᵢ - f̃ᵢ_min) / (f̃ᵢ_max - f̃ᵢ_min)                 (Eq.2b, 动态更新)
  f_tch = max_i(wᵢ·f̄ᵢ) + η·Σᵢ(wᵢ·f̄ᵢ),   η=0.05             (Eq.1)

权重集合 W：
  通过 Riesz s-energy 最小化在单纯形上预生成 N 个均匀分布的权重向量，
  每次迭代从 W 中随机选取一个，探索不同 Pareto 前沿区域。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from DataBase.database import ObservationDB, DEFAULT_REF_POINT, DEFAULT_BOUNDS
from llmbo.gp_model import build_gp_stack
from llm.llm_interface import build_llm_interface
from llmbo.acquisition import build_acquisition_function
from pybamm_simulator import PyBaMMSimulator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# §A  超参数默认配置
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # ── 实验规模 ──────────────────────────────────────────────────────────
    "max_iterations":   20,
    "n_warmstart":      10,
    "n_candidates":     15,
    "n_select":         1,

    # ── LLM 配置 ──────────────────────────────────────────────────────────
    # FIX: api_base 不含 /chat 后缀；OpenAI SDK 会自动拼接 /chat/completions
    "llm_backend":      "openai",
    "llm_model":        "gpt-4o",
    "llm_api_base":     "https://api.nuwaapi.com/v1",   # ← 修正（去掉 /chat）
    "llm_api_key":      "sk-Sq1zyC8PLM8gafI2fpAccWpzBAzZvuNOPU6ZC9aWA6C883IK",
    "llm_n_samples":    5,
    "llm_temperature":  0.7,

    # ── GP 超参数 ─────────────────────────────────────────────────────────
    "gamma_max":        0.3,
    "gamma_min":        0.05,
    "gamma_t_decay":    20.0,

    # ── Acquisition 超参数 ────────────────────────────────────────────────
    "alpha_max":        0.7,
    "alpha_min":        0.05,
    "t_decay_alpha":    60.0,
    "kappa":            0.20,
    "eps_sigma":        0.001,
    "rho":              0.1,

    # ── Riesz s-energy 权重集合 ───────────────────────────────────────────
    "n_weights":        66,       # 预生成权重向量数量（Das-Dennis n_div=10 → 66 点）
    "riesz_s":          2.0,      # Riesz s 参数（s=2 为球形势能）
    "riesz_n_iter":     500,      # 梯度下降迭代次数
    "riesz_lr":         5e-3,     # 学习率
    "w_sample_seed":    None,     # 随机选取种子（None=随机）
    "riesz_seed":       42,       # 权重集合生成种子（固定保证复现性）

    # ── Tchebycheff 参数 ──────────────────────────────────────────────────
    "eta":              0.05,     # 严格 Pareto 支配 tiebreaker（Eq.1）

    # ── 检查点 ────────────────────────────────────────────────────────────
    "checkpoint_dir":   "checkpoints",
    "checkpoint_every": 5,

    # ── 电池模型 ──────────────────────────────────────────────────────────
    "battery_model":    "LG M50 (Chen2020)",
}


# ═══════════════════════════════════════════════════════════════════════════
# §B  目标变换与归一化工具函数
# ═══════════════════════════════════════════════════════════════════════════

def log_transform_objectives(Y_raw: np.ndarray) -> np.ndarray:
    """
    对原始目标矩阵应用 log₁₀ 变换（仅对 aging 维度）。

    f̃₁ = t_charge      (不变)
    f̃₂ = T_peak        (不变)
    f̃₃ = log₁₀(ΔQ_aging)  (Eq.2a)

    Parameters
    ----------
    Y_raw : (n, 3)  原始目标 [time_s, temp_K, aging_pct]

    Returns
    -------
    Y_tilde : (n, 3)  变换后
    """
    Y_raw = np.atleast_2d(Y_raw).astype(float)
    Y_tilde = Y_raw.copy()
    # aging_pct 可能极小（≈1e-8），clip 避免 log10(0)
    Y_tilde[:, 2] = np.log10(np.maximum(Y_raw[:, 2], 1e-12))
    return Y_tilde


def compute_dynamic_bounds(Y_tilde: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    动态计算每个维度的历史最小/最大值（Eq.2b 分母）。

    Parameters
    ----------
    Y_tilde : (n, 3)  log 变换后的目标

    Returns
    -------
    y_min : (3,)
    y_max : (3,)
    """
    y_min = Y_tilde.min(axis=0)
    y_max = Y_tilde.max(axis=0)
    return y_min, y_max


def normalize_objectives(
    Y_tilde: np.ndarray,
    y_min:   np.ndarray,
    y_max:   np.ndarray,
) -> np.ndarray:
    """
    Min-max 归一化到 [0, 1]（Eq.2b）。

    f̄ᵢ = (f̃ᵢ - f̃ᵢ_min) / (f̃ᵢ_max - f̃ᵢ_min)

    当 max≈min 时（单点情况）置 f̄ᵢ = 0。
    """
    denom = y_max - y_min
    denom = np.where(denom < 1e-10, 1.0, denom)
    return (Y_tilde - y_min) / denom


def compute_tchebycheff(
    Y_bar:  np.ndarray,   # (n, 3)  已归一化目标
    w_vec:  np.ndarray,   # (3,)
    eta:    float = 0.05,
) -> np.ndarray:
    """
    Tchebycheff 标量化（Eq.1）：
      f_tch(θ) = max_i(wᵢ·f̄ᵢ) + η·Σᵢ(wᵢ·f̄ᵢ)

    Parameters
    ----------
    Y_bar : (n, 3)  已归一化目标（通过 normalize_objectives 得到）
    w_vec : (3,)    权重向量（Σwᵢ=1）
    eta   : float   tiebreaker 权重（默认 0.05）

    Returns
    -------
    f_tch : (n,)
    """
    Y_bar  = np.atleast_2d(Y_bar)
    w      = np.asarray(w_vec, dtype=float).ravel()
    Wf     = w[np.newaxis, :] * Y_bar            # (n, 3)
    tch    = Wf.max(axis=1) + eta * Wf.sum(axis=1)  # (n,)
    return tch.squeeze()


def compute_tchebycheff_from_raw(
    Y_raw:  np.ndarray,   # (n, 3) 原始目标
    w_vec:  np.ndarray,   # (3,)
    y_min:  np.ndarray,   # (3,) 动态下界（log 空间）
    y_max:  np.ndarray,   # (3,) 动态上界（log 空间）
    eta:    float = 0.05,
) -> np.ndarray:
    """便捷函数：原始目标 → log 变换 → 归一化 → Tchebycheff。"""
    Y_tilde = log_transform_objectives(Y_raw)
    Y_bar   = normalize_objectives(Y_tilde, y_min, y_max)
    return compute_tchebycheff(Y_bar, w_vec, eta)


# ═══════════════════════════════════════════════════════════════════════════
# §C  Riesz s-energy 权重集合生成
# ═══════════════════════════════════════════════════════════════════════════

def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    将向量 v 投影到概率单纯形 {x ≥ 0, Σx = 1}。
    使用 O(n log n) 排序算法（Duchi et al., 2008）。
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = float(cssv[rho_idx] - 1) / (rho_idx + 1.0)
    return np.maximum(v - theta, 0.0)


def generate_riesz_weight_set(
    n_obj:   int   = 3,
    n_div:   int   = 10,       # Das-Dennis 分割数（n_div=10 → 66 个初始点）
    s:       float = 2.0,      # Riesz s 参数
    n_iter:  int   = 500,      # 梯度下降迭代次数
    lr:      float = 5e-3,     # 学习率
    seed:    int   = 42,
    eps_min: float = 0.01,     # 最小权重分量（避免退化点）
) -> np.ndarray:
    """
    在 (n_obj-1)-单纯形上生成均匀分布的权重向量集合 W。

    算法：
    1. Das-Dennis 均匀权重初始化（保证初始点有规律覆盖）
    2. Riesz s-energy 梯度下降（最小化 Σ_{i≠j} ‖wᵢ-wⱼ‖⁻ˢ）
    3. 每步投影回单纯形，并保证最小分量 ≥ eps_min

    Parameters
    ----------
    n_obj   : 目标维度（默认 3）
    n_div   : Das-Dennis 分割数（决定点集密度；n_div=10, n_obj=3 → 66 点）
    s       : Riesz 势能指数（s=2 等价于 Coulomb 势）
    n_iter  : 梯度下降步数（500 次通常已收敛）
    lr      : 学习率
    seed    : 生成种子（固定保证复现性）
    eps_min : 最小权重分量

    Returns
    -------
    W : (N, n_obj)  N 个权重向量，每行满足 Σ=1, 各分量 ≥ eps_min
    """
    # ── Step 1: Das-Dennis 均匀初始化 ─────────────────────────────────
    from itertools import product as iproduct

    def das_dennis(n_div, n_obj):
        """递归生成 Das-Dennis 权重（n_obj 维）。"""
        if n_obj == 1:
            return [[n_div]]
        pts = []
        for i in range(n_div + 1):
            for rest in das_dennis(n_div - i, n_obj - 1):
                pts.append([i] + rest)
        return pts

    pts = das_dennis(n_div, n_obj)
    W = np.array(pts, dtype=float) / n_div   # (N, n_obj)

    # 确保最小分量
    W = np.maximum(W, eps_min)
    W = W / W.sum(axis=1, keepdims=True)

    N = len(W)
    logger.info("Riesz weight set: %d 初始点（Das-Dennis n_div=%d）", N, n_div)

    # ── Step 2: Riesz s-energy 梯度下降 ──────────────────────────────
    for iteration in range(n_iter):
        # 计算所有点对的梯度
        grad = np.zeros_like(W)          # (N, n_obj)
        for i in range(N):
            diff  = W[i] - W             # (N, n_obj)
            dist2 = np.sum(diff ** 2, axis=1)   # (N,)
            dist2[i] = np.inf            # 排除自身
            # ∂E_s/∂wᵢ = -s · Σ_{j≠i} (wᵢ-wⱼ) / ‖wᵢ-wⱼ‖^(s+2)
            factor = s / (dist2 ** ((s + 2) / 2) + 1e-15)  # (N,)
            factor[i] = 0.0
            grad[i] = np.sum(factor[:, np.newaxis] * diff, axis=0)

        # 梯度步（最大化扩散 → 最小化能量 → 梯度方向为排斥方向，取负号）
        W_new = W + lr * grad

        # 投影回单纯形，并保证最小分量
        for i in range(N):
            W_new[i] = _project_to_simplex(W_new[i])
        W_new = np.maximum(W_new, eps_min)
        W_new = W_new / W_new.sum(axis=1, keepdims=True)
        W = W_new

    logger.info("Riesz weight set 生成完毕: shape=%s", W.shape)
    return W


# ═══════════════════════════════════════════════════════════════════════════
# §D  BayesOptimizer 主类
# ═══════════════════════════════════════════════════════════════════════════

class BayesOptimizer:
    """
    LLAMBO-MO 贝叶斯优化主类。

    使用方式（最简）::

        from llmbo.optimizer import BayesOptimizer
        opt = BayesOptimizer()
        opt.run()
        opt.save_results("results/")

    使用方式（自定义配置）::

        cfg = {
            "llm_backend": "openai",
            "llm_model": "gpt-4o",
            "max_iterations": 30,
            "n_warmstart": 8,
        }
        opt = BayesOptimizer(config=cfg)
        opt.run()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}

        # ── 随机数生成器 ──────────────────────────────────────────────
        seed = self.cfg.get("w_sample_seed")
        self._rng = np.random.default_rng(seed)

        # ── 组件占位 ──────────────────────────────────────────────────
        self.simulator:    Optional[PyBaMMSimulator] = None
        self.database:     Optional[ObservationDB]   = None
        self.llm:          Any                       = None
        self.psi_fn:       Any                       = None
        self.coupling_mgr: Any                       = None
        self.gamma_ann:    Any                       = None
        self.gp:           Any                       = None
        self.af:           Any                       = None

        # ── 预生成 Riesz 权重集合 ─────────────────────────────────────
        logger.info("预生成 Riesz s-energy 权重集合 (n_div=%d, n_iter=%d) ...",
                    self.cfg.get("riesz_n_div", 10),
                    self.cfg["riesz_n_iter"])
        self._weight_set: np.ndarray = generate_riesz_weight_set(
            n_obj   = 3,
            n_div   = self.cfg.get("riesz_n_div", 10),
            s       = self.cfg["riesz_s"],
            n_iter  = self.cfg["riesz_n_iter"],
            lr      = self.cfg["riesz_lr"],
            seed    = self.cfg["riesz_seed"],
        )
        logger.info("权重集合 W: %d 个向量", len(self._weight_set))

        # ── 运行时状态 ────────────────────────────────────────────────
        self._current_iter:  int                  = 0
        self._current_w_vec: Optional[np.ndarray] = None
        # 动态 log 空间 min/max（供 Tchebycheff 归一化）
        self._y_tilde_min: Optional[np.ndarray] = None
        self._y_tilde_max: Optional[np.ndarray] = None

        Path(self.cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # ── §1 初始化 ──────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        初始化所有组件（按框图 §1 顺序）。
        """
        logger.info("=" * 60)
        logger.info("LLAMBO-MO 初始化开始")
        logger.info("=" * 60)

        # 1. PyBaMM 仿真器
        self.simulator = PyBaMMSimulator()
        param_bounds = self.simulator.param_bounds

        # 2. ObservationDB
        self.database = ObservationDB(
            param_bounds=param_bounds,
            ref_point=DEFAULT_REF_POINT,
            normalize=True,
        )

        # 3. LLM 接口
        # FIX: api_base 从 cfg 中取（已在 DEFAULT_CONFIG 修正），
        #      确保 optimizer 和 llm_interface 使用同一 key/url
        self.llm = build_llm_interface(
            param_bounds   = param_bounds,
            backend        = self.cfg["llm_backend"],
            model          = self.cfg["llm_model"],
            api_base       = self.cfg["llm_api_base"],   # "https://api.nuwaapi.com/v1"
            api_key        = self.cfg["llm_api_key"],
            n_samples      = self.cfg["llm_n_samples"],
            temperature    = self.cfg["llm_temperature"],
            battery_model  = self.cfg["battery_model"],
        )

        # 4. Touchpoint 1a：耦合矩阵
        logger.info("Touchpoint 1a: 生成耦合矩阵 ...")
        W_time, W_temp, W_aging = self.llm.generate_coupling_matrices()

        # 5. GP 栈
        self.psi_fn, self.coupling_mgr, self.gamma_ann, self.gp = build_gp_stack(
            param_bounds  = param_bounds,
            gamma_max     = self.cfg["gamma_max"],
            gamma_min     = self.cfg["gamma_min"],
            gamma_t_decay = self.cfg["gamma_t_decay"],
        )
        self.coupling_mgr.set_llm_matrices(W_time, W_temp, W_aging)
        logger.info("耦合矩阵已注入 CouplingMatrixManager")

        # 6. AcquisitionFunction
        self.af = build_acquisition_function(
            gp            = self.gp,
            psi_fn        = self.psi_fn,
            param_bounds  = param_bounds,
            n_select      = self.cfg["n_select"],
            alpha_max     = self.cfg["alpha_max"],
            alpha_min     = self.cfg["alpha_min"],
            t_decay_alpha = self.cfg["t_decay_alpha"],
            kappa         = self.cfg["kappa"],
            eps_sigma     = self.cfg["eps_sigma"],
            rho           = self.cfg["rho"],
        )

        logger.info("所有组件初始化完成")

    # ── §2 Warm-start 评估 ────────────────────────────────────────────────

    def run_warmstart(self) -> None:
        """Touchpoint 1b：评估所有 warm-start 候选点。"""
        logger.info("=" * 60)
        logger.info("§2 Warm-Start 评估阶段 (N_ws=%d)", self.cfg["n_warmstart"])
        logger.info("=" * 60)

        n_ws   = self.cfg["n_warmstart"]
        n_llm  = max(int(n_ws * 0.7), 1)    # 70% 由 LLM 提供（含物理先验）
        n_lhs  = n_ws - n_llm               # 30% 由 LHS 补全（确保空间覆盖）

        llm_candidates = self.llm.generate_warmstart_candidates(n=n_llm)
        lhs_candidates = self._generate_lhs_candidates(n_lhs)
        warmstart_candidates = llm_candidates + lhs_candidates
        logger.info(
            "Warmstart: LLM=%d 个 + LHS=%d 个 = 共 %d 个候选",
            len(llm_candidates), len(lhs_candidates), len(warmstart_candidates)
        )

        for i, theta in enumerate(warmstart_candidates):
            logger.info("  Warm-start [%d/%d]: θ=%s",
                        i + 1, len(warmstart_candidates), theta.round(4))
            t0 = time.perf_counter()
            result = self.simulator.evaluate(theta)
            elapsed = time.perf_counter() - t0

            self.database.add_from_simulator(
                theta=theta, result=result, source="llm", iteration=0
            )

            feasible_str = "✓" if result["feasible"] else f"✗ ({result.get('violation', '?')})"
            logger.info(
                "    → %s  objectives=%s  (%.1fs)",
                feasible_str, np.round(result["raw_objectives"], 4), elapsed
            )

        logger.info("Warm-start 完成: %d 可行解 / %d 总计",
                    self.database.n_feasible, self.database.size)

        # warm-start 后用均匀权重初始化 Tchebycheff 上下文
        # 同时更新动态 min/max 用于 log 空间归一化
        w_init = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        self._update_dynamic_bounds()
        self.database.update_tchebycheff_context(
            w_vec  = w_init,
            y_min  = self._y_tilde_min,
            y_max  = self._y_tilde_max,
            eta    = self.cfg["eta"],
        )

    def _update_dynamic_bounds(self) -> None:
        """
        从可行解历史中更新 log 空间的动态 min/max（Eq.2b 分母）。
        在每次迭代开始时调用，确保归一化基于当前全部历史数据。
        """
        feasible = self.database.get_feasible()
        if not feasible:
            self._y_tilde_min = np.zeros(3)
            self._y_tilde_max = np.ones(3)
            return
        Y_raw   = np.array([o.objectives for o in feasible])  # (n, 3)
        Y_tilde = log_transform_objectives(Y_raw)
        self._y_tilde_min, self._y_tilde_max = compute_dynamic_bounds(Y_tilde)

    def _generate_lhs_candidates(self, n: int) -> list:
        """
        Latin Hypercube Sampling — 覆盖参数空间边角区域。
        用于补充 warmstart，确保 Pareto 极端方向有初始观测。

        Parameters
        ----------
        n : int
            需要生成的候选点数量

        Returns
        -------
        list[np.ndarray]
            物理空间中的候选点列表 (m, 3)
        """
        lo = np.array([self.simulator.param_bounds[k][0] for k in ["I1", "SOC1", "I2"]])
        hi = np.array([self.simulator.param_bounds[k][1] for k in ["I1", "SOC1", "I2"]])
        rng = np.random.default_rng(seed=42)
        # 分层采样：每个维度分 n 段，各取一个
        intervals = np.linspace(0.0, 1.0, n + 1)
        pts = []
        for d in range(3):
            perm = rng.permutation(n)
            lower = intervals[perm]
            upper = intervals[perm + 1]
            samples = lower + rng.random(n) * (upper - lower)
            pts.append(samples)
        X_lhs = np.column_stack(pts)   # (n, 3) in [0,1]³
        return [lo + row * (hi - lo) for row in X_lhs]

    # ── §3 采集函数初始化 ─────────────────────────────────────────────────

    def initialize_acquisition(self) -> None:
        """Algorithm 步骤 5：在 warm-start 后初始化 μ / σ 追踪器。"""
        logger.info("§3 采集函数初始化 ...")
        self.af.initialize(self.database, llm_prior=self.llm)
        state = self.af.get_state()
        logger.info(
            "AcquisitionFunction 就绪: μ=%s  σ=%s",
            state.mu.round(4), state.sigma.round(4)
        )

    # ── §4 主优化循环 ─────────────────────────────────────────────────────

    def run_optimization_loop(self) -> None:
        """
        主 BO 循环（Algorithm §6 步骤 25-35）。

        关键流程：
          1. 从预生成集合 W 随机选取 w_vec（不用 Dirichlet 采样）
          2. 更新动态 min/max，注入 Tchebycheff 上下文
          3. 对可行解应用 log10(aging) + min-max 归一化 + Tchebycheff(Eq.1)
          4. 用归一化标量值训练 GP
          5. LLM Touchpoint 2 生成候选点 → af.step() → PyBaMM 评估
        """
        logger.info("=" * 60)
        logger.info("§4 主优化循环开始 (max_iterations=%d, |W|=%d)",
                    self.cfg["max_iterations"], len(self._weight_set))
        logger.info("=" * 60)

        eta = self.cfg["eta"]

        for t in range(self.cfg["max_iterations"]):
            self._current_iter = t
            iter_start = time.perf_counter()

            logger.info("\n─── 迭代 t=%d ─────────────────────────────────────", t)

            # ── 步骤 1：从预生成集合 W 随机选取 w_vec ─────────────────
            # FIX: 不用 Dirichlet，从 Riesz 集合中随机选一个
            idx_w = int(self._rng.integers(0, len(self._weight_set)))
            w_vec = self._weight_set[idx_w].copy()
            self._current_w_vec = w_vec
            logger.info("  w_vec[%d] = [%.3f, %.3f, %.3f]", idx_w, *w_vec)

            # ── 步骤 2：更新动态 min/max & Tchebycheff 上下文 ─────────
            self._update_dynamic_bounds()
            self.database.update_tchebycheff_context(
                w_vec = w_vec,
                y_min = self._y_tilde_min,
                y_max = self._y_tilde_max,
                eta   = eta,
            )

            # ── 步骤 3：训练 GP ────────────────────────────────────────
            X_train, Y_raw_train = self.database.get_train_XY(
                feasible_only=True, normalize_X=True, normalize_Y=False
            )
            if len(X_train) < 3:
                logger.warning("  可行解不足 3 个，随机采样回退")
                self._evaluate_random_candidates(t, source="random")
                continue

            # 对训练集应用 Eq.2a + Eq.2b + Eq.1
            _, Y_raw_unnorm = self.database.get_train_XY(
                feasible_only=True, normalize_X=False, normalize_Y=False
            )
            F_tch = compute_tchebycheff_from_raw(
                Y_raw  = Y_raw_unnorm,
                w_vec  = w_vec,
                y_min  = self._y_tilde_min,
                y_max  = self._y_tilde_max,
                eta    = eta,
            )
            # GP 训练在归一化的标量空间（zero-mean, unit-var）
            f_mean = float(F_tch.mean())
            f_std  = float(F_tch.std()) + 1e-8
            F_tch_norm = (F_tch - f_mean) / f_std

            self.gp.fit(X_train, F_tch_norm, w_vec, t=t)
            summary = self.gp.training_summary()
            logger.info(
                "  GP 训练完成: n=%d  l=%.4f  γ=%.4f",
                summary["n_train"], summary["l"], summary["gamma"]
            )

            # ── 步骤 4：Touchpoint 2 生成候选点 ─────────────────────
            af_state = self.af.get_state()
            data_summary = self.database.to_llm_context(
                max_observations=20, include_pareto=True, include_top_k=5
            )
            state_dict = {
                "iteration":        t,
                "max_iterations":   self.cfg["max_iterations"],
                "theta_best":       af_state.theta_best,
                "f_min":            af_state.f_min,
                "mu":               af_state.mu,
                "sigma":            af_state.sigma,
                "stagnation_count": af_state.stagnation_count,
                "w_vec":            w_vec,
                "data_summary":     data_summary,
                "sensitivity_info": (
                    f"∂Ψ/∂I₁={af_state.grad_psi_at_best[0]:.3f}, "
                    f"∂Ψ/∂SOC₁={af_state.grad_psi_at_best[1]:.4f}, "
                    f"∂Ψ/∂I₂={af_state.grad_psi_at_best[2]:.3f}"
                ),
            }
            X_candidates = self.llm.generate_iteration_candidates(
                n=self.cfg["n_candidates"],
                state_dict=state_dict,
            )
            logger.info("  LLM 生成 %d 个候选点", X_candidates.shape[0])

            # 归一化候选点（GP 在归一化 θ 空间预测）
            lo = np.array([self.simulator.param_bounds[k][0] for k in ["I1", "SOC1", "I2"]])
            hi = np.array([self.simulator.param_bounds[k][1] for k in ["I1", "SOC1", "I2"]])
            X_cand_norm = (X_candidates - lo) / (hi - lo + 1e-12)

            # ── 步骤 5：af.step() 选 top-k ────────────────────────────
            # f_min 需要与 GP 预测（归一化 F_tch）同量纲
            f_min_normalized = float((self.database.get_f_min() - f_mean) / f_std)
            db_proxy = _DBProxy(self.database, f_min_override=f_min_normalized)

            result_af = self.af.step(
                X_candidates=X_cand_norm,
                database=db_proxy,
                t=t,
                w_vec=w_vec,
            )
            logger.info(
                "  top-%d α 分值: %s",
                len(result_af.selected_thetas),
                result_af.selected_scores.round(6)
            )

            # ── 步骤 6：PyBaMM 评估 top-k ────────────────────────────
            n_new = 0
            for rank, sel_idx in enumerate(result_af.selected_indices):
                theta_orig = X_candidates[sel_idx]    # 原始物理空间
                logger.info(
                    "  评估候选 [rank=%d]: I1=%.3f  SOC1=%.3f  I2=%.3f",
                    rank, *theta_orig
                )
                t_eval = time.perf_counter()
                sim_result = self.simulator.evaluate(theta_orig)
                elapsed_eval = time.perf_counter() - t_eval

                self.database.add_from_simulator(
                    theta      = theta_orig,
                    result     = sim_result,
                    source     = "llm_gp",
                    iteration  = t + 1,
                    acq_value  = float(result_af.selected_scores[rank]),
                    acq_type   = "EI_Wcharge",
                    gp_pred    = {
                        "mean": float(result_af.all_mean[sel_idx]),
                        "std":  float(result_af.all_std[sel_idx]),
                    },
                )
                n_new += 1
                feasible_str = "✓" if sim_result["feasible"] else "✗"
                logger.info(
                    "    → %s  obj=%s  (%.1fs)",
                    feasible_str,
                    np.round(sim_result["raw_objectives"], 4),
                    elapsed_eval
                )

            # ── 步骤 7：记录统计，保存检查点 ─────────────────────────
            iter_elapsed = time.perf_counter() - iter_start
            hv = self.database.compute_hypervolume()
            self.database.record_iteration_stats(extra={
                "t":           t,
                "w_vec":       w_vec.tolist(),
                "n_new_evals": n_new,
                "iter_time_s": round(iter_elapsed, 2),
            })
            logger.info(
                "  迭代 t=%d 完成: HV=%.6f  |PF|=%d  总评估=%d  (%.1fs)",
                t, hv, self.database.pareto_size, self.database.size, iter_elapsed
            )

            if (t + 1) % self.cfg["checkpoint_every"] == 0:
                self._save_checkpoint(t)

    def _evaluate_random_candidates(self, t: int, source: str = "random") -> None:
        """可行解不足时的随机采样回退。"""
        bounds = self.simulator.param_bounds
        lo = np.array([bounds[k][0] for k in ["I1", "SOC1", "I2"]])
        hi = np.array([bounds[k][1] for k in ["I1", "SOC1", "I2"]])
        for _ in range(self.cfg["n_select"]):
            theta = self._rng.uniform(lo, hi)
            result = self.simulator.evaluate(theta)
            self.database.add_from_simulator(
                theta=theta, result=result, source=source, iteration=t + 1
            )

    # ── 公开入口 ─────────────────────────────────────────────────────────

    def run(self) -> ObservationDB:
        """完整运行：setup → warm-start → 初始化采集函数 → 主循环。"""
        self.setup()
        self.run_warmstart()
        self.initialize_acquisition()
        self.run_optimization_loop()
        logger.info(
            "\n优化完成！总评估: %d  最终 HV: %.6f",
            self.database.size, self.database.compute_hypervolume()
        )
        return self.database

    # ── 结果保存 ─────────────────────────────────────────────────────────

    def save_results(self, output_dir: str = "results") -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.database.save(str(Path(output_dir) / "database.json"))
        logger.info("结果已保存至: %s", output_dir)

    # ── 检查点 ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, t: int) -> None:
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        self.database.save(str(ckpt_dir / f"db_t{t:04d}.json"))
        af_state = self.af.save_state()
        with open(ckpt_dir / f"af_t{t:04d}.json", "w", encoding="utf-8") as f:
            json.dump(af_state, f, indent=2)
        summary = {
            "t":           t,
            "n_total":     self.database.size,
            "hv":          self.database.compute_hypervolume(),
            "pareto_size": self.database.pareto_size,
            "config":      {k: v for k, v in self.cfg.items()
                            if isinstance(v, (int, float, str, bool, type(None)))},
        }
        with open(ckpt_dir / f"summary_t{t:04d}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("  检查点已保存: t=%d", t)


# ═══════════════════════════════════════════════════════════════════════════
# §E  _DBProxy — 临时 f_min 覆盖代理
# ═══════════════════════════════════════════════════════════════════════════

class _DBProxy:
    """
    ObservationDB 的轻量级代理。

    GP 训练在归一化 F_tch 空间，但 database._f_min 存储原始 Tchebycheff 值。
    EI 计算中 f_min 需要与 GP 预测在同一量纲，因此用归一化值覆盖 get_f_min()。
    其余方法全部转发给真实 database。
    """

    def __init__(self, db: ObservationDB, f_min_override: float):
        self._db = db
        self._f_min_override = f_min_override

    def get_f_min(self) -> float:
        return self._f_min_override

    def get_theta_best(self):
        return self._db.get_theta_best()

    def has_improved(self) -> bool:
        return self._db.has_improved()

    def get_stagnation_count(self) -> int:
        return self._db.get_stagnation_count()


# ═══════════════════════════════════════════════════════════════════════════
# §F  CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="LLAMBO-MO 锂电池快充优化")
    parser.add_argument("--backend",    default="openai",    help="LLM 后端")
    parser.add_argument("--model",      default="gpt-4o",    help="LLM 模型")
    parser.add_argument("--api-base",   default="https://api.nuwaapi.com/v1", help="API 地址")
    parser.add_argument("--api-key",    default="sk-Sq1zyC8PLM8gafI2fpAccWpzBAzZvuNOPU6ZC9aWA6C883IK", help="API Key")
    parser.add_argument("--iters",      type=int, default=50, help="优化迭代次数")
    parser.add_argument("--warmstart",  type=int, default=10, help="Warm-start 点数")
    parser.add_argument("--candidates", type=int, default=15, help="每迭代候选点数")
    parser.add_argument("--output",     default="results",   help="结果输出目录")
    args = parser.parse_args()

    cfg = {
        "llm_backend":    args.backend,
        "llm_model":      args.model,
        "llm_api_base":   args.api_base,
        "llm_api_key":    args.api_key,
        "max_iterations": args.iters,
        "n_warmstart":    args.warmstart,
        "n_candidates":   args.candidates,
    }

    opt = BayesOptimizer(config=cfg)
    db  = opt.run()
    opt.save_results(args.output)
    print("\n" + db.summary())