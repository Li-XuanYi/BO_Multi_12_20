"""
parego_optimizer.py — ParEGO 对比实验优化器
=============================================
ParEGO (Pareto Efficient Global Optimization) 实现，作为 LLAMBO-MO 的对比基线。

与 LLAMBO-MO 保持一致的部分：
  - 相同的 ObservationDB（独立实例）
  - 相同的 HV 计算方式（含 log₁₀(aging) 变换）
  - 相同的 Tchebycheff 标量化函数（Eq.1）
  - 相同的动态 min-max 归一化（Eq.2b）
  - 相同的 PyBaMM 仿真器
  - 相同的 Riesz s-energy 权重集合（每迭代随机选取）

与 LLAMBO-MO 不同的部分：
  - 初始化：使用 Latin Hypercube Sampling（无 LLM warmstart）
  - 代理模型：sklearn GaussianProcessRegressor（标准 Matérn 5/2 核）
  - 采集函数：标准 EI（无 W_charge 加权）
  - 候选点生成：在参数空间内均匀随机采样（无 LLM 引导）
  - 无物理先验、无耦合矩阵、无 μ/σ 动态追踪

算法流程：
  §1  LHS 初始化 (N_ws 个点) → PyBaMM 评估 → 填充 ObservationDB
  §2  主循环 (t = 0..T)：
        a. 随机选取 w_vec（从 Riesz 集合）
        b. 更新动态 min/max，Tchebycheff 标量化
        c. 训练 sklearn GP
        d. 随机生成候选点 → EI 评分 → 选最优点
        e. PyBaMM 评估 → 更新 ObservationDB
  §3  返回 ObservationDB（与 LLAMBO-MO 结果对比）

对外接口：
  ParEGOOptimizer(config)
  ParEGOOptimizer.run() → ObservationDB
  ParEGOOptimizer.save_results(output_dir)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm as scipy_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from DataBase.database import ObservationDB, DEFAULT_REF_POINT, DEFAULT_BOUNDS
from pybamm_simulator import PyBaMMSimulator

# 复用 optimizer.py 中的工具函数（Tchebycheff 相关）
from llmbo.optimizer import (
    log_transform_objectives,
    compute_dynamic_bounds,
    normalize_objectives,
    compute_tchebycheff,
    compute_tchebycheff_from_raw,
    generate_riesz_weight_set,
    _project_to_simplex,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# §A  默认配置
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PAREGO_CONFIG: Dict[str, Any] = {
    # ── 实验规模 ──────────────────────────────────────────────────────────
    "max_iterations":    300,     # 300 回合长程优化
    "n_warmstart":       15,      # LHS 初始点数量（增加以适应长程运行）
    "n_random_cands":    500,     # 每迭代随机候选点数（EI 优化用）

    # ── Riesz 权重集合（与 LLAMBO-MO 完全一致） ───────────────────────────
    "riesz_n_div":       10,
    "riesz_s":           2.0,
    "riesz_n_iter":      500,
    "riesz_lr":          5e-3,
    "riesz_seed":        42,      # 固定保证权重集合与 LLAMBO-MO 一致

    # ── Tchebycheff 参数（与 LLAMBO-MO 完全一致） ─────────────────────────
    "eta":               0.05,

    # ── sklearn GP 超参数 ─────────────────────────────────────────────────
    "gp_n_restarts":     5,       # GP 优化重启次数（防止局部最优）
    "gp_normalize_y":    True,    # 是否对 y 做标准化（GP 内部）

    # ── EI 参数 ───────────────────────────────────────────────────────────
    "xi":                0.0,     # EI 探索奖励（0 = 纯利用）

    # ── 随机种子 ──────────────────────────────────────────────────────────
    "lhs_seed":          0,       # LHS 初始化种子
    "w_sample_seed":     None,    # 权重采样种子（None = 随机）
    "cand_seed":         None,    # 候选点采样种子（None = 随机）

    # ── 检查点 ────────────────────────────────────────────────────────────
    "checkpoint_dir":    "checkpoints_parego",
    "checkpoint_every":  10,      # 每 10 回合保存（300 回合→30 个检查点）

    # ── 电池模型（与 LLAMBO-MO 一致） ─────────────────────────────────────
    "battery_model":     "LG M50 (Chen2020)",
}


# ═══════════════════════════════════════════════════════════════════════════
# §B  Latin Hypercube Sampling
# ═══════════════════════════════════════════════════════════════════════════

def lhs_sampling(
    param_bounds: Dict[str, Tuple[float, float]],
    n:            int,
    seed:         int = 0,
) -> np.ndarray:
    """
    Latin Hypercube Sampling，生成 n 个均匀分布的初始点。

    Parameters
    ----------
    param_bounds : {"I1": (lo, hi), "SOC1": (lo, hi), "I2": (lo, hi)}
    n            : 采样点数
    seed         : 随机种子

    Returns
    -------
    X : (n, 3)  原始物理空间的候选点
    """
    rng = np.random.default_rng(seed)
    keys = ["I1", "SOC1", "I2"]
    d = len(keys)
    lo = np.array([param_bounds[k][0] for k in keys])
    hi = np.array([param_bounds[k][1] for k in keys])

    # LHS 核心：每维分 n 等份，各取一个随机点，再对各维做随机排列
    cut = np.linspace(0, 1, n + 1)
    u = np.zeros((n, d))
    for j in range(d):
        u[:, j] = rng.uniform(cut[:-1], cut[1:])  # 每格随机取一点
        u[:, j] = rng.permutation(u[:, j])         # 随机打乱

    # 映射到参数空间
    X = lo + u * (hi - lo)
    return X


# ═══════════════════════════════════════════════════════════════════════════
# §C  标准 EI 计算器（无 W_charge）
# ═══════════════════════════════════════════════════════════════════════════

class StandardEICalculator:
    """
    标准期望改进（EI）计算器。

    与 LLAMBO-MO 的 EICalculator 接口兼容，但不依赖 GPProtocol；
    直接接收 sklearn GP 的预测结果。

    EI(θ) = (f_min − μ(θ) − ξ)·Φ(z) + σ(θ)·φ(z)
    z     = (f_min − μ(θ) − ξ) / σ(θ)
    """

    def __init__(self, xi: float = 0.0):
        """
        Parameters
        ----------
        xi : float  探索奖励（默认 0，纯利用）
        """
        self.xi = float(xi)

    def compute(
        self,
        mean:  np.ndarray,   # (m,)  GP 后验均值
        std:   np.ndarray,   # (m,)  GP 后验标准差
        f_min: float,        # 当前最优标量值
    ) -> np.ndarray:
        """
        批量计算 EI。

        Parameters
        ----------
        mean  : (m,)  GP 预测均值
        std   : (m,)  GP 预测标准差
        f_min : float  当前历史最优标量值

        Returns
        -------
        ei : (m,)  EI 值（非负）
        """
        std = np.maximum(std, 1e-12)  # 数值保护
        improvement = f_min - mean - self.xi

        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(std > 1e-12, improvement / std, 0.0)

        ei = improvement * scipy_norm.cdf(z) + std * scipy_norm.pdf(z)
        return np.maximum(ei, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# §D  ParEGO sklearn GP 代理模型
# ═══════════════════════════════════════════════════════════════════════════

class ParEGOSurrogate:
    """
    ParEGO 使用的 sklearn GP 代理模型。

    核函数：ConstantKernel × Matérn(ν=5/2) + WhiteKernel
    - Matérn 5/2 是 GP 优化领域最常用的核（比 RBF 更鲁棒）
    - WhiteKernel 处理观测噪声
    - ConstantKernel 学习信号幅度

    接口与 LLAMBO-MO 的 PhysicsGPModel 保持功能一致：
      fit(X, F_tch) → self
      predict(X_new) → (mean, std)
      training_summary() → dict
    """

    def __init__(
        self,
        n_restarts: int  = 5,
        normalize_y: bool = True,
        seed:        int  = 0,
    ):
        """
        Parameters
        ----------
        n_restarts  : GP 优化重启次数
        normalize_y : 是否标准化 y（GP 内部，建议 True）
        seed        : 随机种子
        """
        self.n_restarts  = n_restarts
        self.normalize_y = normalize_y
        self.seed        = seed

        self._gp: Optional[GaussianProcessRegressor] = None
        self._n_train: int = 0

    def _build_kernel(self):
        """构建 Matérn 5/2 核（带幅度和噪声项）。"""
        # ConstantKernel：信号幅度（1.0，范围 0.01~100）
        # Matérn(ν=5/2)：平滑度合理，长度尺度从 [0.1, 5.0] 优化
        # WhiteKernel：观测噪声（1e-4，范围 1e-8~1e-1）
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(0.01, 100.0)) *
            Matern(
                length_scale=1.0,
                length_scale_bounds=(0.1, 10.0),
                nu=2.5,
            ) +
            WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 0.1))
        )
        return kernel

    def fit(self, X: np.ndarray, F_tch: np.ndarray) -> "ParEGOSurrogate":
        """
        训练 GP。

        Parameters
        ----------
        X      : (n, 3)  归一化到 [0,1] 的决策向量
        F_tch  : (n,)    Tchebycheff 标量目标（动态归一化空间）
        """
        self._gp = GaussianProcessRegressor(
            kernel=self._build_kernel(),
            n_restarts_optimizer=self.n_restarts,
            normalize_y=self.normalize_y,
            random_state=self.seed,
        )
        self._gp.fit(X, F_tch)
        self._n_train = X.shape[0]

        logger.debug(
            "ParEGOSurrogate.fit: n=%d  kernel=%s",
            self._n_train, self._gp.kernel_
        )
        return self

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GP 后验预测。

        Returns
        -------
        mean : (m,)
        std  : (m,)  >= 0
        """
        if self._gp is None:
            raise RuntimeError("ParEGOSurrogate 尚未训练，请先调用 fit()")
        X_new = np.atleast_2d(X_new)
        mean, std = self._gp.predict(X_new, return_std=True)
        std = np.maximum(std, 0.0)
        return mean.ravel(), std.ravel()

    def training_summary(self) -> Dict[str, Any]:
        """返回训练状态摘要（供日志）。"""
        if self._gp is None:
            return {"fitted": False, "n_train": 0}
        return {
            "fitted":       True,
            "n_train":      self._n_train,
            "kernel":       str(self._gp.kernel_),
            "log_marginal": float(self._gp.log_marginal_likelihood_value_),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §E  ParEGOOptimizer 主类
# ═══════════════════════════════════════════════════════════════════════════

class ParEGOOptimizer:
    """
    ParEGO 多目标贝叶斯优化器。

    与 LLAMBO-MO BayesOptimizer 保持相同的外部接口：
      run() → ObservationDB
      save_results(output_dir)

    关键设计决策：
    ──────────────
    1. 权重集合与 LLAMBO-MO 完全相同（riesz_seed=42），保证公平比较。
    2. HV 计算、log 变换、归一化与 LLAMBO-MO 完全一致（复用 database.py）。
    3. 候选点完全随机（参数空间均匀采样），无任何 LLM 或物理先验引导。
    4. GP 使用 sklearn，无物理复合核（纯数据驱动）。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = {**DEFAULT_PAREGO_CONFIG, **(config or {})}

        # 随机数生成器
        w_seed = self.cfg["w_sample_seed"]
        cand_seed = self.cfg["cand_seed"]
        self._rng_w    = np.random.default_rng(w_seed)
        self._rng_cand = np.random.default_rng(cand_seed)

        # 组件占位
        self.simulator: Optional[PyBaMMSimulator] = None
        self.database:  Optional[ObservationDB]   = None
        self._surrogate: Optional[ParEGOSurrogate] = None
        self._ei_calc:   Optional[StandardEICalculator] = None

        # 动态 log 空间 min/max
        self._y_tilde_min: Optional[np.ndarray] = None
        self._y_tilde_max: Optional[np.ndarray] = None

        # 预生成 Riesz 权重集合（与 LLAMBO-MO 完全一致）
        logger.info(
            "ParEGO: 预生成 Riesz 权重集合 (n_div=%d, n_iter=%d, seed=%d) ...",
            self.cfg["riesz_n_div"],
            self.cfg["riesz_n_iter"],
            self.cfg["riesz_seed"],
        )
        self._weight_set: np.ndarray = generate_riesz_weight_set(
            n_obj   = 3,
            n_div   = self.cfg["riesz_n_div"],
            s       = self.cfg["riesz_s"],
            n_iter  = self.cfg["riesz_n_iter"],
            lr      = self.cfg["riesz_lr"],
            seed    = self.cfg["riesz_seed"],
        )
        logger.info("ParEGO: 权重集合 |W|=%d", len(self._weight_set))

        # 检查点目录
        Path(self.cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # ── §1 初始化 ──────────────────────────────────────────────────────────

    def setup(self) -> None:
        """初始化仿真器、数据库、代理模型。"""
        logger.info("=" * 60)
        logger.info("ParEGO 初始化开始")
        logger.info("=" * 60)

        # 1. PyBaMM 仿真器（与 LLAMBO-MO 相同）
        self.simulator = PyBaMMSimulator()

        # 2. ObservationDB（独立实例，与 LLAMBO-MO 隔离）
        self.database = ObservationDB(
            param_bounds=self.simulator.param_bounds,
            ref_point=DEFAULT_REF_POINT,
            normalize=True,
        )

        # 3. 代理模型和 EI
        self._surrogate = ParEGOSurrogate(
            n_restarts  = self.cfg["gp_n_restarts"],
            normalize_y = self.cfg["gp_normalize_y"],
            seed        = 0,
        )
        self._ei_calc = StandardEICalculator(xi=self.cfg["xi"])

        logger.info("ParEGO: 所有组件初始化完成")

    # ── §2 LHS 初始化评估 ─────────────────────────────────────────────────

    def run_warmstart(self) -> None:
        """
        LHS 采样 N_ws 个初始点并评估。

        与 LLAMBO-MO 的 run_warmstart 对应，但用 LHS 替换 LLM 候选。
        评估次数相同，保证后续 HV 曲线在相同评估预算下对比。
        """
        logger.info("=" * 60)
        logger.info("ParEGO §2 LHS 初始化 (N_ws=%d)", self.cfg["n_warmstart"])
        logger.info("=" * 60)

        X_init = lhs_sampling(
            self.simulator.param_bounds,
            n    = self.cfg["n_warmstart"],
            seed = self.cfg["lhs_seed"],
        )

        for i, theta in enumerate(X_init):
            logger.info(
                "  LHS [%d/%d]: I1=%.3f  SOC1=%.3f  I2=%.3f",
                i + 1, len(X_init), theta[0], theta[1], theta[2]
            )
            t0 = time.perf_counter()
            result = self.simulator.evaluate(theta)
            elapsed = time.perf_counter() - t0

            self.database.add_from_simulator(
                theta=theta, result=result, source="lhs", iteration=0
            )

            feasible_str = "✓" if result["feasible"] else f"✗ ({result.get('violation', '?')})"
            logger.info(
                "    → %s  objectives=%s  (%.1fs)",
                feasible_str,
                np.round(result["raw_objectives"], 4),
                elapsed,
            )

        logger.info(
            "ParEGO LHS 完成: %d 可行解 / %d 总计",
            self.database.n_feasible, self.database.size
        )

        # 初始化 Tchebycheff 上下文（均匀权重）
        w_init = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        self._update_dynamic_bounds()
        self.database.update_tchebycheff_context(
            w_vec = w_init,
            y_min = self._y_tilde_min,
            y_max = self._y_tilde_max,
            eta   = self.cfg["eta"],
        )

    def _update_dynamic_bounds(self) -> None:
        """
        从可行解历史更新 log 空间动态 min/max。

        与 LLAMBO-MO 的 _update_dynamic_bounds 完全相同，保证归一化一致。
        """
        feasible = self.database.get_feasible()
        if not feasible:
            self._y_tilde_min = np.zeros(3)
            self._y_tilde_max = np.ones(3)
            return
        Y_raw   = np.array([o.objectives for o in feasible])
        Y_tilde = log_transform_objectives(Y_raw)
        self._y_tilde_min, self._y_tilde_max = compute_dynamic_bounds(Y_tilde)

    # ── §3 主优化循环 ─────────────────────────────────────────────────────

    def run_optimization_loop(self) -> None:
        """
        ParEGO 主循环。

        每次迭代：
          1. 从 Riesz 集合随机选 w_vec（与 LLAMBO-MO 一致）
          2. 更新动态 min/max & Tchebycheff 标量化
          3. 训练 sklearn GP
          4. 随机生成候选点 → EI 评分 → 选最优
          5. PyBaMM 评估 → 更新 DB
        """
        logger.info("=" * 60)
        logger.info(
            "ParEGO §3 主循环 (max_iterations=%d, |W|=%d, n_cands=%d)",
            self.cfg["max_iterations"],
            len(self._weight_set),
            self.cfg["n_random_cands"],
        )
        logger.info("=" * 60)

        eta = self.cfg["eta"]
        param_bounds = self.simulator.param_bounds
        lo = np.array([param_bounds[k][0] for k in ["I1", "SOC1", "I2"]])
        hi = np.array([param_bounds[k][1] for k in ["I1", "SOC1", "I2"]])

        for t in range(self.cfg["max_iterations"]):
            iter_start = time.perf_counter()
            logger.info("\n─── ParEGO 迭代 t=%d ─────────────────────────────", t)

            # ── 步骤 1：从 Riesz 集合随机选 w_vec ─────────────────────
            idx_w = int(self._rng_w.integers(0, len(self._weight_set)))
            w_vec = self._weight_set[idx_w].copy()
            logger.info("  w_vec[%d] = [%.3f, %.3f, %.3f]", idx_w, *w_vec)

            # ── 步骤 2：更新动态归一化 & Tchebycheff 上下文 ────────────
            self._update_dynamic_bounds()
            self.database.update_tchebycheff_context(
                w_vec = w_vec,
                y_min = self._y_tilde_min,
                y_max = self._y_tilde_max,
                eta   = eta,
            )

            # ── 步骤 3：训练 sklearn GP ────────────────────────────────
            # 获取归一化 X（[0,1] 空间）
            X_train_norm, Y_raw_train = self.database.get_train_XY(
                feasible_only=True, normalize_X=True, normalize_Y=False
            )
            if len(X_train_norm) < 3:
                logger.warning("  可行解不足 3 个，随机评估回退")
                self._evaluate_random(t)
                continue

            # 计算 Tchebycheff 标量目标（原始空间 Y，log 变换 + 归一化）
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

            # 标准化 F_tch（减均值除标准差，保持与 LLAMBO-MO GP 训练一致）
            f_mean = float(F_tch.mean())
            f_std  = float(F_tch.std()) + 1e-8
            F_tch_norm = (F_tch - f_mean) / f_std
            f_min_normalized = float(F_tch_norm.min())

            # 训练 GP
            self._surrogate.fit(X_train_norm, F_tch_norm)
            summary = self._surrogate.training_summary()
            logger.info(
                "  GP 训练完成: n=%d  log_marginal=%.4f",
                summary["n_train"],
                summary.get("log_marginal", float("nan")),
            )

            # ── 步骤 4：随机生成候选点 → EI 评分 → 选最优 ────────────
            n_cands = self.cfg["n_random_cands"]

            # 完全随机均匀采样（无 LLM / 物理先验引导）
            X_cands_raw = self._rng_cand.uniform(lo, hi, size=(n_cands, 3))

            # 归一化候选点（GP 在 [0,1] 空间预测）
            X_cands_norm = (X_cands_raw - lo) / (hi - lo + 1e-12)

            # GP 预测
            mean, std = self._surrogate.predict(X_cands_norm)

            # EI 评分
            ei = self._ei_calc.compute(mean, std, f_min_normalized)

            # 选 EI 最大的候选点
            best_idx = int(np.argmax(ei))
            theta_next = X_cands_raw[best_idx]

            logger.info(
                "  EI 最优点: I1=%.3f  SOC1=%.3f  I2=%.3f  EI=%.6f",
                theta_next[0], theta_next[1], theta_next[2], float(ei[best_idx])
            )

            # ── 步骤 5：PyBaMM 评估 ──────────────────────────────────
            t_eval = time.perf_counter()
            sim_result = self.simulator.evaluate(theta_next)
            elapsed_eval = time.perf_counter() - t_eval

            self.database.add_from_simulator(
                theta      = theta_next,
                result     = sim_result,
                source     = "pareto_ei",
                iteration  = t + 1,
                acq_value  = float(ei[best_idx]),
                acq_type   = "EI",
                gp_pred    = {
                    "mean": float(mean[best_idx]),
                    "std":  float(std[best_idx]),
                },
            )

            feasible_str = "✓" if sim_result["feasible"] else "✗"
            logger.info(
                "    → %s  obj=%s  (%.1fs)",
                feasible_str,
                np.round(sim_result["raw_objectives"], 4),
                elapsed_eval,
            )

            # ── 步骤 6：记录统计，保存检查点 ─────────────────────────
            iter_elapsed = time.perf_counter() - iter_start
            hv = self.database.compute_hypervolume()

            # 记录迭代统计
            self.database.record_iteration_stats(extra={
                "t":           t,
                "w_vec":       w_vec.tolist(),
                "n_new_evals": 1,
                "iter_time_s": round(iter_elapsed, 2),
                "best_ei":     float(ei[best_idx]),
            })

            # 保存当前权重向量供检查点使用
            self._current_w_vec = w_vec

            logger.info(
                "  迭代 t=%d 完成: HV=%.6f  |PF|=%d  总评估=%d  (%.1fs)",
                t, hv, self.database.pareto_size, self.database.size, iter_elapsed,
            )

            if (t + 1) % self.cfg["checkpoint_every"] == 0:
                self._save_checkpoint(t)

    def _evaluate_random(self, t: int) -> None:
        """可行解不足时随机评估一个点（回退策略）。"""
        bounds = self.simulator.param_bounds
        lo = np.array([bounds[k][0] for k in ["I1", "SOC1", "I2"]])
        hi = np.array([bounds[k][1] for k in ["I1", "SOC1", "I2"]])
        theta = self._rng_cand.uniform(lo, hi)
        result = self.simulator.evaluate(theta)
        self.database.add_from_simulator(
            theta=theta, result=result, source="random", iteration=t + 1
        )

    # ── 公开入口 ─────────────────────────────────────────────────────────

    def run(self) -> ObservationDB:
        """完整运行：setup → LHS warmstart → 主循环。"""
        self.setup()
        self.run_warmstart()
        self.run_optimization_loop()

        logger.info(
            "\nParEGO 优化完成！总评估: %d  最终 HV: %.6f",
            self.database.size, self.database.compute_hypervolume()
        )
        return self.database

    # ── 结果保存 ─────────────────────────────────────────────────────────

    def save_results(self, output_dir: str = "results_pareto") -> None:
        """保存 ObservationDB 和统计信息到 output_dir。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.database.save(str(out / "pareto_database.json"))

        # 保存 HV 收敛曲线（供绘图）
        hv_trace = self.database.get_hv_trace()
        stats = self.database.get_iteration_stats()

        summary = {
            "algorithm":      "ParEGO",
            "n_total":        self.database.size,
            "final_hv":       self.database.compute_hypervolume(),
            "pareto_size":    self.database.pareto_size,
            "hv_trace":       hv_trace.tolist(),
            "config":         {
                k: v for k, v in self.cfg.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
        }
        with open(out / "pareto_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("ParEGO 结果已保存至: %s", output_dir)


    def save_final_summary(self, output_dir: str = "results_parego") -> None:
        """Save final complete summary file for plotting"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stats = self.database.get_iteration_stats()
        hv_trace = self.database.get_hv_trace()
        y_stats = self.database.get_Y_stats(feasible_only=True)

        # Get best values per objective with corresponding parameters
        best_per_obj = self.database.get_best_per_objective()
        best_objectives = {}
        obj_names = ["time_s", "temp_K", "aging_pct"]
        for i, name in enumerate(obj_names):
            if name in best_per_obj:
                obs = best_per_obj[name]
                best_objectives[name] = {
                    "value": float(obs.objectives[i]),
                    "theta": obs.theta.tolist(),
                    "I1": float(obs.theta[0]),
                    "SOC1": float(obs.theta[1]),
                    "I2": float(obs.theta[2]),
                }

        final_summary = {
            "algorithm":         "ParEGO",
            "total_iterations":  self.cfg["max_iterations"],
            "n_warmstart":       self.cfg["n_warmstart"],
            "final_hv":          float(self.database.compute_hypervolume()),
            "hv_trace":          hv_trace.tolist(),
            "iteration_stats":   stats,
            "pareto_front":      [o.to_dict() for o in self.database.get_pareto_front()],
            "best_per_objective": best_objectives,
            "best_objectives_raw": {
                "time_s":  float(y_stats["min"][0]),
                "temp_K":  float(y_stats["min"][1]),
                "aging_pct": float(y_stats["min"][2]),
            },
            "config": {
                k: v for k, v in self.cfg.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
        }

        with open(out / "parego_final_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)

        logger.info("ParEGO final summary saved to: %s/parego_final_summary.json", output_dir)

    # ── 检查点 ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, t: int) -> None:
        """保存完整检查点，格式与 LLAMBO 一致"""
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存完整数据库
        self.database.save(str(ckpt_dir / f"pareto_db_t{t:04d}.json"))

        # 2. 保存摘要文件（含 HV 曲线、权重向量，供制图使用）
        y_stats = self.database.get_Y_stats(feasible_only=True)
        hv_trace = self.database.get_hv_trace()

        summary = {
            "iteration":    t + 1,
            "n_total":      self.database.size,
            "n_feasible":   self.database.n_feasible,
            "pareto_size":  self.database.pareto_size,
            "hypervolume":  float(self.database.compute_hypervolume()),
            "hypervolume_raw": float(self.database.compute_hypervolume_raw()),
            "best_objectives": {
                "time_s":  float(y_stats["min"][0]),
                "temp_K":  float(y_stats["min"][1]),
                "aging_pct": float(y_stats["min"][2]),
            },
            "hv_trace":     hv_trace.tolist(),
            "w_vec":        self._current_w_vec.tolist() if hasattr(self, '_current_w_vec') and self._current_w_vec is not None else None,
        }

        with open(ckpt_dir / f"pareto_summary_t{t:04d}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("  ParEGO 检查点已保存：t=%d, HV=%.6f", t, summary["hypervolume"])


# ═══════════════════════════════════════════════════════════════════════════
# §F  联合运行工具（同时运行 LLAMBO-MO 和 ParEGO，输出对比结果）
# ═══════════════════════════════════════════════════════════════════════════

def run_comparison(
    shared_config: Optional[Dict[str, Any]] = None,
    llambo_extra:  Optional[Dict[str, Any]] = None,
    pareto_extra:  Optional[Dict[str, Any]] = None,
    output_dir:    str = "comparison_results",
) -> Dict[str, Any]:
    """
    同时运行 LLAMBO-MO 和 ParEGO，输出对比统计。

    Parameters
    ----------
    shared_config : 两个算法共享的配置（如 max_iterations, n_warmstart）
    llambo_extra  : LLAMBO-MO 专有配置（如 llm_model）
    pareto_extra  : ParEGO 专有配置（如 n_random_cands）
    output_dir    : 结果输出目录

    Returns
    -------
    dict : {"llambo_hv": float, "pareto_hv": float, "llambo_db": ObservationDB, ...}
    """
    from llmbo.optimizer import BayesOptimizer

    shared = shared_config or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── 运行 ParEGO ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  开始运行 ParEGO 对比实验")
    logger.info("=" * 70)
    pareto_cfg = {**shared, **(pareto_extra or {})}
    pareto_cfg["checkpoint_dir"] = str(out / "checkpoints_pareto")
    pareto_opt = ParEGOOptimizer(config=pareto_cfg)
    pareto_db  = pareto_opt.run()
    pareto_opt.save_results(str(out / "pareto"))
    results["pareto_db"]  = pareto_db
    results["pareto_hv"]  = pareto_db.compute_hypervolume()
    results["pareto_hv_trace"] = pareto_db.get_hv_trace()

    # ── 运行 LLAMBO-MO ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  开始运行 LLAMBO-MO")
    logger.info("=" * 70)
    llambo_cfg = {**shared, **(llambo_extra or {})}
    llambo_cfg["checkpoint_dir"] = str(out / "checkpoints_llambo")
    llambo_opt = BayesOptimizer(config=llambo_cfg)
    llambo_db  = llambo_opt.run()
    llambo_opt.save_results(str(out / "llambo"))
    results["llambo_db"]  = llambo_db
    results["llambo_hv"]  = llambo_db.compute_hypervolume()
    results["llambo_hv_trace"] = llambo_db.get_hv_trace()

    # ── 输出对比摘要 ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  对比结果汇总")
    logger.info("=" * 70)
    logger.info(
        "  LLAMBO-MO:  最终 HV = %.6f  |PF| = %d  总评估 = %d",
        results["llambo_hv"], llambo_db.pareto_size, llambo_db.size
    )
    logger.info(
        "  ParEGO:     最终 HV = %.6f  |PF| = %d  总评估 = %d",
        results["pareto_hv"], pareto_db.pareto_size, pareto_db.size
    )
    hv_gain = (results["llambo_hv"] - results["pareto_hv"]) / (results["pareto_hv"] + 1e-12) * 100
    logger.info("  LLAMBO-MO 相对 ParEGO HV 提升: %.2f%%", hv_gain)

    # 保存对比摘要
    cmp_summary = {
        "llambo_final_hv":   results["llambo_hv"],
        "pareto_final_hv":   results["pareto_hv"],
        "hv_gain_pct":       hv_gain,
        "llambo_n_total":    llambo_db.size,
        "pareto_n_total":    pareto_db.size,
        "llambo_pareto_size": llambo_db.pareto_size,
        "pareto_pareto_size": pareto_db.pareto_size,
        "llambo_hv_trace":   results["llambo_hv_trace"].tolist(),
        "pareto_hv_trace":   results["pareto_hv_trace"].tolist(),
    }
    with open(out / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(cmp_summary, f, indent=2)

    logger.info("对比结果已保存至: %s", output_dir)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# §G  自测
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("ParEGO 组件单元测试（不运行 PyBaMM）")
    print("=" * 60)

    BOUNDS = {"I1": (3.0, 7.0), "SOC1": (0.1, 0.7), "I2": (1.0, 5.0)}

    # ── 1. LHS 采样测试 ────────────────────────────────────────────────────
    print("\n1. LHS 采样测试")
    X_lhs = lhs_sampling(BOUNDS, n=10, seed=42)
    assert X_lhs.shape == (10, 3), f"shape 错误: {X_lhs.shape}"
    lo = np.array([3.0, 0.1, 1.0])
    hi = np.array([7.0, 0.7, 5.0])
    assert np.all(X_lhs >= lo) and np.all(X_lhs <= hi), "LHS 超出边界!"
    print(f"  ✓ LHS (10 点) shape={X_lhs.shape}, 全部在边界内")
    print(f"  示例: {X_lhs[:3].round(4).tolist()}")

    # ── 2. ParEGOSurrogate 测试 ────────────────────────────────────────────
    print("\n2. ParEGOSurrogate (sklearn GP) 测试")
    rng = np.random.default_rng(0)
    X_tr = rng.uniform(0, 1, size=(15, 3))
    F_tr = rng.uniform(0, 1, size=15)

    surrogate = ParEGOSurrogate(n_restarts=2, normalize_y=True, seed=0)
    surrogate.fit(X_tr, F_tr)
    summary = surrogate.training_summary()
    print(f"  训练完成: n={summary['n_train']}  log_marginal={summary['log_marginal']:.4f}")

    X_te = rng.uniform(0, 1, size=(5, 3))
    mean, std = surrogate.predict(X_te)
    assert mean.shape == (5,), f"mean shape 错误: {mean.shape}"
    assert std.shape  == (5,), f"std shape 错误: {std.shape}"
    assert (std >= 0.0).all(), "std 含负值!"
    print(f"  mean in [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  std  in [{std.min():.4f}, {std.max():.4f}]")
    print("  ✓ ParEGOSurrogate 预测通过")

    # ── 3. StandardEICalculator 测试 ──────────────────────────────────────
    print("\n3. StandardEICalculator 测试")
    ei_calc = StandardEICalculator(xi=0.0)
    f_min = float(F_tr.min())
    ei = ei_calc.compute(mean, std, f_min)
    assert ei.shape == (5,), f"EI shape 错误: {ei.shape}"
    assert (ei >= 0.0).all(), "EI 含负值!"
    print(f"  EI in [{ei.min():.6f}, {ei.max():.6f}]")
    print("  ✓ StandardEICalculator 通过")

    # ── 4. Tchebycheff 与 LLAMBO-MO 一致性验证 ────────────────────────────
    print("\n4. Tchebycheff 一致性验证（复用 optimizer.py 函数）")
    Y_raw  = np.array([[2800, 305.0, 0.0012],
                       [2100, 312.0, 0.0035],
                       [1200, 322.0, 0.0150]])
    w_test = np.array([0.4, 0.3, 0.3])

    # 模拟动态 min/max
    Y_tilde = log_transform_objectives(Y_raw)
    y_min, y_max = compute_dynamic_bounds(Y_tilde)
    F_tch = compute_tchebycheff_from_raw(Y_raw, w_test, y_min, y_max, eta=0.05)

    print(f"  Y_raw  = {Y_raw.tolist()}")
    print(f"  y_min  = {y_min.round(4).tolist()}")
    print(f"  y_max  = {y_max.round(4).tolist()}")
    print(f"  F_tch  = {F_tch.round(6).tolist()}")
    assert F_tch.shape == (3,), f"F_tch shape 错误: {F_tch.shape}"
    assert (F_tch >= 0.0).all(), "F_tch 含负值（归一化后应≥0）!"
    print("  ✓ Tchebycheff 函数与 LLAMBO-MO 一致")

    # ── 5. Riesz 权重集合测试 ──────────────────────────────────────────────
    print("\n5. Riesz 权重集合测试（seed=42，与 LLAMBO-MO 完全一致）")
    W = generate_riesz_weight_set(n_obj=3, n_div=5, s=2.0, n_iter=50, lr=5e-3, seed=42)
    assert W.ndim == 2 and W.shape[1] == 3, f"W shape 错误: {W.shape}"
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-6), "权重不满足 Σ=1!"
    assert (W >= 0.0).all(), "权重含负值!"
    print(f"  |W| = {len(W)}")
    print(f"  示例: {W[:3].round(4).tolist()}")
    print("  ✓ Riesz 权重集合通过")

    # ── 6. ParEGOOptimizer 配置测试（不运行 PyBaMM）────────────────────────
    print("\n6. ParEGOOptimizer 配置检验")
    cfg_test = {
        "max_iterations": 3,
        "n_warmstart":    5,
        "riesz_n_iter":   10,   # 快速测试
    }
    opt_test = ParEGOOptimizer(config=cfg_test)
    assert opt_test._weight_set is not None
    assert len(opt_test._weight_set) > 0
    print(f"  ParEGOOptimizer 实例化成功")
    print(f"  cfg: max_iter={opt_test.cfg['max_iterations']}  n_warmstart={opt_test.cfg['n_warmstart']}")
    print("  ✓ ParEGOOptimizer 配置通过")

    print("\n✓ pareto_optimizer.py 全部单元测试通过")
    print("\n使用示例:")
    print("  from pareto_optimizer import ParEGOOptimizer")
    print("  opt = ParEGOOptimizer({'max_iterations': 20, 'n_warmstart': 10})")
    print("  db  = opt.run()")
    print("  opt.save_results('results_pareto')")
    print("\n联合对比:")
    print("  from pareto_optimizer import run_comparison")
    print("  results = run_comparison(")
    print("      shared_config={'max_iterations': 20, 'n_warmstart': 10},")
    print("      llambo_extra={'llm_model': 'gpt-4o'},")
    print("  )")