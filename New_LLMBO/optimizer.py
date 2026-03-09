"""
optimizer.py — LLAMBO-MO 主优化器
===================================
编排所有子模块，实现完整 Bayesian 优化主循环。

调用顺序（严格按照系统框图 §1-§4）：
  §1  初始化：构建组件 → LLM Touchpoint 1a（耦合矩阵）→ Touchpoint 1b（warm-start）
  §2  Warm-start 评估：PyBaMM 评估所有候选 → 填充 ObservationDB
  §3  采集函数初始化：af.initialize(database, llm_prior=llm)
  §4  主循环（t=0..T）：
        采样 w_vec → update_tchebycheff_context → gp.fit →
        LLM Touchpoint 2 → af.step → PyBaMM 评估 → 记录统计 → 保存检查点
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from database import ObservationDB, DEFAULT_REF_POINT, DEFAULT_BOUNDS
from gp_model import build_gp_stack
from llm_interface import build_llm_interface
from acquisition import build_acquisition_function
from pybamm_simulator import PyBaMMSimulator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# §A  超参数默认配置
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # ── 实验规模 ──────────────────────────────────────────────────────────
    "max_iterations":   50,       # 主循环迭代次数 T
    "n_warmstart":      10,       # warm-start 候选点数量 N_ws
    "n_candidates":     15,       # 每迭代 LLM 生成候选点数量 m
    "n_select":         3,        # 每迭代 PyBaMM 评估数量 N_select

    # ── LLM 配置 ──────────────────────────────────────────────────────────
    "llm_backend":      "openai", # "ollama" / "openai" / "anthropic" / "mock"
    "llm_model":        "gpt-4o",
    "llm_api_base":     "https://api.nuwaapi.com/v1/chat",
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

    # ── Tchebycheff 权重采样 ───────────────────────────────────────────────
    "w_sample_seed":    None,     # 固定随机种子（None = 随机）

    # ── 检查点 ────────────────────────────────────────────────────────────
    "checkpoint_dir":   "checkpoints",
    "checkpoint_every": 5,        # 每隔多少迭代保存一次

    # ── 电池模型 ──────────────────────────────────────────────────────────
    "battery_model":    "LG M50 (Chen2020)",
}


# ═══════════════════════════════════════════════════════════════════════════
# §B  Tchebycheff 权重采样工具函数
# ═══════════════════════════════════════════════════════════════════════════

def sample_tchebycheff_weights(
    n_obj: int = 3,
    rng:   Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    从均匀 Dirichlet 分布采样 Tchebycheff 权重向量。

    w_i ≥ 0, Σ w_i = 1

    采用 Dirichlet(1,1,...,1) 等价于在 simplex 上均匀采样，
    保证每次迭代以随机权重组合三个目标，探索不同 Pareto 前沿区域。
    """
    if rng is None:
        rng = np.random.default_rng()
    w = rng.dirichlet(np.ones(n_obj))
    return w.astype(float)


def compute_tchebycheff(
    objectives: np.ndarray,   # (n, 3) 或 (3,)
    w_vec:      np.ndarray,   # (3,)
    z_star:     np.ndarray,   # (3,) 理想点
) -> np.ndarray:
    """
    Tchebycheff 标量化：f_tch(θ) = max_i [ w_i * |f_i - z_i*| ]

    Parameters
    ----------
    objectives : (n, 3) 或 (3,)
    w_vec      : (3,) 权重向量
    z_star     : (3,) 理想点（各目标历史最小值）

    Returns
    -------
    np.ndarray  (n,) 或标量
    """
    objectives = np.atleast_2d(objectives)           # (n, 3)
    diff = np.abs(objectives - z_star[np.newaxis, :])  # (n, 3)
    weighted = w_vec[np.newaxis, :] * diff             # (n, 3)
    return weighted.max(axis=1).squeeze()               # (n,) or scalar


# ═══════════════════════════════════════════════════════════════════════════
# §C  BayesOptimizer 主类
# ═══════════════════════════════════════════════════════════════════════════

class BayesOptimizer:
    """
    LLAMBO-MO 贝叶斯优化主类。

    使用方式（最简）::

        from optimizer import BayesOptimizer
        opt = BayesOptimizer()
        opt.run()
        opt.save_results("results/")

    使用方式（自定义配置）::

        cfg = {
            "llm_backend": "ollama",
            "llm_model": "qwen2.5:7b",
            "max_iterations": 30,
            "n_warmstart": 8,
        }
        opt = BayesOptimizer(config=cfg)
        opt.run()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}

        seed = self.cfg.get("w_sample_seed")
        self._rng = np.random.default_rng(seed)

        self.simulator:    Optional[PyBaMMSimulator] = None
        self.database:     Optional[ObservationDB]   = None
        self.llm:          Any                       = None
        self.psi_fn:       Any                       = None
        self.coupling_mgr: Any                       = None
        self.gamma_ann:    Any                       = None
        self.gp:           Any                       = None
        self.af:           Any                       = None

        self._current_iter:  int                     = 0
        self._current_w_vec: Optional[np.ndarray]    = None
        self._z_star:        np.ndarray              = np.zeros(3)

        Path(self.cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # ── §1 初始化 ──────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        初始化所有组件。

        按照框图 §1 顺序：
          1. 构建 PyBaMM 仿真器
          2. 构建 ObservationDB
          3. 构建 LLM 接口
          4. Touchpoint 1a：生成耦合矩阵 → 注入 CouplingMatrixManager
          5. 构建 GP 栈（使用来自 1a 的耦合矩阵）
          6. 构建 AcquisitionFunction
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
        self.llm = build_llm_interface(
            param_bounds=param_bounds,
            backend=self.cfg["llm_backend"],
            model=self.cfg["llm_model"],
            api_base=self.cfg["llm_api_base"],
            api_key=self.cfg["llm_api_key"],
            n_samples=self.cfg["llm_n_samples"],
            temperature=self.cfg["llm_temperature"],
            battery_model=self.cfg["battery_model"],
        )

        # 4. Touchpoint 1a：耦合矩阵
        logger.info("Touchpoint 1a: 生成耦合矩阵 ...")
        W_time, W_temp, W_aging = self.llm.generate_coupling_matrices()

        # 5. GP 栈
        self.psi_fn, self.coupling_mgr, self.gamma_ann, self.gp = build_gp_stack(
            param_bounds=param_bounds,
            gamma_max=self.cfg["gamma_max"],
            gamma_min=self.cfg["gamma_min"],
            gamma_t_decay=self.cfg["gamma_t_decay"],
        )
        self.coupling_mgr.set_llm_matrices(W_time, W_temp, W_aging)
        logger.info("耦合矩阵已注入 CouplingMatrixManager")

        # 6. AcquisitionFunction
        self.af = build_acquisition_function(
            gp=self.gp,
            psi_fn=self.psi_fn,
            param_bounds=param_bounds,
            n_select=self.cfg["n_select"],
            alpha_max=self.cfg["alpha_max"],
            alpha_min=self.cfg["alpha_min"],
            t_decay_alpha=self.cfg["t_decay_alpha"],
            kappa=self.cfg["kappa"],
            eps_sigma=self.cfg["eps_sigma"],
            rho=self.cfg["rho"],
        )

        logger.info("所有组件初始化完成")

    # ── §2 Warm-start 评估 ────────────────────────────────────────────────

    def run_warmstart(self) -> None:
        """
        Touchpoint 1b：生成 warm-start 候选点并全部评估。
        """
        logger.info("=" * 60)
        logger.info("§2 Warm-Start 评估阶段 (N_ws=%d)", self.cfg["n_warmstart"])
        logger.info("=" * 60)

        warmstart_candidates = self.llm.generate_warmstart_candidates(
            n=self.cfg["n_warmstart"]
        )

        for i, theta in enumerate(warmstart_candidates):
            logger.info("  Warm-start [%d/%d]: θ=%s",
                        i + 1, len(warmstart_candidates), theta.round(4))
            t_start = time.perf_counter()
            result = self.simulator.evaluate(theta)
            elapsed = time.perf_counter() - t_start

            self.database.add_from_simulator(
                theta=theta,
                result=result,
                source="llm",
                iteration=0,
            )

            feasible_str = "✓" if result["feasible"] else f"✗ ({result.get('violation', '?')})"
            logger.info(
                "    → %s  objectives=%s  (%.1fs)",
                feasible_str,
                np.round(result["raw_objectives"], 3),
                elapsed,
            )

        logger.info("Warm-start 完成: %d 可行解 / %d 总计",
                    self.database.n_feasible, self.database.size)

        # warm-start 结束后，用均匀权重初始化 Tchebycheff 上下文
        w_init = np.array([1.0/3, 1.0/3, 1.0/3])
        self.database.update_tchebycheff_context(w_init)

    # ── §3 采集函数初始化 ─────────────────────────────────────────────────

    def initialize_acquisition(self) -> None:
        """
        Algorithm 步骤 5：在 warm-start 评估完成后初始化 μ / σ 追踪器。
        """
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
        """
        logger.info("=" * 60)
        logger.info("§4 主优化循环开始 (max_iterations=%d)", self.cfg["max_iterations"])
        logger.info("=" * 60)

        for t in range(self.cfg["max_iterations"]):
            self._current_iter = t
            iter_start = time.perf_counter()

            logger.info("\n─── 迭代 t=%d ─────────────────────────────────────", t)

            # ── 步骤 1：采样 Tchebycheff 权重 ───────────────────────────
            w_vec = sample_tchebycheff_weights(n_obj=3, rng=self._rng)
            self._current_w_vec = w_vec
            logger.info("  w_vec = [%.3f, %.3f, %.3f]", *w_vec)

            # ── 步骤 2：更新 DB 的 Tchebycheff 上下文 ───────────────────
            self.database.update_tchebycheff_context(w_vec)
            self._z_star = self.database._z_star.copy()

            # ── 步骤 3：训练 GP ───────────────────────────────────────
            X_train, Y_train = self.database.get_train_XY(
                feasible_only=True, normalize_X=True, normalize_Y=False
            )
            if len(X_train) < 3:
                logger.warning("  可行解不足 3 个，跳过本轮 GP 训练，随机采样回退")
                self._evaluate_random_candidates(t, source="random")
                continue

            # 计算 Tchebycheff 标量目标
            _, Y_raw = self.database.get_train_XY(
                feasible_only=True, normalize_X=False, normalize_Y=False
            )
            F_tch = compute_tchebycheff(Y_raw, w_vec, self._z_star)
            f_mean = float(F_tch.mean())
            f_std  = float(F_tch.std()) + 1e-8
            F_tch_norm = (F_tch - f_mean) / f_std

            self.gp.fit(X_train, F_tch_norm, w_vec, t=t)
            summary = self.gp.training_summary()
            logger.info(
                "  GP 训练完成: n=%d  l=%.4f  γ=%.4f",
                summary["n_train"], summary["l"], summary["gamma"],
            )

            # ── 步骤 4：Touchpoint 2 生成候选点 ─────────────────────────
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

            # 归一化候选点（GP 在归一化空间预测）
            lo = np.array([self.simulator.param_bounds[k][0] for k in ["I1", "SOC1", "I2"]])
            hi = np.array([self.simulator.param_bounds[k][1] for k in ["I1", "SOC1", "I2"]])
            X_cand_norm = (X_candidates - lo) / (hi - lo + 1e-12)

            # ── 步骤 5：af.step() 选 top-k ────────────────────────────
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
                result_af.selected_scores.round(6),
            )

            # ── 步骤 6：PyBaMM 评估 top-k ────────────────────────────
            n_new = 0
            for rank, idx in enumerate(result_af.selected_indices):
                theta_orig = X_candidates[idx]
                logger.info(
                    "  评估候选 [rank=%d]: I1=%.3f  SOC1=%.3f  I2=%.3f",
                    rank, *theta_orig
                )
                t_eval = time.perf_counter()
                sim_result = self.simulator.evaluate(theta_orig)
                elapsed_eval = time.perf_counter() - t_eval

                self.database.add_from_simulator(
                    theta=theta_orig,
                    result=sim_result,
                    source="llm_gp",
                    iteration=t + 1,
                    acq_value=float(result_af.selected_scores[rank]),
                    acq_type="EI_Wcharge",
                    gp_pred={
                        "mean": float(result_af.all_mean[idx]),
                        "std":  float(result_af.all_std[idx]),
                    },
                )
                n_new += 1
                feasible_str = "✓" if sim_result["feasible"] else "✗"
                logger.info(
                    "    → %s  obj=%s  (%.1fs)",
                    feasible_str, np.round(sim_result["raw_objectives"], 3), elapsed_eval
                )

            # ── 步骤 7：记录统计，保存检查点 ─────────────────────────────
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
        """
        完整运行：setup → warm-start → 初始化采集函数 → 主循环。

        Returns
        -------
        ObservationDB — 包含所有评估结果的数据库
        """
        self.setup()
        self.run_warmstart()
        self.initialize_acquisition()
        self.run_optimization_loop()
        logger.info("\n优化完成！总评估: %d  最终 HV: %.6f",
                    self.database.size, self.database.compute_hypervolume())
        return self.database

    # ── 结果保存 ─────────────────────────────────────────────────────────

    def save_results(self, output_dir: str = "results") -> None:
        """将数据库和统计保存到 output_dir/。"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.database.save(str(Path(output_dir) / "database.json"))
        logger.info("结果已保存至: %s", output_dir)

    # ── 检查点 ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, t: int) -> None:
        """保存优化器状态检查点。"""
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
            "config":      self.cfg,
        }
        with open(ckpt_dir / f"summary_t{t:04d}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("  检查点已保存: t=%d", t)


# ═══════════════════════════════════════════════════════════════════════════
# §D  _DBProxy — 临时 f_min 覆盖代理
# ═══════════════════════════════════════════════════════════════════════════

class _DBProxy:
    """
    ObservationDB 的轻量级代理。

    GP 训练使用归一化的 F_tch，但 database._f_min 存储原始量尺度的值。
    EI 计算中 f_min 必须与 GP 预测值（归一化空间）处于同一量纲。
    此代理类仅覆盖 get_f_min()，其余方法全部转发给真实 database。
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
# §E  CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="LLAMBO-MO 锂电池快充优化")
    parser.add_argument("--backend",    default="ollama",      help="LLM 后端")
    parser.add_argument("--model",      default="qwen2.5:7b",  help="LLM 模型")
    parser.add_argument("--iters",      type=int, default=50,  help="优化迭代次数")
    parser.add_argument("--warmstart",  type=int, default=10,  help="Warm-start 点数")
    parser.add_argument("--candidates", type=int, default=15,  help="每迭代候选点数")
    parser.add_argument("--output",     default="results",     help="结果输出目录")
    args = parser.parse_args()

    cfg = {
        "llm_backend":    args.backend,
        "llm_model":      args.model,
        "max_iterations": args.iters,
        "n_warmstart":    args.warmstart,
        "n_candidates":   args.candidates,
    }

    opt = BayesOptimizer(config=cfg)
    db  = opt.run()
    opt.save_results(args.output)

    print("\n" + db.summary())
