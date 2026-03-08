"""
LLM-MOBO 主程序（FrameWork.md 实现）

决策空间: 3D (I1, SOC1, I2)
目标空间: 3D (time, temp, aging)

约束实现:
  C-1: Ψ 焦耳热代理函数及解析梯度 (psi_function.py)
  C-2: W^(t) = (w1·W_time + w2·W_temp + w3·W_aging)/Σwi，无 W_data
  C-3: Riesz s-energy 权重集合 (N=15) via tchebycheff.py
  C-4: W_charge 三维高斯搜索权重 (Eq.17)
  C-5: 无条件 gamma 更新（无 if/else）
  C-6: GAMMA_INIT=0.1, ALPHA_MAX=0.7, ALPHA_MIN=0.05, T_DECAY=60

用法:
  python main.py --demo            # Mock 模式快速演示
  python exp/runner.py --method V0 --seed 42  # 推荐完整实验入口
"""

import argparse
import asyncio
import os
import numpy as np
from typing import Dict, List, Optional

from config import (
    BATTERY_CONFIG, PARAM_BOUNDS, MOBO_CONFIG, BO_CONFIG,
    LLM_CONFIG, DATA_CONFIG, Q_NOM, SOC0, SOC_END,
    get_algorithm_param,
)
from battery_env.wrapper import BatterySimulator
from models.gp_model import MOGPModel
from acquisition.tchebycheff import TchebycheffScalarizer
from acquisition.acquisition import PhysicsWeightedAcquisition
from psi_function import PsiFunction
from database import ExperimentDatabase, compute_hypervolume_normalized
from utils.transforms import DataTransformer


# ============================================================
# 约束 C-5: 无条件 gamma 更新（无 if/else）
# ============================================================
def update_gamma(
    gamma: float,
    f_min_t: float,
    f_min_prev: float,
    rho: float = 0.1,
    gamma_min: float = 0.001,
    gamma_max: float = 2.0,
) -> float:
    """无条件 gamma 更新 — C-5。
    gamma = gamma * (1 + rho * (f_min_t - f_min_prev) / |f_min_prev|)
    无论 Δf 正负每轮都执行，禁止任何 if/else 条件。
    """
    eps = 1e-10
    gamma_new = gamma * (1.0 + rho * (f_min_t - f_min_prev) / (abs(f_min_prev) + eps))
    return float(np.clip(gamma_new, gamma_min, gamma_max))


# ============================================================
# 主类: LLMMOBO
# ============================================================
class LLMMOBO:
    """
    LLM 增强多目标贝叶斯优化器（FrameWork.md §6 完整实现）

    消融标志（exp/runner.py 透传）:
      use_warmstart     — Touchpoint 1b LLM 热启动
      use_coupling      — 物理复合核（Eq.11）
      use_adaptive_W    — W^(t) 自适应耦合矩阵（Eq.11'）
      use_llm_sampling  — Touchpoint 2 LLM 候选采样
      gamma_adaptive    — 超体积反馈驱动 HV 记录
    """

    def __init__(
        self,
        llm_api_key: str = None,
        n_warmstart: int = BO_CONFIG.get("n_warmstart", 5),
        n_random_init: int = BO_CONFIG.get("n_random_init", 10),
        n_iterations: int = BO_CONFIG.get("n_iterations", 50),
        # 消融标志
        use_warmstart: bool = True,
        use_coupling: bool = True,
        use_adaptive_W: bool = True,
        use_llm_sampling: bool = True,
        gamma_adaptive: bool = True,
        # 运行控制
        gamma_init: float = BO_CONFIG.get("gamma_init", 0.1),
        verbose: bool = True,
        db_path: str = ":memory:",
    ):
        self.verbose = verbose
        self.n_warmstart = n_warmstart
        self.n_random_init = n_random_init
        self.n_iterations = n_iterations
        self.use_warmstart = use_warmstart
        self.use_coupling = use_coupling
        self.use_adaptive_W = use_adaptive_W
        self.use_llm_sampling = use_llm_sampling
        self.gamma_adaptive = gamma_adaptive

        # C-6: gamma_init = 0.1
        self.gamma = float(gamma_init)

        # 数据库
        self.db = ExperimentDatabase(db_path)
        self.f_min_history: List[float] = []
        self.hv_history: List[float] = []

        # 仿真器
        self.simulator = BatterySimulator(
            param_set=BATTERY_CONFIG["param_set"],
            init_voltage=BATTERY_CONFIG["init_voltage"],
            init_temp=BATTERY_CONFIG["init_temp"],
            sample_time=BATTERY_CONFIG["sample_time"],
            voltage_max=BATTERY_CONFIG["voltage_max"],
            temp_max=BATTERY_CONFIG["temp_max"],
            soc_target=BATTERY_CONFIG["soc_target"],
        )

        # Ψ 代理函数
        self.psi = PsiFunction()

        # 数据变换器
        self.data_transformer = DataTransformer(
            enable_log_aging=DATA_CONFIG.get("enable_log_aging", True),
            verbose=False,
        )

        # 参考点 & 理想点
        self.reference_point = np.array([
            MOBO_CONFIG["reference_point"]["time"],
            MOBO_CONFIG["reference_point"]["temp"],
            MOBO_CONFIG["reference_point"]["aging"],
        ])
        self.ideal_point = np.array([
            MOBO_CONFIG["ideal_point"]["time"],
            MOBO_CONFIG["ideal_point"]["temp"],
            MOBO_CONFIG["ideal_point"]["aging"],
        ])

        # C-3: Riesz s-energy 权重集合
        self.scalarizer = TchebycheffScalarizer(
            ideal_point=self.ideal_point,
            reference_point=self.reference_point,
            eta=MOBO_CONFIG.get("eta", 0.05),
            n_weights=MOBO_CONFIG.get("N_WEIGHTS", 15),
        )

        # GP 模型
        self.mogp = MOGPModel(
            use_coupling=self.use_coupling,
            gamma_init=self.gamma,
            n_dims=3,
            verbose=False,
        )

        # 采集函数 (C-4: W_charge 三维高斯)
        self.acquisition_fn = PhysicsWeightedAcquisition(
            alpha_max=BO_CONFIG.get("alpha_max", 0.7),
            alpha_min=BO_CONFIG.get("alpha_min", 0.05),
            t_decay_alpha=BO_CONFIG.get("t_decay_alpha", 60),
            kappa=BO_CONFIG.get("kappa", 0.20),
            n_cand=get_algorithm_param("acquisition", "N_cand", 15),
            n_select=get_algorithm_param("acquisition", "N_select", 3),
            verbose=False,
        )

        # LLM 客户端
        self.llm_api_key = llm_api_key or LLM_CONFIG.get("api_key")
        self.llm_enabled = bool(self.llm_api_key)
        self._warmstart_client = None
        self._coupling_client = None
        self._llm_sampler = None

        if self.llm_enabled:
            from components.warmstart import LLMWarmStart
            from components.coupling_inference import LLMCouplingInference
            from components.llm_weighting import LLAMBOWeighting

            self._warmstart_client = LLMWarmStart(
                api_key=self.llm_api_key, verbose=self.verbose
            )
            self._coupling_client = LLMCouplingInference(
                api_key=self.llm_api_key, verbose=self.verbose
            )
            if self.use_llm_sampling:
                self._llm_sampler = LLAMBOWeighting(
                    param_bounds=PARAM_BOUNDS,
                    llm_api_key=self.llm_api_key,
                    base_url=LLM_CONFIG.get("base_url"),
                    model=LLM_CONFIG.get("model"),
                    verbose=self.verbose,
                )

        # C-2: W_time, W_temp, W_aging（Touchpoint 1a 初始化时一次获取）
        self.W_time: Optional[np.ndarray] = None
        self.W_temp: Optional[np.ndarray] = None
        self.W_aging: Optional[np.ndarray] = None

        if self.verbose:
            print("=" * 60)
            print("LLMBO-MO 初始化")
            print(f"  gamma_init={self.gamma} (C-6) | N_WEIGHTS={MOBO_CONFIG.get('N_WEIGHTS',15)} (C-3)")
            print(f"  LLM={'启用' if self.llm_enabled else '禁用'}")
            ablation_str = (f"ws={use_warmstart}, pk={use_coupling}, "
                            f"ga={use_adaptive_W}, llm={use_llm_sampling}, hv={gamma_adaptive}")
            print(f"  消融: {ablation_str}")
            print("=" * 60)

    # ── 约束 C-2: 构建 W^(t) ────────────────────────────────────
    def _build_W_t(self, w: np.ndarray) -> np.ndarray:
        """W^(t) = (w1·W_time + w2·W_temp + w3·W_aging) / Σwi  (Eq.11')
        不含任何 W_data 项 — C-2 约束。"""
        if not self.use_adaptive_W or self.W_time is None:
            return np.eye(3)
        s = w.sum() + 1e-12
        return (w[0] * self.W_time + w[1] * self.W_temp + w[2] * self.W_aging) / s

    # ── 随机采样 ──────────────────────────────────────────────
    def _random_sample(self) -> Dict:
        return {
            "I1": float(np.random.uniform(*PARAM_BOUNDS["I1"])),
            "SOC1": float(np.random.uniform(*PARAM_BOUNDS["SOC1"])),
            "I2": float(np.random.uniform(*PARAM_BOUNDS["I2"])),
            "source": "random",
        }

    # ── 评估 + 存储 ───────────────────────────────────────────
    def _evaluate_and_store(self, params: Dict):
        sim_p = {k: params[k] for k in ("I1", "SOC1", "I2") if k in params}
        result = self.simulator.simulate_3d(**sim_p)
        meta = {"source": params.get("source", "unknown"),
                "reasoning": params.get("reasoning", "")}
        self.db.add_experiment_3d(sim_p, result, meta)
        if self.verbose:
            tag = f"[OK] t={result['time']:.0f}s T={result['temp']:.1f}K a={result['aging']:.2e}" \
                  if result["valid"] else f"[FAIL] {result['violation']}"
            print(f"    {tag}")

    # ── f_min 计算 ────────────────────────────────────────────
    def _compute_f_min(self, w: np.ndarray) -> float:
        valid_data = self.db.get_valid_experiments()
        if not valid_data:
            return float("inf")
        return min(
            float(self.scalarizer.scalarize(np.array([r["time"], r["temp"], r["aging"]]), w))
            for r in valid_data
        )

    # ── HV 更新 ───────────────────────────────────────────────
    def _update_hv(self) -> float:
        pareto = self.db.get_pareto_front()
        if not pareto:
            hv = 0.0
        else:
            pf = np.array([[r["time"], r["temp"], r["aging"]] for r in pareto])
            hv = compute_hypervolume_normalized(pf, self.reference_point, self.ideal_point)
        self.hv_history.append(hv)
        return hv

    # ── Touchpoint 1a: 耦合矩阵初始化 ────────────────────────
    async def _init_coupling_matrices(self):
        """从 LLM 获取 W_time, W_temp, W_aging（仅在初始化时调用一次）"""
        if not (self.llm_enabled and self.use_coupling and self._coupling_client):
            self.W_time = np.eye(3)
            self.W_temp = np.eye(3)
            self.W_aging = np.eye(3)
            return
        try:
            matrices = await self._coupling_client.get_objective_coupling_matrices()
            self.W_time = np.array(matrices.get("W_time", np.eye(3)))
            self.W_temp = np.array(matrices.get("W_temp", np.eye(3)))
            self.W_aging = np.array(matrices.get("W_aging", np.eye(3)))
            if self.verbose:
                print("  [1a] 耦合矩阵已从 LLM 获取")
        except Exception as e:
            if self.verbose:
                print(f"  [1a] 失败: {e}，使用单位矩阵")
            self.W_time = np.eye(3)
            self.W_temp = np.eye(3)
            self.W_aging = np.eye(3)

    # ── Touchpoint 1b: LLM 热启动 ────────────────────────────
    async def _llm_warmstart(self) -> List[Dict]:
        if not (self.use_warmstart and self.llm_enabled and self._warmstart_client):
            return [self._random_sample() for _ in range(self.n_warmstart)]
        try:
            strategies = await self._warmstart_client.generate_strategies(
                n_strategies=self.n_warmstart, param_bounds=PARAM_BOUNDS
            )
            return [
                {
                    "I1": float(np.clip(s.get("I1", s.get("current1", 3.0)),
                                        *PARAM_BOUNDS["I1"])),
                    "SOC1": float(np.clip(s.get("SOC1", s.get("soc1", 0.4)),
                                          *PARAM_BOUNDS["SOC1"])),
                    "I2": float(np.clip(s.get("I2", s.get("current2", 2.0)),
                                        *PARAM_BOUNDS["I2"])),
                    "reasoning": s.get("reasoning", s.get("rationale", "")),
                    "source": "llm_warmstart",
                }
                for s in strategies
            ]
        except Exception as e:
            if self.verbose:
                print(f"  [1b] 失败: {e}，回退随机")
            return [self._random_sample() for _ in range(self.n_warmstart)]

    # ── Touchpoint 2: LLM 候选采样（迭代中）────────────────────
    async def _llm_sample_candidates(
        self, iteration: int, w: np.ndarray, theta_best: np.ndarray
    ) -> List[Dict]:
        n_cand = get_algorithm_param("acquisition", "N_cand", 15)
        if not (self.use_llm_sampling and self.llm_enabled and self._llm_sampler):
            return [self._random_sample() for _ in range(n_cand)]
        try:
            grad = self.psi.gradient(theta_best)
            candidates = await self._llm_sampler.generate_candidates(
                database=self.db.to_legacy_format_3d(),
                weights=w,
                grad_psi=grad,
                iteration=iteration,
                total_iterations=self.n_iterations,
                n_candidates=n_cand,
            )
            return [
                {
                    "I1": float(np.clip(c.get("I1", 3.0), *PARAM_BOUNDS["I1"])),
                    "SOC1": float(np.clip(c.get("SOC1", 0.4), *PARAM_BOUNDS["SOC1"])),
                    "I2": float(np.clip(c.get("I2", 2.0), *PARAM_BOUNDS["I2"])),
                    "source": "llm_sample",
                }
                for c in candidates
            ]
        except Exception as e:
            if self.verbose:
                print(f"  [2] 失败: {e}，使用随机候选")
            return [self._random_sample() for _ in range(n_cand)]

    # ── 主优化循环 ────────────────────────────────────────────
    async def optimize(self) -> Dict:
        """FrameWork.md §6 Algorithm 主循环。"""
        #
        # INITIALIZATION
        #
        if self.verbose:
            print("\n[初始化] Touchpoint 1a: 获取耦合矩阵...")
        await self._init_coupling_matrices()

        if self.verbose:
            print(f"[初始化] 热启动({self.n_warmstart}) + 随机({self.n_random_init})")
        warmstart_samples = await self._llm_warmstart()
        for s in warmstart_samples:
            self._evaluate_and_store(s)
        for _ in range(self.n_random_init):
            self._evaluate_and_store(self._random_sample())

        n_valid = len(self.db.get_valid_experiments())
        if self.verbose:
            print(f"  完成: {n_valid} 个有效样本")

        #
        # OPTIMIZATION LOOP
        #
        for t in range(self.n_iterations):
            if self.verbose:
                print(f"\n[迭代 {t+1}/{self.n_iterations}]")

            # A. 采样 w^(t) (C-3)
            w = self.scalarizer.sample_weight_vector(iteration=t)
            if self.verbose:
                print(f"  w=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})")

            # B. 数据变换（使用 3D 格式：I1/SOC1/I2）
            legacy_db = self.db.to_legacy_format_3d()
            transformed_db = self.data_transformer.fit_transform_database(legacy_db)

            # C. 构建 W^(t) (C-2: 无 W_data)
            W_t = self._build_W_t(w)

            # D. 计算 θ_best 和 ∇Ψ
            valid_data = self.db.get_valid_experiments()
            if valid_data:
                best = min(
                    valid_data,
                    key=lambda r: float(self.scalarizer.scalarize(
                        np.array([r["time"], r["temp"], r["aging"]]), w)),
                )
                # DB 存储映射：current1=I1, time1=SOC1, current2=I2
                theta_best = np.array([best["current1"], best["time1"], best["current2"]])
            else:
                theta_best = np.array([3.0, 0.4, 2.0])
            grad_psi = self.psi.gradient(theta_best)

            # E. 训练 GP（复合核使用 W^(t) 和 ∇Ψ）
            try:
                self.mogp.set_coupling_matrix(W_t)
                self.mogp.set_gamma(self.gamma)
                self.mogp.train(transformed_db, weights=w)
                if self.verbose:
                    print(f"  GP 完成 γ={self.gamma:.4f}")
            except Exception as e:
                if self.verbose:
                    print(f"  GP 失败: {e}")

            # F. 更新标量化器动态边界
            try:
                bounds = self.data_transformer.get_transformed_bounds()
                self.scalarizer.ideal_point = bounds["ideal"]
                self.scalarizer.reference_point = bounds["reference"]
                rng = bounds["reference"] - bounds["ideal"]
                self.scalarizer.range = np.where(rng > 1e-10, rng, 1.0)
            except Exception:
                pass

            # G. Touchpoint 2: LLM 候选采样
            candidates = await self._llm_sample_candidates(t + 1, w, theta_best)

            # H. 采集函数打分 α = EI × W_charge (C-4)，选 top-N_select
            n_select = get_algorithm_param("acquisition", "N_select", 3)
            try:
                gp_list = self.mogp.get_gp_list()
                f_min_cur = self._compute_f_min(w)
                self.acquisition_fn.update_search_params(
                    theta_best=theta_best,
                    grad_psi=grad_psi,
                    iteration=t,
                    f_min_improved=(not self.f_min_history
                                    or f_min_cur < self.f_min_history[-1]),
                )
                scored = sorted(
                    [(c, self.acquisition_fn.score(
                        np.array([c["I1"], c["SOC1"], c["I2"]]),
                        gp_list, f_min_cur, w))
                     for c in candidates],
                    key=lambda x: x[1], reverse=True,
                )
                top = [s[0] for s in scored[:n_select]]
            except Exception as e:
                if self.verbose:
                    print(f"  采集打分失败: {e}")
                top = candidates[:n_select]

            # I. 评估 top 候选
            for cand in top:
                self._evaluate_and_store(cand)

            # J. 无条件 gamma 更新 (C-5)
            f_min_t = self._compute_f_min(w)
            f_min_prev = self.f_min_history[-1] if self.f_min_history else f_min_t
            self.gamma = update_gamma(self.gamma, f_min_t, f_min_prev)
            self.f_min_history.append(f_min_t)

            # K. HV 记录
            hv = self._update_hv()
            if self.verbose:
                print(f"  f={f_min_t:.4f} γ={self.gamma:.4f} HV={hv:.4f} "
                      f"Pareto={len(self.db.get_pareto_front())}")

            # L. 周期性检查点
            if (t + 1) % DATA_CONFIG.get("save_interval", 10) == 0:
                self._save_checkpoint(t + 1)

        return self._build_results()

    def _save_checkpoint(self, iteration: int):
        import json
        save_dir = DATA_CONFIG.get("save_dir", "./results")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"checkpoint_iter{iteration}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._build_results(), f, indent=2, default=str)
        if self.verbose:
            print(f"  [checkpoint] {path}")

    def _build_results(self) -> Dict:
        import json
        valid_data = self.db.get_valid_experiments()
        pareto = self.db.get_pareto_front()
        return {
            "n_evaluations": len(self.db.get_all_experiments()),
            "n_valid": len(valid_data),
            "hv_history": self.hv_history,
            "f_min_history": self.f_min_history,
            "pareto_front": [[r["time"], r["temp"], r["aging"]] for r in pareto],
            "n_pareto": len(pareto),
            "gamma_final": self.gamma,
            "database": [
                {
                    # DB 存储映射：current1→I1, time1→SOC1, current2→I2
                    "params": {
                        "I1": r.get("current1"),
                        "SOC1": r.get("time1"),
                        "I2": r.get("current2"),
                    },
                    "time": r.get("time"), "temp": r.get("temp"),
                    "aging": r.get("aging"), "valid": bool(r.get("valid")),
                }
                for r in self.db.get_all_experiments()
            ],
        }


# ============================================================
# CLI 入口
# ============================================================
async def _demo():
    """快速演示（使用正式 LLM，电池仿真使用轻量数学模型）"""
    import json

    class _MockSim:
        def simulate_3d(self, I1, SOC1, I2):
            Q = 18000
            I1 = max(float(I1), 0.01)
            I2 = max(float(I2), 0.01)
            SOC1 = float(np.clip(SOC1, 0.1, 0.7))
            t = (SOC1 - 0.1) * Q / I1 + (0.8 - SOC1) * Q / I2
            temp = 298.15 + 5 * I1 + 3 * I2
            aging = 1e-4 * (I1 ** 2 + I2 ** 2) * t / 3600
            valid = temp < 323.15
            return {"time": t, "temp": temp, "aging": max(aging, 1e-6),
                    "valid": valid, "violation": None if valid else "temp",
                    "current_profile": [], "temp_profile": [], "voltage_profile": [],
                    "soc_profile": [], "phase_profile": [], "cv_time": 0.0, "final_soc": 0.8}

    opt = LLMMOBO(
        llm_api_key=LLM_CONFIG.get("api_key"),
        n_warmstart=5, n_random_init=5, n_iterations=5,
        use_warmstart=True, use_coupling=True, use_adaptive_W=True,
        use_llm_sampling=True, verbose=True, db_path=":memory:",
    )
    opt.simulator = _MockSim()  # type: ignore（保留轻量仿真以加速演示）
    results = await opt.optimize()
    out = "results/demo_results.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n演示完成 → {out}")
    print(f"  n_valid={results['n_valid']}  n_pareto={results['n_pareto']}  "
          f"HV={results['hv_history'][-1] if results['hv_history'] else 0:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="LLMBO-MO 主程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行实验（推荐）:
  python exp/runner.py --method V0 --seed 42
  python exp/runner.py --list             # 查看全部方法

快速演示（正式 LLM + 轻量仿真）:
  python main.py --demo

生成图表:
  python plots/hv_comparison.py --results_dir results/
  python plots/pareto_3d.py --results results/V0/seed_42.json

结果分析:
  python analysis.py --mode pareto
        """
    )
    parser.add_argument("--demo", action="store_true", help="快速演示（正式 LLM + 轻量仿真）")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(_demo())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
