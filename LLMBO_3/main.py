"""
main.py  —  LLMBO-MO 主循环入口
=================================
算法结构（对应论文伪代码）：

  Phase 1 (WarmStart):
    t ← 0
    从 LLM Agent 采样 N_LLM 充电协议 → 评估 → D_LLM^M（固定不变）

  Phase 2 (BO 主循环):
    while t < t_max:
      随机选权重向量 w^(t)  [RISE 方法]
      合并 D_LLM^M 和 D_T^M，用增广 Chebyshev 标量化
      基于合并数据训练 GP（标准 ARD Matérn 5/2，无 LLM 先验）
      EI + DE 优化（全局搜索空间 [0,1]^5，无 LLM 边界）
      评估最优候选点 → 加入 D_T^M
      t ← t + 1

运行方式：
    python main.py --soh 0.7 --n_llm 10 --t_max 30
    python main.py --soh 0.7 --n_llm 10 --t_max 30 --skip_llm   # 调试：跳过LLM
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from database      import Database
from scalarization import equal_weight, sample_weight, scalarize
from gp_model      import GPModel
from acquisition   import neg_ei_scalar
from optimizer     import de_optimize
from pareto        import pareto_front, normalized_hypervolume

# llm_warmstart.py 与本文件同目录
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_warmstart import main as llm_warmstart_main, protocols_to_normalized

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  物理边界（与 pybamm_simulator.py / Evaluate_Bat.m 完全对齐）
# ---------------------------------------------------------------------------
PHYS_LB = np.array([2.0, 2.0, 2.0, 0.10, 0.10])
PHYS_UB = np.array([6.0, 5.0, 3.0, 0.40, 0.30])

NORM_LB = np.zeros(5)
NORM_UB = np.ones(5)


def _to_physical(x_norm: np.ndarray) -> np.ndarray:
    """归一化空间 → 物理空间，仅在传入 simulator 时调用。"""
    return np.asarray(x_norm) * (PHYS_UB - PHYS_LB) + PHYS_LB


# ---------------------------------------------------------------------------
#  仿真接口（懒加载，避免每次 import 都初始化 PyBaMM）
# ---------------------------------------------------------------------------

def _load_simulator(soh: float, use_crate: bool = True):
    """懒加载 PyBaMMSimulator。"""
    try:
        from pybamm_simulator import PyBaMMSimulator
        return PyBaMMSimulator(SOH=soh, use_crate=use_crate)
    except ImportError:
        logger.warning("PyBaMM 不可用，使用虚拟仿真器（仅供调试）")
        return _DummySimulator()


class _DummySimulator:
    """调试用虚拟仿真器，返回带噪声的随机目标值。"""
    def evaluate(self, theta):
        x = np.asarray(theta, dtype=float)
        t    = 3600 / x[0] + np.random.normal(0, 10)
        dT   = 5 * x[0] + np.random.normal(0, 0.5)
        ag   = 0.01 * x[0] ** 2 + np.random.normal(0, 0.001)
        return {
            "raw_objectives": np.array([max(t, 300), max(dT, 1), max(ag, 1e-4)]),
            "feasible": True,
            "violation": None,
        }


# ---------------------------------------------------------------------------
#  SanityCheck（纯 Python 版，与 SanityCheck.m 逻辑一致）
# ---------------------------------------------------------------------------

def sanity_check(x_norm: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    对 LLM 输出的归一化协议做合法性修复。
    - 电流单调性：I1 >= I2 >= I3
    - SOC 跨度：dSOC1 + dSOC2 <= 0.70
    返回修复后的 x_clean。
    """
    x = np.clip(x_norm, 0.0, 1.0).copy()
    n_fixed = 0

    for i in range(len(x)):
        phys = x[i] * (PHYS_UB - PHYS_LB) + PHYS_LB
        changed = False

        # 电流单调性
        if phys[1] > phys[0]:
            phys[1] = phys[0]; changed = True
        if phys[2] > phys[1]:
            phys[2] = min(phys[1], PHYS_UB[2]); changed = True

        # SOC 跨度
        if phys[3] + phys[4] > 0.70:
            scale    = 0.70 / (phys[3] + phys[4])
            phys[3] *= scale
            phys[4] *= scale
            changed  = True

        if changed:
            phys = np.clip(phys, PHYS_LB, PHYS_UB)
            x[i] = (phys - PHYS_LB) / (PHYS_UB - PHYS_LB)
            n_fixed += 1

    if verbose and n_fixed > 0:
        logger.info(f"SanityCheck: 输入 {len(x_norm)} 条，修复 {n_fixed} 条")
    return x


def _lhs_sampling(
    n: int,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """拉丁超立方采样。"""
    dim = len(lb)
    x   = np.zeros((n, dim))
    for j in range(dim):
        perm    = rng.permutation(n)
        x[:, j] = lb[j] + (perm + rng.random(n)) / n * (ub[j] - lb[j])
    return np.clip(x, lb, ub)


# ---------------------------------------------------------------------------
#  Phase 1: WarmStart
#  仅调用 LLM 生成协议，不使用 LLM GP 先验或搜索边界
# ---------------------------------------------------------------------------

def warm_start(
    n_llm: int,
    soh: float,
    output_dir: Path,
    skip_llm: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    LLM WarmStart：调用 LLM 推荐 N_LLM 个充电协议。

    若 LLM 调用失败或返回协议数不足，用 LHS 补全到 n_llm 条。

    Parameters
    ----------
    n_llm      : 期望的初始协议数量
    soh        : 电池健康状态
    output_dir : LLM 原始回复保存目录
    skip_llm   : True = 跳过 LLM，直接用 LHS（调试模式）
    rng        : 随机数生成器

    Returns
    -------
    x_llm_norm : (N, 5) 归一化协议，N <= n_llm（全部来自 LLM）
                 若 LLM 失败则退回到 LHS
    """
    if rng is None:
        rng = np.random.default_rng(42)

    logger.info("=" * 50)
    logger.info("Phase 1: LLM WarmStart")
    logger.info("=" * 50)

    x_llm = np.empty((0, 5))

    if not skip_llm:
        try:
            json_path = str(output_dir / "llm_warmstart.json")
            llm_data  = llm_warmstart_main(
                n_llm=n_llm, soh=soh, output_path=json_path
            )
            x_llm_raw = np.array(llm_data["protocols_normalized"])
            x_llm     = sanity_check(x_llm_raw, verbose=True)
            logger.info(f"LLM 协议: {len(x_llm)} 条（通过 SanityCheck）")
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}，退回 LHS")

    # ── LLM 失败或不足时用 LHS 补全 ─────────────────────────────
    if len(x_llm) == 0:
        logger.warning(f"LLM 返回 0 条协议，改用 LHS 生成 {n_llm} 条")
        x_lhs  = _lhs_sampling(n_llm, NORM_LB, NORM_UB, rng)
        x_llm  = sanity_check(x_lhs, verbose=False)

    logger.info(f"WarmStart 初始种群: {len(x_llm)} 条")
    return x_llm


# ---------------------------------------------------------------------------
#  Phase 2: BO 主循环
#  使用两个独立数据库：db_llm（固定）+ db_t（增长）
# ---------------------------------------------------------------------------

def bo_loop(
    db_llm: Database,
    db_t: Database,
    simulator,
    t_max: int = 30,
    rng: Optional[np.random.Generator] = None,
    output_dir: Path = Path("."),
) -> Database:
    """
    ParEGO-style 多目标贝叶斯优化主循环。

    算法对应伪代码 Step 4–14：
      while t < t_max:
        w ← 随机权重向量 (RISE)
        合并 D_LLM^M 和 D_T^M → 增广 Chebyshev 标量化
        GP.fit(X_combined, y_agg_norm)   [标准 ARD Matérn 5/2]
        x_new ← argmax EI via DE        [全局搜索 [0,1]^5]
        y_new ← simulate(x_new)
        D_T^M.add(x_new, y_new)
        t ← t + 1

    Parameters
    ----------
    db_llm     : D_LLM^M，LLM 热启动评估结果，全程固定
    db_t       : D_T^M，BO 收集的点，每轮增加一条
    simulator  : 仿真器接口
    t_max      : 最大迭代次数
    rng        : 随机数生成器
    output_dir : 检查点保存目录

    Returns
    -------
    db_t : 更新后的 BO 数据库
    """
    if rng is None:
        rng = np.random.default_rng()

    # 权重向量（RISE：从单纯形均匀分布随机采样）
    W = equal_weight(H=30, M=3)
    logger.info(f"权重向量数: {len(W)}")

    # GP 模型（标准 ARD Matérn 5/2，不使用 LLM 先验）
    gp = GPModel(dim=5, length_scales=None, n_restarts=3)

    logger.info("=" * 50)
    logger.info(f"Phase 2: BO 主循环（t_max={t_max}）")
    logger.info(f"  初始 D_LLM 大小: {db_llm.n} | 初始 D_T 大小: {db_t.n}")
    logger.info("=" * 50)

    hv_history = []

    for t in range(1, t_max + 1):
        t_iter = time.time()
        logger.info(f"\n── 迭代 {t}/{t_max} | D_LLM={db_llm.n} | D_T={db_t.n} ──")

        # ── Step 5: 随机选权重向量 (RISE) ────────────────────────
        w = sample_weight(W, rng)

        # ── Step 8: 合并两库并标量化 ─────────────────────────────
        # 合并 D_LLM^M 和 D_T^M（若 D_T 为空则仅用 D_LLM）
        if db_t.n > 0:
            X_combined = np.vstack([db_llm.x, db_t.x])
            Y_combined = np.vstack([db_llm.y, db_t.y])
        else:
            X_combined = db_llm.x.copy()
            Y_combined = db_llm.y.copy()

        y_agg_norm, _ = scalarize(Y_combined, w, rho=0.05)
        f_best = float(y_agg_norm.min())
        logger.info(f"  w = {w.round(3)} | f_best = {f_best:.4f} | N_train = {len(X_combined)}")

        # ── Step 9: GP 拟合（标准 ARD Matérn 5/2）────────────────
        gp.fit(X_combined, y_agg_norm)

        # ── Step 11-13: EI + DE 优化（全局 [0,1]^5）────────────
        def obj_func(x: np.ndarray) -> float:
            return neg_ei_scalar(x, gp, f_best, xi=0.01)

        x_new, neg_ei_val = de_optimize(
            obj_func=obj_func,
            lb=NORM_LB,
            ub=NORM_UB,
            pop_size=30,
            max_iter=200,
            rng=rng,
        )
        logger.info(f"  EI = {-neg_ei_val:.6f} | θ_norm = {x_new.round(3)}")
        logger.info(f"  θ_phys = {_to_physical(x_new).round(2)}")

        # ── Step 13: 仿真评估 ────────────────────────────────────
        result = simulator.evaluate(_to_physical(x_new))
        if not result["feasible"]:
            logger.warning(f"  仿真不可行: {result['violation']}，使用惩罚值")
        y_new = result["raw_objectives"]
        logger.info(
            f"  y = [time={y_new[0]:.1f}s, ΔT={y_new[1]:.2f}K, aging={y_new[2]:.4f}%]"
        )

        # ── Step 14: 加入 D_T^M ──────────────────────────────────
        db_t.add(x_new, y_new)

        # ── 进度监控 ─────────────────────────────────────────────
        Y_all = np.vstack([db_llm.y, db_t.y])
        _, pf_idx = pareto_front(Y_all)
        logger.info(f"  Pareto 前沿点数 (D_LLM ∪ D_T): {len(pf_idx)}")

        iter_time = time.time() - t_iter
        logger.info(f"  本轮耗时: {iter_time:.1f}s")

        # 保存检查点
        db_t.save(output_dir / "db_t_checkpoint.json")

    return db_t


# ---------------------------------------------------------------------------
#  完整入口
# ---------------------------------------------------------------------------

def main(
    soh: float = 0.7,
    n_llm: int = 10,
    t_max: int = 30,
    skip_llm: bool = False,
    output_dir: str = "./results",
    seed: int = 42,
):
    t_start    = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║         LLMBO-MO  启动               ║")
    logger.info(f"║  SOH={soh:.2f} | n_llm={n_llm} | t_max={t_max}  ║")
    logger.info("╚══════════════════════════════════════╝")

    simulator = _load_simulator(soh)

    # ────────────────────────────────────────────────────────────────
    #  Phase 1: LLM WarmStart
    #  生成 N_LLM 个协议 → 评估 → D_LLM^M（全程固定）
    # ────────────────────────────────────────────────────────────────
    x_llm_norm = warm_start(
        n_llm=n_llm, soh=soh,
        output_dir=output_dir,
        skip_llm=skip_llm,
        rng=rng,
    )

    db_llm = Database(dim=5, n_obj=3)
    logger.info(f"\nPhase 1: 评估 {len(x_llm_norm)} 条 LLM 协议...")

    for i, xi in enumerate(x_llm_norm):
        result = simulator.evaluate(_to_physical(xi))
        db_llm.add(xi, result["raw_objectives"])
        logger.info(
            f"  [{i+1}/{len(x_llm_norm)}] "
            f"θ_phys={_to_physical(xi).round(2)} | "
            f"y={result['raw_objectives'].round(2)} | "
            f"feasible={result['feasible']}"
        )

    db_llm.save(output_dir / "db_llm.json")
    logger.info(f"\nD_LLM^M 完成 | {db_llm.summary()}")

    # ────────────────────────────────────────────────────────────────
    #  Phase 2: BO 主循环
    #  D_T^M 从空开始，每轮追加一条
    # ────────────────────────────────────────────────────────────────
    db_t = Database(dim=5, n_obj=3)

    db_t = bo_loop(
        db_llm=db_llm,
        db_t=db_t,
        simulator=simulator,
        t_max=t_max,
        rng=rng,
        output_dir=output_dir,
    )

    # ────────────────────────────────────────────────────────────────
    #  最终输出
    # ────────────────────────────────────────────────────────────────
    db_t.save(output_dir / "db_t_final.json")

    # 合并两库计算最终 Pareto 前沿
    Y_all = np.vstack([db_llm.y, db_t.y])
    X_all = np.vstack([db_llm.x, db_t.x])
    pf_y, pf_idx = pareto_front(Y_all)
    pf_x      = X_all[pf_idx]
    pf_x_phys = pf_x * (PHYS_UB - PHYS_LB) + PHYS_LB

    logger.info("\n" + "=" * 50)
    logger.info("优化完成")
    logger.info(f"总评估次数: {db_llm.n + db_t.n}  (D_LLM={db_llm.n}, D_T={db_t.n})")
    logger.info(f"Pareto 前沿点数: {len(pf_y)}")
    logger.info(f"总耗时: {time.time() - t_start:.1f}s")

    pareto_result = {
        "pareto_x_norm": pf_x.tolist(),
        "pareto_x_phys": pf_x_phys.tolist(),
        "pareto_y":      pf_y.tolist(),
        "dim_names":     ["I1(C)", "I2(C)", "I3(C)", "dSOC1", "dSOC2"],
        "obj_names":     ["time_s", "delta_temp_K", "aging_pct"],
        "n_llm_pts":     db_llm.n,
        "n_bo_pts":      db_t.n,
    }
    with open(output_dir / "pareto_front.json", "w") as f:
        json.dump(pareto_result, f, indent=2)
    logger.info(f"Pareto 前沿已保存至 {output_dir / 'pareto_front.json'}")

    return db_llm, db_t, pf_y, pf_x_phys


# ---------------------------------------------------------------------------
#  命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMBO-MO 多目标充电协议优化")
    parser.add_argument("--soh",        type=float, default=0.7)
    parser.add_argument("--n_llm",      type=int,   default=10,
                        help="LLM 生成的初始协议数量（D_LLM^M 大小）")
    parser.add_argument("--t_max",      type=int,   default=30,
                        help="BO 主循环迭代次数（D_T^M 最终大小）")
    parser.add_argument("--output_dir", type=str,   default="./results")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--skip_llm",   action="store_true",
                        help="跳过 LLM 调用，仅用 LHS（调试模式）")
    args = parser.parse_args()

    main(
        soh=args.soh,
        n_llm=args.n_llm,
        t_max=args.t_max,
        skip_llm=args.skip_llm,
        output_dir=args.output_dir,
        seed=args.seed,
    )
