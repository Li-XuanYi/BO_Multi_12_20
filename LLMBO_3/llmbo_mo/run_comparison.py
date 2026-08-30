"""
LLMBO-MO vs NSGA-II 对比实验主脚本

运行多个独立实验，收集 HV 曲线和 Pareto 前沿数据，
导出到 xlsx 文件用于后续制图分析。

Usage:
    python run_comparison.py --runs 5 --output comparison_results.xlsx

实验配置:
    - 两种算法使用相同的评估预算
    - 每次运行使用不同的随机种子
    - 记录完整的优化过程数据
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from config import MOLLMBOConfig
from pareto import compute_hypervolume, log_transform_objectives
from battery_model import filter_valid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  实验配置
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """实验配置参数"""
    # 运行配置
    n_runs: int = 5              # 独立运行次数
    verbose: bool = True         # 打印进度

    # MO-LLMBO 配置
    llmbo_n_init: int = 15       # 暖启动评估数
    llmbo_t_max: int = 50        # BO 迭代次数
    llmbo_batch_size: int = 3    # 批次大小
    llmbo_use_llm: bool = True   # 是否使用 LLM

    # NSGA-II 配置
    nsgaii_pop_size: int = 50    # 种群大小

    # 通用配置
    random_seed_base: int = 42   # 基础随机种子
    n_workers: int = 3           # PyBaMM 并行进程数

    @property
    def total_evals(self) -> int:
        """总评估次数"""
        return self.llmbo_n_init + self.llmbo_t_max * self.llmbo_batch_size


@dataclass
class RunResult:
    """单次运行的结果"""
    run_id: int
    algorithm: str  # "LLMBO" or "NSGAII"
    seed: int

    # HV 曲线
    hv_history: list[float] = field(default_factory=list)
    n_evals: list[int] = field(default_factory=list)

    # 最终 Pareto 前沿
    pareto_thetas: list[list[float]] = field(default_factory=list)
    pareto_objectives: list[list[float]] = field(default_factory=list)

    # 统计信息
    final_hv: float = 0.0
    n_pareto: int = 0
    n_valid: int = 0
    total_time: float = 0.0
    convergence_evals: int = 0  # 达到 90% 最终 HV 所需评估数


# ══════════════════════════════════════════════════════════════════════════════
#  运行 LLMBO-MO
# ══════════════════════════════════════════════════════════════════════════════

def run_llmbo(cfg: MOLLMBOConfig, seed: int, verbose: bool = True) -> RunResult:
    """
    运行一次 MO-LLMBO 优化

    Args:
        cfg: 配置对象
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        RunResult 对象
    """
    from optimizer import MOLLMBOptimizer

    result = RunResult(
        run_id=seed,
        algorithm="LLMBO",
        seed=seed,
    )

    t_start = time.time()

    # 创建并运行优化器
    config = MOLLMBOConfig(
        use_llm=cfg.llmbo_use_llm,
        use_gek=True,
        use_dpp=True,
        use_turbo=True,
        n_init=cfg.llmbo_n_init,
        t_max=cfg.llmbo_t_max,
        batch_size=cfg.llmbo_batch_size,
        n_workers=cfg.n_workers,
        random_seed=seed,
    )

    optimizer = MOLLMBOptimizer(config)
    db, state = optimizer.run()

    result.total_time = time.time() - t_start

    # 提取 HV 曲线
    result.hv_history = list(db.hv_history)

    # 计算每次迭代的评估次数
    result.n_evals = [
        cfg.llmbo_n_init + i * cfg.llmbo_batch_size
        for i in range(len(result.hv_history))
    ]

    # 提取最终 Pareto 前沿
    pf_thetas = db.pareto_front_thetas()
    pf_objs = db.pareto_front_objs()

    if len(pf_objs) > 0:
        result.pareto_thetas = pf_thetas.tolist()
        result.pareto_objectives = pf_objs.tolist()
        result.n_pareto = len(pf_objs)

    # 最终 HV
    if result.hv_history:
        result.final_hv = result.hv_history[-1]

    # 统计有效评估数
    result.n_valid = db.n_valid

    # 计算收敛评估数（达到 90% 最终 HV）
    if result.final_hv > 0:
        threshold = 0.9 * result.final_hv
        for i, hv in enumerate(result.hv_history):
            if hv >= threshold:
                result.convergence_evals = result.n_evals[i] if i < len(result.n_evals) else 0
                break

    if verbose:
        log.info(f"LLMBO 运行完成 (seed={seed}): HV={result.final_hv:.4f}, "
                 f"Pareto 点数={result.n_pareto}, 时间={result.total_time:.1f}s")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  运行 NSGA-II
# ══════════════════════════════════════════════════════════════════════════════

def run_nsgaii_wrapper(cfg: MOLLMBOConfig, seed: int, verbose: bool = True) -> RunResult:
    """
    运行一次 NSGA-II 优化

    Args:
        cfg: 配置对象
        seed: 随机种子
        verbose: 是否打印进度

    Returns:
        RunResult 对象
    """
    from nsgaii_optimizer import NSGAIOptimizer

    result = RunResult(
        run_id=seed,
        algorithm="NSGAII",
        seed=seed,
    )

    t_start = time.time()

    # 创建配置
    config = MOLLMBOConfig(
        random_seed=seed,
    )

    # 计算 NSGA-II 代数，使总评估数与 LLMBO 相近
    total_evals = cfg.llmbo_n_init + cfg.llmbo_t_max * cfg.llmbo_batch_size
    pop_size = cfg.nsgaii_pop_size
    n_gen = max(1, total_evals // pop_size)

    # 创建并运行优化器
    optimizer = NSGAIOptimizer(config, pop_size=pop_size, seed=seed)
    optimizer.run(verbose=verbose)
    optimizer.n_gen = n_gen  # Update for correct history

    result.total_time = time.time() - t_start

    # 提取 HV 历史
    result.hv_history = list(optimizer.hv_history)

    # 计算评估次数（每代）
    result.n_evals = [
        (i + 1) * pop_size
        for i in range(len(result.hv_history))
    ]

    # 提取最终 Pareto 前沿
    pf_X, pf_F = optimizer.get_pareto_front()

    if len(pf_F) > 0:
        result.pareto_thetas = pf_X.tolist()
        result.pareto_objectives = pf_F.tolist()
        result.n_pareto = len(pf_F)

    # 最终 HV
    if result.hv_history:
        result.final_hv = result.hv_history[-1]

    # 总评估数
    result.n_valid = optimizer.problem.eval_count

    # 计算收敛评估数
    if result.final_hv > 0:
        threshold = 0.9 * result.final_hv
        for i, hv in enumerate(result.hv_history):
            if hv >= threshold:
                result.convergence_evals = result.n_evals[i] if i < len(result.n_evals) else 0
                break

    if verbose:
        log.info(f"NSGA-II 运行完成 (seed={seed}): HV={result.final_hv:.4f}, "
                 f"Pareto 点数={result.n_pareto}, 时间={result.total_time:.1f}s")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  主对比实验
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison(
    exp_cfg: ExperimentConfig,
    output_path: str = "comparison_results.xlsx",
) -> dict[str, list[RunResult]]:
    """
    运行完整的对比实验

    Args:
        exp_cfg: 实验配置
        output_path: xlsx 输出路径

    Returns:
        {"LLMBO": [results], "NSGAII": [results]}
    """
    print("=" * 70)
    print("LLMBO-MO vs NSGA-II 对比实验")
    print("=" * 70)
    print(f"独立运行次数：{exp_cfg.n_runs}")
    print(f"总评估预算：{exp_cfg.total_evals} 每运行")
    print(f"LLMBO 配置：n_init={exp_cfg.llmbo_n_init}, t_max={exp_cfg.llmbo_t_max}, "
          f"batch={exp_cfg.llmbo_batch_size}, use_llm={exp_cfg.llmbo_use_llm}")
    print(f"NSGA-II 配置：pop_size={exp_cfg.nsgaii_pop_size}")
    print("=" * 70)

    all_results = {
        "LLMBO": [],
        "NSGAII": [],
    }

    for run_idx in range(exp_cfg.n_runs):
        seed = exp_cfg.random_seed_base + run_idx
        print(f"\n{'='*70}")
        print(f"运行 {run_idx + 1}/{exp_cfg.n_runs} (seed={seed})")
        print(f"{'='*70}")

        # 运行 LLMBO-MO
        print("\n[1/2] 运行 LLMBO-MO...")
        try:
            llmbo_result = run_llmbo(exp_cfg, seed, verbose=exp_cfg.verbose)
            all_results["LLMBO"].append(llmbo_result)
            print(f"  ✓ LLMBO 完成：HV={llmbo_result.final_hv:.4f}, "
                  f"Pareto={llmbo_result.n_pareto}, 时间={llmbo_result.total_time:.1f}s")
        except Exception as e:
            log.error(f"LLMBO 运行失败：{e}")
            print(f"  ✗ LLMBO 失败：{e}")

        # 运行 NSGA-II
        print("\n[2/2] 运行 NSGA-II...")
        try:
            nsgaii_result = run_nsgaii_wrapper(exp_cfg, seed, verbose=exp_cfg.verbose)
            all_results["NSGAII"].append(nsgaii_result)
            print(f"  ✓ NSGA-II 完成：HV={nsgaii_result.final_hv:.4f}, "
                  f"Pareto={nsgaii_result.n_pareto}, 时间={nsgaii_result.total_time:.1f}s")
        except Exception as e:
            log.error(f"NSGA-II 运行失败：{e}")
            print(f"  ✗ NSGA-II 失败：{e}")

    # 导出结果
    print(f"\n{'='*70}")
    print(f"导出结果到 {output_path}...")
    export_to_xlsx(all_results, exp_cfg, output_path)

    # 打印汇总统计
    print_summary(all_results, exp_cfg)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
#  导出到 xlsx
# ══════════════════════════════════════════════════════════════════════════════

def export_to_xlsx(
    all_results: dict[str, list[RunResult]],
    exp_cfg: ExperimentConfig,
    output_path: str,
):
    """导出所有结果到 xlsx 文件"""
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas 未安装，尝试使用 openpyxl 直接写入")
        _export_xlsx_openpyxl(all_results, exp_cfg, output_path)
        return

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: HV 曲线
        hv_data = []
        for algo, results in all_results.items():
            for r in results:
                for i, (hv, n_eval) in enumerate(zip(r.hv_history, r.n_evals)):
                    hv_data.append({
                        "Run_ID": r.seed,
                        "Algorithm": algo,
                        "Iteration": i,
                        "N_Evaluations": n_eval,
                        "HV": hv,
                    })
        hv_df = pd.DataFrame(hv_data)
        hv_df.to_excel(writer, sheet_name="HV_Curves", index=False)

        # Sheet 2: LLMBO Pareto 前沿
        llmbo_pf = []
        for r in all_results["LLMBO"]:
            for theta, obj in zip(r.pareto_thetas, r.pareto_objectives):
                llmbo_pf.append({
                    "Run_ID": r.seed,
                    "I1": theta[0], "I2": theta[1], "I3": theta[2],
                    "SOC_sw1": theta[3], "SOC_sw2": theta[4],
                    "V_CV": theta[5], "I_cutoff": theta[6],
                    "t_charge": obj[0], "T_peak": obj[1], "delta_Q": obj[2],
                })
        if llmbo_pf:
            pd.DataFrame(llmbo_pf).to_excel(writer, sheet_name="Pareto_LLMBO", index=False)
        else:
            pd.DataFrame(columns=["Run_ID", "I1", "I2", "I3", "SOC_sw1", "SOC_sw2",
                                  "V_CV", "I_cutoff", "t_charge", "T_peak", "delta_Q"]
                        ).to_excel(writer, sheet_name="Pareto_LLMBO", index=False)

        # Sheet 3: NSGA-II Pareto 前沿
        nsgaii_pf = []
        for r in all_results["NSGAII"]:
            for theta, obj in zip(r.pareto_thetas, r.pareto_objectives):
                nsgaii_pf.append({
                    "Run_ID": r.seed,
                    "I1": theta[0], "I2": theta[1], "I3": theta[2],
                    "SOC_sw1": theta[3], "SOC_sw2": theta[4],
                    "V_CV": theta[5], "I_cutoff": theta[6],
                    "t_charge": obj[0], "T_peak": obj[1], "delta_Q": obj[2],
                })
        if nsgaii_pf:
            pd.DataFrame(nsgaii_pf).to_excel(writer, sheet_name="Pareto_NSGAII", index=False)
        else:
            pd.DataFrame(columns=["Run_ID", "I1", "I2", "I3", "SOC_sw1", "SOC_sw2",
                                  "V_CV", "I_cutoff", "t_charge", "T_peak", "delta_Q"]
                        ).to_excel(writer, sheet_name="Pareto_NSGAII", index=False)

        # Sheet 4: 收敛指标
        conv_data = []
        for algo, results in all_results.items():
            for r in results:
                conv_data.append({
                    "Run_ID": r.seed,
                    "Algorithm": algo,
                    "Final_HV": r.final_hv,
                    "Convergence_Evals": r.convergence_evals,
                    "Total_Time_s": r.total_time,
                    "N_Pareto": r.n_pareto,
                    "N_Valid": r.n_valid,
                })
        pd.DataFrame(conv_data).to_excel(writer, sheet_name="Convergence", index=False)

        # Sheet 5: 统计汇总
        stats_data = []
        for algo, results in all_results.items():
            if not results:
                continue
            hvs = [r.final_hv for r in results if r.final_hv > 0]
            times = [r.total_time for r in results]
            pareto_sizes = [r.n_pareto for r in results]
            conv_evals = [r.convergence_evals for r in results if r.convergence_evals > 0]

            stats_data.append({
                "Algorithm": algo,
                "N_Runs": len(results),
                "HV_Mean": np.mean(hvs) if hvs else 0,
                "HV_Std": np.std(hvs) if hvs else 0,
                "HV_Best": np.max(hvs) if hvs else 0,
                "HV_Worst": np.min(hvs) if hvs else 0,
                "Time_Mean_s": np.mean(times) if times else 0,
                "Time_Std_s": np.std(times) if times else 0,
                "Pareto_Size_Mean": np.mean(pareto_sizes) if pareto_sizes else 0,
                "Convergence_Evals_Mean": np.mean(conv_evals) if conv_evals else 0,
            })
        pd.DataFrame(stats_data).to_excel(writer, sheet_name="Statistics", index=False)

        # Sheet 6: 实验配置
        config_data = {
            "n_runs": exp_cfg.n_runs,
            "total_evals": exp_cfg.total_evals,
            "llmbo_n_init": exp_cfg.llmbo_n_init,
            "llmbo_t_max": exp_cfg.llmbo_t_max,
            "llmbo_batch_size": exp_cfg.llmbo_batch_size,
            "llmbo_use_llm": exp_cfg.llmbo_use_llm,
            "nsgaii_pop_size": exp_cfg.nsgaii_pop_size,
            "random_seed_base": exp_cfg.random_seed_base,
            "timestamp": datetime.now().isoformat(),
        }
        config_df = pd.DataFrame(list(config_data.items()), columns=["Parameter", "Value"])
        config_df.to_excel(writer, sheet_name="Config", index=False)

    log.info(f"结果已导出到：{output_path}")


def _export_xlsx_openpyxl(
    all_results: dict[str, list[RunResult]],
    exp_cfg: ExperimentConfig,
    output_path: str,
):
    """使用 openpyxl 直接写入（pandas 不可用时的备选方案）"""
    try:
        from openpyxl import Workbook
    except ImportError:
        log.error("openpyxl 未安装，无法导出 xlsx")
        print("错误：请安装 pandas 或 openpyxl: pip install pandas openpyxl")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "HV_Curves"

    # 写入 HV 曲线
    header = ["Run_ID", "Algorithm", "Iteration", "N_Evaluations", "HV"]
    ws.append(header)

    for algo, results in all_results.items():
        for r in results:
            for i, (hv, n_eval) in enumerate(zip(r.hv_history, r.n_evals)):
                ws.append([r.seed, algo, i, n_eval, hv])

    # 创建其他 Sheet
    for sheet_name in ["Pareto_LLMBO", "Pareto_NSGAII", "Convergence", "Statistics"]:
        wb.create_sheet(sheet_name)

    wb.save(output_path)
    log.info(f"结果已导出到：{output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  打印汇总统计
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: dict[str, list[RunResult]], exp_cfg: ExperimentConfig):
    """打印对比实验的汇总统计"""
    print("\n" + "=" * 70)
    print("对比实验汇总")
    print("=" * 70)

    for algo, results in all_results.items():
        if not results:
            continue

        hvs = [r.final_hv for r in results if r.final_hv > 0]
        times = [r.total_time for r in results]
        pareto_sizes = [r.n_pareto for r in results]

        print(f"\n{algo} ({len(results)} 次运行):")
        print(f"  HV 指标:")
        print(f"    平均值：{np.mean(hvs):.6f} ± {np.std(hvs):.6f}")
        print(f"    最优值：{np.max(hvs):.6f}")
        print(f"    最劣值：{np.min(hvs):.6f}")
        print(f"  运行时间：{np.mean(times):.1f} ± {np.std(times):.1f} 秒")
        print(f"  Pareto 点数：{np.mean(pareto_sizes):.1f} ± {np.std(pareto_sizes):.1f}")

    # 显著性比较
    if len(all_results["LLMBO"]) > 0 and len(all_results["NSGAII"]) > 0:
        llmbo_hvs = [r.final_hv for r in all_results["LLMBO"] if r.final_hv > 0]
        nsgaii_hvs = [r.final_hv for r in all_results["NSGAII"] if r.final_hv > 0]

        if llmbo_hvs and nsgaii_hvs:
            llmbo_mean = np.mean(llmbo_hvs)
            nsgaii_mean = np.mean(nsgaii_hvs)
            improvement = (llmbo_mean - nsgaii_mean) / nsgaii_mean * 100 if nsgaii_mean > 0 else 0

            print(f"\n性能对比:")
            print(f"  LLMBO 平均 HV 优于 NSGA-II: {improvement:+.2f}%")
            if improvement > 0:
                print(f"  ✓ LLMBO 表现更好")
            else:
                print(f"  ✗ NSGA-II 表现更好")

    print("\n" + "=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLMBO-MO vs NSGA-II 对比实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=5,
        help="独立运行次数"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="comparison_results.xlsx",
        help="输出 xlsx 文件路径"
    )

    parser.add_argument(
        "--llmbo-n-init",
        type=int,
        default=15,
        help="LLMBO 暖启动评估数"
    )

    parser.add_argument(
        "--llmbo-t-max",
        type=int,
        default=50,
        help="LLMBO BO 迭代次数"
    )

    parser.add_argument(
        "--llmbo-batch",
        type=int,
        default=3,
        help="LLMBO 批次大小"
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="使用 LLM 指导"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="不使用 LLM 指导 (消融实验)"
    )

    parser.add_argument(
        "--nsgaii-pop",
        type=int,
        default=50,
        help="NSGA-II 种群大小"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="基础随机种子"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="PyBaMM 并行进程数"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，减少输出"
    )

    args = parser.parse_args()

    # 创建实验配置
    exp_cfg = ExperimentConfig(
        n_runs=args.runs,
        llmbo_n_init=args.llmbo_n_init,
        llmbo_t_max=args.llmbo_t_max,
        llmbo_batch_size=args.llmbo_batch,
        llmbo_use_llm=not args.no_llm,
        nsgaii_pop_size=args.nsgaii_pop,
        random_seed_base=args.seed,
        n_workers=args.workers,
        verbose=not args.quiet,
    )

    # 运行对比实验
    run_comparison(exp_cfg, output_path=args.output)


if __name__ == "__main__":
    main()
