"""
parego_runner.py — ParEGO 300 回合运行器
=========================================

ParEGO (Pareto Efficient Global Optimization) 长程优化运行器。
支持命令行配置，自动保存检查点和最终汇总文件。

与 LLAMBO-MO 对比：
  - 相同的 Tchebycheff 标量化函数（Eq.1）
  - 相同的动态 min-max 归一化（Eq.2b）
  - 相同的 Riesz s-energy 权重集合
  - 相同的 ObservationDB 和 HV 计算

与 LLAMBO-MO 不同：
  - 初始化：Latin Hypercube Sampling（无 LLM warmstart）
  - 代理模型：sklearn GaussianProcessRegressor（标准 Matern 5/2 核）
  - 采集函数：标准 EI（无 W_charge 加权）
  - 候选点生成：均匀随机采样（无 LLM 引导）

用法示例:
    # 运行 300 回合（默认配置）
    python parego_runner.py --output results_parego_300

    # 自定义迭代次数和 warmstart 点数
    python parego_runner.py --iterations 300 --warmstart 15 --output results_parego

    # 使用自定义随机种子
    python parego_runner.py --seed 42 --verbose

    # 快速测试（10 回合）
    python parego_runner.py --iterations 10 --demo
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional
from config.settings import Settings

# 导入 ParEGO 优化器
from llmbo.ParEGO import ParEGOOptimizer

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="ParEGO 300 回合优化运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行 300 回合（默认配置）
  python parego_runner.py --output results_parego_300

  # 自定义迭代次数
  python parego_runner.py --iterations 500 --warmstart 20

  # 快速测试（10 回合）
  python parego_runner.py --iterations 10 --demo

  # 指定随机种子和检查点频率
  python parego_runner.py --seed 42 --checkpoint-every 10 --verbose
        """
    )

    # 实验规模
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=None,
        help="优化迭代次数（默认：从 Settings.PAREGO.N_ITERATIONS 读取）"
    )

    parser.add_argument(
        "--warmstart", "-w",
        type=int,
        default=None,
        help="LHS 初始化点数（默认：从 Settings.PAREGO.N_WARMSTART 读取）"
    )

    parser.add_argument(
        "--n-cands",
        type=int,
        default=None,
        help="每迭代候选点数量（默认：从 Settings.PAREGO.N_RANDOM_CANDS 读取）"
    )

    # 输出配置
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results_parego_300",
        help="结果输出目录（默认：results_parego_300）"
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="每 N 回合保存检查点（默认：10）"
    )

    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）"
    )

    parser.add_argument(
        "--lhs-seed",
        type=int,
        default=None,
        help="LHS 初始化种子（默认：与 seed 相同）"
    )

    # 运行模式
    parser.add_argument(
        "--demo",
        action="store_true",
        help="演示模式（快速测试，10 回合）"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出（DEBUG 级别日志）"
    )

    return parser


def run_parego(
    iterations: int = None,
    warmstart: int = None,
    n_cands: int = None,
    output_dir: str = "results_parego_300",
    checkpoint_every: int = 10,
    seed: int = 42,
    lhs_seed: Optional[int] = None,
    demo: bool = False,
    verbose: bool = False,
) -> None:
    """
    运行 ParEGO 优化

    Args:
        iterations: 迭代次数
        warmstart: LHS 初始点数
        n_cands: 候选点数量
        output_dir: 输出目录
        checkpoint_every: 检查点保存频率
        seed: 随机种子
        lhs_seed: LHS 种子
        demo: 演示模式
        verbose: 详细输出
    """
    # 设置日志级别
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 使用 Settings 默认值（如果未指定）
    if iterations is None:
        iterations = Settings.PAREGO.N_ITERATIONS
    if warmstart is None:
        warmstart = Settings.PAREGO.N_WARMSTART
    if n_cands is None:
        n_cands = Settings.PAREGO.N_RANDOM_CANDS

    # 演示模式：快速测试
    if demo:
        logger.info("[模式] 演示模式：10 回合快速测试")
        iterations = 10
        warmstart = 5
        checkpoint_every = 5

    # 设置默认 LHS 种子
    if lhs_seed is None:
        lhs_seed = seed

    # 配置优化器
    config = {
        "max_iterations":   iterations,
        "n_warmstart":      warmstart,
        "n_random_cands":   n_cands,
        "checkpoint_dir":   str(Path(output_dir) / "checkpoints"),
        "checkpoint_every": checkpoint_every,
        "lhs_seed":         lhs_seed,
        "w_sample_seed":    seed,
        "cand_seed":        seed,
        "riesz_seed":       42,  # 固定，与 LLAMBO-MO 一致
    }

    # 打印配置
    logger.info("=" * 60)
    logger.info("ParEGO 优化配置")
    logger.info("=" * 60)
    logger.info(f"  迭代次数：     {iterations}")
    logger.info(f"  Warmstart 点数：{warmstart}")
    logger.info(f"  候选点数量：   {n_cands}")
    logger.info(f"  检查点频率：   每 {checkpoint_every} 回合")
    logger.info(f"  随机种子：     {seed}")
    logger.info(f"  LHS 种子：      {lhs_seed}")
    logger.info(f"  输出目录：     {output_dir}")
    logger.info("=" * 60)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("已创建输出目录：%s", output_dir)

    # 创建并运行优化器
    start_time = time.time()

    try:
        optimizer = ParEGOOptimizer(config=config)
        db = optimizer.run()

        # 保存结果
        optimizer.save_results(output_dir)

        # 保存最终汇总（供制图使用）
        optimizer.save_final_summary(output_dir)

        # 打印最终统计
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("ParEGO 优化完成!")
        logger.info("=" * 60)
        logger.info(f"  总运行时间：   {elapsed / 60:.1f} 分钟 ({elapsed:.1f} 秒)")
        logger.info(f"  总评估数：     {db.size}")
        logger.info(f"  可行解数：     {db.n_feasible}")
        logger.info(f"  Pareto 大小：  {db.pareto_size}")
        logger.info(f"  最终 HV:       {db.compute_hypervolume():.6f}")
        logger.info(f"  结果保存至：   {output_dir}/")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\n[中断] 用户终止优化")
        sys.exit(0)

    except Exception as e:
        logger.error("优化失败：%s", e)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    run_parego(
        iterations=args.iterations,
        warmstart=args.warmstart,
        n_cands=args.n_cands,
        output_dir=args.output,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        lhs_seed=args.lhs_seed,
        demo=args.demo,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()