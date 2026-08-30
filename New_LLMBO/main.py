"""
New_LLMBO/main.py
===================
LLM-MOBO 主程序（显式依赖注入版本）

设计目标:
1. 显式配置注入 - 不再使用全局 from config import ...
2. 配置与代码分离 - 所有超参数通过配置文件管理
3. 类型安全 - 使用 Pydantic Config 对象进行运行时校验

用法示例:
    # 方式 1: 使用默认配置
    python main.py --demo

    # 方式 2: 从 JSON 配置文件加载
    python main.py --config config.json

    # 方式 3: 命令行覆盖
    python main.py --config config.json --bo.n_iterations=100 --acquisition.n_cand=20
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 使用显式导入（不再使用 from config import ...）
from config.schema import Config, create_minimal_config
from config.load import load_config, parse_cli_overrides
from config.presets import EXPERIMENT_PRESETS
from utils.constants import DSOC_SUM_MAX


# ═══════════════════════════════════════════════════════════════════════════
# §A  命令行参数解析
# ═══════════════════════════════════════════════════════════════════════════

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="LLM-MOBO 贝叶斯优化器（显式配置注入版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置优先级（后者覆盖前者）:
  1. 默认配置（schema.py 中的 default_factory）
  2. JSON 配置文件（--config 指定）
  3. 环境变量（LLM_API_KEY, BO_N_ITERATIONS 等）
  4. 命令行覆盖（--bo.n_iterations=100 等）

示例:
  # 快速演示（Mock 模式）
  python main.py --demo

  # 从配置文件加载
  python main.py --config config.json

  # 命令行覆盖
  python main.py --config config.json --bo.n_iterations=100 --acquisition.n_cand=20

  # 生成配置模板
  python main.py --generate-template --output template_config.json
        """
    )

    # 配置文件
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="JSON 配置文件路径"
    )

    # 运行模式
    parser.add_argument(
        "--demo",
        action="store_true",
        help="演示模式（使用最小配置快速测试）"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock 模式（不调用真实 LLM）"
    )

    # 输出控制
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=[
            "warmstart_plain_ei",
            "warmstart_portfolio_plain_ei",
            "strict_baseline",
            "parego_baseline",
            "warmstart_safe_tiebreak",
            "warmstart_risk_veto",
            "warmstart_region_lifted_gp",
            "warmstart_region_lifted_gp_guarded_pool",
            "warmstart_region_lifted_gp_force_pool_tuned",
        ],
        help="Experiment preset. warmstart_plain_ei is the recommended mainline.",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="结果输出目录（默认：results）"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )

    # 配置模板生成
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="生成配置模板文件"
    )

    parser.add_argument(
        "--template-mode",
        type=str,
        choices=["full", "minimal"],
        default="full",
        help="模板模式（full/minimal）"
    )

    # 剩余参数为 CLI 覆盖（--bo.n_iterations=100 等）
    parser.add_argument(
        "overrides",
        nargs="*",
        help="配置覆盖参数（格式：--key=value 或 --key value）"
    )

    return parser


# ═══════════════════════════════════════════════════════════════════════════
# §B  配置加载
# ═══════════════════════════════════════════════════════════════════════════

def load_configuration(args: argparse.Namespace) -> Config:
    """
    根据命令行参数加载配置

    Args:
        args: 命令行参数

    Returns:
        Config: 配置对象
    """
    # Step 1: 生成模板（如果请求）
    if args.generate_template:
        from config.load import generate_config_template
        output_path = args.output if args.output.endswith('.json') else f"{args.output}/config_template.json"
        generate_config_template(output_path, mode=args.template_mode)
        print(f"配置模板已生成：{output_path}")
        sys.exit(0)

    # Step 2: 演示模式（最小配置）
    if args.demo:
        print("[配置] 使用最小配置（演示模式）")
        config = create_minimal_config(
            n_iterations=5,
            n_warmstart=3,
            n_candidates=5,
        )
        return config

    # Step 3: 解析 CLI 覆盖参数
    overrides = {}
    if args.overrides:
        overrides = parse_cli_overrides(args.overrides)

    # Step 4: Mock 模式配置
    if args.mock:
        overrides.setdefault('llm', {})['acquisition'] = {
            'gen_max_retries': 0,  # 不重试
        }

    # Step 5: 加载配置
    try:
        config = load_config(
            config_path=args.config,
            overrides=overrides,
            strict=True,
        )
        print(f"[配置] 已加载配置")
        print(f"  n_iterations: {config.bo.n_iterations}")
        print(f"  n_warmstart: {config.bo.n_warmstart}")
        print(f"  n_candidates: {config.acquisition.n_cand}")
        print(f"  LLM model: {config.llm.model}")
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        print("提示：使用 --generate-template 生成配置模板")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 配置加载失败：{e}")
        sys.exit(1)

    return config


# ═══════════════════════════════════════════════════════════════════════════
# §C  优化器运行
# ═══════════════════════════════════════════════════════════════════════════

def build_optimizer_config(config: Config, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    """从 Pydantic Config 构建优化器 flat dict.

    只传递 Config 中有的字段；BayesOptimizer.__init__ 会先用 DEFAULT_CONFIG 填充默认值，
    再用本函数的返回值覆盖。因此无需重复与 DEFAULT_CONFIG 相同的值。
    """
    preset = getattr(args, "preset", None)
    flat: Dict[str, Any] = {
        "experiment_preset": preset,
        # ── BO ──
        "max_iterations": config.bo.n_iterations,
        "n_warmstart": config.bo.n_warmstart,
        "n_random_init": config.bo.n_random_init,
        "warmstart_batch_size": config.bo.warmstart_batch_size,
        "warmstart_max_attempts": config.bo.warmstart_max_llm_attempts,
        "warmstart_hv_log_interval": config.bo.warmstart_hv_log_interval,
        # ── GP ──
        "kernel_nu": config.gp.kernel_nu,
        "gp_alpha": config.gp.alpha,
        "gp_normalize_y": config.gp.normalize_y,
        "gp_n_restarts_optimizer": config.gp.n_restarts_optimizer,
        # ── MOBO ──
        "eta": config.mobo.eta,
        "weight_count": config.mobo.n_weights,
        # ── Acquisition ──
        "n_candidates": config.acquisition.n_cand,
        "n_select": config.acquisition.n_select,
        # ── LLM ──
        "llm_backend": "mock" if getattr(args, "mock", False) else (config.llm.api_key and "openai" or "mock"),
        "llm_model": config.llm.model,
        "llm_api_base": config.llm.base_url,
        "llm_api_key": config.llm.api_key,
        "llm_n_samples": getattr(config.llm, "n_samples", 1),
        "llm_temperature": config.llm.warmstart.temperature,
        "battery_param_set": config.battery.param_set,
        "warmstart_context_level": config.llm.warmstart.context_level,
        "warmstart_max_tokens": config.llm.warmstart.max_tokens,
        "warmstart_max_retries": config.llm.warmstart.max_retries,
        "warmstart_temperature": config.llm.warmstart.temperature,
        # ── Charging range ──
        "soc_start": config.charging_range.soc0,
        "soc_end": config.charging_range.soc_end,
        "dsoc_sum_max": DSOC_SUM_MAX,
        # ── Checkpoint ──
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "checkpoint_every": config.data.save_interval,
    }
    if preset:
        if str(preset) not in EXPERIMENT_PRESETS:
            available = ", ".join(sorted(EXPERIMENT_PRESETS))
            raise ValueError(f"Unknown experiment preset '{preset}'. Available: {available}")
        flat.update(EXPERIMENT_PRESETS[str(preset)])
    return flat


async def run_optimization(config: Config, args: argparse.Namespace) -> None:
    """运行优化器"""
    output_dir = Path(args.output)
    if not args.demo:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[初始化] 创建优化器...")
    from llmbo.optimizer import BayesOptimizer
    optimizer = BayesOptimizer(config=build_optimizer_config(config, args, output_dir))

    # 运行优化
    print("\n[优化] 开始运行...")
    db = optimizer.run()

    # 保存结果
    print("\n[完成] 保存结果...")
    optimizer.save_results(str(output_dir))

    # 打印摘要
    print("\n" + db.summary())


# ═══════════════════════════════════════════════════════════════════════════
# §D  主函数
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 加载配置
    config = load_configuration(args)

    # 运行优化
    try:
        asyncio.run(run_optimization(config, args))
    except KeyboardInterrupt:
        print("\n[中断] 用户终止")
        sys.exit(0)
    except Exception as e:
        print(f"\n[错误] 优化失败：{e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
