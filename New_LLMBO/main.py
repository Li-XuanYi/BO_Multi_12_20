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
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 使用显式导入（不再使用 from config import ...）
from config.schema import Config, create_minimal_config, get_default_config
from config.load import load_config, parse_cli_overrides
from llmbo.optimizer import BayesOptimizer, EXPERIMENT_PRESETS
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
            "warmstart_region_lgbo_proposition1",
            "random_region_lgbo_proposition1",
            "baseline_plain_ei",
            "llm_region_lgbo_prior",
            "llm_region_lgbo_posterior",
            "llm_region_lgbo_posterior_calibrated",
            "llm_region_lgbo_posterior_adaptive",
            "random_region_lgbo_prior",
            "random_region_lgbo_posterior",
            "warmstart_lgbo_prior",
            "warmstart_lgbo_posterior",
            "llmbo_mo",
            "LLMBO-MO",
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
    preset = getattr(args, "preset", None)
    flat: Dict[str, Any] = {
        "experiment_preset": preset,
        "max_iterations": config.bo.n_iterations,
        "n_warmstart": config.bo.n_warmstart,
        "n_random_init": getattr(config.bo, "n_random_init", 3),
        "n_candidates": config.acquisition.n_cand,
        "n_select": config.acquisition.n_select,
        "llm_backend": "mock" if getattr(args, "mock", False) else (config.llm.api_key and "openai" or "mock"),
        "llm_model": config.llm.model,
        "llm_api_base": config.llm.base_url,
        "llm_api_key": config.llm.api_key,
        "llm_n_samples": getattr(config.llm, "n_samples", 1),
        "llm_temperature": config.llm.warmstart.temperature,
        "battery_param_set": config.battery.param_set,
        "warmstart_context_level": config.llm.warmstart.context_level,
        "warmstart_max_tokens": config.llm.warmstart.max_tokens,
        "region_preference_max_tokens": 4096,
        "region_preference_prompt_version": "default",
        "warmstart_max_retries": config.llm.warmstart.max_retries,
        "warmstart_temperature": config.llm.warmstart.temperature,
        "enable_warmstart_portfolio": True,
        "warmstart_pool_size": 16,
        "warmstart_cache_path": None,
        "warmstart_cache_mode": "read_write",
        "warmstart_cache_use_selected": False,
        "soc_start": config.charging_range.soc0,
        "soc_end": config.charging_range.soc_end,
        "dsoc_sum_max": DSOC_SUM_MAX,
        "enable_iterative_guidance": False,
        "enable_gp_llm_coupling": False,
        "enable_acq_prior_coupling": False,
        "enable_proposal_sampler": False,
        "enable_llm_rerank": False,
        "llm_rerank_mode": "none",
        "llm_rerank_top_m": 5,
        "llm_rerank_parse_fail_open": True,
        "target_transform_mode": "none",
        "objective_preprocess_mode": "minmax",
        "weight_strategy": "riesz_relaxed_cycle",
        "weight_simplex_divisions": 10,
        "weight_count": 30,
        "acquisition_strategy": "ei_lbfgsb",
        "parego_lcb_variance_weight": 0.5,
        "parego_de_population": 30,
        "parego_de_maxiter": 200,
        "enable_region_lifted_gp": False,
        "region_lift_mode": "heuristic_correlation",
        "region_lift_control_mode": "none",
        "region_lift_random_width_norm": 0.15,
        "region_lift_random_confidence": 0.5,
        "region_lift_apply_override": False,
        "region_lift_external_influence_mode": "diagnostic_only",
        "region_lift_include_raw_candidates": True,
        "region_lift_lambda_max": 0.25,
        "region_lift_min_confidence": 0.60,
        "region_lift_n_anchors": 32,
        "region_lift_max_shift_std": 0.25,
        "region_lift_active_until": 12,
        "region_lift_anneal": "linear_decay",
        "region_lift_max_plain_ei_gap": 0.25,
        "region_lift_log_ei_eps": 1e-12,
        "region_lift_kernel_jitter": 1e-6,
        "region_lift_min_norm_sq": 1e-12,
        "region_lift_min_volume": 1e-5,
        "region_lift_max_volume": 0.25,
        "region_lift_min_width": 0.03,
        "region_lift_max_width": 0.80,
        "region_lift_close_distance": 0.05,
        "region_lift_max_close_fraction": 0.5,
        "region_lift_min_feasible_anchor_ratio": 0.6,
        "region_lift_near_region_tol": 0.05,
        "region_lift_trust_init": 0.5,
        "region_lift_trust_beta": 0.2,
        "region_lift_anchor_weighting": "ei_softmax",
        "region_lift_anchor_temperature": 0.35,
        "region_lift_require_inside": True,
        "region_lift_min_sigma_ratio": 0.85,
        "region_lift_candidate_oversample": 8,
        "region_lift_point_current_probe_levels": 0,
        "region_lift_point_current_probe_keep": 0,
        "region_lift_dsoc_margin": 0.02,
        "region_lift_guard_min_anchor_consistency": 0.35,
        "region_lift_guard_min_reliability": 0.20,
        "region_lift_guard_max_plain_ei_gap": 0.25,
        "region_lift_guard_require_inside": True,
        "region_lift_guard_require_positive_corr": True,
        "region_lift_lgbo_min_variance": 1e-12,
        "region_lift_lgbo_shift_source": "prior_kernel",
        "region_lift_confidence_scale": 1.0,
        "region_lift_adaptive_confidence_enabled": False,
        "region_lift_adaptive_confidence_floor": 0.35,
        "region_lift_adaptive_base_scale": 0.85,
        "region_lift_adaptive_width_min_factor": 0.80,
        "region_lift_adaptive_repeat_min_factor": 0.85,
        "region_lift_adaptive_late_min_factor": 0.85,
        "region_lift_adaptive_width_start": 0.30,
        "region_lift_adaptive_repeat_distance": 0.18,
        "region_lift_lgbo_shift_mean_budget": 0.0,
        "ei_n_external_restarts": 16,
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
    """
    运行优化器

    Args:
        config: 配置对象
        args: 命令行参数
    """
    # 创建结果目录
    output_dir = Path(args.output)
    if not args.demo:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 创建优化器（显式注入配置）
    print("\n[初始化] 创建优化器...")

    # 从配置构建优化器参数字典
    optimizer_kwargs = {
        'config': {
            'experiment_preset': args.preset,
            'max_iterations': config.bo.n_iterations,
            'n_warmstart': config.bo.n_warmstart,
            'n_random_init': getattr(config.bo, 'n_random_init', 3),
            'n_candidates': config.acquisition.n_cand,
            'n_select': config.acquisition.n_select,

            # LLM 配置
            'llm_backend': 'mock' if args.mock else config.llm.api_key and 'openai' or 'mock',
            'llm_model': config.llm.model,
            'llm_api_base': config.llm.base_url,
            'llm_api_key': config.llm.api_key,
            'llm_n_samples': config.llm.n_samples if hasattr(config.llm, 'n_samples') else 1,
            'llm_temperature': config.llm.warmstart.temperature,
            'battery_param_set': config.battery.param_set,
            'warmstart_context_level': config.llm.warmstart.context_level,
            'warmstart_max_tokens': config.llm.warmstart.max_tokens,
            'region_preference_max_tokens': 4096,
            'warmstart_max_retries': config.llm.warmstart.max_retries,
            'warmstart_temperature': config.llm.warmstart.temperature,
            'soc_start': config.charging_range.soc0,
            'soc_end': config.charging_range.soc_end,
            'dsoc_sum_max': DSOC_SUM_MAX,

            # Legacy GP/AF schema keys are intentionally not forwarded here.
            # 检查点
            'checkpoint_dir': str(output_dir / 'checkpoints'),
            'checkpoint_every': config.data.save_interval,
        }
    }
    if args.preset:
        # Let BayesOptimizer's named preset own the mainline/research flags.
        for key in (
            'n_warmstart',
            'n_random_init',
            'enable_iterative_guidance',
            'enable_gp_llm_coupling',
            'enable_acq_prior_coupling',
            'enable_proposal_sampler',
            'enable_llm_rerank',
        ):
            optimizer_kwargs['config'].pop(key, None)

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
