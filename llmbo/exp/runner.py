"""
统一实验执行器
调度消融变体(V0-V6) + Baseline(Random/NSGA2/MOEAD/ParEGO/Sobol+GP)
支持：断点续跑、单方法重跑、结果自动收集
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, Optional

# 项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PARAM_BOUNDS, BATTERY_CONFIG
from battery_env.wrapper import BatterySimulator
from exp.configs import (
    SEEDS, DEFAULT_BUDGET, REFERENCE_POINT,
    ABLATION_CONFIGS, BASELINE_CONFIGS
)
from exp.baselines import run_baseline
try:
    from pymoo.config import Config
    Config.warnings['not_compiled'] = False
except Exception:
    pass

def _get_result_dir(base_dir: str = "./results") -> str:
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _result_path(base_dir: str, method: str, seed: int) -> str:
    method_dir = os.path.join(base_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    return os.path.join(method_dir, f"seed_{seed}.json")


def _already_done(base_dir: str, method: str, seed: int) -> bool:
    path = _result_path(base_dir, method, seed)
    return os.path.exists(path)


def _save_result(base_dir: str, method: str, seed: int, result: Dict):
    path = _result_path(base_dir, method, seed)
    # numpy数组转list
    serializable = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            serializable[k] = v  # evaluations列表直接存
        else:
            serializable[k] = v
    
    serializable["_meta"] = {
        "method": method,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"  ✓ 已保存: {path}")


# ============================================================
# 消融变体执行（V0-V6走LLMMOBO）
# ============================================================
async def run_ablation_variant(
    method_name: str,
    seed: int,
    llm_api_key: str = None,
    base_dir: str = "./results"
) -> Dict:
    """运行单个消融变体"""
    from main import LLMMOBO
    
    config = ABLATION_CONFIGS[method_name]
    budget = DEFAULT_BUDGET
    
    ref_point = np.array([REFERENCE_POINT["time"], REFERENCE_POINT["temp"], REFERENCE_POINT["aging"]])
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 确定是否需要LLM
    needs_llm = any([
        config["use_warmstart"],
        config["use_llm_acq"],
        config.get("use_llm_weighting", False),
    ])
    api_key = llm_api_key if needs_llm else None
    
    print(f"\n{'='*60}")
    print(f"[消融] {method_name} | seed={seed}")
    print(f"  配置: {config}")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    optimizer = LLMMOBO(
        llm_api_key=api_key,
        n_warmstart=budget["n_warmstart"],
        n_random_init=budget["n_random_init"],
        n_iterations=budget["n_iterations"],
        gamma_init=config["gamma_init"],
        verbose=True,
        use_coupling=config["use_coupling"],
        use_warmstart=config["use_warmstart"],
        use_llm_sampling=config["use_llm_sampling"],   # 与 LLMMOBO 一致
        use_adaptive_W=config["use_adaptive_W"],        # 与 LLMMOBO 一致
        gamma_adaptive=config.get("gamma_adaptive", True),
        db_path=":memory:"
    )
    
    results = await optimizer.optimize()
    wall_time = time.time() - t0
    
    # 整理输出
    output = {
        "evaluations": results["database"],
        "hv_history": results["hv_history"],
        "pareto_front": results["pareto_front"],
        "wall_time": wall_time,
        "n_valid": results["n_valid"],
        "n_violations": results["n_evaluations"] - results["n_valid"],
    }
    
    _save_result(base_dir, method_name, seed, output)
    return output


# ============================================================
# Baseline执行
# ============================================================
def run_baseline_method(
    method_name: str,
    seed: int,
    base_dir: str = "./results"
) -> Dict:
    """运行单个baseline"""
    config = BASELINE_CONFIGS[method_name]
    
    ref_point = np.array([REFERENCE_POINT["time"], REFERENCE_POINT["temp"], REFERENCE_POINT["aging"]])
    
    print(f"\n{'='*60}")
    print(f"[Baseline] {method_name} | seed={seed}")
    print(f"{'='*60}")
    
    simulator = BatterySimulator(
        param_set=BATTERY_CONFIG["param_set"],
        init_voltage=BATTERY_CONFIG["init_voltage"],
        init_temp=BATTERY_CONFIG["init_temp"],
        sample_time=BATTERY_CONFIG["sample_time"],
        voltage_max=BATTERY_CONFIG["voltage_max"],
        temp_max=BATTERY_CONFIG["temp_max"],
        soc_target=BATTERY_CONFIG["soc_target"]
    )
    
    np.random.seed(seed)
    
    result = run_baseline(
        method_type=config["type"],
        simulator=simulator,
        seed=seed,
        config=config,
        param_bounds=PARAM_BOUNDS,
        ref_point=ref_point
    )
    
    _save_result(base_dir, method_name, seed, result)
    return result


# ============================================================
# 批量执行
# ============================================================
async def run_all_ablations(
    llm_api_key: str = None,
    seeds: list = None,
    methods: list = None,
    base_dir: str = "./results",
    skip_existing: bool = True
):
    """运行所有消融变体"""
    seeds = seeds or SEEDS
    methods = methods or list(ABLATION_CONFIGS.keys())
    
    for method in methods:
        for seed in seeds:
            if skip_existing and _already_done(base_dir, method, seed):
                print(f"  [跳过] {method}/seed_{seed} 已存在")
                continue
            try:
                await run_ablation_variant(method, seed, llm_api_key, base_dir)
            except Exception as e:
                print(f"  [失败] {method}/seed_{seed}: {e}")
                import traceback
                traceback.print_exc()


def run_all_baselines(
    seeds: list = None,
    methods: list = None,
    base_dir: str = "./results",
    skip_existing: bool = True
):
    """运行所有baseline"""
    seeds = seeds or SEEDS
    methods = methods or list(BASELINE_CONFIGS.keys())
    
    for method in methods:
        for seed in seeds:
            if skip_existing and _already_done(base_dir, method, seed):
                print(f"  [跳过] {method}/seed_{seed} 已存在")
                continue
            try:
                run_baseline_method(method, seed, base_dir)
            except Exception as e:
                print(f"  [失败] {method}/seed_{seed}: {e}")
                import traceback
                traceback.print_exc()


# ============================================================
# 入口
# ============================================================
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="PE-GenBO 实验执行器")
    parser.add_argument("--mode", choices=["ablation", "baseline", "all"], default="all")
    parser.add_argument("--methods", nargs="*", default=None, help="指定方法名，如 V0_Full V1_NoWarmStart")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--result-dir", default="./results")
    parser.add_argument("--no-skip", action="store_true", help="不跳过已有结果")
    args = parser.parse_args()
    
    llm_api_key = os.getenv("LLM_API_KEY", None)
    skip = not args.no_skip
    
    if args.mode in ("ablation", "all"):
        await run_all_ablations(
            llm_api_key=llm_api_key,
            seeds=args.seeds,
            methods=args.methods if args.mode == "ablation" else None,
            base_dir=args.result_dir,
            skip_existing=skip
        )
    
    if args.mode in ("baseline", "all"):
        run_all_baselines(
            seeds=args.seeds,
            methods=args.methods if args.mode == "baseline" else None,
            base_dir=args.result_dir,
            skip_existing=skip
        )


if __name__ == "__main__":
    asyncio.run(main())
