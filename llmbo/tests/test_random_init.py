"""
PE-GenBO 超参数灵敏度分析（精简快速版）

修复原版3个致命问题：
1. 每次实验生成2张matplotlib图 → 内存泄漏导致死机
2. LLM未启用时仍扫描LLM参数 → 浪费时间
3. 无增量保存 → 崩溃后丢失全部结果

用法：
    python hyperparam_sweep_fast.py                     # 扫描全部有效参数
    python hyperparam_sweep_fast.py --param gamma_init  # 只扫单个
    python hyperparam_sweep_fast.py --quick              # 快速模式
"""

import os
import sys
import gc
import json
import time
import asyncio
import numpy as np
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Any

# ============================================================
# 搜索空间（移除LLM-only参数，因为use_llm=False时无意义）
# ============================================================
SWEEP_SPACE = {
    # --- Priority 1: 对HV影响最大 ---
    "coupling_alpha": {
        "path": ("ALGORITHM_CONFIG", "composite_kernel", "coupling_matrix_alpha"),
        "values": [0.2, 0.5, 0.7, 0.833, 0.95],
        "quick":  [0.5, 0.7, 0.95],
        "desc": "W融合权重 α（data vs LLM）",
    },
    "gamma_init": {
        "path": ("BO_CONFIG", "gamma_init"),
        "values": [0.1, 0.3, 0.5, 1.0, 1.5],
        "quick":  [0.3, 0.5, 1.0],
        "desc": "初始耦合强度 γ₀",
    },

    # --- Priority 2: 中等影响 ---
    "gamma_rate": {
        "path": ("BO_CONFIG", "gamma_update_rate"),
        "values": [0.05, 0.1, 0.2, 0.3, 0.5],
        "quick":  [0.1, 0.2, 0.5],
        "desc": "γ自适应调整率 ρ",
    },
    "eta": {
        "path": ("MOBO_CONFIG", "eta"),
        "values": [0.01, 0.05, 0.1, 0.2],
        "quick":  [0.01, 0.05, 0.1],
        "desc": "Tchebycheff增强系数 η",
    },
    "gp_alpha": {
        "path": ("ALGORITHM_CONFIG", "gp", "alpha"),
        "values": [1e-6, 1e-5, 1e-4, 1e-3],
        "quick":  [1e-5, 1e-4, 1e-3],
        "desc": "GP噪声正则化 α（防PSD崩溃）",
    },
}

# aging_ref 和 time_ref 不应该扫描 —— 它们改变的是HV度量标尺，不是优化行为
# beta_lcb / length_scale_base / sigma_scale —— 只在LLM启用时有意义，单独测试


# ============================================================
# Config补丁工具
# ============================================================
def patch_config(param_name: str, value: Any):
    """运行时修改全局config"""
    import config as cfg
    spec = SWEEP_SPACE[param_name]
    path = spec["path"]
    obj = getattr(cfg, path[0])
    if len(path) == 2:
        obj[path[1]] = value
    elif len(path) == 3:
        obj[path[1]][path[2]] = value
    # 联动同步
    if param_name == "gamma_init":
        cfg.ALGORITHM_CONFIG["composite_kernel"]["gamma_init"] = value


def get_default_value(param_name: str) -> Any:
    """读取当前config中的默认值"""
    import config as cfg
    spec = SWEEP_SPACE[param_name]
    path = spec["path"]
    obj = getattr(cfg, path[0])
    if len(path) == 2:
        return obj[path[1]]
    elif len(path) == 3:
        return obj[path[1]][path[2]]


# ============================================================
# 单次实验（关键优化：禁用一切IO）
# ============================================================
async def run_once(seed: int, n_iter: int = 15) -> Dict:
    """
    运行一次精简实验，只返回HV数字
    
    关键优化：
    - verbose=False 关闭所有打印
    - monkey-patch _save_final_results 跳过图片/JSON/DB保存
    - monkey-patch _save_checkpoint 跳过中间检查点
    - 结束后显式回收内存
    """
    from main_r3 import LLMMOBO
    import config as cfg

    np.random.seed(seed)

    optimizer = LLMMOBO(
        llm_api_key=None,          # 禁用LLM
        n_warmstart=0,
        n_random_init=10,
        n_iterations=n_iter,
        gamma_init=cfg.BO_CONFIG["gamma_init"],
        verbose=False,             # 静默
        use_coupling=True,
        use_warmstart=False,
        use_llm_acq=False,
        use_llm_weighting=False,
        gamma_adaptive=True,
        db_path=":memory:",
    )

    # ★ 关键：替换掉生成图片和保存文件的方法
    optimizer._save_final_results = lambda results: None
    optimizer._save_checkpoint = lambda iteration: None

    t0 = time.time()
    try:
        results = await optimizer.optimize()
        final_hv = results["hv_history"][-1] if results["hv_history"] else 0.0
        n_valid = results["n_valid"]
        n_eval = results["n_evaluations"]
    except Exception as e:
        final_hv = 0.0
        n_valid = 0
        n_eval = 0
        print(f"      [ERROR] seed={seed}: {e}")
    finally:
        # ★ 显式释放内存
        try:
            optimizer.db.close()
        except:
            pass
        del optimizer
        gc.collect()

    return {
        "final_hv": final_hv,
        "n_valid": n_valid,
        "n_eval": n_eval,
        "wall_time": time.time() - t0,
    }


# ============================================================
# 单参数扫描
# ============================================================
async def sweep_param(
    param_name: str,
    seeds: List[int],
    n_iter: int,
    quick: bool,
    output_dir: str,
) -> Dict:
    """扫描单个参数，增量保存结果"""
    spec = SWEEP_SPACE[param_name]
    values = spec["quick"] if quick else spec["values"]
    default = get_default_value(param_name)

    print(f"\n{'='*60}")
    print(f"扫描: {param_name} — {spec['desc']}")
    print(f"  候选值: {values}  (默认={default})")
    print(f"  Seeds: {seeds}, 迭代: {n_iter}")
    print(f"{'='*60}")

    results = {}
    best_hv = -1
    best_val = None

    for val in values:
        patch_config(param_name, val)

        hvs = []
        for seed in seeds:
            r = await run_once(seed, n_iter)
            hvs.append(r["final_hv"])

            # 即时进度反馈（单行）
            status = "✓" if r["final_hv"] > 0 else "✗"
            sys.stdout.write(f"  {param_name}={val}, seed={seed}: "
                             f"HV={r['final_hv']:.4f} valid={r['n_valid']}/{r['n_eval']} "
                             f"({r['wall_time']:.0f}s) {status}\n")
            sys.stdout.flush()

        mean_hv = np.mean(hvs)
        std_hv = np.std(hvs)
        results[str(val)] = {
            "mean_hv": round(float(mean_hv), 6),
            "std_hv": round(float(std_hv), 6),
            "raw": [round(float(h), 6) for h in hvs],
        }

        marker = ""
        if mean_hv > best_hv:
            best_hv = mean_hv
            best_val = val
            marker = " ★"

        print(f"  → {param_name}={val:>10}: HV={mean_hv:.4f} ± {std_hv:.4f}{marker}")

    # 恢复默认值
    patch_config(param_name, default)

    summary = {
        "param": param_name,
        "desc": spec["desc"],
        "default": default,
        "tested": values,
        "results": results,
        "best_value": best_val,
        "best_hv": round(float(best_hv), 6),
        "default_hv": results.get(str(default), {}).get("mean_hv", "N/A"),
    }

    print(f"\n  最优: {param_name}={best_val} (HV={best_hv:.4f})")
    if str(default) in results:
        print(f"  默认: {param_name}={default} (HV={results[str(default)]['mean_hv']:.4f})")

    # ★ 增量保存（每扫完一个参数就存一次，崩溃不丢数据）
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"sweep_{param_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"  已保存: {path}")

    return summary


# ============================================================
# 汇总
# ============================================================
def print_final_summary(all_summaries: Dict, output_dir: str):
    """打印最终推荐 + 保存汇总"""

    print(f"\n{'='*60}")
    print("超参数扫描汇总")
    print(f"{'='*60}")
    print(f"{'参数':<20} {'默认值':>10} {'推荐值':>10} {'默认HV':>10} {'最优HV':>10} {'提升':>8}")
    print("-" * 70)

    recommendations = {}
    for name, s in all_summaries.items():
        default_hv = s.get("default_hv", 0)
        if isinstance(default_hv, str):
            default_hv = 0
        best_hv = s["best_hv"]
        delta = best_hv - default_hv if default_hv else 0

        changed = str(s["best_value"]) != str(s["default"])
        marker = "★" if changed else " "

        print(f"{marker} {name:<18} {str(s['default']):>10} {str(s['best_value']):>10} "
              f"{default_hv:>10.4f} {best_hv:>10.4f} {delta:>+8.4f}")

        recommendations[name] = {
            "default": s["default"],
            "recommended": s["best_value"],
            "hv_improvement": round(delta, 6),
            "path": SWEEP_SPACE[name]["path"],
        }

    # 保存汇总
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations,
            "details": {k: v for k, v in all_summaries.items()},
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n汇总已保存: {path}")

    # 打印config.py修改建议
    changes = {k: v for k, v in recommendations.items() if str(v["default"]) != str(v["recommended"])}
    if changes:
        print(f"\n{'='*60}")
        print("建议修改 config.py:")
        print(f"{'='*60}")
        for name, info in changes.items():
            path_str = " → ".join(info["path"])
            print(f"  {path_str}")
            print(f"    {info['default']} → {info['recommended']}  (HV +{info['hv_improvement']:.4f})")
    else:
        print("\n当前默认值已是最优，无需修改。")


# ============================================================
# 主入口
# ============================================================
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="PE-GenBO 超参数扫描（快速版）")
    parser.add_argument("--param", type=str, default=None, help="只扫单个参数")
    parser.add_argument("--quick", action="store_true", help="快速模式（更少候选值）")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=15, help="BO迭代次数")
    parser.add_argument("--output-dir", type=str, default="./sweep_results")
    args = parser.parse_args()

    seeds = args.seeds or ([42, 123] if args.quick else [42, 123, 456])

    # 确定要扫描的参数
    if args.param:
        if args.param not in SWEEP_SPACE:
            print(f"未知参数: {args.param}")
            print(f"可选: {list(SWEEP_SPACE.keys())}")
            return
        params_to_scan = [args.param]
    else:
        params_to_scan = list(SWEEP_SPACE.keys())

    total_runs = sum(
        len(SWEEP_SPACE[p]["quick"] if args.quick else SWEEP_SPACE[p]["values"])
        for p in params_to_scan
    ) * len(seeds)
    print(f"\n将扫描 {len(params_to_scan)} 个参数，共 {total_runs} 次实验")
    print(f"预计耗时: ~{total_runs * 30 / 60:.0f} 分钟（按每次30秒估算）")

    all_summaries = {}
    for param_name in params_to_scan:
        summary = await sweep_param(
            param_name, seeds, args.n_iter, args.quick, args.output_dir
        )
        all_summaries[param_name] = summary

    print_final_summary(all_summaries, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())