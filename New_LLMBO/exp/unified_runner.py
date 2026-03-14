from __future__ import annotations
"""
unified_runner.py — LLAMBO-MO + ParEGO 统一运行入口
====================================================
统一运行 EIMO（LLAMBO-MO）和 ParEGO 两个优化器，支持多随机种子。
运行完成后自动调用 export_to_xlsx.py 生成 XLSX（供 plot_hv.py / plot_pareto3d.py 使用）。

用法：
  # 快速演示（5 回合，Mock LLM，seed=42）
  python unified_runner.py --demo

  # 标准运行（3 seed，EIMO 50 回合 + ParEGO 300 回合）
  python unified_runner.py --seeds 0 1 2 --eimo-iterations 50 --parego-iterations 300

  # 仅运行 EIMO
  python unified_runner.py --method eimo --seeds 0 1 2 --eimo-iterations 50

  # 仅运行 ParEGO
  python unified_runner.py --method parego --seeds 0 1 2 --parego-iterations 300

  # 并行运行多 seed（使用多进程）
  python unified_runner.py --seeds 0 1 2 --parallel --max-workers 3

  # 只导出已有结果为 XLSX（跳过运行）
  python unified_runner.py --export-only --output my_results/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from config.settings import Settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# §A  默认配置
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_EIMO_CONFIG: Dict[str, Any] = {
    # ── 实验规模 ──────────────────────────────────────────────────────────
    "max_iterations":   Settings.BO.N_ITERATIONS,
    "n_warmstart":      Settings.BO.N_WARMSTART,
    "n_candidates":     Settings.ACQUISITION.N_CANDIDATES,
    "n_select":         Settings.ACQUISITION.N_SELECT,

    # ── LLM 配置 ──────────────────────────────────────────────────────────
    "llm_backend":      Settings.LLM.BACKEND,
    "llm_model":        Settings.LLM.MODEL,
    "llm_api_base":     Settings.LLM.API_BASE,
    "llm_api_key":      Settings.LLM.API_KEY,
    "llm_n_samples":    Settings.LLM.N_SAMPLES,
    "llm_temperature":  Settings.LLM.TEMPERATURE,

    # ── GP/Coupling 超参数 ───────────────────────────────────────────────
    "gamma_max":        Settings.COUPLING.GAMMA_MAX,
    "gamma_min":        Settings.COUPLING.GAMMA_MIN,
    "gamma_t_decay":    Settings.COUPLING.GAMMA_T_DECAY,
    "alpha_max":        Settings.COUPLING.ALPHA_MAX,
    "alpha_min":        Settings.COUPLING.ALPHA_MIN,
    "t_decay_alpha":    Settings.COUPLING.T_DECAY_ALPHA,

    # ── Acquisition 超参数 ───────────────────────────────────────────────
    "kappa":            Settings.ACQUISITION.KAPPA,
    "eps_sigma":        Settings.ACQUISITION.EPS_SIGMA,
    "rho":              Settings.ACQUISITION.RHO,

    # ── Riesz s-energy 权重集合 ──────────────────────────────────────────
    "riesz_n_div":      Settings.RIESZ.N_DIV,
    "riesz_s":          Settings.RIESZ.S,
    "riesz_n_iter":     Settings.RIESZ.N_ITER,
    "riesz_lr":         Settings.RIESZ.LR,
    "riesz_seed":       Settings.RIESZ.SEED,

    # ── Tchebycheff 参数 ─────────────────────────────────────────────────
    "eta":              Settings.MOBO.ETA,

    # ── 检查点 ───────────────────────────────────────────────────────────
    "checkpoint_every": Settings.OUTPUT.CHECKPOINT_EVERY,

    # ── 电池模型 ─────────────────────────────────────────────────────────
    "battery_model":    Settings.PYBAMM.BATTERY_MODEL,
}

DEFAULT_PAREGO_CONFIG: Dict[str, Any] = {
    # ── 实验规模 ──────────────────────────────────────────────────────────
    "max_iterations":   Settings.PAREGO.N_ITERATIONS,
    "n_warmstart":      Settings.PAREGO.N_WARMSTART,
    "n_random_cands":   Settings.PAREGO.N_RANDOM_CANDS,

    # ── Riesz s-energy 权重集合 ──────────────────────────────────────────
    "riesz_n_div":      Settings.RIESZ.N_DIV,
    "riesz_s":          Settings.RIESZ.S,
    "riesz_n_iter":     Settings.RIESZ.N_ITER,
    "riesz_lr":         Settings.RIESZ.LR,
    "riesz_seed":       Settings.RIESZ.SEED,

    # ── Tchebycheff 参数 ─────────────────────────────────────────────────
    "eta":              Settings.MOBO.ETA,

    # ── GP 超参数 ────────────────────────────────────────────────────────
    "gp_n_restarts":    Settings.PAREGO.GP_N_RESTARTS,
    "gp_normalize_y":   True,

    # ── EI 参数 ──────────────────────────────────────────────────────────
    "xi":               Settings.ACQUISITION.XI,

    # ── 检查点 ───────────────────────────────────────────────────────────
    "checkpoint_every": Settings.OUTPUT.CHECKPOINT_EVERY,

    # ── 电池模型 ─────────────────────────────────────────────────────────
    "battery_model":    Settings.PYBAMM.BATTERY_MODEL,
}


# ═══════════════════════════════════════════════════════════════════════════
# §B  单次运行函数
# ═══════════════════════════════════════════════════════════════════════════

def _run_eimo_seed(
    seed:       int,
    config:     Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """运行单次 EIMO（LLAMBO-MO）实验，保存数据库并返回结果摘要。"""
    from llmbo.optimizer import BayesOptimizer

    out = Path(output_dir) / "eimo" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        **config,
        "w_sample_seed":  seed,
        "checkpoint_dir": str(out / "checkpoints"),
    }

    t0 = time.time()
    optimizer = BayesOptimizer(config=cfg)
    db = optimizer.run()

    db_path = str(out / "db_final.json")
    db.save(db_path)
    optimizer.save_results(str(out))

    elapsed = time.time() - t0
    hv      = db.compute_hypervolume()

    logger.info(
        "[EIMO seed=%d] 完成: HV=%.6f  |PF|=%d  n=%d  (%.1f min)",
        seed, hv, db.pareto_size, db.size, elapsed / 60,
    )
    return {
        "seed":        seed,
        "hv":          hv,
        "pareto_size": db.pareto_size,
        "n_total":     db.size,
        "db_path":     db_path,
        "elapsed":     elapsed,
    }


def _run_parego_seed(
    seed:       int,
    config:     Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """运行单次 ParEGO 实验，保存汇总 JSON 并返回结果摘要。"""
    from llmbo.ParEGO import ParEGOOptimizer

    out = Path(output_dir) / "parego" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        **config,
        "lhs_seed":       seed,
        "w_sample_seed":  seed,
        "cand_seed":      seed,
        "checkpoint_dir": str(out / "checkpoints"),
    }

    t0 = time.time()
    optimizer = ParEGOOptimizer(config=cfg)
    db = optimizer.run()
    optimizer.save_results(str(out))
    optimizer.save_final_summary(str(out))

    summary_path = str(out / "parego_final_summary.json")
    elapsed = time.time() - t0
    hv      = db.compute_hypervolume()

    logger.info(
        "[ParEGO seed=%d] 完成: HV=%.6f  |PF|=%d  n=%d  (%.1f min)",
        seed, hv, db.pareto_size, db.size, elapsed / 60,
    )
    return {
        "seed":         seed,
        "hv":           hv,
        "pareto_size":  db.pareto_size,
        "n_total":      db.size,
        "summary_path": summary_path,
        "elapsed":      elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# §C  多 seed 调度
# ═══════════════════════════════════════════════════════════════════════════

def run_seeds(
    seeds:         List[int],
    method:        str,
    output_dir:    str,
    eimo_config:   Dict[str, Any],
    parego_config: Dict[str, Any],
    parallel:      bool = False,
    max_workers:   int  = 2,
) -> Dict[str, List[Dict]]:
    """
    对所有 seed 运行指定方法（顺序或多进程并行）。

    Returns
    -------
    {"eimo_results": [...], "parego_results": [...]}
    """
    run_eimo   = method in ("eimo",   "both")
    run_parego = method in ("parego", "both")

    eimo_results:   List[Dict] = []
    parego_results: List[Dict] = []

    if parallel and len(seeds) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        logger.info("[调度] 并行模式：max_workers=%d", max_workers)
        futures: Dict[Any, Tuple[str, int]] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for seed in seeds:
                if run_eimo:
                    f = executor.submit(_run_eimo_seed, seed, eimo_config, output_dir)
                    futures[f] = ("eimo", seed)
                if run_parego:
                    f = executor.submit(_run_parego_seed, seed, parego_config, output_dir)
                    futures[f] = ("parego", seed)

            for future in as_completed(futures):
                method_name, seed = futures[future]
                try:
                    result = future.result()
                    (eimo_results if method_name == "eimo" else parego_results).append(result)
                    logger.info("[完成] %s seed=%d  HV=%.6f", method_name, seed, result["hv"])
                except Exception as exc:
                    logger.error("[失败] %s seed=%d: %s", method_name, seed, exc)
    else:
        logger.info("[调度] 顺序模式：%d seed(s) × method=%s", len(seeds), method)
        for seed in seeds:
            if run_eimo:
                try:
                    eimo_results.append(_run_eimo_seed(seed, eimo_config, output_dir))
                except Exception:
                    import traceback
                    logger.error("[失败] EIMO seed=%d", seed)
                    traceback.print_exc()

            if run_parego:
                try:
                    parego_results.append(_run_parego_seed(seed, parego_config, output_dir))
                except Exception:
                    import traceback
                    logger.error("[失败] ParEGO seed=%d", seed)
                    traceback.print_exc()

    eimo_results.sort(key=lambda r: r["seed"])
    parego_results.sort(key=lambda r: r["seed"])
    return {"eimo_results": eimo_results, "parego_results": parego_results}


# ═══════════════════════════════════════════════════════════════════════════
# §D  XLSX 导出
# ═══════════════════════════════════════════════════════════════════════════

def export_xlsx_from_results(
    eimo_results:   List[Dict],
    parego_results: List[Dict],
    output_dir:     str,
) -> Tuple[Optional[str], Optional[str]]:
    """从运行结果收集各 seed 路径，调用 export_to_xlsx 生成 XLSX。"""
    try:
        from DataBase.export_to_xlsx import (
            load_eimo_stats, load_parego_stats,
            export_hv_curves, export_pareto_front,
        )
    except ImportError as exc:
        logger.error("无法导入 export_to_xlsx: %s", exc)
        return None, None

    out = Path(output_dir)

    eimo_stats_list:  List = []
    eimo_pareto_list: List = []
    for r in eimo_results:
        db_path = r.get("db_path", "")
        if Path(db_path).exists():
            stats, pareto = load_eimo_stats(db_path)
            eimo_stats_list.append(stats)
            eimo_pareto_list.append(pareto)
        else:
            logger.warning("[导出] EIMO db 文件不存在，已跳过 seed=%d: %s", r["seed"], db_path)

    parego_stats_list: List = []
    for r in parego_results:
        summary_path = r.get("summary_path", "")
        if Path(summary_path).exists():
            stats, _ = load_parego_stats(summary_path)
            parego_stats_list.append(stats)
        else:
            logger.warning("[导出] ParEGO summary 不存在，已跳过 seed=%d: %s", r["seed"], summary_path)

    hv_path     = None
    pareto_path = None

    if eimo_stats_list or parego_stats_list:
        hv_path = export_hv_curves(
            eimo_stats_list   or [[]],
            parego_stats_list or [[]],
            str(out / "hv_curves.xlsx"),
        )

    if eimo_pareto_list:
        pareto_path = export_pareto_front(
            eimo_pareto_list,
            str(out / "pareto_front.xlsx"),
            use_last_seed_only=False,
        )

    return hv_path, pareto_path


def _collect_existing_results(output_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    扫描 output_dir 下已存在的结果目录，收集各 seed 路径，
    用于 --export-only 模式。
    """
    out = Path(output_dir)
    eimo_results:   List[Dict] = []
    parego_results: List[Dict] = []

    for seed_dir in sorted((out / "eimo").glob("seed_*")) if (out / "eimo").exists() else []:
        db_path = seed_dir / "db_final.json"
        if db_path.exists():
            seed = int(seed_dir.name.replace("seed_", ""))
            eimo_results.append({"seed": seed, "db_path": str(db_path),
                                  "hv": 0.0, "pareto_size": 0, "n_total": 0, "elapsed": 0.0})

    for seed_dir in sorted((out / "parego").glob("seed_*")) if (out / "parego").exists() else []:
        summary_path = seed_dir / "parego_final_summary.json"
        if summary_path.exists():
            seed = int(seed_dir.name.replace("seed_", ""))
            parego_results.append({"seed": seed, "summary_path": str(summary_path),
                                   "hv": 0.0, "pareto_size": 0, "n_total": 0, "elapsed": 0.0})

    return eimo_results, parego_results


# ═══════════════════════════════════════════════════════════════════════════
# §E  汇总统计
# ═══════════════════════════════════════════════════════════════════════════

def _stat_str(results: List[Dict], key: str) -> str:
    vals = [r[key] for r in results if key in r and r[key] != 0.0]
    if not vals:
        return "N/A"
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
        return f"{mean:.6f} ± {std:.6f}"
    return f"{mean:.6f}"


def print_summary(eimo_results: List[Dict], parego_results: List[Dict]) -> None:
    print("\n" + "=" * 70)
    print("  运行汇总")
    print("=" * 70)
    for tag, results in [("EIMO", eimo_results), ("ParEGO", parego_results)]:
        if not results:
            continue
        print(f"\n  [{tag}]  {len(results)} 次运行")
        print(f"    最终 HV:     {_stat_str(results, 'hv')}")
        print(f"    Pareto 大小: {_stat_str(results, 'pareto_size')}")
        print(f"    总评估数:    {_stat_str(results, 'n_total')}")
        for r in results:
            elapsed_str = f"{r['elapsed'] / 60:.1f} min" if r["elapsed"] else "-"
            print(f"    seed={r['seed']}: HV={r['hv']:.6f}  "
                  f"|PF|={r['pareto_size']}  n={r['n_total']}  ({elapsed_str})")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# §F  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLAMBO-MO + ParEGO 统一运行入口（多 seed + 自动 XLSX 导出）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速演示（5 回合，Mock LLM）
  python unified_runner.py --demo

  # 运行两方法各 3 个 seed
  python unified_runner.py --seeds 0 1 2 --eimo-iterations 50 --parego-iterations 300

  # 仅运行 ParEGO
  python unified_runner.py --method parego --seeds 0 1 2 --parego-iterations 300

  # 只导出已有结果为 XLSX
  python unified_runner.py --export-only --output unified_results/
        """,
    )
    parser.add_argument(
        "--method", choices=["eimo", "parego", "both"], default="both",
        help="运行的算法（默认：both）",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="随机种子列表（默认：42）",
    )
    parser.add_argument("--eimo-iterations",   type=int, default=50,
                        help="EIMO 迭代次数（默认：50）")
    parser.add_argument("--parego-iterations", type=int, default=300,
                        help="ParEGO 迭代次数（默认：300）")
    parser.add_argument("--eimo-warmstart",    type=int, default=10,
                        help="EIMO warmstart 点数（默认：10）")
    parser.add_argument("--parego-warmstart",  type=int, default=15,
                        help="ParEGO warmstart 点数（默认：15）")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM 模型名称（默认：从 config/settings.py 读取）")
    parser.add_argument(
        "--mock-llm", action="store_true",
        help="使用 Mock LLM（不调用真实 API，用于测试）",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="多进程并行运行各 seed",
    )
    parser.add_argument("--max-workers", type=int, default=2,
                        help="并行进程数（默认：2）")
    parser.add_argument("--output", "-o", type=str, default="unified_results",
                        help="结果输出目录（默认：unified_results）")
    parser.add_argument(
        "--demo", action="store_true",
        help="演示模式：5 回合 × 1 seed，Mock LLM",
    )
    parser.add_argument(
        "--export-only", action="store_true",
        help="跳过运行，直接将已有结果导出为 XLSX",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="跳过 XLSX 导出步骤",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    parser = _create_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Demo 模式覆盖参数
    if args.demo:
        logger.info("[演示] Demo 模式：5 回合，Mock LLM，seed=42")
        args.seeds             = [42]
        args.eimo_iterations   = 5
        args.parego_iterations = 5
        args.eimo_warmstart    = 3
        args.parego_warmstart  = 3
        args.mock_llm          = True
        args.method            = "both"

    print("\n" + "=" * 70)
    print("  LLAMBO-MO + ParEGO 统一运行器")
    print("=" * 70)

    # ── 仅导出模式 ────────────────────────────────────────────────────────
    if args.export_only:
        print(f"  [模式] 仅导出：扫描 {args.output}/")
        eimo_results, parego_results = _collect_existing_results(args.output)
        print(f"  找到 {len(eimo_results)} 个 EIMO 结果, {len(parego_results)} 个 ParEGO 结果")
        if not eimo_results and not parego_results:
            print("[错误] 未找到任何结果，请先运行优化器", file=sys.stderr)
            sys.exit(1)
        hv_path, pareto_path = export_xlsx_from_results(eimo_results, parego_results, args.output)
        if hv_path:
            print(f"\n  HV 曲线:     {hv_path}")
            print(f"  绘图:        python plot_hv.py {hv_path} hv_output.png")
            print(f"               python plot_optimal_count.py {hv_path} count_output.png")
        if pareto_path:
            print(f"  Pareto 前沿: {pareto_path}")
            print(f"  绘图:        python plot_pareto3d.py {pareto_path} pareto_output.png")
        return

    # ── 构建优化器配置 ────────────────────────────────────────────────────
    eimo_config = {
        **DEFAULT_EIMO_CONFIG,
        "max_iterations": args.eimo_iterations,
        "n_warmstart":    args.eimo_warmstart,
        "llm_model":      args.llm_model,
        "llm_backend":    "mock" if args.mock_llm else "openai",
    }
    parego_config = {
        **DEFAULT_PAREGO_CONFIG,
        "max_iterations": args.parego_iterations,
        "n_warmstart":    args.parego_warmstart,
    }

    print(f"  方法:        {args.method}")
    print(f"  随机种子:    {args.seeds}")
    if args.method in ("eimo", "both"):
        print(f"  EIMO 迭代:   {args.eimo_iterations}  warmstart={args.eimo_warmstart}")
        print(f"  LLM 后端:    {'mock' if args.mock_llm else 'openai'} / {args.llm_model}")
    if args.method in ("parego", "both"):
        print(f"  ParEGO 迭代: {args.parego_iterations}  warmstart={args.parego_warmstart}")
    parallel_info = f"是 (workers={args.max_workers})" if args.parallel else "否"
    print(f"  并行:        {parallel_info}")
    print(f"  输出目录:    {args.output}")
    print("=" * 70)

    # ── 运行 ──────────────────────────────────────────────────────────────
    t0 = time.time()
    results = run_seeds(
        seeds=args.seeds,
        method=args.method,
        output_dir=args.output,
        eimo_config=eimo_config,
        parego_config=parego_config,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )
    logger.info("总运行时间: %.1f 分钟", (time.time() - t0) / 60)

    eimo_results   = results["eimo_results"]
    parego_results = results["parego_results"]

    print_summary(eimo_results, parego_results)

    # ── 导出 XLSX ─────────────────────────────────────────────────────────
    if not args.no_export:
        print("\n[导出] 生成 XLSX 文件...")
        hv_path, pareto_path = export_xlsx_from_results(
            eimo_results, parego_results, args.output
        )
        if hv_path or pareto_path:
            print()
        if hv_path:
            print(f"  HV 曲线:     {hv_path}")
            print(f"  绘图命令:    python plot_hv.py {hv_path} hv_output.png")
            print(f"               python plot_optimal_count.py {hv_path} count_output.png")
        if pareto_path:
            print(f"  Pareto 前沿: {pareto_path}")
            print(f"  绘图命令:    python plot_pareto3d.py {pareto_path} pareto_output.png")


if __name__ == "__main__":
    main()