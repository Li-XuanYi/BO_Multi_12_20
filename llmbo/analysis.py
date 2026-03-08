"""
LLMBO-MO 结果分析工具

合并了原来的三个分析脚本：
  - analyze_database.py  → --mode db
  - analyze_pareto.py    → --mode pareto
  - verify_results.py    → --mode verify

用法:
  python analysis.py --mode pareto [--results results/results.json]
  python analysis.py --mode db     [--results results/results.json]
  python analysis.py --mode verify [--results results/results.json]
  python analysis.py               (默认: 全部三种模式)
"""

import argparse
import json
import numpy as np
import os


# ============================================================
# 通用工具
# ============================================================

def load_results(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"结果文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pareto_front(valid_points: list) -> list:
    """从有效点列表中提取 Pareto 前沿（最小化三目标）"""
    objectives = np.array([p["objectives"] for p in valid_points])
    n = len(objectives)
    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and pareto_mask[i]:
                # j 支配 i: j 所有维度 <= i 且至少一维严格 <
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    pareto_mask[i] = False
                    break
    return [valid_points[i] for i in range(n) if pareto_mask[i]]


def compute_hv(pareto_objectives: np.ndarray, ref: np.ndarray, ideal: np.ndarray) -> float:
    """计算归一化超体积"""
    try:
        from pymoo.indicators.hv import HV
        norm = (pareto_objectives - ideal) / (ref - ideal + 1e-12)
        # 过滤出在 [0,1) 范围内的点
        mask = np.all(norm < 1.0, axis=1) & np.all(norm >= 0.0, axis=1)
        pf_norm = norm[mask]
        if len(pf_norm) == 0:
            return 0.0
        hv_indicator = HV(ref_point=np.ones(3))
        return float(hv_indicator(pf_norm))
    except ImportError:
        return -1.0  # pymoo 未安装


# ============================================================
# Mode 1: db — 数据库统计
# ============================================================

def mode_db(data: dict):
    print("=" * 70)
    print("数据库统计分析")
    print("=" * 70)

    entries = data.get("database", [])
    total = len(entries)
    valid_entries = [e for e in entries if e.get("valid", False)]
    n_valid = len(valid_entries)
    valid_rate = 100.0 * n_valid / max(total, 1)

    print(f"\n总评估次数  : {total}")
    print(f"有效样本数  : {n_valid}")
    print(f"有效率      : {valid_rate:.1f}%")

    if n_valid == 0:
        print("无有效样本，退出分析。")
        return

    # 目标值统计
    times = [e["time"] for e in valid_entries]
    temps = [e["temp"] for e in valid_entries]
    agings = [e["aging"] for e in valid_entries]

    print(f"\n{'目标':<15} {'最小':>10} {'最大':>10} {'均值':>10}")
    print("-" * 50)
    print(f"{'time [s]':<15} {min(times):>10.1f} {max(times):>10.1f} {np.mean(times):>10.1f}")
    print(f"{'temp [K]':<15} {min(temps):>10.2f} {max(temps):>10.2f} {np.mean(temps):>10.2f}")
    print(f"{'aging':<15} {min(agings):>10.6f} {max(agings):>10.6f} {np.mean(agings):>10.6f}")

    # Pareto 贡献分析
    valid_pts = [
        {"index": i, "objectives": [e["time"], e["temp"], e["aging"]], "params": e.get("params", {})}
        for i, e in enumerate(entries) if e.get("valid", False)
    ]
    pareto_pts = extract_pareto_front(valid_pts)
    print(f"\nPareto 前沿点数 : {len(pareto_pts)}")

    initial_in_pareto = sum(1 for pt in pareto_pts if pt["index"] < 15)
    print(f"初始化（前15）贡献 : {initial_in_pareto}/{len(pareto_pts)}")


# ============================================================
# Mode 2: pareto — Pareto 前沿详细报告
# ============================================================

def mode_pareto(data: dict):
    print("=" * 70)
    print("Pareto 前沿分析")
    print("=" * 70)

    ref = np.array([7200.0, 323.15, 0.008])
    ideal = np.array([1200.0, 298.15, 1e-6])

    pf = data.get("pareto_front", [])
    if not pf:
        # 从 database 重新计算
        entries = data.get("database", [])
        valid_pts = [
            {"index": i, "objectives": [e["time"], e["temp"], e["aging"]], "params": e.get("params", {})}
            for i, e in enumerate(entries) if e.get("valid", False)
        ]
        pareto_pts = extract_pareto_front(valid_pts)
        pf = [pt["objectives"] for pt in pareto_pts]

    print(f"\nPareto 前沿（{len(pf)} 个点）:")
    print(f"\n{'#':<4} {'time[s]':>10} {'temp[K]':>10} {'aging':>12}  {'norm_t':>7} {'norm_T':>7} {'norm_a':>7}")
    print("-" * 70)

    for i, p in enumerate(pf):
        p_arr = np.array(p)
        norm = (p_arr - ideal) / (ref - ideal + 1e-12)
        print(f"{i+1:<4} {p[0]:>10.1f} {p[1]:>10.2f} {p[2]:>12.6f}  "
              f"{norm[0]:>7.3f} {norm[1]:>7.3f} {norm[2]:>7.3f}")

    # HV 计算与对比
    pf_arr = np.array(pf) if pf else np.empty((0, 3))
    hv_calc = compute_hv(pf_arr, ref, ideal)
    hv_recorded = data.get("hv_history", [0])
    hv_latest = hv_recorded[-1] if hv_recorded else 0

    print(f"\n{'=' * 70}")
    print(f"当前计算 HV : {hv_calc:.4f}")
    print(f"记录最终 HV : {hv_latest:.4f}")
    if hv_calc > 0:
        consistent = abs(hv_calc - hv_latest) < 0.002
        print(f"一致性检验   : {'✓' if consistent else '✗'}")


# ============================================================
# Mode 3: verify — 结果有效性验证
# ============================================================

def mode_verify(data: dict):
    print("=" * 70)
    print("实验结果验证")
    print("=" * 70)

    checks = []

    # 1. 有效率
    n_eval = data.get("n_evaluations", len(data.get("database", [])))
    n_valid = data.get("n_valid", sum(1 for e in data.get("database", []) if e.get("valid", False)))
    valid_rate = 100.0 * n_valid / max(n_eval, 1)
    ok1 = valid_rate > 60
    checks.append(ok1)
    print(f"\n1. 有效率: {valid_rate:.1f}% ({'✓' if ok1 else '✗ <60%'})")

    # 2. HV 历史单调性（不应有超长连续相同段）
    hv_history = data.get("hv_history", [])
    max_consec = 1
    cur_consec = 1
    for i in range(1, len(hv_history)):
        if abs(hv_history[i] - hv_history[i - 1]) < 1e-8:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 1
    ok2 = max_consec < 10
    checks.append(ok2)
    print(f"2. HV 最长连续相同: {max_consec} ({'✓' if ok2 else '✗ 可能存在 bug'})")

    # 3. 参数多样性（前15条记录无重复 I1）
    entries = data.get("database", [])[:15]
    try:
        i1_vals = [round(e["params"].get("I1", e["params"].get("current1", 0)), 2) for e in entries]
        unique_i1 = len(set(i1_vals))
        ok3 = unique_i1 == len(i1_vals)
    except Exception:
        ok3 = True  # 无法验证时跳过
        unique_i1 = -1
    checks.append(ok3)
    print(f"3. I1 唯一性（前15）: {unique_i1} 个 ({'✓' if ok3 else '✗ 重复采样'})")

    # 汇总
    print(f"\n{'=' * 70}")
    if all(checks):
        print("✓ 全部验证通过，实验结果有效！")
    else:
        print("✗ 部分检查未通过：")
        if not checks[0]:
            print("  → 建议适当提高 temp_max (如 323.15 K)")
        if not checks[1]:
            print("  → 建议检查 HV 计算或随机状态保护")
        if not checks[2]:
            print("  → 建议检查随机数生成器状态管理")
    print("=" * 70)


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLMBO-MO 结果分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analysis.py                                  # 全部模式
  python analysis.py --mode pareto                   # 仅 Pareto 分析
  python analysis.py --mode db --results r/r.json    # 数据库统计
  python analysis.py --mode verify                   # 有效性验证
        """
    )
    parser.add_argument("--mode", choices=["pareto", "db", "verify", "all"],
                        default="all", help="分析模式")
    parser.add_argument("--results", default="results/results.json",
                        help="结果文件路径 (默认: results/results.json)")
    args = parser.parse_args()

    data = load_results(args.results)

    if args.mode in ("db", "all"):
        mode_db(data)
        print()
    if args.mode in ("pareto", "all"):
        mode_pareto(data)
        print()
    if args.mode in ("verify", "all"):
        mode_verify(data)


if __name__ == "__main__":
    main()
