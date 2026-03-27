"""
实验结果导出工具 - 将 LLMBO-MO 和 NSGA-II 的对比结果导出为 xlsx 文件

支持多种数据源格式：
    1. 从 run_comparison.py 的结果直接导出
    2. 从 JSON 文件导入后导出
    3. 从原始数据库文件导入

Usage:
    # 从已有的对比结果导出
    python export_results.py --input comparison_results.xlsx --output formatted_results.xlsx

    # 从 JSON 导入
    python export_results.py --json llmbo_results.json --json-nsgaii nsgaii_results.json --output comparison.xlsx
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_json_results(path: str) -> dict:
    """从 JSON 文件加载实验结果"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_comparison_to_xlsx(
    llmbo_data: dict,
    nsgaii_data: dict,
    output_path: str,
    metadata: dict | None = None,
):
    """
    导出对比结果到 xlsx 文件

    Args:
        llmbo_data: LLMBO 结果数据
        nsgaii_data: NSGA-II 结果数据
        output_path: 输出文件路径
        metadata: 元数据（配置信息等）
    """
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas 未安装：pip install pandas")
        return

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: HV 曲线
        hv_data = []

        # LLMBO HV 曲线
        if 'hv_history' in llmbo_data:
            for run_id, (hvs, n_evals) in enumerate(zip(
                llmbo_data.get('hv_history', []),
                llmbo_data.get('n_evals', [])
            )):
                if isinstance(hvs, list):
                    for i, hv in enumerate(hvs):
                        eval_count = n_evals[i] if i < len(n_evals) else i
                        hv_data.append({
                            'Run_ID': run_id + 1,
                            'Algorithm': 'LLMBO',
                            'Iteration': i,
                            'N_Evaluations': eval_count,
                            'HV': hv,
                        })

        # NSGA-II HV 曲线
        if 'hv_history' in nsgaii_data:
            for run_id, (hvs, n_evals) in enumerate(zip(
                nsgaii_data.get('hv_history', []),
                nsgaii_data.get('n_evals', [])
            )):
                if isinstance(hvs, list):
                    for i, hv in enumerate(hvs):
                        eval_count = n_evals[i] if i < len(n_evals) else (i + 1) * 50
                        hv_data.append({
                            'Run_ID': run_id + 1,
                            'Algorithm': 'NSGAII',
                            'Iteration': i,
                            'N_Evaluations': eval_count,
                            'HV': hv,
                        })

        if hv_data:
            pd.DataFrame(hv_data).to_excel(writer, sheet_name='HV_Curves', index=False)

        # Sheet 2: LLMBO Pareto 前沿
        llmbo_pf = _extract_pareto_data(llmbo_data, 'LLMBO')
        if llmbo_pf:
            pd.DataFrame(llmbo_pf).to_excel(writer, sheet_name='Pareto_LLMBO', index=False)

        # Sheet 3: NSGA-II Pareto 前沿
        nsgaii_pf = _extract_pareto_data(nsgaii_data, 'NSGAII')
        if nsgaii_pf:
            pd.DataFrame(nsgaii_pf).to_excel(writer, sheet_name='Pareto_NSGAII', index=False)

        # Sheet 4: 收敛统计
        conv_data = []
        _add_convergence_stats(conv_data, llmbo_data, 'LLMBO')
        _add_convergence_stats(conv_data, nsgaii_data, 'NSGAII')
        if conv_data:
            pd.DataFrame(conv_data).to_excel(writer, sheet_name='Convergence', index=False)

        # Sheet 5: 统计汇总
        stats = _compute_statistics(llmbo_data, nsgaii_data)
        if stats:
            pd.DataFrame(stats).to_excel(writer, sheet_name='Statistics', index=False)

        # Sheet 6: 配置信息
        if metadata:
            config_df = pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value'])
            config_df.to_excel(writer, sheet_name='Config', index=False)
        else:
            config_data = {
                'timestamp': datetime.now().isoformat(),
                'llmbo_source': 'user_provided',
                'nsgaii_source': 'user_provided',
            }
            config_df = pd.DataFrame(list(config_data.items()), columns=['Parameter', 'Value'])
            config_df.to_excel(writer, sheet_name='Config', index=False)

    log.info(f"结果已导出到：{output_path}")


def _extract_pareto_data(data: dict, algorithm: str) -> list[dict]:
    """提取 Pareto 前沿数据"""
    pf_data = []

    pareto_thetas = data.get('pareto_thetas', [])
    pareto_objectives = data.get('pareto_objectives', [])

    if isinstance(pareto_thetas, dict):
        # 多次运行的格式
        for run_id, (thetas, objs) in enumerate(zip(
            pareto_thetas.values() if hasattr(pareto_thetas, 'values') else [pareto_thetas],
            pareto_objectives.values() if hasattr(pareto_objectives, 'values') else [pareto_objectives]
        )):
            for theta, obj in zip(
                thetas if isinstance(thetas, list) else [thetas],
                objs if isinstance(objs, list) else [objs]
            ):
                if len(theta) >= 7 and len(obj) >= 3:
                    pf_data.append({
                        'Run_ID': run_id + 1,
                        'Algorithm': algorithm,
                        'I1': theta[0], 'I2': theta[1], 'I3': theta[2],
                        'SOC_sw1': theta[3], 'SOC_sw2': theta[4],
                        'V_CV': theta[5], 'I_cutoff': theta[6],
                        't_charge': obj[0], 'T_peak': obj[1], 'delta_Q': obj[2],
                    })
    elif pareto_thetas and isinstance(pareto_thetas[0], list):
        # 单次运行的格式
        for theta, obj in zip(pareto_thetas, pareto_objectives):
            if len(theta) >= 7 and len(obj) >= 3:
                pf_data.append({
                    'Run_ID': 1,
                    'Algorithm': algorithm,
                    'I1': theta[0], 'I2': theta[1], 'I3': theta[2],
                    'SOC_sw1': theta[3], 'SOC_sw2': theta[4],
                    'V_CV': theta[5], 'I_cutoff': theta[6],
                    't_charge': obj[0], 'T_peak': obj[1], 'delta_Q': obj[2],
                })

    return pf_data


def _add_convergence_stats(stats: list, data: dict, algorithm: str):
    """添加收敛统计数据"""
    if not data:
        return

    final_hv = data.get('final_hv', 0)
    hv_history = data.get('hv_history', [])
    n_evals = data.get('n_evals', [])

    # 计算收敛评估数
    convergence_evals = 0
    if final_hv > 0 and hv_history:
        threshold = 0.9 * final_hv
        for i, hv in enumerate(hv_history):
            if hv >= threshold:
                convergence_evals = n_evals[i] if i < len(n_evals) else (i + 1) * 50
                break

    stats.append({
        'Algorithm': algorithm,
        'Run_ID': data.get('run_id', 1),
        'Final_HV': final_hv,
        'Convergence_Evals': convergence_evals,
        'Total_Time_s': data.get('total_time', 0),
        'N_Pareto': data.get('n_pareto', len(data.get('pareto_thetas', []))),
        'N_Valid': data.get('n_valid', 0),
    })


def _compute_statistics(llmbo_data: dict, nsgaii_data: dict) -> list[dict]:
    """计算统计汇总"""
    stats = []

    for algo, data in [('LLMBO', llmbo_data), ('NSGAII', nsgaii_data)]:
        if not data:
            continue

        # 处理多次运行的情况
        hvs = []
        times = []
        pareto_sizes = []

        if 'runs' in data:
            # 多次运行的格式
            for run in data['runs']:
                if 'final_hv' in run and run['final_hv'] > 0:
                    hvs.append(run['final_hv'])
                if 'total_time' in run:
                    times.append(run['total_time'])
                if 'n_pareto' in run:
                    pareto_sizes.append(run['n_pareto'])
        else:
            # 单次运行
            if data.get('final_hv', 0) > 0:
                hvs.append(data['final_hv'])
            if data.get('total_time', 0) > 0:
                times.append(data['total_time'])
            if data.get('n_pareto', 0) > 0:
                pareto_sizes.append(data['n_pareto'])

        if hvs:
            stats.append({
                'Algorithm': algo,
                'N_Runs': len(hvs),
                'HV_Mean': np.mean(hvs),
                'HV_Std': np.std(hvs) if len(hvs) > 1 else 0,
                'HV_Best': np.max(hvs),
                'HV_Worst': np.min(hvs),
                'Time_Mean_s': np.mean(times) if times else 0,
                'Time_Std_s': np.std(times) if len(times) > 1 else 0,
                'Pareto_Size_Mean': np.mean(pareto_sizes) if pareto_sizes else 0,
            })

    return stats


def export_from_run_comparison(
    comparison_results_path: str,
    output_path: str,
):
    """
    从 run_comparison.py 的输出文件重新格式化导出

    Args:
        comparison_results_path: run_comparison.py 生成的 xlsx 文件
        output_path: 输出文件路径
    """
    try:
        import pandas as pd
    except ImportError:
        log.error("pandas 未安装")
        return

    # 读取已有的 xlsx
    xl = pd.ExcelFile(comparison_results_path)

    # 直接复制所有 sheet 到新文件
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    log.info(f"已格式化导出到：{output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='导出 LLMBO vs NSGA-II 对比结果到 xlsx',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--json-llmbo',
        type=str,
        help='LLMBO 结果 JSON 文件路径'
    )

    parser.add_argument(
        '--json-nsgaii',
        type=str,
        help='NSGA-II 结果 JSON 文件路径'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='已有的对比结果 xlsx 文件路径（用于重新格式化）'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='comparison_export.xlsx',
        help='输出 xlsx 文件路径'
    )

    args = parser.parse_args()

    if args.input:
        # 从已有 xlsx 重新格式化
        export_from_run_comparison(args.input, args.output)
    elif args.json_llmbo and args.json_nsgaii:
        # 从 JSON 导入
        llmbo_data = load_json_results(args.json_llmbo)
        nsgaii_data = load_json_results(args.json_nsgaii)

        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'llmbo_source': args.json_llmbo,
            'nsgaii_source': args.json_nsgaii,
        }

        export_comparison_to_xlsx(llmbo_data, nsgaii_data, args.output, metadata)
    else:
        parser.print_help()
        print("\n示例:")
        print("  # 从 JSON 导入")
        print("  python export_results.py --json-llmbo llmbo.json --json-nsgaii nsgaii.json -o comparison.xlsx")
        print("\n  # 重新格式化已有 xlsx")
        print("  python export_results.py --input comparison_results.xlsx -o formatted.xlsx")


if __name__ == '__main__':
    main()
