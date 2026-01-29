"""
完整对比实验脚本

运行算法：
1. LLM-MOBO (完整版)
2. Standard-BO (无LLM，无耦合核)
3. Random Search

输出：
- HV收敛曲线对比图
- Pareto前沿对比图
- 详细结果JSON
"""

import asyncio
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入算法
from main import LLMMOBO
from exp.baseline_runner import BaselineOptimizer

# 导入可视化
from utils.visualization import (
    plot_comparison_hv,
    plot_pareto_comparison_3d,
    compute_hv_history
)

# 导入配置
from config import (
    BATTERY_CONFIG, AGING_CONFIG, PARAM_BOUNDS, MOBO_CONFIG
)
from battery_env.wrapper import BatterySimulator


# ========== 实验配置 ==========
EXPERIMENT_CONFIG = {
    'n_iterations': 30,
    'n_warmstart': 5,
    'n_random_init': 10,
    'save_dir': './exp_results/full_comparison',
    'seed': 42,
    'algorithms': [
        'LLM-MOBO',
        'Standard-BO',
        'Random'
    ]
}


async def run_llm_mobo(api_key: str , config: Dict) -> Dict:
    """
    运行LLM-MOBO（完整版）
    
    参数：
        api_key: LLM API密钥
        config: 实验配置
    
    返回：
        results: 结果字典
    """
    print(f"\n{'='*60}")
    print(f"运行算法: LLM-MOBO (完整版)")
    print(f"{'='*60}")
    print(f"  物理耦合核: 启用")
    print(f"  LLM热启动: 启用")
    print(f"  LLM空间加权: 启用")
    
    optimizer = LLMMOBO(
        llm_api_key="sk-Evfy9FZGKZ31bpgdNsDSFfkWMopRE6EN4V4r801oRaIi8jm7",
        n_warmstart=config['n_warmstart'],
        n_random_init=config['n_random_init'],
        n_iterations=config['n_iterations'],
        gamma_init=0.5,
        verbose=True,
        use_coupling=True,
        use_warmstart=True,
        use_llm_acq=True
    )
    
    results = await optimizer.optimize()
    return results


async def run_standard_bo(config: Dict) -> Dict:
    """
    运行Standard-BO（消融版）
    
    配置：
    - 无LLM（api_key=None）
    - 无物理耦合核
    - 无热启动
    - 无LLM加权
    
    参数：
        config: 实验配置
    
    返回：
        results: 结果字典
    """
    print(f"\n{'='*60}")
    print(f"运行算法: Standard-BO (消融版)")
    print(f"{'='*60}")
    print(f"  物理耦合核: 禁用")
    print(f"  LLM热启动: 禁用")
    print(f"  LLM空间加权: 禁用")
    
    optimizer = LLMMOBO(
        llm_api_key=None,
        n_warmstart=config['n_warmstart'],
        n_random_init=config['n_random_init'],
        n_iterations=config['n_iterations'],
        gamma_init=0.5,
        verbose=True,
        use_coupling=False,
        use_warmstart=False,
        use_llm_acq=False
    )
    
    results = await optimizer.optimize()
    return results


def run_random(config: Dict) -> Dict:
    """
    运行Random Search
    
    参数：
        config: 实验配置
    
    返回：
        results: 结果字典
    """
    print(f"\n{'='*60}")
    print(f"运行算法: Random Search")
    print(f"{'='*60}")
    
    # 初始化仿真器
    simulator = BatterySimulator(
        param_set=BATTERY_CONFIG['param_set'],
        init_voltage=BATTERY_CONFIG['init_voltage'],
        init_temp=BATTERY_CONFIG['init_temp'],
        sample_time=BATTERY_CONFIG['sample_time'],
        voltage_max=BATTERY_CONFIG['voltage_max'],
        temp_max=BATTERY_CONFIG['temp_max'],
        soc_target=BATTERY_CONFIG['soc_target'],
        aging_config=AGING_CONFIG
    )
    
    # 创建优化器
    optimizer = BaselineOptimizer(
        simulator=simulator,
        param_bounds=PARAM_BOUNDS,
        n_iterations=config['n_iterations'] + config['n_warmstart'] + config['n_random_init'],
        seed=config['seed'],
        verbose=True
    )
    
    # 运行
    optimizer.run_random()
    results = optimizer.get_results()
    
    # 计算HV历史
    ref_point = np.array([
        MOBO_CONFIG['reference_point']['time'],
        MOBO_CONFIG['reference_point']['temp'],
        -1.0  # Log空间
    ])
    
    hv_history = compute_hv_history(results['database'], ref_point)
    results['hv_history'] = hv_history
    
    return results


def save_results(algorithm: str, results: Dict, save_dir: Path):
    """
    保存结果到JSON
    
    参数：
        algorithm: 算法名称
        results: 结果字典
        save_dir: 保存目录
    """
    valid_data = [r for r in results['database'] if r['valid']]
    
    # 提取统计信息
    stats = {
        'algorithm': algorithm,
        'timestamp': datetime.now().isoformat(),
        'n_evaluations': len(results['database']),
        'n_valid': len(valid_data),
        'final_hv': results['hv_history'][-1] if len(results['hv_history']) > 0 else 0.0,
        'n_pareto': len(results['pareto_front'])
    }
    
    # 提取最佳值
    if len(valid_data) > 0:
        stats['best_time'] = min([r['time'] for r in valid_data])
        stats['best_temp'] = min([r['temp'] for r in valid_data])
        stats['best_aging'] = min([r['aging'] for r in valid_data])
    
    # 保存简化版
    simplified_results = {
        'statistics': stats,
        'hv_history': results['hv_history'],
        'pareto_front_sample': results['pareto_front'][:20]
    }
    
    result_path = save_dir / f"{algorithm}_results.json"
    with open(result_path, 'w') as f:
        json.dump(simplified_results, f, indent=2)
    
    print(f"  结果已保存: {result_path}")


async def main():
    """主实验流程"""
    # 设置随机种子
    np.random.seed(EXPERIMENT_CONFIG['seed'])
    
    # 创建保存目录
    save_dir = Path(EXPERIMENT_CONFIG['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取API密钥
    api_key = os.getenv("LLM_API_KEY")
    if api_key is None:
        print("[Warning] 未设置LLM_API_KEY，LLM-MOBO将跳过")
        if 'LLM-MOBO' in EXPERIMENT_CONFIG['algorithms']:
            EXPERIMENT_CONFIG['algorithms'].remove('LLM-MOBO')
    
    print(f"\n{'='*70}")
    print(f"完整对比实验")
    print(f"{'='*70}")
    print(f"评估预算: {EXPERIMENT_CONFIG['n_iterations']}")
    print(f"随机种子: {EXPERIMENT_CONFIG['seed']}")
    print(f"保存目录: {save_dir}")
    print(f"算法列表: {', '.join(EXPERIMENT_CONFIG['algorithms'])}")
    
    # 运行所有算法
    all_results = {}
    
    for algo in EXPERIMENT_CONFIG['algorithms']:
        try:
            if algo == 'LLM-MOBO':
                results = await run_llm_mobo(api_key, EXPERIMENT_CONFIG)
            elif algo == 'Standard-BO':
                results = await run_standard_bo(EXPERIMENT_CONFIG)
            elif algo == 'Random':
                results = run_random(EXPERIMENT_CONFIG)
            else:
                print(f"[Warning] 未知算法: {algo}")
                continue
            
            all_results[algo] = results
            
            # 保存结果
            save_results(algo, results, save_dir)
            
            print(f"\n算法{algo}完成")
        
        except Exception as e:
            print(f"\n算法{algo}运行失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成对比图
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("生成对比图...")
        print(f"{'='*60}")
        
        try:
            # HV收敛曲线
            plot_comparison_hv(
                all_results,
                save_path=str(save_dir / 'hv_comparison.png'),
                show=False
            )
            
            # Pareto前沿对比
            plot_pareto_comparison_3d(
                all_results,
                save_path=str(save_dir / 'pareto_comparison.png'),
                show=False
            )
            
            print(f"对比图已保存")
        
        except Exception as e:
            print(f"绘图失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print(f"\n{'='*70}")
    print(f"实验总结")
    print(f"{'='*70}")
    
    for algo, res in all_results.items():
        valid_count = res['n_valid']
        final_hv = res['hv_history'][-1] if len(res['hv_history']) > 0 else 0.0
        n_pareto = len(res['pareto_front'])
        
        print(f"\n{algo}:")
        print(f"  有效点: {valid_count}/{res['n_evaluations']}")
        print(f"  Pareto点数: {n_pareto}")
        print(f"  最终HV: {final_hv:.4f}")
        
        if valid_count > 0:
            valid_data = [r for r in res['database'] if r['valid']]
            best_time = min([r['time'] for r in valid_data])
            best_temp = min([r['temp'] for r in valid_data])
            best_aging = min([r['aging'] for r in valid_data])
            print(f"  最佳Time: {best_time}")
            print(f"  最佳Temp: {best_temp:.1f}K")
            print(f"  最佳Aging: {best_aging:.6f}")
    
    print(f"\n实验完成！结果保存在: {save_dir}")


if __name__ == "__main__":
    asyncio.run(main())
