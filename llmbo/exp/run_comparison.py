"""
对比实验运行脚本

运行：
1. Random Search
2. Standard-BO

输出：
- 各算法的database和pareto_front
- 结果JSON文件
- （可选）可视化对比图
"""

import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入基准算法
from exp.baseline_runner import BaselineOptimizer

# 导入配置
from config import BATTERY_CONFIG, AGING_CONFIG, PARAM_BOUNDS
from battery_env.wrapper import BatterySimulator
from utils.visualization import plot_pareto_front_3d, plot_optimization_history


# ========== 实验配置 ==========
EXPERIMENT_CONFIG = {
    'n_iterations': 30,        # 评估预算
    'n_random_init': 10,       # Standard-BO的随机初始化
    'save_dir': './exp_results',
    'seed': 42,                # 随机种子（可复现）
    'algorithms': [
        'Random',
        'Standard-BO'
    ]
}


def run_experiment(algorithm: str, config: Dict) -> Dict:
    """
    运行单个算法
    
    参数：
        algorithm: 算法名称
        config: 实验配置
    
    返回：
        results: 结果字典
    """
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
        n_iterations=config['n_iterations'],
        seed=config['seed'],
        verbose=True
    )
    
    # 运行算法
    if algorithm == 'Random':
        optimizer.run_random()
    elif algorithm == 'Standard-BO':
        optimizer.run_standard_bo(n_random_init=config['n_random_init'])
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # 获取结果
    results = optimizer.get_results()
    
    return results


def save_results(algorithm: str, results: Dict, save_dir: Path):
    """
    保存结果到JSON
    
    参数：
        algorithm: 算法名称
        results: 结果字典
        save_dir: 保存目录
    """
    # 简化database用于JSON保存
    simplified_results = {
        'algorithm': algorithm,
        'timestamp': datetime.now().isoformat(),
        'n_evaluations': results['n_evaluations'],
        'n_valid': results['n_valid'],
        'best_time': min([r['time'] for r in results['database'] if r['valid']], default=np.inf),
        'best_temp': min([r['temp'] for r in results['database'] if r['valid']], default=np.inf),
        'best_aging': min([r['aging'] for r in results['database'] if r['valid']], default=np.inf),
        'database_sample': results['database'][:20]  # 只保存前20条
    }
    
    # 保存
    result_path = save_dir / f"{algorithm}_results.json"
    with open(result_path, 'w') as f:
        json.dump(simplified_results, f, indent=2)
    
    print(f"  结果已保存: {result_path}")


def main():
    """主实验流程"""
    # 设置随机种子
    np.random.seed(EXPERIMENT_CONFIG['seed'])
    
    # 创建保存目录
    save_dir = Path(EXPERIMENT_CONFIG['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"对比实验开始")
    print(f"{'='*60}")
    print(f"评估预算: {EXPERIMENT_CONFIG['n_iterations']}")
    print(f"随机种子: {EXPERIMENT_CONFIG['seed']}")
    print(f"保存目录: {save_dir}")
    print(f"算法列表: {', '.join(EXPERIMENT_CONFIG['algorithms'])}")
    
    # 运行所有算法
    all_results = {}
    
    for algo in EXPERIMENT_CONFIG['algorithms']:
        try:
            print(f"\n{'='*60}")
            print(f"运行算法: {algo}")
            print(f"{'='*60}")
            
            results = run_experiment(algo, EXPERIMENT_CONFIG)
            all_results[algo] = results
            
            # 保存结果
            save_results(algo, results, save_dir)
            
            # 生成可视化
            plot_pareto_front_3d(
                results['database'],
                save_path=str(save_dir / f"{algo}_pareto.png"),
                show=False
            )
            
            plot_optimization_history(
                results['database'],
                save_path=str(save_dir / f"{algo}_history.png"),
                show=False
            )
            
            print(f"\n算法{algo}完成")
        
        except Exception as e:
            print(f"\n算法{algo}运行失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"实验总结")
    print(f"{'='*60}")
    for algo, res in all_results.items():
        valid_count = res['n_valid']
        best_time = min([r['time'] for r in res['database'] if r['valid']], default=np.inf)
        best_temp = min([r['temp'] for r in res['database'] if r['valid']], default=np.inf)
        best_aging = min([r['aging'] for r in res['database'] if r['valid']], default=np.inf)
        
        print(f"\n{algo}:")
        print(f"  有效点: {valid_count}/{res['n_evaluations']}")
        print(f"  最佳Time: {best_time}")
        print(f"  最佳Temp: {best_temp:.1f}K")
        print(f"  最佳Aging: {best_aging:.6f}")
    
    print(f"\n实验完成！结果保存在: {save_dir}")


if __name__ == "__main__":
    main()
