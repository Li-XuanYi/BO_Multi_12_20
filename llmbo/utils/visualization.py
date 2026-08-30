"""
可视化模块
实现Pareto前沿绘图、优化历史、HV计算等功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional
try:
    import pygmo as pg
    HAS_PYGMO = True
except ImportError:
    HAS_PYGMO = False
    pg = None


def compute_hv_history(database: List[Dict], ref_point: np.ndarray) -> List[float]:
    """
    计算优化过程中的Hypervolume历史
    
    参数：
        database: 评估历史
        ref_point: 参考点（最坏情况）
    
    返回：
        hv_history: HV值的历史序列
    """
    hv_history = []
    
    for i in range(1, len(database) + 1):
        # 截取前i个点
        current_data = database[:i]
        
        # 提取有效点
        valid_data = [r for r in current_data if r['valid']]
        
        if len(valid_data) == 0:
            hv_history.append(0.0)
            continue
        
        # 提取Pareto前沿
        pareto_front = extract_pareto_front(valid_data)
        
        if len(pareto_front) == 0:
            hv_history.append(0.0)
            continue
        
        # 构建目标矩阵
        objectives = np.array([[
            r['time'],
            r['temp'],
            r['aging']
        ] for r in pareto_front])
        
        # 计算HV
        try:
            hv = pg.hypervolume(objectives)
            hv_value = hv.compute(ref_point)
            hv_history.append(hv_value)
        except Exception as e:
            # HV计算失败（例如参考点不合法）
            hv_history.append(hv_history[-1] if hv_history else 0.0)
    
    return hv_history


def extract_pareto_front(data: List[Dict]) -> List[Dict]:
    """
    提取Pareto前沿（3目标最小化）
    
    参数：
        data: 有效数据点
    
    返回：
        pareto_front: Pareto最优点
    """
    if len(data) == 0:
        return []
    
    # 提取目标值
    objectives = np.array([[
        r['time'],
        r['temp'],
        r['aging']
    ] for r in data])
    
    # Pareto支配关系
    is_pareto = np.ones(len(data), dtype=bool)
    
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                continue
            
            # j支配i：j在所有目标上都不差于i，且至少一个更好
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                is_pareto[i] = False
                break
    
    pareto_front = [data[i] for i in range(len(data)) if is_pareto[i]]
    
    return pareto_front


def plot_pareto_front_3d(
    database: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    绘制3D Pareto前沿
    
    参数：
        database: 评估历史
        save_path: 保存路径（可选）
        show: 是否显示
    """
    # 提取有效数据
    valid_data = [r for r in database if r['valid']]
    invalid_data = [r for r in database if not r['valid']]
    
    if len(valid_data) == 0:
        print("[警告] 无有效数据，跳过绘图")
        return
    
    # 提取Pareto前沿
    pareto_front = extract_pareto_front(valid_data)
    
    # 提取坐标
    valid_time = [r['time'] for r in valid_data]
    valid_temp = [r['temp'] for r in valid_data]
    valid_aging = [r['aging'] for r in valid_data]
    
    pareto_time = [r['time'] for r in pareto_front]
    pareto_temp = [r['temp'] for r in pareto_front]
    pareto_aging = [r['aging'] for r in pareto_front]
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制所有有效点
    ax.scatter(
        valid_time, valid_temp, valid_aging,
        c='lightblue', marker='o', s=30, alpha=0.5,
        label=f'Valid points ({len(valid_data)})'
    )
    
    # 绘制Pareto前沿
    if len(pareto_front) > 0:
        ax.scatter(
            pareto_time, pareto_temp, pareto_aging,
            c='red', marker='*', s=200, edgecolors='black', linewidths=1.5,
            label=f'Pareto front ({len(pareto_front)})'
        )
    
    # 绘制无效点
    if len(invalid_data) > 0:
        invalid_time = [r['time'] for r in invalid_data]
        invalid_temp = [r['temp'] for r in invalid_data]
        invalid_aging = [r['aging'] for r in invalid_data]
        
        ax.scatter(
            invalid_time, invalid_temp, invalid_aging,
            c='gray', marker='x', s=50, alpha=0.3,
            label=f'Invalid ({len(invalid_data)})'
        )
    
    # 设置标签
    ax.set_xlabel('Time (steps)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_zlabel('Aging (%)', fontsize=12)
    ax.set_title('3D Pareto Front', fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Pareto前沿图已保存: {save_path}")
    
    # 显示
    if show:
        plt.show()
    else:
        plt.close()


def plot_optimization_history(
    database: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    绘制优化历史（3个子图）
    
    参数：
        database: 评估历史
        save_path: 保存路径（可选）
        show: 是否显示
    """
    valid_data = [r for r in database if r['valid']]
    
    if len(valid_data) == 0:
        print("[警告] 无有效数据，跳过绘图")
        return
    
    # 提取数据
    iterations = list(range(1, len(valid_data) + 1))
    times = [r['time'] for r in valid_data]
    temps = [r['temp'] for r in valid_data]
    agings = [r['aging'] for r in valid_data]
    
    # 计算累计最小值
    time_min = np.minimum.accumulate(times)
    temp_min = np.minimum.accumulate(temps)
    aging_min = np.minimum.accumulate(agings)
    
    # 创建子图
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # 子图1: Time
    axes[0].plot(iterations, times, 'o-', color='steelblue', alpha=0.6, label='Observed')
    axes[0].plot(iterations, time_min, 'r-', linewidth=2, label='Best so far')
    axes[0].set_ylabel('Time (steps)', fontsize=11)
    axes[0].set_title('Optimization History - Time', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: Temperature
    axes[1].plot(iterations, temps, 'o-', color='orange', alpha=0.6, label='Observed')
    axes[1].plot(iterations, temp_min, 'r-', linewidth=2, label='Best so far')
    axes[1].set_ylabel('Temperature (K)', fontsize=11)
    axes[1].set_title('Optimization History - Temperature', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: Aging
    axes[2].plot(iterations, agings, 'o-', color='green', alpha=0.6, label='Observed')
    axes[2].plot(iterations, aging_min, 'r-', linewidth=2, label='Best so far')
    axes[2].set_xlabel('Iteration', fontsize=11)
    axes[2].set_ylabel('Aging (%)', fontsize=11)
    axes[2].set_title('Optimization History - Aging', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  优化历史图已保存: {save_path}")
    
    # 显示
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_hv(
    all_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    绘制多算法HV收敛对比曲线
    
    参数：
        all_results: {算法名: results}
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'd', 'v']
    
    for i, (algo_name, results) in enumerate(all_results.items()):
        hv_history = results['hv_history']
        iterations = list(range(1, len(hv_history) + 1))
        
        ax.plot(
            iterations, hv_history,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=5,
            markevery=max(1, len(hv_history) // 10),
            label=algo_name
        )
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Hypervolume', fontsize=12)
    ax.set_title('Hypervolume Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  HV对比图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_pareto_comparison_3d(
    all_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    绘制多算法Pareto前沿3D对比
    
    参数：
        all_results: {算法名: results}
        save_path: 保存路径
        show: 是否显示
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['*', 'o', '^', 's', 'd']
    
    for i, (algo_name, results) in enumerate(all_results.items()):
        pareto_front = results['pareto_front']
        
        if len(pareto_front) == 0:
            continue
        
        times = [r['time'] for r in pareto_front]
        temps = [r['temp'] for r in pareto_front]
        agings = [r['aging'] for r in pareto_front]
        
        ax.scatter(
            times, temps, agings,
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=150,
            edgecolors='black',
            linewidths=1,
            label=f'{algo_name} ({len(pareto_front)})',
            alpha=0.8
        )
    
    ax.set_xlabel('Time (steps)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_zlabel('Aging (%)', fontsize=12)
    ax.set_title('Pareto Front Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Pareto对比图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_results_json(results: Dict, save_path: str):
    """
    保存结果到JSON文件
    
    参数：
        results: 结果字典
        save_path: 保存路径
    """
    import json
    
    # 将numpy数组转换为列表
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"  结果已保存: {save_path}")


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("测试 visualization...")
    
    # 创建虚拟数据库
    np.random.seed(42)
    fake_database = []
    
    for i in range(30):
        fake_database.append({
            'params': {
                'current1': np.random.uniform(3.0, 6.0),
                'switch_soc': np.random.uniform(0.3, 0.7),
                'current2': np.random.uniform(1.0, 4.0)
            },
            'time': np.random.randint(30, 100),
            'temp': np.random.uniform(300, 310),
            'aging': np.random.uniform(0.001, 0.05),
            'valid': np.random.rand() > 0.2  # 80%有效
        })
    
    # 测试Pareto前沿提取
    valid_data = [r for r in fake_database if r['valid']]
    pareto_front = extract_pareto_front(valid_data)
    print(f"\n有效点: {len(valid_data)}")
    print(f"Pareto点: {len(pareto_front)}")
    
    # 测试HV计算
    ref_point = np.array([150, 313, 0.1])
    hv_history = compute_hv_history(fake_database, ref_point)
    print(f"\nHV历史: {len(hv_history)}个点")
    print(f"最终HV: {hv_history[-1]:.4f}")
    
    # 测试绘图
    print("\n生成可视化...")
    plot_pareto_front_3d(fake_database, save_path='test_pareto.png', show=False)
    plot_optimization_history(fake_database, save_path='test_history.png', show=False)
    
    print("\n测试完成！")


def select_pareto_representatives(
    pareto_front: List[Dict],
    k: int = 6
) -> List[Dict]:
    """
    混合策略选择 Pareto 代表点（约束 C-7 实现）
    
    策略：
    1. 强制选 3 个极端点（每个目标的最优解）
    2. 选 1 个膝点（距 CHIM 平面最远）
    3. FPS 补充剩余名额（最多 6 个）
    
    参数：
        pareto_front: Pareto 前沿
        k: 最多选择 k 个代表点（约束 C-7: 最多 6 个）
    
    返回：
        representatives: 代表点列表
    """
    if len(pareto_front) == 0:
        return []
    
    # 提取目标值
    objectives = np.array([[
        r['time'],
        r['temp'],
        r['aging']
    ] for r in pareto_front])
    
    n = len(objectives)
    selected_indices = []
    
    # ========== Step 1: 强制选 3 个极端点 ==========
    for dim in range(3):
        idx = int(np.argmin(objectives[:, dim]))
        if idx not in selected_indices:
            selected_indices.append(idx)
    
    # ========== Step 2: 选 1 个膝点（距 CHIM 平面最远） ==========
    if len(selected_indices) >= 3:
        # 三个极端点构成 CHIM 平面
        anchors = objectives[selected_indices[:3]]
        
        # 平面法向量
        v1 = anchors[1] - anchors[0]
        v2 = anchors[2] - anchors[0]
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm > 1e-10:
            normal = normal / normal_norm
            
            # 计算每个点到 CHIM 平面的距离
            dists = np.abs((objectives - anchors[0]) @ normal)
            
            # 选择距离最远的点（膝点）
            remaining_dists = dists.copy()
            for idx in selected_indices:
                remaining_dists[idx] = -1
            
            knee_idx = int(np.argmax(remaining_dists))
            if remaining_dists[knee_idx] > 0:
                selected_indices.append(knee_idx)
    
    # ========== Step 3: FPS 补充剩余名额 ==========
    while len(selected_indices) < k and len(selected_indices) < n:
        min_dists = []
        for i in range(n):
            if i in selected_indices:
                min_dists.append(-1)
                continue
            
            dists_to_selected = [np.linalg.norm(objectives[i] - objectives[j]) for j in selected_indices]
            min_dists.append(np.min(dists_to_selected))
        
        remaining_indices = [i for i in range(n) if i not in selected_indices]
        if not remaining_indices:
            break
        
        best_idx = max(remaining_indices, key=lambda i: min_dists[i])
        selected_indices.append(best_idx)
    
    return [pareto_front[i] for i in selected_indices]
