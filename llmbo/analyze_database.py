"""分析数据库中的目标值和Pareto前沿"""
import json
import numpy as np

# 加载结果
with open('results/results.json', 'r') as f:
    data = json.load(f)

# 提取所有有效点的目标值
valid_points = []
for i, entry in enumerate(data['database']):
    if entry['valid']:
        obj = [entry['time'], entry['temp'], entry['aging']]
        valid_points.append({
            'index': i,
            'objectives': obj,
            'params': entry['params']
        })

print(f"有效点总数: {len(valid_points)}")
print("=" * 80)

# 提取非支配点（Pareto前沿）
def is_dominated(p1, p2):
    """检查p1是否被p2支配"""
    better = False
    for i in range(len(p1)):
        if p2[i] > p1[i]:  # 最小化问题
            return False
        if p2[i] < p1[i]:
            better = True
    return better

objectives = np.array([p['objectives'] for p in valid_points])
pareto_mask = np.ones(len(objectives), dtype=bool)

for i in range(len(objectives)):
    for j in range(len(objectives)):
        if i != j and pareto_mask[i]:
            if is_dominated(objectives[i], objectives[j]):
                pareto_mask[i] = False
                break

pareto_points = [valid_points[i] for i in range(len(valid_points)) if pareto_mask[i]]

print(f"Pareto前沿点数: {len(pareto_points)}\n")

# 显示Pareto前沿
ref = np.array([10000, 318.15, 0.008])
ideal = np.array([300, 298.15, 1e-6])

for i, pt in enumerate(pareto_points):
    obj = pt['objectives']
    params = pt['params']
    norm = (np.array(obj) - ideal) / (ref - ideal)
    
    print(f"点{i+1} (数据库索引{pt['index']+1}):")
    print(f"  目标: time={obj[0]:.0f}s, temp={obj[1]:.2f}K, aging={obj[2]:.6f}")
    print(f"  归一化: [{norm[0]:.3f}, {norm[1]:.3f}, {norm[2]:.3f}]")
    print(f"  参数: I1={params['current1']:.2f}, T1={params['time1']:.0f}, "
          f"I2={params['current2']:.2f}, Vsw={params['v_switch']:.2f}")
    print()

# 计算HV
pf_norm = []
for pt in pareto_points:
    obj = np.array(pt['objectives'])
    norm = (obj - ideal) / (ref - ideal)
    if np.all(norm < 1.0) and np.all(norm >= 0.0):
        pf_norm.append(norm)

if len(pf_norm) > 0:
    pf_norm = np.array(pf_norm)
    
    try:
        from pymoo.indicators.hv import HV
        hv_indicator = HV(ref_point=np.ones(3))
        hv_calc = hv_indicator(pf_norm)
        print("=" * 80)
        print(f"计算的HV: {hv_calc:.4f}")
        print(f"记录的HV: {data['hv_history'][0]:.4f}")
    except:
        print("无法计算HV（pymoo未安装）")

# 检查前5个点是否在Pareto前沿中
print("\n" + "=" * 80)
print("初始化阶段（前15个点）的Pareto贡献:")
initial_in_pareto = sum(1 for pt in pareto_points if pt['index'] < 15)
print(f"  初始化阶段在Pareto前沿中的点: {initial_in_pareto}/{len(pareto_points)}")

bo_in_pareto = sum(1 for pt in pareto_points if pt['index'] >= 15)
print(f"  BO迭代阶段在Pareto前沿中的点: {bo_in_pareto}/{len(pareto_points)}")

if bo_in_pareto == 0:
    print("\n  ⚠ BO阶段没有产生任何Pareto点！")
    print("  这解释了为什么HV没有改善。")
