"""分析Pareto前沿"""
import json
import numpy as np

# 加载结果
with open('results/results.json', 'r') as f:
    data = json.load(f)

pf = data['pareto_front']
ref = np.array([10000, 318.15, 0.008])
ideal = np.array([300, 298.15, 1e-6])

print(f"Pareto前沿 ({len(pf)}个点):")
print("=" * 80)

for i, p in enumerate(pf):
    p_arr = np.array(p)
    norm = (p_arr - ideal) / (ref - ideal)
    print(f"{i+1}. time={p[0]:.0f}s, temp={p[1]:.2f}K, aging={p[2]:.6f}")
    print(f"   归一化: [{norm[0]:.3f}, {norm[1]:.3f}, {norm[2]:.3f}]")
    print()

# 计算实际HV（用于对比）
from pymoo.indicators.hv import HV

pf_norm = []
for p in pf:
    p_arr = np.array(p)    
    norm = (p_arr - ideal) / (ref - ideal)
    if np.all(norm < 1.0) and np.all(norm >= 0.0):
        pf_norm.append(norm.tolist())

pf_norm = np.array(pf_norm)
hv_indicator = HV(ref_point=np.ones(3))
hv_calc = hv_indicator(pf_norm)

print("=" * 80)
print(f"计算的HV: {hv_calc:.4f}")
print(f"记录的HV: {data['hv_history'][0]:.4f}")
print(f"一致性: {'✓' if abs(hv_calc - data['hv_history'][0]) < 0.001 else '✗'}")
