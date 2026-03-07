"""验证实验结果是否满足指令5的要求"""
import json

# 加载结果
with open('results/results.json', 'r') as f:
    data = json.load(f)

print("=" * 70)
print("实验结果验证")
print("=" * 70)

# 1. 检查有效率
n_eval = data['n_evaluations']
n_valid = data['n_valid']
valid_rate = n_valid / n_eval * 100

print(f"\n1. 有效率检查:")
print(f"   总评估: {n_eval}")
print(f"   有效样本: {n_valid}")
print(f"   有效率: {valid_rate:.1f}%")

if valid_rate > 60:
    print(f"   ✓ 有效率超过60%")
else:
    print(f"   ✗ 有效率低于60%，建议提高temp_max")

# 2. 检查HV历史是否有连续相同值（bug检测）
hv_history = data['hv_history']
print(f"\n2. HV历史检查（共{len(hv_history)}个记录）:")
print(f"   HV值: {hv_history}")

max_consecutive = 1
current_consecutive = 1
for i in range(1, len(hv_history)):
    if abs(hv_history[i] - hv_history[i-1]) < 1e-8:  # 浮点数比较
        current_consecutive += 1
        max_consecutive = max(max_consecutive, current_consecutive)
    else:
        current_consecutive = 1

print(f"   最长连续相同: {max_consecutive}")
if max_consecutive >= 10:
    print(f"   ✗ 有连续{max_consecutive}个相同HV值，可能存在bug")
else:
    print(f"   ✓ 无异常连续相同值")

# 3. 检查time1是否有重复（随机采样bug检测）
valid_data = [d for d in data['database'] if d['valid']][:15]
times = [round(d['params']['time1'], 2) for d in valid_data]

print(f"\n3. 前15条记录time1值检查:")
print(f"   time1值: {times}")

unique_times = len(set(times))
print(f"   唯一值数量: {unique_times}/{len(times)}")

if unique_times == len(times):
    print(f"   ✓ 所有time1值均不同，无重复采样bug")
else:
    print(f"   ✗ 存在重复time1值，可能有采样bug")

# 4. 最终评估
print(f"\n" + "=" * 70)
print("最终评估:")
print("=" * 70)

checks = [
    valid_rate > 60,
    max_consecutive < 10,
    unique_times == len(times)
]

if all(checks):
    print("✓ 所有检查通过，实验结果有效！")
else:
    print("✗ 部分检查未通过，需要调整配置")
    if not checks[0]:
        print("  - 建议：提高 temp_max 从 318.15K 到 323.15K")
    if not checks[1]:
        print("  - 建议：检查HV计算或wrapper随机状态保护")
    if not checks[2]:
        print("  - 建议：检查随机数生成器状态管理")

print("=" * 70)
