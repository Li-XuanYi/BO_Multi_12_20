# LLM_Region 技术报告

**作者**: Claude  
**日期**: 2026-05-17  
**版本**: v1.0  
**读者**: 新入职工程师

---

## 目录

1. [背景介绍](#1-背景介绍)
2. [LLM_Region 是什么](#2-llm_region-是什么)
3. [工作原理](#3-工作原理)
4. [代码架构](#4-代码架构)
5. [当前状态](#5-当前状态)
6. [存在的问题](#6-存在的问题)
7. [修复历史](#7-修复历史)
8. [下一步工作](#8-下一步工作)

---

## 1. 背景介绍

### 1.1 项目背景

本项目是 **LLAMBO-MO** (LLM-Augmented Multi-Objective Bayesian Optimization)，用于电池快充协议的多目标贝叶斯优化。目标是同时优化三个冲突的目标：

- **充电时间** (time): 越短越好
- **峰值温升** (temperature): 越低越好  
- **容量衰减** (aging): 越小越好

### 1.2 为什么需要 LLM_Region

传统的贝叶斯优化 (BO) 使用采集函数（如 EI - Expected Improvement）来选择下一个评估点。但 EI 是**纯粹数学驱动**的，它不知道电池物理。

**LLM_Region** 的设计思想：让 LLM（大语言模型）基于电池领域知识，推荐一个"有希望的区域"，然后 BO 在这个区域内重点搜索。

---

## 2. LLM_Region 是什么

### 2.1 核心概念

LLM_Region 是 Region-Lifted GP 的一部分，它：

1. **查询 LLM**: 在每个 BO 迭代中，询问 LLM "哪里可能有更好的充电协议？"
2. **获得建议**: LLM 返回一个区域 (region) 或点 (point)
3. **提升 GP**: 在这个区域内"抬高"高斯过程 (GP) 的均值预测
4. **指导搜索**: 采集函数更倾向于选择该区域内的点

### 2.2 两种建议类型

```python
# 类型 1: 具体点 (当 LLM 有信心时)
{
    "kind": "point",
    "point": {"I1": 4.5, "I2": 3.8, "I3": 2.5, "dSOC1": 0.25, "dSOC2": 0.20},
    "confidence": 0.75,
    "reason": "高电流+中等SOC分配可平衡时间和温度"
}

# 类型 2: 区域 (当 LLM 不太确定但知道大致方向)
{
    "kind": "region",
    "lb": {"I1": 4.0, "I2": 3.5, "I3": 2.2, "dSOC1": 0.22, "dSOC2": 0.18},  # 下界
    "ub": {"I1": 5.0, "I2": 4.2, "I3": 2.8, "dSOC1": 0.28, "dSOC2": 0.25},  # 上界
    "confidence": 0.60,
    "reason": "这个电流范围内可能有好的折衷"
}
```

### 2.3 决策变量

电池协议由 5 个参数决定：

| 参数 | 范围 | 含义 |
|------|------|------|
| I1 | [2.0, 6.0] A | 第一阶段电流 |
| I2 | [2.0, 5.0] A | 第二阶段电流 |
| I3 | [2.0, 3.0] A | 第三阶段电流 |
| dSOC1 | [0.10, 0.40] | 第一阶段 SOC 宽度 |
| dSOC2 | [0.10, 0.30] | 第二阶段 SOC 宽度 |

约束: `dSOC1 + dSOC2 < 0.70` (保证第三阶段 SOC 为正)

---

## 3. 工作原理

### 3.1 整体流程

```
BO Iteration t
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 查询 LLM                                           │
│  - 构建 prompt (包含当前优化状态、历史数据)                    │
│  - 调用 deepseek-v3 API                                      │
│  - 获得 region/point 建议                                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 解析响应                                            │
│  - 提取 JSON                                                 │
│  - 验证字段 (kind, lb/ub/point, confidence)                  │
│  - 转换为内部格式 (LLMRegionPreference)                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 评估 Region (Guard Rails)                          │
│  - 检查体积是否足够大                                         │
│  - 检查宽度是否合理                                           │
│  - 检查是否在可行域内                                         │
│  - 计算可靠性分数                                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Region-Lifted EI 计算                               │
│  - 在 region 内采样锚点                                       │
│  - 计算 GP 均值和方差                                         │
│  - 对 region 内点 "降低" 均值 (使 EI 变大)                    │
│  - 重新计算提升后的 EI                                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: 选择候选点                                          │
│  - 比较 lifted EI 和 plain EI                                │
│  - 如果 lifted EI 更好，接受 region 影响                      │
│  - 否则回退到标准 EI                                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心公式: Mean Shift

Region-Lifted GP 的核心是**均值提升 (Mean Shift)**：

```
mean_lifted(x) = mean_gp(x) - shift(x)

其中:
shift(x) = λ_t × reliability × max(correlation(x, center), 0)
```

参数解释：
- `λ_t`: 退火系数，随迭代递减 (前期激进，后期保守)
- `reliability`: LLM 建议的可靠性 (0-1)
- `correlation`: x 与 region 中心的高斯相关性

**直观理解**: 在 region 内，GP 均值被人为"降低"（因为是最小化问题），这使得 EI 变大，采集函数更倾向于选这些点。

### 3.3 锚点加权 (Anchor Weighting)

为了选择 region 内的代表点，使用加权策略：

```python
weights = 0.60 × ei_score + 0.25 × sigma_score + 0.15 × novelty_score
```

- **60% EI**: 优先选高期望改进的点
- **25% σ**: 兼顾不确定性探索
- **15% novelty**: 避免聚类，促进分散

---

## 4. 代码架构

### 4.1 关键文件

```
New_LLMBO/
├── llm/
│   ├── llm_interface.py          # 主要入口
│   │   ├── query_region_preference()   # 查询 LLM
│   │   └── _extract_json_flexible()    # JSON 提取
│   └── region_prompt.py          # Prompt 生成
│       └── render_region_preference_prompt()
│
├── llmbo/
│   ├── region_lifted_gp.py       # Region-Lifted GP 核心
│   │   ├── parse_region_preference_payload()  # 解析 LLM 响应
│   │   ├── evaluate_region_lift_on_pool()     # 评估 region
│   │   └── _coerce_param_dict()               # 参数强制转换
│   │
│   └── optimizer.py              # BO 主循环
│       └── _maybe_apply_region_lifted_gp()    # 应用 region lift
│
└── Ablation_Exp/                 # 消融实验
    └── Process/tools/
        └── run_ablation_8409_8413_exp_prompt.py
```

### 4.2 关键类

#### LLMRegionPreference

```python
@dataclass
class LLMRegionPreference:
    kind: str                      # "point", "region", or "none"
    point: Optional[Dict]          # 具体点坐标
    lb: Optional[Dict]             # 区域下界
    ub: Optional[Dict]             # 区域上界
    confidence: float              # 置信度 (0-1)
    reason: str                    # 理由
    mechanistic_thinking: str      # 机理解释
    parser_status: str             # 解析状态 ("ok" or error)
```

#### RegionLiftConfig

配置参数（关键）：
- `region_lift_active_until`: region lift 活跃到第几轮 (默认 40)
- `region_lift_min_confidence`: 最小置信度 (默认 0.60)
- `region_lift_lambda_max`: 最大提升强度 (默认 0.25)
- `region_lift_trust_init`: 初始信任度 (默认 0.5)
- `region_lift_trust_beta`: 信任度调整速率 (默认 0.2)

### 4.3 调用链

```
BayesOptimizer.run()
    └── _maybe_apply_region_lifted_gp()
            └── llm_interface.query_region_preference()
                    └── render_region_preference_prompt()  # 生成 prompt
                    └── LLMCaller.call()                  # API 调用
                    └── parse_region_preference_payload() # 解析响应
            └── evaluate_region_lift_on_pool()
                    └── _validate_region()                # 验证 region
                    └── _compute_lifted_ei()              # 计算提升 EI
```

---

## 5. 当前状态

### 5.1 修复完成 (2026-05-17)

**问题**: 所有 LLM 调用都返回 `parse_fail` (100% 失败)

**根本原因**: 
1. `_coerce_param_dict` 要求所有 5 个参数键必须存在
2. JSON 提取器不支持 markdown 代码块、多余文本等 LLM 常见输出
3. 缺乏容错机制

**修复内容**:

| 模块 | 改进 |
|------|------|
| `_coerce_param_dict` | 支持部分字典 (≥3 键自动填充缺失值) |
| `_extract_region_bounds` | 支持 `center+width` 格式 |
| `parse_region_preference_payload` | 添加详细日志 |
| `_extract_json_flexible` | 支持 markdown/多余文本/尾部逗号/单引号 |

### 5.2 修复验证

**10 轮消融实验结果** (2 seeds × 4 variants × 10 iters):

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| `parse_fail` | 25/25 (100%) | 0/20 (0%) ✅ |
| JSON 解析成功率 | 0% | 100% ✅ |

**LLM 响应现在能正确解析**：
```
Iter 1: kind=region, confidence=0.70, status=ok
Iter 2: kind=point, confidence=0.70, status=ok
Iter 3: kind=region, confidence=0.70, status=ok
```

---

## 6. 存在的问题

### 6.1 主要问题：Region-Lift 接受率低

虽然解析成功，但 region-lift 建议**很少被采纳**。

**Fallback 分布** (Baseline+LLM_Region, seed 8409):
```json
{
    "same_as_plain": 4,        # 40% - LLM 建议与标准 EI 相同
    "override_disabled": 1,     # 10% - 覆盖被禁用
    "too_close_to_existing": 5  # 50% - 建议点太接近已有观测
}
```

### 6.2 问题分析

#### 问题 1: same_as_plain (40-60%)

**现象**: LLM 推荐的 region 与 GP 认为的最佳区域重合。

**可能原因**:
1. LLM 只是重复了 GP 的预测（没有提供新信息）
2. Region 太小，大部分点本来就在 EI 高的区域
3. GP 已经收敛到该区域

#### 问题 2: too_close_to_existing (10-50%)

**现象**: LLM 推荐的点在已有观测附近。

**代码逻辑**:
```python
if min_distance_to_existing < threshold:
    fallback_reason = "too_close_to_existing"
```

**可能原因**:
1. LLM 缺乏历史数据上下文
2. Prompt 中 top_scalar_points 信息不足
3. 阈值设置过于严格

#### 问题 3: outside_region (0-20%)

**现象**: 好的候选点落在 LLM 推荐的 region 之外。

**可能原因**:
1. LLM 的 region 估计过窄
2. GP 发现了 LLM 没考虑到的区域

### 6.3 信任度机制

当前信任度更新：
```python
new_trust = trust + beta × (1 - relative_hv_drop)
```

**问题**: 
- 如果 HV 没提升，信任度会**下降**
- 但 LLM 的 region 可能仍然有效，只是 GP 采样不够

### 6.4 参数敏感性

关键超参数及其影响：

| 参数 | 当前值 | 影响 |
|------|--------|------|
| `region_lift_min_confidence` | 0.60 | 高于此值才接受建议 |
| `region_lift_active_until` | 16/24 | 仅前 16/24 轮激活 |
| `region_lift_lambda_max` | 0.25 | 最大提升强度 |
| `region_lift_min_width` | 0.03 | 最小 region 宽度 |

**问题**: 这些参数是启发式设置的，可能没有针对 deepseek-v3 优化。

---

## 7. 修复历史

### 7.1 第一阶段：解析层修复 (2026-05-17) ✅

**问题**: `parse_fail: 25/25`

**修复**:
```python
# 修复前：要求所有 5 个键
for key in PARAM_KEYS:
    if key not in value:
        return None  # 直接失败

# 修复后：允许部分填充
defaults = {"I1": 4.0, "I2": 3.5, "I3": 2.5, "dSOC1": 0.25, "dSOC2": 0.20}
if len(out) >= 3:  # 只要有 3 个键
    for key in missing_keys:
        out[key] = defaults[key]  # 填充默认值
```

**结果**: 解析成功率 0% → 100%

### 7.2 第二阶段：待进行

需要解决的问题：
1. 提高 region-lift 接受率
2. 优化 prompt 使 LLM 提供更有区分度的建议
3. 调整超参数适应 deepseek-v3

---

## 8. 下一步工作

### 8.1 短期任务

#### 任务 1: 分析失败原因分布

**目标**: 理解为什么大多数建议被 fallback

**方法**:
```bash
# 运行 50 轮实验收集详细日志
python run_ablation_8409_8413_exp_prompt.py --iterations 50

# 分析 fallback_distribution
python analyze_fallbacks.py
```

**预期输出**:
- 各 fallback 原因的占比饼图
- 不同 seed 的稳定性分析
- 随 iteration 变化的 fallback 趋势

#### 任务 2: 调整接受阈值

**假设**: `region_lift_min_confidence=0.60` 过高

**实验设计**:
```python
confidence_thresholds = [0.40, 0.50, 0.60, 0.70]
for threshold in confidence_thresholds:
    run_experiment(threshold=threshold)
    measure_accept_rate()
```

#### 任务 3: 改进 Prompt

**当前 Prompt 问题**:
- 给 LLM 的上下文信息不够
- 没有明确告诉 LLM "要探索 GP 不知道的区域"

**改进方向**:
```python
# 在 prompt 中加入
"""
GP 当前预测的最佳区域是: {gp_best_region}
但 GP 在以下区域不确定性高: {high_uncertainty_regions}
请推荐一个与 GP 最佳区域不同的、有潜力的区域。
"""
```

### 8.2 中期任务

#### 任务 4: 自适应信任度机制

当前信任度只基于 HV 变化：
```python
# 当前
new_trust = trust + beta × (1 - relative_hv_drop)
```

改进为考虑更多信息：
```python
# 改进
quality_score = compute_region_quality(region, historical_data)
exploration_bonus = compute_exploration_bonus(region, gp_uncertainty)
new_trust = trust + beta × (quality_score + exploration_bonus)
```

#### 任务 5: Region 尺寸自适应

根据优化阶段动态调整 region 大小：
```python
if iteration < max_iterations × 0.3:
    # 早期：大 region 探索
    width_scale = 0.20
elif iteration < max_iterations × 0.7:
    # 中期：中等 region
    width_scale = 0.10
else:
    # 后期：小 region 精细搜索
    width_scale = 0.05
```

### 8.3 长期任务

#### 任务 6: 多 LLM 调用集成

当前：单次调用，失败则 fallback
改进：多次调用 + 投票/集成

```python
responses = llm.call(prompt, n=3)  # 调用 3 次
regions = [parse(r) for r in responses]
consensus_region = compute_consensus(regions)  # 求交集或平均
```

#### 任务 7: 在线学习 Prompt

根据历史成功率，自动调整 prompt：
```python
# 如果 "explore more" 类型的建议成功率高
prompt_template = add_emphasis("exploration", weight=1.5)
```

---

## 9. 给新工程师的建议

### 9.1 学习路径

1. **第 1 周**: 理解贝叶斯优化基础
   - 阅读 `docs/bo_intro.md` (如果存在)
   - 理解 EI (Expected Improvement) 公式
   - 运行 `main.py --demo` 观察基础流程

2. **第 2 周**: 深入 Region-Lifted GP
   - 阅读本报告
   - 精读 `llmbo/region_lifted_gp.py`
   - 画流程图理解 `evaluate_region_lift_on_pool`

3. **第 3 周**: 实验与调试
   - 运行消融实验 (10 iterations)
   - 查看生成的日志
   - 修改超参数观察效果

### 9.2 调试技巧

**如何查看 LLM 响应**:
```bash
# 设置详细日志
export LLM_LOG_LEVEL=DEBUG

# 运行实验
python main.py --config config.json --verbose

# 查看原始 LLM 响应
cat experiment_log.txt | grep "raw_text_preview"
```

**如何测试解析逻辑**:
```python
# 测试新的 LLM 响应格式
from llmbo.region_lifted_gp import parse_region_preference_payload

test_response = {
    "kind": "region",
    "lb": {"I1": 3.8, "I2": 3.0, "I3": 2.2, "dSOC1": 0.20, "dSOC2": 0.18},
    "ub": {"I1": 4.5, "I2": 4.0, "I3": 2.8, "dSOC1": 0.30, "dSOC2": 0.25},
    "confidence": 0.7
}

result = parse_region_preference_payload(test_response)
print(f"Parsed: kind={result.kind}, status={result.parser_status}")
```

**如何可视化 region**:
```python
# 在 optimizer.py 中添加
import matplotlib.pyplot as plt

def plot_region(lb, ub, title="LLM Region"):
    fig, ax = plt.subplots()
    # 2D projection (I1-I2 plane)
    rect = plt.Rectangle((lb['I1'], lb['I2']), 
                         ub['I1']-lb['I1'], ub['I2']-lb['I2'],
                         fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.set_xlabel('I1')
    ax.set_ylabel('I2')
    ax.set_title(title)
    plt.savefig(f'region_iter_{iteration}.png')
```

### 9.3 常见陷阱

1. **不要修改 parse 逻辑而不测试**: 
   - 每次修改 `_coerce_param_dict` 都要运行 `test_region_fix.py`

2. **注意 numpy vs list**:
   - LLM 接口期望 list，内部用 numpy
   - 转换: `np.array(list)` 或 `arr.tolist()`

3. **日志级别**:
   - 生产用 INFO
   - 调试用 DEBUG
   - 不要提交 DEBUG 级别的代码

4. **API 密钥**:
   - 永远不要硬编码在代码中
   - 使用环境变量或配置文件

---

## 10. 参考资源

### 10.1 内部文档

- `CLAUDE.md`: 项目整体架构
- `llm/README_WarmStart.md`: WarmStart 说明
- `Ablation_Exp/README.md`: 消融实验说明

### 10.2 关键代码行号

| 功能 | 文件 | 行号 |
|------|------|------|
| 查询 LLM | `llm/llm_interface.py` | 1229-1330 |
| 解析响应 | `llmbo/region_lifted_gp.py` | 166-250 |
| 评估 Region | `llmbo/region_lifted_gp.py` | 508-721 |
| 应用 Lift | `llmbo/optimizer.py` | 1303-1424 |
| Prompt 生成 | `llm/region_prompt.py` | 52-90 |

### 10.3 实验脚本

```bash
# 快速测试 (10 iterations, 2 seeds)
python test_region_10iter.py

# 完整消融实验 (50 iterations, 5 seeds)
python Ablation_Exp/Process/tools/run_ablation_8409_8413_exp_prompt.py \
    --iterations 50 \
    --model deepseek-v3

# 查看报告
python Ablation_Exp/Process/tools/run_ablation_8409_8413_exp_prompt.py \
    --summarize-only \
    --output-root Ablation_Exp/experiment_records/quick_10iter_test
```

---

## 附录 A: 术语表

| 术语 | 解释 |
|------|------|
| BO | Bayesian Optimization (贝叶斯优化) |
| GP | Gaussian Process (高斯过程) |
| EI | Expected Improvement (期望改进) |
| HV | Hypervolume (超体积，多目标优化指标) |
| Region | 参数空间中的一个超矩形区域 |
| Lift | 提升，指对 GP 均值的人工调整 |
| Fallback | 回退，当 LLM 建议无效时使用标准方法 |
| Tchebycheff | 切比雪夫标量化方法 |

---

## 附录 B: 关键指标解释

### 指标 1: `fallback_distribution`

记录 LLM 建议被拒绝的原因分布。

**健康状态**: 
- `parse_fail` ≈ 0%
- `same_as_plain` < 30%
- `region_accepted` > 50%

### 指标 2: `region_lift_accept_rate`

实际应用 region-lift 的迭代比例。

**目标**: > 40%

### 指标 3: `effective_lift_accept_rate`

接受 lift 后 HV 确实提升的比例。

**目标**: > 30%

### 指标 4: `region_pool_influenced_acquisition_count`

候选池受 region 影响的次数。

**说明**: 即使最终没选 region 内的点，只要候选池被影响就有价值。

---

**报告结束**

如有疑问，请联系项目维护者或查阅相关代码注释。
