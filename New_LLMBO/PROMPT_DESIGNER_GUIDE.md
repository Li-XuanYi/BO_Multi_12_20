# LLMBO Prompt 设计专家指南

> **文档用途**: 帮助新加入的Prompt设计专家快速理解LLMBO代码框架，掌握现有Prompt设计，并提供改进方向。
>
> **版本**: 2025-05-02
> **项目**: LLAMBO-MO (LLM-Augmented Multi-Objective Bayesian Optimization)

---

## 1. 项目概述

### 1.1 核心任务

LLMBO是一个**多目标贝叶斯优化框架**，用于优化锂电池快速充电协议。我们优化一个**5维决策变量**的3阶段恒流（CC）充电协议：

| 变量 | 范围 | 物理含义 |
|------|------|----------|
| **I1** | [2.0, 6.0] A | 阶段1电流（低SOC快速充电） |
| **I2** | [2.0, 5.0] A | 阶段2电流（中SOC过渡） |
| **I3** | [2.0, 3.0] A | 阶段3电流（高SOC保护） |
| **dSOC1** | [0.10, 0.40] | 阶段1的SOC跨度 |
| **dSOC2** | [0.10, 0.30] | 阶段2的SOC跨度 |

**约束**: `dSOC1 + dSOC2 ≤ 0.70`（确保阶段3有最小SOC窗口）

**3个优化目标**（全部最小化）：
1. **充电时间** (time_s) — 从0%到80% SOC的总时间
2. **峰值温升** (delta_temp_K) — 充电过程中电池最高温度相对于起始温度的增量
3. **老化程度** (aging_pct) — 电池容量衰减百分比

### 1.2 LLM介入的两个Touchpoint

我们的核心创新是在BO流程中引入**两个LLM交互点**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LLMBO 工作流程                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────┐     ┌──────────────┐     ┌──────────────────┐    │
│  │  Touchpoint 1b   │     │   PyBaMM     │     │  Touchpoint 2    │    │
│  │  WarmStart       │────▶│  Simulator   │────▶│  Iter Guidance   │    │
│  │  (初始化阶段)     │     │  (物理仿真)   │     │  (迭代Guidance)  │    │
│  └──────────────────┘     └──────────────┘     └──────────────────┘    │
│          │                                            │                 │
│          ▼                                            ▼                 │
│   LLM生成初始候选                              LLM提供区域/点偏好        │
│   替换随机初始化                               影响GP采集函数            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 系统架构

### 2.1 模块职责

```
New_LLMBO/
├── llm/                          # 【Prompt设计核心区域】
│   ├── llm_interface.py          # LLM交互主接口（两个Touchpoint的实现）
│   ├── warmstart_prompt.py       # WarmStart Prompt构建器
│   ├── iteration_prompt.py       # 迭代Guidance Prompt构建器
│   ├── rerank_prompt.py          # 候选重排序Prompt构建器
│   ├── region_prompt.py          # Region-Lifted GP Prompt构建器
│   └── templates/                # 【Prompt模板文件】
│       ├── warmstart/
│       │   ├── basic.txt         # 最小化模板（baseline）
│       │   ├── problem.txt       # 中等细节模板
│       │   └── detailed.txt      # 完整模板（推荐）
│       ├── iteration_guidance.txt # 迭代Guidance模板
│       └── candidate_rerank.txt  # 候选重排序模板
│
├── llmbo/                        # BO核心算法
│   ├── optimizer.py              # 主优化器（编排整个流程）
│   ├── gp_model.py               # GP模型（Region-Lifted GP实现）
│   ├── acquisition.py            # 采集函数（EI优化）
│   ├── scalarization.py          # Tchebycheff标量化
│   ├── region_lifted_gp.py       # Region-Lifted GP核心逻辑
│   ├── rerank.py                 # LLM重排序逻辑
│   └── warmstart_selector.py     # WarmStart候选选择器
│
├── DataBase/
│   └── database.py               # 观测数据库 + Pareto追踪
│
├── pybamm_simulator.py           # PyBaMM电池物理仿真
└── config/                       # 配置管理
```

### 2.2 核心数据流

```
状态字典 (state_dict) ──────────────────────────────────────────────▶
    │
    ├── 迭代信息: iteration, max_iterations
    ├── 权重向量: w_vec ([time_weight, temp_weight, aging_weight])
    ├── 当前最优: theta_best, f_min
    ├── GP状态: mu, sigma
    ├── 停滞计数: stagnation_count
    ├── 历史观测: database
    ├── Pareto信息: pareto_size, current_hv
    └── 不确定性热点: uncertainty_hotspots
```

---

## 3. 现有Prompt设计详解

### 3.1 WarmStart Prompt (`warmstart_prompt.py`)

**触发时机**: 优化开始前，生成初始候选点（替代随机LHS采样）

**三级上下文模式**:

| Level | 模板文件 | 适用场景 |
|-------|----------|----------|
| `none` | `basic.txt` | Baseline对比实验 |
| `partial` | `problem.txt` | 中等信息量的消融实验 |
| `full` | `detailed.txt` | **主推荐配置**，完整领域知识 |

**核心设计要素**:

```python
# 1. 领域元数据注册表 (BATTERY_METADATA_REGISTRY)
#    - 电池型号、化学体系、容量
#    - 专家知识条目（自然语言描述物理规律）

# 2. 权衡方向桶 (DEFAULT_TRADEOFF_BUCKETS)
#    指导LLM覆盖不同的优化策略：
#    - fast_charge: 时间优先
#    - thermal_safe: 温度优先
#    - aging_safe: 老化优先
#    - balanced: 均衡策略
#    - front_loaded_fast: 前段激进+后段保守
#    - high_margin_safe: 高安全余量

# 3. 防坍塌规则 (DEFAULT_ANTI_COLLAPSE_RULES)
#    - 禁止在单个方向上过度聚集
#    - 禁止过度边缘搜索
#    - 禁止返回近似重复的候选点

# 4. Few-Shot示例 (DEFAULT_FEW_SHOT_EXAMPLES)
#    - 正面示例：标注为"balanced", "front_loaded_fast"等
#    - 负面示例：标注为"avoid"，说明为什么不好
```

**示例Prompt结构**（detailed.txt）:

```
You are an expert in lithium-ion battery fast charging optimization.

Task: [TASK_BRIEF]

Battery and charging window:
- Cell: [BATTERY_NAME]
- Parameter set: [PARAM_SET_DISPLAY]
- SOC window: [SOC_START] to [SOC_END]

Decision variables and bounds:
- I1 (stage-1 current): [I1_RANGE]
...

Constraints:
- Practical hard limit: keep dSOC1 + dSOC2 strictly below [DSOC_SUM_MAX]
- Safety margin: keep dSOC1 + dSOC2 <= [SAFE_DSOC_SUM_MAX]
...

Trade-off buckets to cover:
[TRADEOFF_BUCKETS]

[FEW_SHOT_BLOCK]

[NEGATIVE_EXAMPLE_BLOCK]

Generate [NUM_RECOMMENDATION] warm-start protocols...
```

### 3.2 迭代Guidance Prompt (`iteration_prompt.py`)

**触发时机**: 每轮BO迭代，在拟合GP之后、采集函数优化之前

**核心设计目标**: 让LLM基于当前优化状态生成**Point**或**Region**形式的偏好指导

**关键Prompt元素**:

```
Iteration [ITERATION]/[MAX_ITERATIONS]
Weight vector: [WEIGHT_VECTOR]  ← 核心耦合信号
Current focus: [FOCUS_TEXT]      ← 根据w_vec生成的自然语言描述

Current scalarized best value: [F_MIN]
Current best protocol: [BEST_PROTOCOL]
Stagnation count: [STAGNATION_COUNT]  ← 停滞检测，触发探索

High-uncertainty hotspots from the current GP:
[HOTSPOTS_BLOCK]  ← GP不确定性信息

Observed optimization history:
[PARETO_CONTEXT]  ← 历史观测和Pareto前沿

Task:
Return exactly one JSON value in one of these formats:
["region", [[lb1,...], [ub1,...]], confidence]
["point", [I1, I2, I3, dSOC1, dSOC2], confidence]
```

**权重向量解读**（用于生成FOCUS_TEXT）:

```python
if w_time dominant:
    "Prioritize faster charging time while respecting..."
elif w_temp dominant:
    "Prioritize lower peak temperature even if charging time..."
elif w_aging dominant:
    "Prioritize lower aging and gentler late-stage charging."
```

### 3.3 Region Preference Prompt (`region_prompt.py`)

**触发时机**: 启用`enable_region_lifted_gp`时，每轮迭代生成区域偏好

**独特之处**: 使用**JSON结构化Prompt**（非自然语言模板）

```python
payload = {
    "task": "Return exactly one promising raw-coordinate point or region...",
    "rules": [
        "Lower scalarized objective is better under the current weight vector.",
        "Return a promising region only; do not return avoid-only regions.",
        "If no defensible promising region exists, return kind='none'.",
        ...
    ],
    "parameters": [
        {"name": "I1", "lower": 2.0, "upper": 6.0, "unit": "A"},
        ...
    ],
    "current_context": state_dict,  # 完整状态注入
    "output_schema": {
        "kind": "point | region | none",
        "coordinate_space": "raw",
        "preference_direction": "promising",
        "point": {"I1": None, ...},
        "lb": {"I1": None, ...},
        "ub": {"I1": None, ...},
        "confidence": "float in [0,1]",
        "preference_type": "balanced | fast_charge | ...",
        "reason": "short rationale",
        "risk_flags": ["optional strings"],
    }
}
```

### 3.4 候选重排序Prompt (`rerank_prompt.py`)

**触发时机**: 启用`enable_llm_rerank`时，在EI选择后对Top-M候选进行重排序

**输入**: GP评估后的候选池（包含μ, σ, EI等统计信息）

**输出**: 每个候选的`q_good`（质量评分）和`confidence`

---

## 4. Prompt与算法的耦合机制

### 4.1 WarmStart → 初始化

```python
# llm_interface.py:1269-1423
def generate_warmstart_candidates(n, batch_size, max_attempts):
    # 1. 构建Prompt上下文
    context = warmstart_context_builder.build(num_recommendation=n)
    prompt = render_warmstart_prompt("full", context)
    
    # 2. 调用LLM
    responses = llm_caller.call(prompt, temperature=0.7)
    
    # 3. 解析候选点
    candidates = response_parser.parse_candidates(responses)
    
    # 4. Portfolio选择（多目标权衡）
    selected = warmstart_portfolio_selector.select(candidates, config)
    
    # 5. 回退机制（LLM失败时）
    if not candidates:
        candidates = physics_heuristic_fallback.lhs_candidates(n)
    
    return selected
```

### 4.2 Iteration Guidance → GP耦合

```python
# llm_interface.py:1075-1110
def query_iteration_guidance(state_dict):
    # 1. 构建Prompt
    prompt = render_iteration_guidance_prompt(state_dict, param_bounds, pareto_context)
    
    # 2. 调用LLM（低temperature，更确定性）
    responses = llm_caller.call(prompt, temperature=0.4)
    
    # 3. 解析Guidance
    guidance = response_parser.parse_guidance(responses)
    
    # 4. 失败回退到启发式
    if guidance is None:
        guidance = fallback_iteration_guidance(state_dict)
    
    return guidance
```

**Guidance → GP耦合** (`gp_model.py:406-476`):

```python
# 将LLM的区域/点偏好转换为GP的均值偏移
def build_preference_coupling(grid, weights, confidence, ...):
    # 计算后验方差作为Gram矩阵元素
    sigma_gg = self.posterior_covariance(grid, grid)
    posterior_variance = weights @ sigma_gg @ weights
    
    # 计算耦合强度（随迭代衰减）
    base_lambda = confidence / sqrt(posterior_variance)
    annealed_lambda = base_lambda * (decay_rate ** t)
    
    return LLMPreferenceCoupling(
        mode=mode,  # "region" or "point"
        grid=grid,
        weights=weights,
        confidence=confidence,
        lambda_value=annealed_lambda,  # 控制偏移幅度
        ...
    )
```

**GP预测时的均值偏移** (`gp_model.py:357-377`):

```python
def predict_with_coupling(X_new, coupling):
    mean, std = self.predict(X_new)
    if coupling is None:
        return mean, std
    
    # 计算后验协方差
    sigma_xg_z = self.posterior_covariance_standardized(X_new, coupling.grid)
    base_z = sigma_xg_z @ coupling.weights
    
    # 应用门控和局部掩码
    mask = self._coupling_local_mask(X_new, coupling)
    shift_z = coupling.lambda_value * coupling.gate * mask * base_z
    
    # 转换回原始空间并应用偏移
    shift_y = shift_z * y_std
    return mean - shift_y, std  # 均值向LLM偏好方向偏移
```

---

## 5. 当前Prompt设计的评估与改进方向

### 5.1 优势

1. **分层上下文**: WarmStart的三级模板（none/partial/full）支持系统性的消融实验
2. **领域知识注入**: BATTERY_METADATA_REGISTRY和EXPERT_KNOWLEDGE机制清晰
3. **Few-Shot引导**: 正负示例帮助LLM理解期望的输出分布
4. **防坍塌机制**: ANTI_COLLAPSE_RULES和TRADEOFF_BUCKETS有效防止候选聚集
5. **权重向量耦合**: Iteration Prompt通过w_vec实现与标量化目标的强耦合
6. **失败回退**: 所有LLM调用都有物理启发式回退，保证系统鲁棒性

### 5.2 可改进方向（供专家参考）

#### A. WarmStart Prompt

**当前**: 使用自然语言描述权衡方向（"fast_charge", "thermal_safe"等）

**潜在改进**:
- **量化目标引导**: 在Prompt中明确告知LLM不同权衡方向的预期目标值范围
  ```
  "fast_charge方向预期充电时间 < 800s，但峰值温度可能 > 310K"
  ```
- **物理模型简化解释**: 用1-2句话解释为什么I1高→时间短但温度高
- **动态Few-Shot**: 从已有仿真结果中动态选择最接近当前任务的Few-Shot示例

#### B. Iteration Guidance Prompt

**当前**: 提供原始统计信息（mu, sigma, EI等）

**潜在改进**:
- **可视化描述**: 用文字描述GP预测的"地形"（"在I1=5.0附近有一个性能高原"）
- **历史Guidance效果反馈**: 告知LLM上一次的Guidance是否带来了HV提升
- **反事实建议**: 让LLM解释为什么某个区域应该被探索（"如果I2降低到3.0，温度可能降低2-3K"）

#### C. Region vs Point选择策略

**当前**: 由LLM自行决定返回region还是point

**潜在改进**:
- **基于不确定性的建议**: Prompt中明确提示"当前GP在X区域不确定性高，建议返回region"
- **阶段性策略**: 早期迭代偏好region（探索），后期偏好point（利用）

#### D. 多轮对话设计

**当前**: 单次调用，无上下文保持

**潜在改进**:
- **跨迭代上下文**: 让LLM记住之前的Guidance和结果（使用LLM的message history）
- **自我反思**: 要求LLM解释上一次Guidance的效果，并调整策略

#### E. 约束处理

**当前**: 使用自然语言描述约束（"I1 >= I2 >= I3 is recommended"）

**潜在改进**:
- **违反示例**: 提供约束违反的具体例子和后果
- **软约束量化**: 区分硬约束（物理可行）和软约束（性能偏好）

### 5.3 实验设计建议

为验证Prompt设计的效果，建议采用以下实验流程：

```
1. Baseline: LLM-Mock（纯物理启发式回退）
2. V1: Basic WarmStart Prompt
3. V2: Detailed WarmStart Prompt（当前主推荐）
4. V3: Detailed + 动态Few-Shot
5. V4: V3 + 改进的Iteration Guidance
6. V5: V4 + 多轮对话

评估指标:
- 最终Hypervolume
- 达到80%最大HV所需的迭代数
- 初始HV（WarmStart质量）
- LLM调用成功率
- 约束违反率
```

---

## 6. 开发工具与调试

### 6.1 Prompt渲染调试

```python
# warmstart_prompt.py:384-400
if __name__ == "__main__":
    # 测试Prompt渲染
    builder = WarmStartPromptContextBuilder(...)
    ctx = builder.build(num_recommendation=6)
    for level in ("none", "partial", "full"):
        text = render_warmstart_prompt(level, ctx)
        print(f"[{level}] {len(text)} chars")
```

### 6.2 LLM响应解析测试

```python
# llm_interface.py:1659-1674 (main block)
# 包含完整的ResponseParser自测用例
test_responses = [
    '[{"I1":5.0,"I2":4.0,...}, {...}]',  # 合法
    '{"I1":9.0,...}',  # 越界
    '{"I1":5.0,...,"dSOC1":0.45,...}',  # dSOC1越界
    'invalid json',  # 解析失败
]
```

### 6.3 Mock模式开发

无需真实LLM API即可开发Prompt：

```python
llm = build_llm_interface(
    DEFAULT_BOUNDS,
    backend="mock",  # 触发回退机制
    warmstart_context_level="full",
)
```

---

## 7. 关键代码参考

| 功能 | 文件 | 行号范围 |
|------|------|----------|
| WarmStart Prompt构建 | `warmstart_prompt.py` | 238-362 |
| Iteration Prompt构建 | `iteration_prompt.py` | 62-188 |
| Region Prompt构建 | `region_prompt.py` | 9-55 |
| LLM接口主类 | `llm_interface.py` | 793-888 |
| WarmStart生成 | `llm_interface.py` | 1269-1423 |
| Guidance查询 | `llm_interface.py` | 1075-1110 |
| GP耦合构建 | `gp_model.py` | 406-476 |
| Prompt模板目录 | `llm/templates/` | - |

---

## 8. 联系与资源

- **项目根目录**: `D:\Users\aa133\Desktop\BO_Multi_12_20\New_LLMBO`
- **配置入口**: `main.py` --generate-template
- **文档**: `CLAUDE.md`（项目级设计决策）

---

**欢迎提出任何关于Prompt设计的问题或改进建议！**
