# LLMBO-MO 多目标贝叶斯优化框架学习大纲

## 项目概述

**LLMBO-MO** (LLM-Assisted Multi-Objective Bayesian Optimization) 是一个用于锂电池快充协议优化的多目标贝叶斯优化框架。该项目结合了大语言模型 (LLM) 的物理洞察力和贝叶斯优化的数学严谨性，实现了高效的三目标优化。

### 应用场景

- **优化对象**: 两阶段恒流 (CC-CC) 充电协议
- **决策变量**:
  - I₁ ∈ [3, 7] A — 第一阶段恒流电流
  - SOC₁ ∈ [0.10, 0.70] — 阶段切换 SOC
  - I₂ ∈ [1, 5] A — 第二阶段恒流电流
- **优化目标** (均最小化):
  1. 充电时间 t_charge [s]
  2. 峰值温度 T_peak [K]
  3. 老化程度 L_aging [%]

---

## 第一部分：核心概念基础

### 1.1 贝叶斯优化基础

**学习要点**:
- 代理模型 (Surrogate Model) 的作用
- 采集函数 (Acquisition Function) 的设计原理
- 探索 - 开发权衡 (Exploration-Exploitation Tradeoff)

**关键公式**:
```
Expected Improvement (EI):
EI(θ) = (f_min - f̂(θ))·Φ(z) + s(θ)·φ(z)
其中 z = (f_min - f̂(θ)) / s(θ)
```

**推荐学习资源**:
- 《Gaussian Processes for Machine Learning》Chapter 1-3
- A Tutorial on Bayesian Optimization (Frazier, 2018)

### 1.2 多目标优化基础

**学习要点**:
- Pareto 前沿概念
- Tchebycheff 标量化方法
- 超体积 (Hypervolume) 指标

**关键公式**:
```
Tchebycheff 标量化 (Eq.1):
f_tch = max_i(wᵢ·f̄ᵢ) + η·Σᵢ(wᵢ·f̄ᵢ)
其中 η = 0.05 为 tiebreaker 权重
```

### 1.3 高斯过程回归

**学习要点**:
- RBF 核函数
- Universal Kriging 预测
- Cholesky 分解与数值稳定性

**关键公式**:
```
RBF 核：k(θ,θ') = exp(-‖θ-θ'‖²/(2l²))
GP 预测均值：f̂(θ) = μ̂ + cᵀC⁻¹(F-1μ̂)
GP 预测方差：s²(θ) = σ̂²[1 - cᵀC⁻¹c + (1-1ᵀC⁻¹c)²/(1ᵀC⁻¹1)]
```

---

## 第二部分：框架架构总览

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LLMBO-MO 优化循环                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │  LLM 接口    │────▶│  GP 代理模型  │────▶│  采集函数    │        │
│  │  (Touchpoint)│     │  (复合核)    │     │  (EI×W)     │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   ObservationDB (观测数据库)                  │   │
│  │  - 存储所有评估结果                                          │   │
│  │  - 维护 Pareto 前沿                                          │   │
│  │  - 计算超体积                                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                    ┌─────────────────┐                             │
│                    │  PyBaMM 仿真器   │                             │
│                    │  (锂电池模型)    │                             │
│                    └─────────────────┘                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
optimizer.py (主编排)
    │
    ├──► database.py (观测数据库)
    │       └── ObservationDB, Observation
    │
    ├──► pybamm_simulator.py (仿真器)
    │       └── PyBaMMSimulator
    │
    ├──► llm_interface.py (LLM 接口)
    │       ├── LLMInterface
    │       ├── LLMConfig, LLMCaller
    │       ├── TemplateEngine, ResponseParser
    │       └── PhysicsHeuristicFallback
    │
    ├──► gp_model.py (高斯过程)
    │       ├── PhysicsGPModel
    │       ├── PsiFunction (物理代理函数)
    │       ├── CouplingMatrixManager (耦合矩阵)
    │       └── GammaAnnealer (γ退火)
    │
    └── acquisition.py (采集函数)
            ├── AcquisitionFunction
            ├── SearchMuTracker (μ追踪器)
            ├── SearchSigmaTracker (σ追踪器)
            ├── EICalculator
            └── WChargeCalculator
```

---

## 第三部分：核心模块详解

### 3.1 database.py — 观测数据库

**核心类**:
- `Observation`: 单条观测记录数据类
- `ObservationDB`: 主数据库类

**关键方法**:
| 方法 | 功能 | 对应论文 |
|------|------|----------|
| `add_observation()` | 添加新观测 | - |
| `get_train_XY()` | 获取 GP 训练数据 | - |
| `get_pareto_front()` | 获取 Pareto 前沿 | - |
| `compute_hypervolume()` | 计算归一化超体积 | - |
| `update_tchebycheff_context()` | 更新 Tchebycheff 上下文 | Eq.1-2 |
| `to_llm_context()` | 生成 LLM 上下文 | - |

**核心算法流程**:
```python
# Tchebycheff 标量化计算流程 (Eq.1-2)
# Step 1: 原始目标 → log 变换 (Eq.2a)
Y_tilde = Y_raw.copy()
Y_tilde[:, 2] = log10(aging)  # 仅 aging 做 log 变换

# Step 2: 动态 min-max 归一化 (Eq.2b)
Y_bar = (Y_tilde - y_min) / (y_max - y_min)

# Step 3: Tchebycheff + η tiebreaker (Eq.1)
f_tch = max(w * Y_bar) + η * sum(w * Y_bar)
```

**学习重点**:
1. 理解 Tchebycheff 标量化的数学原理
2. 掌握动态归一化边界的作用
3. 了解超体积计算的 3D 增量算法

### 3.2 pybamm_simulator.py — PyBaMM 仿真器

**核心类**: `PyBaMMSimulator`

**决策变量映射**:
```
θ = [I₁, SOC₁, I₂]
  │       │       └─ 第二阶段恒流电流
  │       └─ 阶段切换 SOC
  └─ 第一阶段恒流电流
```

**充电协议流程**:
```
CC₁ ──────────→ CC₂ ──────────→ 结束
 SOC₀         SOC₁           SOC_end
 I₁ (大电流)   I₂ (小电流)
```

**三目标提取**:
```python
f₁ = 充电时间 [s]    # minimize
f₂ = 峰值温度 [K]    # minimize
f₃ = 老化程度 [%]   # minimize (SEI + 锂沉积)
```

**约束条件**:
- 电压约束：V ≤ 4.3 V
- 温度约束：T ≤ 328.15 K (55°C)
- 违规惩罚：返回固定惩罚值 (7200s, 338K, 0.5%)

**学习重点**:
1. PyBaMM SPMe 模型的基本原理
2. 两阶段恒流充电的物理过程
3. SEI 膜生长和锂沉积的老化机理

### 3.3 gp_model.py — 物理信息复合核 GP

**核心组件**:

| 类 | 功能 | 对应公式 |
|----|------|----------|
| `PsiFunction` | 欧姆热代理函数 Ψ(θ) | Eq.5-7 |
| `CouplingMatrixManager` | 耦合矩阵管理 | Eq.11' |
| `GammaAnnealer` | γ退火调度 | - |
| `PhysicsGPModel` | 复合核 GP 预测 | Eq.11-13 |

**物理代理函数 Ψ(θ)**:
```
Ψ(θ) = I₁²·R̄₁·t₁ + I₂²·R̄₂·t₂   (欧姆热代理，Eq.6)

其中：
  t₁ = (SOC₁ - SOC₀)·Q_nom / I₁
  t₂ = (SOC_end - SOC₁)·Q_nom / I₂

化简后：
  Ψ(θ) = Q_nom · [I₁·R̄₁·(SOC₁-SOC₀) + I₂·R̄₂·(SOC_end-SOC₁)]
```

**梯度计算 (Eq.8-10)**:
```
∂Ψ/∂I₁   = R̄₁·(SOC₁-SOC₀)·Q_nom
∂Ψ/∂SOC₁ = Q_nom·[I₁(R̄₁+(SOC₁-SOC₀)·dR̄₁/dSOC₁) - I₂(R̄₂-(SOC_end-SOC₁)·dR̄₂/dSOC₁)]
∂Ψ/∂I₂   = R̄₂·(SOC_end-SOC₁)·Q_nom
```

**复合核函数 (Eq.11)**:
```
k^(t)(θ,θ') = RBF(θ̃,θ̃') + γ · ∇Ψ(θ)ᵀ W^(t) ∇Ψ(θ')

其中：
  - θ̃ = min-max 归一化到 [0,1]
  - W^(t) = Σ wᵢ Wᵢ / Σ wᵢ  (Eq.11'，权重合成)
  - γ(t) = (γ_max-γ_min)·exp(-t/t_decay) + γ_min  (退火)
```

**GP 预测 (Universal Kriging)**:
```python
# 均值估计
μ̂ = (1ᵀC⁻¹1)⁻¹ (1ᵀC⁻¹F)

# 方差估计
σ̂² = (F-1μ̂)ᵀ C⁻¹ (F-1μ̂) / n

# 后验预测 (Eq.12)
f̂(θ) = μ̂ + cᵀ C⁻¹ (F - 1μ̂)

# 后验方差 (Eq.13)
s²(θ) = σ̂² [1 - cᵀC⁻¹c + (1-1ᵀC⁻¹c)²/(1ᵀC⁻¹1)]
```

**学习重点**:
1. 理解物理代理函数的设计动机
2. 掌握复合核函数的数学推导
3. 了解 Universal Kriging 与普通 Kriging 的区别

### 3.4 llm_interface.py — LLM 接口

**三个 Touchpoint**:

| Touchpoint | 方法 | 功能 | 输出 |
|------------|------|------|------|
| **1a** | `generate_coupling_matrices()` | 生成耦合矩阵 | W_time, W_temp, W_aging |
| **1b** | `generate_warmstart_candidates(n)` | 生成初始候选 | N_ws 个充电协议 |
| **2** | `generate_iteration_candidates(n, state)` | 迭代生成候选 | m 个新候选点 |

**核心组件**:

| 类 | 功能 |
|----|------|
| `LLMConfig` | LLM 调用配置 |
| `TemplateEngine` | Prompt 模板引擎 |
| `LLMCaller` | 多后端 LLM API 调用 |
| `ResponseParser` | JSON 响应解析验证 |
| `PhysicsHeuristicFallback` | 物理启发式回退 |

**模板引擎工作原理**:
```python
# 模板文件使用 [PLACEHOLDER] 占位符
# 运行时动态替换

kwargs = {
    "BATTERY_MODEL": "LG M50",
    "I1_RANGE": "3.0 - 7.0",
    "NUM_CANDIDATES": "10",
}

prompt = engine.render("warmstart_candidates", **kwargs)
```

**响应解析流程**:
```
LLM 响应 → JSON 提取 → 结构验证 → 边界检查 → 有效候选点
   │           │           │           │
   ▼           ▼           ▼           ▼
原始文本    提取对象    检查键名    检查数值范围
```

**LLM 后端支持**:
- **Ollama**: 本地部署 (Qwen2.5 等)
- **OpenAI**: GPT-4/GPT-3.5
- **Anthropic**: Claude 系列
- **Mock**: 测试用（返回物理启发式默认值）

**学习重点**:
1. 理解 Prompt 工程的基本原则
2. 掌握 JSON 响应的容错解析技巧
3. 了解物理启发式回退的设计思想

### 3.5 acquisition.py — 采集函数

**核心组件**:

| 类 | 功能 | 对应公式 |
|----|------|----------|
| `SearchMuTracker` | μ动态漂移追踪 | Eq.18-19 |
| `SearchSigmaTracker` | σ敏感度引导 | Eq.20-22 |
| `EICalculator` | EI 计算 | Eq.15-16 |
| `WChargeCalculator` | 物理加权 | Eq.17 |
| `AcquisitionScorer` | 综合评分 | Eq.14 |

**采集函数设计**:
```
α(θ) = EI(θ) × W_charge(θ)   (Eq.14)

其中：
  EI(θ)      = (f_min - f̂(θ))·Φ(z) + s(θ)·φ(z)  (期望改进)
  W_charge(θ) = Πⱼ N(θⱼ; μⱼ, σⱼ²)               (物理加权)
```

**μ动态漂移 (Eq.18-19)**:
```python
# 信任度衰减
α_t = α_max · exp(-t / t_decay_α) + α_min

# μ更新规则
μⱼ^(t+1) = α_t · μⱼ^(t) + (1 - α_t) · θⱼ^best

# 物理含义:
# - 早期 (t≈0): α_t≈0.75, μ缓慢漂移，LLM 先验主导
# - 晚期 (t→∞): α_t≈0.05, μ快速跟踪θ_best，数据主导
```

**σ敏感度引导 (Eq.20-22)**:
```python
# 灵敏度缩放
c = κ · maxⱼ |∂Ψ/∂θⱼ|_{θ_best}         (Eq.21)

# 搜索范围
σⱼ = c / (|∂Ψ/∂θⱼ| + ε_σ)              (Eq.20)

# 停滞扩张
σⱼ^(t+1) = σⱼ^(t) · (1 + ρ)  if stagnated  (Eq.22)

# 物理含义:
# - |∂Ψ/∂θⱼ|大 → Ψ对θⱼ敏感 → σⱼ小 → 细粒度搜索
# - |∂Ψ/∂θⱼ|小 → Ψ对θⱼ不敏感 → σⱼ大 → 粗粒度搜索
# - 停滞时 σ 扩张 10%，鼓励探索新区域
```

**完整优化循环**:
```python
# Algorithm 步骤 26-29
for t in range(max_iterations):
    # 步骤 1: 选 w_vec (Riesz 集合)
    w_vec = weight_set[random_index]

    # 步骤 2: 更新 Tchebycheff 上下文
    database.update_tchebycheff_context(w_vec, y_min, y_max, eta)

    # 步骤 3: 训练 GP
    gp.fit(X_train, F_tch, w_vec, t)

    # 步骤 4: LLM 生成候选点 (Touchpoint 2)
    X_candidates = llm.generate_iteration_candidates(n=15, state_dict)

    # 步骤 5: 计算 α并选 top-k
    result = af.step(X_candidates, database, t, w_vec)

    # 步骤 6: PyBaMM 评估
    for theta in result.selected_thetas:
        sim_result = simulator.evaluate(theta)
        database.add_from_simulator(theta, sim_result)
```

**学习重点**:
1. 理解 EI×W 乘积形式的设计动机
2. 掌握 μ/σ动态追踪的物理意义
3. 了解停滞检测与扩张机制

---

## 第四部分：优化器主循环

### 4.1 optimizer.py — 主优化器

**核心类**: `BayesOptimizer`

**完整运行流程**:
```
setup()                    # §1 初始化所有组件
    │
    ▼
run_warmstart()            # §2 LLM Touchpoint 1b + PyBaMM 评估
    │
    ▼
initialize_acquisition()   # §3 采集函数初始化 (Algorithm 步骤 5)
    │
    ▼
run_optimization_loop()    # §4 主循环 (Algorithm 步骤 25-35)
    │
    ▼
save_results()             # 保存结果
```

**Riesz s-energy 权重集合生成**:
```python
# 算法流程:
# 1. Das-Dennis 均匀初始化 (n_div=10 → 66 个点)
# 2. Riesz s-energy 梯度下降 (最小化Σ‖wᵢ-wⱼ‖⁻ˢ)
# 3. 投影回单纯形 + 最小分量保护

W = generate_riesz_weight_set(
    n_obj=3, n_div=10, s=2.0, n_iter=500, lr=5e-3
)
# 输出：W.shape = (66, 3), 每行Σ=1
```

**目标变换与归一化**:
```python
# Step 1: log 变换 (Eq.2a)
Y_tilde = log_transform_objectives(Y_raw)
Y_tilde[:, 2] = log10(aging)

# Step 2: 动态边界 (Eq.2b 分母)
y_min, y_max = compute_dynamic_bounds(Y_tilde)

# Step 3: Min-max 归一化
Y_bar = (Y_tilde - y_min) / (y_max - y_min)

# Step 4: Tchebycheff 标量化 (Eq.1)
f_tch = max(w * Y_bar) + η * sum(w * Y_bar)
```

**GP 训练数据准备**:
```python
# 归一化 F_tch 用于 GP 训练 (zero-mean, unit-var)
f_mean = F_tch.mean()
f_std  = F_tch.std() + 1e-8
F_tch_norm = (F_tch - f_mean) / f_std

# GP 在归一化空间训练
gp.fit(X_train_norm, F_tch_norm, w_vec, t)
```

**学习重点**:
1. 理解 Riesz 权重集合的生成原理
2. 掌握动态归一化与 GP 训练的关系
3. 了解检查点保存机制

---

## 第五部分：实战练习

### 5.1 环境配置

```bash
# 必需依赖
pip install numpy scipy pybamm openai anthropic

# PyBaMM 需要 Python 3.9+
python --version  # 推荐 3.10+
```

### 5.2 快速开始

```python
from optimizer import BayesOptimizer

# 最简运行
opt = BayesOptimizer()
opt.run()
opt.save_results("results/")

# 自定义配置
cfg = {
    "llm_backend": "openai",
    "llm_model": "gpt-4o",
    "max_iterations": 30,
    "n_warmstart": 10,
    "n_candidates": 15,
}
opt = BayesOptimizer(config=cfg)
db = opt.run()
```

### 5.3 命令行运行

```bash
cd New_LLMBO

# 默认配置运行
python optimizer.py

# 自定义参数
python optimizer.py \
    --backend openai \
    --model gpt-4o \
    --iters 50 \
    --warmstart 10 \
    --candidates 15 \
    --output results/
```

### 5.4 调试技巧

**单模块测试**:
```bash
# 测试 database
python database.py

# 测试 gp_model
python gp_model.py

# 测试 acquisition
python acquisition.py

# 测试 llm_interface
python llm_interface.py

# 测试 pybamm_simulator
python pybamm_simulator.py
```

**日志级别调整**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)  # 或 INFO/WARNING/ERROR
```

### 5.5 消融实验建议

| 实验名称 | 修改内容 | 预期效果 |
|----------|----------|----------|
| 无 LLM 耦合矩阵 | 使用单位矩阵 | 验证 LLM 物理洞察价值 |
| 无复合核 | γ=0，只用 RBF | 验证物理核贡献 |
| 无 μ追踪 | 固定μ=θ_best | 验证漂移机制作用 |
| 无 σ扩张 | ρ=0 | 验证停滞恢复能力 |
| 无 Riesz 集合 | Dirichlet 采样 | 验证权重分布影响 |

---

## 第六部分：进阶主题

### 6.1 数学推导深入

**Tchebycheff 标量化性质**:
- 定理：对任意 Pareto 最优解，存在权重向量使之为 Tchebycheff 最优
- 推论：通过遍历权重集合可逼近完整 Pareto 前沿

**复合核正定性证明**:
- RBF 核正定 (Mercer 定理)
- 物理耦合项 γ·∇ΨᵀW∇Ψ在 W为 PSD 时半正定
- 正定 + 半正定 = 正定

### 6.2 数值稳定性技巧

**Cholesky 分解保护**:
```python
# 添加 jitter 保证正定性
C += (obs_noise + nugget) * np.eye(n)
# obs_noise = 1e-4, nugget = 1e-6
```

**log 空间计算**:
```python
# W_charge 连乘易下溢，转 log 空间
log_W = Σⱼ [-½log(2π) - log(σⱼ) - (θⱼ-μⱼ)²/(2σⱼ²)]
W = exp(log_W - log_W.max())  # shift 防溢出
```

### 6.3 扩展方向

**新目标函数**:
- 添加能量效率目标
- 考虑循环寿命预测
- 引入充电成本

**新代理模型**:
- 神经网络代理
- 集成高斯过程
- 深度核学习

**新采集策略**:
- q-EI 批量采集
- 多保真度采集
- 约束感知采集

---

## 第七部分：参考资源

### 7.1 核心论文

1. **LLAMBO 原始论文**:
   - Langer, M., et al. (2024). "LLAMBO: Large Language Model-Assisted Multi-Objective Bayesian Optimization"

2. **贝叶斯优化综述**:
   - Frazier, P. I. (2018). "A Tutorial on Bayesian Optimization"

3. **高斯过程经典**:
   - Rasmussen, C. E., & Williams, C. K. I. (2006). "Gaussian Processes for Machine Learning"

4. **多目标优化**:
   - Deb, K. (2001). "Multi-Objective Optimization Using Evolutionary Algorithms"

### 7.2 PyBaMM 资源

- **官方文档**: https://docs.pybamm.org/
- **教程视频**: PyBaMM YouTube 频道
- **模型论文**: Sulzer, V., et al. (2021). "Python Battery Mathematical Modelling"

### 7.3 LLM API 文档

- **OpenAI API**: https://platform.openai.com/docs/
- **Anthropic API**: https://docs.anthropic.com/
- **Ollama**: https://ollama.ai/

---

## 第八部分：学习检查清单

### 基础概念
- [ ] 理解贝叶斯优化的基本流程
- [ ] 掌握高斯过程回归原理
- [ ] 了解多目标 Pareto 前沿概念
- [ ] 熟悉 Tchebycheff 标量化方法

### 代码理解
- [ ] 能解释 database.py 的 Tchebycheff 计算流程
- [ ] 理解 gp_model.py 的复合核函数实现
- [ ] 掌握 acquisition.py 的 μ/σ追踪机制
- [ ] 了解 llm_interface.py 的 Prompt 模板系统

### 实践能力
- [ ] 能独立运行优化器
- [ ] 能修改配置进行实验
- [ ] 能解读优化结果和日志
- [ ] 能进行简单的消融实验

### 进阶能力
- [ ] 能推导复合核函数的梯度
- [ ] 能扩展新的采集函数
- [ ] 能集成新的 LLM 后端
- [ ] 能优化数值稳定性

---

## 附录：关键公式速查

### Eq.1: Tchebycheff 标量化
```
f_tch = max_i(wᵢ·f̄ᵢ) + η·Σᵢ(wᵢ·f̄ᵢ)
```

### Eq.2a: Log 变换
```
f̃₃ = log₁₀(aging)
```

### Eq.2b: 动态归一化
```
f̄ᵢ = (f̃ᵢ - y_min_i) / (y_max_i - y_min_i)
```

### Eq.5-7: 物理代理函数
```
Ψ(θ) = I₁²·R̄₁·t₁ + I₂²·R̄₂·t₂
t₁ = (SOC₁-SOC₀)·Q_nom/I₁
t₂ = (SOC_end-SOC₁)·Q_nom/I₂
```

### Eq.11: 复合核函数
```
k^(t)(θ,θ') = RBF(θ̃,θ̃') + γ·∇Ψ(θ)ᵀW^(t)∇Ψ(θ')
```

### Eq.14: 采集函数
```
α(θ) = EI(θ) × W_charge(θ)
```

### Eq.15-16: 期望改进
```
EI(θ) = (f_min-f̂(θ))·Φ(z) + s(θ)·φ(z)
z = (f_min-f̂(θ))/s(θ)
```

### Eq.17: 物理加权
```
W_charge(θ) = Πⱼ N(θⱼ; μⱼ, σⱼ²)
```

### Eq.18-19: μ漂移
```
α_t = α_max·exp(-t/t_decay_α) + α_min
μⱼ^(t+1) = α_t·μⱼ^(t) + (1-α_t)·θⱼ^best
```

### Eq.20-22: σ引导
```
c = κ·maxⱼ|∂Ψ/∂θⱼ|
σⱼ = c/(|∂Ψ/∂θⱼ|+ε_σ)
σⱼ^(t+1) = σⱼ^(t)·(1+ρ)  [if stagnated]
```

---

*文档版本：v1.0*
*创建日期：2026-03-10*
*基于代码版本：New_LLMBO*
