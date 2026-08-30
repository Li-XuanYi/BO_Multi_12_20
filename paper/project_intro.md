# LLAMBO-MO 项目简介

## 一句话概述

LLAMBO-MO 是一个**大语言模型增强的多目标贝叶斯优化框架**，用于锂电池快充协议设计——在有限的仿真预算内，同时最小化充电时间、峰值温升和容量衰减。

---

## 问题背景

锂电池快充协议设计本质上是一个**受约束多目标优化问题 (CMOP)**：

- 三个冲突目标：充电时间 $t_c$、峰值温升 $\Delta T_p$、容量衰减 $Q_s$
- 5 个决策变量：三阶段恒流电流 $I_1, I_2, I_3$ + 两段 SOC 宽度 $\Delta s_1, \Delta s_2$
- 约束：$\Delta s_1 + \Delta s_2 \leq 0.70$，以及电压/温度/SOC 运行约束
- 仿真代价高：基于 PyBaMM 的 SPMe 电化学-热-老化耦合模型，单次评估数秒

传统 BO 方法（如 ParEGO）仅依赖数值数据构建代理模型，**忽略了电池文献中丰富的领域知识**。

---

## 核心创新：LLM 双触点架构

LLAMBO-MO 的关键洞察：**LLM 不替代代理模型，而是在关键节点补充领域知识**。

### Touchpoint 1 — 初始化阶段：LLM 生成物理信息引导的 warmstart 候选

- 向 LLM 提供电池规格、变量边界、目标含义，请求生成覆盖不同 trade-off 区域的候选协议
- 替代随机/LHS 初始化，为 GP 提供高质量初始数据
- 失败时静默回退到物理启发式候选

### Touchpoint 2 — 迭代阶段：LLM 提供搜索引导 + GP-LLM 耦合

- 每次迭代向 LLM 提供当前权重向量、最优协议、GP 状态、不确定性热点、历史上下文
- LLM 返回：**点模式**（一个推荐协议 + 置信度）或**区域模式**（一个待探索超矩形 + 置信度）
- 通过**有界采集函数均值偏移**机制将 LLM 引导耦合到 GP 后验中：
  - 仅偏移后验均值，保持方差不变
  - 偏移强度随迭代衰减（$\rho = 0.75$），随 GP 确定性自适应调节
  - 空间掩码将偏移局部化到 LLM 指定区域

---

## 技术细节要点

| 组件 | 方案 |
|------|------|
| 多目标分解 | 增广 Tchebycheff 标量化 + Riesz s-energy 权重向量（66个，2-simplex 上均匀分布） |
| 代理模型 | Matérn 5/2 ARD 核 GP，每轮重新拟合 |
| 目标变换 | $t_c, Q_s$ 取 log10；$\Delta T_p$ 线性；动态 min-max 归一化 |
| 采集函数 | EI + LLM 先验奖励 - 约束风险惩罚；候选池融合多源（LLM、不确定性热点、L-BFGS-B、随机） |
| 停滞检测 | 滑动窗口监测 HV 改善；停滞时扩大采集 $\sigma$ 鼓励探索 |
| LLM 后端 | GPT-4.1-mini（可替换）；warmstart 温度 0.7，迭代引导温度 ≤ 0.4 |

---

## 实验结果摘要

**数据集**：LG INR21700-M50 (Chen2020) + Ecker 18650 (Ecker2015)

**评估预算**：56 次仿真评估（6 初始化 + 50 BO 迭代），5 个随机种子

### Chen2020 数据集 — 归一化超体积 (HV)

| 算法 | Mean HV | Std HV |
|------|:-------:|:------:|
| **LLAMBO-MO** | **0.3872** | 0.0142 |
| ParEGO | 0.3763 | 0.0041 |
| NSGA-II | 0.3273 | 0.0216 |
| DISK | 0.3091 | 0.0274 |
| PIMD | 0.2982 | 0.0128 |

### Ecker2015 数据集

LLAMBO-MO HV = 1.8684 vs ParEGO 1.5866（**+17.8%**），标准差仅 0.0024。

### 消融实验关键发现

- **Warmstart 贡献最大**：V0(无LLM) → V1(仅warmstart)，HV +1.4%，标准差从 0.0054 降至 0.0019
- 迭代引导在特定 seed 和跨数据集时优势更明显

### 计算开销

总运行约 440s（vs ParEGO 252s），其中 LLM API 约 30-60s，GP-LLM 耦合约 5-15s/iter。额外开销 < 40%，在可接受范围内。

---

## 代码结构

```
main.py                  # CLI 入口
config/
  schema.py              # Pydantic 配置 (11 classes)
  load.py                # JSON → env → CLI 优先级加载
llmbo/
  optimizer.py           # 优化主循环 BayesOptimizer.run()
  gp_model.py            # 物理信息 GP 核
  acquisition.py         # EI × W_charge 采集函数
  ParEGO.py              # Riesz s-energy 权重生成
llm/
  llm_interface.py       # 两个 LLM 调用函数 + LHS fallback
DataBase/
  database.py            # 观测存储、HV 计算、Pareto 前沿追踪
pybamm_simulator.py      # SPMe 仿真接口
exp/                     # 消融 (V0-V6) 和基线实验
plot/                    # Pareto 前沿、HV 曲线可视化
```

## 快速运行

```bash
# Demo（5 次迭代，无需 LLM API）
pixi run python main.py --demo

# 完整运行
pixi run python main.py --config config.json --verbose
```
