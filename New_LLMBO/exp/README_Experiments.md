# LLMBO-MO 对比实验和消融实验指南

本目录包含 LLMBO-MO 框架的完整实验工具链，用于复现论文中的对比实验和消融实验结果。

---

## 文件结构

```
New_LLMBO/
├── experiment_configs.py      # 实验配置（消融变体、Baseline、参数）
├── baselines.py               # Baseline 方法实现（5 种）
├── experiment_runner.py       # 实验执行器（支持断点续跑）
├── analyze_results.py         # 统计分析（Wilcoxon 检验、表格生成）
├── plot_comparison.py         # 可视化（HV 曲线、Pareto Front、箱线图）
└── run_all_experiments.py     # 主脚本（一键运行全部流程）
```

---

## 实验配置

### 参数边界（与 `pybamm_simulator.py` 一致）

| 参数 | 下界 | 上界 | 单位 |
|------|------|------|------|
| I1   | 3.0  | 7.0  | A    |
| SOC1 | 0.10 | 0.70 | -    |
| I2   | 1.0  | 5.0  | A    |

### 实验预算

- `n_warmstart = 5`（Warmstart 候选数）
- `n_random_init = 10`（随机初始化数）
- `n_iterations = 50`（迭代次数）
- **总评估次数 = 65**

### 参考点（HV 计算）

```python
REFERENCE_POINT = {"time": 5400.0, "temp": 318.0, "aging": 0.1}
```

### 随机种子

默认 5 个种子：`[42, 123, 456, 789, 1024]`

---

## 消融变体（V0-V6）

| 变体 | Warmstart | Coupling Matrix | Stagnation Fix | Wcharge Fix | LLM Sampling |
|------|-----------|-----------------|----------------|--------------|--------------|
| **V0_Full** | ✓ | ✓ | ✓ | ✓ | ✓ |
| V1_NoWarmStart | ✗ | ✓ | ✓ | ✓ | ✓ |
| V2_NoCoupling | ✓ | ✗ | ✓ | ✓ | ✓ |
| V3_NoStagnationFix | ✓ | ✓ | ✗ | ✓ | ✓ |
| V4_NoWchargeFix | ✓ | ✓ | ✓ | ✗ | ✓ |
| V5_NoLLM | ✗ | ✗ | ✓ | ✓ | ✗ |
| V6_VanillaBO | ✗ | ✗ | ✗ | ✗ | ✗ |

---

## Baseline 方法

1. **RandomSearch** - 随机搜索
2. **SobolGP** - Sobol 初始化 + GP-EI（固定权重）
3. **ParEGO** - 随机权重 Tchebycheff + GP-EI
4. **NSGA2** - NSGA-II 进化算法（via pymoo）
5. **MOEAD** - MOEA/D 分解算法（via pymoo）

---

## 使用方法

### 1. 运行 Baseline 对比实验

```bash
# 运行所有 Baseline（5 个种子）
python run_all_experiments.py --mode baseline --seeds 5

# 运行指定 Baseline
python run_all_experiments.py --mode baseline --methods ParEGO NSGA2 --seeds 3
```

### 2. 运行消融实验

```bash
# 运行所有消融变体
python run_all_experiments.py --mode ablation --seeds 5

# 运行指定变体
python run_all_experiments.py --mode ablation --methods V0_Full V1_NoWarmStart V6_VanillaBO
```

### 3. 运行全部实验

```bash
python run_all_experiments.py --mode all --seeds 5
```

### 4. 分析结果

```bash
# 生成 Markdown 格式分析表格
python analyze_results.py --results-dir ./results --output markdown --save results/analysis.md

# 生成 LaTeX 格式表格
python analyze_results.py --results-dir ./results --output latex --save results/analysis.tex
```

### 5. 可视化

```bash
# HV 收敛曲线
python plot_comparison.py --results-dir ./results --type hv --save results/hv_comparison.png

# Pareto Front 对比
python plot_comparison.py --results-dir ./results --type pareto --save results/pareto_comparison.png

# 箱线图
python plot_comparison.py --results-dir ./results --type box --save results/box_comparison.png
```

### 6. 一键完整流程

```bash
# 实验 + 分析 + 可视化
python run_all_experiments.py --mode all --seeds 5 --analyze --visualize
```

---

## 输出结构

```
results/
├── V0_Full/
│   ├── seed_42.json
│   ├── seed_123.json
│   └── ...
├── V1_NoWarmStart/
├── RandomSearch/
├── ParEGO/
├── NSGA2/
├── MOEAD/
├── analysis_results.md        # 分析表格
├── hv_comparison.png          # HV 收敛曲线
├── pareto_comparison.png      # Pareto Front 对比
└── box_comparison.png         # 箱线图
```

---

## 结果文件格式

每个 `seed_*.json` 包含：

```json
{
  "evaluations": [...],
  "hv_history": [0.01, 0.05, ...],
  "pareto_front": [...],
  "wall_time": 123.45,
  "n_feasible": 60,
  "n_violations": 5,
  "_meta": {
    "method": "V0_Full",
    "seed": 42,
    "timestamp": "2026-03-11T..."
  }
}
```

---

## 依赖

```bash
# 核心依赖
pip install numpy scipy matplotlib

# Baseline 需要
pip install pymoo scikit-learn

# 电池仿真需要
pip install pybamm
```

---

## 统计检验

分析脚本自动执行 **Wilcoxon 符号秩检验**（配对样本），比较各方法与基准方法（V0_Full）的最终 HV 值差异：

- **原假设 H0**: 两种方法的中位数无显著差异
- **备择假设 H1**: 两种方法的中位数有显著差异
- **显著性水平**: α = 0.05

输出表格中：
- ✓ 表示 p < 0.05（差异显著）
- ✗ 表示 p ≥ 0.05（差异不显著）

---

## 常见问题

### Q: 如何修改实验预算？

编辑 `experiment_configs.py` 中的 `DEFAULT_BUDGET` 字典。

### Q: 如何添加新的 Baseline？

1. 在 `baselines.py` 中实现 `run_xxx()` 函数
2. 在 `BASELINE_RUNNERS` 字典中注册
3. 在 `BASELINE_CONFIGS` 中添加配置

### Q: 如何自定义参数边界？

编辑 `experiment_configs.py` 中的 `PARAM_BOUNDS` 字典（需与 `pybamm_simulator.py` 一致）。

---

## 参考

- 主优化器：`optimizer.py`
- 电池仿真：`pybamm_simulator.py`
- 采集函数：`acquisition.py`
- GP 模型：`gp_model.py`
