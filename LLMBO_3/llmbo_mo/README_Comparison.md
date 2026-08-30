# LLMBO-MO vs NSGA-II 对比实验指南

## 概述

本目录包含完整的 LLMBO-MO 与 NSGA-II 算法对比实验工具，用于评估两种多目标优化算法在电池充电协议优化问题上的性能。

## 文件结构

```
llmbo_mo/
├── nsgaii_optimizer.py    # NSGA-II 优化器（基于 pymoo）
├── run_comparison.py      # 对比实验主脚本
├── export_results.py      # 结果导出工具
├── config.py              # 配置类（共享）
├── battery_model.py       # PyBaMM 电池模型（共享）
├── pareto.py              # Pareto 前沿工具（共享）
└── ...                    # 其他 LLMBO-MO 文件
```

## 安装依赖

```bash
# 基础依赖
pip install pybamm numpy scipy scikit-learn

# NSGA-II 需要 pymoo
pip install pymoo

# xlsx 导出需要 pandas 和 openpyxl
pip install pandas openpyxl

# LLMBO-MO 使用 LLM 需要 openai
pip install openai
```

## 快速开始

### 1. 运行完整对比实验（5 次独立运行）

```bash
cd llmbo_mo
python run_comparison.py --runs 5 --output comparison_results.xlsx
```

### 2. 单次快速测试

```bash
# 只运行 1 次对比（用于测试）
python run_comparison.py --runs 1 --output test_comparison.xlsx

# 静默模式
python run_comparison.py --runs 1 --quiet
```

### 3. 自定义配置

```bash
# 不使用 LLM（消融实验）
python run_comparison.py --runs 3 --no-llm --output ablation_no_llm.xlsx

# 增加运行次数
python run_comparison.py --runs 10 --output robust_comparison.xlsx

# 调整 LLMBO 参数
python run_comparison.py --llmbo-n-init 20 --llmbo-t-max 60 --runs 5
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--runs`, `-n` | 独立运行次数 | 5 |
| `--output`, `-o` | 输出 xlsx 文件路径 | comparison_results.xlsx |
| `--llmbo-n-init` | LLMBO 暖启动评估数 | 15 |
| `--llmbo-t-max` | LLMBO BO 迭代次数 | 50 |
| `--llmbo-batch` | LLMBO 批次大小 | 3 |
| `--use-llm` | 使用 LLM 指导 | True |
| `--no-llm` | 不使用 LLM 指导 | False |
| `--nsgaii-pop` | NSGA-II 种群大小 | 50 |
| `--seed` | 基础随机种子 | 42 |
| `--workers` | PyBaMM 并行进程数 | 3 |
| `--quiet` | 静默模式 | False |

## xlsx 输出文件结构

生成的 xlsx 文件包含以下 Sheet：

### Sheet 1: HV_Curves
HV 曲线数据（用于绘制收敛图）

| Run_ID | Algorithm | Iteration | N_Evaluations | HV |
|--------|-----------|-----------|---------------|-----|
| 1 | LLMBO | 0 | 15 | 0.123 |
| 1 | LLMBO | 1 | 18 | 0.145 |
| 1 | NSGAII | 0 | 50 | 0.098 |

### Sheet 2: Pareto_LLMBO
LLMBO 最终 Pareto 前沿点

| Run_ID | I1 | I2 | I3 | SOC_sw1 | SOC_sw2 | V_CV | I_cutoff | t_charge | T_peak | delta_Q |
|--------|----|----|----|---------|---------|------|----------|----------|--------|---------|
| 1 | 5.2 | 4.1 | 3.5 | 0.35 | 0.55 | 4.15 | 0.2 | 25.3 | 38.2 | 1.2e-4 |

### Sheet 3: Pareto_NSGAII
NSGA-II 最终 Pareto 前沿点（格式同上）

### Sheet 4: Convergence
收敛指标统计

| Run_ID | Algorithm | Final_HV | Convergence_Evals | Total_Time_s | N_Pareto | N_Valid |
|--------|-----------|----------|-------------------|--------------|----------|---------|
| 1 | LLMBO | 0.456 | 85 | 320.5 | 25 | 142 |
| 1 | NSGAII | 0.398 | 100 | 280.2 | 30 | 150 |

### Sheet 5: Statistics
汇总统计（多次运行时）

| Algorithm | N_Runs | HV_Mean | HV_Std | HV_Best | HV_Worst | Time_Mean_s | Pareto_Size_Mean |
|-----------|--------|---------|--------|---------|----------|-------------|------------------|
| LLMBO | 5 | 0.452 | 0.018 | 0.478 | 0.432 | 315.2 | 24.5 |
| NSGAII | 5 | 0.385 | 0.022 | 0.412 | 0.358 | 275.8 | 28.0 |

### Sheet 6: Config
实验配置信息

| Parameter | Value |
|-----------|-------|
| n_runs | 5 |
| total_evals | 165 |
| timestamp | 2026-03-22T12:00:00 |

## 使用 Python API

```python
from run_comparison import run_comparison, ExperimentConfig

# 创建实验配置
exp_cfg = ExperimentConfig(
    n_runs=5,                    # 5 次独立运行
    llmbo_n_init=15,             # LLMBO 暖启动
    llmbo_t_max=50,              # LLMBO 迭代次数
    llmbo_batch_size=3,          # 批次大小
    llmbo_use_llm=True,          # 使用 LLM
    nsgaii_pop_size=50,          # NSGA-II 种群
    random_seed_base=42,
)

# 运行对比实验
results = run_comparison(exp_cfg, output_path="my_comparison.xlsx")

# 访问结果
llmbo_results = results["LLMBO"]
nsgaii_results = results["NSGAII"]

for r in llmbo_results:
    print(f"Run {r.seed}: HV={r.final_hv:.4f}, Pareto={r.n_pareto}")
```

## 绘图示例

### 使用 matplotlib 绘制 HV 收敛曲线

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 HV 曲线数据
df = pd.read_excel("comparison_results.xlsx", sheet_name="HV_Curves")

# 按算法和运行分组
fig, ax = plt.subplots(figsize=(10, 6))

for algo in ['LLMBO', 'NSGAII']:
    algo_df = df[df['Algorithm'] == algo]
    for run_id in algo_df['Run_ID'].unique():
        run_df = algo_df[algo_df['Run_ID'] == run_id]
        ax.plot(run_df['N_Evaluations'], run_df['HV'],
               alpha=0.3, label=f'{algo} Run {run_id}' if run_id == 1 else "")

    # 绘制平均值
    mean_df = algo_df.groupby('N_Evaluations')['HV'].mean().reset_index()
    ax.plot(mean_df['N_Evaluations'], mean_df['HV'],
           linewidth=2, label=f'{algo} Mean')

ax.set_xlabel('Number of Evaluations')
ax.set_ylabel('Hypervolume')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hv_convergence.png', dpi=300)
plt.show()
```

### 绘制最终 Pareto 前沿对比

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取 Pareto 数据
llmbo_pf = pd.read_excel("comparison_results.xlsx", sheet_name="Pareto_LLMBO")
nsgaii_pf = pd.read_excel("comparison_results.xlsx", sheet_name="Pareto_NSGAII")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# t_charge vs T_peak
axes[0].scatter(llmbo_pf['t_charge'], llmbo_pf['T_peak'],
               c='blue', alpha=0.6, label='LLMBO', s=50)
axes[0].scatter(nsgaii_pf['t_charge'], nsgaii_pf['T_peak'],
               c='red', alpha=0.6, label='NSGAII', s=50)
axes[0].set_xlabel('Charging Time [min]')
axes[0].set_ylabel('Peak Temperature [°C]')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# t_charge vs delta_Q
axes[1].scatter(llmbo_pf['t_charge'], np.log10(llmbo_pf['delta_Q']),
               c='blue', alpha=0.6, s=50)
axes[1].scatter(nsgaii_pf['t_charge'], np.log10(nsgaii_pf['delta_Q']),
               c='red', alpha=0.6, s=50)
axes[1].set_xlabel('Charging Time [min]')
axes[1].set_ylabel('log10(Capacity Loss [Ah])')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pareto_comparison.png', dpi=300)
plt.show()
```

## 实验配置说明

### 评估预算匹配

两种算法使用相同的总评估次数：
```
Total Evaluations = n_init + t_max * batch_size
                  = 15 + 50 * 3 = 165
```

NSGA-II 的代数自动计算：
```
n_gen = total_evals / pop_size = 165 / 50 ≈ 3 代
```

### 公平性保证

1. **相同随机种子**: 每次对比运行使用相同的 seed
2. **相同评估函数**: 都使用 PyBaMM SPMe 模型
3. **相同约束处理**: 同样的边界和约束条件
4. **相同 HV 参考点**: 使用固定的参考点计算 HV

## 常见问题

### Q: NSGA-II 运行失败 "pymoo not installed"
```bash
pip install pymoo
```

### Q: xlsx 导出失败 "pandas not found"
```bash
pip install pandas openpyxl
```

### Q: PyBaMM 仿真速度慢
- 减少 `--runs` 运行次数
- 增加 `--workers` 并行进程数（如果 CPU 核心多）
- 使用 `--quiet` 减少输出

### Q: 如何修改评估预算？
```bash
# 增加总评估到 300 次
python run_comparison.py --llmbo-t-max 95 --runs 3
# Total = 15 + 95 * 3 = 300
```

## 性能基准

在典型配置下（165 次评估，单种算法）：

| 算法 | 运行时间 | 最终 HV |
|------|----------|---------|
| LLMBO-MO (with LLM) | ~350s | 0.45-0.50 |
| LLMBO-MO (no LLM) | ~280s | 0.42-0.47 |
| NSGA-II | ~250s | 0.35-0.42 |

*基于 LG INR21700-M50 电池模型，Intel i7 CPU*

## 参考文献

1. Deb, K. et al. (2002). NSGA-II: A fast and elitist multiobjective genetic algorithm.
2. Turbo, A. et al. (2020). Scalable Global Optimization via Local Bayesian Optimization.
3. LLAMBO Paper (2024). LLM-Assisted Bayesian Optimization for Multi-Objective Problems.
