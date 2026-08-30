# Scalarization 对比实验 - 运行指南

## 环境准备

在 PowerShell 中设置 API 密钥（一次性）：

```powershell
$env:LLM_API_KEY="你的APIKey"
$env:LLM_BASE_URL="https://api.nuwaapi.com/v1"
```

## 运行完整实验

```powershell
cd D:\Users\aa133\Desktop\BO_Multi_12_20\New_LLMBO

# 5 seeds × 3 modes × 50 iterations = 15 组实验
python scalarization_Exp\run_scalarization_experiments.py `
    --seeds 8409 8410 8411 8412 8413 `
    --iterations 50 `
    --skip-existing
```

## 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | gpt-4.1-nano（临时） |
| Seeds | 8409, 8410, 8411, 8412, 8413 |
| 迭代数 | 50 |
| Modes | minmax, zscore, none |
| 温度 | 0.0 |
| Warmstart | 3 |
| Random init | 3 |

## 绘制结果

实验完成后运行：

```powershell
python scalarization_Exp\plot_scalarization_hv.py `
    --exp-root scalarization_Exp\experiment_records\<本次实验目录>
```

## 输出结构

```
scalarization_Exp/experiment_records/
└── scalarization_llmbo_mo_5seeds_50iter_gpt41nano_2026_05_09/
    ├── seed8409/
    │   ├── minmax/summary.json
    │   ├── zscore/summary.json
    │   └── none/summary.json
    ├── seed8410/...
    ├── report_5seeds.json
    └── images/
        └── scalarization_hv_comparison.png
```

## 完成后切回默认模型

实验完成后，编辑 `scalarization_Exp/run_scalarization_experiments.py` 第 23 行：

```python
# 改回 gpt-4.1-mini
MODEL_NAME = "gpt-4.1-mini"
```
