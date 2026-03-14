# LLMBO 运行指南

## 一、快速开始

### 1. 配置 LLM API

首先编辑 `config/settings.py` 文件（第 66-75 行），设置你的 API：

```python
@dataclass(frozen=True)
class LLM:
    BACKEND = "openai"                    # 后端类型：openai / ollama / anthropic
    MODEL = "gpt-4o"                      # 模型名称
    API_KEY = "sk-你的 API 密钥"           # 替换为你的密钥
    API_BASE = "https://api.nuwaapi.com/v1"  # API 地址
    TEMPERATURE = 0.7
    N_SAMPLES = 5
```

### 2. 运行对比实验（LLMBO + ParEGO）

```powershell
# 进入项目目录
cd d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO

# 快速演示（5 回合，测试配置是否正确）
python exp/unified_runner.py --demo

# 正式运行（3 个随机种子）
python exp/unified_runner.py --seeds 0 1 2 --eimo-iterations 50 --parego-iterations 300
```

### 3. 查看结果

运行完成后，结果保存在 `unified_results/` 目录：
```
unified_results/
├── eimo/
│   └── seed_42/
│       ├── db_final.json        # 优化结果数据库
│       └── checkpoints/         # 检查点
├── parego/
│   └── seed_42/
│       └── ...
└── hv_curves.xlsx              # HV 收敛曲线数据
    pareto_front.xlsx           # Pareto 前沿数据
```

---

## 二、完整运行流程

### 步骤 1：检查环境

```powershell
# 激活 conda 环境（如果有）
conda activate llambo

# 检查 Python 版本（需要 3.9+）
python --version

# 检查依赖
pip list | grep -E "numpy|scipy|sklearn|pybamm"
```

### 步骤 2：修改配置

编辑 `config/settings.py`：

| 配置项 | 位置 | 说明 |
|--------|------|------|
| `LLM.MODEL` | 第 70 行 | LLM 模型名称 |
| `LLM.API_KEY` | 第 71 行 | API 密钥 |
| `LLM.API_BASE` | 第 72 行 | API 地址 |
| `BO.N_ITERATIONS` | 第 118 行 | LLMBO 迭代次数 |
| `PAREGO.N_ITERATIONS` | 第 214 行 | ParEGO 迭代次数 |

### 步骤 3：运行优化

#### 方式 A：同时运行 LLMBO + ParEGO（推荐）

```powershell
# 演示模式（快速测试，各 5 回合）
python exp/unified_runner.py --demo

# 单一种子运行
python exp/unified_runner.py --seeds 42 --eimo-iterations 50 --parego-iterations 300

# 多种子并行运行
python exp/unified_runner.py --seeds 0 1 2 --parallel --max-workers 3
```

#### 方式 B：仅运行 LLMBO

```powershell
python exp/unified_runner.py --method eimo --seeds 0 1 2 --eimo-iterations 50
```

#### 方式 C：仅运行 ParEGO

```powershell
python exp/unified_runner.py --method parego --seeds 0 1 2 --parego-iterations 300
```

#### 方式 D：独立运行 ParEGO

```powershell
# 标准 300 回合
python exp/parego_runner.py --output results_parego

# 自定义配置
python exp/parego_runner.py --iterations 500 --warmstart 20 --seed 42
```

### 步骤 4：可视化结果

```powershell
# 绘制 HV 收敛曲线
python plot/plot_hv.py unified_results/hv_curves.xlsx hv_output.png

# 绘制 3D Pareto 前沿
python plot/plot_pareto3d.py unified_results/pareto_front.xlsx pareto_3d.png

# 绘制最优解数量变化
python plot/plot_optimal_count.py unified_results/hv_curves.xlsx count.png
```

---

## 三、命令行参数说明

### unified_runner.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--demo` | 演示模式（快速测试） | - |
| `--seeds` | 随机种子列表 | [42] |
| `--eimo-iterations` | LLMBO 迭代次数 | 50 |
| `--parego-iterations` | ParEGO 迭代次数 | 300 |
| `--method` | 运行方法：eimo/parego/both | both |
| `--parallel` | 启用并行运行 | - |
| `--max-workers` | 最大工作进程数 | 1 |
| `--output` | 输出目录 | unified_results |

### parego_runner.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--iterations` | 迭代次数 | 从 Settings 读取 |
| `--warmstart` | LHS 初始化点数 | 从 Settings 读取 |
| `--n-cands` | 候选点数量 | 从 Settings 读取 |
| `--output` | 输出目录 | results_parego_300 |
| `--seed` | 随机种子 | 42 |
| `--demo` | 演示模式 | - |
| `--verbose` | 详细输出 | - |

---

## 四、常见问题

### Q1: 如何修改 LLM 模型？

编辑 `config/settings.py` 第 70 行：
```python
MODEL = "claude-sonnet-4-20250514"  # 或 "gpt-4o", "qwen2.5:7b" 等
```

### Q2: 如何增加迭代次数？

编辑 `config/settings.py`：
```python
# 第 118 行 - LLMBO
class BO:
    N_ITERATIONS = 100  # 从 50 增加到 100

# 第 214 行 - ParEGO
class ParEGO:
    N_ITERATIONS = 500  # 从 300 增加到 500
```

### Q3: 如何修改候选点数量？

编辑 `config/settings.py` 第 146 行：
```python
class Acquisition:
    N_CANDIDATES = 20  # 每迭代生成 20 个候选点
```

### Q4: 运行时出现 API 错误？

检查 `config/settings.py` 中的配置：
```python
class LLM:
    API_KEY = "正确的密钥"      # 确保密钥有效
    API_BASE = "正确的 API 地址"  # 确保地址正确
    MODEL = "正确的模型名称"     # 确保模型可用
```

### Q5: 如何跳过 Mock 模式？

确保 `config/settings.py` 中：
```python
class LLM:
    BACKEND = "openai"  # 不要设置为 "mock"
```

---

## 五、运行时间估算

| 配置 | 迭代次数 | 预计时间 |
|------|----------|----------|
| 演示模式 | 5 回合 | ~1-2 分钟 |
| 快速测试 | 20 回合 | ~10-20 分钟 |
| 标准运行 | 50 回合 | ~30-60 分钟 |
| 长程运行 | 300 回合 | ~3-6 小时 |

*时间取决于电池模型复杂度和 LLM 响应速度*

---

## 六、输出文件说明

### 数据库文件（db_final.json）

包含所有评估数据：
- `X`: 决策变量 [I1, SOC1, I2]
- `Y`: 目标值 [time, temp, aging]
- `pareto_front`: Pareto 最优解集

### Excel 文件（hv_curves.xlsx）

用于绘制收敛曲线：
- HV: Hypervolume 值
- iteration: 迭代次数
- method: 方法名称（EIMO/ParEGO）

---

## 七、示例运行脚本

创建 `run_experiment.ps1`：

```powershell
# 设置配置
$SEEDS = @(0, 1, 2)
$EIMO_ITERS = 50
$PAREGO_ITERS = 300

# 运行对比实验
python exp/unified_runner.py `
    --seeds $SEEDS `
    --eimo-iterations $EIMO_ITERS `
    --parego-iterations $PAREGO_ITERS `
    --parallel --max-workers 3

# 绘制结果
python plot/plot_hv.py unified_results/hv_curves.xlsx results/hv_comparison.png
python plot/plot_pareto3d.py unified_results/pareto_front.xlsx results/pareto_3d.png

Write-Host "实验完成！"
```

运行：
```powershell
.\run_experiment.ps1
```

---

## 八、配置验证

运行前验证配置：

```powershell
python -c "
from config.settings import Settings
print('LLM:', Settings.LLM.MODEL)
print('BO iterations:', Settings.BO.N_ITERATIONS)
print('ParEGO iterations:', Settings.PAREGO.N_ITERATIONS)
print('配置验证通过！')
"
```
