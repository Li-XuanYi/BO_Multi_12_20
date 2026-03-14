# LLMBO 统一配置使用说明

## 目录

1. [快速开始](#快速开始)
2. [配置结构](#配置结构)
3. [修改配置](#修改配置)
4. [运行优化](#运行优化)

---

## 快速开始

**所有配置只需修改一个文件：`config/settings.py`**

### 1. 修改 LLM 配置（API Key、模型等）

打开 `config/settings.py`，找到第 63 行起的 `LLM` 类：

```python
@dataclass(frozen=True)
class LLM:
    """LLM API 配置"""
    BACKEND = "openai"           # 后端：openai / ollama / anthropic
    MODEL = "gpt-4o"             # 模型名称
    API_KEY = "sk-xxx"           # 你的 API 密钥
    API_BASE = "https://..."     # API 地址
    TEMPERATURE = 0.7
    N_SAMPLES = 5
```

### 2. 修改优化器配置

**LLAMBO-MO 迭代次数**（第 114 行）：
```python
@dataclass(frozen=True)
class BO:
    N_ITERATIONS = 50        # 迭代次数
    N_WARMSTART = 10         # Warmstart 点数
```

**ParEGO 迭代次数**（第 194 行）：
```python
@dataclass(frozen=True)
class ParEGO:
    N_ITERATIONS = 300       # ParEGO 迭代次数
    N_WARMSTART = 15         # LHS 初始化点数
```

---

## 配置结构

| 配置类 | 说明 | 行号 |
|--------|------|------|
| `LLM` | LLM API 配置（模型、API Key、温度等） | 63-72 |
| `PyBaMM` | 电池仿真配置 | 75-84 |
| `ParamBounds` | 决策变量参数边界 | 87-95 |
| `BO` | 贝叶斯优化配置 | 114-119 |
| `GP` | 高斯过程配置 | 122-132 |
| `Acquisition` | 采集函数配置 | 135-144 |
| `Coupling` | 耦合矩阵配置 | 147-155 |
| `MOBO` | 多目标优化配置 | 158-167 |
| `Riesz` | Riesz 权重配置 | 170-177 |
| `Output` | 输出/检查点配置 | 180-185 |
| `ParEGO` | ParEGO 专用配置 | 194-199 |

---

## 修改配置

### 示例 1：更换 LLM 模型

```python
# 在 config/settings.py 中修改
class LLM:
    MODEL = "claude-sonnet-4-20250514"  # 换成 Claude
    API_KEY = "sk-你的新密钥"
    API_BASE = "https://你的 api 地址"
```

### 示例 2：增加迭代次数

```python
# 在 config/settings.py 中修改
class BO:
    N_ITERATIONS = 100  # 从 50 增加到 100
```

### 示例 3：修改候选点数量

```python
# 在 config/settings.py 中修改
class Acquisition:
    N_CANDIDATES = 20  # 每迭代生成 20 个候选点
```

### 示例 4：临时覆盖配置（代码中）

```python
from llm.llm_interface import build_llm_interface

BOUNDS = {"I1": (3.0, 7.0), "SOC1": (0.1, 0.7), "I2": (1.0, 5.0)}

# 使用统一配置（默认）
llm = build_llm_interface(BOUNDS)

# 临时覆盖某个参数
llm = build_llm_interface(BOUNDS, model="claude-sonnet-4-20250514")
```

---

## 运行优化

### 方式 1：运行 LLAMBO-MO + ParEGO 对比实验

```powershell
# 快速演示（5 迭代，Mock 模式）
python exp/unified_runner.py --demo

# 标准运行（3 个随机种子）
python exp/unified_runner.py --seeds 0 1 2 `
    --eimo-iterations 50 `
    --parego-iterations 300

# 仅运行 LLAMBO-MO
python exp/unified_runner.py --method eimo --seeds 0 1 2

# 仅运行 ParEGO
python exp/unified_runner.py --method parego --seeds 0 1 2
```

### 方式 2：运行 ParEGO

```powershell
# 标准 300 回合
python exp/parego_runner.py --output results_parego_300

# 自定义迭代次数
python exp/parego_runner.py --iterations 500 --warmstart 20
```

### 方式 3：运行 LLAMBO-MO

```powershell
python main.py --iterations 50
```

---

## 配置优先级

配置值来源的优先级（后者覆盖前者）：

1. **统一配置** `config/settings.py`（基础默认值）
2. **构造函数参数**（临时覆盖）
3. **命令行参数**（最高优先级）

---

## 文件修改记录

已完成统一的文件：

- ✅ `config/settings.py` - 统一配置（新建）
- ✅ `llm/llm_interface.py` - 从 settings 导入
- ✅ `llmbo/optimizer.py` - 从 settings 导入
- ✅ `llmbo/ParEGO.py` - 从 settings 导入
- ✅ `exp/unified_runner.py` - 从 settings 导入
- ✅ `exp/parego_runner.py` - 从 settings 导入

---

## 故障排除

### 问题：修改配置后没有生效

**检查**：
1. 确认修改的是 `config/settings.py`
2. 确认保存了文件
3. 重启 Python 进程/终端

### 问题：导入错误 `ModuleNotFoundError`

确保在项目根目录运行：
```powershell
cd d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO
python exp/unified_runner.py --demo
```

### 问题：LLM API 调用失败

检查 `config/settings.py` 中的配置：
```python
class LLM:
    API_KEY = "正确的密钥"
    API_BASE = "正确的 API 地址"
    MODEL = "正确的模型名称"
```
