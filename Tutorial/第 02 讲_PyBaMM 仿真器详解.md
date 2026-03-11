# 第 02 讲：PyBaMM 仿真器详解

> **学习目标**：深入理解锂电池仿真器的工作原理，掌握决策变量到三目标的映射关系

---

## 1. PyBaMM 仿真器的角色

### 1.1 在框架中的位置

```
LLMBO-MO 优化循环
    │
    ├──► LLM 生成候选点 θ = [I₁, SOC₁, I₂]
    │
    ▼
┌─────────────────────┐
│  PyBaMMSimulator    │  ← 本讲核心
│  evaluate(θ)        │
└─────────────────────┘
    │
    ▼
返回三目标：
  - f₁ = 充电时间 [s]
  - f₂ = 峰值温度 [K]
  - f₃ = 老化程度 [%]
```

### 1.2 仿真器的职责

PyBaMM 仿真器是**目标函数评估器**，它的作用是：

| 输入 | 处理 | 输出 |
|------|------|------|
| θ = [I₁, SOC₁, I₂] | 运行 PyBaMM SPMe 模型 | [t_charge, T_peak, L_aging] |

**关键特点**：
- **黑盒函数**：无法获得解析表达式，只能通过仿真得到数值
- **计算昂贵**：每次评估需要数秒到数十秒
- **带约束**：电压、温度超过阈值则判定为违规

---

## 2. 决策变量详解

### 2.1 两阶段 CC-CC 充电协议

```python
# pybamm_simulator.py:4-22
"""
决策变量：θ = (I₁, SOC₁, I₂)
  I₁   ∈ [3, 7]  A       第一阶段恒流电流
  SOC₁ ∈ [0.10, 0.70]    阶段切换 SOC
  I₂   ∈ [1, 5]  A       第二阶段恒流电流

充电协议：CC₁ → CC₂ (两阶段恒流)
  阶段 1: 以 I₁ 充电，SOC0 → SOC₁
  阶段 2: 以 I₂ 充电，SOC₁ → SOC_end (0.80)
"""
```

**物理含义图解**：

```
SOC
  │
SOC_end=0.80 ─────────────┼────────  充电结束
        │                 │
SOC₁    │◄── 阶段 2 ──►│    │  I = I₂ (较小)
        │       (I₂)      │
        │                 │
        │◄── 阶段 1 ──►│  │  I = I₁ (较大)
SOC₀    │       (I₁)      │
  │
  └────────────────────────────────────→ 时间
        t₁              t₂
        │◄─ t_charge ─►│
```

### 2.2 阶段时间计算

**理论时间公式**（不考虑电压截止）：

```python
# pybamm_simulator.py:168-170
# t = ΔSOC × Q_nom(Ah) × 3600 / I(A)

t1 = (SOC1 - soc0) * Q_nom_Ah * 3600.0 / I1
t2 = (soc_end - SOC1) * Q_nom_Ah * 3600.0 / I2
```

**示例计算**：
- Q_nom = 5.0 Ah
- SOC₀ = 0.035, SOC₁ = 0.4, SOC_end = 0.8
- I₁ = 5.0 A, I₂ = 2.5 A

```
t1 = (0.4 - 0.035) × 5.0 × 3600 / 5.0 = 1314 s ≈ 22 min
t2 = (0.8 - 0.4) × 5.0 × 3600 / 2.5 = 2880 s ≈ 48 min
t_total = 4194 s ≈ 70 min
```

**注意**：实际仿真中，由于电压截止保护，实际时间可能略长。

---

## 3. 仿真器核心代码分析

### 3.1 初始化过程

```python
# pybamm_simulator.py:86-112
class PyBaMMSimulator:
    def __init__(
        self,
        battery_config: Optional[Dict] = None,
        constraints:    Optional[Dict] = None,
        penalty:        Optional[Dict] = None,
    ):
        # 1. 检查 PyBaMM 是否安装
        if not PYBAMM_AVAILABLE:
            raise ImportError("PyBaMM is not installed.")

        # 2. 合并配置
        self.battery     = {**DEFAULT_BATTERY_CONFIG, **(battery_config or {})}
        self.constraints = {**DEFAULT_CONSTRAINTS,    **(constraints or {})}
        self.penalty     = {**DEFAULT_PENALTY,        **(penalty or {})}

        # 3. 计算标称容量
        self.Q_nom_Ah = self.battery["nominal_capacity"]  # 5.0 Ah
        self.Q_nom_C  = self.Q_nom_Ah * 3600.0            # 18000 C

        # 4. 精确计算初始 SOC
        self.soc0 = self._compute_soc0()
```

**关键配置** (`DEFAULT_BATTERY_CONFIG`)：

```python
DEFAULT_BATTERY_CONFIG = {
    "param_set":        "Chen2020",       # 电池参数集
    "nominal_capacity": 5.0,              # 标称容量 [Ah]
    "init_voltage":     3.0,              # 初始电压 [V]
    "init_temp":        298.15,           # 初始温度 [K] = 25°C
    "ambient_temp":     298.15,           # 环境温度 [K]
    "soc_end":          0.80,             # 目标 SOC
}
```

### 3.2 SOC0 的精确计算

**问题**：为什么需要精确计算 SOC0？

**答案**：电池的初始 SOC 与初始电压存在非线性关系，需要通过电化学计量比反算。

```python
# pybamm_simulator.py:394-436
def _compute_soc0(self) -> float:
    """从 init_voltage 精确计算 SOC0"""
    try:
        param = pybamm.ParameterValues(self.battery["param_set"])
        c_n_max = param["Maximum concentration in negative electrode [mol.m-3]"]

        # 读取默认满充浓度 (100% SOC)
        c_n_full = param["Initial concentration in negative electrode [mol.m-3]"]
        x_n_full = c_n_full / c_n_max  # ≈ 0.9014 for Chen2020

        # 设置到 init_voltage
        param.set_initial_stoichiometries(f"{self.battery['init_voltage']} V")
        c_n_init = param["Initial concentration in negative electrode [mol.m-3]"]
        x_n_init = c_n_init / c_n_max

        # Chen2020 空电化学计量比 (≈0% SOC)
        x_n_empty = 0.028

        # 线性映射到 SOC
        soc0 = (x_n_init - x_n_empty) / (x_n_full - x_n_empty)
        soc0 = float(np.clip(soc0, 0.0, 1.0))

        return soc0
    except Exception as e:
        logger.warning(f"SOC0 计算失败 ({e}), 使用近似值 0.04")
        return 0.04
```

**计算结果**：
- init_voltage = 3.0 V → SOC0 ≈ 0.035 (约 3.5%)

### 3.3 核心仿真流程

```python
# pybamm_simulator.py:117-144
def evaluate(self, theta) -> Dict:
    """评估单条充电协议"""
    rng_state = np.random.get_state()  # 保存随机状态
    try:
        theta = np.asarray(theta, dtype=float)
        I1, SOC1, I2 = float(theta[0]), float(theta[1]), float(theta[2])
        return self._run(I1, SOC1, I2)
    except Exception as e:
        logger.error(f"evaluate 意外异常：{e}")
        return self._make_penalty(f"unexpected: {str(e)[:200]}")
    finally:
        np.random.set_state(rng_state)  # 恢复随机状态
```

**为什么要保存/恢复随机状态？**

PyBaMM 内部可能调用 `np.random.seed()`，会破坏外部优化器的随机数生成。保存和恢复确保可复现性。

---

## 4. _run 方法详解

### 4.1 参数验证

```python
# pybamm_simulator.py:153-165
def _run(self, I1: float, SOC1: float, I2: float) -> Dict:
    # 1. 电流必须为正
    if I1 <= 0 or I2 <= 0:
        return self._make_penalty("invalid: current <= 0")

    # 2. SOC1 必须大于 SOC0
    if SOC1 <= self.soc0:
        return self._make_penalty(f"invalid: SOC1={SOC1:.3f} <= SOC0={self.soc0:.3f}")

    # 3. SOC1 必须小于 SOC_end
    if SOC1 >= self.battery["soc_end"]:
        return self._make_penalty(f"invalid: SOC1={SOC1:.3f} >= SOC_end={self.battery['soc_end']}")
```

### 4.2 构建 PyBaMM 实验

```python
# pybamm_simulator.py:167-174
# 计算近似阶段时间（10% 安全余量）
t1 = (SOC1 - self.soc0) * self.Q_nom_Ah * 3600.0 / I1
t2 = (self.battery["soc_end"] - SOC1) * self.Q_nom_Ah * 3600.0 / I2

t1_safe = t1 * 1.10  # 10% 安全余量
t2_safe = t2 * 1.10
```

**为什么需要安全余量？**

实际充电过程中，由于电压截止，可能无法在理论时间内完成充电。安全余量确保 PyBaMM 有足够时间求解。

```python
# pybamm_simulator.py:241-250
experiment = pybamm.Experiment([
    (
        f"Charge at {I1:.4f} A for {t1_safe:.1f} seconds "
        f"or until {V_max} V"  # 电压截止保护
    ),
    (
        f"Charge at {I2:.4f} A for {t2_safe:.1f} seconds "
        f"or until {V_max} V"
    ),
])
```

### 4.3 模型与参数设置

```python
# pybamm_simulator.py:178-187
model = pybamm.lithium_ion.SPMe(
    options={
        "thermal":         "lumped",       # 集总热模型
        "SEI":             "reaction limited",  # SEI 生长模型
        "lithium plating": "irreversible",     # 不可逆锂沉积
    }
)
```

**模型选项解释**：

| 选项 | 值 | 含义 |
|------|------|------|
| thermal | lumped | 使用 0D 集总热模型（通过散热系数耦合） |
| SEI | reaction limited | SEI 膜生长由反应动力学控制 |
| lithium plating | irreversible | 锂沉积为不可逆过程（死锂） |

```python
# pybamm_simulator.py:189-226
param = pybamm.ParameterValues(self.battery["param_set"])

# 关键参数设置
param["Upper voltage cut-off [V]"] = V_max + 0.1  # 4.4V（比约束高 0.1V）
param["SEI growth activation energy [J.mol-1]"] = 37500.0
param["Initial temperature [K]"] = self.battery["init_temp"]
param["Ambient temperature [K]"] = self.battery["ambient_temp"]

# 注入析锂参数（Chen2020 缺失，从 O'Kane2022 借用）
plating_params = {
    "Exchange-current density for plating [A.m-2]": 0.001,
    "Lithium plating open-circuit potential [V]": 0.0,
    "Dead lithium decay constant [s-1]": 3.33e-7,
    "Lithium plating transfer coefficient": 0.65,
    # ... 更多参数
}
param.update(plating_params, check_already_exists=False)
```

### 4.4 求解器配置

```python
# pybamm_simulator.py:253-266
try:
    # 优先使用 IDAKLUSolver（更快）
    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
except Exception:
    # 回退到 ScipySolver
    solver = pybamm.ScipySolver(atol=1e-6, rtol=1e-6)

sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=param,
    solver=solver,
)
sol = sim.solve()
```

---

## 5. 三目标提取

### 5.1 时间与温度提取

```python
# pybamm_simulator.py:279-301
def _extract(self, sol) -> Dict:
    # 充电时间
    time_entries = sol["Time [s]"].entries
    total_time = float(time_entries[-1])  # 最终时刻

    # 峰值温度（兼容多版本变量名）
    for temp_var in (
        "Cell temperature [K]",
        "X-averaged cell temperature [K]",
        "Volume-averaged cell temperature [K]",
    ):
        try:
            temp_entries = sol[temp_var].entries
            break
        except KeyError:
            continue
    peak_temp = float(np.max(temp_entries))  # 最高温度
```

### 5.2 老化提取（容量损失）

```python
# pybamm_simulator.py:345-378
def _extract_aging(self, sol) -> float:
    """
    从 SEI + 锂沉积提取容量损失百分比

    aging(%) = (Q_loss / Q_nom) × 100
    Q_loss   = Σ(li_loss_mol) × F / 3600
    """
    F = 96485.33212  # 法拉第常数 [C/mol]
    total_loss_mol = 0.0

    # SEI 锂损失（兼容多版本变量名）
    for name in [
        "Loss of lithium to SEI [mol]",
        "Loss of lithium to negative SEI [mol]",
        "Loss of lithium to negative SEI on cracks [mol]",
    ]:
        val = self._safe_final(sol, name)
        if val is not None:
            total_loss_mol += val

    # 锂沉积损失
    for name in [
        "Loss of lithium to lithium plating [mol]",
        "Loss of lithium to negative lithium plating [mol]",
    ]:
        val = self._safe_final(sol, name)
        if val is not None:
            total_loss_mol += val

    if total_loss_mol <= 0:
        return 1e-8  # 避免 log(0)

    Q_loss_Ah = total_loss_mol * F / 3600.0
    return float(max((Q_loss_Ah / self.Q_nom_Ah) * 100.0, 1e-8))
```

**老化机理**：

```
容量损失来源：
│
├─ SEI 膜生长
│  └─ 消耗活性锂离子 → 不可逆容量损失
│
└─ 锂沉积（析锂）
   └─ 锂离子在负极表面沉积为金属锂 → 死锂
```

### 5.3 约束检查

```python
# pybamm_simulator.py:313-324
V_max = self.constraints["voltage_max"]  # 4.3 V
T_max = self.constraints["temp_max"]     # 328.15 K (55°C)

if peak_voltage > V_max:
    return self._make_penalty(f"voltage={peak_voltage:.3f}V > {V_max}V")
if peak_temp > T_max:
    return self._make_penalty(f"temp={peak_temp:.2f}K > {T_max}K")
```

### 5.4 惩罚结果

```python
# pybamm_simulator.py:441-454
def _make_penalty(self, violation: str) -> Dict:
    """违规/失败 → 返回固定惩罚值"""
    logger.warning(f"惩罚：{violation}")

    raw = np.array([
        self.penalty["time"],   # 7200 s
        self.penalty["temp"],   # 338 K
        self.penalty["aging"],  # 0.5 %
    ])

    return {
        "raw_objectives": raw,
        "feasible":       False,
        "violation":      violation,
        "details":        None,
    }
```

---

## 6. 物理一致性验证

### 6.1 预期物理趋势

| 趋势 | 描述 | 验证方法 |
|------|------|----------|
| 时间 | I₁ 越大 → 时间越短 | 比较不同 I₁ 的 t_charge |
| 温度 | I₁ 越大 → 温度越高 | 比较不同 I₁ 的 T_peak |
| 老化 | I₁ 越大 → 老化越高 | 比较不同 I₁ 的 L_aging |

### 6.2 自测代码

```python
# pybamm_simulator.py:478-567
if __name__ == "__main__":
    sim = PyBaMMSimulator()

    tests = [
        ("保守 (低电流)", [3.5, 0.40, 2.0]),
        ("平衡",         [5.0, 0.35, 2.5]),
        ("激进 (高电流)", [7.0, 0.20, 4.0]),
    ]

    for name, theta in tests:
        res = sim.evaluate(theta)
        if res["feasible"]:
            obj = res["raw_objectives"]
            print(f"{name}: time={obj[0]:.0f}s, temp={obj[1]:.2f}K, aging={obj[2]:.6f}%")
```

---

## 7. 理解检查

完成本讲后，你应该能够：

- [ ] 解释两阶段 CC-CC 充电协议的工作原理
- [ ] 推导阶段时间的计算公式
- [ ] 说明为什么需要精确计算 SOC0
- [ ] 描述 PyBaMM 模型的配置选项
- [ ] 解释 SEI 和锂沉积对老化的贡献
- [ ] 说明约束检查的机制

---

## 8. 课后练习

### 练习 2-1: 手动计算

给定参数：
- Q_nom = 5.0 Ah
- SOC₀ = 0.035, SOC₁ = 0.35, SOC_end = 0.80
- I₁ = 6.0 A, I₂ = 3.0 A

**任务**：
1. 计算 t1 和 t2 的理论值
2. 如果 I₁ 增加到 7.0 A，t1 变为多少？

<details>
<summary>点击查看答案</summary>

```
t1 = (0.35 - 0.035) × 5.0 × 3600 / 6.0 = 945 s ≈ 15.75 min
t2 = (0.80 - 0.35) × 5.0 × 3600 / 3.0 = 2700 s ≈ 45 min

I₁=7.0A 时：
t1 = (0.35 - 0.035) × 5.0 × 3600 / 7.0 = 810 s ≈ 13.5 min
```

</details>

### 练习 2-2: 代码运行

在本地运行：

```bash
cd New_LLMBO
python pybamm_simulator.py
```

观察输出，记录：
1. 三个测试点的目标值
2. 哪个点的时间最短？
3. 哪个点的温度最高？

### 练习 2-3: 参数敏感性

修改 `DEFAULT_CONSTRAINTS` 中的 `voltage_max` 为 4.2 V，重新运行测试，观察结果变化。

**问题**：违规点数量是否增加？为什么？

---

## 9. 预习下一讲

下一讲我们将深入 **GP 模型**，了解：
- 物理代理函数 Ψ(θ) 的推导
- 复合核函数的设计原理
- γ退火机制的作用
- Universal Kriging 预测

**预习任务**：
1. 阅读 `gp_model.py` 的 `PsiFunction` 类
2. 推导 Ψ(θ) 对三个变量的偏导数
3. 思考：为什么需要复合核而不是标准 RBF？

---

*第 02 讲 完*
