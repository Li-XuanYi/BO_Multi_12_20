# LLMBO-MO 实现约束文件
## ⚠️ 本文件优先级高于你自己的判断，必须逐条遵守

你将收到两份参考文档：
- `FrameWork.md`：框架完整规格（权威来源）
- `LLMBO_MO_Architecture_Blueprint.pdf`：模块接口蓝图

**在开始任何 Batch 之前，先通读本文件。本文件解决了在此之前反复出现的5类实现错误。**

---

## 约束 C-1：Ψ(θ) 必须是焦耳热代理函数

**禁止**使用任何高斯响应面形式，例如：
```python
# ❌ 禁止
psi = exp(-2*I1_norm**2 - 3*SOC_dev**2 - 1.5*I2_norm**2)
```

**必须**按 FrameWork.md §2 Eq.6 实现：

```python
# ✅ 唯一合法实现
Q_NOM = 18000  # C（5Ah × 3600）
SOC0 = 0.1
SOC_END = 0.8
R1_BAR = 0.01  # Ω，在 config.py 以 PSI_R1 配置
R2_BAR = 0.01  # Ω，在 config.py 以 PSI_R2 配置

def psi(I1, SOC1, I2):
    t1 = (SOC1 - SOC0) * Q_NOM / I1
    t2 = (SOC_END - SOC1) * Q_NOM / I2
    return R1_BAR * I1**2 * t1 + R2_BAR * I2**2 * t2

# 解析梯度（FrameWork.md Eqs.8-10，简化形式，忽略∂R/∂SOC）
def grad_psi(I1, SOC1, I2):
    dI1   = R1_BAR * (SOC1 - SOC0) * Q_NOM
    dSOC1 = Q_NOM * (I1 * R1_BAR - I2 * R2_BAR)
    dI2   = R2_BAR * (SOC_END - SOC1) * Q_NOM
    return np.array([dI1, dSOC1, dI2])
```

**梯度验证容差**：1e-4（中心差分，δ=1e-5）。

---

## 约束 C-2：W^(t) 不存在 W_data，没有融合系数

**禁止**任何以下形式：
```python
# ❌ 禁止
W_final = alpha * W_data + (1 - alpha) * W_LLM
W_data = gradient_outer_product_average(...)
```

**必须**按 FrameWork.md Eq.11' 实现：

```python
# ✅ 唯一合法实现
# W_time, W_temp, W_aging 由 Touchpoint 1a 在初始化时生成一次，之后永远不变
def build_W_t(w, W_time, W_temp, W_aging):
    """w = 当轮 Tchebycheff 权重向量，形状 (3,)"""
    s = w[0] + w[1] + w[2]
    return (w[0] * W_time + w[1] * W_temp + w[2] * W_aging) / s
```

W^(t) 的"适应性"完全来自每轮不同的 λ 权重，不来自数据更新。

---

## 约束 C-3：λ 权重策略必须使用 Riesz s-energy 集合

**禁止**任何以下形式：
```python
# ❌ 禁止
weights = np.random.dirichlet([1, 1, 1])
weights = np.array([1/3, 1/3, 1/3])
```

**必须**在初始化时预生成结构化权重集合，每轮随机选取：

```python
# ✅ 唯一合法实现（在 tchebycheff.py 的 __init__ 中执行一次）
from pymoo.util.ref_dirs import get_reference_directions
weight_set = get_reference_directions("energy", 3, 15, seed=42)  # shape (15, 3)

# 每轮迭代
w_t = weight_set[np.random.randint(len(weight_set))]
```

---

## 约束 C-4：W_charge 是三维高斯，不是 Ψ(θ)

**禁止**任何以下形式：
```python
# ❌ 禁止
W_charge = psi_function(theta)
alpha = EI * psi_function(theta)
```

**必须**按 FrameWork.md §5 Eqs.17-22 实现：

```python
# ✅ 唯一合法实现
def compute_W_charge(theta, mu, sigma):
    """
    三维高斯搜索权重
    theta: 候选点，shape (3,)
    mu:    搜索中心（动态漂移，Eqs.18-19），shape (3,)
    sigma: 搜索宽度（灵敏度引导，Eq.20），shape (3,)
    """
    return float(np.prod(
        np.exp(-0.5 * ((theta - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
    ))

# mu 更新（Eqs.18-19）：
#   alpha_t = alpha_max * exp(-t / t_decay_alpha) + alpha_min
#   mu_j^(t+1) = alpha_t * mu_j^(t) + (1 - alpha_t) * theta_j_best

# sigma 计算（Eq.20）：
#   grad_at_best = |grad_psi(theta_best)|  # shape (3,)
#   c = kappa * max(grad_at_best)
#   sigma_j = c / (grad_at_best[j] + eps_sigma)

# 采集函数：
#   alpha(theta) = EI(theta) * W_charge(theta)
```

---

## 约束 C-5：γ 更新无条件执行，无 if/else

**禁止**任何以下形式：
```python
# ❌ 禁止
if delta_f > 0:
    gamma = gamma * (1 + 0.1 * delta_f / f_prev)
# else: 保持不变
```

**必须**无条件执行（manuscript1 规则）：

```python
# ✅ 唯一合法实现
def update_gamma(gamma, f_min_t, f_min_prev, gamma_min=0.1, gamma_max=2.0):
    gamma = gamma * (1 + 0.1 * (f_min_t - f_min_prev) / abs(f_min_prev))
    return float(np.clip(gamma, gamma_min, gamma_max))
# 无论 Δf 正负，每轮都执行此函数
```

理由：条件版本产生单向棘轮效应（γ只能增大），当物理先验有误时无法自动修正。

---

## 约束 C-6：config.py 超参数

```python
# ✅ 必须使用的超参数值
GAMMA_INIT    = 0.1    # ❌ 不是 0.5
ALPHA_MAX     = 0.7
ALPHA_MIN     = 0.05
T_DECAY_ALPHA = 60
KAPPA         = 0.20
EPS_SIGMA     = 0.001
RHO           = 0.1
N_WEIGHTS     = 15     # Riesz s-energy 权重向量数
PSI_R1        = 0.01   # Ω
PSI_R2        = 0.01   # Ω
```

---

## 约束 C-7：Pareto 代表点选择（可视化用，最多6个）

**禁止**纯粹的随机起点 FPS：

```python
# ❌ 禁止
first = random.choice(pareto)
# 然后 FPS...
```

**必须**使用以下混合策略：

```python
# ✅ 唯一合法实现
def select_representatives(pareto_objs, k=6):
    """
    pareto_objs: array shape (N, 3)，已归一化
    返回索引列表，长度最多 k
    """
    selected = []
    # Step 1：强制选入3个极端点（每个目标的最优解）
    for dim in range(3):
        idx = int(np.argmin(pareto_objs[:, dim]))
        if idx not in selected:
            selected.append(idx)

    # Step 2：选1个膝点（距 CHIM 平面最远）
    anchors = pareto_objs[selected[:3]]  # 3个极端点
    v1 = anchors[1] - anchors[0]
    v2 = anchors[2] - anchors[0]
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-10)
    dists = np.abs((pareto_objs - anchors[0]) @ normal)
    for i in np.argsort(-dists):
        if i not in selected:
            selected.append(int(i))
            break

    # Step 3：FPS 补充剩余名额
    while len(selected) < k:
        remaining = [i for i in range(len(pareto_objs)) if i not in selected]
        if not remaining:
            break
        min_dists = np.min(
            np.linalg.norm(
                pareto_objs[remaining][:, None] - pareto_objs[selected][None, :],
                axis=2
            ), axis=1
        )
        selected.append(remaining[int(np.argmax(min_dists))])

    return selected
```

---

## 约束 C-8：关于 FrameWork.md §3.4 gp_model.py 中的 W 矩阵注释

Architecture Blueprint PDF 的 §3.4 中有一行注释：
> "权重 α_t, β_t, δ_t 随时间更新（由 γ 动态更新驱动）"

这行注释是**错误的**，请忽略。W^(t) 的权重就是当轮 λ（Tchebycheff 权重），不存在独立的 α_t/β_t/δ_t 参数，不由 γ 驱动。正确实现见约束 C-2。

---

## 自检清单（每个 Batch 完成后执行）

| 检查项 | 通过条件 |
|--------|---------|
| psi_function.py | `verify_gradient(psi, theta_test)` 误差 < 1e-4 |
| tchebycheff.py | `weight_set.shape == (15, 3)`，所有行和为1 |
| gp_model.py | `build_W_t` 中无 `W_data` 相关变量 |
| acquisition.py | `compute_W_charge` 函数签名含 `mu, sigma` 参数 |
| optimizer.py | `update_gamma` 无 if/else，`gamma_init=0.1` |
| config.py | `GAMMA_INIT=0.1`，`N_WEIGHTS=15`，`PSI_R1=0.01` |

---

## 开始执行

确认以上所有约束后，按以下顺序执行：

```
Batch 1: config.py + utils.py + psi_function.py  → 验证梯度 (tol=1e-4)
Batch 2: battery_model.py                          → 单次仿真测试
Batch 3: tchebycheff.py + gp_model.py + acquisition.py → 10轮无LLM BO
Batch 4: database.py + database_summarizer.py + llm_interface.py → Mock测试
Batch 5: optimizer.py                              → 端到端测试
Batch 6: visualization.py + run.py                 → 完整流程
```

每个 Batch 完成后，运行对应验证步骤，通过后再进入下一个 Batch。
