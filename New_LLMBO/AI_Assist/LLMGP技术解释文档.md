# New_LLMBO 技术解释文档

更新时间：2026-04-30

本文根据 `New_LLMBO` 仓库当前代码、配置、测试与实验结果整理，重点解释仓库中的 `LLMGP` 部分。这里的 `LLMGP` 不是“LLM 直接训练一个新的 GP”，而是当前仓库里名为 `warmstart_region_lifted_gp` 的研究分支，也就是一个保守的 `Region-Lifted GP` 机制。

## 1. 项目整体在做什么

`New_LLMBO` 是一个面向电池三阶段恒流快充协议设计的多目标贝叶斯优化系统。决策变量是 5 维：

- `I1, I2, I3`：三个充电阶段的电流
- `dSOC1, dSOC2`：前两个阶段覆盖的 SOC 区间宽度

第三段宽度隐式为：

```text
dSOC3 = 0.8 - dSOC1 - dSOC2
```

并且要求：

```text
dSOC1 + dSOC2 < 0.70
```

优化目标是 3 个最小化指标：

- 充电时间 `time_s`
- 温升 `delta_temp_K`
- 老化 `aging_%`

对应的关键代码位置：

- 入口与流程编排：`main.py`、`llmbo/optimizer.py`
- 仿真器：`pybamm_simulator.py`
- 数据库与 Pareto/HV：`DataBase/database.py`
- 标量化：`llmbo/scalarization.py`
- GP：`llmbo/gp_model.py`
- 采集函数：`llmbo/acquisition.py`
- LLM 接口：`llm/llm_interface.py`
- Region-Lifted GP：`llmbo/region_lifted_gp.py`

## 2. 主线算法不是“LLM 替代 BO”

这套系统的默认主线仍然是：

```text
WarmStart + ParEGO 风格标量化 + 单输出 Matern GP + Plain EI
```

也就是说，LLM 不是 surrogate 本体，也不是 acquisition 本体。它在当前工程里主要出现在两个触点：

1. `WarmStart`
2. `Region preference`（也就是本文重点的 `LLMGP`）

默认生产主线对应的 preset 是：

- `warmstart_plain_ei`

对照组是：

- `strict_baseline`

研究分支是：

- `warmstart_region_lifted_gp`

这些 preset 定义在 `llmbo/optimizer.py` 的 `EXPERIMENT_PRESETS` 中。

## 3. BO 主循环的真实数据流

每轮 BO 的核心流程如下：

1. 从 Riesz-relaxed 权重集中取一个当前权重向量 `w_vec`
2. 对全部可行历史样本的三目标做变换与标量化
3. 用得到的单输出 `scalar_y` 拟合一个 Matern GP
4. 在候选池上计算 Plain EI
5. 如果打开 `Region-Lifted GP`，再尝试对同一候选池做一次“lifted EI”
6. 只有 guard 通过时，才允许 lifted 结果替换 plain EI 结果
7. 最后仍然调用仿真器真实评估，并更新数据库、Pareto front、HV

这意味着 `LLMGP` 的插入位置是 acquisition-time shaping，而不是 model-training-time replacement。

## 4. 标量化是整个 GP 建模的前提

当前仓库不是直接做 multi-output GP，而是每轮都先把三目标压成一个标量目标。

### 4.1 目标变换

在 `llmbo/scalarization.py` 中：

- 时间做 `log10`
- 老化做 `log10`
- 温升保持原值

记为：

```text
Y_raw -> Y_tilde
```

### 4.2 动态归一化与理想点 gap

当前实现不是简单用历史 `min/max` 做普通归一化，而是围绕当前 `ideal_point_raw` 计算 gap：

```text
Y_gap = |Y_tilde - ideal_tilde| / (y_max - y_min)
```

然后在当前 `w_vec` 下做 augmented Tchebycheff：

```text
scalar_y = max_i (w_i * Y_gap_i) + eta * sum_i (w_i * Y_gap_i)
```

其中 `eta` 默认是 `0.05`。

因此 GP 拟合的对象不是原始三目标，也不是固定不变的单目标，而是“当前这轮权重上下文下的单输出标量目标”。

## 5. GP 主体是什么

`llmbo/gp_model.py` 里的 `MaternGPModel` 是当前 surrogate 主体。

它的主要特征是：

- 输入是 5 维 charging protocol 参数
- 输入先按物理边界线性归一化到 `[0,1]`
- 核函数是 `ConstantKernel * Matern(nu=2.5) + WhiteKernel`
- 使用 `sklearn.gaussian_process.GaussianProcessRegressor`
- 拟合目标是上一节定义的 `scalar_y`

所以当前主线的 GP 仍然是一个标准的单输出 Matern GP，而不是 deep kernel、multi-output GP 或 BoTorch/qNEHVI。

## 6. 这里的 LLMGP 到底是什么

仓库里“真正有实现、而且和 GP 直接耦合”的研究分支，是 `Region-Lifted GP`。

它的语义可以概括成一句话：

```text
LLM 只提供一个 promising raw-space point/region；
系统把它转换成 standardized target space 中的 bounded posterior mean shift；
不重训 GP，不改 posterior covariance，不改 EI 公式主体；
任何不满足条件的情况都 fail-open 回 plain EI。
```

这也是 `README.md` 明确写出来的语义。

## 7. LLMGP 的技术实现细节

## 7.1 LLM 的输入是什么

当 `enable_region_lifted_gp=True` 时，`llmbo/optimizer.py` 会构造 region preference state，并通过 `llm/llm_interface.py` 调用 `query_region_preference()`。

传给 LLM 的上下文主要包括：

- 当前迭代 `t`
- 当前权重 `w_vec`
- 当前 `ideal_point_raw`
- 当前 `y_min / y_max`
- 当前最优标量值 `f_min`
- 最近若干观测
- 当前最好的若干 `top_scalar_points`
- 最近 HV feedback
- boundary failure 统计

Prompt 模板在 `llm/region_prompt.py`，它强制要求 LLM 输出 JSON，且只能返回：

- `kind = point`
- `kind = region`
- `kind = none`

并要求：

- `coordinate_space = "raw"`
- `preference_direction = "promising"`

这两个字段很关键，因为后续 guard 会直接验证它们。

## 7.2 LLM 的输出会先被严格解析

`llmbo/region_lifted_gp.py` 中的 `parse_region_preference_payload()` 会把 LLM 返回解析成 `LLMRegionPreference`。

若出现以下情况，会直接变成 fail-open：

- 不是合法 JSON
- `kind` 非法
- `point` 或 `region` 结构不完整
- `coordinate_space` 不是 `raw`
- `preference_direction` 不是 `promising`
- `confidence` 低于阈值

默认最小置信度由 `region_lift_min_confidence=0.60` 控制。

## 7.3 region 会先变成 anchors

如果 LLM 给的是 region，系统不会直接把这个 region 当成最终候选，而是会：

1. 把 `lb/ub` 转成数组
2. 验证 region 的体积和每维宽度是否在允许范围内
3. 用 Sobol 采样在 region 内生成 anchor 点
4. 检查这些 anchors 是否大多可行、是否离历史点太近

关键约束来自 `RegionLiftConfig`：

- `region_lift_min_volume=1e-4`
- `region_lift_max_volume=0.25`
- `region_lift_min_width=0.05`
- `region_lift_max_width=0.80`
- `region_lift_min_feasible_anchor_ratio=0.6`
- `region_lift_max_close_fraction=0.5`

这一步的工程目标很明确：LLM 只能给 coarse preference，不能给一个过窄、过宽、不可行或者和历史 archive 几乎重合的无效区域。

## 7.4 lift 发生在 GP standardized target space

这是当前实现最重要的一点。

`MaternGPModel` 除了普通的 `predict()` 之外，还提供了：

- `target_standardization()`
- `predict_standardized()`
- `posterior_covariance_standardized()`
- `posterior_covariance_raw()`

其关系是：

```text
mu_z  = (mu_y - y_mean) / y_std
cov_z = cov_y / y_std^2
sigma_z = sqrt(diag(cov_z))
```

其中：

- `y` 是 GP 训练用的原始 `scalar_y`
- `z` 是 GP 目标标准化后的空间

当前的 `Region-Lifted GP` 明确只在 `z-space` 里做均值修正。

## 7.5 数学上怎么做 mean shift

在候选池 `X_pool` 上，系统先得到 plain GP 的：

```text
mu_z(x), sigma_z(x)
```

然后在 LLM 指出的 region 内得到 anchor 集合 `G={g_j}`，再构造均匀权重 `w_j`。

接着计算：

```text
K_gg = Cov_z(G, G)
K_xg = Cov_z(X_pool, G)
norm_sq = w^T K_gg w
lambda_t = anneal(t) * lambda_max * confidence * trust / sqrt(norm_sq)
shift_z(x) = clip(lambda_t * K_xg w, 0, max_shift_std)
mu_lifted_z(x) = mu_z(x) - shift_z(x)
```

这里有几个要点：

- 只下调均值，不上调均值
- shift 被 `max_shift_std` 截断，默认 `0.25`
- `lambda_t` 同时受 `confidence`、`trust`、`anneal(t)` 影响

因为这是最小化问题，所以均值下调会让该区域在 EI 里更有吸引力。

## 7.6 EI 没有被改写，只是换了均值

`llmbo/acquisition.py` 里的 `expected_improvement()` 仍然是普通 minimization EI。

在 region lift 分支里，系统做的是：

```text
plain EI   = EI(mu_z,        sigma_z, f_min_z)
lifted EI  = EI(mu_lifted_z, sigma_z, f_min_z)
```

注意：

- `sigma_z` 不变
- posterior covariance 不变
- GP 不 refit

所以本质上这是一个 bounded posterior-mean shaping，而不是 covariance shaping，更不是重新训练 surrogate。

## 7.7 为什么它是保守的 fail-open 设计

在 `evaluate_region_lift_on_pool()` 中，只有当以下条件都满足时，lifted 结果才会替换 plain 结果：

- preference 本身合法
- region 体积/宽度合法
- anchor 大部分可行
- anchor 不要和历史 archive 过近
- `anneal(t) > 0`
- `lambda_t > 0`
- `max_shift_z > 0`
- lifted 最优点不能和 plain 最优点完全相同
- lifted 候选的 plain-log-EI 不能比 plain 最优差太多

最后一条通过 `region_lift_max_plain_ei_gap=0.25` 控制，本质上是为了避免 LLM 把选择拉到一个 plain EI 明显不支持的位置。

如果任意一条失败，系统会直接退回 plain EI。也就是说：

```text
LLMGP 失败 != BO 失败
LLMGP 失败 -> 退回 plain EI
```

## 7.8 早期窗口与 trust 机制

当前实现还对 `Region-Lifted GP` 加了两个很强的保守器：

### 早期窗口

默认：

```text
region_lift_active_until = 5
region_lift_anneal = linear_decay
```

也就是说，它只希望在 BO 早期起作用，后面快速衰减到 0。

### trust

`llmbo/optimizer.py` 中维护了 `_region_lift_trust`，默认值：

```text
region_lift_trust_init = 0.5
```

如果某次 lifted 选择带来了更好的 HV 增益，trust 会向上更新；如果 preference 本身经常无效，trust 会有小幅衰减。这个设计的意图是让系统逐步学习“当前 LLM region hint 到底值不值得信”。

## 8. 它和另外两条 LLM 分支有什么区别

仓库里有三种容易混淆的 LLM 介入方式：

### 8.1 WarmStart

位置：

- `llm/llm_interface.py`
- `llmbo/warmstart_selector.py`

作用：

- 在 BO 正式开始前生成一个候选池
- 再经过 deterministic selector 选出初始化组合

它影响的是 initialization，不影响后续 GP 数学结构。

### 8.2 Iterative guidance / legacy coupling

位置：

- `llm/llm_interface.py` 里的 `query_iteration_guidance()`
- `llmbo/gp_model.py` 里的 `predict_with_coupling()`

这条线是更早的 “guidance -> coupling” 路径，但在当前主流 preset 中默认关闭。

### 8.3 Region-Lifted GP

位置：

- `llmbo/region_lifted_gp.py`
- `llmbo/optimizer.py`

这是当前最清晰、最保守、语义也最容易解释的一条 GP-LLM 分支：LLM 只给 region preference，系统只做 bounded mean shift。

## 9. 当前代码语义下，应该如何准确描述 LLMGP

最准确的说法是：

```text
LLMGP = an acquisition-time, fail-open, region-preference-driven posterior-mean shaping method in standardized scalarized-objective space.
```

翻成中文就是：

```text
LLMGP 是一种发生在采集阶段、可失败回退、由 LLM 提供区域偏好驱动、只在标准化标量目标空间中修正 GP 后验均值的方法。
```

不准确的说法包括：

- “LLM 训练了一个新的 GP”
- “LLM 替代了 EI”
- “LLM 直接决定下一次实验点”
- “Region-Lifted GP 会修改 posterior variance”

这些都和当前代码实现不一致。

## 10. 结合当前实验结果如何理解这条分支

优先看你当前打开的实验目录：

- `optimized_experiments/region_lift_v2_50iter_seed01234_2026_04_29/report_5seeds.json`

这份 5-seed、50-iteration 的 v2 报告显示：

### 10.1 三个 variant 的聚合 HV

按 `canonical_hv` 统计：

| Variant | Mean canonical HV | Median | Worst quartile |
|---|---:|---:|---:|
| `strict_baseline` | 0.3821 | 0.3826 | 0.3762 |
| `warmstart_plain_ei` | 0.3753 | 0.3741 | 0.3685 |
| `warmstart_region_lifted_gp` | 0.3753 | 0.3741 | 0.3685 |

### 10.2 这次 v2 实验里，Region-Lifted GP 实际上没有生效

从各 seed 的 `summary.json` 可以看到：

- 每个 seed 的 `region_lift_attempt_count = 50`
- 每个 seed 的 `region_lift_accept_count = 0`
- `lift_accept_rate = 0.0`

进一步看 `region_lift_telemetry`，5 个 seed 的合计 fallback 分布是：

| Fallback reason | Count |
|---|---:|
| `bad_region_volume` | 25 |
| `active_until_expired` | 225 |

这意味着：

1. 前 5 轮虽然 LLM 给出了 region，但这些 region 没通过体积约束
2. 之后 45 轮已经超出 `active_until=5` 的 early-only 窗口，系统直接不再启用 lift

因此这组 v2 实验中：

```text
warmstart_region_lifted_gp == warmstart_plain_ei
```

从结果上看是完全一致的，这恰恰说明 fail-open 机制在按设计工作。

### 10.3 这组结果不说明“LLMGP 没价值”，但说明“当前 v2 配置太保守”

根据这批结果，更合理的技术判断不是：

- “Region-Lifted GP 失败了”

而是：

- “Region-Lifted GP 的 guard 与 early-window 过强，导致这组 v2 实验没有形成有效干预”

换句话说，当前 v2 结果验证的是“安全回退能力”，不是“性能增益能力”。

## 11. 工程上这条分支的优点与限制

### 优点

- 语义清晰，容易解释
- 不改 GP 训练流程，工程侵入小
- 不改 posterior covariance，数值风险较低
- 可严格 fail-open，不会因为 LLM 出错把 BO 主循环拖垮
- 便于做消融实验，因为和 plain EI 很容易对齐比较

### 限制

- LLM 只给 coarse region preference，信息密度有限
- 多重 guard 过强时，很容易退化成“有模块但无有效干预”
- 只改均值，不改不确定性，表达能力有限
- 当前默认只在前几轮激活，长期影响很弱
- 依赖 region schema 的稳定性，prompt 轻微波动就可能触发 fail-open

## 12. 一句话结论

如果只用一句话概括当前仓库的 `LLMGP`，我会这样写：

```text
New_LLMBO 中的 LLMGP 不是 LLM 训练 GP，而是一个默认关闭的 Region-Lifted GP 研究分支：LLM 只输出 promising point/region，系统仅在标准化标量目标空间中对 GP posterior mean 做有界下调，并在严格 guard 下与 plain EI 竞争，失败时无条件回退到 plain EI。
```

如果结合你当前打开的 `region_lift_v2_50iter_seed01234_2026_04_29` 实验，再补一句更具体的结论，就是：

```text
当前这组 v2 实验里，LLMGP 因 region 体积约束和 early-only 时间窗而 0 次接受，因此行为上与 warmstart_plain_ei 完全一致；这说明它的 fail-open 设计是成立的，但也说明当前配置尚未释放出有效的 region-lift 增益。
```
