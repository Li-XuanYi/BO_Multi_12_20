# New_LLMBO Codex总结文档

更新时间：2026-04-29

## 1. 当前项目定位

当前推荐主线仍然是：

```text
LLM WarmStart + augmented Tchebycheff scalarization + single-output Matern GP + plain EI
```

一句话说明：

> LLM 主要负责在初始化阶段生成更好的候选池和 warmstart portfolio；GP 仍然负责概率建模；EI 仍然负责正式选点。

当前已经实现、但默认关闭的 GP-LLM 研究支线是：

```text
Region-Lifted GP
```

它的真实语义不是“重训练一个新的 GP”，而是：

```text
在 acquisition 阶段，对标准化目标空间里的 posterior mean 做一个 bounded mean shift；
不 refit GP；
不改 posterior covariance；
不改 EI 公式；
任何异常都 fail-open 回 plain EI。
```

这也是当前 README 中已经明确写出的语义声明。

## 2. 当前代码框架总览

核心入口和主文件：

- `main.py`
- `llmbo/optimizer.py`
- `llmbo/gp_model.py`
- `llmbo/scalarization.py`
- `llmbo/acquisition.py`
- `llm/llm_interface.py`
- `llm/region_prompt.py`
- `llmbo/region_lifted_gp.py`
- `llmbo/warmstart_selector.py`
- `DataBase/database.py`

主流程可以概括为：

```text
1. main.py 读取配置 / preset，并构造 flat config
2. BayesOptimizer.setup()
   - simulator
   - database
   - llm interface
   - GP
   - acquisition
3. run_initialization()
   - LLM warmstart candidate pool
   - deterministic warmstart selector
   - random init
   - simulator evaluate
   - database update
4. run_optimization_loop()
   - sample current weight w_vec
   - 从 raw objectives 现算当前轮 scalarized objective f_w
   - 拟合单输出 Matern GP
   - plain EI 在 candidate pool 上选点
   - 可选：LLM rerank 或 Region-Lifted GP
   - simulator evaluate
   - update HV / Pareto / telemetry
5. save_results()
   - summary.json
   - database.json
   - pareto_front.json
```

## 3. 当前主线的关键语义

### 3.1 Scalarization

当前不是直接在三目标上做 multi-output BO，而是每轮对当前权重 `w_vec` 做一次 ParEGO 风格标量化：

```text
raw objectives
-> log transform
-> dynamic normalization
-> augmented Tchebycheff under current w_vec
-> scalar_y = f_w
```

相关代码在：

- `llmbo/scalarization.py`
- `DataBase/database.py`
- `llmbo/optimizer.py`

重要约束：

- 不要新增永久 `Observation.scalarized` 字段
- `scalar_y` 是“当前轮上下文相关量”，不是静态 observation 属性

### 3.2 GP

当前主线 surrogate 仍然是单输出 Matern GP，代码在 `llmbo/gp_model.py`。

当前没有默认启用：

- deep-kernel GP
- multi-output GP
- qNEHVI / BoTorch 重构
- 持续型旧版 `enable_gp_llm_coupling`

主线仍然是：

```text
fit GP on scalar_y
-> posterior mean / std
-> plain EI
```

### 3.3 Acquisition

当前主线 acquisition 仍然是 plain EI，代码在 `llmbo/acquisition.py`。

当前研究支线有两类：

- post-EI rerank
- Region-Lifted GP

但默认推荐 preset 里，这两类都关闭。

## 4. LLM 当前在项目中的使用位置

当前 LLM 不是一个统一大入口，而是分成几个明确触点。

### 4.1 WarmStart

相关文件：

- `llm/llm_interface.py`
- `llm/warmstart_prompt.py`
- `llmbo/warmstart_selector.py`

当前 WarmStart 不是“直接拿 LLM 给出的前 3 个点”，而是：

```text
LLM over-sample 一个候选池
-> hard filter
-> soft safety / monotone / diversity / archive-aware selector
-> 输出最终 warmstart portfolio
```

实现要点：

- 默认 `warmstart_pool_size = 16`
- 默认 `n_warmstart = 3`
- 默认 `n_random_init = 3`
- selector 不是随机选，而是 deterministic selection
- 已支持磁盘缓存，避免真实 API 抽样波动反复污染实验

WarmStart 相关关键参数：

```text
warmstart_batch_size = 10
warmstart_max_attempts = 4
warmstart_max_retries = 3
warmstart_max_tokens = 2500
warmstart_pool_size = 16
warmstart_diversity_weight = 0.45
warmstart_soft_penalty_weight = 0.65
warmstart_monotone_bonus = 0.08
warmstart_archive_bonus_weight = 0.0
warmstart_boundary_probe_limit = 1
```

WarmStart cache 相关参数：

```text
warmstart_cache_path
warmstart_cache_mode = read / write / read_write
warmstart_cache_use_selected = true / false
random_init_cache_path
```

### 4.2 Region-Lifted GP

相关文件：

- `llm/region_prompt.py`
- `llm/llm_interface.py`
- `llmbo/region_lifted_gp.py`
- `llmbo/optimizer.py`
- `llmbo/gp_model.py`

这是当前真正的 GP-LLM 研究实现。

它不是让 LLM 替代 GP，也不是让 LLM 直接替代 EI，而是：

```text
LLM 输出一个 raw-space promising point/region
-> 转成一个 bounded standardized-space mean shift
-> 用 shifted mean + 原始 sigma 重新算 EI
-> 同一个 candidate pool 上比较 plain EI 与 lifted EI
-> guard 通过才允许使用 lifted 选点
-> 否则 fail-open 回 plain EI
```

### 4.3 Safe Rerank

相关文件：

- `llmbo/rerank.py`
- `llm/rerank_prompt.py`
- `llm/templates/candidate_rerank.txt`

当前支持的 rerank mode：

- `none`
- `ei_preserving_tiebreak`
- `risk_veto_only`
- `unsafe_legacy_const_gate`

但当前推荐主线和当前 region-lift 实验都没有打开 rerank。

## 5. 当前 GP-LLM 的详细实现逻辑

这一节是目前最重要的工程说明。

### 5.1 入口和开关

Region-Lifted GP 的开关在：

- `main.py`
- `llmbo/optimizer.py`

核心参数名：

```text
enable_region_lifted_gp
```

推荐 preset：

- `warmstart_plain_ei`
- `strict_baseline`
- `warmstart_region_lifted_gp`

其中：

- `warmstart_plain_ei`：主线推荐
- `strict_baseline`：无 LLM 对照
- `warmstart_region_lifted_gp`：默认关闭的研究支线

### 5.2 先拟合普通 GP，再决定是否做 lift

Region-Lifted GP 不改训练过程。

真实顺序是：

```text
1. 先用当前轮 scalar_y 拟合 base Matern GP
2. 用 plain EI 在 candidate pool 上得到 x_plain
3. 再向 LLM 查询当前轮 region preference
4. 在同一个 candidate pool 上构造 lifted EI
5. 如果 guard 通过，再允许 x_lift 替换 x_plain
```

也就是说，lift 是 acquisition-time shaping，不是 refit。

### 5.3 标准化目标空间

Region-Lifted GP v1 统一在 GP 标准化目标空间中工作。

`llmbo/gp_model.py` 里新增了：

- `target_standardization()`
- `predict_standardized()`
- `posterior_covariance_standardized()`
- `posterior_covariance_raw()`

核心关系是：

```text
mu_z  = (mu_y - y_mean) / y_std
sigma_z = sigma_y / y_std
cov_z = cov_y / y_std^2
```

因此当前 lift 的语义不是在 raw scalar_y 上直接偏移，而是在 standardized `z-space` 中做偏移。

### 5.4 LLM 输出的 region schema

Region prompt 在 `llm/region_prompt.py`。

当前要求 LLM 返回的 JSON 结构是：

```json
{
  "kind": "point | region | none",
  "coordinate_space": "raw",
  "preference_direction": "promising",
  "point": {"I1": null, "I2": null, "I3": null, "dSOC1": null, "dSOC2": null},
  "lb": {"I1": null, "I2": null, "I3": null, "dSOC1": null, "dSOC2": null},
  "ub": {"I1": null, "I2": null, "I3": null, "dSOC1": null, "dSOC2": null},
  "confidence": 0.72,
  "preference_type": "balanced | fast_charge | thermal_safe | aging_safe | boundary_probe",
  "reason": "short rationale",
  "risk_flags": []
}
```

当前 v1 只接受：

- `coordinate_space = raw`
- `preference_direction = promising`

以下情况会直接 fail-open：

- `kind = none`
- 非法 JSON
- `coordinate_space != raw`
- `preference_direction != promising`
- `confidence < region_lift_min_confidence`

### 5.5 Region-Lifted GP 的数学实现

当前实现位于 `llmbo/region_lifted_gp.py`。

在通过 parser 和初步 validation 后，系统会：

```text
1. 把 region 的 lb / ub 转成 raw-space box
2. 归一化后检查 volume 和每维宽度
3. 在 box 内用 deterministic Sobol 采样 anchors
4. 对 anchors 做 deterministic feasibility check
5. 计算 anchors 到历史点的距离，避免全贴着已采样区域
6. 用 standardized posterior covariance 构造 bounded mean shift
7. 同一 candidate pool 上算 plain EI 和 lifted EI
8. 用 EI gap guard 决定是否接受 lift
```

当前公式是：

```text
lambda_t =
  anneal_t
  * lambda_max
  * confidence
  * trust
  / sqrt(max(a^T (K_GG + jitter I) a, min_norm_sq))
```

```text
shift_z = clip(lambda_t * K_xG @ a, 0, max_shift_std)
```

```text
mu_lifted_z = mu_plain_z - shift_z
sigma_lifted_z = sigma_plain_z
```

也就是说：

- minimization 语义下，promising region 会把局部 mean 往更小方向推
- `sigma` 不变
- LLM 改的是 desirability，不是 epistemic uncertainty

### 5.6 为什么是同一个 candidate pool

当前 v1 明确要求：

```text
x_plain 和 x_lift 必须在同一个离散 candidate pool 上比较
```

这样做的目的，是避免“差异来自优化器随机性”而不是“差异来自 lift”。

当前 guard 用的是：

```text
plain_log_ei_surrogate = log(max(EI_plain, eps))
gap = plain_log_ei_surrogate(x_plain) - plain_log_ei_surrogate(x_lift)
```

只有当：

```text
gap <= region_lift_max_plain_ei_gap
```

时，才允许接受 lifted 候选。

### 5.7 当前 fail-open 机制

当前 branch 非常保守，任何异常都回 plain EI。

当前真正落盘过的 fallback 原因包括：

- `parse_fail`
- `bad_region_volume`
- `bad_region_width`
- `empty_region`

代码里还定义了更多 guard 名称，例如：

- `non_raw_coordinate_space`
- `non_promising_direction`
- `low_confidence`
- `too_close_to_existing`
- `low_feasible_anchor_ratio`
- `no_feasible_anchors`
- `plain_ei_gap`

但在最新 5-seed 实验中，主要触发的是前四类。

### 5.8 trust 更新机制

trust 不是固定常数，而是 EMA 风格更新，逻辑在 `llmbo/optimizer.py` 的 `_finalize_region_lift_trust()`。

关键点：

- 只有 `selected_source == lifted` 时才正常更新 trust
- 如果只是 parser/invalid preference 失败，会做 small decay
- 如果是 `EI gap fail-open`，不会像“真用了 lift 却效果差”那样重罚

当前默认参数：

```text
region_lift_trust_init = 0.5
region_lift_trust_beta = 0.2
```

## 6. 当前 LLM 的使用方式和默认参数

### 6.1 通用 LLM 配置

当前默认配置来自 `main.py` 和 `llmbo/optimizer.py`：

```text
llm_backend = openai
llm_model = gpt-4.1-mini
llm_api_base = https://api.nuwaapi.com/v1
llm_n_samples = 3
llm_temperature = 0.7
```

说明：

- 真实实验里主要使用 `openai` backend
- baseline 组通常用 `mock` backend 或关闭 LLM 触点
- 真实 API 会有波动，因此 cache 很重要

### 6.2 WarmStart 调用方式

WarmStart 由 `llm_interface.generate_warmstart_candidates()` 驱动。

特点：

- 可以多 batch 调 LLM
- 默认目标不是只拿 `n_warmstart` 个点，而是先过采样一个 pool
- 如果 LLM 候选不够，会用 physics-informed fallback 补齐

WarmStart 还支持：

- selected cache replay
- candidate pool cache replay

这也是当前我们能稳定复现实验结果的关键原因之一。

### 6.3 Region preference 调用方式

Region-Lifted GP 的 LLM 调用在 `llm_interface.query_region_preference()`：

```text
n = 1
temperature = min(config.temperature, 0.3)
max_tokens = 1000
```

这个路径没有 heuristic fallback。

也就是说：

```text
解析失败 -> kind = none / parse_fail
-> optimizer fail-open 回 plain EI
```

这是刻意设计的，目的是避免“把 LLM 失败偷偷变成 heuristic 成功”。

### 6.4 Rerank 调用方式

虽然当前主线不用 rerank，但文档里保留一下，方便后续接手。

`score_candidate_goodness()` 的当前行为：

- `mock` backend 下会走 GP-based fallback 打分
- real backend 下会做多样本 candidate scoring 聚合
- rerank 使用的温度更低：

```text
temperature = min(config.temperature, 0.2)
```

不过当前推荐主线和当前 region-lift 实验都没有开 rerank。

## 7. 当前推荐 preset

### 7.1 推荐主线

```text
warmstart_plain_ei
```

含义：

```text
n_warmstart = 3
n_random_init = 3
enable_warmstart_portfolio = true
enable_iterative_guidance = false
enable_gp_llm_coupling = false
enable_acq_prior_coupling = false
enable_proposal_sampler = false
enable_llm_rerank = false
enable_region_lifted_gp = false
```

### 7.2 严格对照

```text
strict_baseline
```

含义：

```text
n_warmstart = 0
n_random_init = 6
no LLM
plain GP
plain EI
```

### 7.3 当前 GP-LLM 研究支线

```text
warmstart_region_lifted_gp
```

它只是在 `warmstart_plain_ei` 的基础上打开：

```text
enable_region_lifted_gp = true
```

其余大部分研究开关仍然关闭。

### 7.4 一个重要提醒

不要裸用 `DEFAULT_CONFIG` 当实验结论。

原因是 `DEFAULT_CONFIG` 里仍然保留了一些 legacy 默认值，例如：

- `enable_iterative_guidance = True`
- `enable_acq_prior_coupling = True`

真正推荐的实验语义来自 preset 覆盖，而不是裸 default。

## 8. 目前 5 组实验效果

这里的“5组”指的是：

```text
seed = 0, 1, 2, 3, 4
```

每个 seed 都跑了 3 个 variant：

- `strict_baseline`
- `warmstart_plain_ei`
- `warmstart_region_lifted_gp`

实验总报告：

- `optimized_experiments/region_lift_20iter_seed012_2026_04_28/report_5seeds.json`

### 8.1 实验设置

本轮设置是：

```text
n_iterations = 20
variants = strict_baseline / warmstart_plain_ei / warmstart_region_lifted_gp
LLM model = gpt-4.1-mini
LLM backend = openai
llm_n_samples = 3
llm_temperature = 0.7
```

此外：

- per-seed warmstart cache 保持一致
- per-seed random init cache 保持一致
- `w_sample_seed` 与 `init_seed` 按 seed 对齐
- region-lift 内部的 plain/lift 比较使用同一 candidate pool

### 8.2 5-seed 汇总结果

按 `canonical_hv` 汇总：

| Variant | Mean | Median | Worst-Quartile | Min | Variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict_baseline | 0.358726 | 0.372048 | 0.327386 | 0.327386 | 0.000391964 |
| warmstart_plain_ei | 0.372059 | 0.371509 | 0.360444 | 0.360444 | 0.000052075 |
| warmstart_region_lifted_gp | 0.374784 | 0.377571 | 0.366655 | 0.366655 | 0.000027118 |

结论：

- `WarmStart + PlainEI` 明显优于 `Baseline`
- `WarmStart + Region-Lifted GP` 的均值、median、worst-quartile 都优于 `WarmStart + PlainEI`
- `Region-Lifted GP` 的方差没有变大，反而更小

### 8.3 每个 seed 的结果

| Seed | Baseline | WarmStart + PlainEI | WarmStart + Region-Lifted GP | Region Lift Accept Rate |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.372729 | 0.360444 | 0.379859 | 0.10 |
| 1 | 0.372048 | 0.379157 | 0.379157 | 0.00 |
| 2 | 0.343472 | 0.371509 | 0.370677 | 0.10 |
| 3 | 0.377997 | 0.380159 | 0.377571 | 0.15 |
| 4 | 0.327386 | 0.369028 | 0.366655 | 0.25 |

观察：

- Region-Lifted GP 不是每个 seed 都赢
- 但从 5-seed aggregate 看，它已经满足“统计上优于 plain warmstart”的当前门槛
- `seed=1` 完全没接受 lift，所以它和 `warmstart_plain_ei` 数字相同

### 8.4 当前 fallback 分布

从 5-seed 的 `region_lift_telemetry` 汇总：

```text
lifted   = 12
fallback = 88
overall lift_accept_rate = 0.12
```

fallback 原因分布：

| Fallback Reason | Count |
| --- | ---: |
| bad_region_volume | 39 |
| empty_region | 32 |
| parse_fail | 12 |
| bad_region_width | 5 |

这说明当前 lift 接受率低的主因不是 `EI gap`，而是：

```text
1. LLM 输出的 region 尺度问题
2. 空 region / 非法 bounds
3. parse_fail
```

目前 5-seed 数据并没有显示：

- `plain_ei_gap` 是主导失败原因
- `too_close_anchors` 是主导失败原因
- `low_feasible_anchor_ratio` 是主导失败原因

因此下一步更合理的是：

- 先修 region output 质量和 parser 稳定性
- 再谈是否放宽 guard

## 9. 当前可以对外怎么讲

目前最稳妥的说法是：

> New_LLMBO 的主线仍然是 WarmStart + PlainGP + PlainEI。LLM 在当前系统中最稳的作用仍然是 WarmStart。与此同时，我们已经实现了一个默认关闭的 Region-Lifted GP 研究支线：LLM 只提供 coarse region preference，系统仅在 standardized objective space 中对 posterior mean 做 bounded shift，并保持 covariance 与 plain EI 主流程不变。5 seeds × 20 iterations 的当前结果表明，这条支线在 aggregate 统计上已经优于 warmstart_plain_ei，但它仍然是保守 fail-open 的研究分支，而不是默认生产主线。

不建议现在直接对外说：

- “LLM 已经稳定替代 GP”
- “LLM-EI 是当前最优主线”
- “Region-Lifted GP 已经每个 seed 都赢”

## 10. 当前最合理的下一步

优先级建议：

1. 继续把 Region-Lifted GP 的 fallback 分类做细
2. 优先修 `bad_region_volume / empty_region / parse_fail`
3. 不要先急着放宽 EI gap guard
4. 之后再补 3 个 ablation：
   - `warmstart_region_lifted_gp_random_region_ablation`
   - `warmstart_region_lifted_gp_no_trust`
   - `warmstart_region_lifted_gp_oracle_region`

这三组的作用分别是：

- random region：验证收益是不是“随便加个 region bias 都能来”
- no trust：验证 trust EMA 是否真的有贡献
- oracle region：验证 lift mechanism 的上限是否存在

## 11. 新工程师建议阅读顺序

建议按下面顺序读：

1. `main.py`
2. `llmbo/optimizer.py`
3. `llmbo/scalarization.py`
4. `llmbo/gp_model.py`
5. `llmbo/acquisition.py`
6. `llmbo/region_lifted_gp.py`
7. `llm/llm_interface.py`
8. `llm/region_prompt.py`
9. `llmbo/warmstart_selector.py`
10. `DataBase/database.py`

推荐先跑的验证：

```powershell
python -m py_compile main.py llm\llm_interface.py llmbo\optimizer.py llmbo\region_lifted_gp.py llmbo\warmstart_selector.py
python -m pytest tests -q
```

## 12. 最短总结

当前最重要的结论可以压缩成三句话：

```text
1. 主线仍然是 warmstart_plain_ei，不是让 LLM 替代 GP 或 EI。
2. 当前真正的 GP-LLM 实现是 Region-Lifted GP：LLM 只改 standardized-space posterior mean，不改 covariance，不 refit GP。
3. 5 seeds × 20 iterations 的最新结果里，warmstart_region_lifted_gp 的 mean / median / worst-quartile canonical HV 都优于 warmstart_plain_ei，但它仍然是一个低接受率、强 fail-open 的研究分支。
```
