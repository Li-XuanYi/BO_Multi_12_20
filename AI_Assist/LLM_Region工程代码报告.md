# LLM_Region 工程代码报告

本文档基于当前工作区代码重新整理，覆盖 `LLM_Region` / `Region-Lifted GP` 的真实工程实现、优化后的 LGBO 分支、旧 heuristic 分支、Prompt、LLM 返回处理、acquisition 接线、telemetry 与当前实验诊断。

适用代码范围：

- `llm/region_prompt.py`
- `llm/llm_interface.py`
- `llmbo/region_lifted_gp.py`
- `llmbo/gp_model.py`
- `llmbo/acquisition.py`
- `llmbo/optimizer.py`
- `main.py`

---

## 1. 模块定位

当前 `LLM_Region` 不是“让 LLM 直接决定下一个实验点”，而是：

1. LLM 输出一个 `promising point` 或 `promising region`。
2. 系统把偏好转成 raw 参数空间中的 box。
3. 在 box 内生成 Sobol anchors。
4. 通过 GP 均值偏移影响 EI。
5. acquisition 仍然按 BO 规则选点。

当前代码中存在两条实现路线：

| 模式 | 配置值 | 定位 |
|---|---|---|
| 旧 heuristic | `region_lift_mode="heuristic_correlation"` | 历史研究支线，保留 EI-softmax、trust、anneal、clip 和经验 guard |
| 新 LGBO | `region_lift_mode="lgbo_proposition1"` | 当前主要对齐论文的分支，uniform anchors、解析 lambda、acquisition 内部 lifted GP |

一句话概括：

`LLM_Region = LLM 给区域偏好 + GP 均值偏移 + EI 仍负责最终选择`

---

## 2. 当前主流程

入口在 `llmbo/optimizer.py::run_optimization_loop()`。

每轮优化的核心顺序是：

1. 用已有实验数据拟合当前标量化 GP。
2. 如果 `enable_region_lifted_gp=True` 且仍在 `region_lift_active_until` 窗口内：
   - 正常模式调用 `_query_region_preference()`。
   - Random-LGBO 模式调用 `_query_region_preference_random_fallback()`，不调用 API。
3. 如果是 `lgbo_proposition1`：
   - `_build_lgbo_acquisition_lift()` 调用 `build_lgbo_region_lift()`。
   - 得到 `LLMPreferenceCoupling(mode="lgbo_region")`。
   - 将该 coupling 传入 `af.step(..., lift=region_acquisition_lift)`。
4. `AcquisitionFunction.step()` 在 candidate pool 和局部优化过程中调用 `gp.predict_with_coupling()`。
5. `_maybe_apply_region_lifted_gp()` 对 LGBO 模式只做 telemetry 汇总，不再后置 override acquisition 结果。
6. 实验点评估结束后，`_finalize_region_lift_trust()` 写入 telemetry；LGBO 模式不更新 trust，只记录 `skipped_lgbo_mode`。

因此，新 LGBO 分支已经不是旧的“先 plain EI，再后置 lifted EI 竞争”的主路径，而是 acquisition 内部使用 shifted GP。

---

## 3. Prompt 构造

Prompt 入口是 `llm/region_prompt.py::render_region_preference_prompt()`。

当前 Prompt 的关键要求：

- 只返回一个 JSON object。
- 返回 raw-coordinate `point` 或 `region`。
- `coordinate_space="raw"`。
- `preference_direction="promising"`。
- `dSOC1+dSOC2 < 0.70`。
- 使用 `w_vec` 理解当前标量化目标权重。
- 证据层级：
  - PRIMARY：领域知识、物理机制、约束、单位。
  - SECONDARY：历史实验数据。
- Anti-collapse：
  - 不要无机理地围绕历史观测点给 region。
- Neutrality rules：
  - 不假设 prompt 中任何具体数值天然好或坏。
  - 不使用固定数值锚点或 canned ranges。
  - 所有数值必须来自 `parameter_bounds`、`current_context` 和机理判断。
- 输出必须包含 `mechanistic_thinking`，但只允许 1-2 句机理摘要，不保存长链式推理。

当前 Prompt 已删除旧版中带具体数值的 JSON 示例，改为字段级 schema：

```text
Point schema keys:
kind, coordinate_space, preference_direction, point, confidence,
preference_type, reason, mechanistic_thinking

Region schema keys:
kind, coordinate_space, preference_direction, lb, ub, confidence,
preference_type, reason, mechanistic_thinking
```

### 3.1 current_context

状态字典在 `llmbo/optimizer.py::_build_region_preference_state()` 中构建，经过 `region_prompt.py::_compact_state()` 压缩。

当前保留字段包括：

- `iteration`
- `w_vec`
- `ideal_point_raw`
- `objective_preprocess_mode`
- `y_min`
- `y_max`
- `eta`
- `f_min`
- `hv_feedback`
- `boundary_failures`
- `previous_region_thinking`
- `previous_thinking`
- `last_region_adoption_note`
- `adoption_note`
- `top_scalar_points`
- `recent_observations`
- `recent_points`
- `uncertainty_hotspots`

`top_scalar_points` 当前保留：

- `theta`
- `raw_objectives`
- `scalar_y`

`recent_observations` 当前保留：

- `theta`
- `objectives`
- `feasible`
- `source`

这意味着旧报告中提到的两个上下文丢失问题已经修复：

- `recent_observations` 不再被 `_compact_state()` 丢弃。
- `raw_objectives` 不再被 `_compact_value()` 丢弃。

### 3.2 当前 Prompt 仍存在的风险

最新实验显示，deepseek-v3 会自行解释 `w_vec` 中每个目标维度的语义，例如把某一维解释成 degradation 或 capacity fade。当前 Prompt 给了 `w_vec` 和 raw objectives，但没有显式提供“目标维度名称/顺序”。

因此当前最重要的 Prompt 风险是：

- LLM 可能猜错 objective 维度含义。
- confidence 集中在 0.65-0.75，校准不足。
- 返回仍偏向 “moderate/balanced currents + SOC steps” 的保守机理。

建议后续在不引导具体参数数值的前提下，加入 objective schema，例如：

```json
{
  "objective_names": ["charge_time", "temperature_rise", "degradation_proxy"],
  "objective_direction": "all minimized"
}
```

---

## 4. LLM 返回值处理

入口在 `llm/llm_interface.py::query_region_preference()`。

处理流程：

1. 调用 `render_region_preference_prompt()`。
2. `LLMCaller.call(..., n=1)` 请求模型。
3. 使用 `_extract_json_flexible()` 做容错 JSON 提取。
4. 调用 `parse_region_preference_payload()`。
5. 如果解析成功，返回 `LLMRegionPreference`。
6. 如果全部失败，返回 `LLMRegionPreference.none(parser_status)`，系统 fail-open 到普通 EI。

### 4.1 JSON 提取容错

`_extract_json_flexible()` 支持：

- 直接 `json.loads(text)`。
- 从 markdown code block 中提取 JSON。
- 从文本首尾 `{...}` 中提取 JSON。
- 修复常见 JSON 错误，例如尾逗号和单引号。

### 4.2 LLMRegionPreference 字段

定义在 `llmbo/region_lifted_gp.py::LLMRegionPreference`，核心字段包括：

- `kind`
- `coordinate_space`
- `preference_direction`
- `point`
- `lb`
- `ub`
- `confidence`
- `preference_type`
- `reason`
- `mechanistic_thinking`
- `risk_flags`
- `raw_response`
- `raw_response_hash`
- `raw_text_preview`
- `llm_call_diagnostics`
- `parser_status`

最近实验中 `LLM_Region` 的 LLM 返回审查结果：

- `30/30` 返回合法 JSON。
- `30/30` parser_status 为 `ok`。
- `30/30` 返回 `kind="region"`。
- `30/30` 都实际改变 acquisition 选点。

审查文件：

- `AI_Assist/LLM_Region_LLM_Returns_Review.md`

---

## 5. Region box、离散化与 anchors

Region 相关工具集中在 `llmbo/region_lifted_gp.py`。

### 5.1 region bounds

核心函数是 `_preference_bounds()`。

如果 LLM 返回：

- `kind="region"`：直接读取 `lb/ub`。
- `kind="point"`：围绕 point 自动扩展成一个小 box。

之后统一执行：

- clip 到参数边界。
- `_repair_region_box(...)` 修复宽度和体积。
- `_apply_dsoc_margin(...)` 处理 `dSOC1+dSOC2` 安全余量。

### 5.2 anchors

anchors 使用 Sobol 在修复后的 box 内生成：

```python
anchors = _sobol_box(lb, ub, int(config.region_lift_n_anchors))
```

之后用 `_deterministic_feasible()` 过滤不可行 anchors。

在 LGBO 模式中：

- anchors 是均匀权重的 region 表示。
- 不使用 EI-softmax。
- 不用 closeness guard 拒绝 region；相关指标只作为 telemetry。

在 heuristic 模式中：

- anchors 还会用于 closeness、feasible ratio、anchor consistency 等 guard 和权重计算。

---

## 6. 新 LGBO 分支

新主分支由 `region_lift_mode="lgbo_proposition1"` 激活。

核心函数：

- `build_lgbo_region_lift()`
- `MaternGPModel.predict_with_coupling()`
- `AcquisitionFunction.step()`
- `_build_lgbo_post_acquisition_summary()`

### 6.1 构造 LGBO coupling

`build_lgbo_region_lift()` 的核心步骤：

1. 验证 LLM preference。
2. 修复 region bounds。
3. 生成 feasible anchors。
4. 使用 uniform weights：

```python
a = np.ones(G) / G
```

5. 计算 anchor 后验标准化协方差：

```python
Sigma_GG = gp.posterior_covariance_standardized(anchors, anchors)
posterior_variance = a.T @ Sigma_GG @ a
```

6. 计算解析 lambda：

```python
lambda = confidence / sqrt(max(posterior_variance, region_lift_lgbo_min_variance))
```

7. 构造：

```python
LLMPreferenceCoupling(
    mode="lgbo_region",
    grid=anchors,
    weights=a,
    confidence=confidence,
    lambda_value=lambda,
    posterior_variance=posterior_variance,
    gate=1.0,
)
```

### 6.2 均值偏移

`MaternGPModel.predict_with_coupling()` 中，如果 `coupling.mode == "lgbo_region"`：

```python
K_XG = prior_kernel_standardized(X_new, anchors)
shift_z = lambda * gate * mask * (K_XG @ weights)
shift_y = shift_z * y_std
mean_lifted = mean - shift_y
```

当前任务是标量化最小化，因此均值向下偏移，使该区域在 EI 中更有吸引力。

关键点：

- 分母使用 posterior standardized covariance。
- 偏移项使用 prior latent standardized kernel。
- 方差不变。
- 不重新训练 GP。
- 不修改 kernel 超参数。

### 6.3 acquisition 内部生效

`AcquisitionFunction.step()` 会在候选池评分和局部优化中使用：

```python
mean, std = gp.predict_with_coupling(candidate_pool, coupling=lift)
ei = expected_improvement(mean, std, f_min)
```

同时为了调试，会计算同一 candidate pool 上的无 lift baseline：

```python
mean_base = gp.predict(candidate_pool)[0]
ei_base = expected_improvement(mean_base, std, f_min)
plain_selected_indices_without_lift = ...
selected_changed_by_lift = ...
```

因此当前 `effective_lift_accept_rate` 对 LGBO 模式已经有实际意义：它比较的是同一候选池下“无 lift EI”和“有 lift EI”的选点差异。

### 6.4 LGBO 模式关闭的东西

在 LGBO 模式下：

- 不使用 `region_lift_lambda_max`。
- 不使用 anneal。
- 不使用 trust 参与 lambda。
- 不使用 anchor consistency 参与 lambda。
- 不做 corr clip。
- 不做 shift clip。
- 不注入 region candidates / restarts。
- 不走后置 override。
- `same_as_plain`、`outside_region`、`plain_ei_gap`、`low_sigma_ratio`、`too_close_to_existing` 只作为 telemetry，不作为 fallback。

保留的结构性 fail-open：

- parse fail / invalid JSON。
- invalid kind。
- 非 raw coordinate space。
- 非 promising direction。
- region repair 失败。
- bad volume / bad width。
- 无 feasible anchors。
- GP kernel/covariance 不可用。

---

## 7. 旧 heuristic 分支

旧分支仍由 `region_lift_mode="heuristic_correlation"` 使用，主要用于历史实验可比性。

它仍然执行：

- EI-softmax anchor weights：

```python
0.60 * log(EI) + 0.25 * sigma + 0.15 * novelty
```

- 后验 covariance correlation：

```python
corr = (K_xg @ weights) / sqrt(var_x * norm_sq)
```

- 经验 reliability：

```python
reliability = confidence * trust * anchor_consistency
```

- 手动 lambda 和退火：

```python
lambda_t = anneal * region_lift_lambda_max
```

- 裁剪：

```python
shift_z = clip(lambda_t * reliability * max(corr, 0), 0, region_lift_max_shift_std)
```

- 经验 guard fallback：
  - `same_as_plain`
  - `too_close_to_existing`
  - `outside_region`
  - `plain_ei_gap`
  - `low_sigma_ratio`

旧 heuristic 分支仍可能通过 `region_lift_external_influence_mode` 影响 acquisition 前候选池，也可能通过 `region_lift_apply_override=True` 后置覆盖最终选点。

---

## 8. Random-LGBO 消融

`random_region_lgbo_proposition1` preset 使用：

```python
region_lift_control_mode = "fixed_random"
```

此模式：

- 不调用 LLM API。
- 使用 deterministic Sobol/random seed 生成固定宽度随机 region。
- 默认 normalized width 为 `0.15`。
- 默认 confidence 为 `0.5`。
- 返回 `preference_type="random_control"`。
- telemetry 中记录 `llm_called_for_region=False`。

该模式用于区分：

- LGBO 均值偏移机制本身的影响。
- LLM 语义区域信息的影响。

---

## 9. Trust 机制

`_finalize_region_lift_trust()` 中有两种行为：

| 模式 | trust 行为 |
|---|---|
| heuristic | 根据 HV gain、inside region、历史表现更新 trust |
| LGBO | 不更新 trust，只记录 `trust_update_reason="skipped_lgbo_mode"` |

因此 LGBO 的偏移强度只由：

```python
confidence / sqrt(a.T @ Sigma_GG @ a)
```

决定，不再乘历史 trust。

---

## 10. Telemetry

当前 telemetry 已经比较完整，尤其是 LGBO 模式会记录：

- `region_lift_mode`
- `region_lift_control_mode`
- `anchor_weighting_mode`
- `lgbo_lambda`
- `lgbo_posterior_variance`
- `lgbo_protected_variance`
- `lgbo_shift_min`
- `lgbo_shift_max`
- `lgbo_shift_mean`
- `lgbo_covariance_source`
- `lgbo_denominator_covariance_source`
- `lgbo_shift_kernel_source`
- `acquisition_used_lift`
- `selected_index_before`
- `selected_index_after`
- `selected_changed_by_lift`
- `selected_score_before`
- `selected_score_after`
- `structural_fallback_reason`
- `fallback_reason`
- `llm_called_for_region`
- `llm_raw_response_hash`

这些字段足以区分：

- LLM 是否成功返回。
- region 是否成功构造。
- coupling 是否进入 acquisition。
- lift 是否真正改变选点。
- shift 是否过大。
- fallback 是否来自结构性失败。

---

## 11. 当前实验诊断

最近三组对照实验：

```text
Baseline / LLM_Region / LLMBO_BO
3 seeds x 10 iter
model = deepseek-v3
```

结果摘要：

| 组别 | mean canonical HV | 相对 Baseline |
|---|---:|---:|
| Baseline | 0.29074 | - |
| LLM_Region | 0.26087 | -0.02987 |
| LLMBO_BO | 0.32980 | +0.03906 |

解释：

- 单独 `LLM_Region` 没有优于 Baseline。
- `LLMBO_BO = WarmStart + LLM_Region/LGBO` 优于 Baseline，且 2/3 seeds 胜。
- `LLMBO_BO` 明显优于单独 `LLM_Region`，3/3 seeds 胜。

`LLM_Region` 的 LLM 返回检查：

- `30/30` parser_status 为 `ok`。
- `30/30` 都是 `kind="region"`。
- `30/30` 都进入 acquisition 并改变选点。
- confidence 集中在 `0.65-0.75`。
- 返回语义仍偏向 “moderate/balanced currents + SOC steps”。

这说明当前问题不是 parse，也不是 guard 全杀，而是：

1. LLM 区域语义仍偏保守。
2. LGBO shift 有时过强。
3. Prompt 还缺少明确 objective 名称/顺序，LLM 可能猜错 `w_vec` 维度含义。

---

## 12. 当前实现的准确结论

当前实现已经完成了几项关键修复：

- Prompt 上下文丢失问题已修复。
- Prompt 删除了引导性数值示例。
- LGBO 主分支已接入 acquisition 内部 lifted GP。
- LGBO 使用 uniform weights。
- LGBO lambda 使用 posterior anchor variance 自适应校准。
- LGBO shift 使用 prior latent kernel。
- LGBO 不再使用经验 guard fallback。
- LGBO trust 更新已跳过。
- Random-LGBO fixed random control 已实现。
- effective lift telemetry 已修正为真实 plain-vs-lift 对比。

但仍不能得出“单独 LLM_Region 已优于 Baseline”的结论。当前更稳妥的结论是：

```text
WarmStart + LGBO Region 的组合当前表现最好；
单独 LLM_Region 仍需要继续优化 prompt 目标语义和 shift 尺度。
```

---

## 13. 后续建议

优先级建议：

1. 给 region prompt 增加 objective schema，明确 `w_vec` 三个维度分别对应什么目标。
2. 对 LLM confidence 做校准，避免长期集中在 0.65-0.75。
3. 增加 LGBO shift 尺度诊断或保守模式，避免 `lgbo_shift_mean` 达到几十到几百的异常强度。
4. 跑 Random-LGBO 与 LLM-LGBO 对照，区分机制贡献和语义贡献。
5. 单独汇总 LLM 返回 region 的分布，检查是否模式坍塌到保守区域。
6. 正式实验至少扩展到更多 seeds 和更长迭代；3 seeds x 10 iter 只能作为 smoke/诊断。

---

## 14. 常见误解澄清

1. **LLM 不直接选实验点。**  
   它只给 point/region preference。

2. **LGBO 不修改 GP 方差。**  
   当前只偏移均值，方差保持不变。

3. **LGBO 不重新训练 GP。**  
   只是 acquisition-time mean shift。

4. **LGBO 模式下 guard 不再全杀建议。**  
   `same_as_plain/outside_region/plain_ei_gap/low_sigma_ratio/too_close` 只记录 telemetry。

5. **旧 heuristic 分支仍保留。**  
   历史实验如果使用 `warmstart_region_lifted_gp_force_pool_tuned`，看到的行为可能仍是旧 guard-heavy 逻辑。

6. **LLMBO_BO 的提升不能归因给单独 LLM_Region。**  
   当前实验显示组合效果好，但单独 LLM_Region 仍低于 Baseline。

