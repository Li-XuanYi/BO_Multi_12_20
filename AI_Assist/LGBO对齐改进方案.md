# LGBO 对齐改进方案

本文结合本地论文 `4356_Unleashing_LLMs_in_Bayesi.pdf`、已有 `LLM_Region工程代码报告.md` 和当前 `New_LLMBO` 代码框架，给出一套可落地的 LLM_Region 改造方案。

## 1. 结论摘要

当前 LLM_Region 不是论文 LGBO 的直接实现，而是一个更保守的启发式工程版本：

- 权重：当前默认 `region_lift_anchor_weighting="ei_softmax"`，论文区域模式更接近 uniform 区域平均。
- 均值偏移：当前使用后验相关系数、退火、trust、clip 和多层 guard；论文使用 Proposition 1 的解析均值平移。
- 干预路径：当前同时存在 acquisition 前置候选池注入和 acquisition 后置 lifted EI override；论文主路径是“偏移 GP 均值，然后正常优化 acquisition”。
- Prompt：当前已经能构造 `recent_observations` 和 `raw_objectives`，但之前被 `region_prompt.py` 压缩器静默丢弃，导致 LLM 看不到最近实验和真实目标值。该 P0 bug 已修复。

最合理的路线不是把现有 heuristic 分支全部覆盖，而是新增 `region_lift_mode="lgbo_proposition1"`，保留现有实现作为 `heuristic_correlation` 研究支线。这样可以做干净的 A/B 对照，避免把论文复现和工程防护混在一起。

## 2. 论文机制和代码差异

### 2.1 区域偏好的数学含义

论文 Section 3.2 将 LLM 的区域建议表示为一个区域内函数值的线性泛函，再通过指数提升把偏好注入 GP。Proposition 1 的核心结论是：指数提升不会改变 GP 协方差，只会把均值改成：

```text
mu_lifted(x) = mu(x) + lambda * Sigma_XG @ a
```

在 region 模式下，`G` 是区域内 Sobol grid，`a` 通常可以取 uniform 权重。置信度 `c` 不需要再变成手写 `lambda_max`，论文用区域泛函的后验方差做归一化：

```text
lambda = c / sqrt(a.T @ Sigma_GG @ a)
```

对于我们当前的标量化最小化问题，符号需要反过来：LLM 说“promising region”意味着该区域的标量化目标应该更低，因此应使用：

```text
mean_lifted_z = mean_z - lambda * (Sigma_XG @ a)
```

当前代码位置：

- `llmbo/region_lifted_gp.py:45`：`RegionLiftConfig`
- `llmbo/region_lifted_gp.py:362`：`_gp_anchor_weights`
- `llmbo/region_lifted_gp.py:508`：`evaluate_region_lift_on_pool`
- `llmbo/optimizer.py:1448`：前置 region pool 是否影响 acquisition
- `llmbo/optimizer.py:1739`：trust update

### 2.2 当前启发式实现的问题

当前 `evaluate_region_lift_on_pool()` 的核心是：

```text
corr = (K_xg @ weights) / sqrt(var_x * (weights.T @ K_gg @ weights))
reliability = confidence * trust * anchor_consistency
lambda_t = anneal * region_lift_lambda_max
shift_z = clip(lambda_t * reliability * max(corr, 0), 0, max_shift)
mean_lifted_z = mean_z - shift_z
```

这套逻辑的工程动机很清楚：减少错误 LLM 输出对 BO 的破坏。但它与 LGBO 的 Proposition 1 有三处根本偏差：

- `max(corr, 0)` 和 `clip` 让偏移不再等价于任何指数提升 GP。
- `lambda_t` 是手动超参，论文的 `lambda` 是由置信度和后验区域方差自适应校准。
- `trust * anchor_consistency` 是额外经验机制，理论上不属于 LGBO。

因此，目前的 LLM_Region 不能直接声称拥有论文 Theorem 1 的最坏情况保证。它更像“LLM 区域候选启发式 + GP 后验相关性重打分”。

## 3. 代码层面的可行改造

### P0：修 Prompt 上下文丢失

状态：已完成。

修复内容：

- `llm/region_prompt.py::_compact_value()` 保留 `raw_objectives`。
- `llm/region_prompt.py::_compact_state()` 保留 `recent_observations`、`hv_feedback`、`boundary_failures`、`objective_preprocess_mode`、`y_min/y_max`、`eta`、`f_min`。
- 新增 `tests/test_region_prompt.py` 覆盖 prompt payload。

验证：

```powershell
pytest tests\test_region_prompt.py tests\test_weight_aware_guidance.py::test_mock_backend_region_preference_returns_heuristic_point -q
```

结果：`2 passed`。

后续还应补两个上下文字段，以更接近论文 Appendix B：

- `previous_region_thinking`：上一轮 LLM 的简短理由或压缩推理。
- `last_region_adoption_note`：上一轮建议是否被用作 guidance，实际测试点是否偏离。

这两个字段不应要求 LLM 输出长链式推理，只保存可审计的短理由、采纳状态和实际点。

### P1：新增 `lgbo_proposition1` 模式

建议在 `RegionLiftConfig` 增加：

```python
region_lift_mode: str = "heuristic_correlation"
region_lift_lgbo_min_denom: float = 1e-8
region_lift_lgbo_confidence_scale: float = 1.0
```

默认仍保持当前行为，避免破坏已有实验；LGBO preset 显式设为：

```python
{
    "region_lift_mode": "lgbo_proposition1",
    "region_lift_anchor_weighting": "uniform",
    "region_lift_external_influence_mode": "diagnostic_only",
    "region_lift_apply_override": True,
}
```

`evaluate_region_lift_on_pool()` 内新增早分支：

```python
if config.region_lift_mode == "lgbo_proposition1":
    weights = np.full(len(anchors), 1.0 / len(anchors))
    Sigma_GG = gp.posterior_covariance_standardized(anchors, anchors)
    Sigma_XG = gp.posterior_covariance_standardized(X_pool, anchors)
    denom_sq = float(weights @ Sigma_GG @ weights)
    denom = np.sqrt(max(denom_sq, config.region_lift_lgbo_min_denom))
    c = np.clip(preference.confidence, 0.0, 1.0) * config.region_lift_lgbo_confidence_scale
    lam = c / denom
    shift_z = lam * (Sigma_XG @ weights)
    mean_lifted_z = mean_z - shift_z
    ei_lifted = expected_improvement(mean_lifted_z, sigma_z, f_min_z)
    lift_index = int(np.argmax(ei_lifted))
    return RegionLiftResult(lift_index, "lifted", True, None, telemetry)
```

注意：

- 这里的 `Sigma_GG/Sigma_XG` 应使用 GP 后验协方差，不是 prior kernel。当前 `gp.posterior_covariance_standardized()` 已经是合适入口。
- 不要做 `max(corr, 0)`，负协方差点被“压低/抬高”是 Proposition 1 的自然结果。
- 结构性校验仍要保留：JSON parse、raw coordinate、bounds repair、deterministic feasibility、空 anchors。这些不是 acquisition guard，而是输入合法性要求。

### P1：LGBO 模式关闭 heuristic guard

在 `lgbo_proposition1` 分支中关闭以下 fallback：

- `same_as_plain`
- `plain_ei_gap`
- `outside_region`
- `low_sigma_ratio`
- `zero_shift`
- `too_close_to_existing`

原因：

- `same_as_plain` 在 LGBO 中不是失败，说明偏移后的 acquisition 仍认为 plain 点最好。
- `outside_region` 也不是失败，论文的均值平移影响整个输入空间，最优 acquisition 点可以位于区域外但受区域协方差牵引。
- `plain_ei_gap`、`low_sigma_ratio`、`too_close_to_existing` 都是当前启发式防护，不属于解析提升。

建议 telemetry 继续记录这些指标，但不再用它们拦截选择。

### P2：关闭前置候选池注入路径 A

当前代码有两条 LLM 干预路径：

- 路径 A：`_should_influence_acquisition_with_region()` 决定 region candidates 是否进入 acquisition pool/restarts。
- 路径 B：`evaluate_region_lift_on_pool()` 对 candidate pool 做 lifted EI 选择。

LGBO 模式建议先关闭路径 A：

```python
region_lift_external_influence_mode = "diagnostic_only"
```

这样可以避免“候选池注入”和“均值偏移”两种影响叠加，实验解释更干净。

更严格的论文复现还需要第二阶段改造：把 shifted mean 直接接入 acquisition optimizer，而不是只在 `acq_result.candidate_pool` 上后置竞争。当前后置 override 是工程上成本最低的过渡方案，但它只在有限候选池中近似 LGBO；真正的 LGBO 是 acquisition 在偏移后的 GP 上完成全流程优化。

推荐分两步：

1. Phase 1：后置 `candidate_pool` 版 LGBO，快速跑对照，验证方向。
2. Phase 2：给 acquisition 增加 `mean_shift_provider` 或 `surrogate_view`，让随机采样、restart ranking、局部优化全都基于 shifted mean。

### P2：LGBO 模式跳过 trust update

当前 `optimizer.py::_finalize_region_lift_trust()` 会根据 HV gain 调整 trust。LGBO 模式建议：

```python
if self.cfg.get("region_lift_mode") == "lgbo_proposition1":
    summary["trust_update_reason"] = "skipped_lgbo_mode"
    self._region_lift_telemetry.append(summary)
    return
```

理由：论文使用 `lambda = c / sqrt(a.T @ Sigma_GG @ a)` 控制偏移强度，额外 trust 会让实际偏移不再对应 LLM confidence，也会增加不可解释超参。

### P3：保留 heuristic 分支

当前实现仍有研究价值，建议命名为：

```python
region_lift_mode = "heuristic_correlation"
```

它适合回答另一个问题：在真实 LLM 输出不稳定、prompt parse 失败、候选池有限的工程条件下，经验 guard 是否能提高鲁棒性。但它不应再被称为 LGBO 论文实现。

## 4. 推荐配置矩阵

新增 preset 建议命名为 `warmstart_region_lgbo_proposition1`：

```python
"warmstart_region_lgbo_proposition1": {
    **EXPERIMENT_PRESETS["warmstart_region_lifted_gp_force_pool_tuned"],
    "region_lift_mode": "lgbo_proposition1",
    "region_lift_anchor_weighting": "uniform",
    "region_lift_external_influence_mode": "diagnostic_only",
    "region_lift_apply_override": True,
    "region_lift_override_uses_diagnostic_pool": False,
    "region_lift_trust_beta": 0.0,
}
```

但有一个关键提醒：如果只做 Phase 1 后置 override，`region_lift_override_uses_diagnostic_pool=False` 会让 lifted EI 只在 acquisition 自己产生的候选池中选点；这比论文更保守，也更容易出现 `same_as_plain`。若要提高 LGBO 影响力，同时仍避免路径 A，可以增加一个全局 Sobol candidate pool 仅用于“偏移后 EI 评估”，但不要把它作为 acquisition restart 注入。

## 5. 实验验证方案

### 5.1 最小 smoke test

目标：确认 LGBO 分支被触发、不会 parse_fail、telemetry 完整。

- backend：mock 或固定 fake LLM。
- iter：5。
- 检查：
  - `region_lift_mode == "lgbo_proposition1"`
  - `anchor_weighting_mode == "uniform"`
  - `lgbo_lambda`、`lgbo_denom`、`max_shift_z` 存在。
  - fallback distribution 不再被 `same_as_plain/plain_ei_gap/outside_region` 主导。

### 5.2 快速真实 LLM 实验

目标：确认 Prompt 修复后 parse success 是否恢复。

- 1 seed x 10 iter。
- 模型：当前实验模型即可。
- 对比：
  - Baseline
  - WarmStart
  - Heuristic LLM_Region
  - LGBO Proposition1
  - LLMBO-MO = WarmStart + LGBO Proposition1

关键指标：

- `parse_success_rate`
- `region_lift_effective_accept_count`
- `selected_source` 分布
- canonical HV / feasible HV
- 每轮 `lgbo_lambda` 与 `max_shift_z`

### 5.3 正式 ablation

建议至少：

- 5 seeds x 50 iter：确认趋势。
- 10 seeds x 50 iter：用于报告级结论。

统计输出必须单独列出：

- WarmStart-only 是否仍约等于 LLMBO-MO。
- Baseline 是否仍约等于 LLM_Region。
- LGBO 分支的有效 LLM 调用次数。
- parse_fail 和结构校验失败的占比。

如果 `Baseline == LLM_Region` 仍出现，优先检查 `parse_success_rate` 和 LGBO mode telemetry，而不是先讨论算法本身无效。

## 6. 风险和注意事项

1. 论文理论是 GP-UCB 风格 regret 讨论，我们当前主 acquisition 是 EI，且是多目标 Tchebycheff 标量化后的最小化。可以借鉴 Proposition 1 的均值平移，但不要过度宣称完全继承 Theorem 1。
2. 当前 Phase 1 若只做后置 candidate pool override，不等价于“在 shifted GP 上优化 acquisition”。这会削弱 LGBO 影响，但实现风险小。
3. 置信度 `c` 来自 LLM 自报，可能校准不可靠。建议先直接使用，再通过 telemetry 分析是否需要 `confidence_scale`。
4. Prompt 历史不要无限增长。建议保留 newest-first 最近 5 条完整观测，再给一段 compact history summary。
5. 点模式在论文里不是简单 point-to-box。当前 `_preference_bounds()` 会把 point 扩成 box，这可以先保留；后续若严格复现，应实现以 point 为中心的平滑衰减权重。

## 7. 建议实施顺序

1. 已完成：修复 Prompt 上下文丢失，并加单测。
2. 新增 `region_lift_mode` 配置，默认 `heuristic_correlation`。
3. 在 `evaluate_region_lift_on_pool()` 新增 `lgbo_proposition1` 早分支。
4. 新增 preset `warmstart_region_lgbo_proposition1`，关闭前置 pool 注入和 trust。
5. 补 LGBO telemetry 和单元测试。
6. 跑 1 seed x 10 iter，确认 parse success 和 lifted 分支有效。
7. 跑 5 seeds x 50 iter，对比旧 LLM_Region、LGBO、WarmStart、LLMBO-MO。
8. 如果 LGBO 有效，再做 Phase 2 acquisition 内部 shifted GP 接入。
