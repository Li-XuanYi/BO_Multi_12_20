# Codex总结文档

## 1. 目标与当前定位

`New_LLMBO` 当前最合理的定位不是“全程持续耦合 LLM 的新 BO 框架”，而是：

`LLM warmstart + ParEGO 风格标量化 + 单输出 GP + plain EI`

也就是：

1. 多目标原始输出是 `time_s / delta_temp_K / aging_pct`
2. 每轮采样一个 `w_vec`
3. 用 augmented Tchebycheff 把三目标压成当前轮单目标 `f_w`
4. 用单输出 `Matern GP` 拟合 `f_w`
5. 用 `plain EI` 选下一个点
6. LLM 主要在 warmstart 上提供帮助，后续 guidance / coupling / rerank 都是研究支线

当前最稳主线仍然是：

- `warmstart_plain_ei`

而不是：

- 持续 `GP-LLM coupling`
- `acq_prior`
- `proposal sampler`
- `LLMEI const gate`

## 2. 当前代码框架

### 2.1 主流程

主入口：

- [main.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/main.py)

核心优化器：

- [llmbo/optimizer.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/optimizer.py)

主流程大致是：

1. `BayesOptimizer.setup()`
   - 初始化 simulator
   - 初始化 database
   - 初始化 LLM 接口
   - 初始化 GP
   - 初始化 acquisition
2. `run_initialization()`
   - 产生 warmstart 点
   - 产生 random init 点
   - 写入 database
3. `run_optimization_loop()`
   - 取当前轮 `w_vec`
   - 根据 raw objectives 重算当前轮 `scalar_y = f_w`
   - 拟合 GP
   - 跑 EI
   - 可选接 rerank / prior / coupling
   - 调 simulator 评估
   - 更新 database / HV / telemetry
4. `save_results()`
   - 输出 `summary.json`
   - 输出 HV trace
   - 输出 rerank telemetry 等

### 2.2 关键模块

#### 标量化与 HV

- [llmbo/scalarization.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/scalarization.py)
- [DataBase/database.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/DataBase/database.py)

这里已经做过一轮语义清理：

- `log_transform_objectives`
- `compute_dynamic_bounds`
- `compute_tchebycheff_from_raw_with_ideal`
- canonical HV 与 display HV 分离

现在要记住：

- `compute_hypervolume_raw()` / `canonical_hv` 是算法比较主指标
- `compute_hypervolume()` / `display_hv` 只是展示值，内部保留了 `/0.4` 缩放

#### GP

- [llmbo/gp_model.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/gp_model.py)

当前仍然是标准单输出 GP：

- `MaternGPModel`

还没有上：

- deep-kernel GP
- multi-output GP
- qNEHVI / BoTorch 主干

#### Acquisition

- [llmbo/acquisition.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/acquisition.py)

当前主线 acquisition 仍然是：

- `plain EI`

`acquisition.py` 负责：

- EI 计算
- candidate pool 生成
- L-BFGS-B 局部优化
- 候选元信息导出

注意：

- battery-specific 风险特征不要继续塞回 acquisition core
- 这些应该留在 optimizer/rerank 层补充

#### LLM 接口

- [llm/llm_interface.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llm/llm_interface.py)
- [llm/warmstart_prompt.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llm/warmstart_prompt.py)
- [llm/rerank_prompt.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llm/rerank_prompt.py)
- [llm/templates/candidate_rerank.txt](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llm/templates/candidate_rerank.txt)

当前活跃 touchpoint：

- warmstart candidate generation
- rerank scoring

之前已经修过一个重要问题：

- prompt 不再依赖不存在的 `Observation.scalarized`
- 所有 scalarized 信息都按当前 `w_vec / ideal / y_min / y_max / eta` 现算

#### Rerank

- [llmbo/rerank.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/rerank.py)

现在支持三类 rerank 模式：

- `none`
- `ei_preserving_tiebreak`
- `risk_veto_only`
- `unsafe_legacy_const_gate`（历史对照）

其中安全模式是新的收紧版实现。

#### 约束策略

- [llmbo/constraint_policy.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/constraint_policy.py)

当前语义已经固定为两层：

- hard feasibility: `dSOC1 + dSOC2 <= 0.70`
- soft safety margin: `dSOC1 + dSOC2 <= 0.65`

并且：

- `I1 >= I2 >= I3` 仍然只是软偏好
- 不能硬写进 simulator 拒绝逻辑

## 3. 当前推荐 preset

在 [llmbo/optimizer.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/optimizer.py) 里当前有这些关键 preset：

- `warmstart_plain_ei`
- `strict_baseline`
- `warmstart_safe_tiebreak`
- `warmstart_risk_veto`

### 3.1 主线

推荐主线：

- `warmstart_plain_ei`

含义：

- `n_warmstart = 3`
- `n_random_init = 3`
- `enable_iterative_guidance = false`
- `enable_gp_llm_coupling = false`
- `enable_acq_prior_coupling = false`
- `enable_proposal_sampler = false`
- `enable_llm_rerank = false`

### 3.2 对照

- `strict_baseline`
  - no warmstart
  - no LLM
  - plain EI

### 3.3 研究支线

- `warmstart_safe_tiebreak`
  - `llm_rerank_mode = "ei_preserving_tiebreak"`
- `warmstart_risk_veto`
  - `llm_rerank_mode = "risk_veto_only"`

## 4. 最近完成的重要工程清理

### 4.1 已完成

1. 主线固化
   - `warmstart_plain_ei` 成为明确主线
2. scalarization 中心化
   - optimizer 与 database 共用一套标量化逻辑
3. canonical/display HV 分离
4. hard/soft dSOC 语义固定
5. `Observation.scalarized` 依赖清理
6. 新版 safe rerank 落地
   - 只在 EI 后处理
   - 不改 GP
   - 不改 EI 主公式
   - 不改 simulator
   - parse fail open
7. rerank telemetry 增强

### 4.2 safe rerank 现在怎么工作

具体实现：

- [llmbo/optimizer.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/optimizer.py:1348)
- [llmbo/rerank.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/rerank.py:184)

流程：

1. 先跑 plain EI 得到候选池
2. 取 `Top-M`（当前默认 5）
3. 计算 `log_ei`
4. 只保留 `eligible`：
   - `best_log_ei - log_ei_i <= max_log_ei_gap`
5. 调 LLM 对 shortlist 打 `q_good`
6. 若空输出 / 解析失败 / 高熵：
   - 直接 fail-open 回 plain EI
7. 若是 `ei_preserving_tiebreak`：
   - 只做很小 tie-break
8. 若是 `risk_veto_only`：
   - 只惩罚高风险点，不奖励低 EI 点

保守性设计：

- `confidence < min_confidence` 时视为中性
- `conf_eff = confidence * (1 - entropy)`
- safe mode 只能在 eligible 内改选

## 5. 当前实验结论

### 5.1 历史上已确认的事实

1. `WarmStart` 有时能明显好于 `Baseline`
2. `LLMEI const gate` 经常拖后腿
3. 之前更激进的 GP mean coupling 没有稳定守住优势
4. 当前最稳仍是 `WarmStart + PlainEI`

### 5.2 已复现实验

#### 一条确认过的历史线

目录：

- [optimized_experiments/replay_multiseed_safe695_seed1_realapi/report.json](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/replay_multiseed_safe695_seed1_realapi/report.json)

关键结果：

- `baseline_strict`: canonical HV `0.311839`
- `warm3_safe695_focus`: canonical HV `0.351113`

也就是：

- `WarmStart > Baseline`

#### 三组对比

目录：

- [optimized_experiments/replay_multiseed_safe695_seed1_realapi/report_with_llmei.json](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/replay_multiseed_safe695_seed1_realapi/report_with_llmei.json)

结果排序：

- `warmstart_plain_ei > baseline > const_llmei`

说明：

- 老版 `ConstLLMEI` 不是当前推荐方向

### 5.3 新版 safe rerank 实验结论

#### 第一轮：非固定初始化

目录：

- [optimized_experiments/replay_seed1_safe695_safe_rerank_v1/report.json](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/replay_seed1_safe695_safe_rerank_v1/report.json)

表面上：

- `warmstart_risk_veto` 看起来最好

但不能直接下结论，因为各组 warmstart 重新调用了真实 API，初始化点不同。

#### 第二轮：固定初始化

目录：

- [optimized_experiments/replay_seed1_safe695_safe_rerank_fixedinit_v1/report.json](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/replay_seed1_safe695_safe_rerank_fixedinit_v1/report.json)

公平结论：

- `warmstart_plain_ei_fixedinit`
- `warmstart_safe_tiebreak_fixedinit`
- `warmstart_risk_veto_fixedinit`

三者最终 canonical HV 完全相同：

- `0.2389694455564263`

同时：

- `rerank_changed_count = 0`
- `rerank_fail_open_count = 5`

所以当前 safe rerank 的真实结论是：

- 它成功做到了“不污染主线”
- 但暂时没有带来正增益

## 6. 当前最关键的问题

目前最大问题已经不是“LLM 会乱改 EI”，而是：

- shortlist 太相似
- `log_ei_gap` 很小
- LLM 对 top candidates 的判断熵很高
- safe rerank 几乎总是 fail-open

也就是说，当前 safe rerank 更像：

- 一个成功的保护层

而不是：

- 一个成功的增益层

## 7. 新对话最建议继续做的事

### 7.1 最高优先级

做 rerank 诊断，而不是立刻继续铺更大实验。

建议顺序：

1. 缩 `top_m`
   - 从 5 降到 3
2. 做 candidate 去冗余
   - 当前 top candidates 过于相似
3. 分析 `eligible_indices` 和 `entropy`
   - 为什么 LLM 几乎轮轮高熵
4. 只在 fixed-init 下比较
   - 避免被 warmstart 漂移污染

### 7.2 暂时不要做

本阶段不建议直接推进：

- deep-kernel GP
- multi-output GP
- qNEHVI
- proposal sampler 大改
- 重新恢复 GP mean coupling
- calibrated utility fusion

原因是主线还没完全吃透，safe rerank 也还在“保护层验证”阶段。

## 8. 当前重要设计边界

新来的工程师不要轻易打破下面这些边界：

1. 不要把 `warmstart_plain_ei` 主线改掉
2. 不要把 `0.65` 和 `0.70` 混成一个阈值
3. 不要把 `I1 >= I2 >= I3` 硬写成 simulator 约束
4. 不要新增持久化 `Observation.scalarized`
5. 不要用 display HV 当论文/benchmark 主指标
6. 不要一次混改 `scalarization / prior / rerank / coupling`
7. 不要把 safe rerank 重新放大成 acquisition 主导者

## 9. 如何快速上手

### 9.1 先读这些文件

建议顺序：

1. [llmbo/optimizer.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/optimizer.py)
2. [llmbo/acquisition.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/acquisition.py)
3. [llmbo/rerank.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/rerank.py)
4. [llm/llm_interface.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llm/llm_interface.py)
5. [llmbo/scalarization.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/scalarization.py)
6. [DataBase/database.py](d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/DataBase/database.py)

### 9.2 推荐先跑的实验

如果要继续接手实验，先跑 fixed-init 的这三组：

1. `warmstart_plain_ei`
2. `warmstart_safe_tiebreak`
3. `warmstart_risk_veto`

并保持：

- 相同 `fixed_init_points`
- 相同 `w_sample_seed`
- 相同 `init_seed`

### 9.3 读报告时先看什么

优先看：

- `canonical_hv`
- `hypervolume_raw`
- `hv_violations`
- `rerank_changed_count`
- `rerank_fail_open_count`
- `mean_ei_ratio_when_changed`
- `mean_hv_gain_when_changed`

不要只看：

- `display_hv`

## 10. 一句话总结

当前工程已经从“LLM 乱入 GP/EI 主循环”的探索期，收敛到了一个更干净的结构：

`WarmStart + PlainEI` 是主线，`safe rerank` 是保护层研究支线。

现在最值得做的，不是继续堆更复杂的算法，而是把：

- shortlist 多样性
- rerank 熵过高
- fixed-init 公平评估

这三件事先彻底搞清楚。
