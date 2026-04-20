# New_LLMBO 工程现状报告

## 1. 项目定位

当前项目是一个面向锂电池 3-stage CC 快充协议优化的多目标贝叶斯优化系统。

核心目标：

- 决策变量：`I1, I2, I3, dSOC1, dSOC2`
- 原始优化目标：`time_s, delta_temp_K, aging_pct`
- 最终评价指标：`HV (hypervolume)`

当前主干并不是直接做多输出 GP，而是：

1. 每轮采样一个 `w_vec`
2. 在该权重下把多目标做 augmented Tchebycheff 标量化，得到单输出 `f_w`
3. 用单个 Matern GP 拟合 `f_w`
4. 用 EI 选点
5. 长期依靠不同 `w_vec` 推动 Pareto front 扩张

关键文件：

- [llmbo/optimizer.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/optimizer.py)
- [llmbo/gp_model.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/gp_model.py)
- [llmbo/acquisition.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/acquisition.py)
- [DataBase/database.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/DataBase/database.py)
- [llm/llm_interface.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llm/llm_interface.py)

## 2. 已完成的核心修复

### 2.1 HV / Pareto / 约束链路修复

已经修复：

- HV 非单调问题
- Pareto duplicate 问题
- `dSOC1 + dSOC2` 约束语义不一致问题
- `lambda` 爆炸问题的 clamp / annealing
- prompt 中缺少安全裕度说明的问题

验证状态：

- 相关测试通过
- 多轮 benchmark 中 `hv_violations = 0`

关键文件：

- [DataBase/database.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/DataBase/database.py)
- [utils/constants.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/utils/constants.py)
- [tests/test_hv_and_coupling.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/tests/test_hv_and_coupling.py)

### 2.2 Proposal sampler V1

已经接入轻量版 weighted GMM proposal sampler。

关键文件：

- [llmbo/proposal.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/proposal.py)
- [tests/test_proposal_sampler.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/tests/test_proposal_sampler.py)

当前判断：

- proposal V1 是“已接通但未证明有效”
- 在真实 benchmark 下暂时没有稳定超越 plain EI

### 2.3 Acquisition prior 路线

实现过一条不改 GP posterior、只在 acquisition 上加 `prior bonus / risk penalty` 的路线。

关键文件：

- [llmbo/acquisition.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/acquisition.py)
- [tests/test_acquisition_prior.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/tests/test_acquisition_prior.py)

当前判断：

- 短预算实验里有希望
- 长预算 / 多 seed benchmark 里没有守住优势

## 3. 当前 Baseline 定义

### 3.1 严格 baseline

这是完全不使用 LLM 的版本：

- `n_warmstart = 0`
- `n_random_init = 6`
- `enable_iterative_guidance = false`
- `enable_gp_llm_coupling = false`
- `enable_acq_prior_coupling = false`
- `enable_proposal_sampler = false`

本质流程：

- random init
- `w_vec` 加权 Tchebycheff 标量化
- 单 GP 拟合 `f_w`
- plain EI

### 3.2 当前实际主 baseline

当前真正最重要的对照组是：

- `n_warmstart = 3`
- `n_random_init = 3`
- `enable_iterative_guidance = false`
- `enable_gp_llm_coupling = false`
- `enable_acq_prior_coupling = false`
- `enable_proposal_sampler = false`

也就是：

- `LLM warmstart + plain EI`

这是目前最稳、最强、最适合当主配置的基线。

## 4. 已验证的关键实验结论

### 4.1 warmstart 是有效的

`LLM warmstart + plain EI` 稳定优于严格 baseline。

参考报告：

- [optimized_experiments/ordinary_ei_comparison/report.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/ordinary_ei_comparison/report.json)

### 4.2 早期直接 GP mean coupling 经常拖后腿

历史消融已经显示：

- 问题不在普通 EI
- 问题主要在 `LLM -> GP mean coupling`

参考报告：

- [optimized_experiments/ei_mechanism_ablation/report.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/ei_mechanism_ablation/report.json)

### 4.3 proposal V1 没有证明有效

参考：

- [optimized_experiments/proposal_batch_v1/report.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/proposal_batch_v1/report.json)

结论：

- proposal V1 不适合当前直接升为主路径

### 4.4 acquisition prior 在 hard benchmark 中失败

参考：

- [optimized_experiments/acq_prior_comparison_v1/report.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/acq_prior_comparison_v1/report.json)
- [optimized_experiments/acq_prior_focus_v2/report.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/acq_prior_focus_v2/report.json)

最可信的 hard benchmark 结论：

- `warmstart_plain_ei`: mean HV `0.3644`
- `warmstart_old_gp_coupling`: mean HV `0.3616`
- `warmstart_acq_prior_proposal_light`: mean HV `0.3445`

对应结论：

- plain EI 仍最强
- old GP coupling 接近 plain EI，但未稳定超过
- acq prior 路线在长预算下失败

## 5. 最新工作：weight-aware gated GP coupling

针对一个核心问题做了新设计：

### 问题诊断

旧的 GP-LLM coupling 有两个明显缺陷：

1. LLM 虽然能看到 `w_vec`，但看到的是简化语义，而不是当前真正的标量化目标
2. LLM 的 `confidence` 被直接映射成 GP 均值偏移强度，耦合过硬

### 新设计

这轮已经实现：

1. `weight-aware guidance state`
2. `prompt` 升级，让 LLM 看见：
   - `w_vec`
   - 标量化公式
   - `eta`
   - `ideal_point / y_min / y_max`
   - 当前 `HV`
   - 最近 `HV delta`
   - 当前权重下最优协议
   - 相似权重下 guidance 历史效果
   - 边界失败统计
3. `gated GP coupling`
   - 不再直接裸用 `confidence`
   - 通过 `gate = align_score * history_score * hv_score * stage_score`
   - 只做局部 mask 下的弱 coupling

关键文件：

- [llmbo/optimizer.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/optimizer.py)
- [llmbo/gp_model.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/gp_model.py)
- [llm/llm_interface.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llm/llm_interface.py)
- [tests/test_weight_aware_guidance.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/tests/test_weight_aware_guidance.py)

## 6. 最新实验结果：weight-aware gated coupling

最新 focused benchmark 报告：

- [optimized_experiments/weight_aware_coupling_v1/report.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/weight_aware_coupling_v1/report.json)

实验口径：

- 真实 API
- `gpt-4.1-mini`
- `5` seeds
- `8` 轮 BO
- `3 warmstart + 3 random init`
- `temperature = 0`
- `llm_safe_dsoc_sum_max = 0.695`

结果：

- `warmstart_plain_ei_current`: mean HV `0.3666`
- `warmstart_weight_aware_gated_coupling`: mean HV `0.3634`

相对 plain：

- 平均差值：`-0.00321`
- 相对差：`-0.88%`
- 胜场：`1/5`

进一步观察：

- 新版 gated coupling 比历史 old coupling 更稳
- 但仍未稳定超过 plain EI
- 最后几轮 gate 普遍偏小，约 `0.036 ~ 0.115`
- 对应 effective lambda 也很小，说明系统已经学会“别过度相信 LLM”，但增益还没有释放出来

## 7. 当前最重要的技术判断

### 7.1 GP 学 `f_w`，最终用 HV 评估，本身不是 bug

这是 decomposition 多目标 BO 的标准思路：

- 每轮优化当前权重下的单目标 `f_w`
- 长期依靠多组权重扩展 Pareto front
- 最终用 HV 评价整体效果

所以：

- `GP target = f_w`
- `final metric = HV`

这个设定本身合理。

### 7.2 但当前系统确实存在目标错位

问题不在 GP 学 `f_w`，而在于：

- LLM 的 guidance 当前主要围绕“如何降低当前标量目标”
- 但缺少足够强的机制把“对 HV 是否真的有帮助”反馈回 coupling trust

当前已经开始做这件事，但还不够强。

### 7.3 当前 GP-LLM coupling 仍然不如 baseline 的主要原因

当前最可能的主因是：

1. `w_vec` 的真实优化含义对 LLM 来说仍然不够“数学可见”
2. `point/region -> grid/weights -> mean shift` 这条链仍然比较粗
3. 即使有 gate，历史信任与 HV 反馈的建模仍然偏弱
4. guidance 对 GP 的作用仍然是“改 posterior”，而不是更稳的“proposal / trust prior”

## 8. 当前主结论

一句话总结：

- **当前最优主配置仍然是 `LLM warmstart + plain EI`**
- `weight-aware gated GP coupling` 是新的次优研究线
- 它相对旧 coupling 有改善，但还没有稳定超过 plain EI

## 9. 建议 Gemini 优先审看的问题

建议 Gemini 重点分析下面几个问题：

1. 当前 `weight-aware guidance state` 是否已经足够表达当前 `f_w` 的真实优化对象
2. `gate = align * history * hv * stage` 是否过于保守，是否应该换成别的 trust 结构
3. `GP 学 f_w，HV 做最终指标` 这个错位，是否应该在权重调度或 coupling trust 层进一步补偿
4. `LLM -> GP` 是否仍然不应该走 posterior mean coupling，而应进一步退回到 proposal / ranking / trust prior
5. 当前 `similar_weight_guidance_success` 的定义是否合理，是否应该引入更细的 weight neighborhood 统计

## 10. 建议 Gemini 优先阅读的文件

最小必要文件包：

- [工作流分析.md](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/工作流分析.md)
- [对话承接总结.md](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/对话承接总结.md)
- [Gemini_工程现状报告.md](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/Gemini_工程现状报告.md)
- [llmbo/optimizer.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/optimizer.py)
- [llmbo/gp_model.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/gp_model.py)
- [llmbo/acquisition.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llmbo/acquisition.py)
- [llm/llm_interface.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/llm/llm_interface.py)
- [DataBase/database.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/DataBase/database.py)
- [tests/test_weight_aware_guidance.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/tests/test_weight_aware_guidance.py)

最关键实验结果：

- [optimized_experiments/acq_prior_focus_v2/report.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/acq_prior_focus_v2/report.json)
- [optimized_experiments/weight_aware_coupling_v1/report.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/weight_aware_coupling_v1/report.json)
- [optimized_experiments/ei_mechanism_ablation/report.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/ei_mechanism_ablation/report.json)

## 11. 当前推荐下一步

当前最推荐的两条后续方向：

1. 固化 `warmstart_plain_ei` 为标准实验配置
2. 继续专攻 `weight-aware gated coupling`，而不是重新铺很多新方法

如果继续做研究，我建议 Gemini 优先帮忙判断：

- 这条 `gated coupling` 应该继续强化
- 还是应该进一步退化为“LLM 只影响 proposal / trust prior，不再直接改 GP”

