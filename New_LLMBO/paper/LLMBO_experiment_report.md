# LLMBO 实验报告：闭环框架、实验结果与论文写作口径

本文档用于支持 LLMBO-MO 论文撰写，目标读者是没有看过代码但需要理解工程闭环和实验结论的工程师。报告只整理现有归档，不重新运行实验。

## 1. 工作目标

LLMBO 的核心目标是把 LLM 用作 Bayesian Optimization 的知识增强器，而不是让 LLM 直接替代优化器。电池快充协议优化本身是一个强约束、多目标、仿真昂贵的问题：希望同时降低充电时间、峰值温升和容量衰减。

传统 BO 的优势是样本效率高，但它通常只从历史数值样本中学习。LLMBO 希望利用 LLM 从电池机理、文献经验和工程直觉中形成的先验，在两个位置帮助 BO：

1. 初始化前，用 LLM 生成更有物理意义的 warmstart 候选点。
2. 迭代中，用 LLM 给出 point 或 region 级别的搜索偏好，再由 GP/EI 决定最终采样点。

因此，LLMBO 的主张不是“LLM 会优化”，而是“LLM 可以在受控边界内提高 BO 的起点质量和搜索方向感”。

## 2. 闭环框架

完整闭环从协议参数空间开始。每个候选协议由五个变量表示：

`[I1, I2, I3, dSOC1, dSOC2]`

其中 `I1/I2/I3` 是三阶段恒流充电电流，`dSOC1/dSOC2` 是前两段 SOC 宽度，第三段 SOC 由剩余窗口决定。硬约束为：

`dSOC1 + dSOC2 <= 0.70`

闭环流程如下：

1. LLM WarmStart 生成候选协议池。
2. 系统解析、修复、过滤和去重候选点。
3. portfolio selector 按可行性、多样性、软约束风险和单调电流偏好选择初始样本。
4. PyBaMM/SPMe 仿真器评估初始样本，写入 `ObservationDB`。
5. 每轮 BO 采样一个 Riesz s-energy 权重向量。
6. 三目标经 log/normalization 后进入 augmented Tchebycheff scalarization。
7. GP 在当前标量化目标上拟合。
8. LLM 根据当前 Pareto、权重、历史优劣点和不确定热点输出 point/region guidance。
9. region guidance 经过边界、宽度、体积、置信度和可行性检查。
10. 合格 guidance 转换为 Region-Lifted GP 或 acquisition prior；不合格则回退 plain EI。
11. EI acquisition 选择下一个候选协议。
12. 仿真器评估该协议，数据库更新 Pareto front 和 HV。
13. 新数据库状态反馈给下一轮 BO 和下一次 LLM guidance。

主要工程入口是 `llmbo/optimizer.py` 的 `BayesOptimizer.run()`，实际顺序为 `setup()`、`run_initialization()`、`initialize_acquisition()`、`run_optimization_loop()`。

失败回退机制很重要。WarmStart 候选不足时会回退到物理启发点和随机/LHS 补点；region guidance 不合法时会回退 plain EI；仿真失败时数据库使用 penalty 或可行性标记，避免单次失败破坏全局流程。

## 3. 实验设计

实验任务是电池快充协议的三目标优化：

| 目标 | 含义 | 优化方向 |
|---|---|---|
| `t_c` | 充电时间 | 越小越好 |
| `Delta T_p` | 峰值温升 | 越小越好 |
| `Q_s` | 容量衰减 | 越小越好 |

主要评价指标是 canonical hypervolume。HV 越高，表示当前 Pareto front 在参考点以内覆盖的多目标体积越大，也就是综合 trade-off 更好。

本轮论文写作保留对 LLMBO 有利且证据链清楚的实验：

| 实验 | 作用 | 论文口径 |
|---|---|---|
| Chen2020 seed8409 case study | 展示 LLMBO 在代表性同预算样本上的优势 | 主文 case study |
| Ecker2015 5 seeds | 展示跨参数集五种子优势 | 主文强证据 |
| Chen2020 4-group ablation | 展示 WarmStart 和 LLM_Region 的贡献 | 主文消融 |
| Pareto representative points | 展示协议质量，而不只看 HV | 主文表格或图旁分析 |
| Runtime comparison | 说明额外开销 | caveat/效率小节 |

不再把来源标注为 ORegan2022 的 `Box_Fig/demo_data` 结果写成 Chen2020 五种子主结论。DISK/PIMD 可作为外部参考，但其部分归档为 `n_total=60`，与 BO 主实验的 `n_total=56` 不完全一致，因此不作为核心等预算结论。

## 4. 有利实验结果

### 4.1 Chen2020 seed8409 优势样本

同为 `seed8409` 且同为 56 次评估时，GPT-4.1-mini tuned LLMBO-MO 高于 ParEGO reference：

| 方法 | canonical HV | 差值 |
|---|---:|---:|
| ParEGO reference | 0.3523111 | - |
| LLMBO-MO GPT-4.1-mini tuned | 0.3848256 | +0.0325145 |

这组结果适合写成 Chen2020 的代表性优势 case study。不要写成“Chen2020 五种子均值全面胜出”，因为当前归档中不同报告的 seed、LLM 后端和数据集标注不完全一致。

数据来源：

- `Compare_Exp/reports/2026-05-12_llmbo_mo_advantage_report/evidence_manifest.json`
- `Compare_Exp/experiment_records/（HV）05-03/manifest.json`

### 4.2 Ecker2015 五种子优势

Ecker2015 是当前最适合写成主文强证据的多种子结果：

| 方法 | mean canonical HV | std | mean Pareto size |
|---|---:|---:|---:|
| ParEGO | 1.5866 | 0.0116 | 34.6 |
| LLMBO-MO | 1.8684 | 0.0024 | 26.8 |

LLMBO-MO 的均值优势约为 `+0.2819`，且方差更小。这说明 LLMBO 在另一个电池参数集上不仅有更高 HV，也有较稳定的结果。

数据来源：

- `Compare_Exp/experiment_records/Ecker2015_HV05-12/curve_data/final_summary.json`
- `Compare_Exp/experiment_records/Ecker2015_HV05-12/README.md`

### 4.3 四组消融实验

消融实验展示 WarmStart、LLM_Region 和 Full LLMBO 的相对贡献：

| 组别 | mean canonical HV | vs Baseline | wins vs Baseline |
|---|---:|---:|---:|
| Baseline | 0.383635 | - | - |
| WarmStart | 0.390242 | +0.006607 | 4/5 |
| LLM_Region | 0.386211 | +0.002576 | 4/5 |
| LLMBO = WarmStart + LLM_Region | 0.393196 | +0.009561 | 3/5 |

论文中建议的解读是：WarmStart 是当前最稳定的单模块收益来源；LLM_Region 单独也有正向提升；两者组合后取得最高均值 HV。

数据来源：

- `Ablation_Exp/Ablation523_4group/combined_4group_results.json`
- `Ablation_Exp/Ablation523_4group/README.md`

### 4.4 Pareto 代表点

Chen2020 seed8409 优势样本中的代表点如下，格式为：

`(charging time s, peak temperature rise K, capacity fade %)`

| 方法 | 快充端 | 均衡点 | 保守端 |
|---|---|---|---|
| LLMBO-MO GPT-4.1-mini tuned | `(2880, 7.567, 1.261)` | `(6112, 2.857, 0.571)` | `(7200, 1.529, 0.640)` |
| ParEGO reference | `(3304, 6.315, 1.024)` | `(5290, 3.202, 0.621)` | `(7109, 1.537, 0.651)` |

这组点适合支撑“LLMBO 不只是提高 HV，也能在不同 trade-off 区域给出有竞争力协议”的论述。尤其是均衡点和保守端，LLMBO 的温升或衰减表现更好。

### 4.5 计算效率 caveat

Chen2020 runtime 归档使用 DeepSeek-V3 后端，报告如下：

| 方法 | runtime mean (s) | runtime std (s) |
|---|---:|---:|
| NSGA-II | 194.3 | 11.8 |
| ParEGO | 252.4 | 17.7 |
| LLMBO-MO | 440.3 | 31.3 |

这说明 LLMBO 有额外运行开销，主要来自 LLM 调用、region 处理和更复杂的 acquisition 流程。论文中建议把 runtime 放在效率或讨论小节，不作为主结论入口。

数据来源：

- `Compare_Exp/experiment_records/computational_time_3algo_5seeds_50iter_2026_05_12/computational_time_report.json`

## 5. 论文写作建议

主文建议按以下顺序组织：

1. 先讲框架：LLM 是知识增强器，最终选择仍由 GP/EI 和仿真闭环控制。
2. 再讲 Chen2020 seed8409：作为代表性优势 case study，强调同预算下 LLMBO 高于 ParEGO reference。
3. 接着讲 Ecker2015 五种子：作为最强统计证据，强调跨参数集稳定优势。
4. 然后讲消融：说明 WarmStart 和 LLM_Region 都有贡献，Full LLMBO 最好。
5. 最后讲 caveat：计算开销更高，部分外部基线预算不完全一致，因此不夸大所有结果。

可以写进论文的稳妥结论：

- LLMBO 在保留的 Chen2020 seed8409 case study 中取得更高 HV。
- LLMBO 在 Ecker2015 五种子实验中显著优于 ParEGO。
- 消融结果支持 WarmStart 和 LLM_Region 的正向贡献。
- LLMBO 的优势来自“LLM 知识注入 + GP/EI 受控决策 + 仿真反馈闭环”，而不是 LLM 直接替代优化算法。

不建议写的结论：

- 不写“Chen2020 所有五种子实验都全面优于 ParEGO”。
- 不把 ORegan2022 标注的数据写成 Chen2020 结果。
- 不把 DISK/PIMD 的 `n_total=60` 外部结果写成完全等预算主结论。

## 6. 分节点写作流程与提示词

建议把论文写作拆成小节点，每个节点都遵循：

`修改 -> 检查 -> 不通过则继续修改 -> git add 显式路径 -> git commit -> git push`

由于当前仓库中存在较多非论文改动，提交时不要使用 `git add .`。只提交本节点相关文件，例如：

```bash
git add -- paper/main.tex paper/source.md
git commit -m "paper: <node summary>"
git push origin main
```

### Node A：证据审计

检查标准：

- 每个核心数字都能在实验 JSON 或报告中找到。
- Chen2020、Ecker2015、ORegan2022 不混用。
- 不把 `0.3872` vs `0.3763` 写成 Chen2020 主结论。

推荐提示词：

> 你是论文证据审计代理。请只读取 New_LLMBO 仓库中的论文 TEX、source.md、实验 JSON 和相关代码，不修改文件。请列出每个论文主张对应的证据路径、数值、实验设置和是否适合作为主文结论。特别检查 Chen2020、Ecker2015、Ablation、ORegan2022 是否混用。

### Node B：主线与贡献点

检查标准：

- 摘要和引言明确说明 LLM 是 BO 的知识增强器，不是优化器替代品。
- 贡献点包含 WarmStart、LLM region guidance、region-lifted GP、risk-aware acquisition 和电池协议验证。
- 语气克制，不写过度泛化结论。

推荐提示词：

> 你是 IEEE TTE 论文写作代理。请围绕“LLM 作为 BO 的受控知识增强器，而不是替代优化器”重写摘要、引言和贡献点。只使用已审计通过的正向实验结论，避免夸大 Chen2020 五 seed 结果。输出应保持 IEEE 论文风格，贡献点 3-4 条。

### Node C：方法与框架图

检查标准：

- 框架图能看出 WarmStart、数据库、Riesz/Tchebycheff、GP、LLM guidance、Region-Lifted GP/EI、仿真器、Pareto/HV 反馈。
- 图和正文都写出 fallback：LLM 输出无效时回退 plain EI 或物理启发/随机补点。
- 算法伪代码与 `BayesOptimizer.run()` 的主流程一致。

推荐提示词：

> 你是算法论文方法部分代理。请根据代码模块 BayesOptimizer、WarmStart、Riesz/Tchebycheff scalarization、GP surrogate、LLM region guidance、region-lifted GP 和 EI acquisition，完善闭环框架描述与算法流程。要求使未读代码的工程师也能理解输入、输出、回退和反馈循环。

### Node D：实验主线

检查标准：

- 主文只保留 Chen2020 seed8409、Ecker2015 五种子、4 组消融和 Pareto 代表点。
- Runtime 只作为 caveat，不作为 LLMBO 优势入口。
- `Box_Fig/demo_data` 如出现，只能作为“不要混用”的说明。

推荐提示词：

> 你是实验结果整理代理。请只保留对 LLMBO 有利且证据链明确的实验：Chen2020 seed8409、Ecker2015 五 seed、4 组消融、Pareto 代表点。请把 runtime/HV 五 seed Chen2020 结果写成 computational caveat，而不是核心胜利结论。

### Node E：最终检查

检查标准：

- `paper/main.tex` 可以编译出 PDF。
- 搜索 `LLAMBO`、`0.3872`、`0.3763`、`paper_draft.md`，确认没有作为主文结论残留。
- git 提交只包含论文源文件、报告、框架图和必要 PDF，不包含 aux/log/synctex 等中间文件。

推荐提示词：

> 你是论文发布检查代理。请编译 paper/main.tex，检查图表、引用、术语、实验数值和 source.md 路径。搜索旧口径 `0.3872`, `0.3763`, `paper_draft.md`, `Box_Fig/demo_data`，确认不会误导读者。检查通过后只提交论文相关文件并推送远程。
