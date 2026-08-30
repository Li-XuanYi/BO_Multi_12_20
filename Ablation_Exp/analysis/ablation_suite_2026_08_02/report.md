# 消融实验统一复核报告

本报告重新读取逐 seed `summary.json`，统一使用 canonical sHV 和 sample SD。
分析过程未调用优化器或 LLM API；实验数值来自已完成的真实归档运行。

## 核心结论

- 同批配置研究中，Warm 相对 Plain 的配对均值差为 0.00661（4/5 seeds 胜出）；Full 相对 Warm 为 -0.00024。Region-bearing arms 的 external EI restart 从 16 增至 32，且 Warm/Full 初始集未完全共享，因此这些数值是配置差异而非 lift 的独立因果效应。
- 共享 WarmStart 初始点后，加入整个 Region preset 的均值差为 -0.00067（3/5 seeds 胜出）。该批次确认了初始点完全一致，但仍存在 16 对 32 external restart 的搜索预算混杂，所以只能判定 Region 配置没有观察到稳定增益。
- 独立随机 seed 复验中，Warm 相对 Plain 的均值差变为 -0.01540；Region 相关运行累计 107 次 parse_fail，且有效 lift 为零。该组作为负结果/鲁棒性边界保留，不用于证明 Region 有效。
- 短预算 prompt 消融中，Experimental 相对 Random 的 canonical sHV 差为 0.05828（10/10），相对 Detailed 为 0.05519（9/10）；组内 Holm 校正 p 分别为 0.0059 和 0.0078。这支持 experimental prompt 在 Chen2020、26-evaluation 协议中的效果，但不能外推到 Region/lift 或 56-evaluation 主协议。

## 分组结果

### Same-batch component bundle

- 角色：Primary five-seed configuration ablation
- 来源：`Ablation_Exp/experiment_records/adaptive4_5seeds_50iter_deepseek_v3_2026_05_22/report_5seeds.json`
- 协议：56 evaluations/run；seeds=[8409, 8410, 8411, 8412, 8413]
- 完整性：通过

| Variant | n | canonical sHV (mean ± sample SD) | Pareto size | Feasible |
|---|---:|---:|---:|---:|
| Plain BO | 5 | 0.38364 ± 0.01128 | 45.20 | 100.0% |
| Warm start | 5 | 0.39024 ± 0.00870 | 44.20 | 100.0% |
| Region | 5 | 0.38621 ± 0.01250 | 45.20 | 100.0% |
| Full | 5 | 0.39000 ± 0.00665 | 44.40 | 100.0% |

| Paired contrast (lhs-rhs) | Mean Δ | 95% CI | W/T/L | Exact p | Holm p | Config confound |
|---|---:|---:|---:|---:|---:|---|
| warm_vs_plain | 0.00661 | [-0.00091, 0.01412] | 4/0/1 | 0.1250 | 0.6250 | none |
| region_vs_plain | 0.00258 | [-0.00981, 0.01496] | 4/0/1 | 0.5625 | 1.0000 | ei_n_external_restarts |
| full_vs_plain | 0.00637 | [-0.00497, 0.01770] | 3/0/2 | 0.2500 | 1.0000 | ei_n_external_restarts |
| full_vs_warm | -0.00024 | [-0.01192, 0.01144] | 3/0/2 | 1.0000 | 1.0000 | ei_n_external_restarts |
| full_vs_region | 0.00379 | [-0.00946, 0.01704] | 3/0/2 | 0.6250 | 1.0000 | none |

### Shared-initialisation Region increment

- 角色：Paired check of the Region preset after a shared warm start
- 来源：`Ablation_Exp/experiment_records/warmstart_vs_llmbo_paired_5seeds_50iter_deepseek_v3_2026_05_23/report_5seeds.json`
- 协议：56 evaluations/run；seeds=[8409, 8410, 8411, 8412, 8413]
- 完整性：通过

| Variant | n | canonical sHV (mean ± sample SD) | Pareto size | Feasible |
|---|---:|---:|---:|---:|
| Warm start | 5 | 0.39386 ± 0.00830 | 44.20 | 100.0% |
| Warm + Region | 5 | 0.39320 ± 0.01304 | 45.00 | 100.0% |

| Paired contrast (lhs-rhs) | Mean Δ | 95% CI | W/T/L | Exact p | Holm p | Config confound |
|---|---:|---:|---:|---:|---:|---|
| region_increment | -0.00067 | [-0.02211, 0.02078] | 3/0/2 | 1.0000 | 1.0000 | ei_n_external_restarts |

### Independent-seed robustness batch

- 角色：Robustness/failure-mode replication on independent seeds
- 来源：`Ablation_Exp/experiment_records/ablation_4way_5randomseeds_50iter_deepseek_v3_2026_05_14_180222_seeds_56702_53604_97885_98126_37310/report_5seeds.json`
- 协议：56 evaluations/run；seeds=[56702, 53604, 97885, 98126, 37310]
- 完整性：通过

| Variant | n | canonical sHV (mean ± sample SD) | Pareto size | Feasible |
|---|---:|---:|---:|---:|
| Plain BO | 5 | 0.39173 ± 0.00778 | 42.80 | 100.0% |
| Warm start | 5 | 0.37633 ± 0.01178 | 42.40 | 100.0% |
| Region | 5 | 0.39260 ± 0.00472 | 42.80 | 100.0% |
| Full | 5 | 0.37924 ± 0.00530 | 42.80 | 100.0% |

| Paired contrast (lhs-rhs) | Mean Δ | 95% CI | W/T/L | Exact p | Holm p | Config confound |
|---|---:|---:|---:|---:|---:|---|
| warm_vs_plain | -0.01540 | [-0.03785, 0.00704] | 1/0/4 | 0.1250 | 0.3750 | none |
| region_vs_plain | 0.00087 | [-0.00611, 0.00786] | 1/3/1 | 1.0000 | 1.0000 | ei_n_external_restarts |
| full_vs_plain | -0.01249 | [-0.02480, -0.00018] | 0/0/5 | 0.0625 | 0.3125 | ei_n_external_restarts |
| full_vs_warm | 0.00291 | [-0.00795, 0.01378] | 1/3/1 | 1.0000 | 1.0000 | ei_n_external_restarts |
| full_vs_region | -0.01336 | [-0.02049, -0.00623] | 0/0/5 | 0.0625 | 0.3125 | none |

### Warm-start prompt ablation

- 角色：Prompt-content ablation under the 26-evaluation batch protocol
- 来源：`experiment_records/prompt_comparison_v3_10seeds_10iter/report.json`
- 协议：26 evaluations/run；seeds=[8409, 8410, 8411, 8412, 8413, 8414, 8415, 8416, 8417, 8418]
- 完整性：通过

| Variant | n | canonical sHV (mean ± sample SD) | Pareto size | Feasible |
|---|---:|---:|---:|---:|
| Random init | 10 | 0.31062 ± 0.04303 | 17.40 | 100.0% |
| Detailed prompt | 10 | 0.31371 ± 0.04675 | 15.60 | 100.0% |
| Experimental prompt | 10 | 0.36890 ± 0.00824 | 17.10 | 100.0% |

| Paired contrast (lhs-rhs) | Mean Δ | 95% CI | W/T/L | Exact p | Holm p | Config confound |
|---|---:|---:|---:|---:|---:|---|
| detailed_vs_random | 0.00310 | [-0.03856, 0.04476] | 3/0/7 | 0.8750 | 0.8750 | none |
| experimental_vs_random | 0.05828 | [0.02642, 0.09015] | 10/0/0 | 0.0020 | 0.0059 | none |
| experimental_vs_detailed | 0.05519 | [0.02195, 0.08842] | 9/0/1 | 0.0039 | 0.0078 | none |

## 结论边界

1. 五 seed 组的 exact paired randomisation test 最小双侧 p 值为 0.0625，
   因此结果应作描述性配置比较，不作显著性或等效性声明。
2. 历史 Region arms 与对照的 external EI restart 预算不一致，nominal factorial contrast
   不能解释为 posterior lift 的独立因果效应。
3. Prompt 组是 6+20=26 evaluations 的短预算批协议；只能在该设置内解释。
4. 所有结果均为 Chen2020 仿真和未校准退化代理，不代表物理或实验室验证。

## 产物

- `report.json`：完整逐 seed 统计、配置审计与 telemetry
- `variant_summary.csv`：各变体汇总
- `paired_comparisons.csv`：配对差值、CI 与 exact/Holm p
- `ablation_suite.png` / `ablation_suite.pdf`：四组配对可视化
