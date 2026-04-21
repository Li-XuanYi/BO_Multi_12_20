# Unified Data Catalog

这个目录是对仓库实验结果的统一索引，不改动原始 `summary.json / report.json / database.json` 文件。

## 文件说明

- `run_manifest.csv/json`: 每个 `summary.json` 一行，适合按 seed、HV、配置字段做筛选。
- `report_group_manifest.csv/json`: 每个 `report.json` 里的实验组聚合结果一行。
- `report_comparison_manifest.csv/json`: 每个 `report.json` 里的 comparison 项一行。
- `inventory_summary.json`: 全仓库结果文件数量、来源目录和实验族统计。

## 当前规模

- 运行级条目: `166`
- 报告组条目: `64`
- 对比条目: `30`

## 推荐用法

- 先看 `run_manifest.csv`，按 `experiment_family / run_name / w_sample_seed / hypervolume` 过滤。
- 要看均值结果时看 `report_group_manifest.csv`。
- 要查历史对比结论时看 `report_comparison_manifest.csv`。
- 原始细节仍回跳到对应 `summary_path / database_path / pareto_front_path`。

## 最近索引到的部分运行

- `optimized_experiments/weight_aware_coupling_v1/warmstart_plain_ei_current_seed4`: HV=0.37553799484636147, seed=4, summary=`optimized_experiments/weight_aware_coupling_v1/warmstart_plain_ei_current_seed4/summary.json`
- `optimized_experiments/weight_aware_coupling_v1/warmstart_weight_aware_gated_coupling_seed0`: HV=0.3616006971504333, seed=0, summary=`optimized_experiments/weight_aware_coupling_v1/warmstart_weight_aware_gated_coupling_seed0/summary.json`
- `optimized_experiments/weight_aware_coupling_v1/warmstart_weight_aware_gated_coupling_seed1`: HV=0.3694953522755169, seed=1, summary=`optimized_experiments/weight_aware_coupling_v1/warmstart_weight_aware_gated_coupling_seed1/summary.json`
- `optimized_experiments/weight_aware_coupling_v1/warmstart_weight_aware_gated_coupling_seed2`: HV=0.348229083677859, seed=2, summary=`optimized_experiments/weight_aware_coupling_v1/warmstart_weight_aware_gated_coupling_seed2/summary.json`
- `optimized_experiments/weight_aware_coupling_v1/warmstart_weight_aware_gated_coupling_seed3`: HV=0.3647122809902613, seed=3, summary=`optimized_experiments/weight_aware_coupling_v1/warmstart_weight_aware_gated_coupling_seed3/summary.json`
- `optimized_experiments/weight_aware_coupling_v1/warmstart_weight_aware_gated_coupling_seed4`: HV=0.37300756212331276, seed=4, summary=`optimized_experiments/weight_aware_coupling_v1/warmstart_weight_aware_gated_coupling_seed4/summary.json`
- `optimized_experiments/weight_aware_smoke/weight_aware_smoke`: HV=0.33560970049583727, seed=0, summary=`optimized_experiments/weight_aware_smoke/summary.json`
- `results/results/results`: HV=0.36708103407358683, seed=0, summary=`results/summary.json`

## 字段约定

- `experiment_family`: 一级实验族目录，例如 `llmei_vs_plain_v1`。
- `variant_group`: 更深层的分组目录；没有时为空。
- `run_name`: 最终运行目录名。
- `init_hv`: `warmstart_trace` 的最后一个超体积值。
- `hv_violations`: `hv_trace` 中超体积下降次数，正常应为 0。
