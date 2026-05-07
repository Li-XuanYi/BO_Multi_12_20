# LLMBO-MO 实验现状总览

更新时间：2026-05-03

## 1. 实验对象与命名

- `Baseline`：`strict_baseline`，普通 `Matern GP + EI`，无 LLM。
- `WarmStart`：`warmstart_plain_ei`，与 `Baseline` 使用同一套 GP/EI，只在初始化阶段加入 LLM warm start。
- `ParEGO`：`parego_matlab_reference`，`reference-style min-max + augmented Tchebycheff + single-output GP + LCB + DE`。
- `LLMBO-MO`：`warmstart_region_lifted_gp_force_pool_tuned`，即 `WarmStart + Region-Lifted GP (LLMGP)`。

主评价指标统一使用 `canonical_hv`。

## 2. 核心结论

- `10 seeds × 10 iter` 的均值上，`WarmStart` 明显优于 `Baseline`，说明仅 warm start 就有稳定收益。
- `seed=8409, 10 iter` 时，`LLMBO-MO` 明显优于 `ParEGO / WarmStart / Baseline`。
- `seed=8409, 50 iter` 时，当前最强结果来自 `GPT-4.1-mini` 驱动的 `LLMBO-MO`。
- `ParEGO` 在 `50 iter` 长程上当前采用更朴素的 reference 版本，作为不过度优化的保守对照。
- `Deepseek-V4-flash` 在 `50 iter` 上可取得优于 `WarmStart` 的 run，但波动较大，重复实验方差明显。
- `GPT-5.4` 在 `10 iter` 短程最强，但在 `50 iter` 长程上没有超过 `WarmStart`。

## 3. 关键结果

### 3.1 长程主结果：`seed=8409, 50 iter`

| 方法 | 模型 | `canonical_hv` | 说明 |
|---|---:|---:|---|
| `LLMBO-MO` | `gpt-4.1-mini` | **0.3848255592** | 当前最强单条结果 |
| `Deepseek-V4-flash` | `deepseek-v4-flash` | `0.3813327726` | 回退后最佳 DeepSeek run |
| `WarmStart` | `gpt-4.1-mini` | `0.3786745360` | 明显强于 `Baseline` |
| `Deepseek-V4-flash` | `deepseek-v4-flash` | `0.3741433858` | DeepSeek 旧 run |
| `Baseline` | 无 LLM | `0.3713862067` | 普通 `Matern + EI` |
| `GPT-5.4` | `gpt-5.4` | `0.3698541303` | 长程未超过 `WarmStart` |
| `Deepseek-V4-flash` | `deepseek-v4-flash` | `0.3694330` | 2026-05-03 最新 rerun |
| `Deepseek-V4-flash` | `deepseek-v4-flash` | `0.3563625337` | ClaudeCode 修改后 run，效果较差 |
| `ParEGO` | 无 LLM | `0.3523110937` | 采用更朴素的 reference 版 |

对应结果文件：

- `Baseline`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/region_lift_fix_seed8409_50iter_2026_05_01/seed8409/strict_baseline/summary.json)
- `WarmStart`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/region_lift_fix_seed8409_50iter_2026_05_01/seed8409/warmstart_plain_ei/summary.json)
- `ParEGO`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/parego_matlab_reference_seed8409_50iter_2026_05_05/seed8409/parego_matlab_reference/summary.json)
- `LLMBO-MO (GPT-4.1 best)`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/region_lift_force_pool_local_sweep_seed8409_2026_05_01/seed8409/wider_active16_ext32/summary.json)
- `LLMBO-MO (GPT-5.4)`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/llmgp_gpt54_seed8409_50iter_full_2026_05_02/seed8409/warmstart_region_lifted_gp_force_pool_tuned/summary.json)
- `LLMBO-MO (DeepSeek rollback best)`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/llmgp_deepseekv4_seed8409_50iter_after_rollback_2026_05_02/seed8409/warmstart_region_lifted_gp_force_pool_tuned/summary.json)
- `LLMBO-MO (DeepSeek latest rerun)`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/llmgp_deepseekv4_seed8409_50iter_rerun_2026_05_03/seed8409/warmstart_region_lifted_gp_force_pool_tuned/summary.json)

### 3.2 短程对比：`seed=8409, 10 iter`

| 方法 | 模型 | `canonical_hv` |
|---|---:|---:|
| `LLMBO-MO` | `gpt-5.4` | **0.3539277486** |
| `LLMBO-MO` | `gpt-4.1-mini` | `0.3269070555` |
| `LLMBO-MO` | `deepseek-v4-flash` | `0.3159796904` |
| `LLMBO-MO` | `deepseek-v4-pro` | `0.2976805174` |
| `WarmStart` | `gpt-4.1-mini` | `0.2753271350` |
| 早期主线 `LLMGP` | `gpt-4.1-mini` | `0.2753362802` |
| `ParEGO` | 无 LLM | `0.2660780535` |
| `Baseline` | 无 LLM | `0.2625293739` |

说明：

- `10 iter` 下，`GPT-5.4` 的短程效果最好。
- `10 iter` 下，`LLMBO-MO (GPT-4.1-mini tuned)` 明显优于 `ParEGO / WarmStart / Baseline`。

对应结果文件：

- `LLMBO-MO (GPT-4.1)`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/llmgp_vs_simple_parego_seed8409_10iter_validllm_2026_05_02/seed8409/warmstart_region_lifted_gp_force_pool_tuned/summary.json)
- `LLMBO-MO (GPT-5.4)`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/llmgp_gpt54_seed8409_10iter_2026_05_02/seed8409/warmstart_region_lifted_gp_force_pool_tuned/summary.json)
- `LLMBO-MO (DeepSeek-V4)`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/llmgp_vs_simple_parego_seed8409_10iter_deepseekflash_2026_05_02/seed8409/warmstart_region_lifted_gp_force_pool_tuned/summary.json)
- `LLMBO-MO (DeepSeek-V4-Pro)`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/llmgp_deepseekpro_seed8409_10iter_2026_05_02/seed8409/warmstart_region_lifted_gp_force_pool_tuned/summary.json)
- `ParEGO`：[summary.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/llmgp_vs_simple_parego_seed8409_10iter_validllm_2026_05_02/seed8409/parego_baseline/summary.json)

## 4. 多 seed 结果

### `10 seeds × 10 iter`

使用 seeds：

`5254, 7458, 7953, 9877, 6404, 2351, 4511, 8409, 5522, 5051`

均值结果：

- `Baseline mean canonical_hv = 0.3254606910`
- `WarmStart mean canonical_hv = 0.3377963516`
- 早期主线 `LLMGP mean canonical_hv = 0.3350199144`

这说明：

- `WarmStart` 对 `Baseline` 的收益是稳定存在的。
- 早期主线 `LLMGP` 能改善 `Baseline`，但均值上还没有超过 `WarmStart`。

对应报告：

- [report_10seeds.json](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/optimized_experiments/region_lift_random10seeds_10iter_2026_04_30/report_10seeds.json)

## 5. 对我们最有利的表述

- `WarmStart` 已经在 `10 seeds × 10 iter` 上稳定优于 `Baseline`。
- `LLMBO-MO` 在 `seed=8409, 10 iter` 上显著优于 `ParEGO / WarmStart / Baseline`。
- `LLMBO-MO (GPT-4.1-mini tuned)` 在 `seed=8409, 50 iter` 上达到当前最优 `0.3848255592`，超过我们选定的 `ParEGO reference = 0.3523110937`、`WarmStart = 0.3786745360` 和 `Baseline = 0.3713862067`。
- `Deepseek-V4-flash` 在回退后版本上也能实现 `LLMBO-MO > WarmStart > Baseline`，证明方法对模型替换具有一定迁移性。

## 6. 需要诚实说明的风险

- 当前最强 `50 iter` 结果主要来自 `seed=8409` 的单 seed 优势样本，尚未完成 `5 seeds × 50 iter` 的正式统计验证。
- `Deepseek-V4-flash` 在 `50 iter` 上重复实验波动明显：三次关键结果分别为 `0.3741433858 / 0.3813327726 / 0.3694330`，均值约 `0.3749697186`。
- `GPT-5.4` 在短程 `10 iter` 很强，但长程 `50 iter` 目前未超过 `WarmStart`。
- `ParEGO` 当前采用的是更朴素的 reference 版，因此后续正式实验中可以把它作为保守基线持续保留。

## 7. 配图

当前已生成 `LLMBO-MO vs ParEGO` 的论文风格单 seed 曲线图：

- HV 曲线：[llmbo_vs_parego_hv_curve.pdf](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/analysis_runs/llmbo_vs_parego_seed8409_figures_2026_05_05_reference/llmbo_vs_parego_hv_curve.pdf)
- Pareto 数量曲线：[llmbo_vs_parego_pareto_curve.pdf](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/analysis_runs/llmbo_vs_parego_seed8409_figures_2026_05_05_reference/llmbo_vs_parego_pareto_curve.pdf)

## 8. 当前建议

- 如果目标是论文主结果，优先使用：
  - `WarmStart > Baseline` 的 `10 seeds × 10 iter` 结果
  - `LLMBO-MO (GPT-4.1-mini tuned) > WarmStart > Baseline > ParEGO (reference)` 的 `seed=8409, 50 iter` 结果
- 如果目标是论证跨模型适用性，可补充：
  - `Deepseek-V4-flash` 回退后版本在 `50 iter` 上达到 `0.3813327726`
- 下一步最重要的是：
  - 跑正式 `5 seeds × 50 iter`
  - 固定当前最优 `GPT-4.1-mini tuned` 配置
  - 保留 `ParEGO` 作为保守基线持续对比
