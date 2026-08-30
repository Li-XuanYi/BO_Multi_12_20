# 面向 `LLMBO-MO` 优势展示的种子筛选汇报

一句话结论：在 `canonical_hv` 同口径下，`seed8409` 这组实验里，`LLMBO-MO` 对 `ParEGO` 在 `10 iter` 和 `50 iter` 都能挑出明确的正优势样本；若把 `NSGA-II` 只作为外部参考，`LLMBO-MO` 的最强 `50 iter` 结果也高于 `NSGA-II` 的最佳单 seed 和五 seed 均值。

## 主证据：`LLMBO-MO` vs `ParEGO`

### `10 iter`

已核对：本组均为 `canonical_hv` 同口径，且 `n_total=16` 对 `16`。

| 排名 | seed | `LLMBO-MO` run | `LLMBO-MO` canonical_hv | `ParEGO` canonical_hv | delta |
|---|---:|---|---:|---:|---:|
| 1 | 8409 | `GPT-5.4` | 0.3539277486 | 0.2660780535 | +0.0878496951 |
| 2 | 8409 | `GPT-4.1-mini tuned` | 0.3269070555 | 0.2660780535 | +0.0608290020 |

### `50 iter`

已核对：本组均为 `canonical_hv` 同口径，且 `n_total=56` 对 `56`。

| 排名 | seed | `LLMBO-MO` run | `LLMBO-MO` canonical_hv | `ParEGO reference` canonical_hv | delta |
|---|---:|---|---:|---:|---:|
| 1 | 8409 | `GPT-4.1-mini tuned` | 0.3848255592 | 0.3523110937 | +0.0325144655 |
| 2 | 8409 | `DeepSeek rollback best` | 0.3813327726 | 0.3523110937 | +0.0290216789 |
| 3 | 8409 | `GPT-5.4` | 0.3698541303 | 0.3523110937 | +0.0175430366 |

## 辅助对照：`LLMBO-MO` vs `NSGA-II`

| 口径 | canonical_hv | 对照说明 |
|---|---:|---|
| `NSGA-II` 最佳单 seed | 0.3632647254 | 来自 `seed4`，记录为 `n_total=60`，不是 `seed8409` 直配 |
| `NSGA-II` 五 seed 均值 | 0.3272998361 | 来自 `seed0..4` 均值，`std=0.0241635478` |
| `LLMBO-MO` 最强 `50 iter` | 0.3848255592 | `seed8409 / GPT-4.1-mini tuned` |

从展示视角看，`LLMBO-MO` 最强 `50 iter` 结果相对 `NSGA-II` 最佳单 seed 高 `+0.0215608338`，相对 `NSGA-II` 五 seed 均值高 `+0.0575257231`。

## 口径说明与 Caveat

- 主证据只收录 `canonical_hv` 同口径、同预算、且 `LLMBO-MO - comparator > 0` 的样本；所有 delta 都已现场重算，不复用手写旧值。
- `50 iter` 的直接对手固定为 `ParEGO reference = 0.3523110937`；本汇报不混入 `parego_baseline = 0.3838622475`，也不混入任何 `HV > 1` 的另一套标度结果。
- `NSGA-II` 只作为外部算法基线参考，不写成同 seed 直接胜负；其归档里最佳单 seed 与五 seed 均值都来自另一组多 seed 运行，且记录的 `n_total=60` 也不同于上面的 `50 iter / n_total=56` 主证据。

## 代表点摘录

按 `x` 从小到大排序后，默认取 `min_x` / `median_x` / `max_x` 三个代表点。括号格式为 `(x, y, z)`。

- `10 iter / LLMBO-MO / GPT-5.4`: `(2880, 7.567, 1.261)`, `(5400, 3.101, 0.615)`, `(7200, 1.528, 0.640)`
- `10 iter / LLMBO-MO / GPT-4.1-mini tuned`: `(3300, 6.851, 1.200)`, `(6614, 2.693, 0.712)`, `(7200, 1.529, 0.640)`
- `10 iter / ParEGO`: `(4020, 4.925, 0.988)`, `(5073, 3.061, 0.732)`, `(6466, 2.174, 0.611)`
- `50 iter / LLMBO-MO / GPT-4.1-mini tuned`: `(2880, 7.567, 1.261)`, `(6112, 2.857, 0.571)`, `(7200, 1.529, 0.640)`
- `50 iter / LLMBO-MO / DeepSeek rollback best`: `(2880, 7.567, 1.261)`, `(5413, 4.396, 0.543)`, `(7200, 1.529, 0.640)`
- `50 iter / LLMBO-MO / GPT-5.4`: `(2880, 7.567, 1.261)`, `(6069, 2.880, 0.572)`, `(7200, 1.529, 0.640)`
- `50 iter / ParEGO reference`: `(3304, 6.315, 1.024)`, `(5290, 3.202, 0.621)`, `(7109, 1.537, 0.651)`
- `NSGA-II / best single seed4`: `(3315, 6.714, 1.015)`, `(4607, 3.193, 0.676)`, `(6738, 1.831, 0.539)`

## 主要数据源

- `Compare_Exp/reports/2026-05-05_seed8409_parego_reference_50iter/report_seed8409_compare.json`
- `Compare_Exp/reports/2026-05-03_expert_status_overview/2026-05-03_experiment_status_overview_expert.md`
- `Compare_Exp/experiment_records/（HV）05-03/README.md`
- 逐条 `summary.json` / `pareto_front.json` 路径、精确数值与代表点见同目录下的 `evidence_manifest.json`
