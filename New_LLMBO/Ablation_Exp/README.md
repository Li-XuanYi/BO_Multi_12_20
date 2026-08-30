# Ablation_Exp

This folder is a curated archive for the three-way ablation experiments:

- `LLMGP+WarmStart` = `warmstart_region_lifted_gp`
- `WarmStart` = `warmstart_plain_ei`
- `Baseline` = `strict_baseline`

The files here were copied from the original experiment tree for easier review.
The original directories under `optimized_experiments/` and `fixed_experiments/`
were not deleted or overwritten.

## Structure

- `Fig/`
  - Archived experiment figures.
  - Currently includes the fixed `seed8409` profile figures.
- `Report/`
  - Top-level `report*.json` files and related logs for the selected 50-iteration
    ablation experiment packages.
  - Also includes `fixed_experiments/report.json` and the fixed `seed8409`
    profile manifests.
- `Raw_JSON/`
  - Raw JSON artifacts copied from the selected experiment packages, including
    `summary.json`, `database.json`, `db_final.json`, `pareto_front.json`, and
    other JSON snapshots preserved under their source package names.
- `Process/`
  - Experiment and post-processing scripts related to the ablation workflow.

## Included 50-Iteration Packages

- `optimized_experiments/baseline_warmstart_llmgp_50iter_seed01234_2026_04_29_run1`
- `optimized_experiments/region_lift_50iter_seed01234_2026_04_29`
- `optimized_experiments/region_lift_v2_50iter_seed01234_2026_04_29`
- `optimized_experiments/region_lift_fix_seed8409_50iter_2026_05_01`
- `optimized_experiments/region_lift_seed8409_50iter_2026_05_01`

## Included Fixed Snapshots

- `fixed_experiments/fixed_no_llm_baseline`
- `fixed_experiments/fixed_real_api_warmstart_only`
- `fixed_experiments/fixed_real_api_fullcoupled`
- `fixed_experiments/fixed_seed8409_llmgp_winner_2026_05_01`

## Included Process Scripts

- `tools/run_warmstart_vs_baseline.py`
- `tools/run_region_lift_v2_50iter.py`
- `tools/freeze_seed8409_and_plot_profiles.py`

## Missing From Current Workspace

The IDE tabs referenced these paths, but they are not present in the current
workspace snapshot, so they were not archived this round:

- `figures/auc_boxplot/hv_nauc_values.csv`
- `scripts/plot_hv_auc_boxplot.py`
