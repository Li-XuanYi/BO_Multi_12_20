# Ablation 2026-05-23: 4-Group Summary

This folder archives the consolidated 5-seed, 50-iteration ablation table used for the current LLMBO-MO comparison.

## Consolidated Results

| Group | Mean canonical HV | vs Baseline | Wins vs Baseline |
|---|---:|---:|---:|
| Baseline | 0.383635 | - | - |
| WarmStart | 0.390242 | +0.006607 | 4/5 |
| LLM_Region | 0.386211 | +0.002576 | 4/5 |
| LLMBO = WarmStart+LLM_Region | 0.393196 | +0.009561 | 3/5 |

## Source Notes

- Baseline, WarmStart, and LLM_Region are from:
  `Ablation_Exp/experiment_records/adaptive4_5seeds_50iter_deepseek_v3_2026_05_22/report_5seeds.json`
- LLMBO is from the paired rerun:
  `Ablation_Exp/experiment_records/warmstart_vs_llmbo_paired_5seeds_50iter_deepseek_v3_2026_05_23/report_5seeds.json`
- In the paired rerun, LLMBO used the same selected WarmStart initialization as the paired WarmStart baseline.

## Files

- `combined_4group_results.csv`: table for spreadsheets.
- `combined_4group_results.json`: machine-readable table with source metadata.
- `source_reports/`: copied original report JSON files.
- `images/`: copied plots from the source experiments.
- `images/summary_compare_style/`: newly generated Compare-style summary figures.

## Compare-Style Figures

- `images/summary_compare_style/ablation523_4group_canonical_hv_compare_style.png`
- `images/summary_compare_style/ablation523_4group_canonical_hv_compare_style.pdf`
- `images/summary_compare_style/ablation523_4group_delta_vs_baseline_compare_style.png`
- `images/summary_compare_style/ablation523_4group_delta_vs_baseline_compare_style.pdf`
- `images/ablation_canonical_hv_box.png`
- `images/ablation_canonical_hv_box.pdf`
- `images/ablation_hv_convergence.png`
- `images/ablation_hv_convergence.pdf`

The figures use a Compare_Exp-like style: serif font, gray axes, light grid, solid colors, seed-level scatter points, and PNG/PDF outputs.
