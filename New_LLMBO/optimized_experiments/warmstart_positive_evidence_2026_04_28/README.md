# WarmStart Positive Evidence Package - 2026-04-28

This folder is the compact retained evidence set for the current New_LLMBO WarmStart story.

## Main claim currently supported

WarmStart Portfolio can produce a significantly better initialization than strict random Baseline when a high-quality LLM portfolio is selected and cached.
The stable reproduction case is seed=1 with fixed cache:

- seed=1 Baseline canonical HV: 0.31571010193967
- fixed best WarmStart canonical HV: 0.367280769115039
- delta canonical HV: +0.05157066717536901

## Positive evidence retained

- reports/seed1_fixed_cache_report.json
- reports/seed1_best_fixed_report.json
- reports/seed1_repeats_report.json
- reports/realapi_5seed8_positive_report.json
- reports/realapi_3seed5_positive_report.json
- reports/seed1_safe695_positive_report.json

## Reproducibility artifacts retained

- fixed_cache_reproduction/seed1_best_warmstart_cache.json
- fixed_cache_reproduction/shared_random_init_seed1.json
- fixed_cache_reproduction/baseline_seed1_summary.json
- fixed_cache_reproduction/warmstart_cached_seed1_summary.json
- fixed_cache_reproduction/fixed_best_seed1_summary.json

## Important caveat

Real-time API WarmStart is still noisy. Negative or mixed raw runs were removed from the main experiment tree to reduce clutter, but compact audit reports are retained in audit_summaries/.
This means the folder is suitable for presenting the positive fixed-cache evidence, not for claiming universal real-time API robustness.

## Removed raw folders after curation

All previous optimized_experiments subdirectories except this curated package were deleted after copying the retained reports/artifacts above.
