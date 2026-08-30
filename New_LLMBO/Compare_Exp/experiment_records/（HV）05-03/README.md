# HV 05-03 data archive

Source figure: `Compare_Exp/images/(HV)05-03_seed8409/hv_convergence_5way.png`
Generation script: `Compare_Exp/Exp/plot_hv_convergence_5way.py`

## Five plotted line sources

| Line | Source used in the plot | Archived raw copy | Curve data |
|---|---|---|---|
| ParEGO | `optimized_experiments/parego_matlab_reference_seed8409_50iter_2026_05_05/seed8409/parego_matlab_reference/summary.json` -> `hv_trace[].canonical_hv` | `raw_sources/main_curves/ParEGO` | `curve_data/parego_curve.csv` |
| LLAMBO-MO | `optimized_experiments/region_lift_force_pool_local_sweep_seed8409_2026_05_01/seed8409/wider_active16_ext32/summary.json` -> `hv_trace[].canonical_hv` | `raw_sources/main_curves/LLAMBO-MO` | `curve_data/llambo_mo_curve.csv` |
| NSGA-II | `optimized_experiments/nsga2_5seeds_56evals_2026_05_07/seed0..seed4/nsga2/summary.json`, mean/std across five seeds | `raw_sources/multiseed/NSGA-II` | `curve_data/nsga2_curve.csv` |
| DISK | `Compare_Exp/experiment_records/disk_python_Chen2020_5seeds_50evals_2026_05_11/seed8409..seed8413/disk_Chen2020/database.json`, canonical HV recomputed over database prefixes, mean/std across five seeds | `raw_sources/multiseed/DISK` | `curve_data/disk_curve.csv` |
| PIMD | `Compare_Exp/experiment_records/pimd_python_Chen2020_5seeds_50evals_2026_05_11/seed8409..seed8413/pimd_Chen2020/database.json`, canonical HV recomputed over database prefixes, mean/std across five seeds | `raw_sources/multiseed/PIMD` | `curve_data/pimd_curve.csv` |

## Useful files

- `curve_data/hv_convergence_5way_plotted_curves.csv`: merged table for all five plotted curves, including centerlines, plotted bands, lower bounds, and upper bounds.
- `curve_data/hv_convergence_5way_plotted_curves.json`: JSON version of the same plotted data.
- `curve_data/per_seed_traces/`: per-seed traces for NSGA-II, DISK, and PIMD, plus proxy traces used for the ParEGO/LLAMBO-MO estimated bands.
- `manifest.json`: machine-readable source map and final HV summary.
- `figure/`: copied png/pdf figure.
- `code/`: copied generation script and reference local-std manifest.

Note: ParEGO and LLAMBO-MO centerlines are single-seed seed8409 traces. Their shaded bands come from proxy five-seed traces and are scaled by the plotting script.
