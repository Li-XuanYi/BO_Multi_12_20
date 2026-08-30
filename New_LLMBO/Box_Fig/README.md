# Box_Fig

`Box_Fig` builds a reference-style HV box plot from Python-side experiment results.

It supports two input styles:

1. Python experiment reports such as `report.json` / `report_5seeds.json`
2. Direct HV lists in JSON config for algorithms whose reports are not ready yet

## Layout

- [plot_hv_box.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/Box_Fig/plot_hv_box.py)
- `demo_data/`
- `output/`
- `demo_config.json`

## Quick start

Generate the bundled real-data reference-style figure:

```powershell
python Box_Fig\plot_hv_box.py --make-demo
```

Use your own config:

```powershell
python Box_Fig\plot_hv_box.py --config Box_Fig\demo_config.json
```

## Supported JSON shapes

- `{"records": [{"display_hv": ...}, ...]}`
- `{"records": [{"canonical_hv": ...}, ...]}`
- `{"values": [...]}`
- a raw JSON list of scalar HV values

## Notes

- The default figure points to real Ecker2015 experiment reports, not synthetic placeholders.
- The plotted HV values are `canonical_hv * 0.2`; the y-axis is labeled `HV`.
- LLMBO-MO and ParEGO are 5-seed Ecker2015 runs with `n_total=56`.
- NSGA-II, DISK, and PIMD are 5-seed Ecker2015 external baselines with `n_total=60`; keep this caveat in the paper text.
- For new runs, replace each group `path` with your own Python-generated `report*.json`.
- If one algorithm has no ready report yet, you can temporarily use `values` in the config.
