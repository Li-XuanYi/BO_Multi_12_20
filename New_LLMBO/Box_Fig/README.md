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

Generate the bundled reference-style demo figure:

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

- The demo values are synthetic so the picture stays close to your reference image.
- For real runs, replace each group `path` with your own Python-generated `report*.json`.
- If one algorithm has no ready report yet, you can temporarily use `values` in the config.
