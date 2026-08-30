# Patero

`Patero` is a small plotting utility folder for building a reference-style 3D Pareto figure from Python experiment results.

It does three things:

1. Reads Python-side result files such as `database.json` or `pareto_front.json`
2. Filters non-dominated solutions with an in-Python Pareto algorithm
3. Draws a reference-style 3D scatter plot with red star highlights

## Layout

- [plot_soh_pareto.py](/d:/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/Patero/plot_soh_pareto.py)
- `demo_data/`
- `output/`
- `demo_config.json`

## Quick start

Generate demo data and a reference-style algorithm comparison image:

```powershell
python Patero\plot_soh_pareto.py --make-demo
```

Use your own JSON files:

```powershell
python Patero\plot_soh_pareto.py --config Patero\demo_config.json
```

## Supported input shapes

The script accepts several JSON layouts:

- `{"observations": [{"objectives": [...]}, ...]}`
- `{"pareto_front": [{"objectives": [...]}, ...]}`
- `{"points": [[x, y, z], ...]}`
- a raw JSON list of objective vectors

## Notes

- Folder name is kept as `Patero` to match your request.
- Demo data is synthetic and only meant to reproduce the plotting style.
- For real runs, just replace the paths in `demo_config.json` with your own Python result files.
