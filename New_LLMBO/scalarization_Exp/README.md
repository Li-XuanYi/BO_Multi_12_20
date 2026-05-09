# Scalarization preprocessing experiments

Runs LLMBO-MO with three log-space objective preprocessing modes:

- `minmax`
- `zscore`
- `none`

The runner keeps `log10(time)` and `log10(aging)` active and compares only the
preprocessing used before weighted Tchebycheff scalarization.

```powershell
$env:LLM_API_KEY = "<your key>"
$env:OPENAI_BASE_URL = "https://api.nuwaapi.com/v1"
python scalarization_Exp/run_scalarization_experiments.py --seeds 8409 8410 8411 8412 8413 --iterations 50 --skip-existing
python scalarization_Exp/plot_scalarization_hv.py --exp-root scalarization_Exp/experiment_records/<run_dir>
```
