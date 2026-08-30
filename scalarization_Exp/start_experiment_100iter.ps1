# Scalarization Experiment Launcher - 100 Iterations
# Run this in PowerShell: .\scalarization_Exp\start_experiment_100iter.ps1

# Check API key
if (-not $env:LLM_API_KEY) {
    Write-Host "ERROR: Please set `$env:LLM_API_KEY first!" -ForegroundColor Red
    Write-Host "Example: `$env:LLM_API_KEY='your-key-here'" -ForegroundColor Yellow
    exit 1
}

# Set default base URL if not set
if (-not $env:LLM_BASE_URL) {
    $env:LLM_BASE_URL = "https://api.nuwaapi.com/v1"
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Scalarization Experiment Launcher" -ForegroundColor Cyan
Write-Host "  100 Iterations Version" -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Model: deepseek-v3-thinking" -ForegroundColor Yellow
Write-Host "Seeds: 8409, 8410, 8411, 8412, 8413" -ForegroundColor White
Write-Host "Modes: minmax, zscore, none" -ForegroundColor White
Write-Host "Iterations: 100" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan

$OutputRoot = "scalarization_Exp/experiment_records/scalarization_llmbo_mo_5seeds_100iter_deepseek_$(Get-Date -Format 'yyyy_MM_dd')"

Write-Host "`nStarting experiments... Output: $OutputRoot" -ForegroundColor Green
Write-Host "(This will take a LONG time - 15 experiments x 100 iterations)`n" -ForegroundColor Yellow

python scalarization_Exp/run_scalarization_experiments.py `
    --seeds 8409 8410 8411 8412 8413 `
    --iterations 100 `
    --skip-existing `
    --output-root $OutputRoot

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Experiments Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Next step - generate plot:" -ForegroundColor White
Write-Host "python scalarization_Exp/plot_scalarization_hv.py --exp-root $OutputRoot --max-iteration 100" -ForegroundColor Yellow
