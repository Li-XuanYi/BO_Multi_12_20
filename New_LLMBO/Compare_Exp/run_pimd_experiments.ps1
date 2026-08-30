# run_pimd_experiments.ps1 - Run PIMD experiments with seeds 8409-8413
# Usage: .\run_pimd_experiments.ps1

$ErrorActionPreference = "Stop"

# Configuration
$Seeds = @(8409, 8410, 8411, 8412, 8413)
$NEvals = 50
$PopulationSize = 20
$ParamSet = "Chen2020"
$Algorithm = "PIMD"
$DateStr = Get-Date -Format "yyyy_MM_dd"
$OutputRoot = "D:\Users\aa133\Desktop\BO_Multi_12_20\New_LLMBO\Compare_Exp\experiment_records\pimd_${ParamSet}_5seeds_${NEvals}evals_${DateStr}"

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

Write-Host "============================================================" -ForegroundColor Green
Write-Host "PIMD Experiments (50 iterations)" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Seeds: $($Seeds -join ', ')"
Write-Host "Evaluations: $NEvals"
Write-Host "Output: $OutputRoot"
Write-Host ""

# Change to project directory
Set-Location "D:\Users\aa133\Desktop\BO_Multi_12_20\New_LLMBO"

# Activate conda environment if needed
# conda activate your_env_name

foreach ($Seed in $Seeds) {
    Write-Host "Running PIMD with seed $Seed..." -ForegroundColor Yellow

    $OutputDir = "$OutputRoot\seed${Seed}"
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

    try {
        python -m Compare_Exp.Exp.run_platemo_experiments `
            --algorithm $Algorithm `
            --seeds $Seed `
            --n-evals $NEvals `
            --population-size $PopulationSize `
            --param-set $ParamSet `
            --output-root $OutputRoot

        Write-Host "  Seed $Seed completed successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "  Seed $Seed failed: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "PIMD Experiments Complete!" -ForegroundColor Green
Write-Host "Results saved to: $OutputRoot" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
