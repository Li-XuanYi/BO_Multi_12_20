# Monitor experiment progress
$ExpRoot = "scalarization_Exp/experiment_records"
$Latest = Get-ChildItem -Path $ExpRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $Latest) {
    Write-Host "No experiment directory found!" -ForegroundColor Red
    exit 1
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Experiment Progress Monitor" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Directory: $($Latest.FullName)" -ForegroundColor Yellow

$ReportPath = Join-Path $Latest.FullName "report_5seeds.json"
if (Test-Path $ReportPath) {
    $Report = Get-Content $ReportPath -Raw | ConvertFrom-Json

    Write-Host "`nExperiment: $($Report.meta.experiment)" -ForegroundColor Green
    Write-Host "Model: $($Report.meta.model)" -ForegroundColor White
    Write-Host "Iterations: $($Report.meta.iterations)" -ForegroundColor White
    Write-Host "Seeds: $($Report.meta.seeds -join ', ')" -ForegroundColor White
    Write-Host "Modes: $($Report.meta.modes -join ', ')" -ForegroundColor White

    Write-Host "`n--- Progress ---" -ForegroundColor Cyan
    foreach ($mode in $Report.meta.modes) {
        $agg = $Report.aggregates.$mode
        $completed = $agg.n_runs
        $total = $Report.meta.seeds.Count
        $pct = [math]::Round(($completed / $total) * 100, 1)
        Write-Host "$mode`: $completed/$total ($pct%)" -ForegroundColor $(if ($completed -eq $total) { 'Green' } else { 'Yellow' })

        if ($completed -gt 0) {
            Write-Host "  HV mean: $([math]::Round($agg.canonical_hv.mean, 4)) ± $([math]::Round($agg.canonical_hv.std, 4))" -ForegroundColor Gray
        }
    }

    Write-Host "`n--- Individual Runs ---" -ForegroundColor Cyan
    foreach ($record in $Report.records | Sort-Object seed, mode) {
        $color = if ($record.status -eq 'ok') { 'Green' } else { 'Red' }
        Write-Host "seed=$($record.seed) mode=$($record.mode) status=$($record.status)" -ForegroundColor $color
    }
} else {
    Write-Host "Report not yet generated. Experiments are still running..." -ForegroundColor Yellow
}

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "Refresh: .\scalarization_Exp\monitor_progress.ps1" -ForegroundColor Gray
