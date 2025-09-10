# Tide Transformer v1 - Hyperparameter Optimization Runner
Write-Host "Tide Transformer v1 - Hyperparameter Sweep" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting hyperparameter optimization..." -ForegroundColor Green
Write-Host "This will run on Modal's cloud infrastructure (H100 GPU, ~$15-25 cost)." -ForegroundColor Gray
Write-Host ""

# Record start time
$startTime = Get-Date
Write-Host "Started at: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
Write-Host ""

# Run the hyperparameter sweep
py -3.11 -m modal run modal_hp_sweep.py::run_hyperparameter_sweep

if ($LASTEXITCODE -eq 0) {
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host ""
    Write-Host "OPTIMIZATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "Total duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Gray
    Write-Host ""
    
    # Download results
    Write-Host "Downloading results..." -ForegroundColor Yellow
    py -3.11 -m modal volume get tide-training-data hp_optimization_results.json .\
    
    if (Test-Path ".\hp_optimization_results.json") {
        Write-Host "Results saved to hp_optimization_results.json" -ForegroundColor Green
        
        # Show summary
        $results = Get-Content ".\hp_optimization_results.json" | ConvertFrom-Json
        Write-Host ""
        Write-Host "Best validation loss: $($results.best_val_loss)" -ForegroundColor Green
        Write-Host "Total trials: $($results.total_trials)" -ForegroundColor Gray
    }
} else {
    Write-Host ""
    Write-Host "OPTIMIZATION FAILED" -ForegroundColor Red
    Write-Host "Check logs: py -3.11 -m modal app logs tide-transformer-v1-hp-sweep" -ForegroundColor Gray
}

Write-Host ""
Read-Host "Press Enter to exit"