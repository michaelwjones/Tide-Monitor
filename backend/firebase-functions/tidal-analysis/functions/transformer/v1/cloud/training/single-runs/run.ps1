# Tide Transformer v1 - Single Training Run
Write-Host "Tide Transformer v1 - Single Training Run" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting single training run..." -ForegroundColor Green
Write-Host "This will run on Modal's H100 GPU (~$15-25 cost for 150 epochs)." -ForegroundColor Gray
Write-Host "Expected training time: 2-4 hours" -ForegroundColor Gray
Write-Host ""

# Record start time
$startTime = Get-Date
Write-Host "Started at: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
Write-Host ""

# Run the single training
py -3.11 -m modal run modal_single_run_seq2seq.py::run_single_training

if ($LASTEXITCODE -eq 0) {
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host ""
    Write-Host "TRAINING COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "Total duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Gray
    Write-Host ""
    
    # Download results
    Write-Host "Downloading trained model and summary..." -ForegroundColor Yellow
    py -3.11 -m modal volume get tide-training-data best_single_pass.pth .\ --force
    py -3.11 -m modal volume get tide-training-data single_pass_summary.json .\ --force
    
    if (Test-Path ".\single_pass_summary.json") {
        Write-Host "Results downloaded successfully!" -ForegroundColor Green
        
        # Show summary
        $summary = Get-Content ".\single_pass_summary.json" | ConvertFrom-Json
        Write-Host ""
        Write-Host "Training Summary:" -ForegroundColor Cyan
        Write-Host "Best validation loss: $($summary.best_val_loss)" -ForegroundColor Green
        Write-Host "Total epochs: $($summary.total_epochs)" -ForegroundColor Gray
        Write-Host "Model parameters: $($summary.model_parameters)" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Files saved:" -ForegroundColor Cyan
        Write-Host "- best_single_pass.pth (trained single-pass transformer)" -ForegroundColor White
        Write-Host "- single_pass_summary.json (training summary)" -ForegroundColor White
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. Copy best_single_pass.pth to ../../../../../../best.pth (tidal-analysis root)" -ForegroundColor White
        Write-Host "2. Test the model using local/testing/start-server.bat" -ForegroundColor White
        Write-Host "3. Deploy to Firebase using ../../../../../../deploy-transformer-v1.bat" -ForegroundColor White
    }
} else {
    Write-Host ""
    Write-Host "TRAINING FAILED" -ForegroundColor Red
    Write-Host "Check logs: py -3.11 -m modal app logs tide-transformer-v1-single-pass-run" -ForegroundColor Gray
}

Write-Host ""
Read-Host "Press Enter to exit"