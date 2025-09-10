# Tide Transformer v1 - Dependencies and Data Setup
Write-Host "Tide Transformer v1 - Setup (Dependencies & Data)" -ForegroundColor Cyan
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
py -3.11 -m pip install modal torch torchvision numpy scikit-learn tensorboard --upgrade

# Check if authenticated (don't try to authenticate)
Write-Host ""
Write-Host "Checking Modal authentication..." -ForegroundColor Yellow
$authOutput = py -3.11 -m modal app list 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "Not authenticated with Modal." -ForegroundColor Red
    Write-Host "Please run .\login.ps1 first to authenticate." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Modal authentication OK" -ForegroundColor Green

# Upload training data
Write-Host ""
Write-Host "Uploading training data..." -ForegroundColor Yellow
py -3.11 -m modal run modal_single_run_seq2seq.py::upload_training_data

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Setup complete! Run .\run.ps1 to start single training run." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Data upload failed. Check the error messages above." -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit"