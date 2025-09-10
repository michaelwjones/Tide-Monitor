# Tide Transformer v1 - Single Run Setup Script
Write-Host "Tide Transformer v1 - Single Training Run Setup" -ForegroundColor Cyan
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
py -3.11 -m pip install modal torch numpy scikit-learn --upgrade

# Authenticate with Modal
Write-Host "Setting up Modal authentication..." -ForegroundColor Yellow
Write-Host "A browser window will open for GitHub authentication." -ForegroundColor Cyan
Write-Host ""

py -3.11 -m modal token new

# Upload training data
Write-Host "Uploading training data..." -ForegroundColor Yellow
py -3.11 -m modal run modal_single_run_seq2seq.py::upload_training_data

Write-Host ""
Write-Host "Setup complete! Run .\run.ps1 to start single training run." -ForegroundColor Green
Read-Host "Press Enter to exit"