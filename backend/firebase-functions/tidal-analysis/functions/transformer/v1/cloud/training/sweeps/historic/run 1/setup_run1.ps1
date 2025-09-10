# Tide Transformer v1 - Modal Setup Script
Write-Host "Tide Transformer v1 - Modal Setup" -ForegroundColor Cyan
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
py -3.11 -m pip install modal torch numpy scikit-learn "ray[tune]" bayesian-optimization --upgrade

# Authenticate with Modal
Write-Host "Setting up Modal authentication..." -ForegroundColor Yellow
Write-Host "A browser window will open for GitHub authentication." -ForegroundColor Cyan
Write-Host ""

py -3.11 -m modal token new

# Upload training data
Write-Host "Uploading training data..." -ForegroundColor Yellow
py -3.11 -m modal run modal_hp_sweep.py::upload_training_data

Write-Host ""
Write-Host "Setup complete! Run .\run.ps1 to start hyperparameter optimization." -ForegroundColor Green
Read-Host "Press Enter to exit"