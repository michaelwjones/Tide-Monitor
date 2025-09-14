Write-Host "Transformer v2 - Setup (Dependencies & Data)" -ForegroundColor Cyan
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install modal torch torchvision numpy scikit-learn tensorboard --upgrade

# Check if authenticated (using app list like v1)
Write-Host ""
Write-Host "Checking Modal authentication..." -ForegroundColor Yellow
$authOutput = python -m modal app list 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "Not authenticated with Modal." -ForegroundColor Red
    Write-Host "Please run .\login.ps1 first to authenticate." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Modal authentication OK" -ForegroundColor Green
Write-Host ""

Write-Host "Verifying training data exists..."
$dataPath = "..\data-preparation\data"

$requiredFiles = @("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "metadata.json", "normalization_params.json")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $dataPath $file
    if (-not (Test-Path $fullPath)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "ERROR: Missing required data files:"
    foreach ($file in $missingFiles) {
        Write-Host "  - $file"
    }
    Write-Host ""
    Write-Host "Please run data preparation first:"
    Write-Host "  cd ..\data-preparation"
    Write-Host "  .\run-data-preparation.ps1"
    exit 1
}

Write-Host "All required data files found:"
foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $dataPath $file
    $size = (Get-Item $fullPath).Length / 1MB
    Write-Host "  $file (${size:F1} MB)"
}

Write-Host ""
Write-Host "Deploying Modal app..." -ForegroundColor Yellow
Write-Host "This will create the Modal app and upload the training function."
Write-Host ""

python -m modal deploy modal_setup.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Modal app deployed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Setup complete. You can now run training with:" -ForegroundColor Green
    Write-Host "  .\train.ps1" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "Modal app deployment failed. Please check the error messages above." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")