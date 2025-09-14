Write-Host "Modal Training Execution for Transformer v2"
Write-Host "============================================"
Write-Host ""

Write-Host "This script will start training on Modal's H100 GPU."
Write-Host ""

Write-Host "Pre-flight checks..."

# Check Modal authentication
Write-Host "Checking Modal authentication..."
$authOutput = python -m modal app list 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Not logged into Modal. Please run .\login.ps1 first"
    exit 1
}
Write-Host "Modal authentication OK"

# Check if app is deployed
Write-Host "Checking Modal app deployment..."
$appList = & python -m modal app list 2>$null
$appCheck = $appList | Select-String -Pattern "deployed" -CaseSensitive:$false
if (-not $appCheck) {
    Write-Host "ERROR: No deployed Modal apps found. Please run .\setup.ps1 first"
    exit 1
}
Write-Host "Modal app deployed"

# Check training data
Write-Host "Checking training data..."
$dataPath = "..\data-preparation\data"
$requiredFiles = @("X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $dataPath $file
    if (-not (Test-Path $fullPath)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "ERROR: Missing training data files:"
    foreach ($file in $missingFiles) {
        Write-Host "  - $file"
    }
    exit 1
}
Write-Host "Training data available"

Write-Host ""
Write-Host "All pre-flight checks passed!"
Write-Host ""

Write-Host "Starting training execution..."
Write-Host "This will:"
Write-Host "  - Load your prepared training data (~12k sequences)"
Write-Host "  - Upload data to Modal cloud"
Write-Host "  - Train transformer v2 on H100 GPU for 50 epochs"
Write-Host "  - Save trained model to Modal volume"
Write-Host ""

Write-Host ""
Write-Host "Executing training..."
Write-Host ""

python run_training.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Training completed successfully!"
    Write-Host ""
    Write-Host "Downloading trained model and logs..."
    Write-Host ""
    
    # Download best model to shared directory
    Write-Host "Downloading best model to ../shared/..."
    python -m modal volume get transformer-v2-storage best_model.pth ../shared/model.pth
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Best model downloaded to ../shared/model.pth"
    } else {
        Write-Host "Failed to download best model"
    }
    
    # Download training log to current directory
    Write-Host "Downloading training log..."
    python -m modal volume get transformer-v2-storage training_log.json ./
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Training log downloaded to ./training_log.json"
    } else {
        Write-Host "Failed to download training log"
    }
    
    Write-Host ""
    Write-Host "Downloaded files:"
    Write-Host "  - ../shared/model.pth (best performing model)"
    Write-Host "  - ./training_log.json (training metrics and config)"
    Write-Host ""
    Write-Host "Additional files remain on Modal volume 'transformer-v2-storage':"
    Write-Host "  - final_model.pth (final epoch model)"
    Write-Host "  - checkpoint_*.pth (periodic checkpoints)"
} else {
    Write-Host ""
    Write-Host "Training failed!"
    Write-Host "Please check the error messages above and try again."
    Write-Host ""
    Write-Host "Common issues:"
    Write-Host "  - Modal authentication expired (run .\login.ps1)"
    Write-Host "  - Modal app not deployed (run .\setup.ps1)" 
    Write-Host "  - Missing training data (run data preparation scripts)"
    exit 1
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")