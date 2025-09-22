# Deploy Transformer v2 Inference Function to Firebase
# Deploys the inference function code without the large model files (downloads from Storage)

Write-Host "=== Transformer v2 Inference Deployment ===" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
$currentDir = Get-Location
if ($currentDir.Path -notmatch "transformer\\v2$") {
    Write-Host "Error: Please run this script from the transformer/v2 directory" -ForegroundColor Red
    Write-Host "Current directory: $currentDir" -ForegroundColor Yellow
    exit 1
}

# Verify required files exist (but not the large model file)
$requiredFiles = @(
    "shared/model.py", 
    "shared/inference.py",
    "shared/normalization_params.json",
    "inference/main.py",
    "inference/requirements.txt",
    "inference/firebase.json"
)

Write-Host "Checking required files..." -ForegroundColor Yellow
foreach ($file in $requiredFiles) {
    if (!(Test-Path $file)) {
        Write-Host "Error: Required file missing: $file" -ForegroundColor Red
        exit 1
    }
    Write-Host "  + $file" -ForegroundColor Green
}

# Check if model has been uploaded to Firebase Storage
Write-Host ""
Write-Host "Checking if model is uploaded to Firebase Storage..." -ForegroundColor Yellow

try {
    # Try common bucket patterns
    $bucketPatterns = @("gs://tide-monitor-boron.firebasestorage.app/transformer-v2-models/model.pth")
    
    $modelFound = $false
    foreach ($bucketPath in $bucketPatterns) {
        $bucketCheck = gsutil ls $bucketPath 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  + Model found in Firebase Storage: $bucketPath" -ForegroundColor Green
            $modelFound = $true
            break
        }
    }
    
    if (-not $modelFound) {
        Write-Host "  ! Model not found in Firebase Storage" -ForegroundColor Yellow
        Write-Host "    Please run: .\upload-transformer-v2-model.ps1 first" -ForegroundColor Red
        
        $response = Read-Host "Continue deployment anyway? (y/N)"
        if ($response -ne "y" -and $response -ne "Y") {
            Write-Host "Deployment cancelled. Upload the model first." -ForegroundColor Yellow
            exit 1
        }
    }
} catch {
    Write-Host "  ! Cannot check Firebase Storage (gsutil not available)" -ForegroundColor Yellow
    Write-Host "    Function will attempt to download model on first run" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Preparing inference directory for deployment..." -ForegroundColor Yellow

# Copy shared code files to inference directory (excluding large model file)
Write-Host "  Copying shared code files to inference directory..."
Copy-Item "shared/model.py" "inference/model.py" -Force  
Copy-Item "shared/inference.py" "inference/inference.py" -Force
Copy-Item "shared/normalization_params.json" "inference/normalization_params.json" -Force

Write-Host "  Code files copied (model will be downloaded from Storage)" -ForegroundColor Green

# Setup virtual environment and install packages
Write-Host "  Setting up virtual environment..." -ForegroundColor Yellow

Push-Location "inference"

try {
    # Create virtual environment if it doesn't exist
    if (!(Test-Path "venv")) {
        Write-Host "    Creating virtual environment..." -ForegroundColor Gray
        python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment"
        }
    }
    
    # Install required packages
    Write-Host "    Installing Firebase packages..." -ForegroundColor Gray
    ./venv/Scripts/pip install firebase-functions firebase-admin google-cloud-storage numpy --quiet --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Firebase packages"
    }
    
    Write-Host "    Installing PyTorch (CPU-only)..." -ForegroundColor Gray
    ./venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install PyTorch"
    }
    
    Write-Host "    Virtual environment setup complete" -ForegroundColor Green
    
} catch {
    Write-Host "    Error setting up virtual environment: $_" -ForegroundColor Red
    Pop-Location
    exit 1
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Deploying to Firebase Functions..." -ForegroundColor Yellow

# Change to inference directory for deployment
Push-Location "inference"

try {
    # Deploy the transformer v2 inference function
    Write-Host "  Running: firebase deploy --project tide-monitor-boron" -ForegroundColor Cyan
    
    firebase deploy --project tide-monitor-boron
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "=== Deployment Successful! ===" -ForegroundColor Green
        Write-Host ""
        Write-Host "Transformer v2 inference function deployed:" -ForegroundColor White
        Write-Host "  - Function: run_transformer_v2_analysis" -ForegroundColor Cyan
        Write-Host "  - Schedule: Every 5 minutes" -ForegroundColor Cyan
        Write-Host "  - Memory: 2048 MB" -ForegroundColor Cyan
        Write-Host "  - Timeout: 540 seconds" -ForegroundColor Cyan
        Write-Host "  - Model: Downloaded from Firebase Storage (12M parameters)" -ForegroundColor Cyan
        Write-Host "  - Output: 144 predictions (24 hours, 10-minute intervals)" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Firebase paths:" -ForegroundColor White
        Write-Host "  - Forecasts: /tidal-analysis/transformer-v2-forecast" -ForegroundColor Cyan
        Write-Host "  - Errors: /tidal-analysis/transformer-v2-error" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Model Storage:" -ForegroundColor White
        Write-Host "  - Storage path: gs://tide-monitor-boron.appspot.com/transformer-v2-models/" -ForegroundColor Cyan
        Write-Host "  - Download on: Function cold start" -ForegroundColor Cyan
        Write-Host "  - Cache location: /tmp/transformer-v2-model.pth" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor White
        Write-Host "  1. Monitor function logs: firebase functions:log --only run_transformer_v2_analysis" -ForegroundColor Yellow
        Write-Host "  2. Check Firebase Console for execution status" -ForegroundColor Yellow
        Write-Host "  3. Verify forecast data appears in Firebase Realtime Database" -ForegroundColor Yellow
        Write-Host "  4. Add dashboard integration to display v2 forecasts" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Model download verification:" -ForegroundColor White
        Write-Host "  - First function execution will download model from Storage" -ForegroundColor Yellow
        Write-Host "  - Download time: ~30-60 seconds for 306MB model" -ForegroundColor Yellow
        Write-Host "  - Subsequent runs use cached model (faster execution)" -ForegroundColor Yellow
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "=== Deployment Failed! ===" -ForegroundColor Red
        Write-Host "Check the error messages above for details." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Common issues:" -ForegroundColor White
        Write-Host "  - Make sure you're logged in: firebase login" -ForegroundColor Yellow
        Write-Host "  - Check project access: firebase projects:list" -ForegroundColor Yellow
        Write-Host "  - Verify virtual environment is set up in inference directory" -ForegroundColor Yellow
        Write-Host "  - Check that Firebase Storage model was uploaded successfully" -ForegroundColor Yellow
        Write-Host ""
    }
    
} finally {
    # Always return to original directory
    Pop-Location
}

Write-Host ""
Write-Host "Deployment script completed." -ForegroundColor Gray
