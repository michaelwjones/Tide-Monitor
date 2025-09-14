# Upload Transformer v2 Model to Firebase Storage
# Uploads the large model files to Firebase Storage for download during function execution

Write-Host "=== Transformer v2 Model Upload to Firebase Storage ===" -ForegroundColor Green
Write-Host ""

# Check if we're in the right directory
$currentDir = Get-Location
if ($currentDir.Path -notmatch "transformer\\v2$") {
    Write-Host "Error: Please run this script from the transformer/v2 directory" -ForegroundColor Red
    Write-Host "Current directory: $currentDir" -ForegroundColor Yellow
    exit 1
}

# Verify required files exist
$requiredFiles = @(
    "shared/model.pth",
    "shared/normalization_params.json"
)

Write-Host "Checking required model files..." -ForegroundColor Yellow
foreach ($file in $requiredFiles) {
    if (!(Test-Path $file)) {
        Write-Host "Error: Required file missing: $file" -ForegroundColor Red
        Write-Host "Make sure you have trained the model and it's in the shared directory." -ForegroundColor Yellow
        exit 1
    }
    $fileSize = (Get-Item $file).Length / 1MB
    Write-Host "  + $file ($([math]::Round($fileSize, 1)) MB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Uploading to Firebase Storage..." -ForegroundColor Yellow

# Check if gsutil is available
try {
    gsutil version > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "gsutil not found"
    }
} catch {
    Write-Host "Error: gsutil not found. Please install Google Cloud SDK:" -ForegroundColor Red
    Write-Host "  1. Download from: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    Write-Host "  2. Run: gcloud auth login" -ForegroundColor Yellow
    Write-Host "  3. Run: gcloud config set project tide-monitor-boron" -ForegroundColor Yellow
    exit 1
}

# Ensure we're using the correct project
Write-Host "  Checking Google Cloud project configuration..." -ForegroundColor Yellow
$currentProject = gcloud config get-value project 2>$null
if ($currentProject -ne "tide-monitor-boron") {
    Write-Host "    Setting project to tide-monitor-boron..." -ForegroundColor Gray
    gcloud config set project tide-monitor-boron > $null 2>&1
}

# Auto-detect Firebase Storage bucket name
Write-Host "  Detecting Firebase Storage bucket..." -ForegroundColor Yellow

# Try common Firebase bucket patterns
$possibleBuckets = @("tide-monitor-boron.firebasestorage.app")

$bucketName = $null
foreach ($bucket in $possibleBuckets) {
    Write-Host "    Checking: gs://$bucket" -ForegroundColor Gray
    $checkResult = gsutil ls "gs://$bucket/" 2>&1
    if ($LASTEXITCODE -eq 0) {
        $bucketName = $bucket
        Write-Host "    Found bucket: $bucket" -ForegroundColor Green
        break
    }
}

if (-not $bucketName) {
    Write-Host "  Error: Could not find Firebase Storage bucket" -ForegroundColor Red
    Write-Host "  Please check these options:" -ForegroundColor Yellow
    Write-Host "    1. Ensure Firebase Storage is enabled in Firebase Console" -ForegroundColor Yellow
    Write-Host "    2. Run: gcloud auth login" -ForegroundColor Yellow
    Write-Host "    3. Run: gcloud config set project tide-monitor-boron" -ForegroundColor Yellow
    Write-Host "    4. Check bucket name in Firebase Console > Storage" -ForegroundColor Yellow
    exit 1
}

$modelPath = "transformer-v2-models"
Write-Host "  Target bucket: gs://$bucketName/$modelPath/" -ForegroundColor Cyan
Write-Host ""

# Upload model file
Write-Host "  Uploading model.pth..." -ForegroundColor Yellow
$modelUpload = gsutil -m cp "shared/model.pth" "gs://$bucketName/$modelPath/model.pth"
if ($LASTEXITCODE -eq 0) {
    Write-Host "    Model uploaded successfully!" -ForegroundColor Green
} else {
    Write-Host "    Model upload failed!" -ForegroundColor Red
    exit 1
}

# Upload normalization parameters
Write-Host "  Uploading normalization_params.json..." -ForegroundColor Yellow
$paramsUpload = gsutil -m cp "shared/normalization_params.json" "gs://$bucketName/$modelPath/normalization_params.json"
if ($LASTEXITCODE -eq 0) {
    Write-Host "    Normalization params uploaded successfully!" -ForegroundColor Green
} else {
    Write-Host "    Normalization params upload failed!" -ForegroundColor Red
    exit 1
}

# Set public read permissions for the model files
Write-Host ""
Write-Host "  Setting permissions..." -ForegroundColor Yellow
gsutil acl ch -u AllUsers:R "gs://$bucketName/$modelPath/model.pth" > $null 2>&1
gsutil acl ch -u AllUsers:R "gs://$bucketName/$modelPath/normalization_params.json" > $null 2>&1

Write-Host ""
Write-Host "=== Model Upload Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Firebase Storage URLs:" -ForegroundColor White
Write-Host "  Model: https://storage.googleapis.com/$bucketName/$modelPath/model.pth" -ForegroundColor Cyan
Write-Host "  Params: https://storage.googleapis.com/$bucketName/$modelPath/normalization_params.json" -ForegroundColor Cyan
Write-Host ""
Write-Host "Bucket used for deployment scripts:" -ForegroundColor White
Write-Host "  $bucketName" -ForegroundColor Cyan
Write-Host ""
Write-Host "File sizes:" -ForegroundColor White
$modelSize = (Get-Item "shared/model.pth").Length / 1MB  
$paramsSize = (Get-Item "shared/normalization_params.json").Length / 1KB
Write-Host "  Model: $([math]::Round($modelSize, 1)) MB" -ForegroundColor Cyan
Write-Host "  Params: $([math]::Round($paramsSize, 1)) KB" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Run: .\deploy-transformer-v2-inference.ps1" -ForegroundColor Yellow
Write-Host "  2. The inference function will download these files on first run" -ForegroundColor Yellow
Write-Host "  3. Monitor function logs to verify successful model download" -ForegroundColor Yellow
Write-Host ""
Write-Host "Model verification:" -ForegroundColor White
Write-Host "  You can verify the upload by visiting Firebase Console > Storage" -ForegroundColor Yellow
Write-Host "  Or test download: gsutil cp gs://$bucketName/$modelPath/model.pth test-model.pth" -ForegroundColor Yellow
Write-Host ""
