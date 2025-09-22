# Configure Firebase Storage Bucket for Transformer v2
# Auto-detects and configures the correct bucket name for upload/deployment scripts

Write-Host "=== Firebase Storage Bucket Configuration ===" -ForegroundColor Green
Write-Host ""

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

# Ensure correct project
Write-Host "Setting up Google Cloud project..." -ForegroundColor Yellow
$currentProject = gcloud config get-value project 2>$null
if ($currentProject -ne "tide-monitor-boron") {
    Write-Host "  Setting project to tide-monitor-boron..." -ForegroundColor Gray
    gcloud config set project tide-monitor-boron
}
Write-Host "  Using project: tide-monitor-boron" -ForegroundColor Cyan

# Auto-detect Firebase Storage bucket
Write-Host ""
Write-Host "Auto-detecting Firebase Storage bucket..." -ForegroundColor Yellow

$possibleBuckets = @(
    "tide-monitor-boron.appspot.com",
    "tide-monitor-boron.firebasestorage.app", 
    "tide-monitor-boron"
)

$bucketName = $null
foreach ($bucket in $possibleBuckets) {
    Write-Host "  Checking: gs://$bucket" -ForegroundColor Gray
    $checkResult = gsutil ls "gs://$bucket/" 2>&1
    if ($LASTEXITCODE -eq 0) {
        $bucketName = $bucket
        Write-Host "  + Found bucket: $bucket" -ForegroundColor Green
        break
    } else {
        Write-Host "    Not found" -ForegroundColor Gray
    }
}

if (-not $bucketName) {
    Write-Host ""
    Write-Host "No Firebase Storage bucket found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please enable Firebase Storage:" -ForegroundColor White
    Write-Host "  1. Go to: https://console.firebase.google.com/project/tide-monitor-boron/storage" -ForegroundColor Yellow
    Write-Host "  2. Click 'Get Started' to enable Cloud Storage" -ForegroundColor Yellow
    Write-Host "  3. Choose storage location (us-central1 recommended)" -ForegroundColor Yellow
    Write-Host "  4. Click 'Done'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or create bucket manually:" -ForegroundColor White
    Write-Host "  gcloud storage buckets create gs://tide-monitor-boron --location=us-central1" -ForegroundColor Cyan
    exit 1
}

# Update upload script with correct bucket name
Write-Host ""
Write-Host "Configuring upload script..." -ForegroundColor Yellow

$uploadScript = Get-Content "upload-transformer-v2-model.ps1" -Raw
$uploadScript = $uploadScript -replace 'possibleBuckets = @\([^)]+\)', "possibleBuckets = @(`"$bucketName`")"
$uploadScript | Set-Content "upload-transformer-v2-model.ps1" -Encoding UTF8

Write-Host "  + Updated upload-transformer-v2-model.ps1" -ForegroundColor Green

# Update deployment script with correct bucket name
Write-Host "Configuring deployment script..." -ForegroundColor Yellow

$deployScript = Get-Content "deploy-transformer-v2-inference.ps1" -Raw
$deployScript = $deployScript -replace 'bucketPatterns = @\([^)]+\)', "bucketPatterns = @(`"gs://$bucketName/transformer-v2-models/model.pth`")"
$deployScript | Set-Content "deploy-transformer-v2-inference.ps1" -Encoding UTF8

Write-Host "  + Updated deploy-transformer-v2-inference.ps1" -ForegroundColor Green

# Update Python inference code with correct bucket
Write-Host "Configuring Python inference code..." -ForegroundColor Yellow

$pythonScript = Get-Content "inference/main.py" -Raw
$pythonScript = $pythonScript -replace 'self\.project_id = "[^"]*"', "self.project_id = `"tide-monitor-boron`""

# Set the first bucket in the bucket detection list to the discovered one
$bucketList = "bucket_names = [`"$bucketName`", `"tide-monitor-boron.appspot.com`", `"tide-monitor-boron`"]"
$pythonScript = $pythonScript -replace 'bucket_names = \[[^\]]+\]', $bucketList

$pythonScript | Set-Content "inference/main.py" -Encoding UTF8

Write-Host "  + Updated inference/main.py" -ForegroundColor Green

Write-Host ""
Write-Host "=== Configuration Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Configured bucket: $bucketName" -ForegroundColor Cyan
Write-Host ""
Write-Host "Ready to use deployment scripts:" -ForegroundColor White
Write-Host "  1. .\upload-transformer-v2-model.ps1      (upload 306MB model to Storage)" -ForegroundColor Yellow
Write-Host "  2. .\deploy-transformer-v2-inference.ps1  (deploy inference function)" -ForegroundColor Yellow
Write-Host ""
Write-Host "All scripts are now configured with your actual bucket name." -ForegroundColor Gray