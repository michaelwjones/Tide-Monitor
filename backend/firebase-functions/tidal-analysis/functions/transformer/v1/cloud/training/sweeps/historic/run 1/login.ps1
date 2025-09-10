# Tide Transformer v1 - Modal Authentication Script
Write-Host "Tide Transformer v1 - Modal Authentication" -ForegroundColor Cyan
Write-Host ""

# Check Modal authentication status
Write-Host "Checking Modal authentication status..." -ForegroundColor Yellow
$authOutput = py -3.11 -m modal token list 2>&1

if ($authOutput -match "No token found" -or $LASTEXITCODE -ne 0) {
    Write-Host "Not logged in to Modal. Setting up authentication..." -ForegroundColor Yellow
    Write-Host "A browser window will open for GitHub authentication." -ForegroundColor Cyan
    Write-Host ""
    
    py -3.11 -m modal token new
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Authentication failed. Please try again." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    } else {
        Write-Host ""
        Write-Host "Authentication successful!" -ForegroundColor Green
        Write-Host "You can now run setup.ps1 and run.ps1" -ForegroundColor Gray
    }
} else {
    Write-Host "Already authenticated with Modal (OK)" -ForegroundColor Green
    Write-Host "You can proceed with setup.ps1 and run.ps1" -ForegroundColor Gray
}

Write-Host ""
Read-Host "Press Enter to exit"