Write-Host "Transformer v2 - Modal Authentication" -ForegroundColor Cyan
Write-Host ""

# Check Modal authentication status
Write-Host "Checking Modal authentication status..." -ForegroundColor Yellow
$authOutput = python -m modal token list 2>&1

if ($authOutput -match "No token found" -or $LASTEXITCODE -ne 0) {
    Write-Host "Not logged in to Modal. Setting up authentication..." -ForegroundColor Yellow
    Write-Host "A browser window will open for authentication." -ForegroundColor Cyan
    Write-Host ""
    
    python -m modal token new
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Authentication failed. Please try again." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    } else {
        Write-Host ""
        Write-Host "Authentication successful!" -ForegroundColor Green
        Write-Host "You can now run setup.ps1 and train.ps1" -ForegroundColor Gray
    }
} else {
    Write-Host "Already authenticated with Modal (OK)" -ForegroundColor Green
    Write-Host "You can proceed with setup.ps1 and train.ps1" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")