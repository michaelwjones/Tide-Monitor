Write-Host "Starting Transformer v2 Data Analysis Server..."
Write-Host ""
python server.py
Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")