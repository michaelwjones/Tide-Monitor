#!/usr/bin/env powershell
# Transformer v2 Discontinuity Analysis Runner
# PowerShell wrapper for the analysis tool

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("training", "inference")]
    [string]$Data,
    
    [Parameter()]
    [ValidateSet("continuity", "direction", "statistics", "sequences")]
    [string[]]$TrainingAnalysis = @("continuity", "direction", "statistics", "sequences"),
    
    [Parameter()]
    [ValidateSet("level_jumps", "direction_changes", "cusps", "phase_errors")]
    [string[]]$InferenceAnalysis = @("level_jumps", "direction_changes", "cusps"),
    
    [Parameter()]
    [int]$NumTests = 50,
    
    [Parameter()]
    [switch]$Help
)

# Show help if requested
if ($Help) {
    Write-Host "Transformer v2 Discontinuity Analysis Tool" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage Examples:" -ForegroundColor Yellow
    Write-Host "  .\analyze.ps1 -Data training" -ForegroundColor White
    Write-Host "  .\analyze.ps1 -Data inference -NumTests 100" -ForegroundColor White
    Write-Host "  .\analyze.ps1 -Data training -TrainingAnalysis continuity,sequences" -ForegroundColor White
    Write-Host "  .\analyze.ps1 -Data inference -InferenceAnalysis level_jumps,cusps" -ForegroundColor White
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor Yellow
    Write-Host "  -Data: 'training' or 'inference'" -ForegroundColor White
    Write-Host "  -TrainingAnalysis: continuity, direction, statistics, sequences" -ForegroundColor White
    Write-Host "  -InferenceAnalysis: level_jumps, direction_changes, cusps, phase_errors" -ForegroundColor White
    Write-Host "  -NumTests: Number of inference tests (default: 50)" -ForegroundColor White
    Write-Host "  -Help: Show this help message" -ForegroundColor White
    Write-Host ""
    Write-Host "Training Analysis Types:" -ForegroundColor Yellow
    Write-Host "  continuity  - Gap analysis between input/output sequences" -ForegroundColor White
    Write-Host "  direction   - Direction consistency at sequence boundaries" -ForegroundColor White
    Write-Host "  statistics  - Basic dataset statistics" -ForegroundColor White
    Write-Host "  sequences   - Identify worst sequences for manual inspection" -ForegroundColor White
    Write-Host ""
    Write-Host "Inference Analysis Types:" -ForegroundColor Yellow
    Write-Host "  level_jumps      - Significant height differences at prediction start" -ForegroundColor White
    Write-Host "  direction_changes - Wrong tidal direction (rising vs falling)" -ForegroundColor White
    Write-Host "  cusps           - Correct start level but wrong tidal phase" -ForegroundColor White
    Write-Host "  phase_errors    - Temporal misalignment in tidal cycles" -ForegroundColor White
    exit 0
}

# Set working directory to script location
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Transformer v2 Discontinuity Analysis" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Working directory: $ScriptDir" -ForegroundColor Gray
Write-Host ""

# Build Python command
$PythonArgs = @("analyze.py", "--data", $Data)

if ($Data -eq "training") {
    Write-Host "Running training data analysis..." -ForegroundColor Yellow
    Write-Host "Analysis types: $($TrainingAnalysis -join ', ')" -ForegroundColor Gray
    $PythonArgs += "--training-analysis"
    $PythonArgs += $TrainingAnalysis
}
elseif ($Data -eq "inference") {
    Write-Host "Running inference discontinuity analysis..." -ForegroundColor Yellow
    Write-Host "Analysis types: $($InferenceAnalysis -join ', ')" -ForegroundColor Gray
    Write-Host "Number of tests: $NumTests" -ForegroundColor Gray
    $PythonArgs += "--inference-analysis"
    $PythonArgs += $InferenceAnalysis
    $PythonArgs += "--num-tests"
    $PythonArgs += $NumTests
}

Write-Host ""
Write-Host "Executing: python $($PythonArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Check if Python is available
try {
    $PythonVersion = python --version 2>&1
    Write-Host "Python version: $PythonVersion" -ForegroundColor Gray
} catch {
    Write-Host "Error: Python not found. Please ensure Python is installed and in PATH." -ForegroundColor Red
    exit 1
}

# Check if analyze.py exists
if (-not (Test-Path "analyze.py")) {
    Write-Host "Error: analyze.py not found in current directory." -ForegroundColor Red
    Write-Host "Please run this script from the discontinuity-analysis folder." -ForegroundColor Red
    exit 1
}

# Run the analysis
try {
    Write-Host "Starting analysis..." -ForegroundColor Yellow
    $StartTime = Get-Date
    
    & python $PythonArgs
    
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    
    Write-Host ""
    Write-Host "Analysis completed in $($Duration.TotalSeconds.ToString('F1')) seconds" -ForegroundColor Green
    
    # Show results location
    if ($Data -eq "inference" -and (Test-Path "discontinuity_results.json")) {
        Write-Host "Results saved to: discontinuity_results.json" -ForegroundColor Green
    }
    
} catch {
    Write-Host "Error running analysis: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Analysis complete!" -ForegroundColor Green