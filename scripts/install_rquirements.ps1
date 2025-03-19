
#### choco install ffmpeg
#### choco install poppler
#### choco install librosa
#### python -m nltk.downloader popular

param(
    [switch]$ForceCPU
)

Write-Host "=== Python Package Installer ===" -ForegroundColor Cyan

# First check GPU availability
$HasGPU = $false
if (-not $ForceCPU) {
    try {
        $nvidia_output = nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            $HasGPU = $true
            Write-Host "NVIDIA GPU detected!" -ForegroundColor Green
        }
    } catch {
        Write-Host "No NVIDIA GPU detected, using CPU version" -ForegroundColor Yellow
        $HasGPU = $false
    }
}

# Get the requirements file path
$reqPath = Join-Path $PSScriptRoot ".." "requirements.txt"
if (-not (Test-Path $reqPath)) {
    Write-Host "requirements.txt not found!" -ForegroundColor Red
    exit 1
}

# Read requirements and modify torch-related packages
$requirements = Get-Content $reqPath
$modified = @()

foreach ($line in $requirements) {
    if ($line -match "^torch==") {
        if ($HasGPU) {
            Write-Host "Installing PyTorch with CUDA support" -ForegroundColor Green
            $modified += "# For CUDA support:"
            $modified += $line
        } else {
            Write-Host "Installing CPU-only PyTorch" -ForegroundColor Yellow
            $modified += "# CPU-only version:"
            $modified += "$line+cpu"
        }
    } elseif ($line -match "^torchvision==") {
        if ($HasGPU) {
            $modified += $line
        } else {
            $modified += "$line+cpu"
        }
    } elseif ($line -match "^torchaudio==") {
        if ($HasGPU) {
            $modified += $line
        } else {
            $modified += "$line+cpu"
        }
    } else {
        $modified += $line
    }
}

# Create temporary requirements file
$tempReqPath = Join-Path $PSScriptRoot "temp_requirements.txt"
$modified | Set-Content $tempReqPath

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Cyan
python -m pip install -r $tempReqPath

if ($LASTEXITCODE -eq 0) {
    Write-Host "Installation completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Installation failed!" -ForegroundColor Red
}

# Cleanup
Remove-Item $tempReqPath

Write-Host "=== Installation Process Finished ===" -ForegroundColor Cyan
