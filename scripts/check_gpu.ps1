Write-Host "Checking GPU and CUDA availability..." -ForegroundColor Cyan

# Check if pytorch is installed
try {
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    if (python -c "import torch; print(torch.cuda.is_available())" -eq "True") {
        python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
        python -c "import torch; print(f'GPU device: {torch.cuda.get_device_name(0)}')"
    }
} catch {
    Write-Host "PyTorch not installed or error checking CUDA" -ForegroundColor Red
}

# Check NVIDIA GPU with system commands
try {
    if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
        Write-Host "`nNVIDIA GPU Information:" -ForegroundColor Cyan
        nvidia-smi
    } else {
        Write-Host "`nNVIDIA GPU driver not found" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error checking NVIDIA GPU" -ForegroundColor Red
}
