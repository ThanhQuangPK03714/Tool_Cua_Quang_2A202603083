$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$venvPath = Join-Path $projectRoot ".venv-sam3"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py) is required. Install Python 3.12, then run this script again."
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3.12 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Python 3.12 environment." }
}

& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip in .venv-sam3." }

# SAM 3 upstream documents the CUDA 12.6 PyTorch wheels for its Windows-capable runtime.
& $pythonPath -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw "Could not install CUDA-enabled PyTorch." }

& $pythonPath -m pip install -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install the application dependencies." }

& $pythonPath -c "import clip" *> $null
if ($LASTEXITCODE -ne 0) {
    & $pythonPath -m pip install --timeout 120 --retries 10 -r (Join-Path $projectRoot "requirements-sam3.txt")
    if ($LASTEXITCODE -ne 0) { throw "Could not install the SAM 3 text-prompt dependency." }
}

& $pythonPath -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
if ($LASTEXITCODE -ne 0) { throw "Could not inspect the PyTorch CUDA runtime." }

Write-Host ""
Write-Host "Environment ready. Start the app with:"
Write-Host ".\run_gpu.ps1"
Write-Host ""
Write-Host "SAM 3 also needs an authorized checkpoint. Place it at models\sam3.pt or set SAM3_WEIGHTS in .env."
