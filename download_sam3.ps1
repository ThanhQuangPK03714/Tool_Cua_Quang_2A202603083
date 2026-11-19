$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$pythonPath = Join-Path $projectRoot ".venv-sam3\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "GPU environment not found. Run .\setup_gpu.ps1 first."
}

& $pythonPath (Join-Path $projectRoot "src\download_sam3_checkpoint.py")
if ($LASTEXITCODE -ne 0) {
    throw "SAM 3 download failed. Request checkpoint access and authenticate with Hugging Face, then retry."
}
