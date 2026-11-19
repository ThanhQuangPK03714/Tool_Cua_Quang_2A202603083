$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$pythonPath = Join-Path $projectRoot ".venv-sam3\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "GPU environment not found. Run .\setup_gpu.ps1 first."
}

$checkpointPath = Join-Path $projectRoot "models\sam3.pt"
if (-not (Test-Path -LiteralPath $checkpointPath)) {
    & $pythonPath (Join-Path $projectRoot "src\download_sam3_checkpoint.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "SAM 3 checkpoint is unavailable; the app will still start, but SAM 3 fallback will stay disabled."
    }
}
if (Test-Path -LiteralPath $checkpointPath) {
    $env:SAM3_WEIGHTS = $checkpointPath
}

& $pythonPath -m streamlit run (Join-Path $projectRoot "app.py")
