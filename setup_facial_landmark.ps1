param(
    [string]$PythonExe = ""
)

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $pythonPath = $pythonCommand.Source
} elseif ([System.IO.Path]::IsPathRooted($PythonExe)) {
    $pythonPath = $PythonExe
} else {
    $pythonPath = Join-Path $PSScriptRoot $PythonExe
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python executable not found: $pythonPath"
}

& $pythonPath -m pip install --no-deps mediapipe==1.2.3
if ($LASTEXITCODE -ne 0) { throw "Could not install mediapipe" }
& $pythonPath -m pip install absl-py flatbuffers sounddevice cffi pycparser
if ($LASTEXITCODE -ne 0) { throw "Could not install MediaPipe runtime dependencies" }

$modelDir = Join-Path $PSScriptRoot "models"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
$modelPath = Join-Path $modelDir "face_landmarker.task"
if (-not (Test-Path -LiteralPath $modelPath)) {
    Invoke-WebRequest `
        -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" `
        -OutFile $modelPath
}
Write-Host "Facial Landmark model ready: $modelPath"
Write-Host "Installed MediaPipe into: $pythonPath"
