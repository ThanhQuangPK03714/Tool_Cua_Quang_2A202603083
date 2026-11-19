param(
    [string]$PythonExe = ".venv-sam3\Scripts\python.exe"
)

$pythonPath = if ([System.IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe
} else {
    Join-Path $PSScriptRoot $PythonExe
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
