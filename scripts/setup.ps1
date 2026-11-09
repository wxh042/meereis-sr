$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath ".venv/Scripts/python.exe")) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 environment creation failed" }
}
& ".venv/Scripts/python.exe" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& ".venv/Scripts/python.exe" scripts/verify.py
if ($LASTEXITCODE -ne 0) { throw "Verification failed" }
