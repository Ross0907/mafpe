param(
    [string]$Python = "py",
    [switch]$FullExportDeps
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $RepoRoot ".venv"
$PyExe = Join-Path $Venv "Scripts\python.exe"

if (!(Test-Path $PyExe)) {
    & $Python -3.10 -m venv $Venv
    if ($LASTEXITCODE -ne 0) {
        & $Python -m venv $Venv
    }
}

& $PyExe -m pip install --upgrade pip
& $PyExe -m pip install -r (Join-Path $RepoRoot "requirements_runtime.txt")
if ($FullExportDeps) {
    & $PyExe -m pip install -r (Join-Path $RepoRoot "requirements_export.txt")
}

Write-Host "[done] Python environment ready at $PyExe"