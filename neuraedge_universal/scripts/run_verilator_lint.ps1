param(
    [string]$MsysRoot = "C:\msys64"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$MsysShell = Join-Path $MsysRoot "msys2_shell.cmd"
if (-not (Test-Path -LiteralPath $MsysShell)) {
    throw "MSYS2 shell was not found at $MsysShell"
}

Push-Location $ProjectRoot
try {
    & $MsysShell -defterm -no-start -ucrt64 -here -c "bash scripts/run_verilator_lint.sh"
    if ($LASTEXITCODE -ne 0) {
        throw "Verilator lint failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "NEURAEDGE_VERILATOR_LINT_PASSED"
