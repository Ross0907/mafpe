[CmdletBinding()]
param(
    [string]$VivadoRoot = "D:\AMDDesignTools\2026.1\Vivado",
    [switch]$CleanLogs,
    [switch]$Resume,
    [ValidateSet("Explore", "AggressiveExplore", "Default")]
    [string]$ResumeDirective = "Explore"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Vivado = Join-Path $VivadoRoot "bin\vivado.bat"
$Tcl = Join-Path $PSScriptRoot "build_pynq_ooc.tcl"
$ResumeTcl = Join-Path $PSScriptRoot "resume_pynq_ooc.tcl"
$OutputRoot = Join-Path $ProjectRoot "outputs\pynq_ooc"
$ResumeCheckpoint = Join-Path $OutputRoot "pre_postroute_physopt.dcp"

if (-not (Test-Path -LiteralPath $Vivado)) {
    throw "Vivado was not found at $Vivado"
}

if ($CleanLogs) {
    foreach ($Name in @("vivado.jou", "vivado.log")) {
        $Path = Join-Path $ProjectRoot $Name
        if (Test-Path -LiteralPath $Path) {
            Remove-Item -LiteralPath $Path -Force
        }
    }
    if (-not $Resume) {
        foreach ($Name in @(
            "pre_postroute_physopt.dcp",
            "postroute_physopt_unrouted.dcp",
            "implementation_summary.txt"
        )) {
            $Path = Join-Path $OutputRoot $Name
            if (Test-Path -LiteralPath $Path) {
                Remove-Item -LiteralPath $Path -Force
            }
        }
    }
}

Push-Location $ProjectRoot
try {
    if ($Resume) {
        if (-not (Test-Path -LiteralPath $ResumeCheckpoint)) {
            throw "No routed resume checkpoint was found at $ResumeCheckpoint"
        }
        & $Vivado -mode batch -nojournal -nolog -source $ResumeTcl `
            -tclargs pynq_ooc $ResumeDirective
        if ($LASTEXITCODE -ne 0) {
            throw "Vivado resume failed with exit code $LASTEXITCODE"
        }
    } else {
        & $Vivado -mode batch -nojournal -nolog -source $Tcl
        $BuildExitCode = $LASTEXITCODE
        if ($BuildExitCode -ne 0 -and (Test-Path -LiteralPath $ResumeCheckpoint)) {
            Write-Host "Primary Vivado flow exited with $BuildExitCode; resuming the saved route with $ResumeDirective."
            & $Vivado -mode batch -nojournal -nolog -source $ResumeTcl `
                -tclargs pynq_ooc $ResumeDirective
            if ($LASTEXITCODE -ne 0) {
                throw "Vivado implementation and routed-checkpoint resume both failed; primary=$BuildExitCode resume=$LASTEXITCODE"
            }
        } elseif ($BuildExitCode -ne 0) {
            throw "Vivado implementation failed with exit code $BuildExitCode before a routed checkpoint was available"
        }
    }
} finally {
    Pop-Location
}

Get-Content -LiteralPath (Join-Path $ProjectRoot "outputs\pynq_ooc\implementation_summary.txt")
