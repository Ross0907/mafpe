[CmdletBinding()]
param(
    [string]$VivadoRoot = "D:\AMDDesignTools\2026.1\Vivado",
    [ValidateSet("validate", "full")]
    [string]$Mode = "full",
    [switch]$CleanLogs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Vivado = Join-Path $VivadoRoot "bin\vivado.bat"
$Tcl = Join-Path $PSScriptRoot "build_pynq_full.tcl"
$DiagramTcl = Join-Path $PSScriptRoot "export_pynq_diagrams.tcl"
$OutputRoot = Join-Path $ProjectRoot "outputs\pynq_full"
$Summary = Join-Path $ProjectRoot "outputs\pynq_full\full_implementation_summary.txt"
$BuildRoot = if ([string]::IsNullOrWhiteSpace($env:NEURAEDGE_VIVADO_BUILD_ROOT)) {
    "D:\NeuraEdgeBuild"
} else {
    [System.IO.Path]::GetFullPath($env:NEURAEDGE_VIVADO_BUILD_ROOT)
}
$VivadoProject = Join-Path $BuildRoot "pynq_full\neuraedge_pynq_full.xpr"

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
}

Push-Location $ProjectRoot
try {
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    $Log = Join-Path $OutputRoot "vivado_${Mode}.log"
    $Journal = Join-Path $OutputRoot "vivado_${Mode}.jou"
    & $Vivado -mode batch -log $Log -journal $Journal -source $Tcl -tclargs $Mode
    if ($LASTEXITCODE -ne 0) {
        throw "PYNQ $Mode build failed with exit code $LASTEXITCODE. See $Log"
    }

    $DiagramLog = Join-Path $OutputRoot "vivado_diagrams.log"
    $DiagramJournal = Join-Path $OutputRoot "vivado_diagrams.jou"
    & $Vivado -mode gui -log $DiagramLog -journal $DiagramJournal `
        -source $DiagramTcl -tclargs $VivadoProject $OutputRoot
    if ($LASTEXITCODE -ne 0) {
        throw "PYNQ diagram export failed with exit code $LASTEXITCODE. See $DiagramLog"
    }
    foreach ($Diagram in @("neuraedge_pynq_block_design.pdf",
            "neuraedge_pynq_block_design.svg")) {
        $DiagramPath = Join-Path $OutputRoot $Diagram
        if (-not (Test-Path -LiteralPath $DiagramPath) -or
                (Get-Item -LiteralPath $DiagramPath).Length -eq 0) {
            throw "PYNQ diagram export did not create $DiagramPath"
        }
    }
} finally {
    Pop-Location
}

if ($Mode -eq "full") {
    Get-Content -LiteralPath $Summary
} else {
    Write-Host "NEURAEDGE_PYNQ_VALIDATION_PASSED report=$OutputRoot\validated_ip_status.rpt"
}
