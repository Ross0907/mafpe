[CmdletBinding()]
param(
    [string]$VivadoRoot = "D:\AMDDesignTools\2026.1\Vivado",
    [string]$Checkpoint = "",
    [string]$OutputDirectory = "",
    [string]$SaifDirectory = "",
    [string]$AcceleratorInstance = "__ROOT__",
    [switch]$ReuseSaif
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Join-Path $ProjectRoot "outputs\pynq_ooc\neuraedge_pynq_axi_top_routed.dcp"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "outputs\power_activity"
}
$Checkpoint = [System.IO.Path]::GetFullPath($Checkpoint)
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if ([string]::IsNullOrWhiteSpace($SaifDirectory)) {
    $SaifDirectory = $OutputDirectory
}
$SaifDirectory = [System.IO.Path]::GetFullPath($SaifDirectory)

$Vivado = Join-Path $VivadoRoot "bin\vivado.bat"
$Tcl = Join-Path $PSScriptRoot "annotate_saif_power.tcl"
$WrapperSaif = Join-Path $SaifDirectory "saif\tb_neuraedge_pynq_axi_top.saif"
$CoreSaif = Join-Path $SaifDirectory "saif\tb_neuraedge_universal_core.saif"
if (-not $ReuseSaif) {
    & (Join-Path $PSScriptRoot "capture_power_activity.ps1") `
        -VivadoRoot $VivadoRoot -OutputDirectory $SaifDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "SAIF capture failed with exit code $LASTEXITCODE"
    }
}
foreach ($Saif in @($WrapperSaif, $CoreSaif)) {
    if (-not (Test-Path -LiteralPath $Saif -PathType Leaf)) {
        throw "Required SAIF does not exist: $Saif"
    }
    if ((Get-Item -LiteralPath $Saif).Length -eq 0) {
        throw "Required SAIF is empty: $Saif"
    }
}
$ReportDirectory = Join-Path $OutputDirectory "vivado_reports"
& $Vivado -mode batch -nojournal -nolog -source $Tcl -tclargs `
    $Checkpoint $WrapperSaif $CoreSaif $ReportDirectory $AcceleratorInstance
if ($LASTEXITCODE -ne 0) {
    throw "Vivado SAIF power analysis failed with exit code $LASTEXITCODE"
}
Write-Output "NEURAEDGE_SAIF_POWER_COMPLETE report=$ReportDirectory\routed_saif_power.rpt"
