param(
    [string]$VivadoBin = "D:\AMDDesignTools\2026.1\Vivado\bin"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$Xvlog = Join-Path $VivadoBin "xvlog.bat"
$Xelab = Join-Path $VivadoBin "xelab.bat"
$Xsim = Join-Path $VivadoBin "xsim.bat"
$Vivado = Join-Path $VivadoBin "vivado.bat"

foreach ($tool in @($Xvlog, $Xelab, $Xsim, $Vivado)) {
    if (!(Test-Path $tool)) { throw "Required Vivado tool not found: $tool" }
}

Push-Location $ProjectDir
try {
    & $Xvlog -sv rtl\neuraedge_mac2_u8.sv tb\tb_neuraedge_mac2_u8.sv
    if ($LASTEXITCODE -ne 0) { throw "xvlog failed with exit code $LASTEXITCODE" }
    & $Xelab tb_neuraedge_mac2_u8 -debug typical
    if ($LASTEXITCODE -ne 0) { throw "xelab failed with exit code $LASTEXITCODE" }
    & $Xsim tb_neuraedge_mac2_u8 -runall
    if ($LASTEXITCODE -ne 0) { throw "xsim failed with exit code $LASTEXITCODE" }
    & $Vivado -mode batch -source scripts\synth_mac2_probe.tcl -notrace
    if ($LASTEXITCODE -ne 0) { throw "Vivado probe implementation failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

Write-Host "Packed MAC arithmetic and 128-DSP mapping probe passed."
