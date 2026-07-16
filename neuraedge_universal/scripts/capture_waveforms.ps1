[CmdletBinding()]
param(
    [string]$VivadoRoot = "D:\AMDDesignTools\2026.1\Vivado",
    [string]$OutputDirectory = "",
    [string[]]$Testbench = @()
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "outputs\waveforms"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$WorkDirectory = Join-Path $OutputDirectory "xsim_work"
$VcdDirectory = Join-Path $OutputDirectory "vcd"
$WdbDirectory = Join-Path $OutputDirectory "wdb"
$LogDirectory = Join-Path $OutputDirectory "logs"
foreach ($Directory in @($OutputDirectory, $WorkDirectory, $VcdDirectory, $WdbDirectory, $LogDirectory)) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}
$FixtureStage = Join-Path $WorkDirectory "tb\fixtures"
New-Item -ItemType Directory -Force -Path $FixtureStage | Out-Null
Copy-Item -Path (Join-Path $ProjectRoot "tb\fixtures\*") -Destination $FixtureStage -Force

$VivadoBin = Join-Path $VivadoRoot "bin"
$Xvlog = Join-Path $VivadoBin "xvlog.bat"
$Xelab = Join-Path $VivadoBin "xelab.bat"
$Xsim = Join-Path $VivadoBin "xsim.bat"
foreach ($Tool in @($Xvlog, $Xelab, $Xsim)) {
    if (-not (Test-Path -LiteralPath $Tool)) {
        throw "Required Vivado simulator tool was not found: $Tool"
    }
}

$Windows = [ordered]@{
    "tb_neuraedge_mac2_u8" = @(2, "us")
    "tb_neuraedge_tensor_tile" = @(3, "us")
    "tb_neuraedge_buffered_tensor_core" = @(4, "us")
    "tb_neuraedge_alpha_xnor" = @(40, "us")
    "tb_neuraedge_vector_alu" = @(170, "us")
    "tb_neuraedge_integer_primitives" = @(90, "us")
    "tb_neuraedge_normalization_units" = @(405, "us")
    "tb_neuraedge_lif_bank" = @(20, "us")
    "tb_neuraedge_linear_backward8" = @(1, "ms")
    "tb_neuraedge_sgd_optimizer8" = @(20, "us")
    "tb_neuraedge_activation_lut8" = @(20, "us")
    "tb_neuraedge_universal_core" = @(50, "us")
    "tb_neuraedge_descriptor_sequencer" = @(2, "us")
    "tb_neuraedge_pynq_descriptor" = @(2, "us")
    "tb_neuraedge_pynq_axi_top" = @(5, "us")
}

if ($Testbench.Count -eq 0) {
    $Selected = @($Windows.Keys)
} else {
    $Selected = @($Testbench)
    foreach ($Top in $Selected) {
        if (-not $Windows.Contains($Top)) {
            throw "Unknown waveform testbench '$Top'"
        }
    }
}

$PackageFile = Join-Path $ProjectRoot "rtl\neuraedge_command_pkg.sv"
$RtlFiles = @($PackageFile) + @(
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "rtl") -Filter "*.sv" -File |
        Where-Object { $_.FullName -ne $PackageFile } |
        Sort-Object Name |
        ForEach-Object FullName
)
$CaptureTcl = Join-Path $PSScriptRoot "capture_waveform.tcl"
$CaptureTclArgument = $CaptureTcl.Replace('\', '/')
$ManifestRows = [System.Collections.Generic.List[object]]::new()

Push-Location $WorkDirectory
try {
    & $Xvlog -sv -i $ProjectRoot @RtlFiles 2>&1 |
        Tee-Object -FilePath (Join-Path $LogDirectory "rtl_compile.log")
    if ($LASTEXITCODE -ne 0) {
        throw "RTL xvlog compilation failed with exit code $LASTEXITCODE"
    }

    foreach ($Top in $Selected) {
        $TestbenchFile = Join-Path $ProjectRoot "tb\$Top.sv"
        $CompileLog = Join-Path $LogDirectory "$Top.compile.log"
        $ElaborateLog = Join-Path $LogDirectory "$Top.elaborate.log"
        $SimulationLog = Join-Path $LogDirectory "$Top.simulate.log"
        $Snapshot = "${Top}_wave_snapshot"
        $VcdPath = Join-Path $VcdDirectory "$Top.vcd"
        $WdbPath = Join-Path $WdbDirectory "$Top.wdb"
        $Amount = [int]$Windows[$Top][0]
        $Unit = [string]$Windows[$Top][1]
        Remove-Item -LiteralPath $VcdPath -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $WdbPath -Force -ErrorAction SilentlyContinue

        & $Xvlog -sv -i $ProjectRoot $TestbenchFile 2>&1 |
            Tee-Object -FilePath $CompileLog
        if ($LASTEXITCODE -ne 0) {
            throw "$Top xvlog compilation failed with exit code $LASTEXITCODE"
        }

        & $Xelab -debug typical -timescale 1ns/1ps -snapshot $Snapshot "work.$Top" 2>&1 |
            Tee-Object -FilePath $ElaborateLog
        if ($LASTEXITCODE -ne 0) {
            throw "$Top xelab failed with exit code $LASTEXITCODE"
        }

        $env:NEURAEDGE_VCD_PATH = $VcdPath
        $env:NEURAEDGE_WAVE_AMOUNT = [string]$Amount
        $env:NEURAEDGE_WAVE_UNIT = $Unit
        & $Xsim $Snapshot -onerror quit -onfinish quit -wdb $WdbPath `
            -tclbatch $CaptureTclArgument 2>&1 |
            Tee-Object -FilePath $SimulationLog
        if ($LASTEXITCODE -ne 0) {
            throw "$Top XSim waveform capture failed with exit code $LASTEXITCODE"
        }
        if (-not (Test-Path -LiteralPath $VcdPath) -or (Get-Item -LiteralPath $VcdPath).Length -eq 0) {
            throw "$Top did not produce a nonempty VCD"
        }

        $ManifestRows.Add([PSCustomObject]@{
            testbench = $Top
            duration = $Amount
            unit = $Unit
            vcd = [System.IO.Path]::GetRelativePath($OutputDirectory, $VcdPath)
            vcd_bytes = (Get-Item -LiteralPath $VcdPath).Length
            wdb = [System.IO.Path]::GetRelativePath($OutputDirectory, $WdbPath)
            wdb_bytes = (Get-Item -LiteralPath $WdbPath).Length
            simulation_log = [System.IO.Path]::GetRelativePath($OutputDirectory, $SimulationLog)
        })
    }
} finally {
    Remove-Item Env:NEURAEDGE_VCD_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:NEURAEDGE_WAVE_AMOUNT -ErrorAction SilentlyContinue
    Remove-Item Env:NEURAEDGE_WAVE_UNIT -ErrorAction SilentlyContinue
    Pop-Location
}

$Manifest = [ordered]@{
    format = "neuraedge-xsim-waveforms-v1"
    vivado_root = $VivadoRoot
    target_part = "xc7z020clg400-1"
    target_clock_mhz = 200
    captures = @($ManifestRows)
}
$ManifestPath = Join-Path $OutputDirectory "waveform_manifest.json"
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestPath -Encoding utf8
Write-Output "NEURAEDGE_WAVEFORMS_CAPTURED count=$($ManifestRows.Count) manifest=$ManifestPath"
