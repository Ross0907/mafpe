[CmdletBinding()]
param(
    [string]$VivadoRoot = "D:\AMDDesignTools\2026.1\Vivado",
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "outputs\power_activity"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$WorkDirectory = Join-Path $OutputDirectory "xsim_work"
$LogDirectory = Join-Path $OutputDirectory "logs"
$SaifDirectory = Join-Path $OutputDirectory "saif"
foreach ($Directory in @($OutputDirectory, $WorkDirectory, $LogDirectory, $SaifDirectory)) {
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

$PackageFile = Join-Path $ProjectRoot "rtl\neuraedge_command_pkg.sv"
$RtlFiles = @($PackageFile) + @(
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "rtl") -Filter "*.sv" -File |
        Where-Object { $_.FullName -ne $PackageFile } |
        Sort-Object Name |
        ForEach-Object FullName
)
$CaptureTcl = (Join-Path $PSScriptRoot "capture_power_activity.tcl").Replace('\', '/')
$Scenarios = @(
    [PSCustomObject]@{
        Top = "tb_neuraedge_pynq_axi_top"
        Scope = "/tb_neuraedge_pynq_axi_top/dut"
        Amount = 5
        Unit = "us"
    },
    [PSCustomObject]@{
        Top = "tb_neuraedge_universal_core"
        Scope = "/tb_neuraedge_universal_core/dut"
        Amount = 50
        Unit = "us"
    }
)
$ManifestRows = [System.Collections.Generic.List[object]]::new()

Push-Location $WorkDirectory
try {
    & $Xvlog -sv -i $ProjectRoot @RtlFiles 2>&1 |
        Tee-Object -FilePath (Join-Path $LogDirectory "rtl_compile.log")
    if ($LASTEXITCODE -ne 0) {
        throw "RTL xvlog compilation failed with exit code $LASTEXITCODE"
    }

    foreach ($Scenario in $Scenarios) {
        $Top = $Scenario.Top
        $Snapshot = "${Top}_power_snapshot"
        $SaifPath = Join-Path $SaifDirectory "$Top.saif"
        Remove-Item -LiteralPath $SaifPath -Force -ErrorAction SilentlyContinue
        & $Xvlog -sv -i $ProjectRoot (Join-Path $ProjectRoot "tb\$Top.sv") 2>&1 |
            Tee-Object -FilePath (Join-Path $LogDirectory "$Top.compile.log")
        if ($LASTEXITCODE -ne 0) {
            throw "$Top xvlog compilation failed with exit code $LASTEXITCODE"
        }
        & $Xelab -debug typical -timescale 1ns/1ps -snapshot $Snapshot "work.$Top" 2>&1 |
            Tee-Object -FilePath (Join-Path $LogDirectory "$Top.elaborate.log")
        if ($LASTEXITCODE -ne 0) {
            throw "$Top xelab failed with exit code $LASTEXITCODE"
        }

        $env:NEURAEDGE_SAIF_PATH = $SaifPath
        $env:NEURAEDGE_SAIF_SCOPE = $Scenario.Scope
        $env:NEURAEDGE_SAIF_AMOUNT = [string]$Scenario.Amount
        $env:NEURAEDGE_SAIF_UNIT = $Scenario.Unit
        & $Xsim $Snapshot -onerror quit -onfinish quit -tclbatch $CaptureTcl 2>&1 |
            Tee-Object -FilePath (Join-Path $LogDirectory "$Top.simulate.log")
        if ($LASTEXITCODE -ne 0) {
            throw "$Top XSim SAIF capture failed with exit code $LASTEXITCODE"
        }
        if (-not (Test-Path -LiteralPath $SaifPath) -or
                (Get-Item -LiteralPath $SaifPath).Length -eq 0) {
            throw "$Top did not produce a nonempty SAIF"
        }
        $ManifestRows.Add([PSCustomObject]@{
            testbench = $Top
            scope = $Scenario.Scope
            duration = $Scenario.Amount
            unit = $Scenario.Unit
            saif = [System.IO.Path]::GetRelativePath($OutputDirectory, $SaifPath)
            saif_bytes = (Get-Item -LiteralPath $SaifPath).Length
        })
    }
} finally {
    foreach ($Name in @("NEURAEDGE_SAIF_PATH", "NEURAEDGE_SAIF_SCOPE",
            "NEURAEDGE_SAIF_AMOUNT", "NEURAEDGE_SAIF_UNIT")) {
        Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
    }
    Pop-Location
}

$Manifest = [ordered]@{
    format = "neuraedge-xsim-saif-v1"
    vivado_root = $VivadoRoot
    target_part = "xc7z020clg400-1"
    target_clock_mhz = 200
    captures = @($ManifestRows)
}
$ManifestPath = Join-Path $OutputDirectory "power_activity_manifest.json"
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestPath -Encoding utf8
Write-Output "NEURAEDGE_POWER_ACTIVITY_CAPTURED count=$($ManifestRows.Count) manifest=$ManifestPath"
