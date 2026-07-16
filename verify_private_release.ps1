[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$ManifestPath = Join-Path $Root "RELEASE_MANIFEST.json"
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Release manifest is missing: $ManifestPath"
}

$Manifest = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
$Failures = [System.Collections.Generic.List[string]]::new()
foreach ($Record in $Manifest.files) {
    $Path = Join-Path $Root ([string]$Record.path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        $Failures.Add("missing: $($Record.path)")
        continue
    }
    $Item = Get-Item -LiteralPath $Path
    if ($Item.Length -ne [long]$Record.bytes) {
        $Failures.Add("size: $($Record.path)")
        continue
    }
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Hash -ne [string]$Record.sha256) {
        $Failures.Add("sha256: $($Record.path)")
    }
}
if ($Failures.Count -ne 0) {
    throw "Release verification failed:`n$($Failures -join [Environment]::NewLine)"
}

$Project = Join-Path $Root "neuraedge_universal"

$SourceManifestPaths = @(
    (Join-Path $Root "docs\neuraedge_line_by_line_source_manifest.json"),
    (Join-Path $Project "docs\neuraedge_line_by_line_source_manifest.json"),
    (Join-Path $Project "outputs\source_reference_final\neuraedge_line_by_line_source_manifest.json")
)
$SourceManifestHash = $null
foreach ($SourceManifestPath in $SourceManifestPaths) {
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceManifestPath).Hash.ToLowerInvariant()
    if ($null -eq $SourceManifestHash) {
        $SourceManifestHash = $Hash
    } elseif ($Hash -ne $SourceManifestHash) {
        throw "Source-reference manifests are not identical"
    }
}
$SourceManifest = Get-Content -Raw -LiteralPath $SourceManifestPaths[0] | ConvertFrom-Json
$ExpectedSourceSummary = @{
    file_count = 99
    source_lines = 28915
    nonblank_lines = 26565
    explained_nonblank_lines = 26565
}
foreach ($Name in $ExpectedSourceSummary.Keys) {
    if ([long]$SourceManifest.summary.$Name -ne $ExpectedSourceSummary[$Name]) {
        throw "Source-reference summary mismatch for $Name"
    }
}
foreach ($Record in $SourceManifest.files) {
    $Path = Join-Path $Project ([string]$Record.path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Source-reference input is missing: $($Record.path)"
    }
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Hash -ne [string]$Record.sha256) {
        throw "Source-reference hash mismatch: $($Record.path)"
    }
}

$Regression = @{}
Get-Content -LiteralPath (Join-Path $Project "outputs\regression\regression_summary.txt") |
    ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            $Regression[$Matches[1]] = $Matches[2]
        }
    }
if ($Regression.result -ne "PASS" -or $Regression.rtl_testbench_count -ne "15") {
    throw "Packaged regression summary is not a 15-testbench pass"
}
$UnitTestLog = Get-Content -Raw -LiteralPath (Join-Path $Project "outputs\regression\python_unittest.log")
if ($UnitTestLog -notmatch "(?m)^Ran 93 tests in " -or $UnitTestLog -notmatch "(?m)^OK\r?$") {
    throw "Packaged Python regression is not a 93-test pass"
}

$WaveformRoot = Join-Path $Project "outputs\waveforms_final"
$WaveformManifest = Get-Content -Raw -LiteralPath (Join-Path $WaveformRoot "waveform_manifest.json") |
    ConvertFrom-Json
if (@($WaveformManifest.captures).Count -ne 15) {
    throw "Waveform manifest does not contain all 15 testbenches"
}
foreach ($Capture in $WaveformManifest.captures) {
    foreach ($Kind in @("vcd", "wdb")) {
        $Relative = [string]$Capture.$Kind
        $Path = Join-Path $WaveformRoot $Relative
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Waveform capture is missing: $Relative"
        }
        if ((Get-Item -LiteralPath $Path).Length -ne [long]$Capture."${Kind}_bytes") {
            throw "Waveform capture size mismatch: $Relative"
        }
    }
    $SimulationLog = Join-Path $WaveformRoot ([string]$Capture.simulation_log)
    if (-not (Test-Path -LiteralPath $SimulationLog -PathType Leaf)) {
        throw "Waveform simulation log is missing: $($Capture.simulation_log)"
    }
}

$PowerRoot = Join-Path $Project "outputs\power_activity_full"
$PowerManifest = Get-Content -Raw -LiteralPath (Join-Path $PowerRoot "power_activity_manifest.json") |
    ConvertFrom-Json
if (@($PowerManifest.captures).Count -ne 2) {
    throw "Power-activity manifest does not contain both SAIF captures"
}
foreach ($Capture in $PowerManifest.captures) {
    $Path = Join-Path $PowerRoot ([string]$Capture.saif)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "SAIF capture is missing: $($Capture.saif)"
    }
    if ((Get-Item -LiteralPath $Path).Length -ne [long]$Capture.saif_bytes) {
        throw "SAIF capture size mismatch: $($Capture.saif)"
    }
}

$Implementation = @{}
Get-Content -LiteralPath (Join-Path $Project "outputs\pynq_full\full_implementation_summary.txt") |
    ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            $Implementation[$Matches[1]] = $Matches[2]
        }
    }
$Expected = @{
    accelerator_clock_mhz = "200.0"
    setup_wns_ns = "0.005"
    hold_whs_ns = "0.050"
    route_error_count = "0"
    drc_error_count = "0"
    dsp48_count = "110"
}
foreach ($Name in $Expected.Keys) {
    if ($Implementation[$Name] -ne $Expected[$Name]) {
        throw "Implementation summary mismatch for $Name"
    }
}

$ExpectedHardwareHashes = @{
    "neuraedge_pynq_full.bit" = "d9bcac6224efd44da8aa2f8a0e674f5e66a67294caac460d6cfc8086d5a341de"
    "neuraedge_pynq_full.hwh" = "79395d99357fa798ecd31b0c9229500dce7e7aada8185c3576b744fd4485712f"
    "neuraedge_pynq_full.xsa" = "8d06b8a0b6e82105992cbf05fd1f0d3cfac8f5316d682f976f88221b0583bd64"
    "neuraedge_pynq_full_routed.dcp" = "750e7185560d3187c7576f84b722918d3c8d3da21738d417b8fa59fc9ca857a1"
}
foreach ($Name in $ExpectedHardwareHashes.Keys) {
    $Path = Join-Path $Project "outputs\pynq_full\$Name"
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($Hash -ne $ExpectedHardwareHashes[$Name]) {
        throw "Canonical hardware hash mismatch for $Name"
    }
}

Write-Output (
    "NEURAEDGE_PRIVATE_RELEASE_VERIFIED files=$(@($Manifest.files).Count) " +
    "source_files=$(@($SourceManifest.files).Count) waveforms=$(@($WaveformManifest.captures).Count) " +
    "saif_captures=$(@($PowerManifest.captures).Count)"
)
