[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$ExcludedNames = @(
    "RELEASE_MANIFEST.json",
    "RELEASE_MANIFEST.csv",
    "RELEASE_SUMMARY.json"
)

$RelativePaths = @(& git -C $Root ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0) {
    throw "git ls-files failed with exit code $LASTEXITCODE"
}

$Rows = $RelativePaths |
    Where-Object { [IO.Path]::GetFileName($_) -notin $ExcludedNames } |
    Sort-Object |
    ForEach-Object {
        $FullName = Join-Path $Root $_
        if (-not (Test-Path -LiteralPath $FullName -PathType Leaf)) {
            throw "Release file is missing from the working tree: $_"
        }
        $Item = Get-Item -LiteralPath $FullName
        [PSCustomObject]@{
            path = $_.Replace("\", "/")
            bytes = $Item.Length
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $FullName).Hash.ToLowerInvariant()
        }
    }

$Manifest = [ordered]@{
    format = "neuraedge-private-release-v2"
    generated_utc = [DateTime]::UtcNow.ToString("o")
    files = @($Rows)
}
$Manifest | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $Root "RELEASE_MANIFEST.json") -Encoding utf8
$Rows | Export-Csv -LiteralPath (Join-Path $Root "RELEASE_MANIFEST.csv") -NoTypeInformation

$Summary = [ordered]@{
    format = "neuraedge-private-release-summary-v2"
    generated_utc = $Manifest.generated_utc
    file_count = @($Rows).Count
    total_bytes = [long](($Rows | Measure-Object bytes -Sum).Sum)
    target_board = "PYNQ-Z2"
    target_part = "xc7z020clg400-1"
    accelerator_clock_mhz = 200
    float_cifar10_top1_percent = 95.55
    integer_cifar10_top1_percent = 95.42
    setup_wns_ns = 0.005
    hold_whs_ns = 0.050
    route_error_count = 0
    drc_error_count = 0
    saif_total_power_w = 1.937
    saif_dynamic_power_w = 1.785
    saif_accelerator_dynamic_power_w = 0.472
}
$Summary | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $Root "RELEASE_SUMMARY.json") -Encoding utf8

Write-Output "NEURAEDGE_RELEASE_MANIFEST files=$(@($Rows).Count) bytes=$($Summary.total_bytes)"
