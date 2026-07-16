param(
    [string]$DatasetRoot = "",
    [string]$OutputDirectory = "",
    [int]$ShardSize = 1000
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = (Resolve-Path (Join-Path $ProjectRoot "..\datasets")).Path
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot "outputs\cifar10_shards"
}
$Python = (Resolve-Path (Join-Path $ProjectRoot "..\external_research\spikingformer_py310\Scripts\python.exe")).Path
& $Python (Join-Path $ProjectRoot "software\export_cifar10_shards.py") `
    --dataset-root $DatasetRoot `
    --output-dir $OutputDirectory `
    --shard-size $ShardSize `
    --no-download
if ($LASTEXITCODE -ne 0) {
    throw "CIFAR-10 shard export failed with exit code $LASTEXITCODE"
}
Write-Host "NEURAEDGE_CIFAR10_SHARDS_READY directory=$OutputDirectory"
