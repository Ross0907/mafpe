param(
    [string]$BoardHost = "xilinx@192.168.2.99",
    [string]$RemoteRoot = "/home/xilinx/neuraedge_universal",
    [string]$DataDirectory = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DataDirectory)) {
    $DataDirectory = Join-Path $ProjectRoot "outputs\cifar10_shards"
}
$DataDirectory = (Resolve-Path -LiteralPath $DataDirectory).Path
$Bitfile = Join-Path $ProjectRoot "outputs\pynq_full\neuraedge_pynq_full.bit"
$Hwh = Join-Path $ProjectRoot "outputs\pynq_full\neuraedge_pynq_full.hwh"
$ParameterDirectory = Join-Path $ProjectRoot "outputs\spikingformer_hardware_parameters"
$ParameterManifest = Join-Path $ParameterDirectory "spikingformer_neuraedge_weights.json"
$ParameterBinary = Join-Path $ParameterDirectory "spikingformer_neuraedge_weights.bin"
$DataManifest = Join-Path $DataDirectory "cifar10_shards_manifest.json"
foreach ($Required in @($Bitfile, $Hwh, $ParameterManifest, $ParameterBinary, $DataManifest)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required deployment artifact is missing: $Required"
    }
}

$RemoteDirectories = @(
    "$RemoteRoot/software",
    "$RemoteRoot/outputs/pynq_full",
    "$RemoteRoot/outputs/spikingformer_hardware_parameters",
    "$RemoteRoot/outputs/cifar10_shards",
    "$RemoteRoot/outputs/pynq_training"
) -join " "
& ssh $BoardHost "mkdir -p $RemoteDirectories"
if ($LASTEXITCODE -ne 0) { throw "Could not create remote PYNQ directories" }

$SoftwareFiles = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "software") -Filter "*.py" -File |
    Sort-Object Name |
    ForEach-Object FullName
& scp @SoftwareFiles "${BoardHost}:$RemoteRoot/software/"
if ($LASTEXITCODE -ne 0) { throw "PYNQ software transfer failed" }
& scp $Bitfile $Hwh "${BoardHost}:$RemoteRoot/outputs/pynq_full/"
if ($LASTEXITCODE -ne 0) { throw "PYNQ overlay transfer failed" }
& scp $ParameterManifest $ParameterBinary `
    "${BoardHost}:$RemoteRoot/outputs/spikingformer_hardware_parameters/"
if ($LASTEXITCODE -ne 0) { throw "PYNQ parameter transfer failed" }
$DataFiles = Get-ChildItem -LiteralPath $DataDirectory -Filter "*.npz" -File |
    Sort-Object Name |
    ForEach-Object FullName
& scp $DataManifest @DataFiles "${BoardHost}:$RemoteRoot/outputs/cifar10_shards/"
if ($LASTEXITCODE -ne 0) { throw "PYNQ CIFAR-10 transfer failed" }
Write-Host "NEURAEDGE_PYNQ_DEPLOYED host=$BoardHost root=$RemoteRoot"
