param(
    [int]$Epochs = 400,
    [int]$BatchSize = 64,
    [int]$ValBatchSize = 64,
    [int]$Workers = 4,
    [string]$OutputRoot = "outputs_upstream_spikingformer",
    [string]$Experiment = "spikingformer_4_384_400e",
    [int]$LogInterval = 200,
    [int]$CooldownEpochs = 0
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $ResolvedOutputRoot = $OutputRoot
} else {
    $ResolvedOutputRoot = Join-Path $ProjectDir $OutputRoot
}
$PauseFile = Join-Path (Join-Path $ResolvedOutputRoot $Experiment) "PAUSE"
if (Test-Path $PauseFile) {
    Remove-Item -LiteralPath $PauseFile -Force
}
& (Join-Path $ProjectDir "run_upstream_spikingformer_full.ps1") `
    -Epochs $Epochs `
    -BatchSize $BatchSize `
    -ValBatchSize $ValBatchSize `
    -Workers $Workers `
    -OutputRoot $OutputRoot `
    -Experiment $Experiment `
    -Resume auto `
    -LogInterval $LogInterval `
    -CooldownEpochs $CooldownEpochs
