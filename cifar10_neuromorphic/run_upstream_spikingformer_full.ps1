param(
    [int]$Epochs = 400,
    [int]$BatchSize = 64,
    [int]$ValBatchSize = 64,
    [int]$Workers = 4,
    [string]$OutputRoot = "outputs_upstream_spikingformer",
    [string]$Experiment = "spikingformer_4_384_400e",
    [string]$Resume = "auto",
    [switch]$Fresh,
    [switch]$Foreground,
    [int]$LogInterval = 200,
    [int]$CooldownEpochs = 0
)

$ErrorActionPreference = "Stop"
$ScriptBoundParameters = @{} + $PSBoundParameters

function Use-PowerShell7 {
    param([hashtable]$BoundParameters)
    if ($PSVersionTable.PSEdition -eq "Core" -and $PSVersionTable.PSVersion.Major -ge 7) { return }
    $pwsh = (Get-Command pwsh -CommandType Application -ErrorAction SilentlyContinue).Source
    if (!$pwsh) { throw "PowerShell 7+ (pwsh) is required. Install it with: winget install --id Microsoft.PowerShell --source winget" }
    $argList = @("-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath)
    foreach ($key in $BoundParameters.Keys) {
        $value = $BoundParameters[$key]
        if ($value -is [System.Management.Automation.SwitchParameter]) {
            if ($value.IsPresent) { $argList += "-$key" }
        } elseif ($null -ne $value) {
            $argList += "-$key"
            $argList += "$value"
        }
    }
    & $pwsh @argList
    exit $LASTEXITCODE
}
Use-PowerShell7 -BoundParameters $ScriptBoundParameters

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ProjectDir "..")
$UpstreamDir = Join-Path $RepoRoot "external_research\Spikingformer\cifar10"
$Python = Join-Path $RepoRoot "external_research\spikingformer_py310\Scripts\python.exe"
$TorchLib = Join-Path $RepoRoot "external_research\spikingformer_py310\Lib\site-packages\torch\lib"
$DataDir = Join-Path $RepoRoot "datasets"

if (!(Test-Path $Python)) {
    throw "Missing upstream Python 3.10 venv at $Python. Recreate it with Python 3.10 and the documented Spikingformer dependencies."
}
if (!(Test-Path $UpstreamDir)) {
    throw "Missing upstream Spikingformer trainer at $UpstreamDir."
}
if (Test-Path $TorchLib) {
    $env:PATH = "$TorchLib;$env:PATH"
}

if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $ResolvedOutputRoot = $OutputRoot
} else {
    $ResolvedOutputRoot = Join-Path $ProjectDir $OutputRoot
}

$OutputDir = Join-Path $ResolvedOutputRoot $Experiment
if ($Fresh -and (Test-Path $OutputDir)) {
    $Experiment = "{0}_{1}" -f $Experiment, (Get-Date -Format "yyyyMMdd_HHmmss")
    $OutputDir = Join-Path $ResolvedOutputRoot $Experiment
}

New-Item -ItemType Directory -Force -Path $ResolvedOutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$PauseFile = Join-Path $OutputDir "PAUSE"
$RunStateFile = Join-Path $OutputDir "run_state.json"
$JobFile = Join-Path $OutputDir "job.json"
$LogStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $OutputDir "train_$LogStamp.out.log"
$ErrLog = Join-Path $OutputDir "train_$LogStamp.err.log"

if (!$Fresh -and (Test-Path $PauseFile)) {
    Remove-Item -LiteralPath $PauseFile -Force
}

$resumePath = ""
if (!$Fresh -and $Resume -ne "") {
    if ($Resume -eq "auto") {
        $candidate = Join-Path $OutputDir "last.pth.tar"
        if (Test-Path $candidate) {
            $resumePath = $candidate
        }
    } else {
        $resumePath = $Resume
    }
}

$trainArgs = @(
    "train.py",
    "--config", "cifar10.yml",
    "-data-dir", "$DataDir",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--val-batch-size", "$ValBatchSize",
    "--workers", "$Workers",
    "--cooldown-epochs", "$CooldownEpochs",
    "--output", "$ResolvedOutputRoot",
    "--experiment", "$Experiment",
    "--pause-file", "$PauseFile",
    "--run-state-file", "$RunStateFile",
    "--log-interval", "$LogInterval"
)
if ($resumePath -ne "") {
    $trainArgs += @("--resume", "$resumePath")
}

$job = [ordered]@{
    status = if ($Foreground) { "foreground_starting" } else { "background_starting" }
    output_dir = "$OutputDir"
    output_root = "$ResolvedOutputRoot"
    experiment = "$Experiment"
    run_state = "$RunStateFile"
    pause_file = "$PauseFile"
    out_log = "$OutLog"
    err_log = "$ErrLog"
    resume = "$resumePath"
    command = "$Python " + ($trainArgs -join " ")
    updated_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}
$job | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $JobFile -Encoding UTF8

if ($Foreground) {
    Push-Location $UpstreamDir
    try {
        & $Python @trainArgs
    } finally {
        Pop-Location
    }
    exit $LASTEXITCODE
}

$process = Start-Process -FilePath $Python `
    -ArgumentList $trainArgs `
    -WorkingDirectory $UpstreamDir `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru `
    -WindowStyle Hidden

$job.status = "running"
$job.pid = $process.Id
$job.started_unix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$job.updated_unix = $job.started_unix
$job | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $JobFile -Encoding UTF8

Write-Host "Started upstream Spikingformer CIFAR-10 run"
Write-Host "  pid:        $($process.Id)"
Write-Host "  output_dir: $OutputDir"
Write-Host "  run_state:  $RunStateFile"
Write-Host "  log:        $OutLog"
