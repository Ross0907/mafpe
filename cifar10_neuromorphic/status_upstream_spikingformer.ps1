param(
    [string]$OutputRoot = "outputs_upstream_spikingformer",
    [string]$Experiment = "spikingformer_4_384_400e"
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
if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    $ResolvedOutputRoot = $OutputRoot
} else {
    $ResolvedOutputRoot = Join-Path $ProjectDir $OutputRoot
}
$OutputDir = Join-Path $ResolvedOutputRoot $Experiment
$RunStateFile = Join-Path $OutputDir "run_state.json"
$JobFile = Join-Path $OutputDir "job.json"
$SummaryFile = Join-Path $OutputDir "summary.csv"
$PauseFile = Join-Path $OutputDir "PAUSE"

$job = $null
if (Test-Path $JobFile) {
    $job = Get-Content -LiteralPath $JobFile -Raw | ConvertFrom-Json
}
$state = $null
if (Test-Path $RunStateFile) {
    $state = Get-Content -LiteralPath $RunStateFile -Raw | ConvertFrom-Json
}
$last = $null
$best = $null
if (Test-Path $SummaryFile) {
    $rows = @(
        foreach ($row in (Import-Csv -LiteralPath $SummaryFile)) {
            $epochNumber = 0
            $top1Number = 0.0
            if ([int]::TryParse([string]$row.epoch, [ref]$epochNumber) -and
                [double]::TryParse([string]$row.eval_top1, [ref]$top1Number)) {
                $row
            }
        }
    )
    if ($rows.Count -gt 0) {
        $last = $rows[-1]
        $best = $rows | Sort-Object { [double]$_.eval_top1 } -Descending | Select-Object -First 1
    }
}

$pidValue = if ($state -and $state.pid) { [int]$state.pid } elseif ($job -and $job.pid) { [int]$job.pid } else { 0 }
$alive = $false
if ($pidValue -gt 0) {
    $alive = $null -ne (Get-Process -Id $pidValue -ErrorAction SilentlyContinue)
}

Write-Host "Upstream Spikingformer CIFAR-10 job"
Write-Host "  output_dir:  $OutputDir"
Write-Host "  status:      $(if ($state) { $state.status } elseif ($job) { $job.status } else { 'unknown' })"
Write-Host "  pid:         $pidValue alive=$alive"
if ($state) {
    $currentEpoch = $state.current_epoch
    if (($null -eq $currentEpoch -or "$currentEpoch" -eq "") -and $last) {
        $currentEpoch = [int]$last.epoch + 1
    }
    $bestMetric = if ($state) { $state.best_metric } else { $null }
    $bestEpoch = if ($state) { $state.best_epoch } else { $null }
    if ($best) {
        $bestMetric = $best.eval_top1
        $bestEpoch = $best.epoch
    }
    Write-Host "  epoch:       $currentEpoch / $($state.target_epochs)"
    Write-Host "  best_top1:   $bestMetric at epoch $bestEpoch"
}
if ($last) {
    Write-Host "  last_epoch:  $($last.epoch)"
    Write-Host "  last_top1:   $($last.eval_top1)"
    Write-Host "  last_loss:   $($last.eval_loss)"
}
Write-Host "  pause_file:  $PauseFile exists=$(Test-Path $PauseFile)"
if ($job) {
    Write-Host "  out_log:     $($job.out_log)"
    Write-Host "  err_log:     $($job.err_log)"
}
