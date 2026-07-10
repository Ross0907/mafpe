param(
    [int]$Samples = 10000,
    [int]$BatchSize = 64,
    [int]$Workers = 4,
    [int]$TraceSamples = 8,
    [int]$InferSkipAbsThresh = 2048,
    [string]$VivadoBin = "D:\AMDDesignTools\2026.1\Vivado\bin",
    [string]$Part = "xc7z020clg400-1",
    [double]$ClockPeriodNs = 10.0,
    [string]$Python = "",
    [switch]$Reexport,
    [switch]$SkipVivado,
    [switch]$SkipOnChipLearning,
    [switch]$SkipSynthesis,
    [switch]$SkipActivityPower,
    [switch]$SkipReports
)

$ErrorActionPreference = "Stop"

function Assert-StepSucceeded {
    param(
        [string]$Step,
        [bool]$CommandSucceeded,
        [object]$ExitCode
    )
    if (!$CommandSucceeded) {
        throw "$Step failed"
    }
    if ($null -ne $ExitCode -and $ExitCode -ne 0) {
        throw "$Step failed with exit code $ExitCode"
    }
}

function Resolve-Python {
    param([string]$Requested)
    if ($Requested -ne "") {
        return $Requested
    }
    $RepoRootLocal = Split-Path -Parent $MyInvocation.MyCommand.Path
    $Candidates = @(
        (Join-Path $RepoRootLocal ".venv\Scripts\python.exe"),
        (Join-Path $RepoRootLocal "external_research\spikingformer_py310\Scripts\python.exe"),
        (Join-Path (Resolve-Path (Join-Path $RepoRootLocal "..\..") -ErrorAction SilentlyContinue) "external_research\spikingformer_py310\Scripts\python.exe")
    )
    foreach ($Candidate in $Candidates) {
        if ($Candidate -and (Test-Path $Candidate)) {
            return $Candidate
        }
    }
    return "python"
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Resolve-Python $Python
$Pipeline = Join-Path $RepoRoot "cifar10_neuromorphic\run_spikingformer_head_accelerator.ps1"
$ActivityPipeline = Join-Path $RepoRoot "cifar10_neuromorphic\run_spikingformer_head_activity_power.ps1"
$ReportScript = Join-Path $RepoRoot "cifar10_neuromorphic\generate_spikingformer_debug_reports.py"

if (!(Test-Path $Pipeline)) { throw "Missing accelerator pipeline: $Pipeline" }

Push-Location $RepoRoot
try {
    Write-Host "[private] running NeuraEdge accelerator verification from private release"
    & $Pipeline `
        -Samples $Samples `
        -BatchSize $BatchSize `
        -Workers $Workers `
        -TraceSamples $TraceSamples `
        -InferSkipAbsThresh $InferSkipAbsThresh `
        -VivadoBin $VivadoBin `
        -Part $Part `
        -ClockPeriodNs $ClockPeriodNs `
        -SkipExport:(!$Reexport) `
        -SkipVivado:$SkipVivado `
        -SkipOnChipLearning:$SkipOnChipLearning `
        -SkipSynthesis:$SkipSynthesis
    Assert-StepSucceeded "NeuraEdge CIFAR-10 accelerator pipeline" $? $LASTEXITCODE

    if (!$SkipActivityPower -and !$SkipVivado -and !$SkipSynthesis) {
        Write-Host "[power] regenerating routed SAIF power reports"
        & $ActivityPipeline `
            -VivadoBin $VivadoBin `
            -Part $Part `
            -ClockPeriodNs $ClockPeriodNs `
            -SkipInferenceSaif `
            -SkipLearningSaif `
            -Python $Python
        Assert-StepSucceeded "NeuraEdge SAIF activity-power pipeline" $? $LASTEXITCODE
    }

    if (!$SkipReports) {
        Write-Host "[reports] regenerating deep debug PDF, line-by-line explainer PDF, and PNG plots"
        & $Python $ReportScript
        Assert-StepSucceeded "NeuraEdge report generation" $? $LASTEXITCODE
    }
} finally {
    Pop-Location
}

Write-Host "[done] private NeuraEdge accelerator flow completed"