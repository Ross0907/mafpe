param(
    [int]$Samples = 10000,
    [int]$BatchSize = 64,
    [int]$Workers = 4,
    [int]$TraceSamples = 8,
    [int]$InferSkipAbsThresh = 2048,
    [string]$VivadoBin = "D:\AMDDesignTools\2026.1\Vivado\bin",
    [string]$Part = "xc7z020clg400-1",
    [double]$ClockPeriodNs = 10.0,
    [switch]$SkipExport,
    [switch]$SkipVivado,
    [switch]$SkipOnChipLearning,
    [switch]$SkipSynthesis,
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

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $RepoRoot "external_research\spikingformer_py310\Scripts\python.exe"
$Pipeline = Join-Path $RepoRoot "cifar10_neuromorphic\run_spikingformer_head_accelerator.ps1"
$ReportScript = Join-Path $RepoRoot "cifar10_neuromorphic\generate_spikingformer_debug_reports.py"

if (!(Test-Path $Pipeline)) { throw "Missing accelerator pipeline: $Pipeline" }
if (!(Test-Path $Python)) { throw "Missing Python environment: $Python" }

Push-Location $RepoRoot
try {
    Write-Host "[private] local-only run; no remote publish or upload is performed"
    & $Pipeline `
        -Samples $Samples `
        -BatchSize $BatchSize `
        -Workers $Workers `
        -TraceSamples $TraceSamples `
        -InferSkipAbsThresh $InferSkipAbsThresh `
        -VivadoBin $VivadoBin `
        -Part $Part `
        -ClockPeriodNs $ClockPeriodNs `
        -SkipExport:$SkipExport `
        -SkipVivado:$SkipVivado `
        -SkipOnChipLearning:$SkipOnChipLearning `
        -SkipSynthesis:$SkipSynthesis
    $PipelineSucceeded = $?
    $PipelineExitCode = $LASTEXITCODE
    Assert-StepSucceeded "NeuraEdge CIFAR-10 accelerator pipeline" $PipelineSucceeded $PipelineExitCode

    if (!$SkipReports) {
        Write-Host "[reports] regenerating deep debug PDF, line-by-line explainer PDF, and PNG plots"
        & $Python $ReportScript
        $ReportSucceeded = $?
        $ReportExitCode = $LASTEXITCODE
        Assert-StepSucceeded "NeuraEdge report generation" $ReportSucceeded $ReportExitCode
    }
} finally {
    Pop-Location
}

Write-Host "[done] private full NeuraEdge accelerator flow completed"
