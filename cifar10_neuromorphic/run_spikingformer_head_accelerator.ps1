param(
    [int]$Samples = 10000,
    [int]$BatchSize = 64,
    [int]$Workers = 4,
    [int]$TraceSamples = 8,
    [int]$InferSkipAbsThresh = 2048,
    [string]$Checkpoint = "cifar10_neuromorphic\outputs_upstream_spikingformer\spikingformer_4_384_400e\model_best.pth.tar",
    [string]$OutDir = "cifar10_neuromorphic\outputs_spikingformer_head_accelerator",
    [string]$VivadoBin = "D:\AMDDesignTools\2026.1\Vivado\bin",
    [string]$Part = "xc7z020clg400-1",
    [double]$ClockPeriodNs = 10.0,
    [switch]$SkipExport,
    [switch]$SkipVivado,
    [switch]$SkipOnChipLearning,
    [switch]$SkipSynthesis
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
$Python = Join-Path $RepoRoot "external_research\spikingformer_py310\Scripts\python.exe"
$TorchLib = Join-Path $RepoRoot "external_research\spikingformer_py310\Lib\site-packages\torch\lib"
$VivadoXvlog = Join-Path $VivadoBin "xvlog.bat"
$VivadoXelab = Join-Path $VivadoBin "xelab.bat"
$VivadoXsim = Join-Path $VivadoBin "xsim.bat"
$Vivado = Join-Path $VivadoBin "vivado.bat"

if (!(Test-Path $Python)) { throw "Missing Python env: $Python" }
if (Test-Path $TorchLib) { $env:PATH = "$TorchLib;$env:PATH" }

function Assert-LastExitCode {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

Push-Location $RepoRoot
try {
    if (!$SkipExport) {
        Write-Host "[export] trained Spikingformer checkpoint to fixed-point accelerator memories"
        & $Python "cifar10_neuromorphic\export_spikingformer_head_accelerator.py" `
            --checkpoint "$Checkpoint" `
            --samples "$Samples" `
            --batch-size "$BatchSize" `
            --workers "$Workers" `
            --trace-samples "$TraceSamples" `
            --infer-skip-abs-thresh "$InferSkipAbsThresh" `
            --out-dir "$OutDir"
        Assert-LastExitCode "Spikingformer head export"
    }
} finally {
    Pop-Location
}

if (!$SkipVivado) {
    if (!(Test-Path $VivadoXvlog)) { throw "Vivado xvlog not found at $VivadoXvlog" }
    if (!(Test-Path $VivadoXelab)) { throw "Vivado xelab not found at $VivadoXelab" }
    if (!(Test-Path $VivadoXsim)) { throw "Vivado xsim not found at $VivadoXsim" }

    Push-Location $ProjectDir
    try {
        Write-Host "[vivado] compiling trained Spikingformer head accelerator"
        & $VivadoXvlog -sv `
            rtl\cifar10_spikingformer_head_accelerator.sv `
            rtl\tb_cifar10_spikingformer_head_accelerator.sv
        Assert-LastExitCode "Vivado xvlog"
        & $VivadoXelab tb_cifar10_spikingformer_head_accelerator -debug typical
        Assert-LastExitCode "Vivado xelab"
        & $VivadoXsim tb_cifar10_spikingformer_head_accelerator -runall
        Assert-LastExitCode "Vivado xsim"
    } finally {
        Pop-Location
    }

    Push-Location $RepoRoot
    try {
        Write-Host "[validate] checking RTL scores against fixed-point reference"
        & $Python "cifar10_neuromorphic\validate_spikingformer_head_rtl.py" --out-dir "$OutDir"
        Assert-LastExitCode "Spikingformer head RTL validation"
    } finally {
        Pop-Location
    }

    if (!$SkipOnChipLearning) {
        Push-Location $ProjectDir
        try {
            Write-Host "[vivado] verifying on-chip learning update datapath"
            & $VivadoXvlog -sv `
                rtl\cifar10_spikingformer_head_accelerator.sv `
                rtl\tb_spikingformer_head_learning_unit.sv
            Assert-LastExitCode "Vivado xvlog on-chip learning unit"
            & $VivadoXelab tb_spikingformer_head_learning_unit -debug typical
            Assert-LastExitCode "Vivado xelab on-chip learning unit"
            & $VivadoXsim tb_spikingformer_head_learning_unit -runall
            Assert-LastExitCode "Vivado xsim on-chip learning unit"

            Write-Host "[vivado] running full CIFAR-10 on-chip learning simulation"
            & $VivadoXvlog -sv `
                rtl\cifar10_spikingformer_head_accelerator.sv `
                rtl\tb_cifar10_spikingformer_head_onchip_learning.sv
            Assert-LastExitCode "Vivado xvlog on-chip learning full"
            & $VivadoXelab tb_cifar10_spikingformer_head_onchip_learning -debug typical
            Assert-LastExitCode "Vivado xelab on-chip learning full"
            & $VivadoXsim tb_cifar10_spikingformer_head_onchip_learning -runall
            Assert-LastExitCode "Vivado xsim on-chip learning full"
        } finally {
            Pop-Location
        }

        Push-Location $RepoRoot
        try {
            Write-Host "[validate] checking RTL on-chip learning against golden online update model"
            & $Python "cifar10_neuromorphic\validate_spikingformer_head_onchip_learning.py" --out-dir "$OutDir"
            Assert-LastExitCode "Spikingformer head on-chip learning validation"
        } finally {
            Pop-Location
        }
    }
}

if (!$SkipSynthesis) {
    if (!(Test-Path $Vivado)) { throw "Vivado not found at $Vivado" }
    $SynthOut = Join-Path (Resolve-Path $OutDir) "vivado"
    New-Item -ItemType Directory -Force -Path $SynthOut | Out-Null
    $env:NEURAEDGE_PART = $Part
    $env:NEURAEDGE_CLOCK_NS = "$ClockPeriodNs"
    $env:NEURAEDGE_OUT_DIR = "$SynthOut"
    Push-Location $ProjectDir
    try {
        Write-Host "[synth] routing trained Spikingformer head accelerator for $Part at ${ClockPeriodNs}ns"
        & $Vivado -mode batch -source "spikingformer_head_synth.tcl"
        Assert-LastExitCode "Spikingformer head synthesis/implementation"
    } finally {
        Pop-Location
        Remove-Item Env:\NEURAEDGE_PART -ErrorAction SilentlyContinue
        Remove-Item Env:\NEURAEDGE_CLOCK_NS -ErrorAction SilentlyContinue
        Remove-Item Env:\NEURAEDGE_OUT_DIR -ErrorAction SilentlyContinue
    }
}

Write-Host "[done] trained Spikingformer head accelerator flow completed"
Write-Host "       outputs: $OutDir"
Write-Host "       rtl csv: cifar10_neuromorphic\cifar10_spikingformer_head_results.csv"
