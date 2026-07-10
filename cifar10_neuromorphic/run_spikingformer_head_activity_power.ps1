param(
    [string]$VivadoBin = "D:\AMDDesignTools\2026.1\Vivado\bin",
    [string]$OutDir = "cifar10_neuromorphic\outputs_spikingformer_head_accelerator",
    [string]$Part = "xc7z020clg400-1",
    [double]$ClockPeriodNs = 10.0,
    [switch]$SkipInferenceSaif,
    [switch]$SkipLearningSaif,
    [switch]$SkipPowerReports
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
$ResolvedOut = Resolve-Path (Join-Path $RepoRoot $OutDir)
$ActivityDir = Join-Path $ResolvedOut "activity_power"
$RoutedDcp = Join-Path $ResolvedOut "vivado\spikingformer_head_routed.dcp"

if (!(Test-Path $Python)) { throw "Missing Python env: $Python" }
if (Test-Path $TorchLib) { $env:PATH = "$TorchLib;$env:PATH" }
foreach ($tool in @($VivadoXvlog, $VivadoXelab, $VivadoXsim, $Vivado)) {
    if (!(Test-Path $tool)) { throw "Missing Vivado tool: $tool" }
}
if (!(Test-Path $RoutedDcp)) { throw "Missing routed checkpoint: $RoutedDcp. Run run_spikingformer_head_accelerator.ps1 first." }
New-Item -ItemType Directory -Force -Path $ActivityDir | Out-Null

function Assert-LastExitCode {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Write-XsimSaifTcl {
    param(
        [string]$Path,
        [string]$SaifPath,
        [string]$Scope
    )
    @(
        ('open_saif {{{0}}}' -f $SaifPath.Replace('\', '/')),
        ('log_saif [get_objects -r {0}/*]' -f $Scope),
        "run all",
        "close_saif",
        "quit"
    ) | Set-Content -LiteralPath $Path -Encoding ASCII
}

function Run-SaifCase {
    param(
        [string]$Name,
        [string]$Top,
        [string]$TbFile
    )
    $saifPath = Join-Path $ActivityDir "spikingformer_head_${Name}.saif"
    $tclName = "xsim_spikingformer_head_${Name}_saif.tcl"
    $tclPath = Join-Path $ProjectDir $tclName
    Write-XsimSaifTcl -Path $tclPath -SaifPath $saifPath -Scope "/$Top/base/dut"
    Push-Location $ProjectDir
    try {
        Write-Host "[saif] compiling $Name"
        & $VivadoXvlog -sv `
            rtl\cifar10_spikingformer_head_accelerator.sv `
            rtl\cifar10_spikingformer_head_pynq_axi_stream_top.sv `
            $TbFile
        Assert-LastExitCode "xvlog $Name"
        & $VivadoXelab $Top -debug typical
        Assert-LastExitCode "xelab $Name"
        Write-Host "[saif] running $Name full XSim SAIF"
        & $VivadoXsim $Top -tclbatch $tclName
        Assert-LastExitCode "xsim $Name"
    } finally {
        Pop-Location
    }
    if (!(Test-Path $saifPath)) { throw "Expected SAIF was not produced: $saifPath" }
}

if (!$SkipInferenceSaif) {
    Run-SaifCase `
        -Name "inference" `
        -Top "tb_cifar10_spikingformer_head_axi_inference_activity" `
        -TbFile "rtl\tb_cifar10_spikingformer_head_axi_activity.sv"
}

if (!$SkipLearningSaif) {
    Run-SaifCase `
        -Name "learning" `
        -Top "tb_cifar10_spikingformer_head_axi_learning_activity" `
        -TbFile "rtl\tb_cifar10_spikingformer_head_axi_activity.sv"
}

if (!$SkipPowerReports) {
    $powerTcl = Join-Path $ActivityDir "spikingformer_head_activity_power.tcl"
    @(
        ('open_checkpoint {{{0}}}' -f $RoutedDcp.Replace('\', '/')),
        ('if {{[llength [get_clocks ap_clk -quiet]] == 0}} {{ create_clock -period {0} -name ap_clk [get_ports ap_clk] }}' -f $ClockPeriodNs),
        ('set inference_saif {{{0}}}' -f (Join-Path $ActivityDir "spikingformer_head_inference.saif").Replace('\', '/')),
        ('set learning_saif {{{0}}}' -f (Join-Path $ActivityDir "spikingformer_head_learning.saif").Replace('\', '/')),
        ('set inference_report {{{0}}}' -f (Join-Path $ActivityDir "spikingformer_head_power_inference_saif.rpt").Replace('\', '/')),
        ('set learning_report {{{0}}}' -f (Join-Path $ActivityDir "spikingformer_head_power_learning_saif.rpt").Replace('\', '/')),
        'if {[file exists $inference_saif]} {',
        '  reset_switching_activity [get_nets -hierarchical *]',
        '  read_saif -strip_path tb_cifar10_spikingformer_head_axi_inference_activity/base/dut $inference_saif',
        '  report_power -file $inference_report -hier all',
        '} else {',
        '  puts "Missing inference SAIF: $inference_saif"',
        '}',
        'if {[file exists $learning_saif]} {',
        '  reset_switching_activity [get_nets -hierarchical *]',
        '  read_saif -strip_path tb_cifar10_spikingformer_head_axi_learning_activity/base/dut $learning_saif',
        '  report_power -file $learning_report -hier all',
        '} else {',
        '  puts "Missing learning SAIF: $learning_saif"',
        '}',
        "exit"
    ) | Set-Content -LiteralPath $powerTcl -Encoding ASCII

    Write-Host "[power] reporting routed power with XSim SAIF activity"
    & $Vivado -mode batch -source $powerTcl -notrace
    Assert-LastExitCode "Vivado SAIF power report"
}

Push-Location $RepoRoot
try {
    & $Python "cifar10_neuromorphic\summarize_spikingformer_head_activity_power.py"
    Assert-LastExitCode "activity power summary"
} finally {
    Pop-Location
}
