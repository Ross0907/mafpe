param(
    [string]$IcarusRoot = "C:\msys64\ucrt64\bin",
    [string]$MsysRoot = "C:\msys64",
    [string]$PythonExe = "",
    [switch]$KeepExecutables
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputRoot = Join-Path $ProjectRoot "outputs\regression"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$SummaryPath = Join-Path $OutputRoot "regression_summary.txt"
@(
    "result=RUNNING",
    "started_utc=$([DateTime]::UtcNow.ToString('o'))"
) | Set-Content -LiteralPath $SummaryPath -Encoding ascii

$Iverilog = Join-Path $IcarusRoot "iverilog.exe"
$Vvp = Join-Path $IcarusRoot "vvp.exe"
if (-not (Test-Path -LiteralPath $Iverilog) -or -not (Test-Path -LiteralPath $Vvp)) {
    throw "Icarus Verilog was not found under $IcarusRoot"
}

function Test-RegressionPython {
    param([string]$Candidate)
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $false
    }
    $Command = Get-Command $Candidate -ErrorAction SilentlyContinue
    if ($null -eq $Command) {
        return $false
    }
    & $Command.Source -c "import numpy, torch" *> $null
    return $LASTEXITCODE -eq 0
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $Candidates = [System.Collections.Generic.List[string]]::new()
    $SearchRoot = Get-Item -LiteralPath $ProjectRoot
    for ($Depth = 0; $Depth -lt 5 -and $null -ne $SearchRoot; $Depth++) {
        $PreferredPython = Join-Path $SearchRoot.FullName `
            "external_research\spikingformer_py310\Scripts\python.exe"
        if (Test-Path -LiteralPath $PreferredPython) {
            $Candidates.Add((Resolve-Path $PreferredPython).Path)
        }
        $SearchRoot = $SearchRoot.Parent
    }
    $Launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $Launcher) {
        foreach ($Version in @("3.14", "3.10")) {
            $Discovered = & $Launcher.Source "-$Version" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($Discovered)) {
                $Candidates.Add([string]($Discovered | Select-Object -Last 1))
            }
        }
    }
    $PathPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $PathPython) {
        $Candidates.Add($PathPython.Source)
    }
    foreach ($Candidate in $Candidates | Select-Object -Unique) {
        if (Test-RegressionPython $Candidate) {
            $PythonExe = (Get-Command $Candidate -ErrorAction Stop).Source
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        throw "No Python interpreter with NumPy and PyTorch was found. Pass -PythonExe explicitly."
    }
} elseif (-not (Test-RegressionPython $PythonExe)) {
    throw "The requested Python interpreter is unavailable or cannot import NumPy and PyTorch: $PythonExe"
}

$PackageFile = Join-Path $ProjectRoot "rtl\neuraedge_command_pkg.sv"
$RtlFiles = @($PackageFile) + @(
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "rtl") -Filter "*.sv" -File |
        Where-Object { $_.FullName -ne $PackageFile } |
        Sort-Object Name |
        ForEach-Object FullName
)
$Testbenches = @(
    "tb_neuraedge_mac2_u8",
    "tb_neuraedge_tensor_tile",
    "tb_neuraedge_buffered_tensor_core",
    "tb_neuraedge_alpha_xnor",
    "tb_neuraedge_vector_alu",
    "tb_neuraedge_integer_primitives",
    "tb_neuraedge_normalization_units",
    "tb_neuraedge_lif_bank",
    "tb_neuraedge_sgd_optimizer8",
    "tb_neuraedge_linear_backward8",
    "tb_neuraedge_activation_lut8",
    "tb_neuraedge_universal_core",
    "tb_neuraedge_descriptor_sequencer",
    "tb_neuraedge_pynq_descriptor",
    "tb_neuraedge_pynq_axi_top"
)

Push-Location $ProjectRoot
try {
    $env:PYTHONPATH = (Join-Path $ProjectRoot "software")
    & $PythonExe (Join-Path $ProjectRoot "software\generate_tensor_tile_fixture.py") `
        --output-dir (Join-Path $ProjectRoot "tb\fixtures") 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputRoot "tensor_fixture_generation.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Tensor fixture generation failed with exit code $LASTEXITCODE"
    }
    & $PythonExe (Join-Path $ProjectRoot "software\generate_operator_fixtures.py") `
        --output-dir (Join-Path $ProjectRoot "tb\fixtures") 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputRoot "operator_fixture_generation.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Operator fixture generation failed with exit code $LASTEXITCODE"
    }

    & (Join-Path $PSScriptRoot "run_verilator_lint.ps1") -MsysRoot $MsysRoot 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputRoot "verilator_driver.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Verilator lint failed with exit code $LASTEXITCODE"
    }

    & $PythonExe -m unittest discover -s tests -v 2>&1 |
        Tee-Object -FilePath (Join-Path $OutputRoot "python_unittest.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Python regression failed with exit code $LASTEXITCODE"
    }

    foreach ($Top in $Testbenches) {
        $TestbenchFile = Join-Path $ProjectRoot "tb\$Top.sv"
        $Executable = Join-Path $OutputRoot "$Top.vvp"
        $CompileLog = Join-Path $OutputRoot "$Top.compile.log"
        $RunLog = Join-Path $OutputRoot "$Top.run.log"

        & $Iverilog -g2012 -I $ProjectRoot -s $Top -o $Executable @RtlFiles $TestbenchFile 2>&1 |
            Tee-Object -FilePath $CompileLog
        if ($LASTEXITCODE -ne 0) {
            throw "$Top compilation failed with exit code $LASTEXITCODE"
        }

        & $Vvp $Executable 2>&1 | Tee-Object -FilePath $RunLog
        if ($LASTEXITCODE -ne 0) {
            throw "$Top simulation failed with exit code $LASTEXITCODE"
        }
        $RunText = Get-Content -LiteralPath $RunLog -Raw
        $PassPattern = "(?m)^$([regex]::Escape($Top)) PASSED\b"
        if ($RunText -notmatch $PassPattern) {
            throw "$Top simulation did not emit its required PASSED marker"
        }
        if ($RunText -match "(?im)^\s*ERROR\b|^.*\bFAILED\b") {
            throw "$Top simulation emitted an ERROR or FAILED marker"
        }
        if (-not $KeepExecutables) {
            Remove-Item -LiteralPath $Executable -Force
        }
    }
} catch {
    @(
        "result=FAIL",
        "message=$($_.Exception.Message)",
        "completed_utc=$([DateTime]::UtcNow.ToString('o'))"
    ) | Set-Content -LiteralPath $SummaryPath -Encoding ascii
    throw
} finally {
    Pop-Location
}

@(
    "result=PASS",
    "python_executable=$PythonExe",
    "iverilog_executable=$Iverilog",
    "verilator_environment=$MsysRoot",
    "rtl_testbench_count=$($Testbenches.Count)",
    "completed_utc=$([DateTime]::UtcNow.ToString('o'))"
) | Set-Content -LiteralPath $SummaryPath -Encoding ascii
Write-Host "NEURAEDGE_REGRESSION_PASSED tests=$($Testbenches.Count) summary=$SummaryPath"
