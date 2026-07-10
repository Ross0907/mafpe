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
$PauseFile = Join-Path $OutputDir "PAUSE"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
"pause requested at $(Get-Date -Format o)" | Set-Content -LiteralPath $PauseFile -Encoding UTF8
Write-Host "Pause requested. The upstream trainer will stop after the current epoch."
Write-Host "  pause_file: $PauseFile"
