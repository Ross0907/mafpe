param(
    [ValidateSet("start", "status", "pause", "resume")]
    [string]$Action = "status",
    [string]$BoardHost = "xilinx@192.168.2.99",
    [string]$RemoteRoot = "/home/xilinx/neuraedge_universal",
    [string]$Python = "python3",
    [int]$Epochs = 400,
    [int]$BatchSize = 1,
    [double]$LearningRate = 0.25,
    [double]$MinimumLearningRate = 0.01,
    [int]$CheckpointInterval = 1
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Output = "$RemoteRoot/outputs/pynq_training"
$PidFile = "$Output/training.pid"
$PauseFile = "$Output/PAUSE"
$StatusFile = "$Output/training_status.json"
$OutLog = "$Output/train.out.log"
$ErrLog = "$Output/train.err.log"

function Invoke-Remote([string]$Command) {
    & ssh $BoardHost "sh -lc '$Command'"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote PYNQ command failed with exit code $LASTEXITCODE"
    }
}

function Start-Training([switch]$Resume) {
    $ResumeArgument = if ($Resume) { "--resume" } else { "" }
    $Command = "mkdir -p $Output; " +
        "if [ -f $PidFile ] && kill -0 `$(cat $PidFile) 2>/dev/null; then " +
        "echo already_running pid=`$(cat $PidFile); exit 2; fi; " +
        "rm -f $PauseFile; " +
        "cd $RemoteRoot; " +
        "nohup $Python software/train_spikingformer_pynq.py " +
        "--bitfile outputs/pynq_full/neuraedge_pynq_full.bit " +
        "--manifest outputs/spikingformer_hardware_parameters/spikingformer_neuraedge_weights.json " +
        "--data-manifest outputs/cifar10_shards/cifar10_shards_manifest.json " +
        "--output-dir outputs/pynq_training --epochs $Epochs --batch-size $BatchSize " +
        "--learning-rate $LearningRate --minimum-learning-rate $MinimumLearningRate " +
        "--checkpoint-interval $CheckpointInterval $ResumeArgument " +
        ">$OutLog 2>$ErrLog </dev/null & echo `$! >$PidFile; " +
        "echo started pid=`$(cat $PidFile)"
    Invoke-Remote $Command
}

switch ($Action) {
    "start" { Start-Training }
    "resume" { Start-Training -Resume }
    "pause" {
        Invoke-Remote "mkdir -p $Output; touch $PauseFile; echo pause_requested=$PauseFile"
    }
    "status" {
        $Command = "if [ -f $PidFile ]; then pid=`$(cat $PidFile); else pid=none; fi; " +
            "if [ `$pid != none ] && kill -0 `$pid 2>/dev/null; then alive=true; else alive=false; fi; " +
            "echo pid=`$pid alive=`$alive; " +
            "if [ -f $StatusFile ]; then cat $StatusFile; else echo no_status_file; fi; " +
            "echo out_log=$OutLog; echo err_log=$ErrLog"
        Invoke-Remote $Command
    }
}
