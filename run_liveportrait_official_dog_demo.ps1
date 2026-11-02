# LivePortrait Animal v1.1 official Shiba sample on Windows PowerShell.
# Run from activated 'liveportrait-onnx' conda environment.
param(
    [int]$TargetFrame = -1,
    [int]$Threads = 4
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
$prep = Join-Path $repo 'prepare_liveportrait_official_dog_demo.py'
$chain = Join-Path $repo 'scripts/run_animal_onnx_frame.py'
if (!(Test-Path $prep)) { throw "Missing preparation script: $prep" }
if (!(Test-Path $chain)) { throw "Missing V0.6 chain script: $chain" }
$base = 'build/onnx_animal_v06/official_dog_demo'
& python $prep --repo $repo --target-frame $TargetFrame --overwrite
if ($LASTEXITCODE -ne 0) { throw 'Demo input preparation failed' }
# Neutral/anchor reconstruction: first == target, no relative driving motion.
& python $chain `
    --source "$base/dog_source.png" `
    --driving-first "$base/driver_first.png" `
    --driving-target "$base/driver_first.png" `
    --threads $Threads `
    --output "$base/baseline.png" `
    --report-json "$base/baseline.json" `
    --overwrite
if ($LASTEXITCODE -ne 0) { throw 'V0.6 neutral baseline inference failed' }
# Motion case: first != target.
& python $chain `
    --source "$base/dog_source.png" `
    --driving-first "$base/driver_first.png" `
    --driving-target "$base/driver_target.png" `
    --threads $Threads `
    --output "$base/result.png" `
    --report-json "$base/result.json" `
    --overwrite
if ($LASTEXITCODE -ne 0) { throw 'V0.6 ONNX chain inference failed' }
Write-Host "[OK] Open driving candidates: $base/contact_sheet.jpg"
Write-Host "[OK] Compare baseline.png (neutral) with result.png (motion)"
Write-Host "[OK] JSON report: $base/result.json"
