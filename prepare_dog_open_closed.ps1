$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\ES\Documents\GitHub\LivePortrait'
Set-Location $repo
python .\prepare_dog_open_closed.py
if ($LASTEXITCODE -ne 0) { throw 'Frame extraction failed' }
$dir = 'build\onnx_animal_v062\yawn_pair'
Write-Host "Check source frames in: $dir\contact_sheet.png"
mspaint.exe "$dir\contact_sheet.png"
