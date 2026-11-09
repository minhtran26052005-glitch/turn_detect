$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -u .\run_turn_detector.py --input .\test.jsonl --train .\train.jsonl --batch-size 1 --output-dir .\outputs\test_cpu_single
if ($LASTEXITCODE -ne 0) { throw "Model evaluation failed with exit code $LASTEXITCODE" }
