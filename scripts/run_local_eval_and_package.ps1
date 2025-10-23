# Run evaluation wrapper and package results for sharing
# Usage: from repo root in PowerShell: .\scripts\run_local_eval_and_package.ps1

$ErrorActionPreference = 'Stop'
$resultsDir = Join-Path -Path (Get-Location) -ChildPath 'results\per_class_summary'
if (-Not (Test-Path $resultsDir)) { New-Item -ItemType Directory -Path $resultsDir | Out-Null }

Write-Output "Running evaluation wrapper (this will call scripts/evaluate.py for each checkpoint folder)..."
python .\analysis\evaluate_checkpoints_per_class.py --checkpoints checkpoints --out $resultsDir

Write-Output "Packaging results into results/per_class_summary.zip (includes all JSONs)"
$zipPath = Join-Path -Path (Get-Location) -ChildPath 'results\per_class_summary.zip'
if (Test-Path $zipPath) { Remove-Item $zipPath }
Compress-Archive -Path (Join-Path $resultsDir '*') -DestinationPath $zipPath
Write-Output "Packed: $zipPath"

Write-Output "Done. If you want me to process these results, please either commit them to the repo or upload '$zipPath' here."