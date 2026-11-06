param([ValidateSet('quick','full')][string]$Stage, [ValidateSet('traci','libsumo')][string]$Backend = 'traci')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot 'T_Junction_RL')
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$researchPython = 'R:/rl-env/Scripts/python.exe'
$researchRoot = 'outputs/v3_research'
$proof = "$researchRoot/verification"
$quick = "$researchRoot/quick"
$training = "$researchRoot/$Stage"
$groups = foreach ($control in 'cycle','direct') {
    $top = Import-Csv "$quick/validation_leaderboard.csv" |
        Where-Object { $_.control_mode -eq $control } |
        Sort-Object @{Expression={ [double]$_.mean_validation_delay_s }},algorithm |
        Select-Object -First 1
    if (!$top) { throw "No validation result for $control" }
    $control + '_' + $top.algorithm
}
$groups | Set-Content -Encoding utf8NoBOM "$proof/selected_from_quick.txt"
if ($Stage -eq 'full') {
    & $researchPython -m scripts.research_matrix --mode full --backend $Backend --groups @groups --jobs 6 --output $training --execute 2>&1 | Tee-Object "$proof/full.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $researchPython -m scripts.summarize_v3 --root $training 2>&1 | Tee-Object "$proof/full_ranking.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $researchPython -m scripts.research_evaluation --training $training --output "$training/validation_eval" --split validation --backend $Backend --seeds 3 --horizon 900 --jobs 8 2>&1 | Tee-Object "$proof/full_validation.log"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$resume = @()
if (Test-Path "$training/final_eval/evaluation_plan.json") { $resume = @('--resume') }
& $researchPython -m scripts.research_evaluation --training $training --output "$training/final_eval" --split final --backend $Backend --seeds 10 --horizon 900 --jobs 8 --groups @groups @resume 2>&1 | Tee-Object "$proof/final.log"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$plots = "$training/figures"
if ($Stage -eq 'quick') { $plots = "$training/final_figures" }
& $researchPython -m scripts.research_report --training $training --evaluation "$training/final_eval" --output $plots
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $researchPython -m scripts.audit_research --output "$proof/evidence_audit.json"
exit $LASTEXITCODE
