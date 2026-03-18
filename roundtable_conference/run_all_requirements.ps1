$ErrorActionPreference = "Stop"

$project = "E:\zz_work_space\likert"
$rtRoot = Join-Path $project "roundtable_conference"
$reqDir = Join-Path $rtRoot "requirements"
$outRoot = Join-Path $rtRoot "output"
$models = Join-Path $rtRoot "models.json"
$script = Join-Path $rtRoot "roundtable_req_reconcile.py"

if (-not (Test-Path $reqDir)) {
    throw "Requirements directory not found: $reqDir"
}

if (-not (Test-Path $models)) {
    throw "models.json not found: $models"
}

if (-not (Test-Path $script)) {
    throw "roundtable_req_reconcile.py not found: $script"
}

New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$csvFiles = Get-ChildItem -Path $reqDir -Filter *.csv -File | Sort-Object Name
if (-not $csvFiles) {
    throw "No CSV files found in $reqDir"
}

foreach ($file in $csvFiles) {
    $csv = $file.FullName
    $name = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    $safe = $name.Trim().TrimEnd(".")
    if ([string]::IsNullOrWhiteSpace($safe)) {
        $safe = "requirements_file"
    }

    $caseOut = Join-Path $outRoot $safe
    $caseLogs = Join-Path $caseOut "logs"
    $excelOut = Join-Path $caseOut ($safe + "_roundtable_report.xlsx")

    New-Item -ItemType Directory -Force -Path $caseOut, $caseLogs | Out-Null

    Write-Host "Running: $($file.Name)" -ForegroundColor Cyan

    python $script `
        --requirements $csv `
        --models $models `
        --out $excelOut `
        --out_dir $caseLogs `
        --topk 10 `
        --rounds 2 `
        --theta_ratio 0.6 `
        --eps_score 0.25 `
        --tau_jacc 0.9
}

Write-Host "All requirement CSV files processed." -ForegroundColor Green
