# Run from an activated Python environment after setting CHBMIT_BIDS_ROOT.
param([switch]$IncludeAblations, [switch]$IncludeBaseline)
$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUNBUFFERED = "1"

function Invoke-Python {
    param([string[]]$PythonArgs)
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline step failed: python $($PythonArgs -join ' ')"
    }
}

Invoke-Python -PythonArgs @("-u", "run_selftest.py")
Invoke-Python -PythonArgs @("-u", "-m", "dataset.preprocess")
Invoke-Python -PythonArgs @("-u", "run_healthcheck.py")
Invoke-Python -PythonArgs @("-u", "run_lopo.py")
if ($IncludeBaseline) {
    Invoke-Python -PythonArgs @("-u", "-m", "baselines.classical")
}
if ($IncludeAblations) {
    Invoke-Python -PythonArgs @("-u", "run_lopo.py", "--all-ablations")
}
Invoke-Python -PythonArgs @("-u", "run_export.py")
Invoke-Python -PythonArgs @("-u", "run_figures.py")
Write-Host "Complete. See paper_results, paper_tables, and paper_figures."
