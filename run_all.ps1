# DynaGAT - full pipeline for the GNN_pytorch_gpu conda environment.
# From VS Code:  right-click -> Run in Integrated Terminal, or
#                powershell -ExecutionPolicy Bypass -File .\run_all.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$env:CHBMIT_BIDS_ROOT = "D:\EEG_Dataset\CHB_MIT\BIDS_CHB-MIT\BIDS_CHB-MIT"
$env:PYTHONUNBUFFERED  = "1"

Write-Host "`n=== 0/5  self-test (no dataset needed) ===" -ForegroundColor Cyan
python -u run_selftest.py
if ($LASTEXITCODE -ne 0) { throw "self-test failed" }

Write-Host "`n=== 1/5  health check ===" -ForegroundColor Cyan
python -u run_healthcheck.py

Write-Host "`n=== 2/5  preprocessing (one-off, ~1-2 h) ===" -ForegroundColor Cyan
python -u -m dataset.preprocess
if ($LASTEXITCODE -ne 0) { throw "preprocessing failed" }

Write-Host "`n=== 3/5  LOPO training, main model ===" -ForegroundColor Cyan
python -u run_lopo.py
if ($LASTEXITCODE -ne 0) { throw "LOPO run failed" }

Write-Host "`n=== 4/5  baseline + ablations (optional, comment out to skip) ===" -ForegroundColor Cyan
python -u -m baselines.classical
python -u run_lopo.py --ablation no_graph
python -u run_lopo.py --ablation no_causal
python -u run_lopo.py --ablation no_adaptive

Write-Host "`n=== 5/5  paper export ===" -ForegroundColor Cyan
python -u run_export.py

Write-Host "`nDone. Tables in paper_tables\, figures in paper_figures\." -ForegroundColor Green
