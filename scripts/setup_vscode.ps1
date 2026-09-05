# One-shot VS Code setup for DynaGAT.
# Writes .vscode/launch.json and .vscode/settings.json for this workspace.
# .vscode is git-ignored, so run this once after cloning:
#     powershell -ExecutionPolicy Bypass -File .\scripts\setup_vscode.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force -Path ".vscode" | Out-Null

$launch = @'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "0 - self-test (no dataset needed)",
      "type": "debugpy", "request": "launch", "program": "${workspaceFolder}/run_selftest.py",
      "console": "integratedTerminal", "cwd": "${workspaceFolder}", "justMyCode": false,
      "env": {"PYTHONUNBUFFERED": "1"}
    },
    {
      "name": "1 - health check",
      "type": "debugpy", "request": "launch", "program": "${workspaceFolder}/run_healthcheck.py",
      "console": "integratedTerminal", "cwd": "${workspaceFolder}",
      "env": {"PYTHONUNBUFFERED": "1", "CHBMIT_BIDS_ROOT": "${env:CHBMIT_BIDS_ROOT}"}
    },
    {
      "name": "2 - preprocess cache",
      "type": "debugpy", "request": "launch", "module": "dataset.preprocess",
      "console": "integratedTerminal", "cwd": "${workspaceFolder}",
      "env": {"PYTHONUNBUFFERED": "1", "CHBMIT_BIDS_ROOT": "${env:CHBMIT_BIDS_ROOT}"}
    },
    {
      "name": "2b - preprocess 2 subjects (smoke)",
      "type": "debugpy", "request": "launch", "module": "dataset.preprocess",
      "args": ["--max-subjects", "2"],
      "console": "integratedTerminal", "cwd": "${workspaceFolder}",
      "env": {"PYTHONUNBUFFERED": "1", "CHBMIT_BIDS_ROOT": "${env:CHBMIT_BIDS_ROOT}"}
    },
    {
      "name": "3 - LOPO (full)",
      "type": "debugpy", "request": "launch", "program": "${workspaceFolder}/run_lopo.py",
      "console": "integratedTerminal", "cwd": "${workspaceFolder}",
      "env": {"PYTHONUNBUFFERED": "1"}
    },
    {
      "name": "3b - LOPO fold 1 only",
      "type": "debugpy", "request": "launch", "program": "${workspaceFolder}/run_lopo.py",
      "args": ["--folds", "1"],
      "console": "integratedTerminal", "cwd": "${workspaceFolder}",
      "env": {"PYTHONUNBUFFERED": "1"}
    },
    {
      "name": "4 - model ablations",
      "type": "debugpy", "request": "launch", "program": "${workspaceFolder}/run_lopo.py",
      "args": ["--all-ablations"],
      "console": "integratedTerminal", "cwd": "${workspaceFolder}",
      "env": {"PYTHONUNBUFFERED": "1"}
    },
    {
      "name": "5 - export paper artifacts",
      "type": "debugpy", "request": "launch", "program": "${workspaceFolder}/run_export.py",
      "console": "integratedTerminal", "cwd": "${workspaceFolder}",
      "env": {"PYTHONUNBUFFERED": "1"}
    }
  ]
}
'@

$settings = @'
{
  "python.terminal.activateEnvironment": true,
  "python.analysis.extraPaths": ["${workspaceFolder}"],
  "terminal.integrated.env.windows": {
    "PYTHONPATH": "${workspaceFolder}"
  }
}
'@

Set-Content -Path ".vscode\launch.json"   -Value $launch   -Encoding UTF8
Set-Content -Path ".vscode\settings.json" -Value $settings -Encoding UTF8
Write-Host "Wrote .vscode\launch.json and .vscode\settings.json" -ForegroundColor Green
Write-Host "In VS Code: Ctrl+Shift+P -> Python: Select Interpreter -> your DynaGAT environment" -ForegroundColor Yellow
