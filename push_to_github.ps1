# Publish the DynaGAT source tree to GitHub.
#
# Commits code only. Caches, results, figures and checkpoints are excluded by
# .gitignore and blocked again by a guard below. Nothing is committed or pushed
# until you type "yes".
#
#   .\push_to_github.ps1
#   .\push_to_github.ps1 -RepoUrl "https://github.com/Mj-heidari/DynaGAT.git"
#   .\push_to_github.ps1 -Force

param(
    [string]$RepoUrl = "https://github.com/Mj-heidari/DynaGAT-Seizure-Detection.git",
    [string]$Branch  = "main",
    [string]$Message = "DynaGAT: causal dual-view graph attention for patient-independent EEG seizure detection",
    [switch]$Force,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not on PATH. Install Git for Windows first."
}

$name  = git config user.name
$email = git config user.email
if ([string]::IsNullOrWhiteSpace($name) -or [string]::IsNullOrWhiteSpace($email)) {
    Write-Host "git identity is not configured. Set it once:" -ForegroundColor Red
    Write-Host '    git config --global user.name  "Your Name"'
    Write-Host '    git config --global user.email "you@example.com"'
    exit 1
}

Write-Host "Author : $name <$email>" -ForegroundColor Cyan
Write-Host "Remote : $RepoUrl"
Write-Host "Branch : $Branch"
Write-Host ""

if (-not (Test-Path ".git")) {
    git init -q
    git checkout -q -b $Branch
    Write-Host "Initialised a new repository here." -ForegroundColor Yellow
} else {
    git checkout -q -B $Branch
}

git add -A

$staged = @(git diff --cached --name-only)
if ($staged.Count -eq 0) {
    Write-Host "Nothing staged - the working tree already matches HEAD." -ForegroundColor Yellow
} else {
    # Refuse to publish data, checkpoints or generated output.
    $blocked = @($staged | Where-Object {
        $_ -match '^(results|data_cache|paper_figures|paper_tables|paper_results)/' -or
        $_ -match '\.(pt|npz|edf|pth|ckpt)$'
    })
    if ($blocked.Count -gt 0) {
        Write-Host ""
        Write-Host "Refusing to continue. These would be published:" -ForegroundColor Red
        foreach ($f in $blocked) { Write-Host "  $f" -ForegroundColor Red }
        Write-Host "Run 'git reset' or fix .gitignore, then try again." -ForegroundColor Red
        exit 1
    }

    Write-Host "$($staged.Count) file(s) staged:" -ForegroundColor Cyan
    foreach ($f in $staged) { Write-Host "  $f" }

    foreach ($f in $staged) {
        if (Test-Path $f) {
            $mb = (Get-Item $f).Length / 1MB
            if ($mb -gt 2) {
                Write-Host ("  large: {0}  {1:N1} MB" -f $f, $mb) -ForegroundColor Yellow
            }
        }
    }

    if (-not $Yes) {
        Write-Host ""
        Write-Host "Commit message: $Message"
        if ($Force) {
            Write-Host "FORCE PUSH: this overwrites '$Branch' on the remote." -ForegroundColor Red
        }
        $answer = Read-Host "Proceed? (type 'yes' to continue)"
        if ($answer -ne "yes") {
            Write-Host "Aborted."
            exit 1
        }
    }

    git commit -q -m $Message
    Write-Host "Committed." -ForegroundColor Green
}

$remotes = @(git remote)
if ($remotes -contains "origin") {
    git remote set-url origin $RepoUrl
} else {
    git remote add origin $RepoUrl
}

Write-Host "Pushing to $RepoUrl ($Branch) ..." -ForegroundColor Cyan
if ($Force) {
    git push -u origin $Branch --force
} else {
    git push -u origin $Branch
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Push failed. Common causes:" -ForegroundColor Yellow
    Write-Host "  * the remote has commits you do not have locally"
    Write-Host "      git pull --rebase origin $Branch    (then run this again)"
    Write-Host "      or re-run with -Force to replace the remote branch"
    Write-Host "  * authentication: use a personal access token as the password"
    exit 1
}

Write-Host ""
Write-Host "Done. $RepoUrl" -ForegroundColor Green
