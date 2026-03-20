# ============================================================
# agent-git.ps1
# Agent-aware git automation for Windsurf IDE
# Usage: .\agent-git.ps1 -Action "StartFeature" -FeatureName "my-feature"
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("StartFeature", "StageChanges", "CompleteFeature", "CancelFeature", "Status")]
    [string]$Action,

    [Parameter(Mandatory=$false)]
    [string]$FeatureName = "",

    [Parameter(Mandatory=$false)]
    [string]$CommitMessage = "",

    [Parameter(Mandatory=$false)]
    [switch]$Force,

    [Parameter(Mandatory=$false)]
    [switch]$SkipTests
)

# ── Paths & Config ───────────────────────────────────────
$ProjectRoot   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$FeatureMeta   = "$ProjectRoot\open-viking\history\active_feature.json"
$FeaturePrefix = "feature"

# ── Colours ─────────────────────────────────────────────────
function Write-Ok  ($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Err ($msg) { Write-Host "[ERR]  $msg" -ForegroundColor Red }
function Write-Inf ($msg) { Write-Host "[AGENT] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

# ── Helper Functions ───────────────────────────────────────
function Get-ActiveFeature {
    if (Test-Path $FeatureMeta) {
        $meta = Get-Content $FeatureMeta | ConvertFrom-Json
        return $meta
    }
    return $null
}

function Set-ActiveFeature($feature) {
    $feature | ConvertTo-Json -Depth 3 | Set-Content $FeatureMeta
}

function Clear-ActiveFeature {
    if (Test-Path $FeatureMeta) {
        Remove-Item $FeatureMeta
    }
}

function Get-CurrentBranch {
    return git branch --show-current
}

function Is-On-Feature-Branch {
    $branch = Get-CurrentBranch
    return $branch -like "$FeaturePrefix/*"
}

# ============================================================
# ACTIONS
# ============================================================

switch ($Action) {
    "StartFeature" {
        Write-Inf "Starting new feature: $FeatureName"
        
        if (-not $FeatureName) {
            Write-Err "FeatureName required for StartFeature"
            exit 1
        }

        # Check if already on a feature branch
        if (Is-On-Feature-Branch) {
            $current = Get-CurrentBranch
            Write-Warn "Already on feature branch: $current"
            Write-Warn "Cancel current feature first or use -Force"
            if (-not $Force) { exit 1 }
        }

        # Create branch name
        $branchName = "$FeaturePrefix/$FeatureName"
        
        # Sync with main first
        Write-Inf "Syncing with main..."
        git checkout main 2>&1 | Out-Null
        git pull origin main 2>&1 | Out-Null
        
        # Create feature branch
        git checkout -b $branchName 2>&1 | Out-Null
        Write-Ok "Created branch: $branchName"

        # Store feature metadata
        $featureMeta = @{
            name = $FeatureName
            branch = $branchName
            startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            files = @()
            status = "active"
        }
        Set-ActiveFeature $featureMeta
        
        Write-Ok "Feature started: $FeatureName"
    }

    "StageChanges" {
        Write-Inf "Staging changes..."
        
        $active = Get-ActiveFeature
        if (-not $active) {
            Write-Warn "No active feature found. Starting auto-feature..."
            $timestamp = Get-Date -Format "yyyyMMdd-HHmm"
            .\agent-git.ps1 -Action "StartFeature" -FeatureName "auto-$timestamp"
            $active = Get-ActiveFeature
        }

        # Stage all changes
        git add . 2>&1 | Out-Null
        
        # Get changed files
        $changed = git diff --cached --name-only
        if ($changed) {
            $active.files = $changed -split "`n"
            Set-ActiveFeature $active
            Write-Ok "Staged $($active.files.Count) files"
        } else {
            Write-Warn "No changes to stage"
        }
    }

    "CompleteFeature" {
        Write-Inf "Completing feature..."
        
        $active = Get-ActiveFeature
        if (-not $active) {
            Write-Err "No active feature to complete"
            exit 1
        }

        if (-not (Is-On-Feature-Branch)) {
            Write-Warn "Not on feature branch. Switching to $($active.branch)..."
            git checkout $active.branch 2>&1 | Out-Null
        }

        # Stage any final changes
        git add . 2>&1 | Out-Null
        
        # Check if there are changes to commit
        $hasChanges = git status --porcelain
        if (-not $hasChanges) {
            Write-Warn "No changes to commit"
            Clear-ActiveFeature
            exit 0
        }

        # Run tests unless skipped
        if (-not $SkipTests) {
            Write-Inf "Running tests..."
            $pytest = Get-Command pytest -ErrorAction SilentlyContinue
            if ($pytest) {
                pytest --tb=short -q 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    Write-Err "Tests failed. Use -SkipTests to bypass."
                    exit 1
                }
            }
        }

        # Generate commit message if not provided
        if (-not $CommitMessage) {
            $fileCount = ($active.files | Measure-Object).Count
            $CommitMessage = "feat($($active.name)): update $fileCount file(s)"
        }

        # Commit
        git commit -m $CommitMessage 2>&1 | Out-Null
        $commitHash = git rev-parse --short HEAD
        Write-Ok "Committed: $commitHash"

        # Push
        git push -u origin $active.branch 2>&1 | Out-Null
        Write-Ok "Pushed branch: $($active.branch)"

        # Generate PR URL
        $remoteUrl = git remote get-url origin
        $repoUrl = $remoteUrl -replace "\.git$", ""
        $prUrl = "$repoUrl/compare/$($active.branch)"

        # Update feature metadata
        $active.status = "completed"
        $active.endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $active.commitHash = $commitHash
        Set-ActiveFeature $active

        Write-Ok "Feature completed: $($active.name)"
        Write-Host "PR URL: $prUrl" -ForegroundColor Magenta

        # Clean up
        Clear-ActiveFeature
    }

    "CancelFeature" {
        Write-Inf "Cancelling current feature..."
        
        $active = Get-ActiveFeature
        if ($active) {
            Write-Warn "Cancelling feature: $($active.name)"
            Clear-ActiveFeature
        }

        # Switch back to main
        git checkout main 2>&1 | Out-Null
        Write-Ok "Switched to main branch"
    }

    "Status" {
        Write-Inf "Agent Git Status"
        Write-Host "Current branch: $(Get-CurrentBranch)"
        
        $active = Get-ActiveFeature
        if ($active) {
            Write-Host "Active feature: $($active.name)" -ForegroundColor Green
            Write-Host "  Branch: $($active.branch)"
            Write-Host "  Started: $($active.startTime)"
            Write-Host "  Files: $($active.files.Count)"
            Write-Host "  Status: $($active.status)"
        } else {
            Write-Host "No active feature" -ForegroundColor Yellow
        }

        # Git status
        $gitStatus = git status --porcelain
        if ($gitStatus) {
            Write-Host "Uncommitted changes:" -ForegroundColor Cyan
            $gitStatus | ForEach-Object { Write-Host "  $_" }
        }
    }
}

Write-Host ""
