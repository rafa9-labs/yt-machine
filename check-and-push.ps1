# ============================================================
# check-and-push.ps1
# Automates: branch verification, test validation, commit & push
# Usage: .\check-and-push.ps1 -BranchName "feature/my-feature" -CommitMessage "My commit"
# ============================================================

param(
    [Parameter(Mandatory=$false)]
    [string]$BranchName = "",

    [Parameter(Mandatory=$false)]
    [string]$CommitMessage = "",

    [Parameter(Mandatory=$false)]
    [switch]$SkipTests
)

# ── Colours ──────────────────────────────────────────────────
function Write-Ok  ($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Err ($msg) { Write-Host "[ERR]  $msg" -ForegroundColor Red }
function Write-Inf ($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "=======================================" -ForegroundColor Magenta
Write-Host "   YT-MACHINE  |  CHECK-AND-PUSH" -ForegroundColor Magenta
Write-Host "=======================================" -ForegroundColor Magenta
Write-Host ""

# ============================================================
# PHASE 1 — Remote Integrity & Branching
# ============================================================
Write-Inf "PHASE 1: Remote Integrity & Branching"
Write-Host "---------------------------------------"

# 1a. Confirm git repo exists
if (-not (Test-Path ".git")) {
    Write-Err "Not a git repository. Run 'git init' first."
    exit 1
}
Write-Ok "Git repository found."

# 1b. Verify remote URL
$remoteUrl = git remote get-url origin 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "No remote 'origin' configured. Add one with: git remote add origin <url>"
    exit 1
}
Write-Ok "Remote: $remoteUrl"

# 1c. Fetch remote state
Write-Inf "Fetching remote..."
git fetch origin 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Err "git fetch failed. Check your network / GitHub credentials."
    exit 1
}
Write-Ok "Fetch complete."

# 1d. Resolve branch name
if ($BranchName -eq "") {
    $timestamp  = Get-Date -Format "yyyyMMdd-HHmm"
    $BranchName = "feature/update-$timestamp"
    Write-Warn "No branch name supplied. Using: $BranchName"
}

# 1e. Create / switch to feature branch
$existingBranch = git branch --list $BranchName
if ($existingBranch -ne "") {
    Write-Warn "Branch '$BranchName' already exists. Switching to it."
    git checkout $BranchName 2>&1 | Out-Null
} else {
    git checkout -b $BranchName 2>&1 | Out-Null
    Write-Ok "Created and switched to branch: $BranchName"
}

# 1f. Pull latest main to avoid conflicts
Write-Inf "Merging latest origin/main into branch..."
git merge origin/main --no-edit 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Err "Merge conflict detected. Resolve conflicts manually then re-run."
    exit 1
}
Write-Ok "Branch is up to date with origin/main."

# Strict Phase 1 output
Write-Host ""
Write-Host "--- PHASE 1 RESULT ---" -ForegroundColor Magenta
Write-Host "Current branch : $(git branch --show-current)"
Write-Host "Remote         : $remoteUrl"
Write-Host ""

# ============================================================
# PHASE 2 — Validation & Staging
# ============================================================
Write-Inf "PHASE 2: Validation & Staging"
Write-Host "---------------------------------------"

# 2a. Detect changed files
$changedFiles = git status --porcelain
if ($changedFiles -eq "") {
    Write-Warn "No changes detected. Nothing to stage or commit."
    exit 0
}

Write-Inf "Changed files detected:"
$changedFiles | ForEach-Object { Write-Host "  $_" }

# 2b. Run tests (unless skipped)
if (-not $SkipTests) {
    Write-Inf "Running test suite..."

    $pytestAvailable = Get-Command pytest -ErrorAction SilentlyContinue
    if ($pytestAvailable) {
        pytest --tb=short -q 2>&1
        $testResult = $LASTEXITCODE
    } else {
        Write-Warn "pytest not found. Trying 'python -m pytest'..."
        python -m pytest --tb=short -q 2>&1
        $testResult = $LASTEXITCODE
    }

    if ($testResult -ne 0) {
        Write-Err "Tests FAILED. Fix failures before committing."
        Write-Warn "To skip tests and force push, re-run with: -SkipTests"
        exit 1
    }
    Write-Ok "All tests passed."
} else {
    Write-Warn "Tests skipped (-SkipTests flag set)."
}

# 2c. Stage all changes
git add .
Write-Ok "All changes staged."

# 2d. Show staged summary
Write-Host ""
Write-Host "--- PHASE 2 RESULT ---" -ForegroundColor Magenta
Write-Host "Staged files:"
git diff --cached --name-status | ForEach-Object { Write-Host "  $_" }
Write-Host ""

# 2e. Await explicit Accept
$accept = Read-Host "Type ACCEPT to proceed to commit, or anything else to abort"
if ($accept -ne "ACCEPT") {
    Write-Warn "Aborted by user. Changes are staged but NOT committed."
    Write-Warn "To commit manually: git commit -m 'your message' && git push -u origin $BranchName"
    exit 0
}

# ============================================================
# PHASE 3 — Atomic Commit & Push
# ============================================================
Write-Inf "PHASE 3: Atomic Commit & Push"
Write-Host "---------------------------------------"

# 3a. Build commit message
if ($CommitMessage -eq "") {
    $fileCount   = ($changedFiles | Measure-Object).Count
    $shortBranch = $BranchName -replace "feature/", ""
    $CommitMessage = "feat($shortBranch): update $fileCount file(s) - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    Write-Warn "No commit message supplied. Using: $CommitMessage"
}

# 3b. Commit
git commit -m $CommitMessage 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Err "Commit failed."
    exit 1
}
$commitHash = git rev-parse --short HEAD
Write-Ok "Committed: $commitHash"

# 3c. Push
git push -u origin $BranchName 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "Push failed. Check credentials or branch protection rules."
    exit 1
}

# 3d. Build PR URL
$repoUrl    = $remoteUrl -replace "\.git$", ""
$prUrl      = "$repoUrl/compare/$BranchName"
$branchUrl  = "$repoUrl/tree/$BranchName"

Write-Host ""
Write-Host "=======================================" -ForegroundColor Magenta
Write-Host "   PHASE 3 RESULT" -ForegroundColor Magenta
Write-Host "=======================================" -ForegroundColor Magenta
Write-Host "Commit hash  : $commitHash"
Write-Host "Branch URL   : $branchUrl"
Write-Host "Open PR at   : $prUrl"
Write-Host "=======================================" -ForegroundColor Magenta
Write-Host ""
