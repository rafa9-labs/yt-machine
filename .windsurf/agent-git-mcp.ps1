# ============================================================
# agent-git-mcp.ps1
# Integrated Agent Git + MCP Server Workflow
# Automatically manages MCP servers during feature development
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
    [switch]$SkipTests,

    [Parameter(Mandatory=$false)]
    [switch]$SkipMCP
)

# ── Paths & Config ───────────────────────────────────────
$ProjectRoot    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$FeatureMeta    = "$ProjectRoot\open-viking\history\active_feature.json"
$FeaturePrefix  = "feature"
$AgentGitScript = Join-Path $PSScriptRoot "agent-git.ps1"
$MCPScript      = Join-Path $PSScriptRoot "mcp-integration.ps1"

# ── Colours ─────────────────────────────────────────────────
function Write-Ok  ($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Err ($msg) { Write-Host "[ERR]  $msg" -ForegroundColor Red }
function Write-Inf ($msg) { Write-Host "[AGENT] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-MCP ($msg) { Write-Host "[MCP]  $msg" -ForegroundColor Magenta }

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

function Start-MCPServers {
    if ($SkipMCP) {
        Write-Warn "MCP servers skipped (-SkipMCP flag)"
        return $true
    }

    Write-MCP "Starting MCP servers..."
    & $MCPScript -Action "StartAll" 2>&1 | ForEach-Object {
        if ($_ -match "\[OK\]") {
            Write-Ok $_
        } elseif ($_ -match "\[ERR\]") {
            Write-Err $_
        } elseif ($_ -match "\[WARN\]") {
            Write-Warn $_
        } else {
            Write-Host $_
        }
    }

    # Check health
    $health = & $MCPScript -Action "Health" 2>&1
    $hasErrors = $health | Select-String "\[ERR\]"
    
    if ($hasErrors) {
        Write-Warn "Some MCP servers failed to start. Continuing anyway..."
        return $false
    }

    Write-Ok "MCP servers ready"
    return $true
}

function Stop-MCPServers {
    if ($SkipMCP) { return }
    
    Write-MCP "Stopping MCP servers..."
    & $MCPScript -Action "StopAll" 2>&1 | Out-Null
    Write-Ok "MCP servers stopped"
}

function Run-AgentTests {
    if ($SkipTests) {
        Write-Warn "Tests skipped (-SkipTests flag)"
        return $true
    }

    Write-Inf "Running agent tests..."
    
    $pytest = Get-Command pytest -ErrorAction SilentlyContinue
    if ($pytest) {
        pytest --tb=short -q 2>&1 | Out-Null
        $testResult = $LASTEXITCODE
    } else {
        Write-Warn "pytest not found. Trying 'python -m pytest'..."
        python -m pytest --tb=short -q 2>&1 | Out-Null
        $testResult = $LASTEXITCODE
    }

    if ($testResult -ne 0) {
        Write-Err "Tests FAILED. Fix failures before committing."
        return $false
    }

    Write-Ok "All tests passed"
    return $true
}

function Validate-MCPHealth {
    Write-Inf "Validating MCP server health..."
    
    $health = & $MCPScript -Action "Health" 2>&1
    $hasErrors = $health | Select-String "\[ERR\]"
    
    if ($hasErrors) {
        Write-Warn "Some MCP servers are unhealthy"
        return $false
    }

    Write-Ok "All MCP servers healthy"
    return $true
}

# ============================================================
# ACTIONS
# ============================================================

Write-Host ""
Write-Host "=======================================" -ForegroundColor Magenta
Write-Host "   AGENT-GIT + MCP INTEGRATION" -ForegroundColor Magenta
Write-Host "=======================================" -ForegroundColor Magenta
Write-Host ""

switch ($Action) {
    "StartFeature" {
        Write-Inf "Starting new feature: $FeatureName"
        
        if (-not $FeatureName) {
            Write-Err "FeatureName required for StartFeature"
            exit 1
        }

        # Start MCP servers first
        if (-not (Start-MCPServers)) {
            Write-Warn "MCP startup had issues, but continuing..."
        }

        # Create feature branch via agent-git
        & $AgentGitScript -Action "StartFeature" -FeatureName $FeatureName
        
        # Store MCP status in feature metadata
        $active = Get-ActiveFeature
        if ($active) {
            $active | Add-Member -NotePropertyName "mcp_enabled" -NotePropertyValue (-not $SkipMCP) -Force
            Set-ActiveFeature $active
        }

        Write-Ok "Feature started with MCP servers ready"
    }

    "StageChanges" {
        Write-Inf "Staging changes..."
        
        $active = Get-ActiveFeature
        if (-not $active) {
            Write-Warn "No active feature found. Starting auto-feature..."
            $timestamp = Get-Date -Format "yyyyMMdd-HHmm"
            & $PSScriptRoot\agent-git-mcp.ps1 -Action "StartFeature" -FeatureName "auto-$timestamp"
            $active = Get-ActiveFeature
        }

        # Validate MCP health before staging
        if ($active.mcp_enabled -and -not $SkipMCP) {
            if (-not (Validate-MCPHealth)) {
                Write-Warn "MCP health check failed, but continuing with staging..."
            }
        }

        # Stage changes via agent-git
        & $AgentGitScript -Action "StageChanges"
        
        Write-Ok "Changes staged"
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

        # Stage final changes
        git add . 2>&1 | Out-Null
        
        # Check if there are changes to commit
        $hasChanges = git status --porcelain
        if (-not $hasChanges) {
            Write-Warn "No changes to commit"
            Clear-ActiveFeature
            Stop-MCPServers
            exit 0
        }

        # Run tests
        if (-not (Run-AgentTests)) {
            Write-Err "Tests failed. Aborting commit."
            Stop-MCPServers
            exit 1
        }

        # Validate MCP health before final commit
        if ($active.mcp_enabled -and -not $SkipMCP) {
            Write-Inf "Final MCP health check..."
            if (-not (Validate-MCPHealth)) {
                Write-Warn "MCP health check failed, but continuing with commit..."
            }
        }

        # Generate commit message if not provided
        if (-not $CommitMessage) {
            $fileCount = ($active.files | Measure-Object).Count
            $CommitMessage = "feat($($active.name)): update $fileCount file(s)"
        }

        # Commit via agent-git
        & $AgentGitScript -Action "CompleteFeature" -CommitMessage $CommitMessage
        
        # Stop MCP servers after feature complete
        Stop-MCPServers

        Write-Ok "Feature completed and MCP servers stopped"
    }

    "CancelFeature" {
        Write-Inf "Cancelling current feature..."
        
        $active = Get-ActiveFeature
        if ($active) {
            Write-Warn "Cancelling feature: $($active.name)"
            Clear-ActiveFeature
        }

        # Stop MCP servers
        Stop-MCPServers

        # Switch back to main
        git checkout main 2>&1 | Out-Null
        Write-Ok "Switched to main branch"
    }

    "Status" {
        Write-Inf "Agent Git + MCP Status"
        Write-Host "Current branch: $(Get-CurrentBranch)"
        
        $active = Get-ActiveFeature
        if ($active) {
            Write-Host "Active feature: $($active.name)" -ForegroundColor Green
            Write-Host "  Branch: $($active.branch)"
            Write-Host "  Started: $($active.startTime)"
            Write-Host "  Files: $($active.files.Count)"
            Write-Host "  MCP enabled: $($active.mcp_enabled)"
            Write-Host "  Status: $($active.status)"
        } else {
            Write-Host "No active feature" -ForegroundColor Yellow
        }

        # MCP Status
        Write-Host ""
        Write-MCP "MCP Server Status:"
        & $MCPScript -Action "Status" 2>&1 | ForEach-Object {
            Write-Host "  $_"
        }

        # Git status
        Write-Host ""
        $gitStatus = git status --porcelain
        if ($gitStatus) {
            Write-Host "Uncommitted changes:" -ForegroundColor Cyan
            $gitStatus | ForEach-Object { Write-Host "  $_" }
        }
    }
}

Write-Host ""
