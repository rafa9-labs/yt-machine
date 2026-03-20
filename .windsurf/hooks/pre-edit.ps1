# ============================================================
# pre-edit.ps1
# Windsurf IDE hook: Runs before agent makes changes
# Automatically creates feature branch if needed
# ============================================================

param(
    [Parameter(Mandatory=$false)]
    [string]$AgentName = "Cascade",

    [Parameter(Mandatory=$false)]
    [string]$FeatureHint = ""
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$AgentScript = Join-Path $PSScriptRoot "..\agent-git.ps1"

# Check if we're already on a feature branch
$branch = git branch --show-current
if ($branch -like "feature/*") {
    Write-Host "[HOOK] Already on feature branch: $branch"
    exit 0
}

# Auto-generate feature name if not provided
if (-not $FeatureHint) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmm"
    $FeatureHint = "$AgentName-$timestamp"
}

# Start new feature
& $AgentScript -Action "StartFeature" -FeatureName $FeatureHint
Write-Host "[HOOK] Started feature: $FeatureHint"
