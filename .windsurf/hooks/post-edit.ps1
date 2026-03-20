# ============================================================
# post-edit.ps1
# Windsurf IDE hook: Runs after agent makes changes
# Automatically stages changes and updates feature metadata
# ============================================================

param(
    [Parameter(Mandatory=$false)]
    [string]$ModifiedFiles = ""
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$AgentScript = Join-Path $PSScriptRoot "..\agent-git.ps1"

# Stage changes
& $AgentScript -Action "StageChanges"

# Show summary
$active = Get-Content "$ProjectRoot\open-viking\history\active_feature.json" | ConvertFrom-Json
if ($active) {
    Write-Host "[HOOK] Feature: $($active.name) | Files: $($active.files.Count)"
}
