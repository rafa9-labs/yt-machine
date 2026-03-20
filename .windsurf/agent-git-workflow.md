---
description: Agent-aware git workflow for Windsurf IDE
---

# Agent Git Workflow

## Overview
When agents (Cascade, etc.) implement features in Windsurf, they automatically:
- Create feature branches on start
- Stage changes during work
- Commit with structured messages on completion
- Push and generate PR URLs

## Trigger Points

### 1. Feature Start (Agent begins work)
```powershell
.\agent-git.ps1 -Action "StartFeature" -FeatureName "redfish-scraper-improvement"
```
- Creates branch: `feature/redfish-scraper-improvement`
- Syncs with main
- Sets feature metadata

### 2. During Development (Agent makes changes)
```powershell
.\agent-git.ps1 -Action "StageChanges"
```
- Stages modified files
- Shows diff summary
- No commit yet

### 3. Feature Complete (Agent finishes)
```powershell
.\agent-git.ps1 -Action "CompleteFeature" -CommitMessage "feat: improve RSS parsing with error handling"
```
- Commits all staged changes
- Pushes branch
- Returns PR URL

## Integration Points

### Windsurf IDE Integration
- Pre-edit hook: runs `StartFeature` when agent begins major changes
- Post-edit hook: runs `StageChanges` after file modifications
- Completion hook: runs `CompleteFeature` when agent signals task done

### Agent Memory Integration
- Feature metadata stored in `open-viking/history/active_feature.json`
- Tracks: feature name, start time, files modified, completion status

## Usage Examples

### Agent Workflow
```powershell
# Agent starts implementing feature
.\agent-git.ps1 -Action "StartFeature" -FeatureName "memory-logger-enhancement"

# Agent makes changes (automatic staging)
.\agent-git.ps1 -Action "StageChanges"

# Agent completes feature
.\agent-git.ps1 -Action "CompleteFeature" -CommitMessage "feat: add structured logging to memory system"
```

### Manual Override
```powershell
# Force complete current feature
.\agent-git.ps1 -Action "CompleteFeature" -Force

# Cancel current feature (switch back to main)
.\agent-git.ps1 -Action "CancelFeature"
```

## File Structure
```
.windsurf/
├── agent-git-workflow.md      # This file
├── agent-git.ps1              # Main automation script
└── hooks/
    ├── pre-edit.ps1           # Runs before agent edits
    └── post-edit.ps1          # Runs after agent edits
```

## Configuration
Edit `agent-git.ps1` to customize:
- Default branch prefixes
- Test requirements
- Commit message templates
- PR templates
