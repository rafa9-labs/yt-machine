---
description: Complete agent feature with automatic git operations
---

# Agent Feature Complete Workflow

## When to Use
When an agent (Cascade, etc.) finishes implementing a feature and wants to automatically commit and push changes.

## Steps

### 1. Complete Current Feature
```powershell
.\.windsurf\agent-git.ps1 -Action "CompleteFeature" -CommitMessage "feat: add your feature description here"
```

### 2. Auto-Generated Actions
- Stages all modified files
- Runs tests (unless `-SkipTests` used)
- Creates structured commit message
- Pushes feature branch
- Generates PR URL
- Clears active feature metadata

### 3. Manual Options
```powershell
# Skip tests
.\.windsurf\agent-git.ps1 -Action "CompleteFeature" -SkipTests

# Force complete even if no active feature
.\.windsurf\agent-git.ps1 -Action "CompleteFeature" -Force

# Cancel current feature (switch back to main)
.\.windsurf\agent-git.ps1 -Action "CancelFeature"
```

## Integration with IDE

### Automatic Triggers
- **Pre-edit**: Creates feature branch when agent starts work
- **Post-edit**: Stages changes after file modifications
- **Complete**: Commits and pushes when agent signals completion

### Feature Metadata
Stored in `open-viking/history/active_feature.json`:
```json
{
  "name": "memory-logger-enhancement",
  "branch": "feature/memory-logger-enhancement",
  "startTime": "2025-03-20 02:45:00",
  "files": ["open-viking/memory_logger.py", "open-viking/memory_reader.py"],
  "status": "completed",
  "endTime": "2025-03-20 02:50:00",
  "commitHash": "a1b2c3d"
}
```

## Example Usage

### Agent Workflow
```powershell
# Agent starts implementing feature
.\.windsurf\agent-git.ps1 -Action "StartFeature" -FeatureName "rss-parser-fix"

# Agent makes changes (automatic staging via post-edit hook)
# ... files modified ...

# Agent completes feature
.\.windsurf\agent-git.ps1 -Action "CompleteFeature" -CommitMessage "fix: handle malformed RSS feeds gracefully"
```

### Output
```
[AGENT] Completing feature...
[OK] Staged 3 files
[OK] Committed: a1b2c3d
[OK] Pushed branch: feature/rss-parser-fix
[OK] Feature completed: rss-parser-fix
PR URL: https://github.com/rafa9-labs/yt-machine/compare/feature/rss-parser-fix
```

## Best Practices

1. **Feature Names**: Use descriptive, kebab-case names
2. **Commit Messages**: Follow conventional commits (`feat:`, `fix:`, `docs:`, etc.)
3. **Test Requirements**: Use `-SkipTests` only when no test suite exists
4. **Branch Cleanup**: Switch back to main after completing features
5. **PR Reviews**: Review PR URLs before merging to main
