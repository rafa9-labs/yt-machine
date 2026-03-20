---
description: Complete MCP + Agent Git workflow for yt-machine development
---

# MCP + Agent Git Integrated Workflow

## Overview
This workflow combines agent git automation with MCP server management, ensuring all critical services (LLM, RSS scraper, memory system, debate engine) are running and healthy during feature development.

## Critical MCP Servers

| Server | Purpose | Status |
|--------|---------|--------|
| **LLM Interface** | Local Ollama for content generation | Critical |
| **RSS Scraper** | Fetches geopolitical news feeds | Critical |
| **Debate Engine** | Orchestrates analysis & script synthesis | Critical |
| **Memory Logger** | Logs videos to persistent storage | Critical |
| **Memory Reader** | Queries video history for duplicates | Critical |
| **Viking Bridge API** | FastAPI orchestration server | Optional |

## Workflow Phases

### Phase 1: Start Feature (Auto-starts MCP servers)
```powershell
.\.windsurf\agent-git-mcp.ps1 -Action "StartFeature" -FeatureName "memory-logger-enhancement"
```

**What happens:**
1. Starts all MCP servers (LLM, scraper, debate engine, memory system)
2. Validates server health
3. Creates feature branch
4. Stores feature metadata with MCP status

**Output:**
```
[MCP]   Starting MCP servers...
[OK]    LLM Interface: LLM connection verified
[OK]    RSS Feed Scraper: Module available
[OK]    Debate Engine: Module available
[OK]    Memory Logger: Module available
[OK]    Memory Reader: Module available
[OK]    MCP servers ready
[OK]    Created branch: feature/memory-logger-enhancement
[OK]    Feature started with MCP servers ready
```

### Phase 2: Stage Changes (Validates MCP health)
```powershell
.\.windsurf\agent-git-mcp.ps1 -Action "StageChanges"
```

**What happens:**
1. Validates MCP server health
2. Stages modified files
3. Updates feature metadata

### Phase 3: Complete Feature (Tests + Commit + Stop MCP)
```powershell
.\.windsurf\agent-git-mcp.ps1 -Action "CompleteFeature" -CommitMessage "feat: improve memory logging with structured output"
```

**What happens:**
1. Stages final changes
2. Runs pytest (optional with `-SkipTests`)
3. Validates MCP health one final time
4. Commits with structured message
5. Pushes branch
6. **Stops all MCP servers**
7. Returns PR URL

**Output:**
```
[AGENT] Completing feature...
[OK]    All tests passed
[AGENT] Final MCP health check...
[OK]    All MCP servers healthy
[OK]    Committed: a1b2c3d
[OK]    Pushed branch: feature/memory-logger-enhancement
[OK]    Feature completed and MCP servers stopped
PR URL: https://github.com/rafa9-labs/yt-machine/compare/feature/memory-logger-enhancement
```

### Phase 4: Cancel Feature (Stops MCP + switches to main)
```powershell
.\.windsurf\agent-git-mcp.ps1 -Action "CancelFeature"
```

**What happens:**
1. Stops all MCP servers
2. Clears feature metadata
3. Switches back to main branch

## MCP Server Management

### Manual Server Control
```powershell
# Start all MCP servers
.\.windsurf\mcp-integration.ps1 -Action "StartAll"

# Check health
.\.windsurf\mcp-integration.ps1 -Action "Health"

# View status
.\.windsurf\mcp-integration.ps1 -Action "Status"

# Restart all
.\.windsurf\mcp-integration.ps1 -Action "Restart"

# Warmup LLM model
.\.windsurf\mcp-integration.ps1 -Action "Warmup"
```

### Health Check Details
- **LLM Interface**: Verifies Ollama connection (requires local Ollama running)
- **RSS Scraper**: Validates module and config file
- **Debate Engine**: Checks dependencies (LLM, scraper, memory)
- **Memory System**: Verifies storage directories

## Configuration Files

### MCP Servers Config
`@.windsurf/mcp-servers.json` - Defines all servers, startup sequence, health checks

### Feature Metadata
`@open-viking/history/active_feature.json` - Stores active feature state:
```json
{
  "name": "memory-logger-enhancement",
  "branch": "feature/memory-logger-enhancement",
  "startTime": "2025-03-20 02:45:00",
  "files": ["open-viking/memory_logger.py"],
  "status": "active",
  "mcp_enabled": true
}
```

## Flags & Options

| Flag | Purpose |
|------|---------|
| `-SkipTests` | Skip pytest before commit |
| `-SkipMCP` | Don't start/stop MCP servers |
| `-Force` | Force feature start even if already on feature branch |

## Troubleshooting

### LLM Connection Failed
```
[ERR] LLM Interface: LLM connection failed - Ollama may not be running
```
**Solution:** Start Ollama locally before running workflow
```powershell
ollama serve
```

### MCP Server Startup Failed
```
[ERR] Some servers need attention
```
**Solution:** Check individual server health
```powershell
.\.windsurf\mcp-integration.ps1 -Action "Health"
```

### Tests Failed Before Commit
```
[ERR] Tests FAILED. Fix failures before committing.
```
**Solution:** Fix code and re-run, or use `-SkipTests` to bypass (not recommended)

## Integration with IDE

### Windsurf Hooks
- **Pre-edit**: Runs `StartFeature` automatically
- **Post-edit**: Runs `StageChanges` automatically
- **Completion**: Runs `CompleteFeature` when agent signals done

### Agent Memory
Feature metadata persists in `open-viking/history/active_feature.json` for cross-session tracking

## Best Practices

1. **Always use integrated workflow**: Use `agent-git-mcp.ps1` instead of `agent-git.ps1` directly
2. **Let MCP servers run**: Don't manually stop servers during feature work
3. **Validate health**: Check MCP health before critical operations
4. **Clean shutdown**: Always complete or cancel features to stop MCP servers
5. **Monitor logs**: Check `logs/mcp-integration.log` for issues

## Example: Full Feature Lifecycle

```powershell
# 1. Start feature with MCP servers
.\.windsurf\agent-git-mcp.ps1 -Action "StartFeature" -FeatureName "rss-parser-optimization"

# 2. Make changes (agent edits files)
# ... code changes ...

# 3. Stage changes (automatic or manual)
.\.windsurf\agent-git-mcp.ps1 -Action "StageChanges"

# 4. Check status
.\.windsurf\agent-git-mcp.ps1 -Action "Status"

# 5. Complete feature (tests + commit + push + stop MCP)
.\.windsurf\agent-git-mcp.ps1 -Action "CompleteFeature" -CommitMessage "feat: optimize RSS feed parsing with caching"

# 6. Create PR on GitHub
# Navigate to PR URL provided in output
```

## Logs & Monitoring

All MCP operations logged to: `logs/mcp-integration.log`

View logs:
```powershell
Get-Content logs/mcp-integration.log -Tail 50
```
