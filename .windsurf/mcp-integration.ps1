# ============================================================
# mcp-integration.ps1
# MCP Server Integration & Health Monitoring
# Manages startup, health checks, and agent coordination
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("StartAll", "StopAll", "Health", "Status", "Restart", "Warmup")]
    [string]$Action,

    [Parameter(Mandatory=$false)]
    [string]$Server = "",

    [Parameter(Mandatory=$false)]
    [switch]$Verbose
)

# ── Paths & Config ───────────────────────────────────────
$ProjectRoot   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ConfigFile    = Join-Path $PSScriptRoot "mcp-servers.json"
$LogDir        = Join-Path $ProjectRoot "logs"
$LogFile       = Join-Path $LogDir "mcp-integration.log"

# Create logs directory
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# ── Colours ─────────────────────────────────────────────────
function Write-Ok  ($msg) { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Write-Err ($msg) { Write-Host "[ERR]  $msg" -ForegroundColor Red }
function Write-Inf ($msg) { Write-Host "[MCP]  $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

function Log-Message($msg, $level = "INFO") {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp [$level] $msg" | Add-Content $LogFile
}

# ── Load MCP Config ──────────────────────────────────────
function Get-MCPConfig {
    if (-not (Test-Path $ConfigFile)) {
        Write-Err "MCP config not found: $ConfigFile"
        return $null
    }
    return Get-Content $ConfigFile | ConvertFrom-Json
}

# ── Server Management ────────────────────────────────────
function Test-ServerHealth($serverName, $config) {
    $server = $config.mcp_servers.$serverName
    
    if (-not $server) {
        return @{ healthy = $false; reason = "Server not found in config" }
    }

    # Check if module/class exists
    $modulePath = Join-Path $ProjectRoot $server.module
    if (-not (Test-Path $modulePath)) {
        return @{ healthy = $false; reason = "Module not found: $($server.module)" }
    }

    # Check config file if specified
    if ($server.config) {
        $configPath = Join-Path $ProjectRoot $server.config
        if (-not (Test-Path $configPath)) {
            return @{ healthy = $false; reason = "Config not found: $($server.config)" }
        }
    }

    # Check storage if specified
    if ($server.storage) {
        $storagePath = Join-Path $ProjectRoot $server.storage
        $storageDir = Split-Path -Parent $storagePath
        if (-not (Test-Path $storageDir)) {
            Write-Warn "Storage directory missing, will be created: $storageDir"
        }
    }

    # Special health check for LLM interface
    if ($serverName -eq "llm_interface") {
        try {
            $pythonCheck = python -c "
import sys
sys.path.insert(0, '$ProjectRoot')
from brain.llm_interface import LLMInterface
llm = LLMInterface()
print('OK' if llm.check_connection() else 'FAIL')
" 2>&1
            
            if ($pythonCheck -like "*OK*") {
                return @{ healthy = $true; reason = "LLM connection verified" }
            } else {
                return @{ healthy = $false; reason = "LLM connection failed - Ollama may not be running" }
            }
        } catch {
            return @{ healthy = $false; reason = "LLM health check error: $_" }
        }
    }

    return @{ healthy = $true; reason = "Module available" }
}

function Start-MCPServer($serverName, $config) {
    $server = $config.mcp_servers.$serverName
    
    if (-not $server) {
        Write-Err "Server not found: $serverName"
        return $false
    }

    Write-Inf "Starting $($server.name)..."

    # Special handling for FastAPI servers
    if ($server.type -eq "fastapi") {
        $pythonScript = Join-Path $ProjectRoot $server.module
        
        # Check if already running
        $existing = Get-Process -Name "python" -ErrorAction SilentlyContinue | 
                    Where-Object { $_.CommandLine -like "*$($server.module)*" }
        
        if ($existing) {
            Write-Warn "Server already running (PID: $($existing.Id))"
            return $true
        }

        # Start in background
        Start-Process -FilePath "python" -ArgumentList $pythonScript -WorkingDirectory $ProjectRoot -NoNewWindow
        Start-Sleep -Seconds 2
        
        # Verify startup
        try {
            $health = Invoke-WebRequest -Uri "http://localhost:$($server.port)/health" -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($health.StatusCode -eq 200) {
                Write-Ok "$($server.name) started on port $($server.port)"
                Log-Message "Started $serverName on port $($server.port)" "INFO"
                return $true
            }
        } catch {
            Write-Warn "Server started but health check failed - may still be initializing"
            return $true
        }
    }

    Write-Ok "$($server.name) ready"
    Log-Message "Started $serverName" "INFO"
    return $true
}

function Stop-MCPServer($serverName, $config) {
    $server = $config.mcp_servers.$serverName
    
    if (-not $server) {
        Write-Err "Server not found: $serverName"
        return $false
    }

    Write-Inf "Stopping $($server.name)..."

    if ($server.type -eq "fastapi") {
        Get-Process -Name "python" -ErrorAction SilentlyContinue | 
            Where-Object { $_.CommandLine -like "*$($server.module)*" } | 
            Stop-Process -Force
        
        Write-Ok "$($server.name) stopped"
        Log-Message "Stopped $serverName" "INFO"
    }

    return $true
}

function Warmup-LLMModel($config) {
    Write-Inf "Warming up LLM model..."
    
    try {
        python -c "
import sys
sys.path.insert(0, '$ProjectRoot')
from brain.llm_interface import LLMInterface
llm = LLMInterface()
llm.warmup_model()
" 2>&1 | ForEach-Object { Write-Host "  $_" }
        
        Write-Ok "LLM warmup complete"
        Log-Message "LLM warmup completed" "INFO"
        return $true
    } catch {
        Write-Err "LLM warmup failed: $_"
        Log-Message "LLM warmup failed: $_" "ERROR"
        return $false
    }
}

# ============================================================
# ACTIONS
# ============================================================

$config = Get-MCPConfig
if (-not $config) { exit 1 }

switch ($Action) {
    "StartAll" {
        Write-Host ""
        Write-Host "=======================================" -ForegroundColor Magenta
        Write-Host "   MCP SERVERS  |  STARTUP" -ForegroundColor Magenta
        Write-Host "=======================================" -ForegroundColor Magenta
        Write-Host ""

        $startupSeq = $config.startup_sequence
        $failedServers = @()

        foreach ($serverName in $startupSeq) {
            $health = Test-ServerHealth $serverName $config
            
            if (-not $health.healthy) {
                Write-Warn "$serverName: $($health.reason)"
                $failedServers += $serverName
                continue
            }

            if (-not (Start-MCPServer $serverName $config)) {
                $failedServers += $serverName
            }
        }

        Write-Host ""
        Write-Host "--- STARTUP SUMMARY ---" -ForegroundColor Magenta
        Write-Host "Total servers: $($startupSeq.Count)"
        Write-Host "Failed: $($failedServers.Count)"
        
        if ($failedServers.Count -gt 0) {
            Write-Host "Failed servers: $($failedServers -join ', ')" -ForegroundColor Red
        } else {
            Write-Ok "All servers started successfully"
        }

        # Warmup LLM if available
        if ($failedServers -notcontains "llm_interface") {
            Write-Host ""
            Warmup-LLMModel $config
        }

        Write-Host ""
    }

    "StopAll" {
        Write-Inf "Stopping all MCP servers..."
        
        foreach ($serverName in $config.startup_sequence) {
            Stop-MCPServer $serverName $config
        }
        
        Write-Ok "All servers stopped"
    }

    "Health" {
        Write-Host ""
        Write-Host "--- MCP SERVERS HEALTH CHECK ---" -ForegroundColor Magenta
        Write-Host ""

        $allHealthy = $true
        
        foreach ($serverName in $config.startup_sequence) {
            $server = $config.mcp_servers.$serverName
            $health = Test-ServerHealth $serverName $config
            
            if ($health.healthy) {
                Write-Ok "$($server.name): $($health.reason)"
            } else {
                Write-Err "$($server.name): $($health.reason)"
                $allHealthy = $false
            }
        }

        Write-Host ""
        if ($allHealthy) {
            Write-Ok "All servers healthy"
        } else {
            Write-Warn "Some servers need attention"
        }
    }

    "Status" {
        Write-Host ""
        Write-Host "--- MCP SERVERS STATUS ---" -ForegroundColor Magenta
        Write-Host ""

        foreach ($serverName in $config.startup_sequence) {
            $server = $config.mcp_servers.$serverName
            $health = Test-ServerHealth $serverName $config
            
            $status = if ($health.healthy) { "✓ READY" } else { "✗ DOWN" }
            Write-Host "$status | $($server.name)"
            Write-Host "     $($server.description)"
            Write-Host ""
        }
    }

    "Restart" {
        Write-Inf "Restarting all MCP servers..."
        
        & $PSScriptRoot\mcp-integration.ps1 -Action "StopAll"
        Start-Sleep -Seconds 2
        & $PSScriptRoot\mcp-integration.ps1 -Action "StartAll"
    }

    "Warmup" {
        $health = Test-ServerHealth "llm_interface" $config
        if ($health.healthy) {
            Warmup-LLMModel $config
        } else {
            Write-Err "LLM interface not healthy: $($health.reason)"
        }
    }
}

Write-Host ""
