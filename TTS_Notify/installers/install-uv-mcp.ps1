# =============================================================================
# TTS-Notify Dynamic UV/UVX MCP Installer for Windows
# =============================================================================
# Auto-detects AI tools and generates MCP configurations
# Supports: Claude Desktop, Claude Code, Cursor, Continue
# Compatible with PowerShell 5.1+ (Windows 10/11 default)
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Position=0, HelpMessage="Installation mode: local, uvx, auto, config-only")]
    [ValidateSet('local', 'uvx', 'auto', 'config-only')]
    [string]$Mode = 'auto'
)

# Exit on any error
$ErrorActionPreference = 'Stop'

# Colors and formatting
$Colors = @{
    Reset  = ''
    Red    = '\033[0;31m'
    Green  = '\033[0;32m'
    Yellow = '\033[1;33m'
    Blue   = '\033[0;34m'
    Cyan   = '\033[0;36m'
    Bold   = '\033[1m'
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [ValidateSet('Reset', 'Red', 'Green', 'Yellow', 'Blue', 'Cyan', 'Bold')]
        [string]$Color = 'Reset'
    )

    $colorMap = @{
        'Reset'  = $Colors.Reset
        'Red'    = $Colors.Red
        'Green'  = $Colors.Green
        'Yellow' = $Colors.Yellow
        'Blue'   = $Colors.Blue
        'Cyan'   = $Colors.Cyan
        'Bold'   = $Colors.Bold
    }

    $foreground = $colorMap[$Color]
    $formattedMessage = "${foreground}${Message}${$Colors.Reset}"
    Write-Host $formattedMessage
}

# Project paths
$SCRIPT_DIR = Split-Path -Parent $PSScriptRoot
$PROJECT_DIR = Split-Path -Parent $SCRIPT_DIR
$PACKAGE_NAME = "tts-notify"
$MCP_ENTRY = "tts_notify.main:sync_main"

# AI Tool detection paths (Windows paths)
    $AI_TOOL_PATHS = @{
        "claude-desktop" = Join-Path $env:APPDATA "Claude"
        "claude-code"    = Join-Path $env:USERPROFILE ".claude"
        "opencode"       = Join-Path $env:APPDATA "Cursor\User\globalStorage"
        "cursor"         = Join-Path $env:APPDATA "Cursor\User\globalStorage"
        "continue"       = Join-Path $env:USERPROFILE ".continue"
    }

$TOOLS = @('claude-desktop', 'claude-code', 'opencode', 'continue')

# Detected tools
$DetectedTools = @()

# Installation mode
$INSTALL_MODE = $Mode
$VENV_PATH = ""
$USE_UVX = $false

# UV/UVX availability flags
$UV_AVAILABLE = $false
$UVX_AVAILABLE = $false
$UV_PATH = ""
$UVX_PATH = ""
$MCP_COMMAND = ""
$MCP_ARGS = ""

# =============================================================================
# Helper Functions
# =============================================================================

function Show-Header {
    Write-ColorOutput "`n" "Cyan"
    Write-ColorOutput "╔══════════════════════════════════════════════════════════════╗" "Cyan"
    Write-ColorOutput "║" "Cyan" -NoNewline
    Write-ColorOutput "  ${$Colors.Bold}🔔 TTS-Notify MCP Installer (UV/UVX)${$Colors.Reset}" "Cyan" -NoNewline
    Write-ColorOutput "                         " "Cyan" -NoNewline
    Write-ColorOutput "║" "Cyan"
    Write-ColorOutput "║" "Cyan" -NoNewline
    Write-ColorOutput "  ${$Colors.Yellow}Dynamic MCP Configuration Generator${$Colors.Reset}" "Cyan" -NoNewline
    Write-ColorOutput "                         " "Cyan" -NoNewline
    Write-ColorOutput "║" "Cyan"
    Write-ColorOutput "╚══════════════════════════════════════════════════════════════╝" "Cyan"
    Write-ColorOutput "`n" "Reset"
}

function Show-Status {
    param([string]$Message)
    Write-ColorOutput "✅ $Message" "Green"
}

function Show-Warning {
    param([string]$Message)
    Write-ColorOutput "⚠️  $Message" "Yellow"
}

function Show-Error {
    param([string]$Message)
    Write-ColorOutput "❌ $Message" "Red"
}

function Show-Info {
    param([string]$Message)
    Write-ColorOutput "ℹ️  $Message" "Blue"
}

function Show-Section {
    param([string]$Message)
    Write-ColorOutput "`n${$Colors.Bold}$Message${$Colors.Reset}`n" "Reset"
    Write-ColorOutput "─────────────────────────────────────────" "Cyan"
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Get-PythonPath {
    $venvs = @(
        (Join-Path $PROJECT_DIR "venv312"),
        (Join-Path $PROJECT_DIR "venv313"),
        (Join-Path $PROJECT_DIR "venv"),
        (Join-Path $PROJECT_DIR "test_venv")
    )

    foreach ($venv in $venvs) {
        if (Test-Path (Join-Path $venv "python.exe")) {
            return (Join-Path $venv "python.exe")
        }
    }

    # Fallback to system Python
    if (Test-Command "python") {
        $pythonPath = (Get-Command python).Path
        if ($pythonPath) { return $pythonPath }
    }

    if (Test-Command "python3") {
        $pythonPath = (Get-Command python3).Path
        if ($pythonPath) { return $pythonPath }
    }

    return $null
}

# =============================================================================
# UV/UVX Detection
# =============================================================================

function Detect-UV {
    Show-Section "🔍 Detectando UV/UVX"

    # Reset flags first
    $script:UV_AVAILABLE = $false
    $script:UVX_AVAILABLE = $false
    $script:UV_PATH = ""
    $script:UVX_PATH = ""

    if (Test-Command "uv") {
        $uvVersion = uv --version 2>$null
        $uvPath = (Get-Command uv).Path
        $script:UV_PATH = $uvPath
        $script:UV_VERSION = $uvVersion
        $script:UV_AVAILABLE = $true
        Show-Status "UV encontrado: $uvPath ($uvVersion)"
        return $true
    }
    elseif (Test-Command "uvx") {
        $uvxPath = (Get-Command uvx).Path
        $script:UVX_PATH = $uvxPath
        $script:UVX_AVAILABLE = $true
        Show-Status "UVX encontrado: $uvxPath"
        return $true
    }
    else {
        $script:UV_AVAILABLE = $false
        $script:UVX_AVAILABLE = $false
        Show-Warning "UV/UVX no encontrado"
        Show-Info "Instala UV: Invoke-WebRequest -Uri https://astral.sh/uv/install.sh -UseBasicParsing | Invoke-Expression"
        return $false
    }
}

function Get-McpCommand {
    $pythonPath = Get-PythonPath

    if ($script:UV_AVAILABLE) {
        $script:MCP_COMMAND = $script:UV_PATH
        $script:MCP_ARGS = '["run", "--python", "' + $pythonPath + '", "-m", "tts_notify", "--mode", "mcp"]'
        return "uv"
    }
    elseif ($script:UVX_AVAILABLE) {
        $script:MCP_COMMAND = $script:UVX_PATH
        $script:MCP_ARGS = '["tts-notify", "--mode", "mcp"]'
        return "uvx"
    }
    else {
        $script:MCP_COMMAND = $pythonPath
        $script:MCP_ARGS = '["-m", "tts_notify", "--mode", "mcp"]'
        return "python"
    }
}

# =============================================================================
# AI Tool Detection
# =============================================================================

function Detect-AITools {
    Show-Section "🔍 Detectando Herramientas de IA"

    $detectedCount = 0

    foreach ($tool in $TOOLS) {
        $path = $AI_TOOL_PATHS[$tool]
        if (Test-Path $path) {
            $DetectedTools += "$tool|$path"
            Show-Status "$tool: $path"
            $detectedCount++
        }
        else {
            Write-Host "  ○ $tool: no detectado" -ForegroundColor Yellow
        }
    }

    if ($detectedCount -eq 0) {
        Show-Warning "No se detectaron herramientas de IA"
        return $false
    }

    Show-Info "$detectedCount herramienta(s) detectada(s)"
    return $true
}

# =============================================================================
# MCP Configuration Generators
# =============================================================================

function Generate-MCPConfig-ClaudeDesktop {
    param([string]$ConfigDir)

    $configFile = Join-Path $ConfigDir "claude_desktop_config.json"

    Show-Info "Generando config para Claude Desktop..."

    # Backup existing config
    if (Test-Path $configFile) {
        $backupFile = "$configFile.backup.$(Get-Date -Format 'yyyyMMddHHmmss')"
        Copy-Item $configFile $backupFile
        Show-Warning "Backup creado: $backupFile"
    }

    $config = @{
        mcpServers = @{
            "tts-notify" = @{
                command = $MCP_COMMAND
                args = $MCP_ARGS -as [object[]]
                env = @{
                    TTS_NOTIFY_ENGINE = "macos"
                    TTS_NOTIFY_VOICE = "Monica"
                    TTS_NOTIFY_RATE = "175"
                }
            }
        }
    }

    $configJson = $config | ConvertTo-Json -Depth 3
    Set-Content -Path $configFile -Value $configJson

    Show-Status "Claude Desktop: $configFile"
}

function Generate-MCPConfig-ClaudeCode {
    param([string]$ConfigDir)

    $configFile = Join-Path $ConfigDir ".mcp.json"

    Show-Info "Generando config para Claude Code..."

    # Backup existing config
    if (Test-Path $configFile) {
        $backupFile = "$configFile.backup.$(Get-Date -Format 'yyyyMMddHHmmss')"
        Copy-Item $configFile $backupFile
    }

    $config = @{
        "tts-notify" = @{
            command = $MCP_COMMAND
            args = $MCP_ARGS -as [object[]]
            env = @{
                TTS_NOTIFY_ENGINE = "macos"
                TTS_NOTIFY_VOICE = "Monica"
                TTS_NOTIFY_RATE = "175"
            }
        }
    }

    $configJson = $config | ConvertTo-Json -Depth 3
    Set-Content -Path $configFile -Value $configJson

    Show-Status "Claude Code: $configFile"
}

function Generate-MCPConfig-Continue {
    param([string]$ConfigDir)

    $configFile = Join-Path $ConfigDir "config.json"

    Show-Info "Generando config para Continue.dev..."

    # Ensure directory exists
    New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null

    # Backup existing config
    if (Test-Path $configFile) {
        $backupFile = "$configFile.backup.$(Get-Date -Format 'yyyyMMddHHmmss')"
        Copy-Item $configFile $backupFile
    }

    $config = @{
        mcpServers = @{
            "tts-notify" = @{
                command = $MCP_COMMAND
                args = $MCP_ARGS -as [object[]]
            }
        }
    }

    $configJson = $config | ConvertTo-Json -Depth 3
    Set-Content -Path $configFile -Value $configJson

    Show-Status "Continue: $configFile"
}

function Generate-MCPConfig-OpenCode {
    param([string]$ConfigDir)

    $configFile = Join-Path $ConfigDir "mcp.json"

    Show-Info "Generando config para OpenCode..."

    # Ensure directory exists
    New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null

    # Backup existing config
    if (Test-Path $configFile) {
        $backupFile = "$configFile.backup.$(Get-Date -Format 'yyyyMMddHHmmss')"
        Copy-Item $configFile $backupFile
    }

    $config = @{
        "tts-notify" = @{
            command = $MCP_COMMAND
            args = $MCP_ARGS -as [object[]]
            env = @{
                TTS_NOTIFY_ENGINE = "macos"
                TTS_NOTIFY_VOICE = "Monica"
            }
        }
    }

    $configJson = $config | ConvertTo-Json -Depth 3
    Set-Content -Path $configFile -Value $configJson

    Show-Status "OpenCode: $configFile"
}

function Generate-MCPConfig-Cursor {
    param([string]$ConfigDir)

    $configFile = Join-Path $ConfigDir "mcp.json"

    Show-Info "Generando config para Cursor..."

    # Ensure directory exists
    $globalStoragePath = Join-Path $ConfigDir "..\mcp.json"
    $parentDir = Split-Path $ConfigDir

    New-Item -ItemType Directory -Force -Path $parentDir | Out-Null

    # Backup existing config
    if (Test-Path $configFile) {
        $backupFile = "$configFile.backup.$(Get-Date -Format 'yyyyMMddHHmmss')"
        Copy-Item $configFile $backupFile
    }

    $config = @{
        mcpServers = @{
            "tts-notify" = @{
                command = $MCP_COMMAND
                args = $MCP_ARGS -as [object[]]
            }
        }
    }

    $configJson = $config | ConvertTo-Json -Depth 3
    Set-Content -Path $configFile -Value $configJson

    Show-Status "Cursor: $configFile"
}

# =============================================================================
# Configuration Generation
# =============================================================================

function Generate-AllConfigs {
    Show-Section "⚙️  Generando Configuraciones MCP"

    foreach ($toolEntry in $DetectedTools) {
        $tool = $toolEntry.Split('|')[0]
        $path = $toolEntry.Split('|')[1]

        switch ($tool) {
            "claude-desktop" {
                Generate-MCPConfig-ClaudeDesktop $path
            }
            "claude-code" {
                Generate-MCPConfig-ClaudeCode $path
            }
            "opencode" {
                Generate-MCPConfig-OpenCode $path
            }
            "cursor" {
                Generate-MCPConfig-Cursor $path
            }
            "continue" {
                Generate-MCPConfig-Continue $path
            }
            default {
                Show-Info "$tool: Configuración no implementada aún"
            }
        }
    }
}

# =============================================================================
# Installation Functions
# =============================================================================

function Install-With-UV {
    Show-Section "📦 Instalación con UV"

    Set-Location $PROJECT_DIR

    # Create venv if needed
    $venv312Path = Join-Path $PROJECT_DIR "venv312"
    $venvPath = Join-Path $PROJECT_DIR "venv"

    if (-not (Test-Path $venv312Path) -and -not (Test-Path $venvPath)) {
        Show-Info "Creando entorno virtual..."
        uv venv venv
    }

    # Install package
    if (Test-Path $venvPath) {
        Show-Info "Instalando dependencias en venv..."
        uv pip install -e ".[fastmcp]" --python "$venvPath\python.exe"
    }
    elseif (Test-Path $venv312Path) {
        Show-Info "Instalando dependencias en venv312..."
        uv pip install -e ".[fastmcp]" --python "$venv312Path\python.exe"
    }

    Show-Status "Instalación completada"
}

function Install-With-UVX {
    Show-Section "📦 Instalación con UVX"

    Show-Info "UVX no requiere instalación local"
    Show-Info "El servidor se ejecutará con: uvx tts-notify --mode mcp"

    $USE_UVX = $true
}

# =============================================================================
# Interactive Menu
# =============================================================================

function Show-Menu {
    Show-Header

    Write-ColorOutput "Modo de instalación:" "Bold"
    Write-Host "  1) 🖥️  Local (venv) - Instalar en proyecto con UV" -ForegroundColor Cyan
    Write-Host "  2) ☁️  UVX (sin venv) - Ejecutar globalmente" -ForegroundColor Cyan
    Write-Host "  3) 🔍  Detectar automáticamente" -ForegroundColor Cyan
    Write-Host "  4) ⚙️  Solo generar configs MCP" -ForegroundColor Cyan
    Write-Host "  5) ❌ Salir" -ForegroundColor Cyan
    Write-Host ""

    Write-ColorOutput "Herramientas de IA detectadas:" "Bold"
    foreach ($toolEntry in $DetectedTools) {
        $tool = $toolEntry.Split('|')[0]
        Write-Host "  ✓ $tool" -ForegroundColor Green
    }
    Write-Host ""

    $choice = Read-Host "Selecciona una opción [1-5]"

    switch ($choice) {
        '1' { $INSTALL_MODE = 'local' }
        '2' { $INSTALL_MODE = 'uvx' }
        '3' { $INSTALL_MODE = 'auto' }
        '4' { $INSTALL_MODE = 'config-only' }
        '5' {
            Write-ColorOutput "Saliendo..." "Reset"
            exit 0
        }
        default {
            Show-Error "Opción inválida"
            exit 1
        }
    }
}

# =============================================================================
# Main
# =============================================================================

function Main {
    # Detect tools first
    if (-not (Detect-AITools)) {
        Write-Host "⚠️  No se detectaron herramientas de IA" -ForegroundColor Yellow
    }

    # Show menu
    Show-Menu

    # Check UV
    if ($INSTALL_MODE -ne 'config-only') {
        if (-not (Detect-UV)) {
            Show-Warning "Continuando sin UV (usando pip)"
        }
    }

    # Execute installation
    switch ($INSTALL_MODE) {
        'local' {
            Install-With-UV
        }
        'uvx' {
            Install-With-UVX
        }
        'auto' {
            if (Detect-UV) {
                Install-With-UV
            }
            else {
                Show-Error "No se puede instalar sin UV"
                exit 1
            }
        }
        'config-only' {
            Show-Info "Solo generando configuraciones..."
        }
    }

    # Get MCP command
    Get-McpCommand
    Show-Info "Using command: $MCP_COMMAND"

    # Generate MCP configs
    Generate-AllConfigs

    # Summary
    Show-Section "✨ Instalación Completada"

    Write-Host ""

    Write-ColorOutput "Configuraciones generadas:" "Bold"
    foreach ($toolEntry in $DetectedTools) {
        $tool = $toolEntry.Split('|')[0]
        Write-Host "  ✓ $tool" -ForegroundColor Green
    }

    Write-Host ""
    Write-ColorOutput "Para usar TTS-Notify:" "Bold"
    Write-Host "  1. Reinicia tu herramienta de IA" -ForegroundColor Cyan
    Write-Host "  2. El servidor MCP debería estar disponible" -ForegroundColor Cyan
    Write-Host ""
    Write-ColorOutput "Comandos disponibles:" "Bold"
    Write-Host "  • speak_text - Reproducir texto" -ForegroundColor Cyan
    Write-Host "  • list_voices - Listar voces" -ForegroundColor Cyan
    Write-Host "  • xtts_synthesize - Síntesis con emociones" -ForegroundColor Cyan
    Write-Host ""
    Write-ColorOutput "¡Disfruta TTS-Notify! 🎉" "Green"
}

# Run main
Main
