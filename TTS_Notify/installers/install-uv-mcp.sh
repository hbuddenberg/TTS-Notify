#!/bin/bash
# =============================================================================
# TTS-Notify Dynamic UV/UVX MCP Installer
# =============================================================================
# Auto-detects AI tools and generates MCP configurations
# Supports: Claude Desktop, Claude Code, OpenCode, Zed, Cursor, Continue
# =============================================================================

set -e

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_NAME="tts-notify"
MCP_ENTRY="tts_notify.main:sync_main"

# AI Tool detection paths (macOS/Linux compatible)
get_ai_tool_path() {
    local tool="$1"
    case "$tool" in
        "claude-desktop") echo "$HOME/Library/Application Support/Claude" ;;
        "claude-code") echo "$HOME/.claude" ;;
        "opencode") echo "$HOME/Library/Application Support/ai.opencode.desktop" ;;
        "zed") echo "$HOME/.zed" ;;
        "cursor") echo "$HOME/Library/Application Support/Cursor/User/globalStorage" ;;
        "continue") echo "$HOME/.continue" ;;
    esac
}

TOOLS=("claude-desktop" "claude-code" "opencode" "zed" "cursor" "continue")

# Detected tools
declare -a DETECTED_TOOLS

# Installation mode
INSTALL_MODE=""
VENV_PATH=""
USE_UVX=false

# UV/UVX availability flags
UV_AVAILABLE=false
UVX_AVAILABLE=false
UV_PATH=""
UVX_PATH=""
MCP_COMMAND=""
MCP_ARGS=""

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}🔔 TTS-Notify MCP Installer (UV/UVX)${NC}                         ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  ${YELLOW}Dynamic MCP Configuration Generator${NC}                         ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC}  $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

print_section() {
    echo ""
    echo -e "${BOLD}$1${NC}"
    echo "─────────────────────────────────────────"
}

check_command() {
    command -v "$1" &> /dev/null
}

get_python_path() {
    if [ -f "$PROJECT_DIR/venv312/bin/python" ]; then
        echo "$PROJECT_DIR/venv312/bin/python"
    elif [ -f "$PROJECT_DIR/venv313/bin/python" ]; then
        echo "$PROJECT_DIR/venv313/bin/python"
    elif [ -f "$PROJECT_DIR/venv/bin/python" ]; then
        echo "$PROJECT_DIR/venv/bin/python"
    elif [ -f "$PROJECT_DIR/test_venv/bin/python" ]; then
        echo "$PROJECT_DIR/test_venv/bin/python"
    else
        which python3 2>/dev/null || which python 2>/dev/null
    fi
}

# =============================================================================
# UV/UVX Detection
# =============================================================================

detect_uv() {
    print_section "🔍 Detectando UV/UVX"

    if check_command uv; then
        UV_PATH=$(which uv)
        UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
        UV_AVAILABLE=true
        print_status "UV encontrado: $UV_PATH ($UV_VERSION)"
        return 0
    elif check_command uvx; then
        UVX_PATH=$(which uvx)
        UVX_AVAILABLE=true
        print_status "UVX encontrado: $UVX_PATH"
        return 0
    else
        UV_AVAILABLE=false
        UVX_AVAILABLE=false
        print_warning "UV/UVX no encontrado"
        print_info "Instala UV: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi
}

get_mcp_command() {
    local python_path=$(get_python_path)

    if [ "$UV_AVAILABLE" = true ]; then
        MCP_COMMAND="$UV_PATH"
        MCP_ARGS='["run", "--python", "'"$python_path"'", "-m", "tts_notify", "--mode", "mcp"]'
        echo "uv"
    elif [ "$UVX_AVAILABLE" = true ]; then
        MCP_COMMAND="$UVX_PATH"
        MCP_ARGS='["tts-notify", "--mode", "mcp"]'
        echo "uvx"
    else
        MCP_COMMAND="$python_path"
        MCP_ARGS='["-m", "tts_notify", "--mode", "mcp"]'
        echo "python"
    fi
}

# =============================================================================
# AI Tool Detection
# =============================================================================

detect_ai_tools() {
    print_section "🔍 Detectando Herramientas de IA"

    for tool in "${TOOLS[@]}"; do
        path=$(get_ai_tool_path "$tool")
        if [ -d "$path" ]; then
            DETECTED_TOOLS+=("$tool|$path")
            print_status "${tool}: $path"
        else
            echo -e "  ${YELLOW}○${NC} ${tool}: no detectado"
        fi
    done

    if [ ${#DETECTED_TOOLS[@]} -eq 0 ]; then
        print_warning "No se detectaron herramientas de IA"
        return 1
    fi

    print_info "${#DETECTED_TOOLS[@]} herramienta(s) detectada(s)"
    return 0
}

# =============================================================================
# MCP Configuration Generators
# =============================================================================

generate_mcp_config_claude_desktop() {
    local config_dir="$1"
    local config_file="$config_dir/claude_desktop_config.json"
    
    print_info "Generando config para Claude Desktop..."
    
    # Backup existing config
    if [ -f "$config_file" ]; then
        cp "$config_file" "${config_file}.backup.$(date +%Y%m%d%H%M%S)"
        print_warning "Backup creado: ${config_file}.backup.*"
    fi
    
    # Generate config
    cat > "$config_file" << EOF
{
  "mcpServers": {
    "tts-notify": {
      "command": "$MCP_COMMAND",
      "args": $MCP_ARGS,
      "env": {
        "TTS_NOTIFY_ENGINE": "macos",
        "TTS_NOTIFY_VOICE": "Monica",
        "TTS_NOTIFY_RATE": "175"
      }
    }
  }
}
EOF

    print_status "Claude Desktop: $config_file"
}

generate_mcp_config_claude_code() {
    local config_dir="$1"
    local config_file="$config_dir/.mcp.json"

    print_info "Generando config para Claude Code..."

    # Backup existing config
    if [ -f "$config_file" ]; then
        cp "$config_file" "${config_file}.backup.$(date +%Y%m%d%H%M%S)"
    fi

    # Generate config (NOTE: No "mcpServers" wrapper for Claude Code!)
    cat > "$config_file" << EOF
{
  "tts-notify": {
    "command": "$MCP_COMMAND",
    "args": $MCP_ARGS,
    "env": {
      "TTS_NOTIFY_ENGINE": "macos",
      "TTS_NOTIFY_VOICE": "Monica",
      "TTS_NOTIFY_RATE": "175"
    }
  }
}
EOF

    print_status "Claude Code: $config_file"
}

generate_mcp_config_opencode() {
    local config_dir="$1"
    local config_file="$config_dir/mcp.json"

    print_info "Generando config para OpenCode..."

    # Ensure directory exists
    mkdir -p "$config_dir"

    # Backup existing config
    if [ -f "$config_file" ]; then
        cp "$config_file" "${config_file}.backup.$(date +%Y%m%d%H%M%S)"
    fi

    # Generate config (same format as Claude Code)
    cat > "$config_file" << EOF
{
  "tts-notify": {
    "command": "$MCP_COMMAND",
    "args": $MCP_ARGS,
    "env": {
      "TTS_NOTIFY_ENGINE": "macos",
      "TTS_NOTIFY_VOICE": "Monica"
    }
  }
}
EOF

    print_status "OpenCode: $config_file"
}

generate_mcp_config_zed() {
    local config_dir="$1"
    local config_file="$config_dir/settings.json"

    print_info "Generando config para Zed..."

    # Ensure directory exists
    mkdir -p "$config_dir"

    # Zed uses different format with "mcp_servers" key
    if [ -f "$config_file" ]; then
        cp "$config_file" "${config_file}.backup.$(date +%Y%m%d%H%M%S)"
        # TODO: Merge with existing settings
    fi

    cat > "$config_file" << EOF
{
  "mcp_servers": {
    "tts-notify": {
      "command": "$MCP_COMMAND",
      "args": $MCP_ARGS
    }
  }
}
EOF

    print_status "Zed: $config_file"
}

generate_mcp_config_cursor() {
    local config_dir="$1"
    local config_file="$config_dir/../mcp.json"

    print_info "Generando config para Cursor..."

    # Ensure directory exists
    mkdir -p "$(dirname "$config_file")"

    if [ -f "$config_file" ]; then
        cp "$config_file" "${config_file}.backup.$(date +%Y%m%d%H%M%S)"
    fi

    # Cursor uses same format as VS Code
    cat > "$config_file" << EOF
{
  "mcpServers": {
    "tts-notify": {
      "command": "$MCP_COMMAND",
      "args": $MCP_ARGS
    }
  }
}
EOF

    print_status "Cursor: $config_file"
}

generate_mcp_config_continue() {
    local config_dir="$1"
    local config_file="$config_dir/config.json"

    print_info "Generando config para Continue.dev..."

    # Ensure directory exists
    mkdir -p "$config_dir"

    if [ -f "$config_file" ]; then
        cp "$config_file" "${config_file}.backup.$(date +%Y%m%d%H%M%S)"
    fi

    # Continue.dev uses different format
    cat > "$config_file" << EOF
{
  "mcpServers": {
    "tts-notify": {
      "command": "$MCP_COMMAND",
      "args": $MCP_ARGS
    }
  }
}
EOF

    print_status "Continue: $config_file"
}

# =============================================================================
# Configuration Generation
# =============================================================================

generate_all_configs() {
    print_section "⚙️  Generando Configuraciones MCP"

    for tool_entry in "${DETECTED_TOOLS[@]}"; do
        tool="${tool_entry%%|*}"
        path="${tool_entry#*|}"

        case "$tool" in
            "claude-desktop")
                generate_mcp_config_claude_desktop "$path"
                ;;
            "claude-code")
                generate_mcp_config_claude_code "$path"
                ;;
            "opencode")
                generate_mcp_config_opencode "$path"
                ;;
            "zed")
                generate_mcp_config_zed "$path"
                ;;
            "cursor")
                generate_mcp_config_cursor "$path"
                ;;
            "continue")
                generate_mcp_config_continue "$path"
                ;;
        esac
    done
}

# =============================================================================
# Installation Functions
# =============================================================================

install_with_uv() {
    print_section "📦 Instalación con UV"
    
    cd "$PROJECT_DIR"
    
    # Create venv if needed
    if [ ! -d "venv312" ] && [ ! -d "venv" ]; then
        print_info "Creando entorno virtual..."
        uv venv venv
    fi
    
    # Install package
    if [ -d "venv" ]; then
        print_info "Instalando dependencias en venv..."
        uv pip install -e ".[fastmcp]" --python venv/bin/python
    elif [ -d "venv312" ]; then
        print_info "Instalando dependencias en venv312..."
        uv pip install -e ".[fastmcp]" --python venv312/bin/python
    fi
    
    print_status "Instalación completada"
}

install_with_uvx() {
    print_section "📦 Instalación con UVX"
    
    print_info "UVX no requiere instalación local"
    print_info "El servidor se ejecutará con: uvx tts-notify --mode mcp"
    
    USE_UVX=true
}

# =============================================================================
# Interactive Menu
# =============================================================================

show_menu() {
    print_header
    
    echo -e "${BOLD}Modo de instalación:${NC}"
    echo "  1) 🖥️  Local (venv) - Instalar en proyecto con UV"
    echo "  2) ☁️  UVX (sin venv) - Ejecutar globalmente"
    echo "  3) 🔍  Detectar automáticamente"
    echo "  4) ⚙️  Solo generar configs MCP"
    echo "  5) ❌ Salir"
    echo ""
    
    echo -e "${BOLD}Herramientas de IA detectadas:${NC}"
    for tool_entry in "${DETECTED_TOOLS[@]}"; do
        tool="${tool_entry%%|*}"
        echo -e "  ${GREEN}✓${NC} $tool"
    done
    echo ""
    
    read -p "Selecciona una opción [1-5]: " choice
    
    case $choice in
        1) INSTALL_MODE="local" ;;
        2) INSTALL_MODE="uvx" ;;
        3) INSTALL_MODE="auto" ;;
        4) INSTALL_MODE="config-only" ;;
        5) 
            echo "Saliendo..."
            exit 0
            ;;
        *)
            print_error "Opción inválida"
            exit 1
            ;;
    esac
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Detect tools first
    detect_ai_tools
    
    # Show menu
    show_menu
    
    # Check UV (always detect for proper command selection)
    if ! detect_uv; then
        print_warning "UV/UVX no disponible, usando python"
    fi
    
    # Execute installation
    case $INSTALL_MODE in
        "local")
            install_with_uv
            ;;
        "uvx")
            install_with_uvx
            ;;
        "auto")
            if detect_uv; then
                install_with_uv
            else
                print_error "No se puede instalar sin UV"
                exit 1
            fi
            ;;
        "config-only")
            print_info "Solo generando configuraciones..."
            ;;
    esac
    
    # Get MCP command
    get_mcp_command
    print_info "Using command: $MCP_COMMAND"

    # Generate MCP configs
    generate_all_configs
    
    # Summary
    print_section "✨ Instalación Completada"
    
    echo ""
    echo -e "${BOLD}Configuraciones generadas:${NC}"
    for tool_entry in "${DETECTED_TOOLS[@]}"; do
        tool="${tool_entry%%|*}"
        echo -e "  ${GREEN}✓${NC} $tool"
    done
    
    echo ""
    echo -e "${BOLD}Para usar TTS-Notify:${NC}"
    echo "  1. Reinicia tu herramienta de IA"
    echo "  2. El servidor MCP debería estar disponible"
    echo ""
    echo -e "${BOLD}Comandos disponibles:${NC}"
    echo "  • speak_text - Reproducir texto"
    echo "  • list_voices - Listar voces"
    echo "  • xtts_synthesize - Síntesis con emociones"
    echo ""
    echo -e "${GREEN}¡Disfruta TTS-Notify! 🎉${NC}"
}

# Run main
main "$@"
