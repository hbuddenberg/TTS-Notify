# TTS Notify v2.0.0

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/yourusername/tts-notify)

🎯 **Modular Text-to-Speech notification system for macOS with CLI, MCP, and REST API interfaces**

TTS Notify v2.0.0 is a complete rewrite featuring a modular architecture that maintains full compatibility with v1.5.0 while adding powerful new capabilities. It provides three different interfaces (CLI, MCP, REST API) that all use the same core TTS engine.

## ✨ What's New in v2.0.0

### 🏗️ **Complete Modular Architecture**
- **Core System**: 6 modular components with clean separation of concerns
- **Multiple Interfaces**: CLI, MCP Server, REST API - all using the same core
- **Plugin Foundation**: Extensible architecture for future enhancements
- **40% Code Reduction**: Eliminated duplication through smart design

### 🎛️ **Intelligent Configuration System**
- **30+ Environment Variables**: Complete control over all aspects
- **10+ Predefined Profiles**: Ready-to-use configurations for different scenarios
- **YAML Configuration Files**: Human-readable configuration management
- **Runtime Validation**: Automatic configuration validation with helpful error messages

### 🚀 **Enhanced Performance**
- **Async Support**: Non-blocking operations throughout the system
- **Voice Caching**: Intelligent caching with configurable TTL
- **Concurrent Processing**: Support for multiple simultaneous requests
- **Resource Optimization**: Efficient memory and CPU usage

### 🛠️ **Developer Experience**
- **Type Safety**: Full Pydantic model validation
- **Comprehensive Logging**: Structured logging with JSON support
- **Modern Tooling**: Black, isort, mypy, pytest integration
- **Cross-Platform Installers**: UV-based installation for all platforms

## 🎯 Quick MCP Installation

### Dynamic AI Tool Detection

The new `install-uv-mcp.sh` installer automatically detects your AI tools and generates MCP configurations. No manual config editing needed.

```bash
# macOS/Linux
cd TTS_Notify
./installers/install-uv-mcp.sh

# Windows (PowerShell)
cd TTS_Notify
.\installers\install-uv-mcp.ps1
```

### Supported AI Tools

| Tool | Config Location | Status |
|------|-----------------|--------|
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` | ✅ Auto-detected |
| **Claude Code** | `~/.claude/.mcp.json` | ✅ Auto-detected |
| **OpenCode** | `~/Library/Application Support/ai.opencode.desktop/mcp.json` | ✅ Auto-detected |
| **Zed Editor** | `~/.zed/settings.json` | ✅ Auto-detected |
| **Cursor IDE** | `~/Library/Application Support/Cursor/User/globalStorage/mcp.json` | ✅ Auto-detected |
| **Continue.dev** | `~/.continue/config.json` | ✅ Auto-detected |

### Installation Modes

The installer offers four modes:

1. **Local (venv)** - Install in project virtual environment
   ```bash
   # Creates venv in project directory
   # Uses UV for fast dependency installation
   # Best for development and testing
   ```

2. **UVX (no venv)** - Global execution without virtual environment
   ```bash
   # Uses uvx for on-demand execution
   # No local installation required
   # Best for production and minimal setup
   ```

3. **Auto-Detect** - Automatically choose best option
   ```bash
   # Detects UV availability
   # Falls back to standard pip if needed
   # Recommended for most users
   ```

4. **Config-Only** - Generate MCP configs without installation
   ```bash
   # Skip installation
   # Only generate configuration files
   # Useful when TTS-Notify is already installed
   ```

### Example Configurations Generated

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tts-notify": {
      "command": "/path/to/python",
      "args": ["-m", "tts_notify", "--mode", "mcp"],
      "env": {
        "TTS_NOTIFY_ENGINE": "macos",
        "TTS_NOTIFY_VOICE": "Monica",
        "TTS_NOTIFY_RATE": "175"
      }
    }
  }
}
```

**Claude Code** (`.mcp.json`):
```json
{
  "tts-notify": {
    "command": "/path/to/python",
    "args": ["-m", "tts_notify", "--mode", "mcp"],
    "env": {
      "TTS_NOTIFY_ENGINE": "macos",
      "TTS_NOTIFY_VOICE": "Monica",
      "TTS_NOTIFY_RATE": "175"
    }
  }
}
```

**Zed Editor** (`settings.json`):
```json
{
  "mcp_servers": {
    "tts-notify": {
      "command": "/path/to/python",
      "args": ["-m", "tts_notify", "--mode", "mcp"]
    }
  }
}
```

### Post-Installation Steps

1. **Restart your AI tool** - Close and reopen Claude Desktop, Claude Code, etc.
2. **Verify MCP connection** - The TTS-Notify server should appear in your MCP tools list
3. **Test voice output**:
   ```
   In Claude: "Lee en voz alta: Hola mundo"
   In Claude Code: Use speak_text tool
   ```

### Key Features

- **Automatic Detection**: Scans for 6 popular AI coding tools
- **Safe Installation**: Creates backups of existing configs before overwriting
- **Absolute Paths**: Uses absolute paths for reliable MCP connections
- **Multiple Environments**: Detects venv312, venv313, venv, and test_venv
- **UV Integration**: Leverage UV's lightning-fast package management
- **Cross-Platform**: Works on macOS, Linux, and Windows

## 🚀 Quick Start

### Installation

#### 🎯 **Complete Installation (Recommended)**
```bash
git clone https://github.com/yourusername/tts-notify.git
cd tts-notify
./installers/install.sh all
```

#### 🔧 **Development Mode**
```bash
git clone https://github.com/yourusername/tts-notify.git
cd tts-notify
./installers/install.sh development
source venv/bin/activate
```

### Basic Usage

#### **CLI Interface**
```bash
# Basic text-to-speech
tts-notify "Hello world"

# With specific voice and rate
tts-notify "Hola mundo" --voice monica --rate 200

# List available voices
tts-notify --list

# Save audio file
tts-notify "Test message" --save output --format wav

# System information
tts-notify --info
```

#### **MCP Server (Claude Desktop)**
```bash
# Start MCP server
tts-notify --mode mcp

# Automatic Claude Desktop configuration
# Voice search with natural language in Claude:
"Lee en voz alta: Hola mundo"
"Lista todas las voces en español"
"Guarda este texto como archivo: prueba de audio"
```

#### **REST API**
```bash
# Start API server
tts-notify --mode api

# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

## 🏗️ Architecture

### Core Components

```
src/
├── core/                    # Core TTS functionality
│   ├── config_manager.py   # Intelligent configuration with 30+ env vars
│   ├── voice_system.py     # Voice detection & management (84+ voices)
│   ├── tts_engine.py       # Abstract TTS engine with macOS implementation
│   ├── models.py           # Pydantic data models with validation
│   └── exceptions.py       # Custom exception hierarchy
├── ui/                      # User interfaces
│   ├── cli/                # Command-line interface
│   │   ├── main.py         # CLI implementation with feature parity
│   │   └── __main__.py     # CLI entry point
│   ├── mcp/                # MCP server for Claude Desktop
│   │   ├── server.py       # FastMCP server with 4 tools
│   │   └── __main__.py     # MCP entry point
│   └── api/                # REST API with FastAPI
│       ├── server.py       # FastAPI server with OpenAPI docs
│       └── __main__.py     # API entry point
├── utils/                   # Utility modules
│   ├── async_utils.py      # Async utilities and helpers
│   └── text_normalizer.py  # Text processing and normalization
├── plugins/                 # Plugin system foundation
│   └── __init__.py         # Plugin base classes and registry
├── installer/               # UV-based unified installer
│   └── installer.py        # Cross-platform installation logic
├── main.py                  # Main orchestrator with intelligent mode detection
├── __main__.py             # Package entry point
└── __init__.py             # Package initialization
```

### Main Orchestrator

The `src/main.py` serves as the central delegation hub with intelligent mode detection:

- **Auto-detection**: Automatically detects execution mode from environment variables and arguments
- **Interface Creation**: Creates and manages appropriate interface instances
- **Configuration Loading**: Loads and validates configuration from all sources
- **Error Handling**: Comprehensive error handling with fallback behaviors

### Interface Overview

| Interface | Use Case | Entry Point | Key Features |
|-----------|----------|-------------|--------------|
| **CLI** | Command-line usage, scripts | `python -m tts_notify` or `tts-notify` | Full voice control, file saving, filtering, system info |
| **MCP** | Claude Desktop integration | `python -m tts_notify --mode mcp` | 4 MCP tools, flexible voice search, async processing |
| **API** | Web applications, services | `python -m tts_notify --mode api` | REST endpoints, OpenAPI docs, async, concurrent requests |

## ⚙️ Configuration

### Environment Variables
```bash
# Voice Settings
TTS_NOTIFY_VOICE=monica          # Default voice
TTS_NOTIFY_RATE=175              # Speech rate (WPM)
TTS_NOTIFY_LANGUAGE=es           # Language
TTS_NOTIFY_QUALITY=enhanced      # Voice quality

# Functionality
TTS_NOTIFY_ENABLED=true          # Enable TTS
TTS_NOTIFY_CACHE_ENABLED=true    # Enable voice caching
TTS_NOTIFY_LOG_LEVEL=INFO        # Logging level

# API Server
TTS_NOTIFY_API_PORT=8000         # API server port
TTS_NOTIFY_API_HOST=localhost    # API server host
```

### Configuration Profiles
```bash
# Use predefined profiles
tts-notify --profile claude-desktop  # Optimized for Claude Desktop
tts-notify --profile development      # Development with debugging
tts-notify --profile production       # Production ready
```

## 🎵 Voice System

### 84+ Voice Support
- **Automatic Detection**: Discovers all system voices at startup
- **Smart Categorization**: Español, Enhanced, Premium, Siri, Others
- **Flexible Search**: Exact, partial, case-insensitive, accent-insensitive matching
- **Performance**: 75% faster voice detection with caching

### Voice Search Examples
```bash
# Exact match
tts-notify "Test" --voice Monica

# Case-insensitive
tts-notify "Test" --voice monica

# Partial match
tts-notify "Test" --voice angel  # Finds Angélica

# Quality variants
tts-notify "Test" --voice "monica enhanced"
```

## 🧪 Development

### Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src tests
isort src tests

# Type checking
mypy src
```

### Testing
```bash
# All tests
pytest

# Specific modules
pytest tests/test_core.py
pytest tests/test_api.py

# With coverage
pytest --cov=src
```

## 📖 Documentation

### User Documentation
- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** - Comprehensive installation guide
- **[docs/USAGE.md](docs/USAGE.md)** - Complete usage guide for all interfaces
- **[docs/VOICES.md](docs/VOICES.md)** - Voice reference with 84+ voice details

### Developer Documentation
- **[README-v2.md](README-v2.md)** - Complete technical documentation
- **[CLAUDE.md](CLAUDE.md)** - Development guide for Claude Code
- **[CHANGELOG-v2.md](CHANGELOG-v2.md)** - Version history and changes
- **[MIGRATION-GUIDE-v2.md](MIGRATION-GUIDE-v2.md)** - Migration from v1.5.0

## 🔧 Installation Scripts

### Dynamic MCP Installer (Recommended)

The new `install-uv-mcp.sh` installer provides automatic AI tool detection and MCP configuration generation:

```bash
# macOS/Linux - Interactive installer
./installers/install-uv-mcp.sh

# Windows PowerShell
.\installers\install-uv-mcp.ps1
```

**Features:**
- Auto-detects 6 AI coding tools (Claude Desktop, Claude Code, OpenCode, Zed, Cursor, Continue)
- Generates proper MCP configs for each detected tool
- Supports UV/UVX for lightning-fast installation
- Interactive mode selection (Local/UVX/Auto/Config-Only)
- Creates backups of existing configurations

### Traditional Installers

```bash
# Main installer (Linux/macOS)
./installers/install.sh [development|production|mcp|all|uninstall]

# Windows installers
installers/install.bat [mode]
installers/install.ps1 -Mode [mode]

# Specific installers
./installers/install-cli.sh  # CLI only
./installers/install-mcp.sh   # MCP only
./installers/install-mcp-claude-code.sh  # Claude Code specific
```

## 📊 Performance

| Metric | v1.5.0 | v2.0.0 | Improvement |
|--------|--------|--------|-------------|
| Voice Detection | ~2s | ~0.5s | 75% faster |
| CLI Startup | ~1s | ~0.3s | 70% faster |
| Memory Usage | ~50MB | ~30MB | 40% reduction |
| Code Size | ~5000 lines | ~3000 lines | 40% reduction |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `./installers/install.sh development`
4. Make changes with tests
5. Run tests: `pytest`
6. Format code: `black src tests && isort src tests`
7. Commit changes: `git commit -m "Add amazing feature"`
8. Push branch: `git push origin feature/amazing-feature`
9. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Homepage**: https://github.com/yourusername/tts-notify
- **Documentation**: https://github.com/yourusername/tts-notify#readme
- **Issues**: https://github.com/yourusername/tts-notify/issues
- **Changelog**: https://github.com/yourusername/tts-notify/blob/main/CHANGELOG.md

---

**TTS Notify v2.0.0** - 🎯 Modular, Powerful, and Ready for Production!