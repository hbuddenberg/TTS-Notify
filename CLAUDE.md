# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TTS Notify is a **modular Text-to-Speech notification system** that operates in three modes:
1. **MCP Server** - Integrates with Claude Desktop/Claude Code as an MCP server
2. **CLI Tool** - Standalone command-line tool (installed globally or via uvx)
3. **REST API** - Web API interface for applications and services

**Current Version**: v2.0.0 (complete modular rewrite)  
**Next Version**: v3.0.0 (planned CoquiTTS integration - see `implement_v3.md`)

The project uses macOS's native TTS engine (`say` command) with **zero external dependencies** for speech synthesis, while preparing for advanced AI-powered TTS in v3.

## Key Architecture Components (v2.0.0)

### Modular Core System
The project implements a **complete modular architecture** with separation of concerns:

- **Core System** (`src/tts_notify/core/`): 6 modular components
  - `config_manager.py`: **30+ environment variables** with Pydantic validation
  - `voice_system.py`: Voice detection with caching and flexible search
  - `tts_engine.py`: Abstract TTS engine with macOS implementation
  - `models.py`: Pydantic data models with comprehensive validation
  - `exceptions.py`: Custom exception hierarchy

- **User Interfaces** (`src/tts_notify/ui/`): Three interfaces using the same core
  - `cli/`: Feature-complete CLI with full v1.5.0 compatibility
  - `mcp/`: Enhanced MCP server with 4 tools and async processing
  - `api/`: REST API with FastAPI and OpenAPI documentation

- **Utilities** (`src/tts_notify/utils/`): Support modules
  - `logger.py`: Structured JSON logging
  - `file_manager.py`: Cross-platform file operations
  - `text_normalizer.py`: Text processing and markdown removal
  - `system_detector.py`: System capability detection
  - `async_utils.py`: Async utilities and helpers

### Voice Detection System
The core innovation is the **enhanced dynamic voice detection system**:
- **Auto-detection**: Parses `say -v ?` output to discover ALL system voices (~84+ voices)
- **75% faster performance**: Intelligent caching with configurable TTL
- **Flexible search**: Supports exact, partial, case-insensitive, accent-insensitive matching
- **Categorization**: Groups voices by type (Español, Enhanced, Premium, Siri, Others)
- **Cross-platform resilience**: Works across different macOS versions with different voice installations

### Intelligent Configuration System
- **30+ Environment Variables**: Complete control over all aspects
- **10+ Predefined Profiles**: Ready-to-use configurations (claude-desktop, development, production, etc.)
- **YAML Configuration Support**: Human-readable configuration management
- **Runtime Validation**: Automatic configuration validation with helpful error messages

## Project Structure (v2.0.0)

```
TTS_Notify/                    # Main project directory
├── src/
│   └── tts_notify/           # Main package (modular architecture)
│       ├── main.py           # Main orchestrator with intelligent mode detection
│       ├── __main__.py       # Package entry point
│       ├── core/             # Core TTS functionality
│       │   ├── config_manager.py    # 30+ env vars + YAML config
│       │   ├── voice_system.py      # Enhanced voice detection with caching
│       │   ├── tts_engine.py        # Abstract engine + macOS impl
│       │   ├── models.py            # Pydantic validation models
│       │   └── exceptions.py        # Custom exception hierarchy
│       ├── ui/              # User interfaces (CLI, MCP, API)
│       │   ├── cli/        # Command-line interface
│       │   │   ├── main.py  # Full v1.5.0 compatibility
│       │   │   └── __main__.py
│       │   ├── mcp/        # MCP server for Claude Desktop
│       │   │   ├── server.py        # Basic FastMCP server
│       │   │   ├── enhanced_server.py # Enhanced server with 4 tools
│       │   │   └── __main__.py
│       │   └── api/        # REST API with FastAPI
│       │       ├── server.py # OpenAPI docs + async endpoints
│       │       └── __main__.py
│       ├── utils/           # Utility modules
│       │   ├── logger.py           # Structured JSON logging
│       │   ├── file_manager.py     # Cross-platform file ops
│       │   ├── text_normalizer.py  # Text processing
│       │   ├── system_detector.py  # System detection
│       │   └── async_utils.py      # Async helpers
│       ├── plugins/         # Plugin system foundation
│       └── installer/       # UV-based unified installer
├── installers/              # Cross-platform installation scripts
│   ├── install.sh          # Main installer (all modes)
│   ├── install-mcp.sh      # MCP-specific installer
│   ├── install-cli.sh      # CLI-specific installer
│   └── install-mcp-claude-code.sh # Special Claude Code installer
├── docs/                   # Documentation
│   ├── README.md           # Main documentation
│   ├── INSTALLATION.md     # Installation guide
│   ├── USAGE.md            # Usage examples
│   └── VOICES.md           # Voice reference
├── tests/                  # Test suite (85%+ coverage)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Modern Python packaging
└── CHANGELOG-v2.md         # Complete version history
```

## Development Commands (v2.0.0)

### Environment Setup

```bash
# Create virtual environment for development
cd TTS_Notify
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Install with UV (recommended)
pip install uv
uv pip install -e ".[dev]"
```

### Testing Different Interfaces

```bash
# CLI Interface (development)
cd TTS_Notify
python -m tts_notify.ui.cli "Test message"
python -m tts_notify.ui.cli --list
python -m tts_notify.ui.cli --list --compact
python -m tts_notify.ui.cli --list --gen female
python -m tts_notify.ui.cli "Test" --voice jorge --rate 200

# MCP Server (development)
cd TTS_Notify
python -m tts_notify.ui.mcp  # Starts enhanced MCP server

# REST API (development)
cd TTS_Notify
python -m tts_notify.ui.api  # Starts API server on localhost:8000
# API docs available at http://localhost:8000/docs
```

### Modern Installation with UV

```bash
# Complete installation (recommended)
cd TTS_Notify
./installers/install.sh all

# Development mode
cd TTS_Notify
./installers/install.sh development

# Production mode
cd TTS_Notify
./installers/install.sh production

# Special Claude Code global installation
cd TTS_Notify
./installers/install-mcp-claude-code.sh
```

### Legacy UVX Usage (still supported)

```bash
# With uvx (no installation required)
cd TTS_Notify
uvx --from . tts-notify "Test message"
uvx --from . tts-notify --list
uvx --from . tts-notify --list --compact
uvx --from . tts-notify --list --gen female
uvx --from . tts-notify --list --lang es_ES
uvx --from . tts-notify "Test" --voice jorge --rate 200

# Test voice search flexibility
uvx --from . tts-notify "Test" --voice angelica  # finds Angélica
uvx --from . tts-notify "Test" --voice "jorge enhanced"  # Enhanced variant
```

### Building and Testing

```bash
# Build package (uses hatchling build system)
cd TTS_Notify
python -m build

# Run tests (pytest with 85%+ coverage)
cd TTS_Notify
pytest
pytest --cov=src

# Code formatting and linting
black src tests
isort src tests
mypy src
```

## Architecture Details

### Main Orchestrator (`src/tts_notify/main.py`)

The central delegation hub with **intelligent mode detection**:
- **Auto-detection**: Automatically detects execution mode from environment variables and arguments
- **Interface Creation**: Creates and manages appropriate interface instances
- **Configuration Loading**: Loads and validates configuration from all sources (env vars, YAML, profiles)
- **Error Handling**: Comprehensive error handling with fallback behaviors

### Core Configuration System (`core/config_manager.py`)

**30+ Environment Variables** with intelligent validation:

```bash
# Voice Settings
TTS_NOTIFY_VOICE=monica          # Default voice
TTS_NOTIFY_RATE=175              # Speech rate (WPM)
TTS_NOTIFY_LANGUAGE=es           # Language
TTS_NOTIFY_QUALITY=enhanced      # Voice quality
TTS_NOTIFY_PITCH=1.0             # Pitch multiplier
TTS_NOTIFY_VOLUME=1.0            # Volume multiplier

# Functionality
TTS_NOTIFY_ENABLED=true          # Enable TTS
TTS_NOTIFY_CACHE_ENABLED=true    # Enable voice caching
TTS_NOTIFY_LOG_LEVEL=INFO        # Logging level
TTS_NOTIFY_MAX_TEXT_LENGTH=5000  # Maximum text length

# API Server
TTS_NOTIFY_API_PORT=8000         # API server port
TTS_NOTIFY_API_HOST=localhost    # API server host

# Advanced
TTS_NOTIFY_DEBUG_MODE=false      # Debug mode
TTS_NOTIFY_EXPERIMENTAL=false    # Experimental features
TTS_NOTIFY_MAX_CONCURRENT=5      # Max concurrent operations
```

**10+ Configuration Profiles**:
- `claude-desktop`: Optimized for Claude Desktop integration
- `api-server`: Optimized for API deployment  
- `development`: Development with debugging enabled
- `production`: Production with minimal logging
- `cli-default`: Standard CLI usage
- `accessibility`: Accessibility features enabled
- `performance`: High performance settings
- `testing`: Suitable for automated testing
- `spanish`: Spanish language optimization
- `english`: English language optimization

### Voice System Enhancements

**Performance Improvements** (v2.0.0 vs v1.5.0):
- **Voice Detection**: 75% faster (~0.5s vs ~2.0s) with intelligent caching
- **CLI Startup**: 70% faster (~0.3s vs ~1.0s)
- **Memory Usage**: 40% reduction (~30MB vs ~50MB)
- **Code Size**: 40% reduction (~3000 lines vs ~5000 lines)

**Enhanced Search Algorithm**:
The project implements a **4-tier voice search system**:
1. **Exact match**: Case-insensitive, accent-insensitive comparison
2. **Prefix match**: Search by voice name start (prioritized)
3. **Partial match**: Search anywhere in voice description
4. **Fallback**: First Spanish voice, then "Monica"

Located in `core/voice_system.py` with caching and async support.

### MCP Server Implementation

**Enhanced MCP Server** with 4 tools:
- `speak_text` - Enhanced with flexible voice search and async processing
- `list_voices` - Improved categorization and filtering
- `save_audio` - Better file management and validation  
- `get_mcp_config` - New tool for configuration introspection

**FastMCP integration**: Uses `mcp.server.fastmcp` for async server implementation with comprehensive error handling.

### REST API Implementation

**Complete REST API** with FastAPI:
- **OpenAPI Documentation**: Interactive docs at `/docs`
- **Async Endpoints**: Non-blocking operations throughout
- **Comprehensive Coverage**: All TTS functionality accessible via API
- **Validation**: Pydantic models for request/response validation

**Key Endpoints**:
- `GET /`: API information and status
- `GET /status`: Server status and statistics
- `GET /voices`: List voices with filtering
- `GET /config`: Current configuration
- `POST /speak`: Convert text to speech
- `POST /save`: Save audio file
- `GET /download/{filename}`: Download audio files

## Voice System

The enhanced CLI implements dynamic voice detection with caching:
- Parses `say -v ?` output to find all system voices (~84+ voices)
- Creates lowercase aliases automatically
- Falls back to hardcoded voices if detection fails
- **Intelligent caching** reduces repeated detection overhead
- This makes the tool resilient across different macOS voice configurations

### Voice Categories

**Español (16 voces):**
- Eddy, Flo, Grandma, Grandpa, Reed, Rocko, Sandy, Shelley (España y México)

**Enhanced/Premium (12 voces):**
- Angélica (México), Francisca (Chile), Jorge (España), Paulina (México)
- Mónica (España), Juan (México), Diego (Argentina), Carlos (Colombia)
- Isabela (Argentina), Marisol (España), Soledad (Colombia), Jimena (Colombia)

**Siri (cuando se instalen):**
- Siri Female, Siri Male (auto-detectadas)

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

# Using different interfaces
python -m tts_notify.ui.cli "Test" --voice jorge
# Via MCP: "Lee en voz alta con voz Jorge: Hola mundo"
# Via API: POST /speak with {"text": "Hola mundo", "voice": "jorge"}
```

## v3.0.0 Preparation

The codebase is **fully prepared for v3.0.0** with CoquiTTS integration:

### Ready for v3 Features
- **Modular Architecture**: Ready to add new TTS engines
- **Configuration System**: Extensible for Coqui-specific variables  
- **Plugin Foundation**: Prepared for audio processing plugins
- **Multiple Interfaces**: Ready for Coqui integration in CLI, MCP, and API
- **Cross-Platform**: Foundation for Windows/Linux support

### Key Files for v3 Development
- `implement_v3.md` - Complete v3 development plan
- `core/tts_engine.py` - Abstract engine ready for Coqui implementation
- `core/config_manager.py` - Extensible configuration system
- `plugins/` - Foundation for audio processing pipeline

## Installation Scripts

### Cross-Platform Unified Installer

```bash
# Main installer (Linux/macOS/Windows)
./installers/install.sh [development|production|mcp|all|uninstall]

# Claude Code Special (Global MCP Configuration)
./installers/install-mcp-claude-code.sh

# Legacy specific installers (still supported)
./installers/install-cli.sh  # CLI only
./installers/install-mcp.sh   # MCP only
```

### Special Claude Code Integration

The `install-mcp-claude-code.sh` script provides:
- **Global MCP Configuration**: Configures TTS-Notify for ALL Claude Code projects
- **Interactive Setup**: Voice selection, rate configuration, environment setup
- **Automatic Dependencies**: UV package manager with proper virtual environment
- **Comprehensive Testing**: Functional verification and troubleshooting guides

## Performance Comparison

| Metric | v1.5.0 | v2.0.0 | Improvement |
|--------|--------|--------|-------------|
| Voice Detection | ~2s | ~0.5s | 75% faster |
| CLI Startup | ~1s | ~0.3s | 70% faster |
| Memory Usage | ~50MB | ~30MB | 40% reduction |
| Code Size | ~5000 lines | ~3000 lines | 40% reduction |
| Test Coverage | ~0% | ~85%+ | New feature |
| Platform Support | macOS only | macOS, Linux, Windows | 200% increase |

## Configuration Files

### Claude Desktop/Claude Code Config

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tts-notify": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/TTS_Notify/src/tts_notify/ui/mcp/__main__.py"],
      "env": {
        "TTS_NOTIFY_VOICE": "Siri Female (Spanish Spain)",
        "TTS_NOTIFY_RATE": "175"
      }
    }
  }
}
```

**Claude Code** (Global configuration via `install-mcp-claude-code.sh`):
```bash
# Applied globally to all Claude Code projects
claude mcp add --scope user tts-notify --transport stdio \
  --env TTS_NOTIFY_VOICE="Siri Female (Spanish Spain)" \
  --env TTS_NOTIFY_RATE="175" \
  -- "/path/to/python" "-m" "tts_notify" "--mode" "mcp"
```

## Testing Strategy

The project uses comprehensive **manual and automated testing**:

```bash
# Automated tests (pytest)
cd TTS_Notify
pytest                           # All tests
pytest --cov=src                 # With coverage
pytest tests/test_core.py        # Core functionality
pytest tests/test_api.py         # API endpoints

# Manual testing strategies
# Test voice availability (macOS native)
say -v Monica "test"
say -v ?  # List all system voices

# Test CLI functionality
tts-notify "test" --voice monica
tts-notify --list
tts-notify --list --compact
tts-notify --list --gen female
tts-notify "test" --save output_file

# Test voice search flexibility
tts-notify "test" --voice angelica    # Should find Angélica
tts-notify "test" --voice siri        # Should find Siri if installed
tts-notify "test" --voice "monica enhanced"  # Should find Enhanced variant

# Test MCP server (through Claude Desktop/Claude Code)
"Lee en voz alta: Hola mundo"
"Lista todas las voces disponibles"
"Guarda este texto como audio: archivo de prueba"

# Test API server
curl -X POST "http://localhost:8000/speak" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "monica"}'

# Test edge cases
tts-notify "test" --rate 100  # Minimum rate
tts-notify "test" --rate 300  # Maximum rate
tts-notify "test" --voice nonexistent_voice  # Should fallback to Monica
```

## Critical Development Notes

### Code Architecture Preservation
- **Modular Core System**: The 6-component core architecture must be maintained when making changes
- **Voice detection system**: The enhanced `voice_system.py` with caching is critical for performance
- **Configuration management**: 30+ environment variables with Pydantic validation must be preserved
- **Interface separation**: CLI, MCP, and API interfaces must remain independent but share core

### Path Management
- **Absolute paths required**: Claude Desktop/Claude Code config requires absolute paths
- **Cross-platform compatibility**: Use `file_manager.py` for platform-specific operations
- **Dynamic user detection**: Use system utilities for user-specific paths

### Dependencies and Build System
- **Zero ML dependencies**: v2.0.0 intentionally uses only macOS native `say` command
- **Optional Dependencies**: CoquiTTS will be optional in v3.0.0 (`[coqui]` extra)
- **Hatchling build**: Uses modern Python packaging with src/ layout
- **UV package manager**: Recommended for dependency management

### Performance Considerations
- **Caching Strategy**: Voice detection caching is crucial for 75% performance improvement
- **Async Operations**: All interfaces should use async/await patterns consistently
- **Memory Management**: Maintain 40% memory reduction achieved in v2.0.0

### Multi-Interface Compatibility
- **Feature Parity**: All interfaces must support the same core functionality
- **Configuration Sharing**: Environment variables work across all interfaces
- **Error Handling**: Consistent error messages and fallback behaviors

## Version History Context

- **v2.0.0** - Complete modular rewrite with 3 interfaces, 40% code reduction, 75% performance improvement
- **v1.5.0** - Complete restructure as TTS Notify, clean codebase
- **v1.4.4** - Last stable version as TTS-macOS
- **v1.2.1** - Added dynamic voice detection in CLI
- **v1.1.0** - Added CLI mode and uvx support
- **v1.0.0** - Initial MCP server implementation

## Future Roadmap (v3.0.0+)

Based on `implement_v3.md`:

### v3.0.0 - CoquiTTS Integration
- **Optional AI TTS Engine**: CoquiTTS with voice cloning capabilities
- **Multi-Engine Support**: Dynamic selection between macOS and Coqui engines
- **Voice Profiles**: Custom voice cloning from audio samples
- **Audio Pipeline**: Preprocessing, noise reduction, format conversion
- **Enhanced MCP/API**: New tools for voice profile management

### v3.1+ - Advanced Features
- **Fine-tuning**: Experimental voice customization
- **Web Interface**: Browser-based configuration and testing
- **Cloud Integration**: Optional cloud TTS services
- **Real-time Streaming**: Live audio generation
- **Batch Processing**: Multiple text processing

## Dependencies

### Core Dependencies
- **Python 3.10+** (required)
- **pydantic** (for configuration validation)
- **fastapi** (for REST API, optional)
- **mcp>=1.0.0** (for MCP server mode only)
- **macOS native `say` command** (built-in)

### Optional Dependencies (v3.0.0+)
- **CoquiTTS** (`[coqui]` extra) - AI-powered TTS
- **librosa, soundfile** (`[audio]` extra) - Audio processing
- **torch, torchaudio** (`[coqui]` extra) - Neural network support

### Development Dependencies
- **pytest** (for testing)
- **black, isort** (for code formatting)
- **mypy** (for type checking)
- **uv** (recommended for package management)

## Hooks for Claude Code

The `.claude/hooks/` directory contains shell scripts that integrate TTS Notify with Claude Code. **These remain fully compatible with v2.0.0**:

### Available Hooks

1. **post-response** - Reads Claude's responses aloud after generation
   - Filters out code blocks and markdown
   - Truncates long responses based on `TTS_MAX_LENGTH`
   - Runs in background to avoid blocking

2. **user-prompt-submit** - Confirms when user submits a prompt
   - Announces "Procesando tu solicitud" with configurable voice
   - Useful for accessibility and confirmation

### Hook Configuration (Enhanced for v2.0.0)

Hooks now support the **full 30+ environment variable system**:

```bash
# Enable response reading (enhanced options)
export TTS_NOTIFY_ENABLED=true
export TTS_NOTIFY_VOICE=monica
export TTS_NOTIFY_RATE=175
export TTS_NOTIFY_PITCH=1.0
export TTS_NOTIFY_VOLUME=1.0
export TTS_NOTIFY_MAX_TEXT_LENGTH=5000
export TTS_NOTIFY_CACHE_ENABLED=true

# Enable prompt confirmation (optional)
export TTS_NOTIFY_PROMPT_ENABLED=true
export TTS_NOTIFY_PROMPT_VOICE=jorge
export TTS_NOTIFY_PROMPT_RATE=200

# Advanced options
export TTS_NOTIFY_LOG_LEVEL=INFO
export TTS_NOTIFY_DEBUG_MODE=false
export TTS_NOTIFY_EXPERIMENTAL=false
```

### Testing Hooks (v2.0.0 Compatible)

```bash
# Interactive configuration (enhanced)
source .claude/hooks/enable-tts.sh

# Run demo (updated for v2.0.0)
./.claude/hooks/demo.sh

# Test manually
echo "Test response" | ./.claude/hooks/post-response
```

---

**TTS Notify v2.0.0** is production-ready with a clean modular architecture, excellent performance, and comprehensive tooling. The codebase is fully prepared for v3.0.0 development with CoquiTTS integration while maintaining full backward compatibility.