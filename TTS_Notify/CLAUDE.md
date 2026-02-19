# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TTS Notify is a modular Text-to-Speech notification system that operates in three modes:
1. **MCP Server** - Integrates with Claude Desktop/Claude Code as an MCP server
2. **CLI Tool** - Standalone command-line tool (installed globally or via uvx)
3. **REST API** - Web API interface for applications and services

**Current Version**: v3.0.0 (complete CoquiTTS integration - **80% functional**)  
**Previous Version**: v2.0.0 (modular rewrite with 3 interfaces - 100% functional)

The project implements a **dual-engine architecture** with macOS native TTS and AI-powered CoquiTTS engine, maintaining 100% backward compatibility while adding voice cloning, multi-language support, and advanced audio processing.

## Key Architecture Concepts

### Dual-Engine System (v3.0.0 Breakthrough)
The `TTSEngineRegistry` in `src/core/tts_engine.py` manages runtime engine selection:
- **macOS Engine**: Native `say` command for guaranteed compatibility
- **CoquiTTS Engine**: AI-powered TTS with 17 languages and voice cloning
- **Intelligent Switching**: Automatic engine selection based on configuration and availability

### Optional Dependency System
```toml
# Core functionality (always available)
dependencies = ["pydantic>=2.0.0", "pyyaml>=6.0"]

# Optional extras for progressive enhancement
coqui = ["coqui-tts>=0.27.0", "torchaudio>=2.0.0"]
coqui-gpu = ["coqui-tts[gpu]>=0.27.0"]
coqui-cloning = ["librosa>=0.10.0", "scipy>=1.10.0", "numpy>=1.24.0"]
audio-pipeline = ["ffmpeg-python>=0.2.0", "pydub>=0.25.0"]
```

### Phase-Based Implementation (Complete)
**Phase A**: Multi-Language CoquiTTS Engine ✅
- 17 languages: en, es, fr, de, it, pt, nl, pl, ru, zh, ja, ko, cs, ar, tr, hu, fi
- Intelligent model management with auto-download, caching, and fallback
- Language detection and forced language options

**Phase B**: Voice Cloning System ✅
- 4 quality levels: low, medium, high, ultra
- Multi-language voice cloning with optimization scoring
- Voice profile management with persistent storage

**Phase C**: Advanced Audio Pipeline ✅
- 8-stage processing: language optimization, noise reduction, format conversion
- 6 format support: WAV, AIFF, MP3, OGG, FLAC, M4A
- Real-time streaming with low-latency processing

### Modular Core System
The core functionality is split across `src/core/`:
- **config_manager.py**: 60+ environment variables with optional CoquiTTS extensions
- **coqui_engine.py**: Complete CoquiTTS implementation with voice cloning
- **audio_pipeline.py**: Advanced audio processing with language optimization
- **voice_system.py**: Enhanced macOS voice detection with caching (maintained)
- **tts_engine.py**: Abstract engine with dual-engine registry
- **models.py**: Extended Pydantic models with voice cloning support
- **installer.py**: Automated CoquiTTS installation and validation

### Interface Layer (Feature Parity)
All user interfaces (`src/ui/`) use the same enhanced core:
- **cli/**: Extended CLI with v3.0.0 flags and voice cloning commands
- **mcp/**: Enhanced MCP server with 6 tools for voice and model management
- **api/**: REST API with extended endpoints for CoquiTTS features

### Configuration Hierarchy
Configuration follows strict precedence with v3.0.0 extensions:
- Environment variables → YAML files → Pydantic defaults → Profile definitions
- **Engine Selection**: `TTS_NOTIFY_ENGINE=macos|coqui` (auto-detect if not set)
- **Backward Compatibility**: All v2.0.0 variables remain functional

## Development Commands

### Installation and Setup
```bash
# Complete v3.0.0 installation (recommended)
./installers/install.sh all

# Core functionality (macOS native always available)
pip install ".[dev]"                    # v2.0.0 development

# CoquiTTS installation (required for v3.0.0 features)
./installers/install.sh coqui              # Auto-installation with validation
tts-notify --install-coqui              # Installation interactiva

# Check installation status
tts-notify --installation-status          # Muestra estado actual
```

### CLI Commands (Current State)

#### Basic Commands (100% Working - macOS Native)
```bash
tts-notify "Hello world"                    # Basic TTS - macOS native
tts-notify --list                             # List voices - 84 voices detected
tts-notify --list --compact                   # Compact voice list
tts-notify --list --gen female                  # Filter by gender
tts-notify --voice "Monica"                   # Voice selection - flexible matching
tts-notify --save output.txt                    # Save audio file
tts-notify --rate 200                          # Speech rate control
```

#### v3.0.0 Commands (Architecture Complete - Require CoquiTTS Installation)
```bash
# Engine Selection
tts-notify --engine {macos,coqui} "Texto"     # Switch between engines
tts-notify --engine coqui --list-languages       # Show 17 available languages
tts-notify --engine coqui --download-language es  # Download Spanish models
tts-notify --engine coqui --language es "Texto"  # Force Spanish language

# XTTS v2 Emotion Support (FASE 3 - NEW)
tts-notify "Great news!" --xtts --emotion happy
tts-notify "Urgent alert" --xtts --emotion urgent
tts-notify "Calm message" --xtts --emotion calm --temperature 0.3
tts-notify "Sad story" --engine coqui --emotion sad --save output.wav

# Voice Cloning (Phase B)
tts-notify --clone-voice sample.wav --clone-language es --clone-quality ultra
tts-notify --engine coqui --voice "ClonedVoice" "Test"
tts-notify --list-cloned                        # List cloned voices

# Audio Processing (Phase C)
tts-notify --process-audio input.wav --output-format mp3
tts-notify --process-audio input.wav --output-format flac --audio-quality high
tts-notify --pipeline-status                     # Check audio pipeline status
```

### FastMCP Server (v3.1.0 - NEW)

TTS Notify now uses FastMCP for improved Claude Desktop integration:

```bash
# Start FastMCP server (default)
tts-notify --mode mcp

# FastMCP provides 8 tools:
# - speak_text: Synthesize speech
# - list_voices: List available voices
# - save_audio: Save audio to file
# - get_config: Get current configuration
# - xtts_synthesize: XTTS v2 synthesis with emotion support
# - clone_voice: Clone voice from audio sample
# - list_cloned_voices: List cloned voices
# - get_xtts_status: Get XTTS engine status

# FastMCP Resources:
# - voices://list - Voice list resource
# - config://current - Current configuration resource
```

### MCP Tools Reference (FastMCP)

| Tool | Description | Parameters |
|------|-------------|------------|
| `speak_text` | Speak text aloud | text, voice, rate, pitch, volume, engine, language |
| `list_voices` | List available voices | engine, language, gender |
| `save_audio` | Save audio to file | text, filename, voice, format, engine |
| `get_config` | Get current config | none |
| `xtts_synthesize` | XTTS v2 synthesis | text, language, emotion, temperature |
| `clone_voice` | Clone voice | audio_file, voice_name, language, quality |
| `list_cloned_voices` | List clones | none |
| `get_xtts_status` | XTTS status | none |

### Installation Requirements
```bash
# v3.0.0 features require CoquiTTS installation:
./installers/install.sh coqui              # Auto-installation with validation
tts-notify --install-coqui              # Interactive installation
tts-notify --installation-status          # Check what's installed

# Installation status command shows:
# macOS Engine: ✅ Always available
# CoquiTTS Engine: ❌ Not installed (run --install-coqui)
# Audio Pipeline: ❌ Dependencies missing (run --install-coqui)
```

#### Installation Status Commands
```bash
# Verify current installation
tts-notify --installation-status          # Shows: macOS ✅, CoquiTTS ❌, Pipeline ❌

# Test CoquiTTS components (if installed)
tts-notify --test-installation           # Tests all v3.0.0 components
tts-notify --cloning-status               # Tests voice cloning functionality
tts-notify --pipeline-status               # Tests audio processing setup
```

#### v3.0.0 Architecture Status
- **Phase A (Multi-Language Engine)**: ✅ Architecture complete, installation pending
- **Phase B (Voice Cloning)**: ✅ Implementation complete, dependencies pending  
- **Phase C (Audio Pipeline)**: ✅ Implementation complete, dependencies pending
- **Dual-Engine System**: ✅ Runtime engine switching implemented
- **Optional Dependencies**: ✅ Architecture for graceful degradation working

### Installation Priority
1. **Install Core Dependencies** (always works):
   ```bash
   pip install ".[dev]"  # v2.0.0 functionality
   ```

2. **Install CoquiTTS Engine** (enables v3.0.0):
   ```bash
   ./installers/install.sh coqui
   tts-notify --test-installation  # Verify installation
   ```

3. **Install Feature-Specific Dependencies**:
   ```bash
   pip install ".[coqui-cloning]"    # Voice cloning
   pip install ".[audio-pipeline]"    # Audio processing
   ```

### Graceful Degradation
All v3.0.0 features fall back to macOS native engine if CoquiTTS is not installed, maintaining 100% backward compatibility.