# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-02-19

### 🎉 Major Release - CoquiTTS Integration

This is a major release featuring complete CoquiTTS integration with voice cloning, emotion presets, and multi-language support. The architecture has been completely redesigned for modularity and maintainability.

### ✨ Added

#### CoquiTTS + XTTS v2 Integration
- **Voice Cloning**: Clone voices from 6-30 second audio samples
- **Emotion Presets**: 5 emotion presets (neutral, happy, sad, urgent, calm)
- **17 Languages**: Cross-lingual synthesis support
  - English, Spanish, French, German, Italian, Portuguese
  - Polish, Turkish, Russian, Dutch, Czech, Arabic
  - Chinese, Japanese, Hungarian, Korean, Hindi
- **CPU Optimized**: Works 100% on CPU, no GPU required
- **macOS Intel Support**: Full compatibility with Intel Macs using PyTorch 2.2.2

#### Modular Architecture
- **6 Core Components**: config, voice, engine, models, exceptions, installer
- **3 Interfaces**: CLI, MCP Server, REST API - unified core
- **40% Code Reduction**: Eliminated duplication

#### Configuration System
- **60+ Environment Variables**: Complete control over all settings
- **11 Predefined Profiles**: Ready-to-use configurations
- **Pydantic Validation**: Automatic validation with helpful error messages
- **YAML Configuration**: Human-readable config files

#### Voice System
- **84+ macOS Voices**: Full support for all system voices
- **Smart Voice Search**: 4-tier matching algorithm
  - Exact match (case/accent-insensitive)
  - Prefix match
  - Partial match
  - Fallback to default
- **Voice Caching**: 75% faster detection with intelligent caching
- **Advanced Voice Cloning**: Persistent voice profiles
  - Clone voices from FLAC, WAV, MP3, OGG, M4A files
  - Auto-trimming for samples >30 seconds
  - Automatic denoising with noisereduce
  - Audio normalization for consistent quality
  - Voice profile storage in `~/.tts-notify/`
  - Cross-lingual synthesis with cloned voices

#### Installer Improvements
- **Cross-platform**: macOS, Linux, Windows support
- **UV Integration**: Lightning-fast package management
- **Auto-detection**: Detects Python 3.10 for CoquiTTS compatibility
- **espeak-ng Support**: Automatic phonemizer dependency installation
- **Complete Dependency Management**: All CoquiTTS language dependencies

### 🔧 Changed

- **Python Version**: Now requires Python 3.10 or 3.11 (for CoquiTTS)
- **Architecture**: Complete modular redesign
- **Configuration**: Environment variables now use `TTS_NOTIFY_` prefix
- **CLI**: Enhanced with new options for CoquiTTS
- **API**: Improved error handling and validation

### 🐛 Fixed

- **macOS Intel Compatibility**: Resolved PyTorch compatibility issues
  - Uses PyTorch 2.2.2 for macOS Intel
  - Compatible TTS 0.22.0 version
  - Fixed llvmlite build issues with `--prefer-binary`
- **Installer**: Removed incorrect architecture blocking code
- **Model List API**: Fixed compatibility with TTS 0.22.0 API changes
- **Voice Detection**: Improved caching and error handling
- **Transformers Compatibility**: Pinned `transformers>=4.33.0,<4.38` for TTS 0.22.0
  - Transformers 5.x removed `BeamSearchScorer` breaking CoquiTTS
- **Audio File Path Resolution**: CLI now searches multiple locations for audio files
  - As provided (absolute or relative to cwd)
  - Current working directory
  - Project root directory
- **FLAC Support**: Voice cloning now works with FLAC audio files

### 📦 Dependencies

#### Core
- `pydantic>=2.0.0` - Configuration validation
- `pyyaml>=6.0` - YAML configuration support

#### CoquiTTS (Optional)
- `TTS==0.22.0` - CoquiTTS engine
- `torch==2.2.2` - PyTorch (macOS Intel compatible)
- `torchaudio==2.2.2` - Audio processing
- `numpy<2` - Array operations (compatibility)
- `llvmlite<0.46` - JIT compilation (prebuilt wheel)
- `numba>=0.60,<0.64` - Numerical computing
- Language support: bangla, bnnumerizer, bnunicodenormalizer, cython, encodec, g2pkk, gruut, hangul_romanize, jamo

#### MCP Server (Optional)
- `mcp>=1.0.0` - Model Context Protocol
- `fastmcp>=2.0.0` - FastMCP server

#### REST API (Optional)
- `fastapi>=0.104.0` - Web framework
- `uvicorn>=0.24.0` - ASGI server

### 🗑️ Removed

- Python 3.12 and 3.13 support (CoquiTTS incompatible)
- Legacy configuration format
- Duplicate code paths

### 📝 Documentation

- Comprehensive README.md with all features documented
- Spanish documentation (README.es.md)
- Updated installation instructions
- API documentation
- Voice cloning guide

---

## [2.0.0] - 2025-01-15

### Added
- Complete modular architecture rewrite
- MCP Server for Claude Desktop integration
- REST API with FastAPI
- Configuration profiles system
- Voice caching system

### Changed
- Major code refactoring
- Improved error handling
- Enhanced voice detection

---

## [1.5.0] - 2024-12-01

### Added
- Basic TTS functionality
- macOS native voice support
- CLI interface
- Configuration via environment variables

---

[3.0.0]: https://github.com/hbuddenb/tts-notify/releases/tag/v3.0.0
[2.0.0]: https://github.com/hbuddenb/tts-notify/releases/tag/v2.0.0
[1.5.0]: https://github.com/hbuddenb/tts-notify/releases/tag/v1.5.0
