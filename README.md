# TTS Notify v3.0.0

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/hbuddenb/tts-notify)
[![Release](https://img.shields.io/github/v/release/hbuddenb/tts-notify?include_prereleases&label=latest)](https://github.com/hbuddenb/tts-notify/releases)
[![Stars](https://img.shields.io/github/stars/hbuddenb/tts-notify)](https://github.com/hbuddenb/tts-notify)

🎯 **Modular Text-to-Speech notification system with dual-engine architecture: macOS native + CoquiTTS AI voices**

TTS Notify v3.0.0 features a complete modular architecture with dual TTS engines: native macOS voices and CoquiTTS with XTTS v2 for AI-powered voice synthesis. Three interfaces (CLI, MCP, REST API) share a unified core.

---

## ✨ What's New in v3.0.0

### 🤖 **CoquiTTS + XTTS v2 Integration**
- **Voice Cloning**: Clone voices from 6-30 second audio samples
- **Emotion Presets**: neutral, happy, sad, urgent, calm
- **17 Languages**: Cross-lingual synthesis (English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi)
- **CPU Optimized**: Works 100% on CPU, no GPU required
- **macOS Intel Compatible**: Full support for Intel Macs with optimized PyTorch 2.2.2

### 🏗️ **Modular Architecture**
- **6 Core Components**: Clean separation of concerns
- **3 Interfaces**: CLI, MCP Server, REST API - all using the same core
- **40% Code Reduction**: Eliminated duplication through smart design

### 🎛️ **Intelligent Configuration**
- **60+ Environment Variables**: Complete control
- **11 Predefined Profiles**: Ready-to-use configurations
- **Pydantic Validation**: Automatic validation with helpful errors

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10 or 3.11** (required for CoquiTTS)
- **macOS** (for native TTS) or Linux/Windows (CoquiTTS only)
- **espeak-ng** (for CoquiTTS phonemization): `brew install espeak-ng`

### Installation

```bash
# Clone the repository
git clone https://github.com/hbuddenb/tts-notify.git
cd tts-notify

# Complete installation with CoquiTTS support
cd TTS_Notify
./installers/install.sh all
```

### Basic Usage

```bash
# CLI - macOS native TTS
tts-notify "Hello world"

# CLI - CoquiTTS AI voices
tts-notify "Hello" --engine coqui --emotion happy

# CLI - Voice cloning
tts-notify "Custom voice speaking" --engine coqui --voice-sample my_voice.wav

# List available voices
tts-notify --list

# MCP Server (for Claude Desktop)
tts-notify --mode mcp

# REST API
tts-notify --mode api
```

---

## 🎵 Voice System

### macOS Native Voices
- **84+ voices** available
- **Smart Search**: 4-tier matching (exact → prefix → partial → fallback)
- **Categories**: Español, Enhanced, Premium, Siri, Others

### CoquiTTS AI Voices
- **Voice Cloning**: Clone any voice from a 6-30 second audio sample
- **Emotion Presets**: 
  - `neutral` - Standard speech (speed 1.0, temperature 0.5)
  - `happy` - Cheerful tone (speed 1.2, temperature 0.7)
  - `sad` - Melancholic tone (speed 0.8, temperature 0.3)
  - `urgent` - Fast, alert tone (speed 1.5, temperature 0.6)
  - `calm` - Relaxed, slow tone (speed 0.9, temperature 0.4)
- **Cross-lingual**: Speak in any of 17 supported languages

---

## 📦 Installation Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Complete** | `./installers/install.sh all` | Full installation with CoquiTTS |
| **Development** | `./installers/install.sh development` | Dev environment with testing tools |
| **Production** | `./installers/install.sh production` | CLI only, minimal |
| **MCP** | `./installers/install.sh mcp` | Claude Desktop integration |

---

## 🔧 Configuration

### Environment Variables

```bash
# Voice Settings
TTS_NOTIFY_VOICE=monica          # Default voice
TTS_NOTIFY_RATE=175              # Speech rate (WPM)
TTS_NOTIFY_LANGUAGE=es           # Language code

# Engine Selection
TTS_NOTIFY_ENGINE=macos          # or "coqui"

# CoquiTTS Settings
TTS_NOTIFY_COQUI_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
TTS_NOTIFY_COQUI_EMOTION=neutral

# API Server
TTS_NOTIFY_API_PORT=8000
TTS_NOTIFY_API_HOST=localhost
```

### Configuration Profiles

```bash
# Use predefined profiles
tts-notify --profile claude-desktop  # Optimized for Claude Desktop
tts-notify --profile development      # Development with debugging
tts-notify --profile production       # Production ready
```

---

## 🖥️ Interfaces

### CLI Interface

```bash
# Basic usage
tts-notify "Your message here"

# With options
tts-notify "Test" --voice monica --rate 200 --engine coqui

# Save to file
tts-notify "Recording" --save output --format wav

# System info
tts-notify --info
tts-notify --test-installation
```

### MCP Server (Claude Desktop)

```bash
# Start MCP server
tts-notify --mode mcp

# Auto-configure Claude Desktop
./installers/install-uv-mcp.sh
```

In Claude Desktop:
- "Lee en voz alta: Hola mundo"
- "Lista todas las voces en español"
- "Usa CoquiTTS para decir: Hello in French"

### REST API

```bash
# Start API server
tts-notify --mode api

# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

API Endpoints:
- `POST /speak` - Synthesize speech
- `GET /voices` - List available voices
- `GET /health` - Health check

---

## 🎤 Voice Cloning

Clone any voice from an audio sample:

```bash
# Clone voice from sample (6-30 seconds recommended)
tts-notify "This is my cloned voice speaking" \
  --engine coqui \
  --voice-sample my_voice_sample.wav

# With emotion
tts-notify "Happy message" \
  --engine coqui \
  --voice-sample my_voice.wav \
  --emotion happy
```

---

## 🌐 Multi-Language Support

CoquiTTS supports 17 languages:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | `en` | Spanish | `es` |
| French | `fr` | German | `de` |
| Italian | `it` | Portuguese | `pt` |
| Polish | `pl` | Turkish | `tr` |
| Russian | `ru` | Dutch | `nl` |
| Czech | `cs` | Arabic | `ar` |
| Chinese | `zh-cn` | Japanese | `ja` |
| Hungarian | `hu` | Korean | `ko` |
| Hindi | `hi` | | |

```bash
# Cross-lingual synthesis
tts-notify "Bonjour le monde" --engine coqui --language fr
tts-notify "Hola mundo" --engine coqui --language es
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Voice Detection | ~0.5s (75% faster with caching) |
| CLI Startup | ~0.3s |
| Memory Usage (CoquiTTS) | <8GB RAM |
| Inference Latency | <5 seconds |
| Supported Platforms | macOS (Intel & Apple Silicon), Linux, Windows |

---

## 🧪 Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src tests && isort src tests

# Type checking
mypy src
```

---

## 📖 Documentation

- **[README.md](README.md)** - English documentation (this file)
- **[README.es.md](README.es.md)** - Spanish documentation
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[TTS_Notify/README.md](TTS_Notify/README.md)** - Detailed technical docs

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Install dev dependencies: `./installers/install.sh development`
4. Make changes with tests
5. Run tests: `pytest`
6. Commit: `git commit -m "Add amazing feature"`
7. Push: `git push origin feature/amazing-feature`
8. Open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [CoquiTTS](https://github.com/coqui-ai/TTS) - AI-powered TTS engine
- [XTTS v2](https://huggingface.co/coqui/XTTS-v2) - Multi-language voice cloning
- [macOS `say` command](https://ss64.com/mac/say.html) - Native TTS engine

---

**TTS Notify v3.0.0** - 🎯 Dual-Engine TTS for macOS + AI Voices