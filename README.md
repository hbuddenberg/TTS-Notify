# TTS Notify v3.0.0

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/hbuddenb/tts-notify)
[![Release](https://img.shields.io/github/v/release/hbuddenb/tts-notify?include_prereleases&label=latest)](https://github.com/hbuddenb/tts-notify/releases)
[![Stars](https://img.shields.io/github/stars/hbuddenb/tts-notify)](https://github.com/hbuddenb/tts-notify)

🎯 **Modular Text-to-Speech notification system with dual-engine architecture: macOS native + CoquiTTS AI voices**

TTS Notify v3.0.0 features a complete modular architecture with dual TTS engines: native macOS voices and CoquiTTS with XTTS v2 for AI-powered voice synthesis. Three interfaces (CLI, MCP, REST API) share a unified core.

## ✨ What's New in v3.0.0

### 🤖 **CoquiTTS + XTTS v2 Integration**
- **Voice Cloning**: Clone voices from 6-30 second audio samples
- **Emotion Presets**: neutral, happy, sad, urgent, calm
- **17 Languages**: Cross-lingual synthesis
- **CPU Optimized**: Works 100% on CPU, no GPU required
- **Fast Inference**: <5 second latency, <8GB RAM

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/hbuddenb/tts-notify.git
cd tts-notify
uv pip install -e "."
```

### MCP Installer

```bash
cd TTS_Notify
./installers/install-uv-mcp.sh
```

### Basic Usage

```bash
# CLI - macOS native
tts-notify "Hello world"

# CLI - CoquiTTS
tts-notify "Hello" --engine coqui --emotion happy

# MCP Server
tts-notify --mode mcp

# REST API
tts-notify --mode api
```

## 🎵 Voice System

- **macOS Native**: 84+ voices
- **CoquiTTS**: Voice cloning, 17 languages, emotions

## 📖 Documentation

- **English**: README.md
- **Español**: README.es.md

## 📄 License

MIT License

---

**TTS Notify v3.0.0** - Dual-Engine TTS for macOS + AI Voices