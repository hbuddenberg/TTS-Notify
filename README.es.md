# TTS Notify v3.0.0

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/hbuddenb/tts-notify)
[![Release](https://img.shields.io/github/v/release/hbuddenb/tts-notify?include_prereleases&label=latest)](https://github.com/hbuddenb/tts-notify/releases)
[![Stars](https://img.shields.io/github/stars/hbuddenb/tts-notify)](https://github.com/hbuddenb/tts-notify)

🎯 **Sistema modular de texto a voz con arquitectura de doble motor: macOS nativo + voces AI de CoquiTTS**

TTS Notify v3.0.0 cuenta con una arquitectura modular completa con dos motores TTS: voces nativas de macOS y CoquiTTS con XTTS v2 para síntesis de voz con IA. Tres interfaces (CLI, MCP, REST API) comparten un núcleo unificado.

## ✨ Novedades en v3.0.0

### 🤖 **Integración CoquiTTS + XTTS v2**
- **Clonación de Voz**: Clona voces de muestras de audio de 6-30 segundos
- **Emociones**: neutral, happy, sad, urgent, calm
- **17 Idiomas**: Síntesis multilingüe
- **Optimizado para CPU**: Funciona 100% en CPU, sin GPU
- **Inferencia Rápida**: Latencia <5 segundos, RAM <8GB

## 🚀 Inicio Rápido

### Instalación

```bash
git clone https://github.com/hbuddenb/tts-notify.git
cd tts-notify
uv pip install -e "."
```

### Instalador MCP

```bash
cd TTS_Notify
./installers/install-uv-mcp.sh
```

### Uso Básico

```bash
# CLI - macOS nativo
tts-notify "Hola mundo"

# CLI - CoquiTTS
tts-notify "Hola" --engine coqui --emotion happy

# Servidor MCP
tts-notify --mode mcp

# API REST
tts-notify --mode api
```

## 🎵 Sistema de Voces

- **macOS Nativo**: 84+ voces
- **CoquiTTS**: Clonación de voz, 17 idiomas, emociones

## 📖 Documentación

- **English**: README.md
- **Español**: README.es.md

## 📄 Licencia

Licencia MIT

---

**TTS Notify v3.0.0** - Motor dual TTS para macOS + Voces AI
