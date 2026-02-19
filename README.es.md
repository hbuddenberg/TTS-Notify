# TTS Notify v3.0.0

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/hbuddenb/tts-notify)
[![Release](https://img.shields.io/github/v/release/hbuddenb/tts-notify?include_prereleases&label=latest)](https://github.com/hbuddenb/tts-notify/releases)
[![Stars](https://img.shields.io/github/stars/hbuddenb/tts-notify)](https://github.com/hbuddenb/tts-notify)

🎯 **Sistema modular de texto a voz con arquitectura de doble motor: macOS nativo + voces AI de CoquiTTS**

TTS Notify v3.0.0 cuenta con una arquitectura modular completa con dos motores TTS: voces nativas de macOS y CoquiTTS con XTTS v2 para síntesis de voz con IA. Tres interfaces (CLI, MCP, REST API) comparten un núcleo unificado.

---

## ✨ Novedades en v3.0.0

### 🤖 **Integración CoquiTTS + XTTS v2**
- **Clonación de Voz**: Clona voces de muestras de audio de 6-30 segundos
- **Emociones**: neutral, happy, sad, urgent, calm
- **17 Idiomas**: Síntesis multilingüe (Inglés, Español, Francés, Alemán, Italiano, Portugués, Polaco, Turco, Ruso, Holandés, Checo, Árabe, Chino, Japonés, Húngaro, Coreano, Hindi)
- **Optimizado para CPU**: Funciona 100% en CPU, sin GPU
- **Compatible con macOS Intel**: Soporte completo para Macs Intel con PyTorch 2.2.2 optimizado

### 🏗️ **Arquitectura Modular**
- **6 Componentes Core**: Separación limpia de responsabilidades
- **3 Interfaces**: CLI, Servidor MCP, REST API - usando el mismo núcleo
- **40% Menos Código**: Eliminación de duplicación mediante diseño inteligente

### 🎛️ **Configuración Inteligente**
- **60+ Variables de Entorno**: Control total
- **11 Perfiles Predefinidos**: Configuraciones listas para usar
- **Validación Pydantic**: Validación automática con mensajes útiles

---

## 🚀 Inicio Rápido

### Requisitos Previos
- **Python 3.10 o 3.11** (requerido para CoquiTTS)
- **macOS** (para TTS nativo) o Linux/Windows (solo CoquiTTS)
- **espeak-ng** (para fonemización de CoquiTTS): `brew install espeak-ng`

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/hbuddenb/tts-notify.git
cd tts-notify

# Instalación completa con soporte CoquiTTS
cd TTS_Notify
./installers/install.sh all
```

### Uso Básico

```bash
# CLI - TTS nativo de macOS
tts-notify "Hola mundo"

# CLI - Voces AI de CoquiTTS
tts-notify "Hola" --engine coqui --emotion happy

# CLI - Clonación de voz
tts-notify "Mi voz personalizada hablando" --engine coqui --voice-sample mi_voz.wav

# Listar voces disponibles
tts-notify --list

# Servidor MCP (para Claude Desktop)
tts-notify --mode mcp

# API REST
tts-notify --mode api
```

---

## 🎵 Sistema de Voces

### Voces Nativas de macOS
- **84+ voces** disponibles
- **Búsqueda Inteligente**: 4 niveles (exacta → prefijo → parcial → fallback)
- **Categorías**: Español, Enhanced, Premium, Siri, Otros

### Voces AI de CoquiTTS
- **Clonación de Voz**: Clona cualquier voz de una muestra de 6-30 segundos
- **Emociones**: 
  - `neutral` - Habla estándar (velocidad 1.0, temperatura 0.5)
  - `happy` - Tono alegre (velocidad 1.2, temperatura 0.7)
  - `sad` - Tono melancólico (velocidad 0.8, temperatura 0.3)
  - `urgent` - Tono rápido y alerta (velocidad 1.5, temperatura 0.6)
  - `calm` - Tono relajado y lento (velocidad 0.9, temperatura 0.4)
- **Multilingüe**: Habla en cualquiera de los 17 idiomas soportados

---

## 📦 Modos de Instalación

| Modo | Comando | Descripción |
|------|---------|-------------|
| **Completo** | `./installers/install.sh all` | Instalación completa con CoquiTTS |
| **Desarrollo** | `./installers/install.sh development` | Entorno de desarrollo con herramientas de testing |
| **Producción** | `./installers/install.sh production` | Solo CLI, minimal |
| **MCP** | `./installers/install.sh mcp` | Integración con Claude Desktop |

---

## 🔧 Configuración

### Variables de Entorno

```bash
# Configuración de Voz
TTS_NOTIFY_VOICE=monica          # Voz por defecto
TTS_NOTIFY_RATE=175              # Velocidad de habla (PPM)
TTS_NOTIFY_LANGUAGE=es           # Código de idioma

# Selección de Motor
TTS_NOTIFY_ENGINE=macos          # o "coqui"

# Configuración CoquiTTS
TTS_NOTIFY_COQUI_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
TTS_NOTIFY_COQUI_EMOTION=neutral

# Servidor API
TTS_NOTIFY_API_PORT=8000
TTS_NOTIFY_API_HOST=localhost
```

### Perfiles de Configuración

```bash
# Usar perfiles predefinidos
tts-notify --profile claude-desktop  # Optimizado para Claude Desktop
tts-notify --profile development      # Desarrollo con debugging
tts-notify --profile production       # Listo para producción
```

---

## 🖥️ Interfaces

### Interfaz CLI

```bash
# Uso básico
tts-notify "Tu mensaje aquí"

# Con opciones
tts-notify "Prueba" --voice monica --rate 200 --engine coqui

# Guardar en archivo
tts-notify "Grabación" --save salida --format wav

# Información del sistema
tts-notify --info
tts-notify --test-installation
```

### Servidor MCP (Claude Desktop)

```bash
# Iniciar servidor MCP
tts-notify --mode mcp

# Auto-configurar Claude Desktop
./installers/install-uv-mcp.sh
```

En Claude Desktop:
- "Lee en voz alta: Hola mundo"
- "Lista todas las voces en español"
- "Usa CoquiTTS para decir: Bonjour en francés"

### API REST

```bash
# Iniciar servidor API
tts-notify --mode api

# API disponible en http://localhost:8000
# Documentación interactiva en http://localhost:8000/docs
```

Endpoints de la API:
- `POST /speak` - Sintetizar voz
- `GET /voices` - Listar voces disponibles
- `GET /health` - Verificación de estado

---

## 🎤 Clonación de Voz

Clona cualquier voz desde una muestra de audio:

```bash
# Clonar voz desde muestra (6-30 segundos recomendado)
tts-notify "Esta es mi voz clonada hablando" \
  --engine coqui \
  --voice-sample mi_muestra_de_voz.wav

# Con emoción
tts-notify "Mensaje feliz" \
  --engine coqui \
  --voice-sample mi_voz.wav \
  --emotion happy
```

---

## 🌐 Soporte Multi-idioma

CoquiTTS soporta 17 idiomas:

| Idioma | Código | Idioma | Código |
|--------|--------|--------|--------|
| Inglés | `en` | Español | `es` |
| Francés | `fr` | Alemán | `de` |
| Italiano | `it` | Portugués | `pt` |
| Polaco | `pl` | Turco | `tr` |
| Ruso | `ru` | Holandés | `nl` |
| Checo | `cs` | Árabe | `ar` |
| Chino | `zh-cn` | Japonés | `ja` |
| Húngaro | `hu` | Coreano | `ko` |
| Hindi | `hi` | | |

```bash
# Síntesis multilingüe
tts-notify "Bonjour le monde" --engine coqui --language fr
tts-notify "Hola mundo" --engine coqui --language es
```

---

## 📊 Rendimiento

| Métrica | Valor |
|---------|-------|
| Detección de Voces | ~0.5s (75% más rápido con caché) |
| Inicio de CLI | ~0.3s |
| Uso de Memoria (CoquiTTS) | <8GB RAM |
| Latencia de Inferencia | <5 segundos |
| Plataformas Soportadas | macOS (Intel & Apple Silicon), Linux, Windows |

---

## 🧪 Desarrollo

```bash
# Instalar dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar tests
pytest

# Formatear código
black src tests && isort src tests

# Verificación de tipos
mypy src
```

---

## 📖 Documentación

- **[README.md](README.md)** - Documentación en inglés
- **[README.es.md](README.es.md)** - Documentación en español (este archivo)
- **[CHANGELOG.md](CHANGELOG.md)** - Historial de versiones
- **[TTS_Notify/README.md](TTS_Notify/README.md)** - Documentación técnica detallada

---

## 🤝 Contribuir

1. Haz fork del repositorio
2. Crea una rama de feature: `git checkout -b feature/caracteristica-increible`
3. Instala dependencias de desarrollo: `./installers/install.sh development`
4. Haz cambios con tests
5. Ejecuta tests: `pytest`
6. Commit: `git commit -m "Agregar característica increíble"`
7. Push: `git push origin feature/caracteristica-increible`
8. Abre un Pull Request

---

## 📄 Licencia

Licencia MIT - ver archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- [CoquiTTS](https://github.com/coqui-ai/TTS) - Motor TTS con IA
- [XTTS v2](https://huggingface.co/coqui/XTTS-v2) - Clonación de voz multilingüe
- [Comando `say` de macOS](https://ss64.com/mac/say.html) - Motor TTS nativo

---

**TTS Notify v3.0.0** - 🎯 Motor dual TTS para macOS + Voces AI
