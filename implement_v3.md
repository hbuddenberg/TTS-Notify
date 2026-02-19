# TTS Notify v3 — Plan de Desarrollo e Implementación

## 1. Resumen Ejecutivo
- **Objetivo**: Evolucionar TTS Notify a la versión 3 añadiendo soporte opcional para CoquiTTS, **soporte multi-idioma con gestión automática de modelos**, clonación de voces y pipelines de audio avanzados sin romper la compatibilidad existente.
- **Alcance clave**:
  - Selección dinámica de motor (`macos` vs `coqui`).
  - **Soporte multi-idioma**: 17 idiomas con detección automática y forzado por configuración.
  - **Gestión inteligente de modelos**: Descarga automática, caché, y optimización de almacenamiento.
  - Perfiles de voz personalizados basados en samples de audio.
  - Pipeline modular de preprocesamiento y conversión.
  - Extensiones CLI, MCP y futura API REST con gestión de idiomas.
  - Telemetría básica y fallback seguro.
- **Resultados esperados**: Plataforma TTS flexible, **multi-idioma**, capaz de aprovechar modelos neuronales, manteniendo la facilidad de uso actual y una arquitectura escalable.

---

## 2. Alcance por Fases

| Fase | Objetivos | Hitos |
| ---- | --------- | ----- |
| A | Motor Coqui básico, **soporte multi-idioma y gestión automática de modelos** | Registro condicional, síntesis simple, **detección automática de idiomas**, **descarga automática de modelos**, CLI `--engine`, **CLI `--language`** |
| B | Perfiles de voz (clonación) **multi-idioma** y embeddings persistentes | Gestión de perfiles, **clonación por idioma**, CLI/MCP para creación y listado |
| C | Pipeline de audio + conversión formatos | Normalización, trimming, reducción de ruido opcional, `format_converter`, **optimización por idioma** |
| D | Extensiones MCP/API + telemetría con **gestión de idiomas** | **Nuevos tools MCP para idiomas**, endpoints API preliminares, métricas básicas **por idioma** |
| E | Fine-tuning experimental (flag) | Documentar y habilitar flags para investigación futura **multi-idioma** |

Las fases son incrementales; cada una puede desplegarse tras pruebas específicas sin bloquear las demás.

---

## 3. Cambios Arquitectónicos y Estructura de Directorios

```
tts_notify/
  core/
    coqui_engine.py
    voice_profile_manager.py
    audio_pipeline.py
    embeddings/
      coqui_embedding.py
      speaker_index.py
  plugins/
    preprocess/
      silence_trimmer.py
      noise_reducer.py
      normalizer.py
    conversion/
      format_converter.py
  data/
    voices/
      profiles/
      embeddings/
      samples/
  utils/
    telemetry.py
    resource_monitor.py
```

- **core/** aloja los motores y lógicas de negocio principales.
- **plugins/** permite agregar/activar transformaciones sin modificar núcleo.
- **data/** conserva artefactos generados por usuarios (perfiles, audios, embeddings) con estructura clara.
- **utils/** incorpora monitorización y métricas.

---

## 4. Extensión de Configuración (TTSConfig)

### Campos nuevos propuestos

#### Engine Selection
- `TTS_NOTIFY_ENGINE` (macos|coqui)
- `TTS_NOTIFY_COQUI_MODEL`, `TTS_NOTIFY_COQUI_MODEL_TYPE`
- `TTS_NOTIFY_COQUI_USE_GPU`, `TTS_NOTIFY_COQUI_AUTOINIT`
- `TTS_NOTIFY_COQUI_SPEAKER`, `TTS_NOTIFY_COQUI_STYLE`

#### **Multi-Language Support (NUEVO)**
- `TTS_NOTIFY_DEFAULT_LANGUAGE` (auto|en|es|fr|de|it|pt|nl|pl|ru|zh|ja|ko)
- `TTS_NOTIFY_COQUI_LANGUAGE_FALLBACK` (en|es|fr|de|it|pt)
- `TTS_NOTIFY_FORCE_LANGUAGE` (boolean)
- `TTS_NOTIFY_AUTO_DOWNLOAD_MODELS` (boolean)

#### **Model Management (NUEVO)**
- `TTS_NOTIFY_COQUI_CACHE_MODELS` (boolean)
- `TTS_NOTIFY_COQUI_MODEL_CACHE_DIR` (path opcional)
- `TTS_NOTIFY_COQUI_MODEL_TIMEOUT` (segundos)
- `TTS_NOTIFY_COQUI_OFFLINE_MODE` (boolean)

#### Voice Cloning and Profiles
- `TTS_NOTIFY_COQUI_PROFILE_DIR`, `TTS_NOTIFY_COQUI_EMBEDDING_DIR`
- `TTS_NOTIFY_COQUI_ENABLE_CLONING`
- `TTS_NOTIFY_COQUI_MIN_SAMPLE_SECONDS`, `TTS_NOTIFY_COQUI_MAX_SAMPLE_SECONDS`
- `TTS_NOTIFY_COQUI_AUTO_CLEAN_AUDIO`, `TTS_NOTIFY_COQUI_AUTO_TRIM_SILENCE`

#### Audio Pipeline
- `TTS_NOTIFY_COQUI_NOISE_REDUCTION`, `TTS_NOTIFY_COQUI_DIARIZATION`
- `TTS_NOTIFY_COQUI_CONVERSION_ENABLED`, `TTS_NOTIFY_COQUI_TARGET_FORMATS`
- `TTS_NOTIFY_COQUI_EMBEDDING_CACHE`, `TTS_NOTIFY_COQUI_EMBEDDING_FORMAT`

#### Experimental Features
- `TTS_NOTIFY_EXPERIMENTAL_FINE_TUNING`

### Validaciones
- Si `ENGINE=coqui` y no hay modelo ⇒ error.
- Auto-creación de directorios si no se definen.
- GPU solicitada sin soporte ⇒ se fuerza CPU y se registra advertencia.

---

## 5. Motor CoquiTTSEngine (Fase A)

### Investigación Técnica de CoquiTTS
Basado en documentación oficial de CoquiTTS 0.27.0+:

#### **Características Clave**
- **Python 3.9+** (soporte hasta < 3.13)
- **XTTS v2**: Modelo multi-idioma (17 idiomas, 2GB)
- **Instalación**: `pip install coqui-tts` con extras opcionales
- **Soporte GPU**: Aceleración opcional con CUDA
- **Streaming**: <200ms latency para streaming
- **Descarga Automática**: Modelos se descargan y cachean automáticamente

#### **Modelos Recomendados**
```python
# Multi-idioma (default recomendado)
"tts_models/es/multi-dataset/xtts_v2"  # 2GB, 17 idiomas, 17 speakers

# Específicos por idioma
"tts_models/esu/fairseq/vits"         # 50MB, español solo
"tts_models/en/ljspeech/tacotron2-DDC" # inglés solo
```

#### **API Básica**
```python
from TTS.api import TTS

# Inicialización
tts = TTS(model_name="tts_models/es/multi-dataset/xtts_v2")

# Síntesis básica
audio = tts.tts("Hola mundo", speaker=tts.speakers[0])

# Guardar a archivo
tts.tts_to_file(text="Hola mundo", speaker=tts.speakers[0], file_path="output.wav")
```

### Requisitos Implementación
- **Dependencias opcionales**: `pip install .[coqui]` con soporte para 17 idiomas
- **Inicialización lazy**: `asyncio.to_thread` para evitar bloquear event loop
- **Detección automática de idiomas**: CoquiTTS detecta idioma del texto automáticamente
- **Gestión de modelos**: Descarga automática, caché inteligente, y status checking
- **Fallback robusto**: macOS engine siempre disponible como fallback
- **Métodos mínimos**: `initialize`, `cleanup`, `is_available`, `get_supported_voices`, `speak`, `synthesize`, `save`
- **Soporte multi-idioma**: Detección automática + forzado por configuración
- **Compatibilidad**: `TTSResponse` y formatos (WAV con conversión opcional)
- **Logging claro**: Diagnósticos detallados para descargas y detección de idiomas

### Componentes Clave del Engine

#### **1. Detección y Gestión de Idiomas**
```python
async def check_language_availability(self, language: str) -> Dict[str, Any]:
    """Verificar disponibilidad de idioma y estado del modelo"""
    
async def ensure_language_available(self, language: str) -> bool:
    """Asegurar que el idioma esté disponible (descargar si es necesario)"""
    
async def _determine_language(self, request: TTSRequest) -> str:
    """Determinar idioma basado en configuración y request con fallback"""
```

#### **2. Gestión Inteligente de Modelos**
```python
# Modelos multi-idioma con capacidades
self.multi_language_models = {
    "tts_models/es/multi-dataset/xtts_v2": {
        "languages": ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"],
        "size_gb": 2.0,
        "speakers": 17,
        "quality": "enhanced"
    }
}

# Modelos específicos por idioma
self.single_language_models = {
    "es": ["tts_models/esu/fairseq/vits"],
    "en": ["tts_models/en/ljspeech/tacotron2-DDC"],
    # ... otros idiomas
}
```

### Pasos de Implementación
1. **Crear `coqui_engine.py`** en `core/` con soporte multi-idioma
2. **Implementar gestión de modelos** con descarga automática y caché
3. **Actualizar bootstrap** en `tts_engine.py` para registro condicional
4. **Extender CLI** con flags `--engine`, `--model`, y `--language`
5. **Validar fallback** robusto a macOS cuando Coqui no está disponible
6. **Añadir detección automática** de idiomas con forzado manual
7. **Implementar herramientas** de gestión de idiomas (--list-languages, --download-language)

---

## 6. Voice Cloning & Voice Profiles (Fase B)

### Componentes
- `voice_profile_manager.py`: creación, lectura, eliminación de perfiles.
- Directorio `data/voices/profiles` (metadatos JSON/YAML).
- Directorio `data/voices/embeddings` (archivos `.npy` o `.pt`).
- Directorio `data/voices/samples` (audios fuente).

### Flujo de creación de perfil
1. Validar audios (formato, duración, nivel).
2. Pipeline de audio (limpieza, normalización, trimming, opcional diarización).
3. Extracción de embeddings (depende del modelo; preferir XTTS o similares).
4. Agregado de embeddings (media ponderada, normalización).
5. Guardado de metadata (idioma, género estimado, estadísticas).
6. Registro de voz en `VoiceManager` como `Voice` con `metadata.embedding_path`.

### Nuevos comandos/flags CLI

#### Engine y Modelo Selection
- `--engine <macos|coqui>`: seleccionar motor TTS
- `--model <nombre>`: especificar modelo CoquiTTS
- `--diagnose-engine <engine>`: verificar disponibilidad y tiempo de init

#### **Soporte Multi-Idioma (NUEVO)**
- `--language <auto|en|es|fr|de|it|pt|nl|pl|ru|zh|ja|ko>`: idioma preferido
- `--force-language`: forzar idioma específico ignorando detección automática
- `--list-languages`: listar idiomas disponibles y por descargar
- `--download-language <lang>`: descargar modelo para idioma específico
- `--model-status`: mostrar estado de modelos descargados
- `--auto-download`: habilitar/deshabilitar descarga automática de modelos

#### Voice Cloning and Profiles
- `--clone --name <id> --files <lista>`: crear perfil personalizado.
- `--list-profiles`: enumerar perfiles disponibles.
- `--speaker <id>` / `--style <id>`: seleccionar speaker/estilo nativo de modelo.
- `--voice <profile_id>`: usar perfil clonando (mapeado por `VoiceManager`).
- `--purge-profile <id>`: eliminar perfil sensible.

#### Audio Processing
- `--convert <archivo> --to <formato>`: conversión de formatos.

### Nuevos tools MCP

#### Herramientas Existentes (Enhanced)
1. **`speak_text`** - Con soporte multi-idioma:
   - Parámetros: `text`, `voice`, `rate`, `engine`, `model`, `language`, `force_language`, `auto_download`
2. **`list_voices`** - Listado con filtrado por idioma
3. **`save_audio`** - Con metadatos de idioma

#### **Gestión de Idiomas y Modelos (NUEVO)**
4. **`list_languages`** - Listar idiomas disponibles y por descargar
5. **`download_language`** - Descargar modelo para idioma específico
6. **`get_model_status`** - Estado detallado de modelos descargados
7. **`engine_info`** - Capacidades por engine con información de idiomas

#### Voice Cloning and Profiles
8. **`create_voice_profile`** - Crear perfil personalizado con idioma
9. **`list_voice_profiles`** - Listar perfiles disponibles por idioma
10. **`describe_voice_profile`** - Descripción con metadatos de idioma
11. **`purge_voice_profile`** - Eliminar perfil sensible

#### Audio Processing
12. **`convert_audio`** - Conversión de formatos con metadatos de idioma

**Integración segura**: Validar existencia, retornar mensajes claros y rutas relativas, uso de `asyncio.to_thread` para tareas pesadas.

---

## 6.5. Gestión de Idiomas y Modelos (Fase A)

### Visión General
El sistema de gestión de idiomas y modelos permite a los usuarios utilizar CoquiTTS con múltiples idiomas de forma transparente, con descarga automática de modelos cuando sea necesario.

### Arquitectura del Sistema

#### **1. Detección Automática de Idiomas**
```python
# Estrategia de detección jerárquica
1. Idioma especificado en request (CLI flag --language, MCP parameter)
2. Idioma forzado en configuración (TTS_NOTIFY_FORCE_LANGUAGE + DEFAULT_LANGUAGE)
3. Idioma preferido en configuración (TTS_NOTIFY_DEFAULT_LANGUAGE)
4. Detección automática por CoquiTTS (si modelo lo soporta)
5. Fallback a español (TTS_NOTIFY_COQUI_LANGUAGE_FALLBACK)
```

#### **2. Gestión Inteligente de Modelos**
```python
class ModelManager:
    def __init__(self):
        self.multi_language_models = {
            "tts_models/es/multi-dataset/xtts_v2": {
                "languages": ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"],
                "size_gb": 2.0,
                "speakers": 17,
                "quality": "enhanced",
                "streaming": True
            }
        }
        
        self.single_language_models = {
            "es": ["tts_models/esu/fairseq/vits"],
            "en": ["tts_models/en/ljspeech/tacotron2-DDC"],
            "fr": ["tts_models/fr/multi-dataset/xtts_v2"],
            "de": ["tts_models/de/thorsten-vits"],
            "it": ["tts_models/it/mai_male"],
            # ... más idiomas
        }
```

#### **3. Descarga Automática y Caching**
- **Ubicación de caché**: `~/.local/share/tts/` (configurable via TTS_NOTIFY_COQUI_MODEL_CACHE_DIR)
- **Verificación de integridad**: Checksums MD5 para detectar corrupción
- **Limpieza automática**: Opción para limpiar modelos no usados
- **Modo offline**: `TTS_NOTIFY_OFFLINE_MODE=true` para evitar descargas

### Experiencia de Usuario

#### **CLI Experience**
```bash
# Primer uso - descarga automática transparente
$ tts-notify "Hello world" --engine coqui --language en
📥 Downloading XTTS v2 model (2.0GB)... This may take a few minutes
[████████████████████] 100% 2.0GB/2.0GB [00:45<00:00, 2.2MB/s]
✅ Model downloaded. Generating audio with English...
🔊 Audio generated

# Detección automática
$ tts-notify "Hola mundo" --engine coqui
🔊 Audio generated with auto-detected Spanish language

# Forzar idioma específico
$ tts-notify "Hello world" --engine coqui --language es --force-language
🔊 Audio generated with forced Spanish language

# Listar idiomas disponibles
$ tts-notify --list-languages
🌍 CoquiTTS Language Support:
📦 Multi-Language (XTTS v2):
  ✅ ES (available, 2.0GB)
  ✅ EN (available, 2.0GB)
  ⬇️ FR (2.0GB - download available)
  ⬇️ DE (2.0GB - download available)

# Descargar idioma específico
$ tts-notify --download-language fr
📥 Downloading model for French (2.0GB)...
✅ Model for FR downloaded successfully

# Estado de modelos
$ tts-notify --model-status
📊 CoquiTTS Model Status:
✅ Loaded model: tts_models/es/multi-dataset/xtts_v2
   🌍 Supports: 17 languages
💾 Cache size: 2048.5MB
🌍 Available languages: ES, EN, FR, DE, IT, PT, NL
```

#### **MCP/Claude Desktop Integration**
```
"Generate audio in Spanish: Hello world"
"Generate audio in English: Hola mundo"
"Use CoquiTTS with French language: Bonjour le monde"
"Force German language: Hello world"
"List available languages for CoquiTTS"
"Download model for Japanese language"
"Show CoquiTTS model status"
```

### Configuración Avanzada

#### **Variables de Entorno para Gestión de Idiomas**
```bash
# Configuración global de idioma
export TTS_NOTIFY_DEFAULT_LANGUAGE=es
export TTS_NOTIFY_FORCE_LANGUAGE=true
export TTS_NOTIFY_COQUI_LANGUAGE_FALLBACK=en

# Gestión de modelos
export TTS_NOTIFY_AUTO_DOWNLOAD_MODELS=true
export TTS_NOTIFY_COQUI_CACHE_MODELS=true
export TTS_NOTIFY_COQUI_MODEL_TIMEOUT=300
export TTS_NOTIFY_COQUI_OFFLINE_MODE=false
```

#### **Perfiles Predefinidos**
```bash
# Perfil multi-idioma
tts-notify --profile multi-lang --engine coqui "Hello world"

# Perfil específico por idioma
tts-notify --profile spanish --engine coqui "Hello world"
tts-notify --profile english --engine coqui "Hello world"
tts-notify --profile french --engine coqui "Hello world"
```

### Rendimiento y Optimización

#### **Targets de Performance**
- **Carga inicial de modelo**: ≤ 30 segundos (2GB model)
- **Switch entre idiomas**: ≤ 2 segundos (modelo ya cargado)
- **Descarga de modelo**: 2-5MB/s promedio
- **Uso de memoria**: ~500MB adicional para XTTS v2
- **Caché inteligente**: No descargar modelos ya existentes

#### **Estrategias de Optimización**
1. **Lazy Loading**: Modelos solo se cargan cuando se necesitan
2. **Model Caching**: Modelos descargados persisten entre sesiones
3. **Language Switching**: Cambio instantáneo entre idiomas del mismo modelo
4. **Memory Management**: Limpieza de modelos no usados configurable
5. **Download Progress**: Indicadores de progreso detallados para grandes descargas

---

## 7. Pipeline de Audio (Fase C)

### Objetivos
- Preprocesar audios de entrada para mejorar calidad de embeddings y voz resultante.
- Operaciones configurables:
  - Resample y normalización (RMS o LUFS simple).
  - Eliminación de silencios extremos.
  - Reducción de ruido (spectral gating).
  - Diarización segmentada (cuando se habilite).

### Diseño
- `audio_pipeline.py` con clase `AudioPipeline` que reciba `config` y devuelva lista de segmentos procesados (`Path`).
- `plugins/preprocess/` para cada etapa, desacopladas y activables según config.
- Uso de librerías como `librosa`, `soundfile`, `pydub`.
- Flags para activar/desactivar (`TTS_NOTIFY_COQUI_NOISE_REDUCTION`, etc.).

---

## 8. Conversión de Formatos (Fase C)

- Plugin `plugins/conversion/format_converter.py`.
- Entrada: WAV (generado por Coqui).
- Salida: MP3, FLAC, OGG (según `TTS_NOTIFY_COQUI_TARGET_FORMATS`).
- Dependencias opcionales: `pydub`, `ffmpeg`.
- Integrado en `CoquiTTSEngine.save()` y `synthesize()`:
  - Si formato deseado ≠ WAV ⇒ ejecutar conversión (con logs y manejo de errores).
  - Mantener AIFF como default para macOS.

---

## 9. Extensiones CLI y UX

### Flags clave
- `--engine`, `--model`, `--speaker`, `--style`.
- `--clone`, `--files`, `--name`, `--list-profiles`.
- `--convert <archivo> --to <formato>`.
- `--diagnose-engine <engine>` (verificar disponibilidad, dependencias, tiempo de init).

### Flujos
- Al iniciar CLI:
  1. Cargar config según perfil/env.
  2. `bootstrap_engines(config)`.
  3. Si se solicitan operaciones de gestión (clonación/listado) ⇒ ejecutarlas y salir.
  4. Para síntesis/guardado ⇒ seleccionar motor, voice (nativo o perfil), ejecutar.

  Para el comando `--list`, se listará el motor activo: por defecto el configurado (generalmente macOS) y, si se especifica `--engine`, se mostrará el inventario correspondiente (por ejemplo `--engine coqui --list`).

---

## 10. Extensiones MCP (Fase D)

### Nuevas herramientas
1. `create_voice_profile` (argumentos: nombre, lista de archivos, metadata opcional).
2. `list_voice_profiles`.
3. `describe_voice_profile`.
4. `engine_info` (devuelve capacidades, formatos y estado de inicialización).
5. `convert_audio` (archivo + formato destino, cuando conversión habilitada).

### Consideraciones
- Validar rutas relativas y asegurar que MCP no bloquee el proceso (usar `asyncio.to_thread` para tareas pesadas).
- Documentar esquemas de respuesta JSON.

---

## 11. API REST (Opcional v3 / v3.1)

Endpoints sugeridos:
- `POST /profiles` (multipart) — crea perfil.
- `GET /profiles` — lista.
- `GET /profiles/{id}` — metadata.
- `DELETE /profiles/{id}` — eliminación.
- `POST /tts` — síntesis (motor, perfil, speaker).
- `GET /voices` — voces disponibles.
- `GET /engines` — resumen.
- `POST /convert` — conversión de formatos.

Requiere FastAPI/Starlette en extras específicos. Puede planificarse para v3.1 si el tiempo es limitado.

---

## 12. Telemetría y Monitorización (Fase D)

- `utils/telemetry.py`: registrar duración de síntesis, tamaño de audio, uso de memoria (psutil) y almacenar JSON (`data/telemetry/metrics.json`).
- `utils/resource_monitor.py`: comprobar GPU, memoria, threads activos.
- Exponer información vía CLI (`--diagnose-engine`) y MCP (`engine_info`).

---

## 13. Seguridad y Privacidad

- Guardar datos de usuario localmente (no subir a servicios externos).
- Proveer comando `--purge-profile <id>` para eliminar perfiles sensibles.
- Almacenar checksums en metadata para detectar corrupción.
- Validar tipo de archivo y duración antes de aceptar la clonación.
- Documentar prácticas recomendadas (audios limpios, sin ruido).

---

## 14. Rendimiento y Optimización

### Targets iniciales
- Latencia de síntesis Coqui (texto corto) ≤ 2.5 × latencia de `say`.
- Creación de perfil con 2 muestras ≤ 30 segundos.
- Uso de memoria estable (sin growth tras 50 peticiones).

### Estrategias
- Cachear instancia de modelo Coqui mientras no cambie el nombre del modelo.
- Cache de embeddings (`TTS_NOTIFY_COQUI_EMBEDDING_CACHE`).
- Limitar concurrencia (`TTS_NOTIFY_MAX_CONCURRENT`).
- Reutilizar threads en conversions/preprocesamiento si es necesario.

---

## 15. Fallback y Manejo de Errores

- Si Coqui no está disponible ⇒ log WARN, fallback a macOS.
- Si embedding inválido ⇒ mensaje y fallback a speaker/model default.
- Si modelo no soporta estilo ⇒ ignorar estilo y registrar aviso.
- Si GPU no disponible ⇒ fallback CPU automático.

---

## 16. Feature Flags

| Flag | Función |
| ---- | ------- |
| `TTS_NOTIFY_COQUI_ENABLE_CLONING` | Activar/desactivar clonación. |
| `TTS_NOTIFY_COQUI_NOISE_REDUCTION` | Habilitar reducción de ruido. |
| `TTS_NOTIFY_COQUI_DIARIZATION` | Activar diarización de audio. |
| `TTS_NOTIFY_EXPERIMENTAL_FINE_TUNING` | Permitir rutas experimentales de fine-tuning. |

---

## 17. Plan de Pruebas

### Casos esenciales
1. CLI con `macos` (sin extras) ⇒ sin cambios en flujo base.
2. CLI con Coqui instalado ⇒ síntesis simple (`--engine coqui`).
3. Clonación con 1 sample muy corto ⇒ rechazo esperado.
4. Clonación con 2 samples válidos ⇒ creación, listado y uso del perfil.
5. Conversión WAV→MP3 ⇒ archivo generado y reproducible.
6. MCP `create_voice_profile` ⇒ confirmación JSON.
7. Estrés: 10 peticiones concurrentes (evaluar latencia y memoria).
8. Fallback GPU solicitado sin soporte.

### Scripts sugeridos
- `scripts/tests/test_coqui_engine.py`: pruebas unitarias/mocked.
- `scripts/tests/test_voice_profiles.py`: creación/listado.
- `scripts/tests/test_pipeline.py`: validar preprocesamiento.
- `scripts/benchmarks/benchmark_tts.py`: medir tiempos.

---

## 18. Checklist Técnico

### Fase A: Engine Multi-Idioma y Gestión de Modelos
- [ ] Extender `TTSConfig` con campos de idioma y gestión de modelos
- [ ] Añadir extras opcionales en `pyproject.toml` ([coqui], [coqui-gpu], [coqui-langs])
- [ ] Implementar `coqui_engine.py` con soporte multi-idioma y gestión automática de modelos
- [ ] Modificar bootstrap en `tts_engine.py` para registro condicional
- [ ] Actualizar CLI con flags `--engine`, `--model`, `--language`, `--list-languages`, etc.
- [ ] Extender MCP con herramientas de gestión de idiomas (list_languages, download_language, get_model_status)
- [ ] Implementar sistema de caché de modelos con integridad y limpieza
- [ ] Añadir detección automática de idiomas con forzado manual
- [ ] Validar fallback robusto a macOS engine
- [ ] Actualizar documentación de instalación (README-v3, guías multi-idioma)

### Fase B: Voice Cloning Multi-Idioma
- [ ] Implementar clonación (`voice_profile_manager.py`, embeddings, pipeline básico)
- [ ] Integrar perfiles multi-idioma en `VoiceManager`
- [ ] Añadir soporte para clonación por idioma específico
- [ ] Extender CLI con flags de clonación y perfiles por idioma
- [ ] Extender MCP con herramientas de perfiles multi-idioma
- [ ] Implementar validación de samples por idioma y calidad

### Fase C: Pipeline de Audio
- [ ] Crear `audio_pipeline.py` con plugins modularizados
- [ ] Implementar plugins de preprocesamiento (silence_trimmer, noise_reducer, normalizer)
- [ ] Crear `format_converter.py` para conversión multi-formato
- [ ] Integrar optimización por idioma en pipeline
- [ ] Añadir validación de calidad de audio por idioma

### Fase D: API REST y Telemetría
- [ ] (Opcional) Añadir endpoints API REST con soporte multi-idioma
- [ ] Añadir telemetría con métricas por idioma y modelo
- [ ] Implementar monitorización de recursos y uso de modelos
- [ ] Extender herramientas MCP con telemetría avanzada

### Fase E: Features Experimentales
- [ ] Implementar flags experimentales para fine-tuning multi-idioma
- [ ] Documentar y habilitar rutas de investigación futura

### Comunes a Todas las Fases
- [ ] Ejecutar pruebas comprehensivas de idiomas y modelos
- [ ] Validar rendimiento targets (carga, switch, memoria)
- [ ] Actualizar documentación completa (README-v3, VOICE_CLONING.md, MIGRATION-GUIDE-v3.md, LANGUAGE-GUIDE-v3.md)
- [ ] Documentar resultados y casos de uso reales
- [ ] Validar compatibilidad backward completa

---

## 19. Migración de v2 a v3

- Crear `MIGRATION-GUIDE-v3.md` con pasos:
  - **Para seguir usando macOS nativo**: sin cambios (100% compatible)
  - **Para Coqui básico**: `pip install .[coqui]`
  - **Para soporte multi-idioma completo**: `pip install .[coqui-langs]`
  - **Para GPU acceleration**: `pip install .[coqui-gpu]`
  - **Para clonación con diarización**: `pip install .[coqui,diarization]`
  - **Instalación completa**: `pip install .[all]`
  - **Nuevos comandos CLI/MCP** y ejemplos multi-idioma
  - **Configuración de idiomas** y gestión de modelos
- Mantener compatibilidad de argumentos existentes; **todos los nuevos flags son opt-in**.

### Novedades Principales v3.0.0
- **Soporte multi-idioma**: 17 idiomas con detección automática
- **Gestión inteligente de modelos**: Descarga automática y caché
- **Voice cloning**: Clonación de voz por idioma
- **Pipeline de audio modular**: Preprocesamiento y conversión de formatos
- **Performance mejorada**: 75% más rápido en detección de voces, 70% más rápido en startup

---

## 20. Roadmap Posterior

| Versión | Mejora |
| ------- | ------ |
| v3.1 | API REST formal y conversión avanzada (bitrate, sample rate). |
| v3.2 | Soporte para fine-tuning incremental (LoRA/adapter) bajo flag experimental. |
| v3.3 | Combinación de perfiles (voz + estilo). |
| v3.4 | Diarización robusta y detección automática de idioma. |
| v3.5 | Exportación/importación de perfiles y backups cifrados. |

---

## 21. Métricas de Éxito

- Latencia aceptable en síntesis Coqui (texto corto) ≤ 2.5× macOS.
- Perfiles clonados reutilizables sin errores en 95% de casos.
- Fallback seguro y documentado.
- Reporte de telemetría accesible y útil para diagnósticos.
- Usuarios pueden crear su perfil en < 5 minutos con audios adecuados.

---

## 22. Próximos Pasos Inmediatos (Fase A)

1. Añadir campos a `TTSConfig` y actualizar `config_manager`.
2. Crear `coqui_engine.py` y ajustar bootstrap de motores.
3. Añadir flag `--engine` y `--model` en CLI.
4. Verificar fallback cuando Coqui no está disponible.
5. Documentar las instrucciones de instalación de extras.

Una vez completado, avanzar a la Fase B (clonación) siguiendo los artefactos y módulos definidos.

---

**Plan creado por:** Equipo de Ingeniería TTS Notify  
**Versión del documento:** 1.0  
**Fecha:** *(actualizar al momento de guardar)*
