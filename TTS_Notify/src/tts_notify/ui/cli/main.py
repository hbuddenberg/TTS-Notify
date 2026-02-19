#!/usr/bin/env python3
"""
TTS Notify v2 - CLI Interface

Modular command-line interface for TTS Notify v2 using the new core architecture.
Maintains full feature parity with v1.5.0 while using the modular backend.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from ...core.config_manager import config_manager
from ...core.voice_system import VoiceManager, VoiceFilter
from ...core.tts_engine import engine_registry, bootstrap_engines
from ...core.models import (
    TTSRequest,
    AudioFormat,
    TTSEngineType,
    Voice,
    VoiceCloningRequest,
    VoiceCloningResponse,
)
from ...core.exceptions import (
    TTSNotifyError,
    VoiceNotFoundError,
    ValidationError,
    TTSError,
    EngineNotAvailableError,
)
from ...core.installer import coqui_installer
from ...utils.logger import setup_logging, get_logger


class TTSNotifyCLI:
    """Main CLI class for TTS Notify v2"""

    def __init__(self):
        self.config_manager = config_manager
        self.voice_manager = VoiceManager()
        self.logger = None
        self._engines_initialized = False

    async def initialize_engines(self):
        """Initialize TTS engines based on configuration"""
        if not self._engines_initialized:
            config = self.config_manager.get_config()
            await bootstrap_engines(config)
            self._engines_initialized = True
            if self.logger:
                self.logger.info("TTS engines initialized successfully")

    def setup_logging(self):
        """Setup logging based on configuration"""
        config = self.config_manager.get_config()
        setup_logging(level=getattr(config, "TTS_NOTIFY_LOG_LEVEL", "INFO"))
        self.logger = get_logger(__name__)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the command-line argument parser"""
        parser = argparse.ArgumentParser(
            prog="tts-notify",
            description="TTS Notify v2 - Sistema de notificaciones Text-to-Speech para macOS",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Ejemplos (v2.0.0 + v3.0.0 CoquiTTS):
  tts-notify "Hola mundo"
  tts-notify "Hola mundo" --voice monica --rate 200
  tts-notify "Hello world" --engine coqui --language en
  tts-notify "Bonjour le monde" --engine coqui --model xtts_v2 --language fr
  tts-notify --list
  tts-notify --list-languages              # Listar idiomas CoquiTTS
  tts-notify --download-language fr        # Descargar idioma frances
  tts-notify --model-status                # Estado de modelos CoquiTTS
  tts-notify --list --compact
  tts-notify --list --gen female
  tts-notify --list --lang es_ES
  tts-notify "Test" --save output_file
  tts-notify --mcp-config                 # Mostrar configuracion MCP para Claude Desktop

Para busqueda flexible de voces:
  tts-notify "Test" --voice angelica     # Encuentra Angelica
  tts-notify "Test" --voice "jorge enhanced"  # Variante Enhanced
  tts-notify "Test" --voice siri         # Siri si esta instalada

CoquiTTS (v3.0.0+):
  tts-notify "Hello" --engine coqui --voice Daniel --language en
  tts-notify "Hola" --engine coqui --language es --force-language

XTTS v2 Emotions (FASE 3):
  tts-notify "Great news!" --xtts --emotion happy
  tts-notify "Urgent alert" --xtts --emotion urgent
  tts-notify "Calm down" --xtts --emotion calm --temperature 0.3
  tts-notify "Save with emotion" --xtts --emotion sad --save output.wav

Voice Cloning (Phase B):
  tts-notify --clone-voice sample.wav --voice-name MiVoz --clone-language es
  tts-notify --list-cloned                        # Listar voces clonadas
  tts-notify --cloning-status                     # Estado del sistema
  tts-notify --delete-clone VOICE_ID              # Eliminar voz clonada
  tts-notify --enable-cloning                     # Activar clonacion

Audio Pipeline (Phase C):
  tts-notify --pipeline-status                    # Estado del pipeline
  tts-notify --process-audio input.wav --output-format mp3
  tts-notify --process-audio input.wav --target-language es --audio-quality ultra
  tts-notify --enable-pipeline                     # Activar pipeline

Installation & Testing:
  tts-notify --install-coqui                      # Instalar CoquiTTS
  tts-notify --install-coqui-gpu                   # Instalar CoquiTTS con GPU
  tts-notify --install-all                          # Instalar todo (CoquiTTS + deps + FFmpeg)
  tts-notify --test-installation                  # Probar instalacion completa
  tts-notify --installation-status               # Estado de la instalacion
            """,
        )

        # Text argument
        parser.add_argument("text", nargs="?", help="Texto a reproducir en voz alta")

        # Voice options
        parser.add_argument(
            "--voice",
            "-v",
            help="Voz a utilizar (búsqueda flexible: exacta, parcial, sin acentos)",
        )
        parser.add_argument(
            "--rate",
            "-r",
            type=int,
            help="Velocidad de habla (palabras por minuto, 100-300)",
        )
        parser.add_argument(
            "--pitch", type=float, help="Tono de la voz (0.5-2.0, donde 1.0 es normal)"
        )
        parser.add_argument(
            "--volume", type=float, help="Volumen (0.0-1.0, donde 1.0 es máximo)"
        )

        # Engine selection (v3.0.0+)
        parser.add_argument(
            "--engine",
            "-e",
            choices=["macos", "coqui"],
            help="Motor TTS a utilizar (default: configurado en TTS_NOTIFY_ENGINE)",
        )
        parser.add_argument(
            "--model", "-m", help="Modelo CoquiTTS específico (default: xtts_v2)"
        )
        parser.add_argument(
            "--language",
            "-L",
            choices=[
                "auto",
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "nl",
                "pl",
                "ru",
                "zh",
                "ja",
                "ko",
            ],
            help="Idioma para CoquiTTS (auto=detección automática)",
        )
        parser.add_argument(
            "--force-language",
            action="store_true",
            help="Forzar idioma específico ignorando detección automática",
        )

        # XTTS v2 Emotion and Style (FASE 3)
        parser.add_argument(
            "--emotion",
            "-E",
            choices=["neutral", "happy", "sad", "urgent", "calm"],
            help="Emocion para XTTS v2 (neutral, happy, sad, urgent, calm)",
        )
        parser.add_argument(
            "--temperature",
            "-T",
            type=float,
            help="Temperatura para XTTS v2 (0.1-1.0, controla variabilidad)",
        )
        parser.add_argument(
            "--xtts",
            action="store_true",
            help="Usar motor XTTS v2 explicitamente (alias para --engine coqui)",
        )

        # CoquiTTS management commands (v3.0.0+)
        parser.add_argument(
            "--list-languages",
            action="store_true",
            help="Listar idiomas disponibles en CoquiTTS",
        )
        parser.add_argument(
            "--download-language",
            metavar="LANG",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "nl",
                "pl",
                "ru",
                "zh",
                "ja",
                "ko",
            ],
            help="Descargar modelo para idioma específico de CoquiTTS",
        )
        parser.add_argument(
            "--model-status",
            action="store_true",
            help="Mostrar estado de los modelos CoquiTTS",
        )

        # Phase B: Voice Cloning Commands
        parser.add_argument(
            "--clone-voice",
            metavar="AUDIO_FILE",
            help="Clonar voz desde archivo de audio (Ej: --clone-voice sample.wav --voice-name MiVoz)",
        )
        parser.add_argument(
            "--voice-name",
            metavar="NAME",
            help="Nombre para la voz clonada (requerido con --clone-voice)",
        )
        parser.add_argument(
            "--clone-language",
            metavar="LANG",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "nl",
                "pl",
                "ru",
                "zh",
                "ja",
                "ko",
            ],
            help="Idioma para clonación de voz",
        )
        parser.add_argument(
            "--clone-quality",
            choices=["low", "medium", "high", "ultra"],
            default="high",
            help="Calidad de clonación de voz (default: high)",
        )
        parser.add_argument(
            "--list-cloned", action="store_true", help="Listar voces clonadas"
        )
        parser.add_argument(
            "--delete-clone", metavar="VOICE_ID", help="Eliminar voz clonada específica"
        )
        parser.add_argument(
            "--cloning-status",
            action="store_true",
            help="Mostrar estado del sistema de clonación de voz",
        )
        parser.add_argument(
            "--max-sample-duration",
            type=float,
            help="Duración máxima de muestra de audio en segundos (0 para ilimitado)",
        )
        parser.add_argument(
            "--enable-cloning",
            action="store_true",
            help="Activar sistema de clonación de voz",
        )
        parser.add_argument(
            "--disable-cloning",
            action="store_true",
            help="Desactivar sistema de clonación de voz",
        )

        # Installation Commands
        parser.add_argument(
            "--install-coqui",
            action="store_true",
            help="Instalar CoquiTTS y dependencias básicas",
        )
        parser.add_argument(
            "--install-coqui-gpu",
            action="store_true",
            help="Instalar CoquiTTS con soporte GPU (requiere CUDA)",
        )
        parser.add_argument(
            "--install-all",
            action="store_true",
            help="Instalar todas las dependencias (CoquiTTS + audio + FFmpeg)",
        )
        parser.add_argument(
            "--install-all-gpu",
            action="store_true",
            help="Instalar todas las dependencias con soporte GPU",
        )
        parser.add_argument(
            "--install-deps",
            action="store_true",
            help="Instalar dependencias de audio (librosa, soundfile, etc.)",
        )
        parser.add_argument(
            "--test-installation",
            action="store_true",
            help="Probar instalación completa de CoquiTTS",
        )
        parser.add_argument(
            "--installation-status",
            action="store_true",
            help="Mostrar estado actual de instalación",
        )

        # Phase C: Audio Pipeline Commands
        parser.add_argument(
            "--pipeline-status",
            action="store_true",
            help="Mostrar estado del pipeline de audio",
        )
        parser.add_argument(
            "--enable-pipeline",
            action="store_true",
            help="Activar pipeline de procesamiento de audio",
        )
        parser.add_argument(
            "--disable-pipeline",
            action="store_true",
            help="Desactivar pipeline de procesamiento de audio",
        )
        parser.add_argument(
            "--process-audio",
            metavar="INPUT_FILE",
            help="Procesar archivo de audio con pipeline avanzado",
        )
        parser.add_argument(
            "--output-format",
            choices=["wav", "mp3", "ogg", "flac", "aiff"],
            help="Formato de salida para procesamiento de audio",
        )
        parser.add_argument(
            "--audio-quality",
            choices=["low", "medium", "high", "ultra"],
            default="high",
            help="Calidad de procesamiento de audio (default: high)",
        )
        parser.add_argument(
            "--target-language",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "nl",
                "pl",
                "ru",
                "zh",
                "ja",
                "ko",
            ],
            help="Idioma objetivo para optimización de audio",
        )

        # Listing options
        parser.add_argument(
            "--list",
            "-l",
            action="store_true",
            help="Listar todas las voces disponibles del sistema",
        )
        parser.add_argument(
            "--compact",
            action="store_true",
            help="Formato compacto para listar voces (solo nombres)",
        )
        parser.add_argument(
            "--gen",
            "--gender",
            choices=["male", "female"],
            help="Filtrar voces por género",
        )
        parser.add_argument(
            "--lang", help="Filtrar voces por idioma (ej: es, en, es_ES, es_MX)"
        )

        # Output options
        parser.add_argument(
            "--save", "-s", help="Guardar audio en archivo en lugar de reproducir"
        )
        parser.add_argument(
            "--format",
            choices=["aiff", "wav", "mp3", "ogg", "m4a", "flac"],
            default="aiff",
            help="Formato de audio para archivo de salida (default: aiff)",
        )

        # Configuration options
        parser.add_argument(
            "--profile", help="Usar perfil de configuración predefinido"
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Mostrar información detallada"
        )
        parser.add_argument(
            "--debug", action="store_true", help="Mostrar información de depuración"
        )

        # MCP configuration
        parser.add_argument(
            "--mcp-config",
            action="store_true",
            help="Mostrar configuración MCP para Claude Desktop con rutas reales",
        )

        # MCP Config Installation
        parser.add_argument(
            "--install-mcp",
            nargs="?",
            const=True,
            metavar="CLIENT",
            choices=["claude", "opencode", "zed", "cursor", "continue"],
            help="Generate MCP config for specified client (claude, opencode, zed, cursor, continue)",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            dest="all_clients",
            help="Generate configs for all supported clients (use with --install-mcp)",
        )
        parser.add_argument(
            "--write",
            nargs="?",
            const=True,
            metavar="PATH",
            dest="write_config",
            help="Write config to file (auto-detect location, or specify path)",
        )
        parser.add_argument(
            "--path",
            metavar="EXEC_PATH",
            dest="custom_exec_path",
            help="Custom executable path for tts-notify",
        )

        # Version
        parser.add_argument(
            "--version",
            action="version",
            version="%(prog)s 3.0.0 (Phase A - CoquiTTS Integration)",
        )

        return parser

    async def list_voices(
        self,
        compact: bool = False,
        gender: Optional[str] = None,
        language: Optional[str] = None,
        engine: Optional[str] = None,
    ) -> None:
        """List available voices with optional filtering"""
        try:
            # Initialize engines first
            await self.initialize_engines()

            if engine == "coqui":
                # For CoquiTTS, list available speakers for the current model
                await self._list_coqui_voices(language=language)
                return

            voices = await self.voice_manager.get_all_voices()

            # Apply filters
            if gender or language:
                voice_filter = VoiceFilter()
                voices = voice_filter.filter_voices(
                    voices, gender=gender, language=language
                )

            if not voices:
                print("No se encontraron voces con los filtros especificados.")
                return

            if compact:
                # Compact format - just names
                for voice in sorted(voices, key=lambda v: v.name):
                    print(voice.name)
            else:
                # Detailed format with categorization
                self._print_voices_categorized(voices)

        except TTSNotifyError as e:
            print(f"Error listando voces: {e}")
            sys.exit(1)

    def _print_voices_categorized(self, voices) -> None:
        """Print voices organized by category like v1.5.0"""
        # Categorize voices
        espanol = []
        enhanced = []
        siri = []
        other = []

        for voice in voices:
            voice_name_lower = voice.name.lower()

            if any(
                espanol_marker in voice_name_lower
                for espanol_marker in [
                    "espanol",
                    "español",
                    "spain",
                    "mexico",
                    "mexico",
                    "argentina",
                    "colombia",
                    "chile",
                ]
            ):
                espanol.append(voice)
            elif "siri" in voice_name_lower:
                siri.append(voice)
            elif "enhanced" in voice_name_lower or "premium" in voice_name_lower:
                enhanced.append(voice)
            else:
                other.append(voice)

        # Print by category
        if espanol:
            print("\n🇪🇸  VOCES ESPAÑOL:")
            for voice in sorted(espanol, key=lambda v: v.name):
                gender_symbol = (
                    "♂" if voice.gender and voice.gender.value == "male" else "♀"
                )
                quality = voice.quality.value if voice.quality else "basic"
                print(f"  {gender_symbol} {voice.name} ({quality})")

        if enhanced:
            print("\n✨ VOCES ENHANCED:")
            for voice in sorted(enhanced, key=lambda v: v.name):
                gender_symbol = (
                    "♂" if voice.gender and voice.gender.value == "male" else "♀"
                )
                quality = voice.quality.value if voice.quality else "enhanced"
                print(f"  {gender_symbol} {voice.name} ({quality})")

        if siri:
            print("\n🍎 VOCES SIRI:")
            for voice in sorted(siri, key=lambda v: v.name):
                gender_symbol = (
                    "♂" if voice.gender and voice.gender.value == "male" else "♀"
                )
                print(f"  {gender_symbol} {voice.name}")

        if other:
            print("\n🌍 OTRAS VOCES:")
            for voice in sorted(other, key=lambda v: v.name):
                gender_symbol = (
                    "♂" if voice.gender and voice.gender.value == "male" else "♀"
                )
                lang = voice.language.value if voice.language else "unknown"
                print(f"  {gender_symbol} {voice.name} ({lang})")

        print(f"\nTotal: {len(voices)} voces disponibles")

    async def _list_coqui_voices(self, language: Optional[str] = None) -> None:
        """List CoquiTTS available voices/speakers

        Args:
            language: Optional language code to filter voices (e.g., 'es', 'en', 'fr')
        """
        try:
            coqui_engine = engine_registry.get("coqui")
            if coqui_engine:
                voices = await coqui_engine.get_supported_voices()

                # Check if the current model is multi-lingual
                is_multi_lingual = False
                if hasattr(coqui_engine, "model_name"):
                    model_name = coqui_engine.model_name
                    # Check if it's one of the known multi-lingual models
                    if (
                        "multilingual" in model_name.lower()
                        or "xtts" in model_name.lower()
                    ):
                        is_multi_lingual = True
                        # Get supported languages from model info
                        if hasattr(coqui_engine, "_get_model_info"):
                            model_info = coqui_engine._get_model_info(model_name)
                            if model_info and hasattr(model_info, "languages"):
                                supported_langs = model_info.languages
                            else:
                                supported_langs = []
                        else:
                            supported_langs = []

                print("\n🤖 VOCES COQUITTS:")
                if language:
                    if is_multi_lingual:
                        # For multi-lingual models, check if the requested language is supported
                        if supported_langs and language in supported_langs:
                            print(
                                f"   Filtrado por idioma: {language} (todas las voces soportan este idioma)"
                            )
                        elif supported_langs and language not in supported_langs:
                            print(
                                f"   ⚠️  Idioma '{language}' no está soportado por este modelo"
                            )
                            print(
                                f"   Idiomas disponibles: {', '.join(supported_langs)}"
                            )
                            voices = []  # Clear voices to show none
                        else:
                            print(f"   Filtrado por idioma: {language}")
                    else:
                        # For single-language models, filter by the voice's assigned language
                        voices = [
                            v
                            for v in voices
                            if v.language and v.language.value == language
                        ]
                        print(f"   Filtrado por idioma: {language}")
                print("=" * 40)

                if voices:
                    for voice in sorted(voices, key=lambda v: v.name):
                        gender_symbol = (
                            "♂"
                            if voice.gender and voice.gender.value == "male"
                            else "♀"
                        )
                        lang = voice.language.value if voice.language else "unknown"
                        quality = voice.quality.value if voice.quality else "basic"
                        print(f"  {gender_symbol} {voice.name} ({lang}, {quality})")
                        if voice.description:
                            print(f"      {voice.description}")
                    print(f"\nTotal: {len(voices)} voces CoquiTTS")
                else:
                    print("  No se encontraron voces CoquiTTS disponibles")
                    print(
                        "  Ejecute: tts-notify --model-status para verificar el estado"
                    )
            else:
                print("❌ Motor CoquiTTS no disponible")
                print(
                    "   Instale con: pip install coqui-tts o configure TTS_NOTIFY_ENGINE=coqui"
                )

        except EngineNotAvailableError:
            print("❌ Motor CoquiTTS no disponible")
            print("   Instale con: pip install coqui-tts")
        except Exception as e:
            print(f"❌ Error listando voces CoquiTTS: {e}")

    async def list_languages(self) -> None:
        """List available CoquiTTS languages"""
        try:
            coqui_engine = engine_registry.get("coqui")
            if coqui_engine and hasattr(coqui_engine, "multi_language_models"):
                print("\n🌍 IDIOMAS DISPONIBLES EN COQUITTS:")
                print("=" * 50)

                current_model = coqui_engine.model_name
                model_info = coqui_engine.multi_language_models.get(current_model)

                if model_info:
                    print(f"\nModelo actual: {current_model}")
                    print(f"Idiomas soportados: {len(model_info.languages)}")
                    print(f"Calidad: {model_info.quality}")
                    print(f"Locutores: {model_info.speakers}")
                    print(f"Tamaño: ~{model_info.size_gb}GB")

                    print(f"\n📋 Idiomas disponibles:")
                    for i, lang in enumerate(model_info.languages, 1):
                        lang_names = {
                            "en": "Inglés",
                            "es": "Español",
                            "fr": "Francés",
                            "de": "Alemán",
                            "it": "Italiano",
                            "pt": "Portugués",
                            "nl": "Neerlandés",
                            "pl": "Polaco",
                            "ru": "Ruso",
                            "zh": "Chino",
                            "ja": "Japonés",
                            "ko": "Coreano",
                        }
                        lang_name = lang_names.get(lang, lang.upper())
                        print(f"  {i:2d}. {lang} ({lang_name})")
                else:
                    print("❌ Información del modelo no disponible")
            else:
                print("❌ Motor CoquiTTS no disponible o no soporta multi-idioma")

        except Exception as e:
            print(f"❌ Error listando idiomas: {e}")

    async def download_language(self, language: str) -> None:
        """Download CoquiTTS language model with progress bar"""
        try:
            print(f"📥 Descargando idioma '{language}' para CoquiTTS...")

            coqui_engine = engine_registry.get("coqui")
            if coqui_engine and hasattr(coqui_engine, "ensure_language_available"):
                model_info = None
                if hasattr(coqui_engine, "multi_language_models"):
                    for model_name, info in coqui_engine.multi_language_models.items():
                        if language in info.languages:
                            model_info = info
                            break

                if TQDM_AVAILABLE and model_info:
                    total_mb = int(model_info.size_gb * 1024)
                    with tqdm(
                        total=total_mb,
                        unit="MB",
                        desc=f"Descargando {language}",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}MB [{elapsed}<{remaining}]",
                    ) as pbar:
                        success = await coqui_engine.ensure_language_available(language)
                        if success:
                            pbar.update(total_mb)
                else:
                    success = await coqui_engine.ensure_language_available(language)

                if success:
                    print(f"✅ Idioma '{language}' descargado y disponible")
                else:
                    print(f"❌ Error descargando idioma '{language}'")
                    print("   Verifique su conexion a internet y espacio en disco")
            else:
                print("❌ Motor CoquiTTS no disponible")

        except Exception as e:
            print(f"❌ Error descargando idioma: {e}")

    async def show_model_status(self) -> None:
        """Show CoquiTTS model status"""
        try:
            config = self.config_manager.get_config()

            print("\n🤖 ESTADO DE MODELOS COQUITTS:")
            print("=" * 50)

            # Check CoquiTTS availability
            try:
                coqui_engine = engine_registry.get("coqui")
                if coqui_engine:
                    current_model = coqui_engine.model_name
                    model_info = coqui_engine.multi_language_models.get(current_model)

                    print(f"✅ Motor CoquiTTS disponible")
                    print(f"📍 Modelo actual: {current_model}")

                    if model_info:
                        print(f"📊 Estado del modelo:")
                        print(f"   • Idiomas: {len(model_info.languages)}")
                        print(f"   • Locutores: {model_info.speakers}")
                        print(f"   • Calidad: {model_info.quality}")
                        print(f"   • Tamaño: ~{model_info.size_gb}GB")
                        print(f"   • GPU: {'Sí' if coqui_engine.use_gpu else 'No'}")

                        # Check model download status
                        if hasattr(coqui_engine, "is_model_downloaded"):
                            downloaded = coqui_engine.is_model_downloaded()
                            print(f"   • Descargado: {'Sí' if downloaded else 'No'}")
                    else:
                        print("⚠️  Información del modelo no disponible")
                else:
                    print("❌ Motor CoquiTTS no registrado")

            except EngineNotAvailableError:
                print("❌ Motor CoquiTTS no disponible")

            # Configuration
            print(f"\n⚙️  Configuración:")
            print(f"   • Motor por defecto: {config.TTS_NOTIFY_ENGINE}")
            print(
                f"   • Auto-descargar modelos: {config.TTS_NOTIFY_AUTO_DOWNLOAD_MODELS}"
            )
            print(f"   • Modo offline: {config.TTS_NOTIFY_COQUI_OFFLINE_MODE}")
            print(f"   • Usar GPU: {config.TTS_NOTIFY_COQUI_USE_GPU}")
            print(f"   • Cache modelos: {config.TTS_NOTIFY_COQUI_CACHE_MODELS}")

            # Installation suggestion
            if not engine_registry.get("coqui"):
                print(f"\n💡 Instalación:")
                print(f"   pip install coqui-tts torchaudio soundfile")
                print(f"   O bien:")
                print(f"   pip install -e .[coqui]")

        except Exception as e:
            print(f"❌ Error obteniendo estado de modelos: {e}")

    # Phase B: Voice Cloning Methods

    async def clone_voice(
        self,
        audio_file: str,
        voice_name: str,
        language: Optional[str] = None,
        quality: str = "high",
        max_sample_duration: Optional[float] = None,
    ) -> None:
        """Clone a voice from audio file"""
        try:
            await self.initialize_engines()

            # Update configuration if max_sample_duration is provided
            if max_sample_duration is not None:
                self.config_manager.get_config().TTS_NOTIFY_COQUI_MAX_SAMPLE_SECONDS = (
                    max_sample_duration
                )

            config = self.config_manager.get_config()

            # Check if CoquiTTS is available
            coqui_engine = engine_registry.get("coqui")
            if not coqui_engine:
                print("❌ Motor CoquiTTS no disponible para clonación")
                print("   Instale con: pip install coqui-tts")
                sys.exit(1)

            # Validate audio file - check multiple locations
            audio_path = Path(audio_file)
            project_root = Path(__file__).parent.parent.parent.parent.parent.parent
            search_paths = [
                audio_path,
                Path.cwd() / audio_file,
                project_root / audio_file,
            ]

            found_path = None
            for search_path in search_paths:
                if search_path.exists():
                    found_path = search_path.resolve()
                    break

            if not found_path:
                print(f"❌ Archivo de audio no encontrado: {audio_file}")
                print(f"   Buscado en:")
                for sp in search_paths:
                    print(f"   • {sp.resolve()}")
                sys.exit(1)

            audio_path = found_path

            # Determine language
            clone_language = language or config.TTS_NOTIFY_DEFAULT_LANGUAGE or "es"

            print(f"🎭 Iniciando clonación de voz...")
            print(f"   Archivo: {audio_path}")
            print(f"   Nombre: {voice_name}")
            print(f"   Idioma: {clone_language}")
            print(f"   Calidad: {quality}")

            # Create cloning request
            cloning_request = VoiceCloningRequest(
                source_audio_path=audio_path,
                voice_name=voice_name,
                language=clone_language,
                quality=quality,
                normalize=config.TTS_NOTIFY_COQUI_CLONING_NORMALIZE,
                denoise=config.TTS_NOTIFY_COQUI_CLONING_DENOISE,
                auto_optimize=config.TTS_NOTIFY_COQUI_AUTO_OPTIMIZE_CLONE,
                batch_size=config.TTS_NOTIFY_COQUI_CLONING_BATCH_SIZE,
                timeout=config.TTS_NOTIFY_COQUI_CLONING_TIMEOUT,
            )

            # Validate request
            if hasattr(coqui_engine, "validate_cloning_request"):
                errors = await coqui_engine.validate_cloning_request(cloning_request)
                if errors:
                    print("❌ Errores de validación:")
                    for error in errors:
                        print(f"   • {error}")
                    sys.exit(1)

            # Perform cloning
            response = await coqui_engine.clone_voice(cloning_request)

            if response.success:
                print(f"✅ Voz clonada exitosamente:")
                print(f"   🎤 Nombre: {response.voice.name}")
                print(f"   🆔 ID: {response.voice.id}")
                print(f"   🌍 Idioma: {response.voice.cloning_language}")
                print(f"   ⭐ Calidad: {response.voice.cloning_quality}")
                print(f"   📊 Duración muestra: {response.sample_duration:.2f}s")

                if response.optimization_score:
                    print(f"   🎯 Optimización: {response.optimization_score:.2f}")

                if response.processing_time:
                    print(
                        f"   ⏱️  Tiempo procesamiento: {response.processing_time:.2f}s"
                    )

                print(f"\n💡 Usa la voz clonada con:")
                print(
                    f'   tts-notify "Hola mundo" --engine coqui --voice "{response.voice.name}"'
                )
            else:
                print(f"❌ Error en clonación: {response.error}")
                if response.warnings:
                    print("Advertencias:")
                    for warning in response.warnings:
                        print(f"   ⚠️  {warning}")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error en clonación de voz: {e}")
            sys.exit(1)

    async def list_cloned_voices(self) -> None:
        """List all cloned voices"""
        try:
            await self.initialize_engines()

            coqui_engine = engine_registry.get("coqui")
            if not coqui_engine:
                print("❌ Motor CoquiTTS no disponible")
                return

            cloned_voices = await coqui_engine.get_cloned_voices()

            print("\n🎭 VOCES CLONADAS:")
            print("=" * 50)

            if not cloned_voices:
                print("   No hay voces clonadas disponibles")
                print("\n💡 Clona una voz con:")
                print(
                    "   tts-notify --clone-voice sample.wav --voice-name MiVoz --clone-language es"
                )
                return

            for i, voice in enumerate(cloned_voices, 1):
                created_at = voice.created_at or "Desconocido"
                quality = voice.cloning_quality or "Desconocida"
                language = voice.cloning_language or "Desconocido"
                score = voice.optimization_score

                print(f"\n{i:2d}. 🎤 {voice.name}")
                print(f"     🆔 ID: {voice.id}")
                print(f"     🌍 Idioma: {language}")
                print(f"     ⭐ Calidad: {quality}")
                print(f"     📅 Creada: {created_at}")

                if score:
                    score_percent = score * 100
                    print(f"     🎯 Optimización: {score_percent:.1f}%")

                if voice.cloning_source:
                    source_file = Path(voice.cloning_source).name
                    print(f"     📁 Fuente: {source_file}")

            print(f"\nTotal: {len(cloned_voices)} voces clonadas")

        except Exception as e:
            print(f"❌ Error listando voces clonadas: {e}")

    async def delete_cloned_voice(self, voice_id: str) -> None:
        """Delete a cloned voice"""
        try:
            await self.initialize_engines()

            coqui_engine = engine_registry.get("coqui")
            if not coqui_engine:
                print("❌ Motor CoquiTTS no disponible")
                return

            print(f"🗑️  Eliminando voz clonada: {voice_id}")

            success = await coqui_engine.delete_cloned_voice(voice_id)

            if success:
                print(f"✅ Voz clonada eliminada exitosamente")
            else:
                print(f"❌ No se encontró la voz clonada: {voice_id}")
                print("\n💡 Lista las voces disponibles con:")
                print("   tts-notify --list-cloned")

        except Exception as e:
            print(f"❌ Error eliminando voz clonada: {e}")

    async def show_cloning_status(self) -> None:
        """Show voice cloning system status"""
        try:
            await self.initialize_engines()

            coqui_engine = engine_registry.get("coqui")
            config = self.config_manager.get_config()

            print("\n🎭 ESTADO DEL SISTEMA DE CLONACIÓN DE VOZ:")
            print("=" * 60)

            # Basic configuration
            print(f"\n⚙️  Configuración:")
            print(
                f"   • Clonación habilitada: {config.TTS_NOTIFY_COQUI_ENABLE_CLONING}"
            )
            print(
                f"   • Calidad por defecto: {config.TTS_NOTIFY_COQUI_CLONING_QUALITY}"
            )
            print(
                f"   • Auto-optimización: {config.TTS_NOTIFY_COQUI_AUTO_OPTIMIZE_CLONE}"
            )
            print(f"   • Normalización: {config.TTS_NOTIFY_COQUI_CLONING_NORMALIZE}")
            print(f"   • Reducción de ruido: {config.TTS_NOTIFY_COQUI_CLONING_DENOISE}")
            print(f"   • Tamaño batch: {config.TTS_NOTIFY_COQUI_CLONING_BATCH_SIZE}")
            print(f"   • Timeout: {config.TTS_NOTIFY_COQUI_CLONING_TIMEOUT}s")

            # Engine status
            if coqui_engine:
                cloning_status = await coqui_engine.get_cloning_status()

                print(f"\n🤖 Motor CoquiTTS:")
                print(f"   • Disponible: ✅")
                print(
                    f"   • Inicializado: {cloning_status.get('engine_initialized', False)}"
                )
                print(
                    f"   • Soporta clonación: {cloning_status.get('supports_cloning', False)}"
                )

                if "profile_count" in cloning_status:
                    print(f"\n📊 Estadísticas:")
                    print(f"   • Voces clonadas: {cloning_status['profile_count']}")
                    print(
                        f"   • Embeddings: {cloning_status.get('embedding_count', 0)}"
                    )

                if "storage_usage" in cloning_status:
                    storage = cloning_status["storage_usage"]
                    print(f"\n💾 Almacenamiento:")
                    print(f"   • Perfiles: {storage.get('profiles_mb', 0):.1f} MB")
                    print(f"   • Embeddings: {storage.get('embeddings_mb', 0):.1f} MB")
                    print(f"   • Total: {storage.get('total_mb', 0):.1f} MB")

                if "directories" in cloning_status:
                    dirs = cloning_status["directories"]
                    print(f"\n📁 Directorios:")
                    print(f"   • Perfiles: {dirs.get('profiles', 'N/A')}")
                    print(f"   • Embeddings: {dirs.get('embeddings', 'N/A')}")
                    print(f"   • Temporal: {dirs.get('temp', 'N/A')}")
            else:
                print(f"\n🤖 Motor CoquiTTS:")
                print(f"   • Disponible: ❌")
                print(f"   • Instalación: pip install coqui-tts")

            # Instructions
            if not config.TTS_NOTIFY_COQUI_ENABLE_CLONING:
                print(f"\n💡 Para habilitar la clonación:")
                print(f"   export TTS_NOTIFY_COQUI_ENABLE_CLONING=true")
                print(f"   O usa: tts-notify --enable-cloning")

        except Exception as e:
            print(f"❌ Error obteniendo estado de clonación: {e}")

    async def toggle_cloning(self, enable: bool) -> None:
        """Enable or disable voice cloning"""
        try:
            # This would require updating the configuration
            # For now, we'll show instructions
            action = "habilitar" if enable else "deshabilitar"
            env_var = "TTS_NOTIFY_COQUI_ENABLE_CLONING"
            value = "true" if enable else "false"

            print(f"\n⚙️  Para {action} la clonación de voz:")
            print(f"   export {env_var}={value}")
            print(f"\nO añádelo a tu archivo ~/.bashrc o ~/.zshrc:")
            print(f"   echo 'export {env_var}={value}' >> ~/.bashrc")

            if enable:
                print(f"\nReinicia tu terminal o ejecuta:")
                print(f"   source ~/.bashrc")
                print(f"\nLuego verifica el estado con:")
                print(f"   tts-notify --cloning-status")

        except Exception as e:
            print(f"❌ Error configurando clonación: {e}")

    # Phase C: Audio Pipeline Methods

    async def process_audio_file(
        self,
        input_file: str,
        output_format: Optional[str] = None,
        target_language: Optional[str] = None,
        audio_quality: str = "high",
    ) -> None:
        """Process audio file through the advanced pipeline"""
        try:
            await self.initialize_engines()

            config = self.config_manager.get_config()

            # Check if CoquiTTS is available
            coqui_engine = engine_registry.get("coqui")
            if not coqui_engine:
                print("❌ Motor CoquiTTS no disponible para procesamiento de audio")
                print("   Instale con: pip install coqui-tts librosa soundfile")
                sys.exit(1)

            # Validate input file - check multiple locations
            input_path = Path(input_file)
            project_root = Path(__file__).parent.parent.parent.parent.parent.parent
            search_paths = [
                input_path,
                Path.cwd() / input_file,
                project_root / input_file,
            ]

            found_path = None
            for search_path in search_paths:
                if search_path.exists():
                    found_path = search_path.resolve()
                    break

            if not found_path:
                print(f"❌ Archivo de audio no encontrado: {input_file}")
                print(f"   Buscado en:")
                for sp in search_paths:
                    print(f"   • {sp.resolve()}")
                sys.exit(1)

            input_path = found_path

            # Determine source and target formats
            source_format = AudioFormat(input_path.suffix.lower().lstrip("."))
            target_format_enum = AudioFormat(output_format or source_format.value)

            # Read input audio
            print(f"🔧 Iniciando procesamiento de audio...")
            print(f"   Entrada: {input_path}")
            print(f"   Formato salida: {target_format_enum.value}")

            if target_language:
                print(f"   Idioma objetivo: {target_language}")
            print(f"   Calidad: {audio_quality}")

            input_data = input_path.read_bytes()

            # Process through pipeline
            processed_data, metrics = await coqui_engine.process_audio_pipeline(
                audio_data=input_data,
                source_format=source_format,
                target_format=target_format_enum,
                language=target_language,
                quality_level=audio_quality,
            )

            # Generate output filename
            output_path = input_path.with_suffix(f".{target_format_enum.value}")

            # Save processed audio
            output_path.write_bytes(processed_data)

            # Display results
            print(f"\n✅ Audio procesado exitosamente:")
            print(f"   📁 Salida: {output_path}")
            print(
                f"   📊 Tamaño original: {metrics.get('input_size_bytes', 0):,} bytes"
            )
            print(
                f"   📊 Tamaño procesado: {metrics.get('output_size_bytes', 0):,} bytes"
            )

            if metrics.get("processing_time"):
                print(f"   ⏱️  Tiempo procesamiento: {metrics['processing_time']:.2f}s")

            if metrics.get("compression_ratio"):
                print(f"   📈 Compresión: {metrics['compression_ratio']:.2f}x")

            if metrics.get("peak_level_dbfs"):
                print(f"   🔊 Nivel pico: {metrics['peak_level_dbfs']:.1f} dBFS")

            if metrics.get("rms_level_dbfs"):
                print(f"   🔊 Nivel RMS: {metrics['rms_level_dbfs']:.1f} dBFS")

            stages = metrics.get("stages_completed", [])
            if stages:
                print(f"   🔄 Etapas completadas: {len(stages)}")

            warnings = metrics.get("warnings", [])
            if warnings:
                print(f"   ⚠️  Advertencias: {len(warnings)}")
                for warning in warnings[:3]:  # Show first 3 warnings
                    print(f"      • {warning}")

            print(f"\n💡 Para más opciones de procesamiento:")
            print(f"   tts-notify --pipeline-status")

        except Exception as e:
            print(f"❌ Error procesando audio: {e}")
            sys.exit(1)

    async def show_pipeline_status(self) -> None:
        """Show audio pipeline system status"""
        try:
            await self.initialize_engines()

            coqui_engine = engine_registry.get("coqui")
            config = self.config_manager.get_config()

            print("\n🔧 ESTADO DEL PIPELINE DE AUDIO:")
            print("=" * 50)

            # Configuration
            print(f"\n⚙️  Configuración:")
            print(
                f"   • Pipeline habilitado: {config.TTS_NOTIFY_COQUI_CONVERSION_ENABLED}"
            )
            print(
                f"   • Limpieza automática: {config.TTS_NOTIFY_COQUI_AUTO_CLEAN_AUDIO}"
            )
            print(
                f"   • Recorte de silencios: {config.TTS_NOTIFY_COQUI_AUTO_TRIM_SILENCE}"
            )
            print(f"   • Reducción de ruido: {config.TTS_NOTIFY_COQUI_NOISE_REDUCTION}")
            print(f"   • Diarización: {config.TTS_NOTIFY_COQUI_DIARIZATION}")
            print(f"   • Formatos objetivo: {config.TTS_NOTIFY_COQUI_TARGET_FORMATS}")
            print(
                f"   • Cache de embeddings: {config.TTS_NOTIFY_COQUI_EMBEDDING_CACHE}"
            )
            print(
                f"   • Formato embeddings: {config.TTS_NOTIFY_COQUI_EMBEDDING_FORMAT}"
            )

            # Engine capabilities
            if coqui_engine:
                try:
                    capabilities = await coqui_engine.get_pipeline_capabilities()

                    print(f"\n🤖 Capacidades del Motor:")
                    print(f"   • Motor disponible: ✅")
                    print(
                        f"   • Procesamiento de audio: {'✅' if capabilities.get('audio_processing_available') else '❌'}"
                    )
                    print(
                        f"   • FFmpeg disponible: {'✅' if capabilities.get('ffmpeg_available') else '❌'}"
                    )
                    print(
                        f"   • Idiomas soportados: {len(capabilities.get('supported_languages', []))}"
                    )

                    supported_formats = capabilities.get("supported_formats", [])
                    if supported_formats:
                        print(
                            f"   • Formatos soportados: {', '.join(supported_formats)}"
                        )

                    stages = capabilities.get("processing_stages", [])
                    if stages:
                        print(f"   • Etapas de procesamiento: {len(stages)}")

                    temp_dir = capabilities.get("temp_directory")
                    if temp_dir:
                        print(f"   • Directorio temporal: {temp_dir}")

                except Exception as e:
                    print(f"\n🤖 Motor CoquiTTS:")
                    print(f"   • Error obteniendo capacidades: {e}")
            else:
                print(f"\n🤖 Motor CoquiTTS:")
                print(f"   • Disponible: ❌")
                print(f"   • Instalación: pip install coqui-tts librosa soundfile")

            # Instructions for enabling
            if not config.TTS_NOTIFY_COQUI_CONVERSION_ENABLED:
                print(f"\n💡 Para habilitar el pipeline de audio:")
                print(f"   export TTS_NOTIFY_COQUI_CONVERSION_ENABLED=true")
                print(f"   O usa: tts-notify --enable-pipeline")

        except Exception as e:
            print(f"❌ Error obteniendo estado del pipeline: {e}")

    async def toggle_pipeline(self, enable: bool) -> None:
        """Enable or disable audio pipeline"""
        try:
            action = "habilitar" if enable else "deshabilitar"
            env_var = "TTS_NOTIFY_COQUI_CONVERSION_ENABLED"
            value = "true" if enable else "false"

            print(f"\n⚙️  Para {action} el pipeline de audio:")
            print(f"   export {env_var}={value}")
            print(f"\nO añádelo a tu archivo ~/.bashrc o ~/.zshrc:")
            print(f"   echo 'export {env_var}={value}' >> ~/.bashrc")

            if enable:
                print(f"\nInstala las dependencias adicionales:")
                print(f"   pip install librosa soundfile ffmpeg-python")
                print(f"\nReinicia tu terminal o ejecuta:")
                print(f"   source ~/.bashrc")
                print(f"\nLuego verifica el estado con:")
                print(f"   tts-notify --pipeline-status")

        except Exception as e:
            print(f"❌ Error configurando pipeline: {e}")

    # Installation & Testing Methods

    async def install_coqui_tts(self, use_gpu: bool = False) -> None:
        """Install CoquiTTS automatically"""
        try:
            print(f"🚀 Iniciando instalación de CoquiTTS...")
            print(f"   GPU: {'Sí' if use_gpu else 'No'}")

            result = await coqui_installer.install_coqui_tts(use_gpu=use_gpu)

            if result.success:
                print(f"\n✅ CoquiTTS instalado exitosamente!")
                print(f"   Componente: {result.component}")
                print(f"   Versión: {result.version}")
                print(f"   Ubicación: {result.installed_path}")

                print(f"\n💡 Ahora puedes usar el motor CoquiTTS:")
                print(f'   tts-notify --engine coqui "Hola mundo"')
                print(f"   tts-notify --list-languages")

                # Re-initialize engines to pick up CoquiTTS
                await self.initialize_engines()
                self._engines_initialized = False  # Force re-initialization
                await self.initialize_engines()

                # Interactive language selection
                try:
                    coqui_engine = engine_registry.get("coqui")
                    if coqui_engine:
                        print(f"\n🌍 Opciones de Modelos de Voz:")
                        print("   1. Modelo Multi-lenguaje (Recomendado)")
                        print("      • Soporta 17 idiomas (en, es, fr, de, etc.)")
                        print("      • Mejor calidad y clonación de voz")
                        print("      • Tamaño: ~2.0 GB")
                        print("   2. Modelo de Idioma Específico")
                        print("      • Solo un idioma")
                        print("      • Menor calidad, sin clonación")
                        print("      • Tamaño: ~100-500 MB")
                        print("   3. Omitir descarga por ahora")

                        choice = input("\n   Seleccione una opción [1-3]: ").strip()

                        if choice == "1":
                            print(
                                f"\n⬇️  Iniciando descarga del modelo multi-lenguaje..."
                            )
                            model_info = coqui_engine.multi_language_models.get(
                                coqui_engine.model_name
                            )
                            if TQDM_AVAILABLE and model_info:
                                total_mb = int(model_info.size_gb * 1024)
                                with tqdm(
                                    total=total_mb,
                                    unit="MB",
                                    desc="Modelo XTTS",
                                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}MB [{elapsed}<{remaining}]",
                                ) as pbar:
                                    success = await coqui_engine.download_model(
                                        force=True
                                    )
                                    if success:
                                        pbar.update(total_mb)
                            else:
                                success = await coqui_engine.download_model(force=True)
                            if success:
                                print(f"✅ Modelo descargado exitosamente!")
                            else:
                                print(f"❌ Error en la descarga del modelo.")

                        elif choice == "2":
                            single_models = coqui_engine.get_single_language_models()
                            available_langs = sorted(single_models.keys())

                            print(f"\n🗣️  Idiomas disponibles:")
                            for i, lang in enumerate(available_langs, 1):
                                print(f"   {i}. {lang}")

                            lang_idx = input(
                                f"\n   Seleccione idioma [1-{len(available_langs)}]: "
                            ).strip()

                            try:
                                idx = int(lang_idx) - 1
                                if 0 <= idx < len(available_langs):
                                    selected_lang = available_langs[idx]
                                    models = single_models[selected_lang]

                                    success = False
                                    for model_name in models:
                                        print(
                                            f"\n⬇️  Intentando descargar modelo: {model_name}..."
                                        )
                                        if await coqui_engine.download_model(
                                            model_name=model_name, force=True
                                        ):
                                            print(
                                                f"✅ Modelo para {selected_lang} descargado exitosamente!"
                                            )
                                            print(f"💡 Para usar este modelo:")
                                            print(
                                                f'   tts-notify --engine coqui --model "{model_name}" ...'
                                            )
                                            success = True
                                            break
                                        else:
                                            print(
                                                f"⚠️  Fallo la descarga de {model_name}, intentando siguiente..."
                                            )

                                    if not success:
                                        print(
                                            f"❌ No se pudo descargar ningún modelo para {selected_lang}."
                                        )
                                else:
                                    print("❌ Selección inválida.")
                            except ValueError:
                                print("❌ Entrada inválida.")

                        elif choice == "3":
                            print("\nℹ️  Omitiendo descarga de modelos adicionales.")

                        else:
                            print(
                                f"\n❌ Opción no válida: '{choice}'. Seleccione 1, 2 o 3."
                            )

                except Exception as e:
                    print(f"⚠️  Error en selección de idiomas: {e}")

            else:
                print(f"\n❌ Error instalando CoquiTTS: {result.error}")
                if result.warnings:
                    print("\nAdvertencias:")
                    for warning in result.warnings:
                        print(f"   ⚠️  {warning}")

        except Exception as e:
            print(f"❌ Error en instalación: {e}")
            sys.exit(1)

    async def install_all_dependencies(
        self, use_gpu: bool = False, include_ffmpeg: bool = False
    ) -> None:
        """Install all dependencies for full functionality"""
        try:
            print(f"🚀 Iniciando instalación completa de dependencias...")
            print(f"   GPU: {'Sí' if use_gpu else 'No'}")
            print(f"   FFmpeg: {'Sí' if include_ffmpeg else 'No'}")

            results = await coqui_installer.install_all_dependencies(
                use_gpu=use_gpu, include_ffmpeg=include_ffmpeg
            )

            # Display results
            print(f"\n📊 Resultados de la instalación:")
            all_success = True

            for component, result in results.items():
                status = "✅" if result.success else "❌"
                print(f"   {status} {component}: ", end="")

                if result.success:
                    if result.version:
                        print(f"v{result.version} ({result.installed_path})")
                    else:
                        print("Instalado")
                else:
                    print(f"Error - {result.error}")
                    all_success = False

            if all_success:
                print(f"\n🎉 ¡Todas las dependencias instaladas exitosamente!")
                print(
                    f"\n⚠️  IMPORTANTE: Debes activar el entorno virtual para usar CoquiTTS:"
                )
                print(f"   source TTS_Notify/venv312/bin/activate")
                print(f"\n💡 Sistema listo para uso completo (una vez activado):")
                print(f'   tts-notify --engine coqui "Hello world"')
                print(f"   tts-notify --pipeline-status")
                print(f"   tts-notify --clone-voice sample.wav --voice-name MiVoz")

                # Re-initialize engines
                await self.initialize_engines()
                self._engines_initialized = False  # Force re-initialization
                await self.initialize_engines()

                # Interactive language selection
                try:
                    coqui_engine = engine_registry.get("coqui")
                    if coqui_engine:
                        print(f"\n🌍 Opciones de Modelos de Voz:")
                        print("   1. Modelo Multi-lenguaje (Recomendado)")
                        print("      • Soporta 17 idiomas (en, es, fr, de, etc.)")
                        print("      • Mejor calidad y clonación de voz")
                        print("      • Tamaño: ~2.0 GB")
                        print("   2. Modelo de Idioma Específico")
                        print("      • Solo un idioma")
                        print("      • Menor calidad, sin clonación")
                        print("      • Tamaño: ~100-500 MB")
                        print("   3. Omitir descarga por ahora")

                        choice = input("\n   Seleccione una opción [1-3]: ").strip()

                        if choice == "1":
                            print(
                                f"\n⬇️  Iniciando descarga del modelo multi-lenguaje..."
                            )
                            model_info = coqui_engine.multi_language_models.get(
                                coqui_engine.model_name
                            )
                            if TQDM_AVAILABLE and model_info:
                                total_mb = int(model_info.size_gb * 1024)
                                with tqdm(
                                    total=total_mb,
                                    unit="MB",
                                    desc="Modelo XTTS",
                                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}MB [{elapsed}<{remaining}]",
                                ) as pbar:
                                    success = await coqui_engine.download_model(
                                        force=True
                                    )
                                    if success:
                                        pbar.update(total_mb)
                            else:
                                success = await coqui_engine.download_model(force=True)
                            if success:
                                print(f"✅ Modelo descargado exitosamente!")
                            else:
                                print(f"❌ Error en la descarga del modelo.")

                        elif choice == "2":
                            single_models = coqui_engine.get_single_language_models()
                            available_langs = sorted(single_models.keys())

                            print(f"\n🗣️  Idiomas disponibles:")
                            for i, lang in enumerate(available_langs, 1):
                                print(f"   {i}. {lang}")

                            lang_idx = input(
                                f"\n   Seleccione idioma [1-{len(available_langs)}]: "
                            ).strip()

                            try:
                                idx = int(lang_idx) - 1
                                if 0 <= idx < len(available_langs):
                                    selected_lang = available_langs[idx]
                                    models = single_models[selected_lang]

                                    success = False
                                    for model_name in models:
                                        print(
                                            f"\n⬇️  Intentando descargar modelo: {model_name}..."
                                        )
                                        if await coqui_engine.download_model(
                                            model_name=model_name, force=True
                                        ):
                                            print(
                                                f"✅ Modelo para {selected_lang} descargado exitosamente!"
                                            )
                                            print(f"💡 Para usar este modelo:")
                                            print(
                                                f'   tts-notify --engine coqui --model "{model_name}" ...'
                                            )
                                            success = True
                                            break
                                        else:
                                            print(
                                                f"⚠️  Fallo la descarga de {model_name}, intentando siguiente..."
                                            )

                                    if not success:
                                        print(
                                            f"❌ No se pudo descargar ningún modelo para {selected_lang}."
                                        )
                                else:
                                    print("❌ Selección inválida.")
                            except ValueError:
                                print("❌ Entrada inválida.")

                        elif choice == "3":
                            print("\nℹ️  Omitiendo descarga de modelos adicionales.")

                        else:
                            print(
                                f"\n❌ Opción no válida: '{choice}'. Seleccione 1, 2 o 3."
                            )
                except Exception as e:
                    print(f"⚠️  Error en selección de idiomas: {e}")

            else:
                print(f"\n⚠️  Algunas dependencias fallaron. Revisa los errores arriba.")
                print(f"   Puedes instalar individualmente con:")
                print(f"   tts-notify --install-coqui")
                print(f"   tts-notify --install-deps")

        except Exception as e:
            print(f"❌ Error en instalación completa: {e}")
            sys.exit(1)

    async def test_installation(self) -> None:
        """Test complete installation"""
        try:
            print(f"🧪 Iniciando pruebas de instalación completa...")

            test_results = await coqui_installer.test_complete_installation()

            print(f"\n📊 Resultados de las pruebas:")

            # CoquiTTS
            coqui_result = test_results["coqui_tts"]
            coqui_status = "✅" if coqui_result.success else "❌"
            print(f"   {coqui_status} CoquiTTS: ", end="")
            if coqui_result.success:
                metadata = coqui_result.metadata or {}
                if "available_models" in metadata:
                    print(f"Funciona ({metadata['available_models']} modelos)")
                else:
                    print("Funciona")
            else:
                print(f"Falló - {coqui_result.error}")

            # Audio Dependencies
            audio_result = test_results["audio_deps"]
            audio_status = "✅" if audio_result.success else "❌"
            print(f"   {audio_status} Deps de audio: ", end="")
            if audio_result.success:
                print("Funcionan correctamente")
            else:
                print(f"Falló - {audio_result.error}")

            # FFmpeg
            ffmpeg_result = test_results["ffmpeg"]
            ffmpeg_status = "✅" if ffmpeg_result.success else "❌"
            print(f"   {ffmpeg_status} FFmpeg: ", end="")
            if ffmpeg_result.success:
                print("Funciona correctamente")
            else:
                print(f"Falló - {ffmpeg_result.error}")

            # Overall
            overall_status = "✅" if test_results["overall_success"] else "❌"
            print(f"\n   {overall_status} Estado general: ", end="")
            if test_results["overall_success"]:
                print("¡Sistema completamente funcional!")
                print(f"\n💡 Prueba las características:")
                print(f'   tts-notify --engine coqui "Test en francés" --language fr')
                print(f"   tts-notify --process-audio audio.wav --output-format mp3")
                print(f"   tts-notify --pipeline-status")
            else:
                print("❌ Sistema no completamente funcional")

        except Exception as e:
            print(f"❌ Error en pruebas: {e}")
            sys.exit(1)

    async def install_audio_dependencies(self) -> None:
        """Install audio processing dependencies only"""
        try:
            print(f"📦 Instalando dependencias de procesamiento de audio...")

            results = await coqui_installer.install_audio_dependencies()

            print(f"\n📊 Resultados de la instalación:")
            all_success = True

            for result in results:
                status = "✅" if result.success else "❌"
                print(f"   {status} {result.component}: ", end="")

                if result.success:
                    print("Instalado correctamente")
                else:
                    print(f"Error - {result.error}")
                    all_success = False

            if all_success:
                print(f"\n🎉 ¡Dependencias de audio instaladas exitosamente!")
                print(f"\n🎛️ Sistema listo para procesamiento de audio avanzado:")
                print(f"   tts-notify --process-audio input.wav --output-format mp3")
                print(f"   tts-notify --pipeline-status")
            else:
                print(f"\n⚠️  Algunas dependencias fallaron. Revisa los errores arriba.")

        except Exception as e:
            print(f"❌ Error en instalación de dependencias: {e}")
            sys.exit(1)

    async def show_installation_status(self) -> None:
        """Show current installation status"""
        try:
            print(f"📊 Estado Actual de la Instalación:")
            print("=" * 50)

            status = await coqui_installer.get_installation_status()

            print(f"\n🐍 Sistema:")
            print(f"   • Python: {status['python_version']}")
            print(f"   • Plataforma: {status['platform']}")

            print(f"\n🤖 CoquiTTS:")
            coqui_status = (
                "✅ Instalado" if status["coqui_tts_installed"] else "❌ No instalado"
            )
            print(f"   • Estado: {coqui_status}")
            if status["coqui_tts_installed"]:
                try:
                    version = coqui_installer._get_coqui_version()
                    print(f"   • Versión: {version}")
                except:
                    print(f"   • Versión: Desconocida")

            print(f"\n🎵 Dependencias de Audio:")
            audio_deps = status["audio_deps"]
            for dep_name, installed in audio_deps.items():
                status_icon = "✅" if installed else "❌"
                print(f"   • {dep_name}: {status_icon}")

            print(f"\n🎬 FFmpeg:")
            ffmpeg_status = (
                "✅ Disponible" if status["ffmpeg_available"] else "❌ No disponible"
            )
            print(f"   • Estado: {ffmpeg_status}")

            # Recommendations
            print(f"\n💡 Recomendaciones:")
            if not status["coqui_tts_installed"]:
                print(f"   • Instalar CoquiTTS: tts-notify --install-coqui")

            if not all(audio_deps.values()):
                missing = [
                    dep for dep, installed in audio_deps.items() if not installed
                ]
                print(f"   • Instalar dependencias: tts-notify --install-deps")
                print(f"     Faltantes: {', '.join(missing)}")

            if not status["ffmpeg_available"]:
                print(f"   • Instalar FFmpeg: tts-notify --install-ffmpeg")

            if status["coqui_tts_installed"] and all(audio_deps.values()):
                print(f'   • Sistema listo: tts-notify --engine coqui "Prueba"')

        except Exception as e:
            print(f"❌ Error obteniendo estado: {e}")

    async def speak_text(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        force_language: bool = False,
        emotion: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Speak text using TTS engine (v3.0.0 with CoquiTTS support)"""
        try:
            await self.initialize_engines()
            config = self.config_manager.get_config()
            engine_name = engine or config.TTS_NOTIFY_ENGINE

            if engine_name == "coqui":
                await self._speak_with_coqui(
                    text,
                    voice,
                    rate,
                    pitch,
                    volume,
                    model,
                    language,
                    force_language,
                    config,
                    emotion,
                    temperature,
                )
            else:
                await self._speak_with_macos(text, voice, rate, pitch, volume, config)

        except (
            VoiceNotFoundError,
            ValidationError,
            TTSError,
            EngineNotAvailableError,
        ) as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error inesperado: {e}")
            if self.logger:
                self.logger.exception("Unexpected error in speak_text")
            sys.exit(1)

    async def _speak_with_macos(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        config=None,
    ) -> None:
        """Speak text using macOS TTS engine"""
        try:
            # Get voice from voice manager
            voice_name = voice or config.TTS_NOTIFY_VOICE
            voice_obj = await self.voice_manager.find_voice(voice_name)
            if not voice_obj:
                voice_obj = await self.voice_manager.find_voice("monica")  # fallback

            # Create TTS request
            request = TTSRequest(
                text=text,
                voice=voice_obj,
                rate=rate or config.TTS_NOTIFY_RATE,
                pitch=pitch or config.TTS_NOTIFY_PITCH,
                volume=volume or config.TTS_NOTIFY_VOLUME,
                output_format=AudioFormat.AIFF,
            )

            # Use engine registry
            macos_engine = engine_registry.get("macos")
            response = await macos_engine.speak(request)

            if response.success:
                if self.logger:
                    self.logger.info(
                        f"Text spoken successfully with macOS engine: {text[:50]}..."
                    )
                engine_used = f"macOS ({voice_obj.name})"
                print(f"✅ Texto reproducido con motor: {engine_used}")
            else:
                print(f"❌ Error: {response.error}")
                sys.exit(1)

        except Exception as e:
            # Fallback to direct subprocess if engine fails
            import subprocess

            voice_to_use = voice or config.TTS_NOTIFY_VOICE
            rate_to_use = str(rate or config.TTS_NOTIFY_RATE)

            cmd = ["say", "-v", voice_to_use, "-r", rate_to_use]
            cmd.append(text)

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(
                    f"✅ Texto reproducido con voz: {voice_to_use} (fallback directo)"
                )
            else:
                print(f"❌ Error: {result.stderr}")
                sys.exit(1)

    async def _speak_with_coqui(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        force_language: bool = False,
        config=None,
        emotion: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Speak text using CoquiTTS engine with emotion support"""
        try:
            coqui_engine = engine_registry.get("coqui")
            if not coqui_engine:
                print(
                    "❌ Motor CoquiTTS no disponible. Instale con: pip install coqui-tts"
                )
                sys.exit(1)

            voices = await coqui_engine.get_supported_voices()
            voice_obj = None

            if voice:
                for v in voices:
                    if v.name.lower() == voice.lower() or v.id.lower() == voice.lower():
                        voice_obj = v
                        break

            if not voice_obj and voices:
                voice_obj = voices[0]

            if not voice_obj:
                print("❌ No hay voces CoquiTTS disponibles")
                sys.exit(1)

            request = TTSRequest(
                text=text,
                voice=voice_obj,
                rate=rate or config.TTS_NOTIFY_RATE,
                pitch=pitch or config.TTS_NOTIFY_PITCH,
                volume=volume or config.TTS_NOTIFY_VOLUME,
                engine_type=TTSEngineType.COQUI,
                language=language or config.TTS_NOTIFY_DEFAULT_LANGUAGE,
                force_language=force_language,
                model_name=model or config.TTS_NOTIFY_COQUI_MODEL,
                auto_download=config.TTS_NOTIFY_AUTO_DOWNLOAD_MODELS,
                output_format=AudioFormat.WAV,
            )

            if emotion and hasattr(coqui_engine, "synthesize"):
                response = await coqui_engine.synthesize(request, emotion=emotion)
            else:
                response = await coqui_engine.speak(request)

            if response.success:
                if self.logger:
                    self.logger.info(
                        f"Text spoken successfully with CoquiTTS: {text[:50]}..."
                    )
                engine_used = f"CoquiTTS ({voice_obj.name})"
                lang_used = language or "auto"
                emotion_info = f" [emocion: {emotion}]" if emotion else ""
                print(
                    f"✅ Texto reproducido con motor: {engine_used} [idioma: {lang_used}]{emotion_info}"
                )
            else:
                print(f"❌ Error CoquiTTS: {response.error}")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error con motor CoquiTTS: {e}")
            if "No module named 'TTS'" in str(e):
                print(
                    "   💡 Instale CoquiTTS: pip install coqui-tts torchaudio soundfile"
                )
            sys.exit(1)

    async def save_audio(
        self,
        text: str,
        filename: str,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        audio_format: str = "aiff",
        engine: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        force_language: bool = False,
        emotion: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Save text as audio file (v3.0.0 with CoquiTTS support)"""
        try:
            await self.initialize_engines()
            config = self.config_manager.get_config()

            output_path = Path(filename)
            if not output_path.suffix:
                output_path = output_path.with_suffix(f".{audio_format}")

            if not output_path.is_absolute():
                output_dir = Path(
                    getattr(config, "TTS_NOTIFY_OUTPUT_DIR", Path.home() / "Desktop")
                )
                output_path = output_dir / output_path

            engine_name = engine or config.TTS_NOTIFY_ENGINE

            if engine_name == "coqui":
                await self._save_with_coqui(
                    text,
                    output_path,
                    voice,
                    rate,
                    pitch,
                    volume,
                    audio_format,
                    model,
                    language,
                    force_language,
                    config,
                    emotion,
                    temperature,
                )
            else:
                await self._save_with_macos(
                    text, output_path, voice, rate, pitch, volume, audio_format, config
                )

        except (
            VoiceNotFoundError,
            ValidationError,
            TTSError,
            EngineNotAvailableError,
        ) as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error inesperado: {e}")
            if self.logger:
                self.logger.exception("Unexpected error in save_audio")
            sys.exit(1)

    async def _save_with_macos(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        audio_format: str = "aiff",
        config=None,
    ) -> None:
        """Save audio using macOS TTS engine"""
        try:
            # Get voice from voice manager
            voice_name = voice or config.TTS_NOTIFY_VOICE
            voice_obj = await self.voice_manager.find_voice(voice_name)
            if not voice_obj:
                voice_obj = await self.voice_manager.find_voice("monica")  # fallback

            # Create TTS request
            request = TTSRequest(
                text=text,
                voice=voice_obj,
                rate=rate or config.TTS_NOTIFY_RATE,
                pitch=pitch or config.TTS_NOTIFY_PITCH,
                volume=volume or config.TTS_NOTIFY_VOLUME,
                output_format=AudioFormat(audio_format),
                output_path=output_path,
            )

            # Use engine registry
            macos_engine = engine_registry.get("macos")
            response = await macos_engine.save(request, output_path)

            if response.success:
                print(f"✅ Audio guardado en: {output_path} (macOS)")
                if self.logger:
                    self.logger.info(f"Audio saved with macOS engine: {output_path}")
            else:
                print(f"❌ Error: {response.error}")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error guardando audio con macOS: {e}")
            sys.exit(1)

    async def _save_with_coqui(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        audio_format: str = "wav",
        model: Optional[str] = None,
        language: Optional[str] = None,
        force_language: bool = False,
        config=None,
        emotion: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Save audio using CoquiTTS engine with emotion support"""
        try:
            coqui_engine = engine_registry.get("coqui")
            if not coqui_engine:
                print(
                    "❌ Motor CoquiTTS no disponible. Instale con: pip install coqui-tts"
                )
                sys.exit(1)

            voices = await coqui_engine.get_supported_voices()
            voice_obj = None

            if voice:
                for v in voices:
                    if v.name.lower() == voice.lower() or v.id.lower() == voice.lower():
                        voice_obj = v
                        break

            if not voice_obj and voices:
                voice_obj = voices[0]

            if not voice_obj:
                print("❌ No hay voces CoquiTTS disponibles")
                sys.exit(1)

            request = TTSRequest(
                text=text,
                voice=voice_obj,
                rate=rate or config.TTS_NOTIFY_RATE,
                pitch=pitch or config.TTS_NOTIFY_PITCH,
                volume=volume or config.TTS_NOTIFY_VOLUME,
                engine_type=TTSEngineType.COQUI,
                language=language or config.TTS_NOTIFY_DEFAULT_LANGUAGE,
                force_language=force_language,
                model_name=model or config.TTS_NOTIFY_COQUI_MODEL,
                auto_download=config.TTS_NOTIFY_AUTO_DOWNLOAD_MODELS,
                output_format=AudioFormat(audio_format),
                output_path=output_path,
            )

            if emotion and hasattr(coqui_engine, "synthesize"):
                response = await coqui_engine.synthesize(request, emotion=emotion)
                if response.success and response.audio_data:
                    output_path.write_bytes(response.audio_data)
            else:
                response = await coqui_engine.save(request, output_path)

            if response.success:
                lang_used = language or "auto"
                emotion_info = f" [emocion: {emotion}]" if emotion else ""
                print(
                    f"✅ Audio guardado en: {output_path} (CoquiTTS [idioma: {lang_used}]{emotion_info})"
                )
                if self.logger:
                    self.logger.info(f"Audio saved with CoquiTTS engine: {output_path}")
            else:
                print(f"❌ Error CoquiTTS: {response.error}")
                sys.exit(1)

        except Exception as e:
            print(f"❌ Error guardando audio con CoquiTTS: {e}")
            if "No module named 'TTS'" in str(e):
                print(
                    "   💡 Instale CoquiTTS: pip install coqui-tts torchaudio soundfile"
                )
            sys.exit(1)

    def show_mcp_config(self) -> None:
        """Show MCP configuration for Claude Desktop with real paths"""
        import json
        import subprocess
        import sys
        from pathlib import Path

        try:
            # Get the current Python executable path
            if hasattr(sys, "real_prefix") or (
                hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
            ):
                # Virtual environment detected
                python_exe = sys.executable
            else:
                # Try to find a virtual environment in common locations
                project_root = Path(__file__).parent.parent.parent.parent
                venv_paths = [
                    project_root / "src" / "venv" / "bin" / "python",
                    project_root / "venv" / "bin" / "python",
                    Path.home()
                    / ".local"
                    / "share"
                    / "tts-notify"
                    / "venv"
                    / "bin"
                    / "python",
                ]

                python_exe = None
                for path in venv_paths:
                    if path.exists():
                        python_exe = str(path)
                        break

                if not python_exe:
                    python_exe = sys.executable

            # Get the MCP server script path
            current_dir = Path(__file__).parent.parent.parent.parent
            mcp_server_paths = [
                current_dir / "src" / "tts_notify" / "ui" / "mcp" / "server.py",
                current_dir / "src" / "mcp_server.py",
                Path(
                    "/Volumes/Resources/Develop/TTS-Notify/TTS_Notify/src/tts_notify/ui/mcp/server.py"
                ),
            ]

            mcp_server_path = None
            for path in mcp_server_paths:
                if path.exists():
                    mcp_server_path = str(path)
                    break

            if not mcp_server_path:
                # Fallback: assume it's installed as a module
                mcp_server_path = "-m tts_notify.ui.mcp.server"

            # Create MCP configuration - use the orchestrator with mode mcp
            mcp_config = {
                "mcpServers": {
                    "tts-notify": {
                        "command": python_exe,
                        "args": ["-m", "tts_notify", "--mode", "mcp"],
                    }
                }
            }

            # Pretty print JSON configuration
            print("📋 Configuración MCP para Claude Desktop")
            print("=" * 50)
            print("Copie y pegue este JSON en su archivo de configuración:")
            print()
            print("📍 Ruta del archivo de configuración:")
            print("   ~/Library/Application Support/Claude/claude_desktop_config.json")
            print()
            print("📝 Configuración JSON:")
            print(json.dumps(mcp_config, indent=2))
            print()
            print("💡 Notas:")
            print("   • Asegúrese de que TTS Notify esté instalado correctamente")
            print("   • Reinicie Claude Desktop después de modificar la configuración")
            print("   • Las rutas mostradas son las rutas reales en su sistema")

            if self.logger:
                self.logger.info("MCP configuration displayed")

        except Exception as e:
            print(f"❌ Error generando configuración MCP: {e}")
            if self.logger:
                self.logger.error(f"Error generating MCP config: {e}")

    async def run(self, args: argparse.Namespace) -> None:
        """Main CLI execution method (v3.0.0 with CoquiTTS support)"""
        # Load configuration profile if specified
        if args.profile:
            self.config_manager.reload_config(args.profile)

        # Setup logging
        self.setup_logging()

        # Handle MCP configuration request
        if args.mcp_config:
            self.show_mcp_config()
            return

        # Handle MCP Config Installation
        if args.install_mcp:
            from .mcp_config_generator import handle_install_mcp

            sys.exit(handle_install_mcp(args))

        # Handle CoquiTTS management commands
        if args.list_languages:
            await self.list_languages()
            return

        if args.download_language:
            await self.download_language(args.download_language)
            return

        if args.model_status:
            await self.show_model_status()
            return

        # Handle Phase B: Voice Cloning Commands
        if args.clone_voice:
            if not args.voice_name:
                print("❌ Error: --voice-name es requerido con --clone-voice")
                sys.exit(1)
            await self.clone_voice(
                audio_file=args.clone_voice,
                voice_name=args.voice_name,
                language=args.clone_language or args.language,
                quality=args.clone_quality,
                max_sample_duration=args.max_sample_duration,
            )
            return

        if args.list_cloned:
            await self.list_cloned_voices()
            return

        if args.delete_clone:
            await self.delete_cloned_voice(args.delete_clone)
            return

        if args.cloning_status:
            await self.show_cloning_status()
            return

        if args.enable_cloning:
            await self.toggle_cloning(enable=True)
            return

        if args.disable_cloning:
            await self.toggle_cloning(enable=False)
            return

        # Handle Installation Commands
        if args.install_coqui:
            await self.install_coqui_tts(use_gpu=False)
            return

        if args.install_coqui_gpu:
            await self.install_coqui_tts(use_gpu=True)
            return

        if args.install_all:
            await self.install_all_dependencies(use_gpu=False, include_ffmpeg=True)
            return

        if args.install_all_gpu:
            await self.install_all_dependencies(use_gpu=True, include_ffmpeg=True)
            return

        if args.test_installation:
            await self.test_installation()
            return

        if args.installation_status:
            await self.show_installation_status()
            return

        if args.install_deps:
            await self.install_audio_dependencies()
            return

        # Handle Phase C: Audio Pipeline Commands
        if args.pipeline_status:
            await self.show_pipeline_status()
            return

        if args.enable_pipeline:
            await self.toggle_pipeline(enable=True)
            return

        if args.disable_pipeline:
            await self.toggle_pipeline(enable=False)
            return

        if args.process_audio:
            await self.process_audio_file(
                input_file=args.process_audio,
                output_format=args.output_format,
                target_language=args.target_language,
                audio_quality=args.audio_quality,
            )
            return

        # Handle voice listing with engine support
        if args.list:
            await self.list_voices(
                compact=args.compact,
                gender=args.gen,
                language=args.lang,
                engine=args.engine,
            )
            return

        # Handle save command with CoquiTTS support
        if args.save:
            if not args.text:
                print("Error: Se requiere texto para guardar archivo de audio")
                sys.exit(1)
            engine = args.engine
            if args.xtts:
                engine = "coqui"
            await self.save_audio(
                text=args.text,
                filename=args.save,
                voice=args.voice,
                rate=args.rate,
                pitch=args.pitch,
                volume=args.volume,
                audio_format=args.format,
                engine=engine,
                model=args.model,
                language=args.language,
                force_language=args.force_language,
                emotion=args.emotion,
                temperature=args.temperature,
            )
            return

        # Handle speak command with CoquiTTS support
        if args.text:
            engine = args.engine
            if args.xtts:
                engine = "coqui"
            await self.speak_text(
                text=args.text,
                voice=args.voice,
                rate=args.rate,
                pitch=args.pitch,
                volume=args.volume,
                engine=engine,
                model=args.model,
                language=args.language,
                force_language=args.force_language,
                emotion=args.emotion,
                temperature=args.temperature,
            )
            return

        # No valid command provided
        print("Error: Se requiere texto o alguna de las opciones disponibles")
        print("Use --help para ver todas las opciones disponibles")
        sys.exit(1)


async def main():
    """Main entry point for CLI"""
    cli = TTSNotifyCLI()
    parser = cli.create_parser()
    args = parser.parse_args()

    try:
        await cli.run(args)
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"Error fatal: {e}")
        sys.exit(1)


def sync_main():
    """Synchronous main entry point for CLI scripts"""
    cli = TTSNotifyCLI()
    parser = cli.create_parser()
    args = parser.parse_args()

    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    sync_main()
