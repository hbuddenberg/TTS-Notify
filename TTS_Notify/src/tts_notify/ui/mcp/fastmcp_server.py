#!/usr/bin/env python3
"""
TTS Notify FastMCP Server

FastMCP server implementation using TTS-Notify core modules.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

from ...core.voice_system import VoiceManager
from ...core.config_manager import config_manager
from ...core.tts_engine import engine_registry
from ...core.exceptions import TTSError, VoiceNotFoundError
from ...core.coqui_engine import CoquiTTSEngine, EMOTION_PRESETS
from ...core.voice_cloner import voice_cloner

logger = logging.getLogger(__name__)

mcp = FastMCP("TTS-Notify")

voice_manager = VoiceManager()


@mcp.tool
def speak_text(text: str, voice: Optional[str] = None, rate: int = 175) -> str:
    """Speak text using TTS with optional voice and rate.

    Args:
        text: Text to speak (max 5000 characters)
        voice: Voice name or ID (optional, uses system default if not specified)
        rate: Speech rate in words per minute (50-500, default: 175)

    Returns:
        Success message with voice used or error details
    """
    try:
        config = config_manager.get_config()

        voice_to_use = voice or config.TTS_NOTIFY_VOICE

        engine = engine_registry.get_engine(config.TTS_NOTIFY_ENGINE)

        asyncio.run(engine.speak(text, voice_to_use, rate))

        return f"✅ Text spoken successfully with voice: {voice_to_use}"

    except VoiceNotFoundError as e:
        return f"❌ Voice not found: {e}"
    except TTSError as e:
        return f"❌ TTS Error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"


@mcp.tool
def list_voices() -> list[dict]:
    """List all available TTS voices on the system.

    Returns:
        List of voice dictionaries containing:
        - id: Voice identifier
        - name: Human-readable voice name
        - language: Voice language code
        - locale: Locale identifier (e.g., es_ES, en_US)
        - gender: Gender (male, female, unknown)
        - quality: Quality tier (basic, enhanced, premium, siri, neural)
        - description: Voice description
    """
    try:
        voices = asyncio.run(voice_manager.get_all_voices())

        return [
            {
                "id": v.id,
                "name": v.name,
                "language": v.language.value,
                "locale": v.locale,
                "gender": v.gender.value,
                "quality": v.quality.value,
                "description": v.description,
            }
            for v in voices
        ]

    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        return []


@mcp.tool
def save_audio(
    text: str, output_path: str, voice: Optional[str] = None, rate: int = 175
) -> dict:
    """Save text as audio file.

    Args:
        text: Text to convert to audio
        output_path: Output file path (extension will be auto-detected)
        voice: Voice name or ID (optional, uses system default if not specified)
        rate: Speech rate in words per minute (50-500, default: 175)

    Returns:
        Dictionary with:
        - success: Boolean indicating if operation succeeded
        - output_path: Full path to saved audio file
        - message: Success or error message
        - file_size: File size in bytes (if successful)
    """
    try:
        config = config_manager.get_config()

        voice_to_use = voice or config.TTS_NOTIFY_VOICE

        output = Path(output_path)
        if not output.suffix:
            output = output.with_suffix(f".{config.TTS_NOTIFY_OUTPUT_FORMAT}")

        engine = engine_registry.get_engine(config.TTS_NOTIFY_ENGINE)

        asyncio.run(engine.save_to_file(text, str(output), voice_to_use, rate))

        file_size = output.stat().st_size if output.exists() else 0

        return {
            "success": True,
            "output_path": str(output),
            "message": f"Audio saved successfully to: {output}",
            "file_size": file_size,
        }

    except VoiceNotFoundError as e:
        return {
            "success": False,
            "output_path": output_path,
            "message": f"Voice not found: {e}",
            "file_size": 0,
        }
    except TTSError as e:
        return {
            "success": False,
            "output_path": output_path,
            "message": f"TTS Error: {e}",
            "file_size": 0,
        }
    except Exception as e:
        return {
            "success": False,
            "output_path": output_path,
            "message": f"Unexpected error: {e}",
            "file_size": 0,
        }


@mcp.tool
def get_config() -> dict:
    """Get current TTS-Notify configuration.

    Returns:
        Dictionary containing all active configuration settings including:
        - Engine selection (macos/coqui)
        - Voice settings (voice, rate, language, quality)
        - Performance settings (cache, timeout, max_concurrent)
        - Output settings (format, directory)
        - All other active environment variables
    """
    try:
        config = config_manager.get_config()

        return config.to_dict()

    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        return {}


@mcp.resource("voices://list")
def voices_list() -> list[dict]:
    """List all available TTS voices on the system.

    Returns:
        List of voice dictionaries containing:
        - id: Voice identifier
        - name: Human-readable voice name
        - language: Voice language code
        - locale: Locale identifier (e.g., es_ES, en_US)
        - gender: Gender (male, female, unknown)
        - quality: Quality tier (basic, enhanced, premium, siri, neural)
        - description: Voice description
    """
    try:
        voices = asyncio.run(voice_manager.get_all_voices())

        return [
            {
                "id": v.id,
                "name": v.name,
                "language": v.language.value,
                "locale": v.locale,
                "gender": v.gender.value,
                "quality": v.quality.value,
                "description": v.description,
            }
            for v in voices
        ]

    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        return []


@mcp.resource("config://current")
def config_current() -> dict:
    """Get current TTS-Notify configuration.

    Returns:
        Dictionary containing all active configuration settings including:
        - Engine selection (macos/coqui)
        - Voice settings (voice, rate, language, quality)
        - Performance settings (cache, timeout, max_concurrent)
        - Output settings (format, directory)
        - All other active environment variables
    """
    try:
        config = config_manager.get_config()

        return config.to_dict()

    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        return {}


@mcp.tool
def xtts_synthesize(
    text: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
    temperature: float = 0.5,
    emotion: Optional[str] = None,
    language: str = "es",
    output: Optional[str] = None,
) -> dict:
    """Synthesize speech using Coqui XTTS with emotion support.

    Args:
        text: Text to synthesize (max 5000 characters)
        voice: Voice name or ID (optional, uses default if not specified)
        speed: Speech speed multiplier (0.5-2.0, default: 1.0)
        temperature: Sampling temperature for variation (0.1-1.0, default: 0.5)
        emotion: Emotion preset (neutral, happy, sad, urgent, calm) or None
        language: Language code (en, es, fr, de, it, pt, nl, pl, ru, zh, ja, ko, etc.)
        output: Output file path (optional, plays audio if not specified)

    Returns:
        Dictionary with:
        - success: Boolean indicating if operation succeeded
        - message: Success or error message
        - output_path: Full path to saved audio file (if saved)
        - file_size: File size in bytes (if saved)
        - emotion: Applied emotion preset
    """
    try:
        config = config_manager.get_config()

        applied_emotion = None
        if emotion:
            emotion_lower = emotion.lower()
            if emotion_lower in EMOTION_PRESETS:
                applied_emotion = emotion_lower

        engine = engine_registry.get_engine("coqui")
        if not isinstance(engine, CoquiTTSEngine):
            return {"success": False, "message": "CoquiTTS engine not available"}

        from ...core.models import (
            TTSRequest,
            AudioFormat,
            Voice,
            VoiceQuality,
            Language,
            Gender,
            TTSEngineType,
        )

        voices = asyncio.run(engine.get_supported_voices())
        voice_to_use = (
            voices[0]
            if voices
            else Voice(
                id="coqui_default",
                name="Coqui Default",
                language=Language.ENGLISH,
                gender=Gender.UNKNOWN,
                quality=VoiceQuality.ENHANCED,
                engine_name="coqui",
                metadata={"engine_type": TTSEngineType.COQUI.value},
            )
        )

        request = TTSRequest(
            text=text,
            voice=voice_to_use,
            output_format=AudioFormat.WAV,
            language=language,
        )

        if output:
            output_path = Path(output)
            if not output_path.suffix:
                output_path = output_path.with_suffix(
                    f".{config.TTS_NOTIFY_OUTPUT_FORMAT}"
                )

            response = asyncio.run(engine.save(request, output_path))

            if response.success:
                file_size = output_path.stat().st_size if output_path.exists() else 0
                return {
                    "success": True,
                    "message": f"Audio synthesized with XTTS and saved to: {output_path}",
                    "output_path": str(output_path),
                    "file_size": file_size,
                    "emotion": applied_emotion,
                    "language": language,
                }
            else:
                return {
                    "success": False,
                    "message": f"Synthesis failed: {response.error}",
                }
        else:
            response = asyncio.run(engine.synthesize(request, emotion=applied_emotion))

            if response.success:
                return {
                    "success": True,
                    "message": f"Audio synthesized with XTTS (emotion: {applied_emotion or 'neutral'})",
                    "emotion": applied_emotion,
                    "language": language,
                    "duration": response.duration,
                }
            else:
                return {
                    "success": False,
                    "message": f"Synthesis failed: {response.error}",
                }

    except TTSError as e:
        return {"success": False, "message": f"TTS Error: {e}"}
    except Exception as e:
        return {"success": False, "message": f"Unexpected error: {e}"}


@mcp.tool
def clone_voice(audio_path: str, name: str, language: str = "es") -> dict:
    """Clone a voice from an audio sample using Coqui XTTS.

    Args:
        audio_path: Path to audio file for cloning (wav, mp3, m4a, etc.)
        name: Name for the cloned voice
        language: Language code of the audio (en, es, fr, de, etc.)

    Returns:
        Dictionary with:
        - success: Boolean indicating if operation succeeded
        - message: Success or error message
        - voice_id: ID of the cloned voice (if successful)
        - voice_name: Name of the cloned voice
        - embedding_path: Path to voice embedding file
        - processing_time: Time taken for cloning in seconds
    """
    try:
        config = config_manager.get_config()

        if not config.TTS_NOTIFY_COQUI_ENABLE_CLONING:
            return {
                "success": False,
                "message": "Voice cloning is disabled in configuration",
            }

        source_path = Path(audio_path).expanduser()
        if not source_path.exists():
            return {"success": False, "message": f"Audio file not found: {audio_path}"}

        from ...core.models import VoiceCloningRequest

        request = VoiceCloningRequest(
            source_audio_path=source_path,
            voice_name=name,
            language=language,
            quality=config.TTS_NOTIFY_COQUI_CLONING_QUALITY or "high",
            auto_optimize=config.TTS_NOTIFY_COQUI_AUTO_OPTIMIZE_CLONE,
            normalize=config.TTS_NOTIFY_COQUI_CLONING_NORMALIZE,
            denoise=config.TTS_NOTIFY_COQUI_CLONING_DENOISE,
        )

        response = asyncio.run(voice_cloner.clone_voice(request))

        if response.success:
            return {
                "success": True,
                "message": f"Voice '{name}' cloned successfully",
                "voice_id": response.voice.id if response.voice else None,
                "voice_name": response.voice.name if response.voice else name,
                "embedding_path": str(response.embedding_path)
                if response.embedding_path
                else None,
                "processing_time": response.processing_time,
                "sample_duration": response.sample_duration,
            }
        else:
            return {"success": False, "message": f"Cloning failed: {response.error}"}

    except TTSError as e:
        return {"success": False, "message": f"TTS Error: {e}"}
    except Exception as e:
        return {"success": False, "message": f"Unexpected error: {e}"}


@mcp.tool
def list_cloned_voices() -> list[dict]:
    """List all cloned voices available in the system.

    Returns:
        List of voice dictionaries containing:
        - id: Voice identifier
        - name: Voice name
        - language: Language code
        - created_at: Creation timestamp
        - sample_duration: Duration of source audio in seconds
        - quality_score: Optimization score (0.0-1.0)
        - embedding_path: Path to voice embedding file
        - profile_path: Path to voice profile file
    """
    try:
        voices = asyncio.run(voice_cloner.list_cloned_voices())

        return [
            {
                "id": v.id,
                "name": v.name,
                "language": v.cloning_language or "unknown",
                "created_at": v.created_at,
                "sample_duration": v.sample_duration,
                "quality_score": v.optimization_score,
                "embedding_path": v.embedding_path,
                "profile_path": v.profile_path,
            }
            for v in voices
        ]

    except Exception as e:
        logger.error(f"Error listing cloned voices: {e}")
        return []


@mcp.tool
def get_xtts_status() -> dict:
    """Get the status of the Coqui XTTS engine and voice cloning system.

    Returns:
        Dictionary containing:
        - available: Whether CoquiTTS is available
        - initialized: Whether the engine is initialized
        - model_name: Current model name
        - device: Device being used (cpu/cuda)
        - supported_languages: List of supported language codes
        - voice_cloning: Voice cloning status dict
        - profiles_count: Number of cloned voice profiles
        - storage_usage: Storage usage information in MB
    """
    try:
        status = {"available": False, "initialized": False}

        try:
            engine = engine_registry.get_engine("coqui")
            if isinstance(engine, CoquiTTSEngine):
                engine_info = asyncio.run(engine.get_engine_info())
                status.update(engine_info)

                cloning_status = asyncio.run(engine.get_cloning_status())
                status["voice_cloning"] = {
                    "enabled": cloning_status.get("enabled", False),
                    "profiles_count": cloning_status.get("profile_count", 0),
                    "storage_usage": cloning_status.get("storage_usage", {}),
                }

        except Exception as e:
            status["error"] = str(e)

        return status

    except Exception as e:
        logger.error(f"Error getting XTTS status: {e}")
        return {"error": str(e)}


if __name__ == "__mcp__":
    mcp.run()
