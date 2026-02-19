"""
Abstract TTS Engine for TTS Notify v2

This module defines the abstract base class for all TTS engines.
Provides a common interface for different TTS providers with async support.
"""

import asyncio
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, AsyncGenerator
import logging

from .models import TTSRequest, TTSResponse, Voice, AudioFormat
from .exceptions import TTSError, EngineNotAvailableError, ValidationError

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Abstract base class for TTS engines with async support"""

    def __init__(self, name: str):
        self.name = name
        self._initialized = False
        self._supported_formats: List[AudioFormat] = []

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the TTS engine"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources used by the TTS engine"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS engine is available on the current system"""
        pass

    @abstractmethod
    async def get_supported_voices(self) -> List[Voice]:
        """Get list of supported voices for this engine"""
        pass

    async def get_supported_formats(self) -> List[AudioFormat]:
        """Get list of supported audio formats"""
        return self._supported_formats

    @abstractmethod
    async def speak(self, request: TTSRequest) -> TTSResponse:
        """Convert text to speech and play it"""
        pass

    @abstractmethod
    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Convert text to speech and return audio data"""
        pass

    @abstractmethod
    async def save(self, request: TTSRequest, output_path: Path) -> TTSResponse:
        """Convert text to speech and save to file"""
        pass

    async def stream(self, request: TTSRequest) -> AsyncGenerator[bytes, None]:
        """Stream audio data as it's generated (optional implementation)"""
        # Default implementation uses synthesize and yields chunks
        response = await self.synthesize(request)
        if response.success and response.audio_data:
            # Yield in chunks (e.g., 8KB chunks)
            chunk_size = 8192
            for i in range(0, len(response.audio_data), chunk_size):
                yield response.audio_data[i:i + chunk_size]

    def validate_request(self, request: TTSRequest) -> None:
        """Validate TTS request"""
        if not isinstance(request, TTSRequest):
            raise ValidationError("Request must be a TTSRequest instance")

        # Validate format compatibility
        if request.output_format not in asyncio.run(self.get_supported_formats()):
            raise ValidationError(f"Format '{request.output_format.value}' is not supported by engine '{self.name}'")

        # Validate rate limits (maintain v1.5.0 behavior)
        if request.rate is not None and (request.rate < 100 or request.rate > 300):
            raise ValidationError("Rate must be between 100 and 300 WPM for optimal performance",
                              field="rate", value=request.rate)

    async def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information and capabilities"""
        return {
            "name": self.name,
            "available": self.is_available(),
            "initialized": self._initialized,
            "supported_voices_count": len(await self.get_supported_voices()),
            "supported_formats": [fmt.value for fmt in await self.get_supported_formats()],
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class SubprocessTTSEngine(TTSEngine):
    """Base class for subprocess-based TTS engines"""

    def __init__(self, name: str, command: str):
        super().__init__(name)
        self.command = command
        self._process_timeout = 60  # Default timeout in seconds

    def is_available(self) -> bool:
        """Check if the command is available"""
        try:
            subprocess.run([self.command, "--version"],
                         capture_output=True, timeout=5, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def _run_command(
        self,
        args: List[str],
        input_data: Optional[bytes] = None,
        timeout: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command asynchronously"""
        cmd = [self.command] + args
        timeout = timeout or self._process_timeout

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                input=input_data
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr
            )
        except asyncio.TimeoutError:
            # Kill the process if it times out
            if 'process' in locals():
                process.kill()
                await process.wait()
            raise TTSError(f"Command timed out after {timeout} seconds", engine_name=self.name)

    def _build_voice_args(self, voice: Voice) -> List[str]:
        """Build command arguments for voice selection"""
        # Override in subclasses
        return ["-v", voice.id]

    def _build_rate_args(self, rate: int) -> List[str]:
        """Build command arguments for speech rate"""
        # Override in subclasses
        return ["-r", str(rate)]

    def _build_pitch_args(self, pitch: float) -> List[str]:
        """Build command arguments for pitch"""
        # Override in subclasses
        return []

    def _build_volume_args(self, volume: float) -> List[str]:
        """Build command arguments for volume"""
        # Override in subclasses
        return []

    def _build_format_args(self, format: AudioFormat, output_path: Path) -> List[str]:
        """Build command arguments for output format"""
        # Override in subclasses
        return []


class MacOSTTSEngine(SubprocessTTSEngine):
    """macOS TTS engine using native say command (enhanced from v1.5.0)"""

    def __init__(self):
        super().__init__("macos", "say")
        self._supported_formats = [AudioFormat.AIFF]  # macOS say only supports AIFF

    async def initialize(self) -> None:
        """Initialize the macOS TTS engine"""
        if not self.is_available():
            raise EngineNotAvailableError("macOS TTS", "say command not available")
        self._initialized = True
        logger.info("macOS TTS engine initialized successfully")

    async def cleanup(self) -> None:
        """Cleanup resources used by the macOS TTS engine"""
        self._initialized = False
        logger.info("macOS TTS engine cleaned up")

    def is_available(self) -> bool:
        """Check if macOS say command is available"""
        try:
            result = subprocess.run([self.command, "-v", "?"],
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def get_supported_voices(self) -> List[Voice]:
        """Get supported voices by delegating to voice manager"""
        from tts_notify.core.voice_system import VoiceManager
        voice_manager = VoiceManager()
        return await voice_manager.get_all_voices()

    async def speak(self, request: TTSRequest) -> TTSResponse:
        """Convert text to speech and play it using macOS say command"""
        start_time = time.time()
        self.validate_request(request)

        try:
            # Build command arguments
            cmd = self._build_voice_args(request.voice)

            # Add rate if specified (maintain v1.5.0 behavior)
            if request.rate is not None:
                cmd.extend(self._build_rate_args(request.rate))

            # Add text to speak
            cmd.append(request.text)

            # Execute the command asynchronously
            completed_process = await self._run_command(cmd)

            if completed_process.returncode == 0:
                duration = time.time() - start_time
                logger.info(f"Successfully spoke text using voice '{request.voice.id}' in {duration:.2f}s")
                return TTSResponse(
                    success=True,
                    duration=duration,
                    format=AudioFormat.AIFF
                )
            else:
                error_msg = completed_process.stderr.decode() if completed_process.stderr else "Unknown error"
                logger.error(f"Failed to speak text: {error_msg}")
                return TTSResponse(
                    success=False,
                    error=error_msg
                )

        except asyncio.TimeoutError:
            error_msg = f"Speech timed out after {self._process_timeout} seconds"
            logger.error(error_msg)
            return TTSResponse(success=False, error=error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during speech: {str(e)}"
            logger.error(error_msg)
            return TTSResponse(success=False, error=error_msg)

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Convert text to speech and return audio data"""
        # macOS say command doesn't directly support returning audio data
        # We need to save to a temporary file and read it back
        import tempfile
        import os

        start_time = time.time()
        self.validate_request(request)

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Save to temporary file first
            save_response = await self.save(request, temp_path)

            if save_response.success and temp_path.exists():
                # Read the audio data
                audio_data = temp_path.read_bytes()
                duration = time.time() - start_time

                logger.info(f"Successfully synthesized {len(audio_data)} bytes using voice '{request.voice.id}'")

                return TTSResponse(
                    success=True,
                    audio_data=audio_data,
                    file_path=temp_path,
                    duration=duration,
                    format=AudioFormat.AIFF
                )
            else:
                return save_response

        except Exception as e:
            error_msg = f"Failed to synthesize audio: {str(e)}"
            logger.error(error_msg)
            return TTSResponse(success=False, error=error_msg)
        finally:
            # Clean up temporary file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    async def save(self, request: TTSRequest, output_path: Path) -> TTSResponse:
        """Convert text to speech and save to file using macOS say command"""
        start_time = time.time()
        self.validate_request(request)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add .aiff extension if not present
        if not output_path.suffix.lower():
            output_path = output_path.with_suffix(".aiff")

        try:
            # Build command arguments
            cmd = self._build_voice_args(request.voice)

            # Add rate if specified
            if request.rate is not None:
                cmd.extend(self._build_rate_args(request.rate))

            # Add output file argument
            cmd.extend(["-o", str(output_path)])

            # Add text to speak
            cmd.append(request.text)

            # Execute the command asynchronously
            completed_process = await self._run_command(cmd)

            if completed_process.returncode == 0 and output_path.exists():
                duration = time.time() - start_time
                file_size = output_path.stat().st_size

                logger.info(f"Successfully saved audio to '{output_path}' ({file_size} bytes) in {duration:.2f}s")

                return TTSResponse(
                    success=True,
                    file_path=output_path,
                    duration=duration,
                    format=AudioFormat.AIFF,
                    metadata={"file_size": file_size}
                )
            else:
                error_msg = completed_process.stderr.decode() if completed_process.stderr else "Unknown error"
                logger.error(f"Failed to save audio: {error_msg}")
                return TTSResponse(
                    success=False,
                    file_path=output_path,
                    error=error_msg
                )

        except Exception as e:
            error_msg = f"Failed to save audio: {str(e)}"
            logger.error(error_msg)
            return TTSResponse(
                success=False,
                file_path=output_path,
                error=error_msg
            )

    def _build_voice_args(self, voice: Voice) -> List[str]:
        """Build command arguments for voice selection"""
        return ["-v", voice.id]

    def _build_rate_args(self, rate: int) -> List[str]:
        """Build command arguments for speech rate"""
        return ["-r", str(rate)]

    def _build_pitch_args(self, pitch: float) -> List[str]:
        """Build command arguments for pitch (not supported by macOS say)"""
        # macOS say doesn't support pitch adjustment
        return []

    def _build_volume_args(self, volume: float) -> List[str]:
        """Build command arguments for volume (not supported by macOS say)"""
        # macOS say doesn't support volume adjustment via command line
        return []

    def _build_format_args(self, format: AudioFormat, output_path: Path) -> List[str]:
        """Build command arguments for output format"""
        # macOS say only supports AIFF format
        if format != AudioFormat.AIFF:
            logger.warning(f"macOS say only supports AIFF format, requested {format.value}")
        return ["-o", str(output_path)]


class EngineRegistry:
    """Registry for managing TTS engines"""

    def __init__(self):
        self._engines: Dict[str, TTSEngine] = {}
        self._default_engine: Optional[str] = None

    def register(self, engine: TTSEngine, is_default: bool = False) -> None:
        """Register a TTS engine"""
        self._engines[engine.name] = engine
        if is_default or self._default_engine is None:
            self._default_engine = engine.name
        logger.info(f"Registered TTS engine: {engine.name}")

    def unregister(self, name: str) -> None:
        """Unregister a TTS engine"""
        if name in self._engines:
            del self._engines[name]
            if self._default_engine == name:
                self._default_engine = next(iter(self._engines.keys()), None)
            logger.info(f"Unregistered TTS engine: {name}")

    def get(self, name: Optional[str] = None) -> TTSEngine:
        """Get a TTS engine by name"""
        if name is None:
            name = self._default_engine

        if name is None:
            raise EngineNotAvailableError("No TTS engine available and no default set")

        if name not in self._engines:
            raise EngineNotAvailableError(f"TTS engine '{name}' not registered")

        return self._engines[name]

    def list_available(self) -> List[str]:
        """List available engine names"""
        return list(self._engines.keys())

    async def initialize_all(self) -> None:
        """Initialize all registered engines"""
        for name, engine in self._engines.items():
            if not engine._initialized:
                try:
                    await engine.initialize()
                    engine._initialized = True
                except Exception as e:
                    logger.warning(f"Failed to initialize engine '{name}': {e}")

    async def cleanup_all(self) -> None:
        """Cleanup all registered engines"""
        for name, engine in self._engines.items():
            if engine._initialized:
                try:
                    await engine.cleanup()
                    engine._initialized = False
                except Exception as e:
                    logger.warning(f"Failed to cleanup engine '{name}': {e}")

    async def get_engine_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about an engine"""
        engine = self.get(name)
        return await engine.get_engine_info()

    async def get_all_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all engines"""
        info = {}
        for name, engine in self._engines.items():
            try:
                info[name] = await engine.get_engine_info()
            except Exception as e:
                info[name] = {"error": str(e)}
        return info


async def bootstrap_engines(config=None):
    """Bootstrap all available TTS engines based on configuration"""
    from .config_manager import ConfigManager

    if config is None:
        config_manager = ConfigManager()
        config = config_manager.get_config()

    # Always register macOS engine (default, guaranteed to work)
    macos_engine = MacOSTTSEngine()
    engine_registry.register(macos_engine, is_default=True)
    logger.info("Registered macOS TTS engine (default)")

    # Conditionally register CoquiTTS if available
    if (config.TTS_NOTIFY_ENGINE == "coqui" or
        config.TTS_NOTIFY_COQUI_AUTO_INIT or
        config.TTS_NOTIFY_AUTO_DOWNLOAD_MODELS):
        try:
            # Import here to avoid circular import issues
            from tts_notify.core.coqui_engine import CoquiTTSEngine

            # Create CoquiTTS engine
            coqui_engine = CoquiTTSEngine(config.TTS_NOTIFY_COQUI_MODEL)

            # Check if it's available before registering
            if coqui_engine.is_available():
                engine_registry.register(coqui_engine)

                # Set as default if explicitly requested
                if config.TTS_NOTIFY_ENGINE == "coqui":
                    # Unregister macOS as default first
                    engine_registry._default_engine = None
                    engine_registry.register(coqui_engine, is_default=True)
                    logger.info("Registered CoquiTTS engine as default")
                else:
                    logger.info("Registered CoquiTTS engine")
            else:
                logger.info("CoquiTTS engine not available, skipping registration")
                if config.TTS_NOTIFY_ENGINE == "coqui":
                    logger.warning("CoquiTTS requested but not available. Install with: pip install coqui-tts")

        except ImportError:
            logger.warning("CoquiTTS dependencies not available. Install with: pip install coqui-tts")
        except Exception as e:
            logger.warning(f"Failed to register CoquiTTS engine: {e}")

    # Also try to register CoquiTTS if available (even if not explicitly requested)
    else:
        try:
            from tts_notify.core.coqui_engine import CoquiTTSEngine
            coqui_engine = CoquiTTSEngine(config.TTS_NOTIFY_COQUI_MODEL)
            if coqui_engine.is_available():
                engine_registry.register(coqui_engine)
                logger.info("Auto-registered available CoquiTTS engine")
        except ImportError:
            logger.debug("CoquiTTS not available")
        except Exception as e:
            logger.debug(f"CoquiTTS auto-registration failed: {e}")


# Global engine registry instance
engine_registry = EngineRegistry()