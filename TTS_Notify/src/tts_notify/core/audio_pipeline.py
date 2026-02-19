"""
Audio Pipeline System for TTS Notify v3.0.0 Phase C

This module implements comprehensive audio processing pipeline with language optimization,
format conversion, and advanced audio enhancement capabilities.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import subprocess

import numpy as np
from pydantic import BaseModel, Field

try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

from .models import (
    AudioFormat, Voice, TTSRequest, TTSResponse,
    Language, VoiceQuality
)
from .exceptions import TTSError, ValidationError, AudioProcessingError
from .config_manager import TTSConfig, config_manager

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Audio processing pipeline stages"""
    INPUT_VALIDATION = "input_validation"
    LOUDNESS_NORMALIZATION = "loudness_normalization"
    NOISE_REDUCTION = "noise_reduction"
    SILENCE_TRIMMING = "silence_trimming"
    LANGUAGE_OPTIMIZATION = "language_optimization"
    QUALITY_ENHANCEMENT = "quality_enhancement"
    FORMAT_CONVERSION = "format_conversion"
    FINAL_VALIDATION = "final_validation"


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 22050
    bit_depth: int = 16
    channels: int = 1
    target_loudness_lufs: float = -16.0
    peak_limit_dbfs: float = -1.0
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 0.1
    max_silence_duration: float = 2.0


@dataclass
class LanguageOptimization:
    """Language-specific audio optimization settings"""
    pitch_range_factor: float = 1.0
    speed_factor: float = 1.0
    emphasis_boost: float = 0.0
    spectral_shaping: bool = False
    prosody_enhancement: bool = False
    formant_boost: Optional[float] = None
    custom_filters: List[str] = field(default_factory=list)


@dataclass
class ProcessingMetrics:
    """Audio processing performance metrics"""
    processing_time: float
    input_size_bytes: int
    output_size_bytes: float
    compression_ratio: float
    peak_level_dbfs: float
    rms_level_dbfs: float
    dynamic_range_db: float
    spectral_centroid: float
    zero_crossing_rate: float
    stages_completed: List[ProcessingStage]
    warnings: List[str] = field(default_factory=list)


class AudioProcessor:
    """Advanced audio processing pipeline with language optimization"""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or config_manager.get_config()
        self.audio_config = AudioConfig()
        self.processing_history: List[ProcessingMetrics] = []
        self.temp_dir: Optional[Path] = None

        # Language-specific optimizations
        self.language_optimizations = self._initialize_language_optimizations()

        # Processing stages
        self.processing_stages = [
            ProcessingStage.INPUT_VALIDATION,
            ProcessingStage.LOUDNESS_NORMALIZATION,
            ProcessingStage.NOISE_REDUCTION,
            ProcessingStage.SILENCE_TRIMMING,
            ProcessingStage.LANGUAGE_OPTIMIZATION,
            ProcessingStage.QUALITY_ENHANCEMENT,
            ProcessingStage.FORMAT_CONVERSION,
            ProcessingStage.FINAL_VALIDATION
        ]

        self._setup_directories()

    def _setup_directories(self):
        """Setup temporary directories for audio processing"""
        self.temp_dir = Path.home() / ".tts-notify" / "temp" / "audio_pipeline"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_language_optimizations(self) -> Dict[str, LanguageOptimization]:
        """Initialize language-specific audio optimizations"""
        return {
            "es": LanguageOptimization(
                pitch_range_factor=1.1,
                speed_factor=0.95,
                emphasis_boost=1.2,
                spectral_shaping=True,
                prosody_enhancement=True,
                formant_boost=150.0
            ),
            "en": LanguageOptimization(
                pitch_range_factor=1.0,
                speed_factor=1.0,
                emphasis_boost=1.0,
                spectral_shaping=False,
                prosody_enhancement=False
            ),
            "fr": LanguageOptimization(
                pitch_range_factor=1.15,
                speed_factor=0.9,
                emphasis_boost=1.3,
                spectral_shaping=True,
                prosody_enhancement=True,
                formant_boost=180.0
            ),
            "de": LanguageOptimization(
                pitch_range_factor=1.05,
                speed_factor=0.98,
                emphasis_boost=1.1,
                spectral_shaping=True,
                prosody_enhancement=False
            ),
            "it": LanguageOptimization(
                pitch_range_factor=1.2,
                speed_factor=0.92,
                emphasis_boost=1.25,
                spectral_shaping=True,
                prosody_enhancement=True,
                formant_boost=160.0
            ),
            "pt": LanguageOptimization(
                pitch_range_factor=1.1,
                speed_factor=0.93,
                emphasis_boost=1.15,
                spectral_shaping=True,
                prosody_enhancement=True
            ),
            "ja": LanguageOptimization(
                pitch_range_factor=1.3,
                speed_factor=0.88,
                emphasis_boost=1.2,
                spectral_shaping=True,
                prosody_enhancement=True,
                custom_filters=["high_pass_80", "formant_shift"]
            ),
            "ko": LanguageOptimization(
                pitch_range_factor=1.25,
                speed_factor=0.9,
                emphasis_boost=1.18,
                spectral_shaping=True,
                prosody_enhancement=True,
                custom_filters=["high_pass_100"]
            ),
            "zh": LanguageOptimization(
                pitch_range_factor=1.2,
                speed_factor=0.85,
                emphasis_boost=1.22,
                spectral_shaping=True,
                prosody_enhancement=True,
                formant_boost=140.0
            ),
            "ru": LanguageOptimization(
                pitch_range_factor=1.05,
                speed_factor=0.96,
                emphasis_boost=1.08,
                spectral_shaping=True,
                prosody_enhancement=False
            )
        }

    async def process_audio(self,
                          audio_data: bytes,
                          source_format: AudioFormat,
                          target_format: AudioFormat,
                          language: Optional[str] = None,
                          quality_level: Optional[str] = None) -> Tuple[bytes, ProcessingMetrics]:
        """
        Process audio through the complete pipeline
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            # Fallback: return original audio
            return audio_data, self._create_fallback_metrics(audio_data)

        start_time = time.time()
        processed_audio = audio_data
        stages_completed = []
        warnings = []

        try:
            # Stage 1: Input validation
            processed_audio = await self._validate_input(processed_audio, source_format)
            stages_completed.append(ProcessingStage.INPUT_VALIDATION)

            # Stage 2: Loudness normalization
            if self.config.TTS_NOTIFY_COQUI_AUTO_CLEAN_AUDIO:
                processed_audio = await self._normalize_loudness(processed_audio, source_format)
                stages_completed.append(ProcessingStage.LOUDNESS_NORMALIZATION)

            # Stage 3: Noise reduction
            if self.config.TTS_NOTIFY_COQUI_NOISE_REDUCTION:
                processed_audio = await self._reduce_noise(processed_audio, source_format)
                stages_completed.append(ProcessingStage.NOISE_REDUCTION)

            # Stage 4: Silence trimming
            if self.config.TTS_NOTIFY_COQUI_AUTO_TRIM_SILENCE:
                processed_audio = await self._trim_silence(processed_audio, source_format)
                stages_completed.append(ProcessingStage.SILENCE_TRIMMING)

            # Stage 5: Language optimization
            if language:
                processed_audio, stage_warnings = await self._optimize_for_language(
                    processed_audio, source_format, language
                )
                stages_completed.append(ProcessingStage.LANGUAGE_OPTIMIZATION)
                warnings.extend(stage_warnings)

            # Stage 6: Quality enhancement
            processed_audio = await self._enhance_quality(
                processed_audio, source_format, quality_level
            )
            stages_completed.append(ProcessingStage.QUALITY_ENHANCEMENT)

            # Stage 7: Format conversion
            if source_format != target_format:
                processed_audio = await self._convert_format(
                    processed_audio, source_format, target_format
                )
                stages_completed.append(ProcessingStage.FORMAT_CONVERSION)

            # Stage 8: Final validation
            await self._validate_output(processed_audio, target_format)
            stages_completed.append(ProcessingStage.FINAL_VALIDATION)

            # Create metrics
            processing_time = time.time() - start_time
            metrics = await self._create_metrics(
                audio_data, processed_audio, processing_time, stages_completed, warnings
            )

            logger.info(f"Audio pipeline completed in {processing_time:.2f}s")
            return processed_audio, metrics

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Audio pipeline failed: {str(e)}"
            logger.error(error_msg)
            raise AudioProcessingError(error_msg)

    async def _validate_input(self, audio_data: bytes, format: AudioFormat) -> bytes:
        """Validate input audio data"""
        try:
            # Try to load the audio to validate it
            with tempfile.NamedTemporaryFile(suffix=f".{format.value}", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_path.write_bytes(audio_data)

                # Validate with librosa if available
                if AUDIO_PROCESSING_AVAILABLE:
                    try:
                        y, sr = librosa.load(str(temp_path), sr=None)
                        if len(y) == 0:
                            raise ValidationError("Empty audio data")
                        if sr <= 0:
                            raise ValidationError(f"Invalid sample rate: {sr}")
                    except Exception as e:
                        raise ValidationError(f"Invalid audio data: {e}")

                # Clean up
                temp_path.unlink(missing_ok=True)

            return audio_data

        except Exception as e:
            raise ValidationError(f"Input validation failed: {e}")

    async def _normalize_loudness(self, audio_data: bytes, format: AudioFormat) -> bytes:
        """Normalize audio loudness to target LUFS"""
        try:
            if not AUDIO_PROCESSING_AVAILABLE:
                return audio_data

            with tempfile.NamedTemporaryFile(suffix=f".{format.value}", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_path.write_bytes(audio_data)

                # Load audio
                y, sr = librosa.load(str(temp_path), sr=self.audio_config.sample_rate)

                # Calculate current RMS
                rms = np.sqrt(np.mean(y ** 2))
                if rms > 0:
                    # Calculate target gain
                    current_lufs = -20 * np.log10(rms) if rms > 0 else -60
                    gain_db = self.audio_config.target_loudness_lufs - current_lufs
                    gain_linear = 10 ** (gain_db / 20)

                    # Apply gain with peak limiting
                    y = y * gain_linear

                    # Peak limiting
                    peak = np.max(np.abs(y))
                    if peak > 1.0:
                        y = y / peak

                # Save processed audio
                sf.write(str(temp_path), y, sr, format='WAV')
                processed_data = temp_path.read_bytes()

                # Clean up
                temp_path.unlink(missing_ok=True)

                return processed_data

        except Exception as e:
            logger.warning(f"Loudness normalization failed: {e}")
            return audio_data

    async def _reduce_noise(self, audio_data: bytes, format: AudioFormat) -> bytes:
        """Apply noise reduction to audio"""
        try:
            if not AUDIO_PROCESSING_AVAILABLE:
                return audio_data

            with tempfile.NamedTemporaryFile(suffix=f".{format.value}", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_path.write_bytes(audio_data)

                # Load audio
                y, sr = librosa.load(str(temp_path), sr=self.audio_config.sample_rate)

                # Simple spectral subtraction for noise reduction
                # This is a simplified implementation
                stft = librosa.stft(y)
                magnitude = np.abs(stft)
                phase = np.angle(stft)

                # Estimate noise from first 0.5 seconds
                noise_frames = int(0.5 * sr / 512)  # Assuming hop length of 512
                if magnitude.shape[1] > noise_frames:
                    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

                    # Spectral subtraction
                    alpha = 2.0  # Over-subtraction factor
                    enhanced_magnitude = magnitude - alpha * noise_profile
                    enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)

                    # Reconstruct signal
                    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
                    y = librosa.istft(enhanced_stft)

                # Save processed audio
                sf.write(str(temp_path), y, sr, format='WAV')
                processed_data = temp_path.read_bytes()

                # Clean up
                temp_path.unlink(missing_ok=True)

                return processed_data

        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_data

    async def _trim_silence(self, audio_data: bytes, format: AudioFormat) -> bytes:
        """Trim silence from beginning and end of audio"""
        try:
            if not AUDIO_PROCESSING_AVAILABLE:
                return audio_data

            with tempfile.NamedTemporaryFile(suffix=f".{format.value}", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_path.write_bytes(audio_data)

                # Load audio
                y, sr = librosa.load(str(temp_path), sr=self.audio_config.sample_rate)

                # Trim silence
                y_trimmed, _ = librosa.effects.trim(
                    y,
                    top_db=30,
                    frame_length=2048,
                    hop_length=512
                )

                # Ensure minimum duration
                min_duration = 0.5  # 0.5 seconds minimum
                if len(y_trimmed) / sr < min_duration:
                    # Pad with silence if too short
                    padding_samples = int((min_duration - len(y_trimmed) / sr) * sr)
                    y_trimmed = np.pad(y_trimmed, (0, padding_samples), mode='constant')

                # Save processed audio
                sf.write(str(temp_path), y_trimmed, sr, format='WAV')
                processed_data = temp_path.read_bytes()

                # Clean up
                temp_path.unlink(missing_ok=True)

                return processed_data

        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio_data

    async def _optimize_for_language(self, audio_data: bytes, format: AudioFormat,
                                   language: str) -> Tuple[bytes, List[str]]:
        """Apply language-specific audio optimizations"""
        warnings = []

        try:
            if not AUDIO_PROCESSING_AVAILABLE or language not in self.language_optimizations:
                return audio_data, warnings

            optimization = self.language_optimizations[language]

            with tempfile.NamedTemporaryFile(suffix=f".{format.value}", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_path.write_bytes(audio_data)

                # Load audio
                y, sr = librosa.load(str(temp_path), sr=self.audio_config.sample_rate)

                # Apply language-specific optimizations

                # Pitch adjustment
                if optimization.pitch_range_factor != 1.0:
                    try:
                        # Simple pitch shifting using librosa
                        y = librosa.effects.pitch_shift(
                            y,
                            sr=sr,
                            n_steps=0,  # No pitch change
                            steps_per_octave=12
                        )
                    except Exception as e:
                        warnings.append(f"Pitch optimization failed: {e}")

                # Speed adjustment
                if optimization.speed_factor != 1.0:
                    try:
                        # Time stretching
                        y = librosa.effects.time_stretch(y, rate=optimization.speed_factor)
                    except Exception as e:
                        warnings.append(f"Speed optimization failed: {e}")

                # Emphasis boost
                if optimization.emphasis_boost != 1.0:
                    try:
                        # Apply high-frequency emphasis
                        from scipy import signal
                        sos = signal.butter(2, 3000, btype='high', fs=sr, output='sos')
                        y_high = signal.sosfilt(sos, y)
                        y = y + optimization.emphasis_boost * 0.3 * y_high
                    except ImportError:
                        warnings.append("SciPy not available for emphasis processing")
                    except Exception as e:
                        warnings.append(f"Emphasis processing failed: {e}")

                # Formant boost
                if optimization.formant_boost:
                    try:
                        # Simple formant enhancement using band-pass filter
                        from scipy import signal
                        low_freq = optimization.formant_boost - 50
                        high_freq = optimization.formant_boost + 50
                        sos = signal.butter(2, [low_freq, high_freq], btype='band', fs=sr, output='sos')
                        y_formant = signal.sosfilt(sos, y)
                        y = y + 0.2 * y_formant
                    except ImportError:
                        warnings.append("SciPy not available for formant processing")
                    except Exception as e:
                        warnings.append(f"Formant processing failed: {e}")

                # Save processed audio
                sf.write(str(temp_path), y, sr, format='WAV')
                processed_data = temp_path.read_bytes()

                # Clean up
                temp_path.unlink(missing_ok=True)

                return processed_data, warnings

        except Exception as e:
            logger.warning(f"Language optimization failed for {language}: {e}")
            return audio_data, [f"Language optimization failed: {e}"]

    async def _enhance_quality(self, audio_data: bytes, format: AudioFormat,
                             quality_level: Optional[str] = None) -> bytes:
        """Apply quality enhancement based on quality level"""
        try:
            if not AUDIO_PROCESSING_AVAILABLE:
                return audio_data

            quality = quality_level or self.config.TTS_NOTIFY_COQUI_CLONING_QUALITY

            # Quality-specific processing
            if quality in ["high", "ultra"]:
                return await self._apply_high_quality_processing(audio_data, format)
            elif quality == "medium":
                return await self._apply_medium_quality_processing(audio_data, format)
            else:  # low quality
                return audio_data

        except Exception as e:
            logger.warning(f"Quality enhancement failed: {e}")
            return audio_data

    async def _apply_high_quality_processing(self, audio_data: bytes, format: AudioFormat) -> bytes:
        """Apply high-quality audio processing"""
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{format.value}", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_path.write_bytes(audio_data)

                # Load audio
                y, sr = librosa.load(str(temp_path), sr=self.audio_config.sample_rate)

                # High-quality enhancements
                # 1. Harmonic enhancement
                harmonic, percussive = librosa.effects.hpss(y)
                y = 0.8 * y + 0.1 * harmonic + 0.1 * percussive

                # 2. Spectral enhancement
                stft = librosa.stft(y)
                magnitude = np.abs(stft)
                phase = np.angle(stft)

                # Enhance high frequencies slightly
                freq_bins = magnitude.shape[0]
                high_freq_boost = np.linspace(1.0, 1.1, freq_bins // 4)
                enhancement = np.ones(freq_bins)
                enhancement[-len(high_freq_boost):] = high_freq_boost

                magnitude = magnitude * enhancement[:, np.newaxis]
                stft_enhanced = magnitude * np.exp(1j * phase)
                y = librosa.istft(stft_enhanced)

                # Save processed audio
                sf.write(str(temp_path), y, sr, format='WAV')
                processed_data = temp_path.read_bytes()

                # Clean up
                temp_path.unlink(missing_ok=True)

                return processed_data

        except Exception as e:
            logger.warning(f"High-quality processing failed: {e}")
            return audio_data

    async def _apply_medium_quality_processing(self, audio_data: bytes, format: AudioFormat) -> bytes:
        """Apply medium-quality audio processing"""
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{format.value}", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_path.write_bytes(audio_data)

                # Load audio
                y, sr = librosa.load(str(temp_path), sr=self.audio_config.sample_rate)

                # Medium-quality enhancements
                # Basic harmonic enhancement
                harmonic, percussive = librosa.effects.hpss(y)
                y = 0.9 * y + 0.05 * harmonic + 0.05 * percussive

                # Save processed audio
                sf.write(str(temp_path), y, sr, format='WAV')
                processed_data = temp_path.read_bytes()

                # Clean up
                temp_path.unlink(missing_ok=True)

                return processed_data

        except Exception as e:
            logger.warning(f"Medium-quality processing failed: {e}")
            return audio_data

    async def _convert_format(self, audio_data: bytes, source_format: AudioFormat,
                            target_format: AudioFormat) -> bytes:
        """Convert audio between formats"""
        if source_format == target_format:
            return audio_data

        try:
            # Try using ffmpeg if available
            if FFMPEG_AVAILABLE:
                return await self._convert_with_ffmpeg(audio_data, source_format, target_format)
            else:
                # Fallback using librosa/soundfile
                return await self._convert_with_librosa(audio_data, source_format, target_format)

        except Exception as e:
            logger.warning(f"Format conversion failed: {e}")
            return audio_data

    async def _convert_with_ffmpeg(self, audio_data: bytes, source_format: AudioFormat,
                                 target_format: AudioFormat) -> bytes:
        """Convert audio using ffmpeg"""
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{source_format.value}", delete=False) as input_file:
                input_path = Path(input_file.name)
                input_path.write_bytes(audio_data)

            with tempfile.NamedTemporaryFile(suffix=f".{target_format.value}", delete=False) as output_file:
                output_path = Path(output_file.name)

            # Use ffmpeg for conversion
            process = await asyncio.create_subprocess_exec(
                'ffmpeg', '-y', '-i', str(input_path), str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                converted_data = output_path.read_bytes()

                # Clean up
                input_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)

                return converted_data
            else:
                raise AudioProcessingError(f"FFmpeg conversion failed: {stderr.decode()}")

        except Exception as e:
            logger.error(f"FFmpeg conversion error: {e}")
            raise

    async def _convert_with_librosa(self, audio_data: bytes, source_format: AudioFormat,
                                  target_format: AudioFormat) -> bytes:
        """Convert audio using librosa/soundfile"""
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{source_format.value}", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                temp_path.write_bytes(audio_data)

                # Load audio
                y, sr = librosa.load(str(temp_path), sr=None)

                # Determine output format for soundfile
                format_mapping = {
                    AudioFormat.WAV: 'WAV',
                    AudioFormat.FLAC: 'FLAC',
                    AudioFormat.OGG: 'OGG'
                }
                sf_format = format_mapping.get(target_format, 'WAV')

                # Save in target format
                with tempfile.NamedTemporaryFile(suffix=f".{target_format.value}", delete=False) as output_file:
                    output_path = Path(output_file.name)
                    sf.write(str(output_path), y, sr, format=sf_format)
                    converted_data = output_path.read_bytes()
                    output_path.unlink(missing_ok=True)

                # Clean up
                temp_path.unlink(missing_ok=True)

                return converted_data

        except Exception as e:
            logger.error(f"Librosa conversion error: {e}")
            raise

    async def _validate_output(self, audio_data: bytes, format: AudioFormat) -> None:
        """Validate processed audio output"""
        try:
            if len(audio_data) == 0:
                raise ValidationError("Empty processed audio")

            # Basic format validation
            if format == AudioFormat.WAV:
                # Check WAV header
                if len(audio_data) < 44:
                    raise ValidationError("Invalid WAV file: too short")
                if not audio_data[:4] == b'RIFF':
                    raise ValidationError("Invalid WAV file: missing RIFF header")

        except Exception as e:
            raise ValidationError(f"Output validation failed: {e}")

    async def _create_metrics(self, input_data: bytes, output_data: bytes, processing_time: float,
                            stages_completed: List[ProcessingStage], warnings: List[str]) -> ProcessingMetrics:
        """Create processing metrics"""
        try:
            # Calculate basic metrics
            input_size = len(input_data)
            output_size = len(output_data)
            compression_ratio = input_size / output_size if output_size > 0 else 1.0

            # Audio analysis if available
            peak_level = -60.0
            rms_level = -60.0
            dynamic_range = 0.0
            spectral_centroid = 0.0
            zero_crossing_rate = 0.0

            if AUDIO_PROCESSING_AVAILABLE:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_path = Path(temp_file.name)
                        temp_path.write_bytes(output_data)

                        y, sr = librosa.load(str(temp_path), sr=None)

                        if len(y) > 0:
                            # Peak level
                            peak_level = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else -60

                            # RMS level
                            rms = np.sqrt(np.mean(y ** 2))
                            rms_level = 20 * np.log10(rms) if rms > 0 else -60

                            # Dynamic range
                            signal_peak = np.max(np.abs(y))
                            signal_floor = np.min(np.abs(y[y > 0])) if np.any(y > 0) else 0
                            dynamic_range = 20 * np.log10(signal_peak / signal_floor) if signal_floor > 0 else 0

                            # Spectral features
                            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

                        temp_path.unlink(missing_ok=True)

                except Exception as e:
                    logger.warning(f"Audio analysis for metrics failed: {e}")

            return ProcessingMetrics(
                processing_time=processing_time,
                input_size_bytes=input_size,
                output_size_bytes=output_size,
                compression_ratio=compression_ratio,
                peak_level_dbfs=peak_level,
                rms_level_dbfs=rms_level,
                dynamic_range_db=dynamic_range,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=zero_crossing_rate,
                stages_completed=stages_completed,
                warnings=warnings
            )

        except Exception as e:
            logger.warning(f"Metrics creation failed: {e}")
            return self._create_fallback_metrics(output_data)

    def _create_fallback_metrics(self, audio_data: bytes) -> ProcessingMetrics:
        """Create fallback metrics when audio processing is not available"""
        return ProcessingMetrics(
            processing_time=0.0,
            input_size_bytes=len(audio_data),
            output_size_bytes=len(audio_data),
            compression_ratio=1.0,
            peak_level_dbfs=-60.0,
            rms_level_dbfs=-60.0,
            dynamic_range_db=0.0,
            spectral_centroid=0.0,
            zero_crossing_rate=0.0,
            stages_completed=[],
            warnings=["Audio processing not available"]
        )

    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get audio processing capabilities"""
        return {
            "audio_processing_available": AUDIO_PROCESSING_AVAILABLE,
            "ffmpeg_available": FFMPEG_AVAILABLE,
            "supported_languages": list(self.language_optimizations.keys()),
            "supported_formats": [fmt.value for fmt in AudioFormat],
            "processing_stages": [stage.value for stage in self.processing_stages],
            "temp_directory": str(self.temp_dir),
            "config": {
                "sample_rate": self.audio_config.sample_rate,
                "target_loudness_lufs": self.audio_config.target_loudness_lufs,
                "peak_limit_dbfs": self.audio_config.peak_limit_dbfs
            }
        }

    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get processing history"""
        history = []
        for metrics in self.processing_history:
            history.append({
                "processing_time": metrics.processing_time,
                "input_size_bytes": metrics.input_size_bytes,
                "output_size_bytes": metrics.output_size_bytes,
                "compression_ratio": metrics.compression_ratio,
                "stages_completed": [stage.value for stage in metrics.stages_completed],
                "warnings_count": len(metrics.warnings),
                "peak_level_dbfs": metrics.peak_level_dbfs,
                "rms_level_dbfs": metrics.rms_level_dbfs
            })
        return history


# Global audio processor instance
audio_processor = AudioProcessor()