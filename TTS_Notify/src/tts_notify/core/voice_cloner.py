"""
Voice Cloning System for TTS Notify v3.0.0 Phase B

This module implements voice cloning capabilities for CoquiTTS with multi-language support.
Provides comprehensive audio processing, embedding generation, and voice profile management.
"""

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging

import numpy as np
import soundfile as sf
import librosa
from pydantic import BaseModel, Field

from .models import (
    Voice,
    VoiceCloningRequest,
    VoiceCloningResponse,
    VoiceProfile,
    Gender,
    Language,
    VoiceQuality,
    TTSEngineType,
)
from .exceptions import TTSError, ValidationError, VoiceNotFoundError
from .config_manager import TTSConfig, config_manager

logger = logging.getLogger(__name__)


@dataclass
class AudioProcessingConfig:
    """Audio processing configuration for voice cloning"""

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    win_length: Optional[int] = None
    normalize: bool = True
    trim_silence: bool = True
    denoise: bool = True


class VoiceCloner:
    """Advanced voice cloning system for CoquiTTS with multi-language support"""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or config_manager.get_config()
        self.processing_config = AudioProcessingConfig()
        self._cloning_cache: Dict[str, VoiceProfile] = {}
        self._temp_dir: Optional[Path] = None

        # Quality settings
        self.quality_settings = {
            "low": {"n_mels": 40, "hop_length": 512, "sample_rate": 16000},
            "medium": {"n_mels": 64, "hop_length": 256, "sample_rate": 22050},
            "high": {"n_mels": 80, "hop_length": 256, "sample_rate": 22050},
            "ultra": {"n_mels": 80, "hop_length": 128, "sample_rate": 24000},
        }

        self._setup_directories()

    def _setup_directories(self):
        """Setup directories for voice cloning"""
        # Profile directory
        self.profile_dir = Path(
            self.config.TTS_NOTIFY_COQUI_PROFILE_DIR or "~/.tts-notify/voice-profiles"
        ).expanduser()
        self.embedding_dir = Path(
            self.config.TTS_NOTIFY_COQUI_EMBEDDING_DIR or "~/.tts-notify/embeddings"
        ).expanduser()
        self.voice_audio_dir = Path("~/.tts-notify/voice-audio").expanduser()

        # Temp directory
        self._temp_dir = Path.home() / ".tts-notify" / "temp" / "cloning"

        # Create directories
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.voice_audio_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Voice cloning directories configured: {self.profile_dir}")

    async def clone_voice(self, request: VoiceCloningRequest) -> VoiceCloningResponse:
        """
        Clone a voice from audio sample with comprehensive processing
        """
        start_time = time.time()

        try:
            logger.info(
                f"Starting voice cloning: {request.voice_name} ({request.language})"
            )

            # Validate and process source audio
            processed_audio, sample_duration = await self._process_source_audio(request)

            # Generate voice embedding
            embedding_path = await self._generate_voice_embedding(
                processed_audio, request.voice_name, request.language
            )

            # Create voice profile
            voice_profile = await self._create_voice_profile(
                request, embedding_path, sample_duration
            )

            # Optimize if requested
            optimization_score = None
            if request.auto_optimize:
                optimization_score = await self._optimize_voice_profile(voice_profile)
                voice_profile.quality_score = optimization_score

            # Create voice object
            voice = await self._create_cloned_voice(
                request, voice_profile, sample_duration
            )

            # Cache the profile
            self._cloning_cache[voice.id] = voice_profile

            processing_time = time.time() - start_time

            logger.info(
                f"Voice cloning completed successfully in {processing_time:.2f}s"
            )

            return VoiceCloningResponse(
                success=True,
                voice=voice,
                embedding_path=embedding_path,
                profile_path=voice_profile.embedding_path,
                processing_time=processing_time,
                optimization_score=optimization_score,
                sample_duration=sample_duration,
                metadata={
                    "quality": request.quality,
                    "language": request.language,
                    "auto_optimized": request.auto_optimize,
                    "processing_config": asdict(self.processing_config),
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Voice cloning failed: {str(e)}"
            logger.error(error_msg)

            return VoiceCloningResponse(
                success=False,
                error=error_msg,
                processing_time=processing_time,
                metadata={"request": asdict(request)},
            )

    def _convert_to_wav(self, input_path: Path) -> Path:
        """Convert audio file to WAV format using ffmpeg"""
        try:
            # Create temp file
            import tempfile
            import subprocess

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = Path(temp_file.name)

            # Run ffmpeg
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(self.processing_config.sample_rate),
                "-ac",
                "1",  # Mono
                str(output_path),
            ]

            # Run with timeout to prevent hanging
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

            return output_path

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise TTSError(f"Failed to convert audio to WAV: {e}")

    def _convert_to_wav_file(self, input_path: Path, output_path: Path) -> None:
        """Convert audio file to WAV format at specific location"""
        try:
            import subprocess

            # Run ffmpeg
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(self.processing_config.sample_rate),
                "-ac",
                "1",  # Mono
                str(output_path),
            ]

            # Run with timeout
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

        except Exception as e:
            logger.error(f"Audio conversion to file failed: {e}")
            raise TTSError(f"Failed to convert audio to WAV file: {e}")

    async def _process_source_audio(
        self, request: VoiceCloningRequest
    ) -> Tuple[np.ndarray, float]:
        """
        Process source audio with comprehensive enhancement
        """
        try:
            logger.info(f"Processing source audio: {request.source_audio_path}")

            # Convert to WAV if needed for better compatibility (e.g. m4a, mp3)
            audio_source = request.source_audio_path
            temp_wav_path = None

            if request.source_audio_path.suffix.lower() != ".wav":
                try:
                    logger.info(
                        f"Converting {request.source_audio_path.suffix} to WAV for processing..."
                    )
                    temp_wav_path = self._convert_to_wav(request.source_audio_path)
                    audio_source = temp_wav_path
                except Exception as e:
                    logger.warning(f"Conversion failed, trying direct load: {e}")

            # Load audio
            try:
                audio, sr = librosa.load(
                    str(audio_source), sr=self.processing_config.sample_rate, mono=True
                )
            finally:
                # Clean up temp file
                if temp_wav_path and temp_wav_path.exists():
                    try:
                        temp_wav_path.unlink()
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete temp file {temp_wav_path}: {e}"
                        )

            # Check duration
            duration = librosa.get_duration(y=audio, sr=sr)
            min_duration = self.config.TTS_NOTIFY_COQUI_MIN_SAMPLE_SECONDS
            max_duration = self.config.TTS_NOTIFY_COQUI_MAX_SAMPLE_SECONDS

            if duration < min_duration:
                raise ValidationError(
                    f"Audio too short: {duration:.2f}s (minimum: {min_duration}s)"
                )
            # AUTO-TRIM: Cut audio to max duration if too long
            if duration > max_duration:
                logger.info(
                    f"Audio too long ({duration:.2f}s), auto-trimming to {max_duration}s"
                )
                max_samples = int(max_duration * sr)
                audio = audio[:max_samples]
                duration = max_duration
                logger.info(f"Audio trimmed to {duration:.2f}s")

            logger.info(f"Audio loaded: {duration:.2f}s at {sr}Hz")

            # Apply quality settings
            quality_config = self.quality_settings[request.quality]
            if quality_config["sample_rate"] != sr:
                audio = librosa.resample(
                    y=audio, orig_sr=sr, target_sr=quality_config["sample_rate"]
                )
                sr = quality_config["sample_rate"]

            # Audio enhancement pipeline
            if request.normalize or self.config.TTS_NOTIFY_COQUI_CLONING_NORMALIZE:
                audio = self._normalize_audio(audio)

            if request.denoise or self.config.TTS_NOTIFY_COQUI_CLONING_DENOISE:
                audio = self._denoise_audio(audio)

            if self.config.TTS_NOTIFY_COQUI_AUTO_TRIM_SILENCE:
                audio, _ = librosa.effects.trim(
                    audio, top_db=20, frame_length=2048, hop_length=512
                )

            # Validate final audio
            if np.max(np.abs(audio)) < 0.001:
                raise ValidationError("Audio is too quiet after processing")

            final_duration = len(audio) / sr
            logger.info(f"Audio processing completed: {final_duration:.2f}s")

            return audio, final_duration

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise TTSError(f"Failed to process source audio: {e}")

    async def _generate_voice_embedding(
        self, audio: np.ndarray, voice_name: str, language: str
    ) -> Path:
        """
        Generate voice embedding using advanced audio processing
        """
        try:
            logger.info(f"Generating voice embedding for {voice_name} ({language})")

            # Apply quality-specific mel spectrogram configuration
            quality_config = self.quality_settings.get(
                self.config.TTS_NOTIFY_COQUI_CLONING_QUALITY,
                self.quality_settings["high"],
            )

            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.processing_config.sample_rate,
                n_fft=self.processing_config.n_fft,
                hop_length=quality_config["hop_length"],
                n_mels=quality_config["n_mels"],
                fmin=self.processing_config.f_min,
                fmax=self.processing_config.f_max,
                win_length=self.processing_config.win_length,
            )

            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Generate embedding (simplified for demonstration)
            # In real implementation, this would use neural network embeddings
            embedding = self._extract_audio_features(log_mel_spec)

            # Save embedding
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(
                c for c in voice_name if c.isalnum() or c in (" ", "-", "_")
            ).strip()
            embedding_filename = f"{safe_name}_{language}_{timestamp}.npy"
            embedding_path = self.embedding_dir / embedding_filename

            np.save(str(embedding_path), embedding)
            logger.info(f"Voice embedding saved: {embedding_path}")

            # Save source audio for persistence (required for XTTS)
            audio_filename = f"{safe_name}_{language}_{timestamp}.wav"
            persistent_audio_path = self.voice_audio_dir / audio_filename

            # Save the processed audio (or copy original if no processing needed, but we always have audio loaded)
            import soundfile as sf

            sf.write(
                str(persistent_audio_path), audio, self.processing_config.sample_rate
            )  # Use processing_config.sample_rate as audio was resampled
            logger.info(f"Source audio persisted: {persistent_audio_path}")

            return embedding_path

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise TTSError(f"Failed to generate voice embedding: {e}")

    def _extract_audio_features(self, log_mel_spec: np.ndarray) -> np.ndarray:
        """
        Extract features from mel spectrogram for voice embedding
        """
        try:
            # Basic feature extraction (simplified - real implementation would use neural networks)

            # Statistical features
            mean_features = np.mean(log_mel_spec, axis=1)
            std_features = np.std(log_mel_spec, axis=1)
            max_features = np.max(log_mel_spec, axis=1)
            min_features = np.min(log_mel_spec, axis=1)

            # Temporal features (derivatives)
            delta_features = np.diff(log_mel_spec, axis=1)
            delta_mean = np.mean(delta_features, axis=1)
            delta_std = np.std(delta_features, axis=1)

            # Combine features
            embedding = np.concatenate(
                [
                    mean_features,
                    std_features,
                    max_features,
                    min_features,
                    delta_mean,
                    delta_std,
                ]
            )

            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            return embedding

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise TTSError(f"Failed to extract audio features: {e}")

    async def _create_voice_profile(
        self, request: VoiceCloningRequest, embedding_path: Path, sample_duration: float
    ) -> VoiceProfile:
        """Create voice profile with metadata"""
        try:
            voice_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            # Save source audio for persistence (required for XTTS)
            safe_name = "".join(
                c for c in request.voice_name if c.isalnum() or c in (" ", "-", "_")
            ).strip()
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"{safe_name}_{request.language}_{timestamp_str}.wav"
            persistent_audio_path = self.voice_audio_dir / audio_filename

            # Copy source audio to persistent location
            # If we converted it, request.source_audio_path might be the original or temp
            # But we want to save a valid WAV.
            # Let's use the file that was actually used for processing if possible,
            # or convert/copy the original.

            # Since we don't have the temp path here easily (it was in clone_voice or _process_source_audio),
            # let's just convert/copy the input path to the persistent path using ffmpeg to ensure it's WAV
            try:
                self._convert_to_wav_file(
                    request.source_audio_path, persistent_audio_path
                )
                logger.info(f"Source audio persisted: {persistent_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to persist audio, using original path: {e}")
                persistent_audio_path = request.source_audio_path

            # Generate metadata
            metadata = {
                "source_file": str(persistent_audio_path),
                "original_source": str(request.source_audio_path),
                "cloning_quality": request.quality,
                "sample_rate": request.sample_rate,
                "auto_optimize": request.auto_optimize,
                "normalize": request.normalize,
                "denoise": request.denoise,
                "batch_size": request.batch_size,
                "language": request.language,
                "sample_duration": sample_duration,
                "config_version": "3.0.0",
            }

            # Add gender if provided
            if request.gender:
                metadata["gender"] = request.gender.value

            profile = VoiceProfile(
                voice_id=voice_id,
                name=request.voice_name,
                language=request.language,
                embedding_path=embedding_path,
                created_at=timestamp,
                metadata=metadata,
            )

            # Save profile metadata
            profile_path = self.profile_dir / f"{voice_id}.json"
            with open(profile_path, "w", encoding="utf-8") as f:
                # Convert to serializable format
                profile_data = {
                    "voice_id": profile.voice_id,
                    "name": profile.name,
                    "language": profile.language,
                    "embedding_path": str(profile.embedding_path),
                    "metadata": profile.metadata,
                    "created_at": profile.created_at,
                    "last_used": profile.last_used,
                    "usage_count": profile.usage_count,
                    "quality_score": profile.quality_score,
                    "is_active": profile.is_active,
                }
                json.dump(profile_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Voice profile created: {profile.voice_id}")
            return profile

        except Exception as e:
            logger.error(f"Profile creation failed: {e}")
            raise TTSError(f"Failed to create voice profile: {e}")

    async def _optimize_voice_profile(self, profile: VoiceProfile) -> float:
        """
        Optimize voice profile for better quality
        """
        try:
            logger.info(f"Optimizing voice profile: {profile.voice_id}")

            # Load embedding
            if not profile.embedding_path.exists():
                raise ValidationError(
                    f"Embedding file not found: {profile.embedding_path}"
                )

            embedding = np.load(profile.embedding_path)

            # Quality optimization (simplified)
            # Real implementation would use neural network fine-tuning

            # Basic quality assessment
            embedding_norm = np.linalg.norm(embedding)
            embedding_std = np.std(embedding)
            embedding_range = np.max(embedding) - np.min(embedding)

            # Calculate optimization score based on multiple factors
            score = min(1.0, (embedding_norm * embedding_std * embedding_range) / 100.0)

            # Enhance embedding if score is low
            if score < 0.5:
                # Apply smoothing and normalization
                embedding = self._enhance_embedding(embedding)

                # Save enhanced embedding
                enhanced_path = profile.embedding_path.with_suffix(".enhanced.npy")
                np.save(enhanced_path, embedding)

                # Update profile path
                profile.embedding_path = enhanced_path

            logger.info(f"Voice profile optimized with score: {score:.3f}")
            return score

        except Exception as e:
            logger.error(f"Profile optimization failed: {e}")
            return 0.0

    def _enhance_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Enhance embedding for better quality"""
        try:
            # Apply Gaussian smoothing
            from scipy.ndimage import gaussian_filter1d

            enhanced = gaussian_filter1d(embedding, sigma=1.0)

            # Re-normalize
            enhanced = enhanced / (np.linalg.norm(enhanced) + 1e-8)

            return enhanced

        except ImportError:
            # Fallback without scipy
            return embedding / (np.linalg.norm(embedding) + 1e-8)
        except Exception as e:
            logger.warning(f"Embedding enhancement failed: {e}")
            return embedding

    async def _create_cloned_voice(
        self,
        request: VoiceCloningRequest,
        profile: VoiceProfile,
        sample_duration: float,
    ) -> Voice:
        """Create Voice object from profile"""
        try:
            # Map language to Language enum
            language_map = {
                "en": Language.ENGLISH,
                "es": Language.SPANISH,
                "fr": Language.FRENCH,
                "de": Language.GERMAN,
                "it": Language.ITALIAN,
                "pt": Language.PORTUGUESE,
                "nl": Language.UNKNOWN,
                "pl": Language.UNKNOWN,
                "ru": Language.UNKNOWN,
                "zh": Language.CHINESE,
                "ja": Language.JAPANESE,
                "ko": Language.KOREAN,
            }

            voice_language = language_map.get(request.language, Language.UNKNOWN)

            # Determine quality
            quality_map = {
                "low": VoiceQuality.BASIC,
                "medium": VoiceQuality.ENHANCED,
                "high": VoiceQuality.PREMIUM,
                "ultra": VoiceQuality.NEURAL,
            }
            voice_quality = quality_map.get(request.quality, VoiceQuality.PREMIUM)

            # Create voice ID
            voice_id = f"cloned_{profile.voice_id[:8]}"

            # Create Voice object
            voice = Voice(
                id=voice_id,
                name=request.voice_name,
                language=voice_language,
                locale=request.language,
                gender=request.gender or Gender.UNKNOWN,
                quality=voice_quality,
                description=f"Cloned voice from {request.source_audio_path.name}",
                engine_name="coqui-cloned",
                sample_rate=request.sample_rate,
                supported_formats=["wav", "mp3", "ogg"],
                is_cloned=True,
                cloning_source=str(request.source_audio_path),
                cloning_quality=request.quality,
                cloning_language=request.language,
                embedding_path=str(profile.embedding_path),
                profile_path=str(self.profile_dir / f"{profile.voice_id}.json"),
                created_at=profile.created_at,
                sample_duration=sample_duration,
                optimization_score=profile.quality_score,
                metadata={
                    "profile_id": profile.voice_id,
                    "cloning_timestamp": profile.created_at,
                    "auto_optimized": request.auto_optimize,
                },
            )

            logger.info(f"Cloned voice created: {voice.name} ({voice.id})")
            return voice

        except Exception as e:
            logger.error(f"Cloned voice creation failed: {e}")
            raise TTSError(f"Failed to create cloned voice: {e}")

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio signal"""
        try:
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio / rms * 0.1  # Target RMS of 0.1

            # Peak normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95

            return audio

        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}")
            return audio

    def _denoise_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply advanced denoising to audio"""
        try:
            # 1. Apply High-Pass Filter (remove rumble < 80Hz)
            from scipy import signal

            sos = signal.butter(
                10,
                80,
                btype="high",
                fs=self.processing_config.sample_rate,
                output="sos",
            )
            audio = signal.sosfilt(sos, audio)

            # 2. Try using noisereduce library (best quality)
            try:
                import noisereduce as nr

                logger.info("Denoising with noisereduce library")
                return nr.reduce_noise(y=audio, sr=self.processing_config.sample_rate)
            except ImportError:
                # 3. Fallback to custom Spectral Gating (remove broadband noise)
                logger.info(
                    "noisereduce not found, falling back to internal spectral gating"
                )
                return self._spectral_gating(audio)

        except ImportError:
            logger.warning("Scipy not available, skipping denoising")
            return audio
        except Exception as e:
            logger.warning(f"Audio denoising failed: {e}")
            return audio

    def _spectral_gating(
        self, audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512
    ) -> np.ndarray:
        """
        Implement basic spectral gating for noise reduction
        Assumes the first 0.5s of audio is noise
        """
        try:
            from scipy import signal

            # Compute STFT
            f, t, Zxx = signal.stft(
                audio,
                fs=self.processing_config.sample_rate,
                nperseg=n_fft,
                noverlap=hop_length,
            )

            # Estimate noise profile from first 0.5 seconds (or 10% if shorter)
            noise_duration_samples = int(0.5 * self.processing_config.sample_rate)
            noise_duration_samples = min(noise_duration_samples, len(audio) // 10)

            # If audio is too short, skip spectral gating
            if noise_duration_samples < hop_length:
                return audio

            # Calculate noise threshold
            noise_segment = audio[:noise_duration_samples]
            _, _, Zxx_noise = signal.stft(
                noise_segment,
                fs=self.processing_config.sample_rate,
                nperseg=n_fft,
                noverlap=hop_length,
            )

            # Mean noise amplitude per frequency bin
            noise_mean = np.mean(np.abs(Zxx_noise), axis=1, keepdims=True)

            # Threshold multiplier (sensitivity)
            threshold = noise_mean * 1.5

            # Generate mask
            spectrogram = np.abs(Zxx)
            mask = spectrogram > threshold

            # Smooth mask (optional, simple version here)

            # Apply mask
            Zxx_denoised = Zxx * mask

            # Inverse STFT
            _, audio_denoised = signal.istft(
                Zxx_denoised,
                fs=self.processing_config.sample_rate,
                nperseg=n_fft,
                noverlap=hop_length,
            )

            # Match length
            if len(audio_denoised) > len(audio):
                audio_denoised = audio_denoised[: len(audio)]
            elif len(audio_denoised) < len(audio):
                audio_denoised = np.pad(
                    audio_denoised, (0, len(audio) - len(audio_denoised))
                )

            return audio_denoised

        except Exception as e:
            logger.warning(f"Spectral gating failed: {e}")
            return audio

    async def list_cloned_voices(self) -> List[Voice]:
        """List all available cloned voices"""
        try:
            voices = []

            if not self.profile_dir.exists():
                return voices

            # Load all voice profiles
            for profile_file in self.profile_dir.glob("*.json"):
                try:
                    with open(profile_file, "r", encoding="utf-8") as f:
                        profile_data = json.load(f)

                    # Create Voice object
                    voice = Voice(
                        id=f"cloned_{profile_data['voice_id'][:8]}",
                        name=profile_data["name"],
                        language=Language.UNKNOWN,  # Would need proper mapping
                        cloning_quality=profile_data["metadata"].get("cloning_quality"),
                        cloning_source=profile_data["metadata"].get("source_file"),
                        is_cloned=True,
                        embedding_path=profile_data["embedding_path"],
                        profile_path=str(profile_file),
                        created_at=profile_data["created_at"],
                        sample_duration=profile_data["metadata"].get("sample_duration"),
                        optimization_score=profile_data.get("quality_score"),
                        metadata=profile_data["metadata"],
                    )

                    voices.append(voice)

                except Exception as e:
                    logger.warning(f"Failed to load profile {profile_file}: {e}")

            return sorted(voices, key=lambda v: v.created_at or "", reverse=True)

        except Exception as e:
            logger.error(f"Failed to list cloned voices: {e}")
            return []

    async def delete_cloned_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice"""
        try:
            # Find profile
            profile_file = self.profile_dir / f"{voice_id}.json"
            if not profile_file.exists():
                # Try to find by voice ID prefix
                for profile_path in self.profile_dir.glob("*.json"):
                    if profile_path.name.startswith(voice_id):
                        profile_file = profile_path
                        break

            if not profile_file.exists():
                return False

            # Load profile to get embedding path
            with open(profile_file, "r", encoding="utf-8") as f:
                profile_data = json.load(f)

            embedding_path = Path(profile_data["embedding_path"])

            # Delete files
            if embedding_path.exists():
                embedding_path.unlink()

            profile_file.unlink()

            # Remove from cache
            if voice_id in self._cloning_cache:
                del self._cloning_cache[voice_id]

            logger.info(f"Cloned voice deleted: {voice_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete cloned voice {voice_id}: {e}")
            return False

    async def get_cloning_status(self) -> Dict[str, Any]:
        """Get voice cloning system status"""
        try:
            # Count profiles
            profile_count = len(list(self.profile_dir.glob("*.json")))

            # Count embeddings
            embedding_count = len(list(self.embedding_dir.glob("*.npy")))

            # Calculate storage usage
            profile_size = sum(
                f.stat().st_size for f in self.profile_dir.glob("*.json") if f.is_file()
            )
            embedding_size = sum(
                f.stat().st_size
                for f in self.embedding_dir.glob("*.npy")
                if f.is_file()
            )

            return {
                "enabled": self.config.TTS_NOTIFY_COQUI_ENABLE_CLONING,
                "profile_count": profile_count,
                "embedding_count": embedding_count,
                "storage_usage": {
                    "profiles_mb": profile_size / (1024 * 1024),
                    "embeddings_mb": embedding_size / (1024 * 1024),
                    "total_mb": (profile_size + embedding_size) / (1024 * 1024),
                },
                "directories": {
                    "profiles": str(self.profile_dir),
                    "embeddings": str(self.embedding_dir),
                    "temp": str(self._temp_dir),
                },
                "quality": self.config.TTS_NOTIFY_COQUI_CLONING_QUALITY,
                "auto_optimize": self.config.TTS_NOTIFY_COQUI_AUTO_OPTIMIZE_CLONE,
            }

        except Exception as e:
            logger.error(f"Failed to get cloning status: {e}")
            return {"error": str(e)}


# Global voice cloner instance
voice_cloner = VoiceCloner()
