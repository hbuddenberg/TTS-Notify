"""
Data models for TTS Notify v2

This module defines all data models used throughout the TTS Notify system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path


class Gender(Enum):
    """Voice gender enumeration"""
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class VoiceQuality(Enum):
    """Voice quality levels"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    PREMIUM = "premium"
    SIRI = "siri"
    NEURAL = "neural"


class Language(Enum):
    """Supported languages"""
    SPANISH = "es"
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    JAPANESE = "ja"
    CHINESE = "zh"
    KOREAN = "ko"
    RUSSIAN = "ru"
    POLISH = "pl"
    DUTCH = "nl"
    CZECH = "cs"
    ARABIC = "ar"
    TURKISH = "tr"
    HUNGARIAN = "hu"
    FINNISH = "fi"
    UNKNOWN = "unknown"


class AudioFormat(Enum):
    """Supported audio formats"""
    AIFF = "aiff"
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    M4A = "m4a"
    FLAC = "flac"


class TTSEngineType(Enum):
    """TTS Engine types"""
    MACOS = "macos"
    COQUI = "coqui"


@dataclass
class Voice:
    """Voice information model (v3.0.0 with cloning support)"""
    id: str
    name: str
    language: Language
    locale: Optional[str] = None
    gender: Gender = Gender.UNKNOWN
    quality: VoiceQuality = VoiceQuality.BASIC
    description: Optional[str] = None
    engine_name: Optional[str] = None
    sample_rate: Optional[int] = None
    supported_formats: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Phase B: Voice Cloning Support
    is_cloned: bool = False
    cloning_source: Optional[str] = None  # Source file or original voice
    cloning_quality: Optional[str] = None  # low, medium, high, ultra
    cloning_language: Optional[str] = None  # Language used for cloning
    embedding_path: Optional[str] = None  # Path to voice embedding
    profile_path: Optional[str] = None  # Path to voice profile
    created_at: Optional[str] = None  # Timestamp when voice was created
    sample_duration: Optional[float] = None  # Duration of source sample
    optimization_score: Optional[float] = None  # Quality optimization score (0.0-1.0)

    def __post_init__(self):
        """Post-initialization processing"""
        if not self.name:
            self.name = self.id

        # Normalize locale format
        if self.locale and '-' in self.locale:
            # Convert to standard format (e.g., es-ES -> es_ES)
            parts = self.locale.split('-')
            if len(parts) == 2:
                self.locale = f"{parts[0].lower()}_{parts[1].upper()}"

    @property
    def display_name(self) -> str:
        """Get display name with quality indicator"""
        quality_suffix = ""
        if self.quality != VoiceQuality.BASIC:
            quality_suffix = f" ({self.quality.value.title()})"
        return f"{self.name}{quality_suffix}"

    def matches_query(self, query: str, fuzzy: bool = True) -> bool:
        """Check if voice matches a search query"""
        if not query:
            return True

        # Normalize query and voice name for comparison
        import unicodedata

        def normalize_text(text: str) -> str:
            nfd = unicodedata.normalize("NFD", text)
            without_accents = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
            return without_accents.lower()

        normalized_query = normalize_text(query.lower())
        normalized_name = normalize_text(self.name.lower())
        normalized_id = normalize_text(self.id.lower())

        # Exact match
        if normalized_query == normalized_name or normalized_query == normalized_id:
            return True

        if not fuzzy:
            return False

        # Prefix match
        if normalized_name.startswith(normalized_query) or normalized_id.startswith(normalized_query):
            return True

        # Partial match
        return normalized_query in normalized_name or normalized_query in normalized_id


@dataclass
class TTSRequest:
    """TTS request configuration"""
    text: str
    voice: Voice
    rate: Optional[int] = None  # Words per minute
    pitch: Optional[float] = None  # Pitch multiplier
    volume: Optional[float] = None  # Volume multiplier (0.0-1.0)
    output_format: AudioFormat = AudioFormat.AIFF
    output_path: Optional[Path] = None
    engine_type: Optional[TTSEngineType] = None  # TTS engine type
    language: Optional[str] = None  # Language override (auto, en, es, fr, etc.)
    force_language: bool = False  # Force specific language
    model_name: Optional[str] = None  # CoquiTTS model name
    auto_download: Optional[bool] = None  # Auto-download models
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate request parameters"""
        if not self.text or not self.text.strip():
            from .exceptions import ValidationError
            raise ValidationError("Text cannot be empty")

        if self.rate is not None and (self.rate < 50 or self.rate > 500):
            from .exceptions import ValidationError
            raise ValidationError("Rate must be between 50 and 500 WPM", field="rate", value=self.rate)

        if self.pitch is not None and (self.pitch < 0.5 or self.pitch > 2.0):
            from .exceptions import ValidationError
            raise ValidationError("Pitch must be between 0.5 and 2.0", field="pitch", value=self.pitch)

        if self.volume is not None and (self.volume < 0.0 or self.volume > 1.0):
            from .exceptions import ValidationError
            raise ValidationError("Volume must be between 0.0 and 1.0", field="volume", value=self.volume)

        if self.metadata is None:
            self.metadata = {}


@dataclass
class TTSResponse:
    """TTS response information"""
    success: bool
    audio_data: Optional[bytes] = None
    file_path: Optional[Path] = None
    duration: Optional[float] = None
    format: Optional[AudioFormat] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self):
        """Post-processing of response"""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class HealthReport:
    """Health check report"""
    status: str  # "healthy", "degraded", "unhealthy"
    voice_system: Dict[str, Any]
    disk_space: Dict[str, Any]
    dependencies: Dict[str, Any]
    configuration: Dict[str, Any]
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceCloningRequest:
    """Voice cloning request model (Phase B)"""
    source_audio_path: Path
    voice_name: str
    language: str  # Language code for cloning
    gender: Optional[Gender] = None
    quality: str = "high"  # low, medium, high, ultra
    sample_rate: int = 22050
    normalize: bool = True
    denoise: bool = True
    description: Optional[str] = None
    auto_optimize: bool = True
    batch_size: int = 1
    timeout: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate cloning request parameters"""
        if not self.source_audio_path.exists():
            from .exceptions import ValidationError
            raise ValidationError(f"Source audio file not found: {self.source_audio_path}")

        if self.quality not in ["low", "medium", "high", "ultra"]:
            from .exceptions import ValidationError
            raise ValidationError(f"Invalid quality level: {self.quality}")

        if self.language not in ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"]:
            from .exceptions import ValidationError
            raise ValidationError(f"Unsupported language for cloning: {self.language}")


@dataclass
class VoiceCloningResponse:
    """Voice cloning response model (Phase B)"""
    success: bool
    voice: Optional[Voice] = None
    embedding_path: Optional[Path] = None
    profile_path: Optional[Path] = None
    processing_time: Optional[float] = None
    optimization_score: Optional[float] = None
    sample_duration: Optional[float] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceProfile:
    """Voice profile for cloned voices (Phase B)"""
    voice_id: str
    name: str
    language: str
    embedding_path: Path
    created_at: str
    last_used: Optional[str] = None
    usage_count: int = 0
    quality_score: Optional[float] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstallationReport:
    """Installation report"""
    success: bool
    components_installed: List[str]
    failed_components: List[str]
    warnings: List[str]
    errors: List[str]
    installation_path: Path
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)