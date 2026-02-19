"""
Configuration Manager for TTS Notify v2

This module provides intelligent environment variable management and configuration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator

from .exceptions import ConfigurationError


class TTSConfig(BaseModel):
    """Configuration model with environment variable support"""

    # Engine Selection
    TTS_NOTIFY_ENGINE: str = Field(default="macos", pattern=r"^(macos|coqui)$", description="Default TTS engine")

    # Voice settings with validation
    TTS_NOTIFY_VOICE: str = Field(default="monica", description="Default voice")
    TTS_NOTIFY_RATE: int = Field(default=175, ge=50, le=500, description="Speech rate in WPM")
    TTS_NOTIFY_LANGUAGE: str = Field(default="es", description="Default language")
    TTS_NOTIFY_QUALITY: str = Field(default="basic", pattern=r"^(basic|enhanced|premium|siri|neural)$", description="Voice quality")
    TTS_NOTIFY_PITCH: float = Field(default=1.0, ge=0.5, le=2.0, description="Pitch multiplier")
    TTS_NOTIFY_VOLUME: float = Field(default=1.0, ge=0.0, le=1.0, description="Volume multiplier (0.0-1.0)")

    # CoquiTTS Engine Configuration
    TTS_NOTIFY_COQUI_MODEL: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2", description="CoquiTTS model name")
    TTS_NOTIFY_COQUI_FALLBACK_MODEL: Optional[str] = Field(default=None, description="Fallback CoquiTTS model if main model fails")
    TTS_NOTIFY_COQUI_AUTO_FALLBACK: bool = Field(default=True, description="Automatically fallback to language-specific models if multilingual fails")
    TTS_NOTIFY_COQUI_DEVICE: str = Field(default="auto", pattern=r"^(auto|cpu|cuda)$", description="CoquiTTS device")
    TTS_NOTIFY_COQUI_USE_GPU: bool = Field(default=True, description="Use GPU acceleration for CoquiTTS")
    TTS_NOTIFY_COQUI_AUTO_INIT: bool = Field(default=True, description="Auto-initialize CoquiTTS on startup")
    TTS_NOTIFY_COQUI_SPEAKER: Optional[str] = Field(default=None, description="Default CoquiTTS speaker")
    TTS_NOTIFY_COQUI_STYLE: Optional[str] = Field(default=None, description="Default CoquiTTS style")

    # Multi-Language Support (NEW)
    TTS_NOTIFY_DEFAULT_LANGUAGE: str = Field(default="auto", pattern=r"^(auto|en|es|fr|de|it|pt|nl|pl|ru|zh|ja|ko)$", description="Preferred language (auto=detection)")
    TTS_NOTIFY_COQUI_LANGUAGE_FALLBACK: str = Field(default="es", pattern=r"^(en|es|fr|de|it|pt)$", description="Fallback language for CoquiTTS")
    TTS_NOTIFY_FORCE_LANGUAGE: bool = Field(default=False, description="Force specific language ignoring detection")

    # Model Management (NEW)
    TTS_NOTIFY_AUTO_DOWNLOAD_MODELS: bool = Field(default=True, description="Automatically download missing models")
    TTS_NOTIFY_COQUI_CACHE_MODELS: bool = Field(default=True, description="Cache downloaded models")
    TTS_NOTIFY_COQUI_MODEL_CACHE_DIR: Optional[str] = Field(default=None, description="Model cache directory")
    TTS_NOTIFY_COQUI_MODEL_TIMEOUT: int = Field(default=300, ge=60, le=1800, description="Model download timeout (seconds)")
    TTS_NOTIFY_COQUI_OFFLINE_MODE: bool = Field(default=False, description="Offline mode (no downloads)")

    # Voice Cloning Configuration (Phase B)
    TTS_NOTIFY_COQUI_ENABLE_CLONING: bool = Field(default=True, description="Enable voice cloning")
    TTS_NOTIFY_COQUI_PROFILE_DIR: Optional[str] = Field(default=None, description="Voice profile directory")
    TTS_NOTIFY_COQUI_EMBEDDING_DIR: Optional[str] = Field(default=None, description="Embedding directory")
    TTS_NOTIFY_COQUI_MIN_SAMPLE_SECONDS: float = Field(default=2.0, ge=1.0, le=30.0, description="Minimum sample length for cloning")
    TTS_NOTIFY_COQUI_MAX_SAMPLE_SECONDS: float = Field(default=300.0, ge=0.0, description="Maximum sample length for cloning (0=unlimited)")
    TTS_NOTIFY_COQUI_CLONING_QUALITY: str = Field(default="high", pattern=r"^(low|medium|high|ultra)$", description="Voice cloning quality level")
    TTS_NOTIFY_COQUI_CLONING_BATCH_SIZE: int = Field(default=1, ge=1, le=16, description="Batch size for cloning processing")
    TTS_NOTIFY_COQUI_CLONING_DENOISE: bool = Field(default=True, description="Enable audio denoising for cloning")
    TTS_NOTIFY_COQUI_CLONING_NORMALIZE: bool = Field(default=True, description="Enable audio normalization for cloning")
    TTS_NOTIFY_COQUI_VOICE_NAME_TEMPLATE: str = Field(default="custom_voice_{lang}_{timestamp}", description="Template for custom voice names")
    TTS_NOTIFY_COQUI_CLONING_TIMEOUT: int = Field(default=300, ge=60, le=1800, description="Voice cloning timeout in seconds")
    TTS_NOTIFY_COQUI_AUTO_OPTIMIZE_CLONE: bool = Field(default=True, description="Automatically optimize cloned voice quality")

    # Audio Pipeline Configuration
    TTS_NOTIFY_COQUI_AUTO_CLEAN_AUDIO: bool = Field(default=True, description="Auto-clean audio for cloning")
    TTS_NOTIFY_COQUI_AUTO_TRIM_SILENCE: bool = Field(default=True, description="Auto-trim silence from audio")
    TTS_NOTIFY_COQUI_NOISE_REDUCTION: bool = Field(default=False, description="Enable noise reduction")
    TTS_NOTIFY_COQUI_DIARIZATION: bool = Field(default=False, description="Enable audio diarization")

    # Format Conversion
    TTS_NOTIFY_COQUI_CONVERSION_ENABLED: bool = Field(default=False, description="Enable format conversion")
    TTS_NOTIFY_COQUI_TARGET_FORMATS: str = Field(default="wav", description="Target audio formats (comma-separated)")
    TTS_NOTIFY_COQUI_EMBEDDING_CACHE: bool = Field(default=True, description="Cache voice embeddings")
    TTS_NOTIFY_COQUI_EMBEDDING_FORMAT: str = Field(default="npy", pattern=r"^(npy|pt)$", description="Embedding format")

    # Functionality flags
    TTS_NOTIFY_ENABLED: bool = Field(default=True, description="Enable TTS globally")
    TTS_NOTIFY_CACHE_ENABLED: bool = Field(default=True, description="Enable voice detection cache")
    TTS_NOTIFY_STREAMING: bool = Field(default=False, description="Enable streaming for long responses")
    TTS_NOTIFY_AUTO_SAVE: bool = Field(default=False, description="Auto-save responses")
    TTS_NOTIFY_CONFIRMATION: bool = Field(default=False, description="Enable command confirmation")

    # System and performance settings
    TTS_NOTIFY_LOG_LEVEL: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARN|ERROR)$", description="Logging level")
    TTS_NOTIFY_MAX_CONCURRENT: int = Field(default=5, ge=1, le=50, description="Max concurrent requests")
    TTS_NOTIFY_TIMEOUT: int = Field(default=60, ge=5, le=300, description="Operation timeout in seconds")
    TTS_NOTIFY_CACHE_TTL: int = Field(default=300, ge=30, le=3600, description="Cache TTL in seconds")
    TTS_NOTIFY_MAX_TEXT_LENGTH: int = Field(default=5000, ge=100, le=50000, description="Max text length")

    # Format and output settings
    TTS_NOTIFY_OUTPUT_FORMAT: str = Field(default="aiff", pattern=r"^(aiff|wav|mp3|ogg|m4a|flac)$", description="Audio output format")
    TTS_NOTIFY_OUTPUT_DIR: str = Field(default="", description="Output directory (default: Desktop)")
    TTS_NOTIFY_SAMPLE_RATE: int = Field(default=22050, description="Sample rate in Hz")
    TTS_NOTIFY_CHANNELS: int = Field(default=1, ge=1, le=2, description="Audio channels (1=mono, 2=stereo)")

    # Interface-specific settings
    TTS_NOTIFY_CLI_FORMAT: str = Field(default="table", pattern=r"^(table|json|yaml|csv)$", description="CLI output format")
    TTS_NOTIFY_API_PORT: int = Field(default=8000, ge=1024, le=65535, description="API server port")
    TTS_NOTIFY_API_HOST: str = Field(default="localhost", description="API server host")
    TTS_NOTIFY_MCP_TOOLS: str = Field(default="all", description="MCP tools to expose")
    TTS_NOTIFY_PROFILE: str = Field(default="default", description="Configuration profile")

    # Advanced settings
    TTS_NOTIFY_DEBUG_MODE: bool = Field(default=False, description="Enable debug mode")
    TTS_NOTIFY_VERBOSE: bool = Field(default=False, description="Enable verbose output")
    TTS_NOTIFY_EXPERIMENTAL: bool = Field(default=False, description="Enable experimental features")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"

    @validator('TTS_NOTIFY_OUTPUT_DIR', pre=True, always=True)
    def set_default_output_dir(cls, v):
        """Set default output directory to Desktop if empty"""
        if not v:
            import subprocess
            desktop_path = Path.home() / "Desktop"
            if desktop_path.exists():
                return str(desktop_path)
        return v

    @validator('TTS_NOTIFY_LANGUAGE', pre=True, always=True)
    def normalize_language(cls, v):
        """Normalize language code"""
        if v:
            return v.lower()
        return v

    @classmethod
    def load_from_env(cls, env_prefix: str = "TTS_NOTIFY_") -> 'TTSConfig':
        """Load configuration from environment variables"""
        try:
            # Create a dictionary of environment variables with the correct prefix
            env_vars = {}
            for key, value in os.environ.items():
                if key.startswith(env_prefix):
                    config_key = key
                    env_vars[config_key] = value

            return cls(**env_vars)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from environment: {e}")

    def get_active_vars(self) -> List[str]:
        """Get list of active environment variables"""
        active_vars = []
        for field_name, field_info in self.__fields__.items():
            value = getattr(self, field_name)
            # Check if the value differs from default
            default_value = field_info.default
            if value != default_value:
                active_vars.append(field_name)
        return active_vars

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.dict(exclude_unset=True)

    def get_profile_config(self, profile_name: str) -> 'TTSConfig':
        """Get configuration for a specific profile"""
        profiles = {
            "claude-desktop": {
                "TTS_NOTIFY_VOICE": "jorge",
                "TTS_NOTIFY_RATE": 175,
                "TTS_NOTIFY_LANGUAGE": "es",
                "TTS_NOTIFY_QUALITY": "enhanced",
                "TTS_NOTIFY_MAX_TEXT_LENGTH": 2000,
                "TTS_NOTIFY_CONFIRMATION": False,
                "TTS_NOTIFY_LOG_LEVEL": "INFO",
                "TTS_NOTIFY_PROFILE": "claude-desktop"
            },
            "api-server": {
                "TTS_NOTIFY_MAX_CONCURRENT": 10,
                "TTS_NOTIFY_TIMEOUT": 120,
                "TTS_NOTIFY_CACHE_ENABLED": True,
                "TTS_NOTIFY_LOG_LEVEL": "INFO",
                "TTS_NOTIFY_API_PORT": 8000,
                "TTS_NOTIFY_PROFILE": "api-server"
            },
            "cli-default": {
                "TTS_NOTIFY_CLI_FORMAT": "table",
                "TTS_NOTIFY_AUTO_SAVE": False,
                "TTS_NOTIFY_VERBOSE": False,
                "TTS_NOTIFY_PROFILE": "cli-default"
            },
            "development": {
                "TTS_NOTIFY_DEBUG_MODE": True,
                "TTS_NOTIFY_VERBOSE": True,
                "TTS_NOTIFY_LOG_LEVEL": "DEBUG",
                "TTS_NOTIFY_CACHE_TTL": 60,
                "TTS_NOTIFY_PROFILE": "development"
            },
            "production": {
                "TTS_NOTIFY_MAX_CONCURRENT": 5,
                "TTS_NOTIFY_TIMEOUT": 60,
                "TTS_NOTIFY_LOG_LEVEL": "WARN",
                "TTS_NOTIFY_EXPERIMENTAL": False,
                "TTS_NOTIFY_PROFILE": "production"
            }
        }

        profile_config = profiles.get(profile_name, {})
        # Merge with current config
        current_config = self.dict()
        current_config.update(profile_config)
        return TTSConfig(**current_config)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        try:
            # Pydantic validation is done automatically
            pass
        except Exception as e:
            errors.append(f"Configuration validation failed: {e}")

        # Custom validations
        if self.TTS_NOTIFY_RATE < 100 or self.TTS_NOTIFY_RATE > 300:
            errors.append("TTS_NOTIFY_RATE should be between 100-300 WPM for optimal performance")

        if self.TTS_NOTIFY_OUTPUT_DIR and not Path(self.TTS_NOTIFY_OUTPUT_DIR).exists():
            try:
                Path(self.TTS_NOTIFY_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            except Exception:
                errors.append(f"Cannot create output directory: {self.TTS_NOTIFY_OUTPUT_DIR}")

        return errors


class ConfigManager:
    """Centralized configuration management"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent.parent.parent / "config"
        self.config_file = self.config_dir / "default.yaml"
        self.profiles_file = self.config_dir / "profiles.yaml"
        self._config: Optional[TTSConfig] = None
        self._profiles: Dict[str, Dict] = {}

    def load_config(self, profile: Optional[str] = None) -> TTSConfig:
        """Load configuration from environment and optional profile"""
        try:
            # Load base configuration from environment
            config = TTSConfig.load_from_env()

            # Apply profile if specified
            if profile and profile != "default":
                config = config.get_profile_config(profile)
            elif config.TTS_NOTIFY_PROFILE != "default":
                config = config.get_profile_config(config.TTS_NOTIFY_PROFILE)

            # Load additional configuration from YAML files if they exist
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        config_dict = config.dict()
                        config_dict.update(yaml_config)
                        config = TTSConfig(**config_dict)

            self._config = config
            return config

        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def get_config(self) -> TTSConfig:
        """Get current configuration"""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def reload_config(self, profile: Optional[str] = None) -> TTSConfig:
        """Reload configuration from sources"""
        self._config = None
        return self.load_config(profile)

    def save_config(self, config: TTSConfig, config_file: Optional[Path] = None) -> None:
        """Save configuration to YAML file"""
        try:
            config_file = config_file or self.config_file
            config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(config_file, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)

        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def load_profiles(self) -> Dict[str, Dict]:
        """Load configuration profiles from YAML file"""
        try:
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r') as f:
                    self._profiles = yaml.safe_load(f) or {}
            else:
                self._profiles = self._get_default_profiles()
            return self._profiles
        except Exception as e:
            raise ConfigurationError(f"Failed to load profiles: {e}")

    def _get_default_profiles(self) -> Dict[str, Dict]:
        """Get default configuration profiles"""
        return {
            "claude-desktop": {
                "TTS_NOTIFY_VOICE": "jorge",
                "TTS_NOTIFY_RATE": 175,
                "TTS_NOTIFY_LANGUAGE": "es",
                "TTS_NOTIFY_QUALITY": "enhanced",
                "TTS_NOTIFY_MAX_TEXT_LENGTH": 2000,
                "TTS_NOTIFY_CONFIRMATION": False,
                "TTS_NOTIFY_LOG_LEVEL": "INFO"
            },
            "api-server": {
                "TTS_NOTIFY_MAX_CONCURRENT": 10,
                "TTS_NOTIFY_TIMEOUT": 120,
                "TTS_NOTIFY_CACHE_ENABLED": True,
                "TTS_NOTIFY_LOG_LEVEL": "INFO",
                "TTS_NOTIFY_API_PORT": 8000
            },
            "cli-default": {
                "TTS_NOTIFY_CLI_FORMAT": "table",
                "TTS_NOTIFY_AUTO_SAVE": False,
                "TTS_NOTIFY_VERBOSE": False
            },
            "development": {
                "TTS_NOTIFY_DEBUG_MODE": True,
                "TTS_NOTIFY_VERBOSE": True,
                "TTS_NOTIFY_LOG_LEVEL": "DEBUG",
                "TTS_NOTIFY_CACHE_TTL": 60
            },
            "production": {
                "TTS_NOTIFY_MAX_CONCURRENT": 5,
                "TTS_NOTIFY_TIMEOUT": 60,
                "TTS_NOTIFY_LOG_LEVEL": "WARN",
                "TTS_NOTIFY_EXPERIMENTAL": False
            }
        }

    def get_execution_context(self) -> str:
        """Detect execution context and suggest appropriate profile"""
        if "CLAUDE_DESKTOP" in os.environ or "MCP_SERVER" in os.environ:
            return "claude-desktop"
        elif "API_SERVER" in os.environ or "FASTAPI" in os.environ:
            return "api-server"
        elif "TESTING" in os.environ:
            return "development"
        else:
            return "cli-default"

    def get_mcp_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for MCP configuration"""
        config = self.get_config()

        # Filter to relevant MCP environment variables
        mcp_vars = {
            "TTS_NOTIFY_VOICE": config.TTS_NOTIFY_VOICE,
            "TTS_NOTIFY_RATE": str(config.TTS_NOTIFY_RATE),
            "TTS_NOTIFY_LANGUAGE": config.TTS_NOTIFY_LANGUAGE,
            "TTS_NOTIFY_QUALITY": config.TTS_NOTIFY_QUALITY,
            "TTS_NOTIFY_ENABLED": str(config.TTS_NOTIFY_ENABLED).lower(),
            "TTS_NOTIFY_CACHE_ENABLED": str(config.TTS_NOTIFY_CACHE_ENABLED).lower(),
            "TTS_NOTIFY_CONFIRMATION": str(config.TTS_NOTIFY_CONFIRMATION).lower(),
            "TTS_NOTIFY_LOG_LEVEL": config.TTS_NOTIFY_LOG_LEVEL,
            "TTS_NOTIFY_MAX_TEXT_LENGTH": str(config.TTS_NOTIFY_MAX_TEXT_LENGTH),
            "TTS_NOTIFY_OUTPUT_FORMAT": config.TTS_NOTIFY_OUTPUT_FORMAT,
            "TTS_NOTIFY_PROFILE": config.TTS_NOTIFY_PROFILE
        }

        return mcp_vars

    def validate_system(self) -> List[str]:
        """Validate system configuration and return issues"""
        issues = []
        config = self.get_config()

        # Validate configuration
        config_errors = config.validate()
        issues.extend(config_errors)

        # Validate system requirements
        try:
            import subprocess
            result = subprocess.run(["say", "-v", "?"],
                                   capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                issues.append("macOS 'say' command is not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            issues.append("macOS 'say' command is not available")

        # Validate output directory
        if config.TTS_NOTIFY_OUTPUT_DIR:
            output_path = Path(config.TTS_NOTIFY_OUTPUT_DIR)
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except Exception:
                    issues.append(f"Cannot access output directory: {output_path}")

        return issues


# Global configuration manager instance
config_manager = ConfigManager()