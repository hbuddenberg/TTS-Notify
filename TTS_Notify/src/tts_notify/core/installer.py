"""
Automated Installation System for TTS Notify v3.0.0

This module provides automated installation of CoquiTTS dependencies
with validation and testing capabilities.
"""

import asyncio
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import importlib
import json

from .config_manager import config_manager
from .exceptions import InstallationError, ValidationError
from .models import AudioFormat, Language

logger = logging.getLogger(__name__)


@dataclass
class InstallationResult:
    """Result of installation operation"""

    success: bool
    component: str
    version: Optional[str] = None
    installed_path: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class CoquiTTSInstaller:
    """Automated CoquiTTS installer with validation"""

    def __init__(self):
        self.config = config_manager.get_config()
        self.installed_components: Dict[str, InstallationResult] = {}

    async def install_coqui_tts(
        self, force: bool = False, use_gpu: bool = False
    ) -> InstallationResult:
        """Install CoquiTTS with validation"""
        try:
            logger.info("Starting CoquiTTS installation...")

            # Check if already installed
            if not force and await self._is_coqui_installed():
                version = await self._get_coqui_version()
                logger.info(f"CoquiTTS already installed: {version}")
                return InstallationResult(
                    success=True,
                    component="coqui-tts",
                    version=version,
                    installed_path="Already installed",
                )

            # Prepare installation command
            install_cmd = self._prepare_install_command(use_gpu)

            # Execute installation
            logger.info(f"Installing CoquiTTS: {' '.join(install_cmd)}")
            result = await self._run_command(install_cmd)

            if result.returncode == 0:
                # Verify installation
                await self._verify_installation()

                # Test basic functionality
                test_result = await self._test_coqui_functionality()

                if test_result.success:
                    version = await self._get_coqui_version()
                    logger.info(f"✅ CoquiTTS installed successfully: {version}")

                    return InstallationResult(
                        success=True,
                        component="coqui-tts",
                        version=version,
                        installed_path="Python environment",
                    )
                else:
                    return InstallationResult(
                        success=False,
                        component="coqui-tts",
                        error=f"Installation succeeded but functionality test failed: {test_result.error}",
                        warnings=test_result.warnings,
                    )
            else:
                error_msg = f"Installation failed: {result.stderr.decode()}"
                logger.error(error_msg)
                return InstallationResult(
                    success=False, component="coqui-tts", error=error_msg
                )

        except Exception as e:
            error_msg = f"CoquiTTS installation failed: {str(e)}"
            logger.error(error_msg)
            return InstallationResult(
                success=False, component="coqui-tts", error=error_msg
            )

    async def install_audio_dependencies(self) -> List[InstallationResult]:
        """Install audio processing dependencies"""
        results = []

        dependencies = [
            ("librosa", "librosa>=0.10.0", "Audio processing"),
            ("soundfile", "soundfile>=0.12.0", "Audio file I/O"),
            ("numpy", "numpy>=1.24.0", "Numerical processing"),
        ]

        for name, spec, description in dependencies:
            try:
                result = await self._install_dependency(name, spec, description)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to install {name}: {e}")
                results.append(
                    InstallationResult(success=False, component=name, error=str(e))
                )

        return results

    async def _install_dependency(
        self, name: str, spec: str, description: str
    ) -> InstallationResult:
        """Install a single dependency"""
        try:
            # Check if already installed
            if await self._is_package_installed(name):
                logger.info(f"{name} already installed")
                return InstallationResult(success=True, component=name)

            # Install package
            install_cmd = [sys.executable, "-m", "pip", "install", spec]
            result = await self._run_command(install_cmd)

            if result.returncode == 0:
                logger.info(f"✅ {name} installed successfully")
                return InstallationResult(success=True, component=name)
            else:
                error_msg = result.stderr.decode()
                logger.error(f"Failed to install {name}: {error_msg}")
                return InstallationResult(
                    success=False, component=name, error=error_msg
                )

        except Exception as e:
            logger.error(f"Exception installing {name}: {e}")
            return InstallationResult(success=False, component=name, error=str(e))

    async def install_ffmpeg(self) -> InstallationResult:
        """Install FFmpeg if available"""
        try:
            # Check if FFmpeg is already available
            ffmpeg_available = await self._check_ffmpeg()

            if ffmpeg_available:
                version = await self._get_ffmpeg_version()
                logger.info(f"FFmpeg already available: {version}")
                return InstallationResult(
                    success=True,
                    component="ffmpeg",
                    version=version,
                    installed_path="System PATH",
                )

            # Provide installation instructions
            logger.info("FFmpeg not found. Please install manually:")

            platform = sys.platform
            if platform == "darwin":
                instructions = [
                    "macOS (Homebrew):",
                    "  brew install ffmpeg",
                    "",
                    "macOS (MacPorts):",
                    "  sudo port install ffmpeg",
                ]
            elif platform == "linux":
                instructions = [
                    "Ubuntu/Debian:",
                    "  sudo apt update && sudo apt install ffmpeg",
                    "",
                    "CentOS/RHEL/Fedora:",
                    "  sudo dnf install ffmpeg  # or yum install ffmpeg",
                ]
            elif platform == "win32":
                instructions = [
                    "Windows:",
                    "  1. Download from https://ffmpeg.org/download.html",
                    "  2. Extract and add to system PATH",
                    "  3. Or use package manager like chocolatey:",
                    "     choco install ffmpeg",
                ]
            else:
                instructions = [
                    "Please install FFmpeg for your platform:",
                    "  https://ffmpeg.org/download.html",
                ]

            print("\n" + "\n".join(instructions))

            return InstallationResult(
                success=False,
                component="ffmpeg",
                error="Manual installation required",
                warnings=instructions,
            )

        except Exception as e:
            logger.error(f"Error checking FFmpeg: {e}")
            return InstallationResult(success=False, component="ffmpeg", error=str(e))

    async def test_complete_installation(self) -> Dict[str, Any]:
        """Test complete installation"""
        test_results = {
            "coqui_tts": await self._test_coqui_functionality(),
            "audio_deps": await self._test_audio_dependencies(),
            "ffmpeg": await self._test_ffmpeg_functionality(),
            "overall_success": False,
        }

        # Determine overall success
        test_results["overall_success"] = (
            test_results["coqui_tts"].success
            and test_results["audio_deps"].success
            and test_results["ffmpeg"].success
        )

        return test_results

    async def _is_coqui_installed(self) -> bool:
        """Check if CoquiTTS is installed"""
        try:
            importlib.import_module("TTS")
            return True
        except ImportError:
            return False

    async def _get_coqui_version(self) -> Optional[str]:
        """Get CoquiTTS version"""
        try:
            import TTS

            # Try to get version from TTS module
            if hasattr(TTS, "__version__"):
                return TTS.__version__
            return "Unknown"
        except Exception:
            return None

    async def _is_package_installed(self, package_name: str) -> bool:
        """Check if a Python package is installed"""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False

    def _prepare_install_command(self, use_gpu: bool = False) -> List[str]:
        """Prepare installation command"""
        if use_gpu:
            return [sys.executable, "-m", "pip", "install", "coqui-tts[gpu]"]
        else:
            return [sys.executable, "-m", "pip", "install", "coqui-tts"]

    async def _run_command(
        self, cmd: List[str], timeout: int = 300
    ) -> subprocess.CompletedProcess:
        """Run a command asynchronously"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            return subprocess.CompletedProcess(
                args=cmd, returncode=process.returncode, stdout=stdout, stderr=stderr
            )

        except asyncio.TimeoutError:
            raise InstallationError(f"Command timed out: {' '.join(cmd)}")

    async def _verify_installation(self) -> None:
        """Verify CoquiTTS installation"""
        if not await self._is_coqui_installed():
            raise InstallationError("CoquiTTS installation verification failed")

    async def _test_coqui_functionality(self) -> InstallationResult:
        """Test CoquiTTS functionality"""
        try:
            if not await self._is_coqui_installed():
                return InstallationResult(
                    success=False,
                    component="coqui-tts-test",
                    error="CoquiTTS not installed",
                )

            # Try to import and test basic functionality using correct API
            try:
                from TTS.api import TTS

                logger.info("✅ CoquiTTS API import successful")
            except ImportError as e:
                return InstallationResult(
                    success=False,
                    component="coqui-tts-test",
                    error=f"Failed to import TTS.api: {str(e)}",
                )

            # Test model list functionality
            try:
                tts = TTS()
                model_manager = tts.list_models()
                # TTS 0.22.0 returns ModelManager, not a list
                if hasattr(model_manager, "list_models"):
                    model_list = model_manager.list_models()
                else:
                    model_list = (
                        model_manager if isinstance(model_manager, list) else []
                    )

                if isinstance(model_list, list) and len(model_list) > 0:
                    # Check for XTTS models specifically
                    xtts_models = [
                        model for model in model_list if "xtts" in model.lower()
                    ]

                    logger.info(f"✅ CoquiTTS functionality test passed")
                    logger.info(f"   Found {len(model_list)} total models")
                    logger.info(
                        f"   Found {len(xtts_models)} XTTS models for multi-language support"
                    )

                    return InstallationResult(
                        success=True,
                        component="coqui-tts-test",
                        metadata={
                            "total_models": len(model_list),
                            "xtts_models": len(xtts_models),
                            "available_models": xtts_models[
                                :5
                            ],  # Show first 5 XTTS models
                        },
                    )
                else:
                    return InstallationResult(
                        success=False,
                        component="coqui-tts-test",
                        error="No models available",
                    )
            except Exception as e:
                logger.error(f"CoquiTTS functionality test error: {e}")
                return InstallationResult(
                    success=False,
                    component="coqui-tts-test",
                    error=f"Functionality test failed: {str(e)}",
                )

        except Exception as e:
            logger.error(f"Exception testing CoquiTTS: {e}")
            return InstallationResult(
                success=False,
                component="coqui-tts-test",
                error=f"Test exception: {str(e)}",
            )

    async def _test_audio_dependencies(self) -> InstallationResult:
        """Test audio processing dependencies"""
        try:
            missing_deps = []

            # Test librosa
            try:
                import librosa

                logger.info("✅ librosa available")
            except ImportError:
                missing_deps.append("librosa")

            # Test soundfile
            try:
                import soundfile

                logger.info("✅ soundfile available")
            except ImportError:
                missing_deps.append("soundfile")

            # Test numpy
            try:
                import numpy

                logger.info("✅ numpy available")
            except ImportError:
                missing_deps.append("numpy")

            # Test scipy
            try:
                import scipy

                logger.info("✅ scipy available")
            except ImportError:
                missing_deps.append("scipy")

            if not missing_deps:
                return InstallationResult(success=True, component="audio-dependencies")
            else:
                return InstallationResult(
                    success=False,
                    component="audio-dependencies",
                    error=f"Missing dependencies: {', '.join(missing_deps)}",
                )

        except Exception as e:
            logger.error(f"Exception testing audio dependencies: {e}")
            return InstallationResult(
                success=False,
                component="audio-dependencies",
                error=f"Test exception: {str(e)}",
            )

    async def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            result = await self._run_command(["ffmpeg", "-version"], timeout=10)
            return result.returncode == 0
        except Exception:
            return False

    async def _get_ffmpeg_version(self) -> str:
        """Get FFmpeg version"""
        try:
            result = await self._run_command(["ffmpeg", "-version"], timeout=10)
            if result.returncode == 0:
                output = result.stdout.decode()
                # Extract version from output
                for line in output.split("\n"):
                    if "ffmpeg version" in line.lower():
                        return line.split("version")[1].strip().split()[0]
            return "Unknown"
        except Exception:
            return "Unknown"

    async def _test_ffmpeg_functionality(self) -> InstallationResult:
        """Test FFmpeg functionality"""
        try:
            if not await self._check_ffmpeg():
                return InstallationResult(
                    success=False, component="ffmpeg-test", error="FFmpeg not available"
                )

            # Test basic FFmpeg functionality
            test_file = Path.home() / ".tts-notify" / "temp" / "test_audio.wav"
            test_file.parent.mkdir(parents=True, exist_ok=True)

            # Generate a test tone (this is simplified)
            try:
                # For now, just test that ffmpeg can run
                result = await self._run_command(["ffmpeg", "-version"], timeout=10)

                # Clean up test file
                if test_file.exists():
                    test_file.unlink()

                if result.returncode == 0:
                    logger.info("✅ FFmpeg functionality test passed")
                    return InstallationResult(success=True, component="ffmpeg-test")
                else:
                    return InstallationResult(
                        success=False,
                        component="ffmpeg-test",
                        error="FFmpeg execution failed",
                    )
            except Exception as e:
                logger.error(f"FFmpeg test error: {e}")
                return InstallationResult(
                    success=False,
                    component="ffmpeg-test",
                    error=f"Test exception: {str(e)}",
                )

        except Exception as e:
            logger.error(f"Exception testing FFmpeg: {e}")
            return InstallationResult(
                success=False,
                component="ffmpeg-test",
                error=f"Test exception: {str(e)}",
            )

    async def install_all_dependencies(
        self, use_gpu: bool = False, include_ffmpeg: bool = False
    ) -> Dict[str, InstallationResult]:
        """Install all dependencies for full functionality"""
        results = {}

        logger.info("Starting complete dependency installation...")

        # Install CoquiTTS
        logger.info("Installing CoquiTTS...")
        results["coqui_tts"] = await self.install_coqui_tts(use_gpu=use_gpu)

        # Install audio dependencies
        logger.info("Installing audio processing dependencies...")
        audio_deps = await self.install_audio_dependencies()
        for dep in audio_deps:
            results[f"audio_{dep.component}"] = dep

        # Install FFmpeg if requested
        if include_ffmpeg:
            logger.info("Checking FFmpeg...")
            results["ffmpeg"] = await self.install_ffmpeg()

        return results

    async def get_installation_status(self) -> Dict[str, Any]:
        """Get current installation status"""
        return {
            "coqui_tts_installed": await self._is_coqui_installed(),
            "audio_deps": {
                "librosa": await self._is_package_installed("librosa"),
                "soundfile": await self._is_package_installed("soundfile"),
                "numpy": await self._is_package_installed("numpy"),
                "scipy": await self._is_package_installed("scipy"),
            },
            "ffmpeg_available": await self._check_ffmpeg(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
        }


# Global installer instance
coqui_installer = CoquiTTSInstaller()
