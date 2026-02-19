#!/usr/bin/env python3
"""
TTS Notify v2 - Main Orchestrator

Unified entry point for all TTS Notify v2 interfaces.
Provides intelligent mode detection and seamless switching between CLI, MCP, and API modes.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
import os

# Core imports will be done lazily to avoid import issues


class TTSNotifyOrchestrator:
    """Main orchestrator for TTS Notify v2"""

    def __init__(self):
        # Lazy loading to avoid import issues
        self._config_manager = None
        self._system_detector = None
        self._logger = None
        self._config = None

        # Interface instances
        self.cli = None
        self.mcp_server = None
        self.api_server = None

        # Setup logging immediately
        self._setup_logging()

    @property
    def config_manager(self):
        if self._config_manager is None:
            from .core.config_manager import config_manager

            self._config_manager = config_manager
        return self._config_manager

    @property
    def system_detector(self):
        if self._system_detector is None:
            from .utils.system_detector import SystemDetector

            self._system_detector = SystemDetector()
        return self._system_detector

    @property
    def logger(self):
        return self._logger

    @property
    def config(self):
        if self._config is None:
            self._config = self.config_manager.get_config()
        return self._config

    def _setup_logging(self):
        """Setup logging based on configuration"""
        from .utils.logger import setup_logging, get_logger

        setup_logging(level=getattr(self.config, "TTS_NOTIFY_LOG_LEVEL", "INFO"))
        self._logger = get_logger(__name__)

        if self._logger:
            self._logger.info("TTS Notify Orchestrator v3.0.0 initialized")

    def detect_execution_mode(self) -> str:
        """Intelligently detect the execution mode"""
        import os

        # Check explicit environment variables
        if "TTS_NOTIFY_MODE" in os.environ:
            mode = os.environ["TTS_NOTIFY_MODE"].lower()
            if mode in ["cli", "mcp", "api", "server"]:
                return mode

        # Check for direct MCP mode (used for subprocess calls)
        if len(sys.argv) > 2 and sys.argv[1] == "--mode" and sys.argv[2] == "mcp":
            return "mcp"

        # Check for MCP server context
        if "MCP_SERVER" in os.environ or "CLAUDE_DESKTOP" in os.environ:
            return "mcp"

        # Check for API server context
        if (
            "API_SERVER" in os.environ
            or "FASTAPI" in os.environ
            or "TTS_NOTIFY_API_PORT" in os.environ
        ):
            return "api"

        # Check command line arguments
        if len(sys.argv) > 1:
            first_arg = sys.argv[1].lower()
            if first_arg in [
                "cli",
                "mcp",
                "api",
                "server",
                "--help",
                "-h",
                "--version",
            ]:
                return "cli"

            # Check for CLI-specific flags
            cli_flags = [
                "--list",
                "--voice",
                "--rate",
                "--save",
                "--compact",
                "--gen",
                "--lang",
            ]
            if any(flag in sys.argv for flag in cli_flags):
                return "cli"

            # Check if first argument is text (CLI mode)
            if not sys.argv[1].startswith("-"):
                return "cli"

        # Check if running as module
        if len(sys.argv) >= 2 and sys.argv[0].endswith("__main__.py"):
            module_parts = sys.argv[0].split("/")
            if len(module_parts) >= 2:
                module_name = module_parts[-2]
                if module_name == "cli":
                    return "cli"
                elif module_name == "mcp":
                    return "mcp"
                elif module_name == "api":
                    return "api"

        # Default based on execution context
        if sys.stdin.isatty() and len(sys.argv) > 1:
            return "cli"
        else:
            return "cli"  # Safe default

    def get_interface(self, mode: str):
        """Get the appropriate interface instance"""
        if mode == "cli":
            if not self.cli:
                from .ui.cli.main import TTSNotifyCLI

                self.cli = TTSNotifyCLI()
            return self.cli
        elif mode == "mcp":
            # For MCP mode, we don't need to instantiate a server class
            # We'll use the simple server directly in the run method
            return None
        elif mode == "api" or mode == "server":
            if not self.api_server:
                from .ui.api.server import TTSNotifyAPIServer

                self.api_server = TTSNotifyAPIServer()
            return self.api_server
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main command-line parser"""
        parser = argparse.ArgumentParser(
            prog="tts-notify",
            description="TTS Notify v2 - Sistema unificado de notificaciones Text-to-Speech para macOS",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Modos de ejecución:
  tts-notify "Texto"                    - Modo CLI (predeterminado)
  tts-notify --mode mcp                 - Modo servidor MCP
  tts-notify --mode api                 - Modo servidor API
  tts-notify --mode cli --list          - Modo CLI con lista de voces

Variables de entorno:
  TTS_NOTIFY_MODE                       - Forzar modo específico (cli/mcp/api)
  TTS_NOTIFY_VOICE                      - Voz predeterminada
  TTS_NOTIFY_RATE                       - Velocidad predeterminada (WPM)
  TTS_NOTIFY_PROFILE                    - Perfil de configuración

Para más información sobre cada modo:
  tts-notify --mode cli --help          - Ayuda del modo CLI
  tts-notify --mode mcp --help          - Ayuda del modo MCP
  tts-notify --mode api --help          - Ayuda del modo API
            """,
        )

        # Mode selection
        parser.add_argument(
            "--mode",
            "-m",
            choices=["cli", "mcp", "api", "server"],
            help="Modo de ejecución (default: detección automática)",
        )

        # Configuration
        parser.add_argument(
            "--profile", help="Usar perfil de configuración predefinido"
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Mostrar información detallada"
        )
        parser.add_argument(
            "--debug", action="store_true", help="Mostrar información de depuración"
        )

        # System information
        parser.add_argument(
            "--info",
            action="store_true",
            help="Mostrar información del sistema y configuración",
        )

        # Version
        parser.add_argument("--version", action="version", version="%(prog)s 3.0.0")

        return parser

    async def show_system_info(self):
        """Display system and configuration information"""
        try:
            # System detection
            system_info = await self.system_detector.detect_system()

            # Voice information
            from core.voice_system import VoiceManager

            voice_manager = VoiceManager()
            voices = await voice_manager.get_all_voices()

            # Configuration info
            mcp_vars = self.config_manager.get_mcp_environment_variables()
            validation_issues = self.config_manager.validate_system()

            print("🔧 TTS NOTIFY v3.0.0 - INFORMACIÓN DEL SISTEMA")
            print("=" * 60)

            print("\n📋 INFORMACIÓN DEL SISTEMA:")
            print(f"  Sistema: {system_info.get('system', 'Unknown')}")
            print(f"  Versión: {system_info.get('version', 'Unknown')}")
            print(f"  Arquitectura: {system_info.get('architecture', 'Unknown')}")
            print(f"  Python: {system_info.get('python_version', 'Unknown')}")

            print(f"\n🎵 VOCES DISPONIBLES: {len(voices)}")
            if voices:
                # Categorize voices
                espanol = [
                    v
                    for v in voices
                    if "espanol" in v.name.lower() or "español" in v.name.lower()
                ]
                enhanced = [
                    v
                    for v in voices
                    if "enhanced" in v.name.lower() or "premium" in v.name.lower()
                ]
                siri = [v for v in voices if "siri" in v.name.lower()]

                print(f"  Español: {len(espanol)}")
                print(f"  Enhanced: {len(enhanced)}")
                print(f"  Siri: {len(siri)}")
                print(
                    f"  Otras: {len(voices) - len(espanol) - len(enhanced) - len(siri)}"
                )

            print(f"\n⚙️  CONFIGURACIÓN ACTUAL:")
            print(f"  Perfil: {getattr(self.config, 'TTS_NOTIFY_PROFILE', 'default')}")
            print(f"  Voz: {getattr(self.config, 'TTS_NOTIFY_VOICE', 'monica')}")
            print(f"  Velocidad: {getattr(self.config, 'TTS_NOTIFY_RATE', 175)} WPM")
            print(
                f"  TTS Habilitado: {getattr(self.config, 'TTS_NOTIFY_ENABLED', True)}"
            )
            print(
                f"  Caché Habilitado: {getattr(self.config, 'TTS_NOTIFY_CACHE_ENABLED', True)}"
            )
            print(
                f"  Nivel de Log: {getattr(self.config, 'TTS_NOTIFY_LOG_LEVEL', 'INFO')}"
            )

            print(f"\n✅ VALIDACIÓN DEL SISTEMA:")
            if validation_issues:
                print("  ❌ Problemas encontrados:")
                for issue in validation_issues:
                    print(f"    - {issue}")
            else:
                print("  ✅ Todo funciona correctamente")

            print(f"\n🌟 MODOS DISPONIBLES:")
            print("  • CLI: Interfaz de línea de comandos")
            print("  • MCP: Servidor para Claude Desktop")
            print("  • API: Servidor REST API")

            if self.logger:
                self.logger.info("System information displayed")

        except Exception as e:
            print(f"❌ Error obteniendo información del sistema: {e}")
            if self.logger:
                self.logger.exception("Error in show_system_info")

    async def run(self, args: argparse.Namespace = None):
        """Main execution method"""
        try:
            # Check for help with specific mode before parsing
            if "--help" in sys.argv or "-h" in sys.argv:
                help_index = next(
                    (i for i, arg in enumerate(sys.argv) if arg in ["--help", "-h"]), -1
                )
                mode_index = next(
                    (i for i, arg in enumerate(sys.argv) if arg == "--mode"), -1
                )

                if mode_index != -1 and mode_index < help_index:
                    # Help is requested after --mode, delegate to specific mode
                    mode = (
                        sys.argv[mode_index + 1]
                        if mode_index + 1 < len(sys.argv)
                        else None
                    )
                    if mode == "cli":
                        from .ui.cli.main import main as cli_main

                        # Modify sys.argv to remove --mode mode and keep --help
                        new_argv = ["tts-notify"] + [
                            arg
                            for i, arg in enumerate(sys.argv)
                            if not (i >= mode_index and i <= mode_index + 1)
                        ]
                        original_argv = sys.argv
                        sys.argv = new_argv
                        try:
                            await cli_main()
                        finally:
                            sys.argv = original_argv
                        return
                    elif mode in ["mcp", "api", "server"]:
                        if mode == "mcp":
                            # Show MCP configuration for Claude Desktop
                            import json
                            from pathlib import Path

                            # Get project root and Python paths
                            project_root = Path.cwd()
                            python_exe = sys.executable

                            # Create MCP configuration
                            mcp_config = {
                                "mcpServers": {
                                    "tts-notify": {
                                        "command": python_exe,
                                        "args": ["-m", "tts_notify", "--mode", "mcp"],
                                    }
                                }
                            }

                            print(
                                "🤖 TTS Notify - Configuración MCP para Claude Desktop"
                            )
                            print("=" * 60)
                            print()
                            print(
                                "📍 Copie y pegue este JSON en su archivo de configuración:"
                            )
                            print(
                                "   ~/Library/Application Support/Claude/claude_desktop_config.json"
                            )
                            print()
                            print("📝 Configuración JSON:")
                            print(json.dumps(mcp_config, indent=2))
                            print()
                            print("🔧 Para iniciar el servidor MCP:")
                            print("   tts-notify --mode mcp")
                            print()
                            print("💡 Notas:")
                            print("   • Reinicie Claude Desktop después de configurar")
                            print(
                                "   • El servidor usará TTS nativo de macOS (say command)"
                            )
                            print(
                                "   • Herramientas disponibles: speak_text, list_voices, save_audio"
                            )
                            return
                        else:
                            print(f"🔧 Help para modo {mode.upper()}")
                            print(f"Para ejecutar el servidor {mode.upper()}:")
                            print(f"  tts-notify --mode {mode}")
                            print()
                            print(f"Características del modo {mode.upper()}:")
                            print("  • Servidor REST API con FastAPI")
                            print("  • Endpoints para TTS y gestión de voces")
                            print("  • Documentación OpenAPI incluida")
                            return

            # Parse arguments if not provided
            if args is None:
                parser = self.create_parser()
                args, unknown = parser.parse_known_args()

            # Load configuration profile if specified
            if args and args.profile:
                self.config_manager.reload_config(args.profile)
                self.config = self.config_manager.get_config()

            # Show system info if requested
            if args and args.info:
                await self.show_system_info()
                return

            # Detect execution mode
            mode = args.mode if args and args.mode else self.detect_execution_mode()

            if self.logger:
                self.logger.info(f"Starting TTS Notify in {mode.upper()} mode")

            # Handle different modes
            if mode == "mcp":
                from .ui.mcp.fastmcp_server import mcp

                mcp.run()
                return

            if mode == "cli":
                # Delegate to CLI interface
                from .ui.cli.main import main as cli_main

                if args:
                    # Filter orchestrator-specific args
                    filtered_args = [
                        arg for arg in sys.argv if arg not in ["--mode", mode]
                    ]
                    if args.profile:
                        filtered_args.extend(["--profile", args.profile])
                    if args.verbose:
                        filtered_args.append("--verbose")
                    if args.debug:
                        filtered_args.append("--debug")

                    # Temporarily modify sys.argv for CLI parser
                    original_argv = sys.argv
                    sys.argv = filtered_args
                    try:
                        await cli_main()
                    finally:
                        sys.argv = original_argv
                else:
                    await cli_main()

            elif mode == "mcp":
                # For MCP mode, we should start a separate process
                # to avoid asyncio conflicts
                import subprocess
                import os

                print("🤖 Starting MCP server in separate process...")

                # Get the current command and modify it for MCP
                cmd = [sys.executable, "-m", "tts_notify", "--mode", "mcp"]

                if self.logger:
                    self.logger.info(
                        f"Starting MCP server with command: {' '.join(cmd)}"
                    )

                # Start the MCP server as a subprocess
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=os.environ.copy(),
                )

                # Wait a moment to see if it starts successfully
                try:
                    import asyncio

                    await asyncio.wait_for(
                        asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        ),
                        timeout=3.0,
                    )
                    print("✅ MCP server started successfully")
                    if self.logger:
                        self.logger.info("MCP server started successfully")

                    # Keep the process alive and wait for it
                    while True:
                        await asyncio.sleep(1)

                except asyncio.TimeoutError:
                    print("⚠️ MCP server startup timeout")
                    if self.logger:
                        self.logger.error("MCP server failed to start within timeout")
                    process.terminate()
                    return

                except Exception as e:
                    print(f"❌ Error starting MCP server: {e}")
                    if self.logger:
                        self.logger.error(f"MCP server error: {e}")
                    return

            elif mode == "api" or mode == "server":
                # Delegate to API server
                from .ui.api.server import main as api_main

                await api_main()

            else:
                raise ValueError(f"Unsupported mode: {mode}")

        except KeyboardInterrupt:
            print("\nOperación cancelada por el usuario")
            if self.logger:
                self.logger.info("Operation cancelled by user")
        except Exception as e:
            print(f"Error fatal: {e}")
            if self.logger:
                self.logger.exception("Fatal error in orchestrator")
            sys.exit(1)


def ensure_venv_execution():
    """
    Automatically switch to venv312 if available and not currently active.
    This ensures CoquiTTS dependencies are available even if run from global context.
    """
    try:
        # Check if we've already switched to avoid infinite loops
        if os.environ.get("TTS_NOTIFY_RELAUNCHED"):
            return

        # Only attempt this if we are in an editable install or source checkout
        current_file = Path(__file__).resolve()
        # src/tts_notify/main.py -> src/tts_notify -> src -> TTS_Notify (project root)
        if len(current_file.parents) < 3:
            return

        project_root = current_file.parents[2]
        venv_path = project_root / "venv312"

        # Check if venv312 exists and has python
        venv_python = venv_path / "bin" / "python"
        if not venv_python.exists():
            return

        # Check if we are already running from this venv
        current_exe = Path(sys.executable).resolve()
        venv_python_resolved = venv_python.resolve()

        if current_exe == venv_python_resolved:
            return

        # Check if sys.prefix matches (another way to check)
        if str(venv_path.resolve()) in sys.prefix:
            return

        # If we are here, we need to switch
        # We use -m tts_notify to ensure we run the module properly
        args = [str(venv_python), "-m", "tts_notify"] + sys.argv[1:]

        # Set flag
        os.environ["TTS_NOTIFY_RELAUNCHED"] = "1"

        # Replace current process
        os.execv(str(venv_python), args)

    except Exception:
        # If anything fails, just proceed with current environment
        pass


async def main():
    """Main entry point for TTS Notify v2"""
    # Ensure we are in the correct venv
    ensure_venv_execution()

    # Check for special modes before creating orchestrator
    if len(sys.argv) > 2 and sys.argv[1] == "--mode" and sys.argv[2] == "mcp":
        from .ui.mcp.fastmcp_server import mcp

        mcp.run()
        return

    orchestrator = TTSNotifyOrchestrator()
    await orchestrator.run()


def sync_main():
    """Synchronous main entry point for CLI scripts"""
    try:
        # Ensure we are in the correct venv
        ensure_venv_execution()

        # Check for special modes before creating orchestrator
        if len(sys.argv) > 2 and sys.argv[1] == "--mode" and sys.argv[2] == "mcp":
            from .ui.mcp.fastmcp_server import mcp

            mcp.run()
            return

        # Create orchestrator and run with asyncio
        orchestrator = TTSNotifyOrchestrator()
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
    except Exception as e:
        print(f"Error fatal: {e}")
        # Fallback to simple CLI if orchestrator fails
        try:
            from .ui.cli.main import sync_main as cli_sync_main

            cli_sync_main()
        except ImportError:
            print("TTS Notify v2.0.0 - Error crítico de importación")
            print("Por favor, reinstale el paquete.")


if __name__ == "__main__":
    sync_main()
