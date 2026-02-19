#!/usr/bin/env python3
"""
MCP Config Generator Module

Core utilities for generating MCP (Model Context Protocol) configurations
for various AI coding tools: Claude Desktop, OpenCode, Zed, Cursor, Continue.dev.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# T1: Config Templates
# =============================================================================

CLIENT_CONFIGS: dict[str, dict[str, Any]] = {
    "claude": {
        "mcpServers": {"tts-notify": {"command": "", "args": ["--mode", "mcp"]}}
    },
    "opencode": {"mcp": {"tts-notify": {"type": "local", "command": []}}},
    "zed": {"context_servers": {"tts-notify": {"command": {"path": "", "args": []}}}},
    "cursor": {
        "mcpServers": {"tts-notify": {"type": "command", "command": "", "args": []}}
    },
    "continue": {
        "name": "TTS Notify Configuration",
        "version": "1.0.0",
        "schema": "v1",
        "mcpServers": {"tts-notify": {"command": "", "args": []}},
    },
}


# =============================================================================
# T2: Path Resolution
# =============================================================================


def resolve_executable_path(custom_path: str | None = None) -> str:
    """
    Resolve the executable path for tts-notify.

    Fallback chain:
    1. custom_path if provided
    2. shutil.which('tts-notify') if in PATH
    3. sys.executable as fallback

    Args:
        custom_path: Custom executable path to use

    Returns:
        Absolute path string to the executable

    Examples:
        >>> resolve_executable_path('/custom/path')
        '/custom/path'
        >>> resolve_executable_path()  # doctest: +SKIP
        '/usr/local/bin/python3'
    """
    if custom_path:
        return str(Path(custom_path).resolve())

    # Check if tts-notify is in PATH
    tts_notify_path = shutil.which("tts-notify")
    if tts_notify_path:
        return str(Path(tts_notify_path).resolve())

    # Fallback to current Python executable
    # Log warning when falling back
    print(
        f"Warning: 'tts-notify' not found in PATH, using sys.executable: {sys.executable}",
        file=sys.stderr,
    )
    return sys.executable


# =============================================================================
# T3: JSON/YAML Formatter
# =============================================================================


def format_config(config: dict[str, Any], output_format: str = "json") -> str:
    """
    Format config dict as JSON or YAML string.

    Args:
        config: Configuration dictionary
        output_format: Output format ("json" or "yaml")

    Returns:
        Formatted configuration string

    Raises:
        ValueError: If output_format is not "json" or "yaml"

    Examples:
        >>> config = {"test": "value"}
        >>> format_config(config, "json")
        '{\\n  "test": "value"\\n}'
    """
    if output_format == "json":
        return json.dumps(config, indent=2, ensure_ascii=False)
    elif output_format == "yaml":
        return yaml.dump(
            config, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    else:
        raise ValueError(f"Unsupported format: {output_format}. Use 'json' or 'yaml'.")


def detect_format(path: str) -> str:
    """
    Detect format from file extension.

    Args:
        path: File path to examine

    Returns:
        "yaml" for .yaml/.yml files, "json" otherwise

    Examples:
        >>> detect_format('/path/to/config.yaml')
        'yaml'
        >>> detect_format('/path/to/config.json')
        'json'
        >>> detect_format('/path/to/config.txt')
        'json'
    """
    path_obj = Path(path)
    extension = path_obj.suffix.lower()
    return "yaml" if extension in [".yaml", ".yml"] else "json"


# =============================================================================
# T4: Config Merge Utility
# =============================================================================


def merge_config(
    existing_path: str, new_config: dict[str, Any], client: str
) -> dict[str, Any]:
    """
    Merge new tts-notify config into existing config file.

    Args:
        existing_path: Path to existing config file
        new_config: New configuration to merge (with tts-notify entry)
        client: Client name (claude, opencode, zed, cursor, continue)

    Returns:
        Merged configuration dictionary (does NOT write to disk)

    Examples:
        >>> existing = {"mcpServers": {"other-server": {"command": "node"}}}
        >>> new_cfg = {"mcpServers": {"tts-notify": {"command": "python"}}}
        >>> merged = merge_config("/tmp/config.json", new_cfg, "claude")
        >>> "tts-notify" in merged["mcpServers"]
        True
    """
    path_obj = Path(existing_path)

    # Determine the config key based on client
    config_keys = {
        "claude": "mcpServers",
        "opencode": "mcp",
        "zed": "context_servers",
        "cursor": "mcpServers",
        "continue": "mcpServers",
    }

    config_key = config_keys.get(client)
    if not config_key:
        raise ValueError(f"Unknown client: {client}")

    # Start with empty config or load existing
    merged: dict[str, Any] = {}
    if path_obj.exists():
        file_format = detect_format(existing_path)
        try:
            content = path_obj.read_text(encoding="utf-8")
            if file_format == "yaml":
                merged = yaml.safe_load(content) or {}
            else:
                merged = json.loads(content)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            print(
                f"Warning: Failed to parse existing config: {e}. Creating new config.",
                file=sys.stderr,
            )
            merged = {}
    else:
        # File doesn't exist, start fresh
        merged = {}

    # Ensure the config key exists
    if config_key not in merged:
        merged[config_key] = {}

    # Merge the tts-notify config
    if config_key in new_config:
        # For continue.dev, we need special handling for mcpServers
        if client == "continue":
            # Continue uses a flat structure at mcpServers level
            merged[config_key].update(new_config[config_key])
        else:
            # Other clients use nested structure
            merged[config_key].update(new_config[config_key])

    return merged


# =============================================================================
# T5: OS-Specific Path Detector
# =============================================================================


def get_config_path(client: str) -> str | None:
    """
    Get OS-specific config path for client.

    Args:
        client: Client name (claude, opencode, zed, cursor, continue)

    Returns:
        Absolute path to config file, or None if unknown client

    Examples:
        >>> get_config_path("claude")  # doctest: +SKIP
        '/Users/username/Library/Application Support/Claude/claude_desktop_config.json'
        >>> get_config_path("unknown") is None
        True
    """
    home = Path.home()

    # macOS paths only (as per task requirements)
    paths = {
        "claude": home
        / "Library/Application Support/Claude/claude_desktop_config.json",
        "opencode": home / "Library/Application Support/ai.opencode.desktop/mcp.json",
        "zed": home / ".zed/settings.json",
        "cursor": home
        / "Library/Application Support/Cursor/User/globalStorage/mcp.json",
        "continue": home / ".continue/config.json",
    }

    return str(paths.get(client)) if client in paths else None


# =============================================================================
# Bonus: Generate Config Function
# =============================================================================


def generate_config(client: str, exec_path: str) -> dict[str, Any]:
    """
    Generate a complete MCP config for the specified client.

    Takes the template from CLIENT_CONFIGS and injects the exec_path.
    Returns a dict ready for formatting.

    Args:
        client: Client name (claude, opencode, zed, cursor, continue)
        exec_path: Executable path to use

    Returns:
        Configuration dictionary with injected executable path

    Raises:
        ValueError: If client is unknown

    Examples:
        >>> config = generate_config("claude", "/usr/bin/python3")
        >>> config["mcpServers"]["tts-notify"]["command"]
        '/usr/bin/python3'
        >>> config["mcpServers"]["tts-notify"]["args"]
        ['--mode', 'mcp']
    """
    if client not in CLIENT_CONFIGS:
        raise ValueError(
            f"Unknown client: {client}. Available: {', '.join(CLIENT_CONFIGS.keys())}"
        )

    # Deep copy the template to avoid modifying the original
    import copy

    config = copy.deepcopy(CLIENT_CONFIGS[client])

    # Inject the executable path based on client format
    if client == "claude":
        config["mcpServers"]["tts-notify"]["command"] = exec_path
    elif client == "opencode":
        # OpenCode uses command array
        config["mcp"]["tts-notify"]["command"] = [exec_path, "--mode", "mcp"]
    elif client == "zed":
        # Zed uses nested command.path
        config["context_servers"]["tts-notify"]["command"]["path"] = exec_path
        config["context_servers"]["tts-notify"]["command"]["args"] = ["--mode", "mcp"]
    elif client == "cursor":
        config["mcpServers"]["tts-notify"]["command"] = exec_path
        config["mcpServers"]["tts-notify"]["args"] = ["--mode", "mcp"]
    elif client == "continue":
        config["mcpServers"]["tts-notify"]["command"] = exec_path
        config["mcpServers"]["tts-notify"]["args"] = ["--mode", "mcp"]

    return config


def handle_install_mcp(args) -> int:
    """
    Handle the --install-mcp CLI command.

    Args:
        args: Parsed argparse namespace with install_mcp, all_clients, write_config, custom_exec_path

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Import necessary modules
    import sys

    # Resolve executable path
    exec_path = resolve_executable_path(args.custom_exec_path)

    # Handle --all flag
    if args.all_clients:
        print("\n🚀 Generando configuraciones MCP para todos los clientes...")
        print("=" * 60)

        for client in CLIENT_CONFIGS:
            config = generate_config(client, exec_path)
            print(f"\n📄 Configuración para {client.capitalize()}:")
            print(format_config(config, "json"))
            print("-" * 60)

        print(f"\n✅ Generadas configuraciones para {len(CLIENT_CONFIGS)} clientes")
        print("\n💡 Para aplicar configuraciones:")
        print(
            f"   - Claude Desktop: Edita ~/Library/Application Support/Claude/claude_desktop_config.json"
        )
        print(
            f"   - OpenCode: Edita ~/Library/Application Support/ai.opencode.desktop/mcp.json"
        )
        print(f"   - Zed: Edita ~/.zed/settings.json")
        print(
            f"   - Cursor: Edita ~/Library/Application Support/Cursor/User/globalStorage/mcp.json"
        )
        print(f"   - Continue: Edita ~/.continue/config.json")

        return 0

    # Handle single client
    if args.install_mcp:
        client = args.install_mcp if isinstance(args.install_mcp, str) else "claude"
        print(f"\n🚀 Generando configuración MCP para {client.capitalize()}...")

        if client not in CLIENT_CONFIGS:
            print(f"❌ Cliente desconocido: {client}")
            print(f"   Clientes disponibles: {', '.join(CLIENT_CONFIGS.keys())}")
            return 1

        config = generate_config(client, exec_path)
        print(f"\n📄 Configuración MCP generada:")
        print(format_config(config, "json"))

        # Handle --write flag
        if args.write_config:
            # Check if custom path was provided or use auto-detection
            if isinstance(args.write_config, str):
                config_path = args.write_config
            else:
                config_path = get_config_path(client)

            if config_path:
                print(f"\n📝 Escribiendo a: {config_path}")

                # Backup existing file if it exists
                if Path(config_path).exists():
                    backup_path = Path(config_path).with_suffix(".backup")
                    shutil.copy2(config_path, backup_path)
                    print(f"   ✅ Copia de seguridad creada: {backup_path.name}")

                # Merge or create new config
                merged_config = merge_config(config_path, config, client)

                # Determine file format from path
                file_format = detect_format(config_path)

                # Write the merged config
                if file_format == "yaml":
                    with open(config_path, "w", encoding="utf-8") as f:
                        yaml.dump(
                            merged_config,
                            f,
                            default_flow_style=False,
                            sort_keys=False,
                            allow_unicode=True,
                        )
                else:
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(merged_config, f, indent=2, ensure_ascii=False)

                print(f"   ✅ Configuración guardada exitosamente")
                print(f"\n💡 Para usar la configuración:")
                print(f"   - Reinicia el cliente (Claude Desktop, Zed, etc.)")
            else:
                print(f"❌ No se encontró ruta de configuración para {client}")
                return 1
        else:
            print("\n💡 Para guardar la configuración, usa:")
            print(f"   tts-notify --install-mcp {client} --write")
            print(f"\n   O para todos los clientes:")
            print(f"   tts-notify --install-mcp --all --write")

        return 0

    # No flags provided
    print("❌ Debes especificar un cliente o usar --all")
    print(f"   Ejemplos:")
    print(f"   tts-notify --install-mcp claude")
    print(f"   tts-notify --install-mcp --all")
    print(f"   tts-notify --install-mcp claude --write")
    return 1
