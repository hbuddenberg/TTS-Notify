"""
TTS Notify v3.0.0 - Modular Text-to-Speech notification system
"""

import sys
import os
from pathlib import Path

# Add the src directory to sys.path to allow absolute imports within the package
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
project_root = src_dir.parent

# Add all necessary paths to make imports work
paths_to_add = [
    str(src_dir),           # For imports like "from core..."
    str(project_root),      # For imports that might expect project root
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Also add to current working directory for imports
os.chdir(project_root)

__version__ = "3.0.0"
__author__ = "TTS Notify Project"
__description__ = "Modular Text-to-Speech notification system for macOS with CLI, MCP, and REST API interfaces"

# Simple package initialization
__all__ = ["__version__"]