#!/bin/bash
# TTS Notify v3.0.0 - Cross-platform Installer Script with CoquiTTS Support
# Supports macOS and Linux systems with AI-powered TTS installation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${PURPLE}🚀 TTS Notify v3.0.0 - AI-Powered TTS Installer${NC}"
echo -e "${BLUE}📍 Project: $PROJECT_DIR${NC}"
echo -e "${YELLOW}✨ NEW: CoquiTTS integration with 17+ languages & voice cloning${NC}"
echo

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)    echo "macos";;
        Linux*)     echo "linux";;
        CYGWIN*|MINGW*|MSYS*) echo "windows";;
        *)          echo "unknown";;
    esac
}

# Function to check if UV is installed
check_uv() {
    if command -v uv &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to install UV
install_uv() {
    print_info "Installing UV package manager..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        # Add UV to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        print_status "UV installed successfully"
        return 0
    else
        print_error "Failed to install UV"
        return 1
    fi
}

# Function to check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
            print_status "Python $PYTHON_VERSION OK"
            echo "PYTHON_CMD=python3" >> .env
            return 0
        else
            print_error "Python 3.10+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Function to check macOS TTS
check_macos_tts() {
    if [ "$(detect_os)" = "macos" ]; then
        if command -v say &> /dev/null; then
            if say -v "?" &> /dev/null; then
                print_status "macOS TTS (say command) OK"
                return 0
            else
                print_error "macOS TTS not working"
                return 1
            fi
        else
            print_error "macOS TTS not available"
            return 1
        fi
    else
        print_warning "Not macOS - TTS support may be limited"
        return 0
    fi
}

# Function to check espeak-ng
check_espeak() {
    if command -v espeak-ng &> /dev/null; then
        print_status "espeak-ng OK"
        return 0
    else
        print_warning "espeak-ng not found (required for CoquiTTS phonemization)"
        print_info "Installing espeak-ng via Homebrew..."
        if command -v brew &> /dev/null; then
            brew install espeak-ng
            print_status "espeak-ng installed"
        else
            print_warning "Homebrew not found - please install espeak-ng manually: brew install espeak-ng"
        fi
        return 0
    fi
}

# Function to run installer
run_installer() {
    local mode="$1"
    print_info "Starting $mode installation..."

    cd "$PROJECT_DIR"

    case "$mode" in
        "all"|"coqui"|"complete")
            print_info "Installing TTS Notify v3.0.0 with CoquiTTS support..."


            RECREATE_VENV=true
            if [ -d "venv312" ]; then
                VENV_PYTHON_VERSION=$(./venv312/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "broken")
                if [[ "$VENV_PYTHON_VERSION" == "3.10" ]]; then
                    print_status "Existing venv312 has Python $VENV_PYTHON_VERSION (compatible with CoquiTTS)"
                    RECREATE_VENV=false
                else
                    print_warning "Existing venv312 has Python $VENV_PYTHON_VERSION (incompatible with CoquiTTS, needs 3.10)"
                    rm -rf venv312
                fi
            fi

            if [ "$RECREATE_VENV" = "true" ]; then
                print_info "Creating virtual environment with Python 3.10 (CoquiTTS requirement)..."

                # Find Python 3.10 path from UV
                PYTHON_310_PATH=$(uv python list 2>/dev/null | grep "cpython-3.10" | grep -v "download available" | head -1 | awk '{print $2}')
                
                if [ -n "$PYTHON_310_PATH" ] && [ -x "$PYTHON_310_PATH" ]; then
                    print_info "Using UV Python 3.10: $PYTHON_310_PATH"
                    uv venv venv312 --python "$PYTHON_310_PATH"
                elif command -v uv &> /dev/null; then
                    # Fallback: let UV find Python 3.10
                    uv venv venv312 --python 3.10
                elif command -v python3.10 &> /dev/null; then
                    python3.10 -m venv venv312
                else
                    print_error "Python 3.10 required for CoquiTTS (supports Python 3.9-3.11)"
                    return 1
                fi
            fi

            # Activate virtual environment and install
            source venv312/bin/activate
            
            # Use UV if available
            if check_uv; then
                print_info "Using UV for installation..."
                uv pip install --upgrade pip
                uv pip install -e .
            else
                pip install --upgrade pip
                pip install -e .
            fi

            print_info "Installing CoquiTTS with compatible versions (TTS 0.22.0 + PyTorch 2.2.2)..."
            
            pip install "numpy<2" "torch==2.2.2" "torchaudio==2.2.2" || true
            pip install --prefer-binary "llvmlite<0.46" "numba>=0.60,<0.64" || true
            pip install "TTS==0.22.0" --no-deps || print_warning "TTS installation had issues"
            
            print_info "Installing TTS dependencies..."
            pip install --prefer-binary coqpit coqpit-config "transformers>=4.33.0" \
                "pandas<2.0,>=1.4" "trainer>=0.0.32" aiohttp anyascii einops flask \
                inflect jieba nltk num2words pypinyin pysbd "spacy[ja]>=3" umap-learn unidecode \
                librosa soundfile scipy noisereduce || true
            
            print_info "Installing language support dependencies..."
            pip install bangla bnnumerizer bnunicodenormalizer cython encodec g2pkk \
                gruut hangul_romanize jamo || true
            
            print_status "CoquiTTS installation completed"

            # Create global symlink
            print_info "Creating global symlink..."
            SYMLINK_PATH="/usr/local/bin/tts-notify"
            if [ -L "$SYMLINK_PATH" ]; then
                rm "$SYMLINK_PATH"
            fi
            ln -s "$PROJECT_DIR/venv312/bin/tts-notify" "$SYMLINK_PATH"
            print_status "Global symlink created at $SYMLINK_PATH"

            # Test installation
            print_info "Testing installation..."
            ./venv312/bin/tts-notify --test-installation

            print_status "TTS Notify v3.0.0 with CoquiTTS installed successfully!"
            ;;
        "development"|"dev")
            print_info "Installing development environment..."
            if [ ! -d "venv" ]; then
                python3 -m venv venv
            fi
            source venv/bin/activate
            
            # Use UV if available
            if check_uv; then
                print_info "Using UV for installation..."
                uv pip install --upgrade pip
                uv pip install -e ".[dev]"
            else
                pip install --upgrade pip
                pip install -e ".[dev]"
            fi
            print_status "Development environment installed"
            ;;
        "production"|"prod")
            print_info "Installing for production use..."
            if command -v uv &> /dev/null; then
                uv pip install -e ".[api,mcp]"
            else
                pip3 install -e ".[api,mcp]"
            fi
            print_status "Production installation completed"
            ;;
        "mcp")
            print_info "Installing MCP server mode..."
            if command -v uv &> /dev/null; then
                uv pip install -e ".[mcp]"
            else
                pip3 install -e ".[mcp]"
            fi
            print_status "MCP server installation completed"
            ;;
        "uninstall")
            print_info "Uninstalling TTS Notify..."
            # Remove virtual environments
            [ -d "venv" ] && rm -rf venv
            [ -d "venv312" ] && rm -rf venv312
            [ -d "venv313" ] && rm -rf venv313
            # Remove package if installed globally
            pip3 uninstall -y tts-notify 2>/dev/null || true
            print_status "Uninstallation completed"
            ;;
        *)
            print_error "Unknown installation mode: $mode"
            return 1
            ;;
    esac
}

# Function to show usage
show_usage() {
    echo -e "${BLUE}TTS Notify v3.0.0 Installation Options:${NC}"
    echo
    echo "  1) Development    - Install for development with virtual environment"
    echo "  2) Production     - Install CLI globally for all users"
    echo "  3) MCP Server     - Install for Claude Desktop integration"
    echo "  4) Complete/Coqui - Install ALL features including CoquiTTS AI voices"
    echo "  5) Uninstall      - Remove TTS Notify completely"
    echo "  6) Exit"
    echo
    echo -e "${YELLOW}🚀 NEW in v3.0.0: CoquiTTS AI-powered TTS with multi-language support${NC}"
    echo "   • 17 languages supported (English, Spanish, French, German, etc.)"
    echo "   • Voice cloning capabilities"
    echo "   • Advanced audio processing pipeline"
    echo "   • Automated installation and testing"
    echo
}

# Main installation flow
main() {
    # Check if we're in the right directory
    if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
        print_error "pyproject.toml not found. Please run from project directory."
        exit 1
    fi

    # Show system information
    OS=$(detect_os)
    print_info "Detected OS: $OS"

    # Check prerequisites
    print_info "Checking prerequisites..."

    if ! check_python; then
        exit 1
    fi

    if ! check_macos_tts; then
        exit 1
    fi

    if ! check_espeak; then
        print_warning "espeak-ng not available - some TTS features may be limited"
    fi

    if ! check_uv; then
        if ! install_uv; then
            print_error "UV is required for installation"
            exit 1
        fi
    fi

    # Show installation menu
    if [ $# -eq 0 ]; then
        while true; do
            show_usage
            read -p "Select installation option (1-6): " choice
            case $choice in
                1)
                    run_installer "development"
                    break
                    ;;
                2)
                    run_installer "production"
                    break
                    ;;
                3)
                    run_installer "mcp"
                    break
                    ;;
                4)
                    run_installer "all"
                    break
                    ;;
                5)
                    run_installer "uninstall"
                    break
                    ;;
                6)
                    print_info "Installation cancelled"
                    exit 0
                    ;;
                *)
                    print_error "Invalid option. Please select 1-6."
                    ;;
            esac
        done
    else
        # Command line mode
        mode="$1"
        case $mode in
            development|dev)
                run_installer "development"
                ;;
            production|prod)
                run_installer "production"
                ;;
            mcp)
                run_installer "mcp"
                ;;
            all|coqui|complete)
                run_installer "all"
                ;;
            uninstall)
                run_installer "uninstall"
                ;;
            *)
                print_error "Invalid mode: $mode"
                echo "Valid modes: development, production, mcp, all/coqui, uninstall"
                echo "New v3.0.0 modes: coqui (install with AI voices), complete (full installation)"
                exit 1
                ;;
        esac
    fi

    print_status "Installation process completed!"
}

# Quick install functions for convenience
install_dev() {
    main "development"
}

install_prod() {
    main "production"
}

install_mcp() {
    main "mcp"
}

install_all() {
    main "all"
}

# New v3.0.0 convenience functions
install_coqui() {
    main "coqui"
}

install_complete() {
    main "complete"
}

uninstall() {
    main "uninstall"
}

# Run main function
main "$@"
