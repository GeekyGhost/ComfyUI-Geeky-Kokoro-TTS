#!/usr/bin/env python3
"""
Installation script for Geeky Kokoro TTS ComfyUI Node
Supports ComfyUI v3.49+ and Python 3.9-3.13
Optimized for ComfyUI portable installations
"""

import sys
import subprocess
import os
from pathlib import Path
import shutil
import platform

# Fix Windows console encoding issues
if platform.system() == 'Windows':
    try:
        # Try to set UTF-8 encoding for Windows console
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # If that fails, we'll fall back to ASCII symbols
        pass

# Detect if we can use Unicode symbols
def can_use_unicode():
    """Check if the console supports Unicode characters."""
    if platform.system() == 'Windows':
        # Windows console may not support Unicode properly
        try:
            # Try to encode Unicode checkmark
            '✓'.encode(sys.stdout.encoding or 'utf-8')
            return True
        except (UnicodeEncodeError, AttributeError):
            return False
    return True

USE_UNICODE = can_use_unicode()


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# Symbols with fallback for systems that don't support Unicode
class Symbols:
    CHECK = '✓' if USE_UNICODE else '+'
    CROSS = '✗' if USE_UNICODE else 'x'
    WARNING = '⚠' if USE_UNICODE else '!'
    BULLET = '•' if USE_UNICODE else '*'
    ARROW = '==>' if not USE_UNICODE else '==>'
    PARTY = '🎉' if USE_UNICODE else '***'
    SPARKLES = '✨' if USE_UNICODE else '***'


def print_step(message):
    """Print a main step with formatting."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{Symbols.ARROW} {message}{Colors.ENDC}")


def print_substep(message):
    """Print a substep with formatting."""
    print(f"  {Colors.OKCYAN}{Symbols.BULLET} {message}{Colors.ENDC}")


def print_success(message):
    """Print a success message."""
    print(f"{Colors.OKGREEN}{Symbols.CHECK} {message}{Colors.ENDC}")


def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.WARNING}{Symbols.WARNING} {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message."""
    print(f"{Colors.FAIL}{Symbols.CROSS} {message}{Colors.ENDC}")


def check_python_compatibility():
    """
    Check if the current Python version is compatible.

    Returns:
    --------
    bool
        True if Python version is compatible (3.9-3.13), False otherwise
    """
    version = sys.version_info
    print_step("Checking Python compatibility")
    print_substep(f"Current Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3:
        print_error(f"Python 3 is required, but you have Python {version.major}")
        return False

    if version.minor < 9:
        print_error(f"Python 3.9 or higher is required, but you have Python 3.{version.minor}")
        print_error("Please upgrade your Python installation.")
        return False

    if version.minor > 13:
        print_warning(f"Python 3.{version.minor} is newer than tested versions (3.9-3.13)")
        print_warning("Installation will continue, but compatibility is not guaranteed.")

    print_success(f"Python {version.major}.{version.minor} is compatible!")
    return True


def find_python_executable():
    """
    Find the correct Python executable for the current environment.
    Handles ComfyUI portable installations.

    Returns:
    --------
    str
        Path to Python executable
    """
    # Check if we're in a ComfyUI portable installation
    script_dir = Path(__file__).resolve().parent

    # Look for portable Python installation
    possible_portable_paths = [
        script_dir.parent.parent.parent / "python_embeded" / "python.exe",  # Windows portable
        script_dir.parent.parent.parent / "python_embedded" / "python.exe",  # Alternative spelling
        script_dir.parent.parent.parent / "python" / "python.exe",  # Another possible location
    ]

    for path in possible_portable_paths:
        if path.exists():
            print_success(f"Found portable Python at: {path}")
            return str(path)

    # Fall back to system Python
    return sys.executable


def find_comfyui_directory(start_path):
    """
    Find the ComfyUI installation directory.

    Parameters:
    -----------
    start_path : Path
        Starting path to search from (usually the script directory)

    Returns:
    --------
    tuple
        (parent_dir, comfyui_dir) or (None, None) if not found
    """
    print_step("Locating ComfyUI installation")

    current = start_path
    max_depth = 5  # Prevent infinite loops
    depth = 0

    while depth < max_depth:
        # Check if current directory is ComfyUI root
        if current.name == "ComfyUI" or current.name == "comfyui":
            print_success(f"Found ComfyUI at: {current}")
            return current.parent, current

        # Check for ComfyUI markers
        if (current / "nodes.py").exists() or (current / "comfy").exists():
            print_success(f"Found ComfyUI at: {current}")
            return current.parent, current

        # Move up one directory
        if current.parent == current:
            break
        current = current.parent
        depth += 1

    return None, None


def setup_modern_directory_structure(comfy_dir, script_dir):
    """
    Set up directory structure following ComfyUI v3.49+ conventions.

    Parameters:
    -----------
    comfy_dir : Path
        ComfyUI installation directory
    script_dir : Path
        Directory where this script is located

    Returns:
    --------
    tuple
        (node_directory, models_directory)
    """
    print_step("Setting up directory structure")

    # Node installation directory (where we are)
    node_dir = script_dir
    print_substep(f"Node directory: {node_dir}")

    # Models directory (following ComfyUI conventions)
    models_dir = comfy_dir / "models" / "kokoro_tts"
    models_dir.mkdir(parents=True, exist_ok=True)
    print_substep(f"Models directory: {models_dir}")

    print_success("Directory structure ready")
    return node_dir, models_dir


def install_kokoro_dependencies():
    """
    Install core Kokoro TTS dependencies.

    Returns:
    --------
    bool
        True if installation succeeded, False otherwise
    """
    print_step("Installing Kokoro TTS core dependencies")

    python_exe = find_python_executable()

    core_packages = [
        "kokoro>=0.9.4",
        "soundfile>=0.12.1",
        "numpy>=1.22.0,<2.0.0",
        "torch>=2.0.0",
        "tqdm>=4.64.0",
        "einops>=0.6.0",
    ]

    for package in core_packages:
        print_substep(f"Installing {package}...")
        try:
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", package, "--no-warn-script-location"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print_warning(f"Failed to install {package}")
                print_substep(f"Error: {result.stderr}")
                # Try without version constraint
                simple_name = package.split(">")[0].split("=")[0].split("<")[0]
                print_substep(f"Retrying with just package name: {simple_name}")
                result = subprocess.run(
                    [python_exe, "-m", "pip", "install", simple_name, "--no-warn-script-location"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode != 0:
                    print_error(f"Failed to install {simple_name}")
                    return False
            print_success(f"Installed {package}")
        except subprocess.TimeoutExpired:
            print_error(f"Timeout installing {package}")
            return False
        except Exception as e:
            print_error(f"Error installing {package}: {e}")
            return False

    return True


def install_audio_dependencies():
    """
    Install audio processing dependencies for Voice Mod features.
    These are optional but recommended for full functionality.

    Returns:
    --------
    bool
        True if installation succeeded, False otherwise
    """
    print_step("Installing audio processing dependencies (optional)")

    python_exe = find_python_executable()

    audio_packages = [
        "librosa>=0.10.0",
        "scipy>=1.9.0,<2.0.0",
        "resampy>=0.4.3",
    ]

    all_success = True
    for package in audio_packages:
        print_substep(f"Installing {package}...")
        try:
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", package, "--no-warn-script-location"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print_warning(f"Failed to install {package} - Voice Mod features may be limited")
                all_success = False
            else:
                print_success(f"Installed {package}")
        except Exception as e:
            print_warning(f"Error installing {package}: {e}")
            all_success = False

    if all_success:
        print_success("All audio dependencies installed successfully")
    else:
        print_warning("Some audio dependencies failed - Voice Mod may have limited functionality")

    return all_success


def install_pyopenjtalk(python_exe):
    """
    Try to install pyopenjtalk with multiple strategies.

    Parameters:
    -----------
    python_exe : str
        Path to Python executable

    Returns:
    --------
    bool
        True if installation succeeded, False otherwise
    """
    print_substep("Installing pyopenjtalk (Japanese support)...")

    # Strategy 1: Try to install pre-built wheel
    print_substep("  Attempting to install pre-built wheel...")
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", "pyopenjtalk>=0.3.0",
             "--only-binary", ":all:", "--no-warn-script-location"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print_success("Installed pyopenjtalk from pre-built wheel")
            return True
    except Exception:
        pass

    # Strategy 2: Try to build from source (requires CMake)
    print_substep("  Pre-built wheel not available, attempting to build from source...")
    print_substep("  (This requires CMake to be installed)")
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", "pyopenjtalk>=0.3.0",
             "--no-warn-script-location"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print_success("Installed pyopenjtalk from source")
            return True
        else:
            # Check if error is CMake-related
            if "CMake" in result.stderr or "cmake" in result.stderr.lower():
                print_warning("pyopenjtalk requires CMake to build from source")
                print_warning("Install CMake from https://cmake.org/download/ to enable Japanese support")
                print_warning("Or use pre-built wheels if available for your platform")
            else:
                print_warning(f"Failed to install pyopenjtalk: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print_warning("Timeout while installing pyopenjtalk")
    except Exception as e:
        print_warning(f"Error installing pyopenjtalk: {e}")

    print_warning("Japanese voices will not be available without pyopenjtalk")
    return False


def install_language_dependencies():
    """
    Install language-specific dependencies for Japanese and Chinese TTS.
    These are optional but required for Japanese and Chinese voice support.

    Returns:
    --------
    bool
        True if installation succeeded, False otherwise
    """
    print_step("Installing language-specific dependencies (optional)")
    print_substep("For Japanese and Chinese language support")

    python_exe = find_python_executable()

    # Try to install pyopenjtalk with special handling
    pyopenjtalk_success = install_pyopenjtalk(python_exe)

    # Install ordered-set for Chinese support (simpler, no build required)
    print_substep("Installing ordered-set (Chinese support)...")
    ordered_set_success = False
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", "ordered-set>=4.1.0",
             "--no-warn-script-location"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print_success("Installed ordered-set")
            ordered_set_success = True
        else:
            print_warning("Failed to install ordered-set - Chinese voices may not be available")
    except Exception as e:
        print_warning(f"Error installing ordered-set: {e}")

    if pyopenjtalk_success and ordered_set_success:
        print_success("All language dependencies installed successfully")
        return True
    elif pyopenjtalk_success or ordered_set_success:
        print_warning("Some language dependencies installed - partial language support available")
        return True
    else:
        print_warning("Language dependencies failed - using English-only mode")
        return False


def copy_updated_files(source_dir, target_dir):
    """
    Copy node files to the target directory.

    Parameters:
    -----------
    source_dir : Path
        Source directory (where install script is)
    target_dir : Path
        Target directory (where to install)

    Returns:
    --------
    bool
        True if copy succeeded, False otherwise
    """
    # If source and target are the same, no need to copy
    if source_dir == target_dir:
        print_substep("Files are already in the correct location")
        return True

    print_step("Copying node files")

    files_to_copy = [
        "__init__.py",
        "node.py",
        "GeekyKokoroVoiceModNode.py",
        "audio_utils.py",
        "voice_profiles_utils.py",
    ]

    for file_name in files_to_copy:
        source_file = source_dir / file_name
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_dir / file_name)
                print_substep(f"Copied {file_name}")
            except Exception as e:
                print_warning(f"Failed to copy {file_name}: {e}")
                return False
        else:
            print_warning(f"File not found: {file_name}")

    print_success("Files copied successfully")
    return True


def verify_installation(node_dir, comfy_dir):
    """
    Verify that the installation was successful.

    Parameters:
    -----------
    node_dir : Path
        Node installation directory
    comfy_dir : Path
        ComfyUI installation directory

    Returns:
    --------
    bool
        True if verification passed, False otherwise
    """
    print_step("Verifying installation")

    required_files = [
        "__init__.py",
        "node.py",
        "GeekyKokoroVoiceModNode.py",
    ]

    all_present = True
    for file_name in required_files:
        file_path = node_dir / file_name
        if file_path.exists():
            print_substep(f"{Symbols.CHECK} Found {file_name}")
        else:
            print_error(f"{Symbols.CROSS} Missing {file_name}")
            all_present = False

    # Check Python imports
    print_substep("Testing Python imports...")
    python_exe = find_python_executable()

    test_imports = [
        "kokoro",
        "soundfile",
        "numpy",
        "torch",
        "einops",
    ]

    for module in test_imports:
        result = subprocess.run(
            [python_exe, "-c", f"import {module}"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_substep(f"{Symbols.CHECK} {module} is importable")
        else:
            print_warning(f"{Symbols.CROSS} {module} import failed - may cause issues")
            all_present = False

    if all_present:
        print_success("Installation verification passed!")
    else:
        print_warning("Installation verification had issues")

    return all_present


def main():
    """
    Main installation function with modern ComfyUI support.

    Returns:
    --------
    bool
        True if installation succeeded, False otherwise
    """
    print_step("Starting Geeky Kokoro TTS installation for ComfyUI v3.49+")
    print_substep("Python 3.9-3.13 compatible")
    print_substep("Optimized for ComfyUI portable installations")

    # Check Python compatibility first
    if not check_python_compatibility():
        return False

    # Get the script directory
    script_dir = Path(__file__).resolve().parent
    print_substep(f"Installation script location: {script_dir}")

    # Find ComfyUI installation
    comfy_parent, comfy_dir = find_comfyui_directory(script_dir)

    if not comfy_dir:
        print_error("Could not find ComfyUI installation.")
        print_error("Please ensure this script is run from within your ComfyUI directory structure.")
        print_error("Supported locations:")
        print_error("  - ComfyUI/custom_nodes/ComfyUI-Geeky-Kokoro-TTS/")
        print_error("  - ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI-Geeky-Kokoro-TTS/")
        return False

    # Set up directory structure following modern ComfyUI conventions
    node_dir, models_dir = setup_modern_directory_structure(comfy_dir, script_dir)

    # Install dependencies
    print_step("Installing Python dependencies")

    if not install_kokoro_dependencies():
        print_error("Failed to install core dependencies. Installation aborted.")
        return False

    # Install audio dependencies (optional for full functionality)
    install_audio_dependencies()

    # Install language-specific dependencies (optional for Japanese/Chinese)
    install_language_dependencies()

    # Copy node files (only if needed)
    if node_dir != script_dir:
        if not copy_updated_files(script_dir, node_dir):
            print_warning("Some files could not be copied. Installation may be incomplete.")

    # Verify installation
    if not verify_installation(node_dir, comfy_dir):
        print_warning("Installation verification had issues.")
        return False

    # Final success message and instructions
    print_success(f"\n{Symbols.PARTY} Geeky Kokoro TTS installation completed!")
    print_success(f"Node installed at: {node_dir}")
    print_success(f"Models will be cached at: {models_dir}")
    print_success("\nNext steps:")
    print_substep("1. Restart ComfyUI completely")
    print_substep("2. Look for 'Geeky Kokoro TTS (Updated)' in the node menu")
    print_substep("3. Look for 'Geeky Kokoro Advanced Voice' for voice effects")
    print_substep("4. Models will be downloaded automatically on first use")
    print_substep("5. Check the console for any error messages")

    if comfy_parent and comfy_parent.name.lower().startswith("comfyui"):
        print_substep("6. For portable installations, models are cached in your user directory")

    print_success(f"\n{Symbols.SPARKLES} Installation complete! Enjoy using Geeky Kokoro TTS!")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_error("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error during installation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
