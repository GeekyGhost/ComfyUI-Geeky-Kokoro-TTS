#!/usr/bin/env python
"""
Updated installation script for the Geeky Kokoro TTS and Voice Mod nodes for ComfyUI.
Compatible with Python 3.12+ and ComfyUI v3.49+.
"""
import os
import sys
import subprocess
import zipfile
import shutil
import platform
import logging
from pathlib import Path
import urllib.request
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GeekyKokoroInstaller")


def is_windows():
    """Check if the operating system is Windows."""
    return platform.system() == "Windows"


def is_mac():
    """Check if the operating system is macOS."""
    return platform.system() == "Darwin"


def is_linux():
    """Check if the operating system is Linux."""
    return platform.system() == "Linux"


def get_python_version():
    """Get current Python version info."""
    version_info = sys.version_info
    return f"{version_info.major}.{version_info.minor}.{version_info.micro}"


def print_step(message):
    """Print a step message with formatting."""
    logger.info(message)
    print(f"\n\033[1;34m==>\033[0m \033[1m{message}\033[0m")


def print_substep(message):
    """Print a substep message with formatting."""
    logger.debug(message)
    print(f"  \033[1;32m->\033[0m \033[1m{message}\033[0m")


def print_error(message):
    """Print an error message with formatting."""
    logger.error(message)
    print(f"\033[1;31mError: {message}\033[0m")


def print_warning(message):
    """Print a warning message with formatting."""
    logger.warning(message)
    print(f"\033[1;33mWarning: {message}\033[0m")


def print_success(message):
    """Print a success message with formatting."""
    logger.info(message)
    print(f"\033[1;32mSuccess: {message}\033[0m")


def check_python_compatibility():
    """Check if Python version is compatible."""
    version_info = sys.version_info
    python_version = f"{version_info.major}.{version_info.minor}"
    
    print_step(f"Checking Python compatibility (current: {get_python_version()})")
    
    if version_info.major != 3:
        print_error("Python 3.x is required for this installation.")
        return False
    
    if version_info.minor < 9:
        print_error("Python 3.9 or newer is required. Please upgrade your Python installation.")
        return False
    
    if version_info.minor >= 15:
        print_warning(f"Python {python_version} is very new and may have compatibility issues.")
    
    print_success(f"Python {python_version} is compatible with ComfyUI and Kokoro TTS.")
    return True


def run_pip(args, desc=None):
    """
    Run pip with the specified arguments.
    
    Parameters:
    -----------
    args : list
        List of arguments to pass to pip
    desc : str
        Description of the operation
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    if desc:
        print_substep(desc)
    
    python = sys.executable
    pip_command = [python, '-m', 'pip']
    pip_command.extend(args)
    
    try:
        result = subprocess.run(pip_command, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"pip command failed: {' '.join(pip_command)}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def install_kokoro_dependencies():
    """
    Install Kokoro TTS and related dependencies with proper version management.
    
    Returns:
    --------
    bool
        True if all critical dependencies were installed, False otherwise
    """
    print_step("Installing Kokoro TTS Dependencies")
    
    # Core dependencies with version constraints for stability
    core_dependencies = [
        "torch>=2.0.0",  # PyTorch first for CUDA compatibility
        "numpy>=1.22.0,<2.0.0",  # Numpy with upper bound for compatibility
        "scipy>=1.9.0,<2.0.0",
        "kokoro>=0.9.4",  # Latest Kokoro version
        "soundfile>=0.12.1",
        "tqdm>=4.64.0",
        "einops>=0.6.0",
    ]
    
    core_success = True
    for dep in core_dependencies:
        if not run_pip(["install", "--upgrade", dep], f"Installing {dep}"):
            print_error(f"Failed to install critical dependency: {dep}")
            core_success = False
    
    if not core_success:
        print_error("Failed to install core dependencies.")
        return False
    
    return True


def install_audio_dependencies():
    """
    Install audio processing dependencies with better error handling.
    
    Returns:
    --------
    bool
        True if audio dependencies were installed, False otherwise
    """
    print_step("Installing Audio Processing Dependencies")
    
    # Check for existing installations first
    audio_deps = [
        ("librosa", "librosa>=0.10.0"),
        ("resampy", "resampy>=0.4.3"),
    ]
    
    success = True
    
    for package_name, pip_spec in audio_deps:
        try:
            # Try importing to see if already installed
            __import__(package_name)
            print_substep(f"{package_name} already available")
        except ImportError:
            # Need to install
            if package_name == "resampy":
                # Resampy often needs numba, install it first
                print_substep("Installing numba (required for resampy)")
                if not run_pip(["install", "numba>=0.56.0"], "Installing numba"):
                    print_warning("Numba installation failed. Resampy may not work properly.")
                
            if not run_pip(["install", pip_spec], f"Installing {package_name}"):
                print_warning(f"{package_name} installation failed. Some audio effects may use fallback methods.")
                success = False
    
    return success


def find_comfyui_directory(script_dir):
    """
    Find the ComfyUI installation directory with improved detection.
    
    Parameters:
    -----------
    script_dir : Path
        Directory of the installation script
        
    Returns:
    --------
    tuple
        (comfy_parent, comfy_dir) or (None, None) if not found
    """
    print_step("Locating ComfyUI installation")
    
    current_dir = script_dir
    
    # Check up to 6 directory levels up
    for level in range(6):
        print_substep(f"Checking directory level {level}: {current_dir}")
        
        # Check if this is the parent of a ComfyUI installation
        comfyui_subdir = current_dir / "ComfyUI"
        if comfyui_subdir.exists() and (comfyui_subdir / "main.py").exists():
            print_success(f"Found ComfyUI installation at: {comfyui_subdir}")
            return current_dir, comfyui_subdir
        
        # Check if this is a ComfyUI installation itself
        if (current_dir / "main.py").exists() and (current_dir / "nodes").exists():
            print_success(f"Found ComfyUI root at: {current_dir}")
            return current_dir.parent, current_dir
        
        # Check for ComfyUI portable structure
        if (current_dir.name.startswith("ComfyUI") and 
            (current_dir / "ComfyUI" / "main.py").exists()):
            comfyui_dir = current_dir / "ComfyUI"
            print_success(f"Found ComfyUI portable at: {comfyui_dir}")
            return current_dir, comfyui_dir
        
        # Move up one directory level
        if current_dir.parent == current_dir:  # Reached filesystem root
            break
        current_dir = current_dir.parent
    
    print_error("Could not locate ComfyUI installation")
    return None, None


def setup_modern_directory_structure(comfy_dir, script_dir):
    """
    Set up the modern ComfyUI directory structure following v3.49+ conventions.
    
    Parameters:
    -----------
    comfy_dir : Path
        ComfyUI directory
    script_dir : Path
        Installation script directory
        
    Returns:
    --------
    Path
        Path to the node directory
    """
    print_step("Setting up modern directory structure")
    
    custom_nodes_dir = comfy_dir / "custom_nodes"
    node_dir = custom_nodes_dir / "ComfyUI-Geeky-Kokoro-TTS"
    models_dir = comfy_dir / "models" / "kokoro_tts"
    
    # Create directories
    custom_nodes_dir.mkdir(exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle existing installation
    if node_dir.exists():
        print_substep("Found existing installation - backing up")
        backup_dir = custom_nodes_dir / "ComfyUI-Geeky-Kokoro-TTS.backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(node_dir, backup_dir)
        shutil.rmtree(node_dir)
    
    node_dir.mkdir(exist_ok=True)
    
    # Create a symlink or info file for model directory
    model_info = node_dir / "model_location.txt"
    model_info.write_text(f"Models are stored in: {models_dir}")
    
    print_success(f"Directory structure created:")
    print_substep(f"Node files: {node_dir}")
    print_substep(f"Models: {models_dir}")
    
    return node_dir


def copy_updated_files(script_dir, node_dir):
    """
    Copy updated node implementation files.
    
    Parameters:
    -----------
    script_dir : Path
        Source directory (script location)
    node_dir : Path
        Destination directory
        
    Returns:
    --------
    bool
        True if all required files were copied, False otherwise
    """
    print_step("Installing updated node files")
    
    # Required files with their descriptions
    required_files = {
        "node.py": "Main TTS node implementation",
        "__init__.py": "Node initialization",
        "requirements.txt": "Python dependencies",
        "pyproject.toml": "Project configuration",
        "README.md": "Documentation",
        "GeekyKokoroVoiceModNode.py": "Voice modification node",
        "audio_utils.py": "Audio processing utilities", 
        "voice_profiles_utils.py": "Voice profile utilities",
        "LICENSE": "License file"
    }
    
    success = True
    copied_files = []
    missing_files = []
    
    for filename, description in required_files.items():
        source_path = script_dir / filename
        dest_path = node_dir / filename
        
        if source_path.exists():
            try:
                shutil.copy2(source_path, dest_path)
                print_substep(f"✓ {filename} ({description})")
                copied_files.append(filename)
            except Exception as e:
                print_error(f"Failed to copy {filename}: {e}")
                success = False
        else:
            print_warning(f"Missing: {filename} ({description})")
            missing_files.append(filename)
    
    if missing_files:
        print_warning(f"Missing files: {', '.join(missing_files)}")
        if len(missing_files) > len(copied_files):
            success = False
    
    print_success(f"Copied {len(copied_files)} files successfully")
    return success


def verify_installation(node_dir, comfy_dir):
    """
    Verify the installation is correct and provide setup guidance.
    
    Parameters:
    -----------
    node_dir : Path
        Node installation directory
    comfy_dir : Path
        ComfyUI directory
        
    Returns:
    --------
    bool
        True if installation looks good, False otherwise
    """
    print_step("Verifying installation")
    
    # Check critical files
    critical_files = ["node.py", "__init__.py", "requirements.txt"]
    missing_critical = []
    
    for filename in critical_files:
        if not (node_dir / filename).exists():
            missing_critical.append(filename)
    
    if missing_critical:
        print_error(f"Missing critical files: {', '.join(missing_critical)}")
        return False
    
    # Check Python imports
    try:
        # Add the node directory to sys.path temporarily
        import sys
        sys.path.insert(0, str(node_dir))
        
        # Try importing kokoro to verify it's working
        import kokoro
        print_substep(f"✓ Kokoro TTS version: {kokoro.__version__ if hasattr(kokoro, '__version__') else 'imported successfully'}")
        
        # Remove from path
        sys.path.remove(str(node_dir))
        
    except ImportError as e:
        print_warning(f"Kokoro import test failed: {e}")
        print_warning("This may be normal if ComfyUI uses a different Python environment")
    
    # Check models directory
    models_dir = comfy_dir / "models" / "kokoro_tts"
    if models_dir.exists():
        print_substep(f"✓ Models directory: {models_dir}")
    else:
        print_substep(f"? Models directory will be created: {models_dir}")
    
    print_success("Installation verification completed")
    return True


def main():
    """
    Main installation function with modern ComfyUI support.
    
    Returns:
    --------
    bool
        True if installation succeeded, False otherwise
    """
    print_step("Starting Geeky Kokoro TTS installation for ComfyUI v3.49+")
    
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
    node_dir = setup_modern_directory_structure(comfy_dir, script_dir)
    
    # Install dependencies
    print_step("Installing Python dependencies")
    
    if not install_kokoro_dependencies():
        print_error("Failed to install core dependencies. Installation aborted.")
        return False
    
    # Install audio dependencies (optional for full functionality)
    install_audio_dependencies()
    
    # Copy node files
    if not copy_updated_files(script_dir, node_dir):
        print_warning("Some files could not be copied. Installation may be incomplete.")
    
    # Verify installation
    if not verify_installation(node_dir, comfy_dir):
        print_warning("Installation verification had issues.")
    
    # Final success message and instructions
    print_success("🎉 Geeky Kokoro TTS installation completed!")
    print_success(f"Node installed at: {node_dir}")
    print_success("Next steps:")
    print_substep("1. Restart ComfyUI completely")
    print_substep("2. Look for '🔊 Geeky Kokoro TTS (Updated)' in the node menu")
    print_substep("3. Models will be downloaded automatically on first use")
    print_substep("4. Check the console for any error messages")
    
    if comfy_parent.name.lower().startswith("comfyui"):
        print_substep("5. For portable installations, models are cached in your user directory")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print_error("\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Installation failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)