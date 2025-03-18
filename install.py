#!/usr/bin/env python
"""
Installation script for the Geeky Kokoro TTS and Voice Mod nodes for ComfyUI.
Handles dependency installation and file setup.
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


def print_step(message):
    """Print a step message with formatting."""
    logger.info(message)
    print("\n\033[1;34m==>\033[0m \033[1m{}\033[0m".format(message))


def print_substep(message):
    """Print a substep message with formatting."""
    logger.debug(message)
    print("  \033[1;32m->\033[0m \033[1m{}\033[0m".format(message))


def print_error(message):
    """Print an error message with formatting."""
    logger.error(message)
    print("\033[1;31mError: {}\033[0m".format(message))


def print_warning(message):
    """Print a warning message with formatting."""
    logger.warning(message)
    print("\033[1;33mWarning: {}\033[0m".format(message))


def print_success(message):
    """Print a success message with formatting."""
    logger.info(message)
    print("\033[1;32mSuccess: {}\033[0m".format(message))


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
        subprocess.check_call(pip_command)
        return True
    except subprocess.CalledProcessError:
        return False


def download_file(url, destination):
    """
    Download a file from a URL to a destination.
    
    Parameters:
    -----------
    url : str
        URL to download from
    destination : Path
        Destination path
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    print_substep(f"Downloading {url} to {destination}")
    try:
        urllib.request.urlretrieve(url, destination)
        return True
    except Exception as e:
        print_error(f"Failed to download {url}: {e}")
        return False


def install_audio_dependencies():
    """
    Install audio processing dependencies with proper handling of prerequisites.
    
    Returns:
    --------
    bool
        True if all critical dependencies were installed, False otherwise
    """
    print_step("Installing Audio Processing Dependencies")
    
    # First, install core dependencies
    base_dependencies = [
        "numpy>=1.22.0", 
        "scipy>=1.9.0"
    ]
    
    base_success = True
    for dep in base_dependencies:
        if not run_pip(["install", dep], f"Installing {dep}"):
            base_success = False
            print_error(f"Failed to install critical dependency: {dep}")
    
    if not base_success:
        print_error("Failed to install basic dependencies. Aborting.")
        return False
    
    # Install numba (required for resampy)
    numba_success = run_pip(["install", "numba>=0.56.0"], "Installing Numba (required for resampy)")
    
    # If numba install fails, try installing llvmlite first (numba dependency)
    if not numba_success:
        print_warning("Numba installation failed, trying to install llvmlite first...")
        run_pip(["install", "llvmlite"], "Installing llvmlite")
        numba_success = run_pip(["install", "numba>=0.56.0"], "Retrying Numba installation")
    
    # Install resampy with specific version
    if numba_success:
        resampy_success = run_pip(["install", "resampy==0.4.2"], "Installing resampy for high-quality audio processing")
        if not resampy_success:
            print_warning("Resampy installation failed. Some audio effects may use fallback methods.")
    else:
        print_warning("Numba installation failed. Audio effects will use fallback methods.")
    
    # Install librosa separately to avoid dependency issues
    librosa_success = run_pip(["install", "librosa>=0.10.0"], "Installing librosa audio processing library")
    if not librosa_success:
        print_warning("Librosa installation failed. Will use basic fallback methods.")
    
    # Install other audio dependencies
    audio_deps = [
        "soundfile>=0.12.1",
        "tqdm>=4.64.0",
    ]
    
    for dep in audio_deps:
        run_pip(["install", dep], f"Installing {dep}")
    
    print_success("Audio processing dependencies installation completed")
    return True


def find_comfyui_directory(script_dir):
    """
    Find the ComfyUI installation directory.
    
    Parameters:
    -----------
    script_dir : Path
        Directory of the installation script
        
    Returns:
    --------
    tuple
        (comfy_parent, comfy_dir) or (None, None) if not found
    """
    # Determine if this is being run within the ComfyUI directory structure
    comfy_parent = None
    current_dir = script_dir
    
    # Check up to 5 directory levels up
    for _ in range(5):
        # Check if this is the parent of a ComfyUI installation
        if (current_dir / "ComfyUI").exists() and (current_dir / "ComfyUI" / "main.py").exists():
            comfy_parent = current_dir
            comfy_dir = current_dir / "ComfyUI"
            return comfy_parent, comfy_dir
        
        # Check if this is a ComfyUI installation itself
        if (current_dir / "main.py").exists() and (current_dir / "nodes").exists():
            comfy_parent = current_dir.parent
            comfy_dir = current_dir
            return comfy_parent, comfy_dir
        
        # Move up one directory level
        current_dir = current_dir.parent
    
    return None, None


def setup_kokoro_directory(comfy_dir, script_dir):
    """
    Set up the Geeky Kokoro TTS directory structure.
    
    Parameters:
    -----------
    comfy_dir : Path
        ComfyUI directory
    script_dir : Path
        Installation script directory
        
    Returns:
    --------
    Path
        Path to the Kokoro directory
    """
    custom_nodes_dir = comfy_dir / "custom_nodes"
    kokoro_dir = custom_nodes_dir / "geeky_kokoro_tts"
    
    # Create custom_nodes directory if it doesn't exist
    if not custom_nodes_dir.exists():
        print_substep("Creating custom_nodes directory")
        custom_nodes_dir.mkdir(exist_ok=True)
    
    # Handle existing installation
    if kokoro_dir.exists():
        print_substep("Updating existing geeky_kokoro_tts directory")
        
        # Preserve the models directory if it exists
        model_dir = kokoro_dir / "models"
        if model_dir.exists():
            print_substep("Preserving existing model files")
            temp_model_dir = comfy_dir / "temp_models"
            if temp_model_dir.exists():
                shutil.rmtree(temp_model_dir)
            shutil.copytree(model_dir, temp_model_dir)
        
        # Clear the directory except for models
        for item in kokoro_dir.iterdir():
            if item.name != "models" and item.name != "temp_models":
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    else:
        print_substep("Creating geeky_kokoro_tts directory")
        kokoro_dir.mkdir(exist_ok=True)
    
    return kokoro_dir


def copy_node_files(script_dir, kokoro_dir):
    """
    Copy node implementation files to the installation directory.
    
    Parameters:
    -----------
    script_dir : Path
        Source directory (script location)
    kokoro_dir : Path
        Destination directory
        
    Returns:
    --------
    bool
        True if all required files were copied, False otherwise
    """
    print_step("Installing Geeky Kokoro TTS and Voice Mod nodes")
    
    # Required node files to copy
    node_files = {
        "node.py": script_dir / "node.py",
        "__init__.py": script_dir / "__init__.py",
        "requirements.txt": script_dir / "requirements.txt",
        "README.md": script_dir / "README.md",
        "GeekyKokoroVoiceModNode.py": script_dir / "GeekyKokoroVoiceModNode.py",
        "audio_utils.py": script_dir / "audio_utils.py",
        "voice_profiles_utils.py": script_dir / "voice_profiles_utils.py"
    }
    
    success = True
    missing_files = []
    
    for dest_name, source_path in node_files.items():
        dest_path = kokoro_dir / dest_name
        
        # Copy if file exists, otherwise log error
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            print_substep(f"Copied {dest_name} to {dest_path}")
        else:
            missing_files.append(dest_name)
            success = False
    
    if missing_files:
        print_warning(f"Could not find these files: {', '.join(missing_files)}")
        print_warning("Some functionality may be limited.")
    
    return success


def setup_model_files(comfy_dir, kokoro_dir):
    """
    Set up model files, either by restoring from temp or downloading.
    
    Parameters:
    -----------
    comfy_dir : Path
        ComfyUI directory
    kokoro_dir : Path
        Kokoro installation directory
        
    Returns:
    --------
    bool
        True if models are available, False otherwise
    """
    print_step("Setting up Kokoro model files")
    
    model_dir = kokoro_dir / "models"
    
    # Restore models from temp directory if it exists
    temp_model_dir = comfy_dir / "temp_models"
    if temp_model_dir.exists():
        print_substep("Restoring preserved model files")
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.copytree(temp_model_dir, model_dir)
        shutil.rmtree(temp_model_dir)
    else:
        model_dir.mkdir(exist_ok=True)
    
    # Check model files
    model_file = model_dir / "kokoro-v1.0.onnx"
    voices_file = model_dir / "voices-v1.0.bin"
    
    # Model URLs
    model_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
    voices_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
    
    # Download model file if missing
    if not model_file.exists():
        print_substep("Model file not found. Downloading...")
        if not download_file(model_url, model_file):
            print_warning("Failed to download model file. You will need to download it manually.")
            return False
    else:
        print_substep("Model file already exists. Skipping download.")
    
    # Download voices file if missing
    if not voices_file.exists():
        print_substep("Voices file not found. Downloading...")
        if not download_file(voices_url, voices_file):
            print_warning("Failed to download voices file. You will need to download it manually.")
            return False
    else:
        print_substep("Voices file already exists. Skipping download.")
    
    return True


def main():
    """
    Main installation function.
    
    Returns:
    --------
    bool
        True if installation succeeded, False otherwise
    """
    # Get the script directory
    script_dir = Path(__file__).resolve().parent
    
    # Find ComfyUI installation
    comfy_parent, comfy_dir = find_comfyui_directory(script_dir)
    
    if not comfy_dir:
        print_error("Could not find ComfyUI installation. Please run this script from within your ComfyUI directory structure.")
        print_warning("If you haven't installed ComfyUI yet, please do so first.")
        print_warning("You can also manually specify the ComfyUI path by editing this script.")
        return False
    
    print_step(f"Found ComfyUI installation at: {comfy_dir}")
    
    # Set up the Kokoro directory
    kokoro_dir = setup_kokoro_directory(comfy_dir, script_dir)
    
    # Copy node files
    if not copy_node_files(script_dir, kokoro_dir):
        print_warning("Some node files could not be copied. Installation may be incomplete.")
    
    # Install dependencies
    print_step("Installing required dependencies")
    
    # Standard Kokoro TTS dependencies
    core_dependencies = [
        "kokoro>=0.8.4",
        "torch>=2.0.0",
        "einops>=0.6.0"
    ]
    
    for dep in core_dependencies:
        run_pip(["install", dep], f"Installing {dep}")
    
    # Install audio dependencies with better error handling
    if not install_audio_dependencies():
        print_warning("Some audio processing features may not be available.")
    
    # Set up model files
    if not setup_model_files(comfy_dir, kokoro_dir):
        print_warning("Model files may be missing. Basic functionality might be limited.")
    
    print_success("Geeky Kokoro TTS node for ComfyUI has been installed successfully!")
    print_success(f"Node files installed at: {kokoro_dir}")
    print_success("Restart ComfyUI to use the new nodes.")
    
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_error(f"Installation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)