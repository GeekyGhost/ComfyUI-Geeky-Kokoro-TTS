#!/usr/bin/env python
"""
Installation script for the Geeky WhisperSpeech node for ComfyUI.
Handles dependency installation and file setup.
"""
import os
import sys
import subprocess
import shutil
import platform
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GeekyWhisperInstaller")


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
        if (current_dir / "main.py").exists() and (current_dir / "web").exists():
            comfy_parent = current_dir.parent
            comfy_dir = current_dir
            return comfy_parent, comfy_dir
        
        # Move up one directory level
        current_dir = current_dir.parent
    
    return None, None


def setup_node_directory(comfy_dir, script_dir):
    """
    Set up the Geeky WhisperSpeech directory structure.
    
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
    custom_nodes_dir = comfy_dir / "custom_nodes"
    node_dir = custom_nodes_dir / "geeky_whisper_speech"
    
    # Create custom_nodes directory if it doesn't exist
    if not custom_nodes_dir.exists():
        print_substep("Creating custom_nodes directory")
        custom_nodes_dir.mkdir(exist_ok=True)
    
    # Handle existing installation
    if node_dir.exists():
        print_substep("Updating existing geeky_whisper_speech directory")
        for item in node_dir.iterdir():
            if item.is_dir():
                if item.name != "temp" and item.name != "models":
                    shutil.rmtree(item)
            else:
                item.unlink()
    else:
        print_substep("Creating geeky_whisper_speech directory")
        node_dir.mkdir(exist_ok=True)
    
    # Create temp directory if it doesn't exist
    temp_dir = node_dir / "temp"
    if not temp_dir.exists():
        temp_dir.mkdir(exist_ok=True)
    
    return node_dir


def copy_node_files(script_dir, node_dir):
    """
    Copy node implementation files to the installation directory.
    
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
    print_step("Installing Geeky WhisperSpeech node")
    
    # Required node files to copy
    node_files = {
        "GeekyWhisperSpeechNode.py": script_dir / "GeekyWhisperSpeechNode.py",
        "__init__.py": script_dir / "__init__.py",
        "requirements.txt": script_dir / "requirements.txt",
        "install.py": script_dir / "install.py",
    }
    
    success = True
    missing_files = []
    
    for dest_name, source_path in node_files.items():
        dest_path = node_dir / dest_name
        
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


def install_dependencies(node_dir):
    """
    Install the required dependencies.
    
    Parameters:
    -----------
    node_dir : Path
        Path to the node directory
        
    Returns:
    --------
    bool
        True if all dependencies were installed, False otherwise
    """
    print_step("Installing required dependencies")
    
    requirements_file = node_dir / "requirements.txt"
    if not requirements_file.exists():
        print_error("Requirements file not found. Cannot install dependencies.")
        return False
    
    # Install packages from requirements.txt
    success = run_pip(["install", "-r", str(requirements_file)], 
                     f"Installing dependencies from {requirements_file}")
    
    if not success:
        print_warning("Some dependencies may not have installed correctly.")
        print_warning("You may need to install them manually:")
        print_warning(f"pip install -r {requirements_file}")
    
    # Try to install additional dependencies that might be needed
    print_substep("Installing additional dependencies")
    
    extra_deps = [
        "torchaudio",  # For audio processing
        "nltk",        # For text processing
        "librosa",     # For audio analysis
    ]
    
    for dep in extra_deps:
        run_pip(["install", dep], f"Installing {dep}")
    
    # Download NLTK data
    try:
        print_substep("Downloading NLTK data for text processing")
        import nltk
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print_warning(f"Failed to download NLTK data: {e}")
    
    return success


def create_portable_install_script(node_dir):
    """
    Create a batch file for installing in the portable version of ComfyUI.
    
    Parameters:
    -----------
    node_dir : Path
        Path to the node directory
    """
    # Only create batch file for Windows
    if platform.system() != "Windows":
        return
    
    batch_content = """@echo off
echo Installing Geeky WhisperSpeech dependencies for ComfyUI portable
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
pause
"""
    
    batch_path = node_dir / "install_portable.bat"
    with open(batch_path, "w") as f:
        f.write(batch_content)
    
    print_substep(f"Created portable installation batch file: {batch_path}")


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
        print_warning("If you haven't installed ComfyUI yet, please do so first