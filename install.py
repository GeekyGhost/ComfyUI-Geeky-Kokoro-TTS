#!/usr/bin/env python
import os
import sys
import subprocess
import zipfile
import shutil
import platform
from pathlib import Path
import urllib.request

def is_windows():
    return platform.system() == "Windows"

def is_mac():
    return platform.system() == "Darwin"

def is_linux():
    return platform.system() == "Linux"

def print_step(message):
    """Print a step message with formatting"""
    print("\n\033[1;34m==>\033[0m \033[1m{}\033[0m".format(message))

def print_substep(message):
    """Print a substep message with formatting"""
    print("  \033[1;32m->\033[0m \033[1m{}\033[0m".format(message))

def print_error(message):
    """Print an error message with formatting"""
    print("\033[1;31mError: {}\033[0m".format(message))

def print_warning(message):
    """Print a warning message with formatting"""
    print("\033[1;33mWarning: {}\033[0m".format(message))

def print_success(message):
    """Print a success message with formatting"""
    print("\033[1;32mSuccess: {}\033[0m".format(message))

def run_pip(args, desc=None):
    """Run pip with the specified arguments"""
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
    """Download a file from a URL to a destination"""
    print_substep(f"Downloading {url} to {destination}")
    try:
        urllib.request.urlretrieve(url, destination)
        return True
    except Exception as e:
        print_error(f"Failed to download {url}: {e}")
        return False

def install_audio_dependencies():
    """Install audio processing dependencies with proper handling of prerequisites"""
    print_step("Installing Audio Processing Dependencies")
    
    # First, install core dependencies
    base_dependencies = [
        "numpy>=1.22.0", 
        "scipy>=1.9.0"
    ]
    
    for dep in base_dependencies:
        run_pip(["install", dep], f"Installing {dep}")
    
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
        print_warning("Librosa installation failed. Will create basic fallback methods.")
    
    # Install other audio dependencies
    audio_deps = [
        "soundfile>=0.12.1",
        "tqdm>=4.64.0",
    ]
    
    for dep in audio_deps:
        run_pip(["install", dep], f"Installing {dep}")
    
    print_success("Audio processing dependencies installation completed")

def main():
    # Get the ComfyUI custom nodes directory
    script_dir = Path(__file__).resolve().parent
    
    # Determine if this is being run within the ComfyUI directory structure
    comfy_parent = None
    current_dir = script_dir
    
    for _ in range(5):  # Check up to 5 directory levels up
        if (current_dir / "ComfyUI").exists() and (current_dir / "ComfyUI" / "main.py").exists():
            comfy_parent = current_dir
            break
        
        if (current_dir / "main.py").exists() and (current_dir / "nodes").exists():
            comfy_parent = current_dir.parent
            break
        
        current_dir = current_dir.parent
    
    if not comfy_parent:
        print_error("Could not find ComfyUI installation. Please run this script from within your ComfyUI directory structure.")
        print_warning("If you haven't installed ComfyUI yet, please do so first.")
        print_warning("You can also manually specify the ComfyUI path by editing this script.")
        return False
    
    if (comfy_parent / "ComfyUI").exists():
        comfy_dir = comfy_parent / "ComfyUI"
    else:
        comfy_dir = comfy_parent
    
    custom_nodes_dir = comfy_dir / "custom_nodes"
    kokoro_dir = custom_nodes_dir / "geeky_kokoro_tts"
    
    print_step(f"Found ComfyUI installation at: {comfy_dir}")
    
    # Create custom_nodes directory if it doesn't exist
    if not custom_nodes_dir.exists():
        print_substep("Creating custom_nodes directory")
        custom_nodes_dir.mkdir(exist_ok=True)
    
    # Create geeky_kokoro_tts directory if it doesn't exist (or clear it if it does)
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
    
    # Copy node.py to custom_nodes/geeky_kokoro_tts
    print_step("Installing Geeky Kokoro TTS node")
    
    # Copy node implementation files
    node_files = {
        "node.py": script_dir / "node.py",
        "__init__.py": script_dir / "__init__.py",
        "requirements.txt": script_dir / "requirements.txt",
        "README.md": script_dir / "README.md",
        "GeekyKokoroVoiceModNode.py": script_dir / "GeekyKokoroVoiceModNode.py",
        "audio_utils.py": script_dir / "audio_utils.py"  # New fallback utilities
    }
    
    for dest_name, source_path in node_files.items():
        dest_path = kokoro_dir / dest_name
        
        # Use the file from this script's directory if it exists, otherwise use embedded content
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            print_substep(f"Copied {dest_name} to {dest_path}")
        else:
            # Fallback to embedded content if the file doesn't exist
            with open(dest_path, "w", encoding="utf-8") as f:
                if dest_name == "node.py":
                    print_substep("Creating node.py from embedded content")
                    # This is just a placeholder - in a real script, you'd have the content embedded
                    f.write("# Created by install.py - node implementation\n")
                elif dest_name == "__init__.py":
                    print_substep("Creating __init__.py from embedded content")
                    f.write('from .node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS\n\nfrom .GeekyKokoroVoiceModNode import NODE_CLASS_MAPPINGS as VOICE_MOD_NODE_CLASS_MAPPINGS\nfrom .GeekyKokoroVoiceModNode import NODE_DISPLAY_NAME_MAPPINGS as VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS\n\n# Merge the mappings\nNODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **VOICE_MOD_NODE_CLASS_MAPPINGS}\nNODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS}\n\n__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]\n\nprint("🔊 Geeky Kokoro TTS nodes and Voice Mod loaded successfully!")')
                elif dest_name == "requirements.txt":
                    print_substep("Creating requirements.txt from embedded content")
                    f.write("kokoro>=0.8.4\nsoundfile>=0.12.1\nnumpy>=1.22.0\ntorch>=2.0.0\ntqdm>=4.64.0\neinops>=0.6.0\nlibrosa>=0.10.0\nscipy>=1.9.0\nresampy>=0.4.0")
    
    # Install dependencies with enhanced method
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
    install_audio_dependencies()
    
    # Download model files
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
    
    print_step("Checking Kokoro model files")
    
    # Model URLs
    model_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
    voices_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
    
    model_file = model_dir / "kokoro-v1.0.onnx"
    voices_file = model_dir / "voices-v1.0.bin"
    
    if not model_file.exists():
        print_substep("Model file not found. Downloading...")
        if not download_file(model_url, model_file):
            print_warning("Failed to download model file. You will need to download it manually.")
    else:
        print_substep("Model file already exists. Skipping download.")
    
    if not voices_file.exists():
        print_substep("Voices file not found. Downloading...")
        if not download_file(voices_url, voices_file):
            print_warning("Failed to download voices file. You will need to download it manually.")
    else:
        print_substep("Voices file already exists. Skipping download.")
    
    print_success("Geeky Kokoro TTS node for ComfyUI has been installed successfully!")
    print_success(f"Node files installed at: {kokoro_dir}")
    print_success("Restart ComfyUI to use the new node.")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_error(f"Installation failed: {e}")
        import traceback
        traceback.print_exc()