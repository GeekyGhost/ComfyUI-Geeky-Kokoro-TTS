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
    # Simplified - no backup handling
    node_dir, _ = setup_modern_directory_structure(comfy_dir, script_dir)
    
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
        return False
    
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
