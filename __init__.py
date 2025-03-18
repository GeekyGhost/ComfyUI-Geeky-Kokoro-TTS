"""
Geeky Kokoro TTS and Voice Mod nodes for ComfyUI.
This package provides TTS functionality and voice effect processing.
"""
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GeekyKokoroTTS")

# Import the TTS node
try:
    from .node import NODE_CLASS_MAPPINGS as TTS_NODE_CLASS_MAPPINGS
    from .node import NODE_DISPLAY_NAME_MAPPINGS as TTS_NODE_DISPLAY_NAME_MAPPINGS
    TTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing TTS node: {e}")
    TTS_NODE_CLASS_MAPPINGS = {}
    TTS_NODE_DISPLAY_NAME_MAPPINGS = {}
    TTS_AVAILABLE = False

# Import the Voice Mod node
try:
    from .GeekyKokoroVoiceModNode import NODE_CLASS_MAPPINGS as VOICE_MOD_NODE_CLASS_MAPPINGS
    from .GeekyKokoroVoiceModNode import NODE_DISPLAY_NAME_MAPPINGS as VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS
    VOICE_MOD_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing Voice Mod node: {e}")
    VOICE_MOD_NODE_CLASS_MAPPINGS = {}
    VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS = {}
    VOICE_MOD_AVAILABLE = False

# Merge the mappings
NODE_CLASS_MAPPINGS = {**TTS_NODE_CLASS_MAPPINGS, **VOICE_MOD_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**TTS_NODE_DISPLAY_NAME_MAPPINGS, **VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Show success message
if TTS_AVAILABLE and VOICE_MOD_AVAILABLE:
    logger.info("🔊 Geeky Kokoro TTS nodes and Voice Mod loaded successfully!")
elif TTS_AVAILABLE:
    logger.info("🔊 Geeky Kokoro TTS nodes loaded successfully! Voice Mod not available.")
elif VOICE_MOD_AVAILABLE:
    logger.info("🔊 Geeky Kokoro Voice Mod loaded successfully! TTS not available.")
else:
    logger.error("❌ Failed to load Geeky Kokoro TTS nodes and Voice Mod!")