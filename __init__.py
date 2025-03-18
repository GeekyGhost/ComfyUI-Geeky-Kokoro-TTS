"""
Geeky Kokoro TTS and Voice Mod nodes for ComfyUI.
This package provides TTS functionality, voice effect processing, and WhisperSpeech voice cloning.
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

# Import the WhisperSpeech node
try:
    from .GeekyWhisperSpeechNode import NODE_CLASS_MAPPINGS as WHISPER_NODE_CLASS_MAPPINGS
    from .GeekyWhisperSpeechNode import NODE_DISPLAY_NAME_MAPPINGS as WHISPER_NODE_DISPLAY_NAME_MAPPINGS
    WHISPER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing WhisperSpeech node: {e}")
    WHISPER_NODE_CLASS_MAPPINGS = {}
    WHISPER_NODE_DISPLAY_NAME_MAPPINGS = {}
    WHISPER_AVAILABLE = False

# Merge the mappings
NODE_CLASS_MAPPINGS = {
    **TTS_NODE_CLASS_MAPPINGS, 
    **VOICE_MOD_NODE_CLASS_MAPPINGS, 
    **WHISPER_NODE_CLASS_MAPPINGS
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **TTS_NODE_DISPLAY_NAME_MAPPINGS, 
    **VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS, 
    **WHISPER_NODE_DISPLAY_NAME_MAPPINGS
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Show success message
available_nodes = []
if TTS_AVAILABLE:
    available_nodes.append("TTS")
if VOICE_MOD_AVAILABLE:
    available_nodes.append("Voice Mod")
if WHISPER_AVAILABLE:
    available_nodes.append("WhisperSpeech")

if available_nodes:
    logger.info(f"🔊 Geeky Kokoro {', '.join(available_nodes)} loaded successfully!")
else:
    logger.error("❌ Failed to load any Geeky Kokoro nodes!")