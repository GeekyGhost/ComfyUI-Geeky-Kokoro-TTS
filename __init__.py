"""
Geeky Kokoro TTS and Voice Mod nodes for ComfyUI - 2025 Edition
Complete rewrite with 54+ voices across 9 languages
Python 3.12 and 3.13 compatible
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
    logger.info("✅ Geeky Kokoro TTS node loaded successfully (2025 Edition with 54+ voices)")
except ImportError as e:
    logger.error(f"❌ Error importing TTS node: {e}")
    TTS_NODE_CLASS_MAPPINGS = {}
    TTS_NODE_DISPLAY_NAME_MAPPINGS = {}
    TTS_AVAILABLE = False

# Import the Voice Mod node
try:
    from .GeekyKokoroVoiceModNode import NODE_CLASS_MAPPINGS as VOICE_MOD_NODE_CLASS_MAPPINGS
    from .GeekyKokoroVoiceModNode import NODE_DISPLAY_NAME_MAPPINGS as VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS
    VOICE_MOD_AVAILABLE = True
    logger.info("✅ Geeky Kokoro Voice Mod node loaded successfully")
except ImportError as e:
    logger.error(f"⚠️  Voice Mod node not available: {e}")
    VOICE_MOD_NODE_CLASS_MAPPINGS = {}
    VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS = {}
    VOICE_MOD_AVAILABLE = False

# Merge the mappings
NODE_CLASS_MAPPINGS = {**TTS_NODE_CLASS_MAPPINGS, **VOICE_MOD_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**TTS_NODE_DISPLAY_NAME_MAPPINGS, **VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Show success message
if TTS_AVAILABLE and VOICE_MOD_AVAILABLE:
    logger.info("🎉 Geeky Kokoro TTS Complete Package loaded successfully!")
    logger.info("   - 54+ voices across 9 languages (US/UK English, Japanese, Chinese, Spanish, French, Hindi, Italian, Portuguese)")
    logger.info("   - Voice blending with adjustable ratios")
    logger.info("   - Advanced voice modification effects")
    logger.info("   - Python 3.12 and 3.13 compatible")
elif TTS_AVAILABLE:
    logger.info("🔊 Geeky Kokoro TTS (2025) loaded successfully! Voice Mod not available.")
elif VOICE_MOD_AVAILABLE:
    logger.info("🔊 Geeky Kokoro Voice Mod loaded successfully! TTS not available.")
else:
    logger.error("❌ Failed to load Geeky Kokoro TTS nodes!")
