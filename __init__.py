# Import the TTS node
from .node import NODE_CLASS_MAPPINGS as TTS_NODE_CLASS_MAPPINGS
from .node import NODE_DISPLAY_NAME_MAPPINGS as TTS_NODE_DISPLAY_NAME_MAPPINGS

# Import the Voice Mod node
from .GeekyKokoroVoiceModNode import NODE_CLASS_MAPPINGS as VOICE_MOD_NODE_CLASS_MAPPINGS
from .GeekyKokoroVoiceModNode import NODE_DISPLAY_NAME_MAPPINGS as VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS

# Merge the mappings
NODE_CLASS_MAPPINGS = {**TTS_NODE_CLASS_MAPPINGS, **VOICE_MOD_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**TTS_NODE_DISPLAY_NAME_MAPPINGS, **VOICE_MOD_NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("🔊 Geeky Kokoro TTS nodes and Voice Mod loaded successfully!")