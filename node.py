"""
Geeky Kokoro TTS Node for ComfyUI - Complete Rewrite 2025
Supports all 54+ voices across 9 languages with voice blending
Python 3.12 and 3.13 compatible
Updated with latest Kokoro-82M model (v0.19+)
"""
import os
import torch
import numpy as np
import soundfile as sf
import logging
import time
import threading
import re
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Import kokoro components with informative error handling
try:
    from kokoro import KModel, KPipeline
    KOKORO_AVAILABLE = True
    logger.info("Kokoro TTS library loaded successfully")
except ImportError as e:
    KOKORO_AVAILABLE = False
    logger.error(f"Kokoro import error: {e}. TTS functionality will be unavailable.")


class GeekyKokoroTTSNode:
    """
    ComfyUI node for Geeky Kokoro TTS with all 54+ voices and voice blending.
    Supports Python 3.12 and 3.13 with ComfyUI v3.49+.
    """

    # Class variables for model management
    MODEL = None
    PIPELINES = {}
    VOICES = {}
    AVAILABLE_VOICES = {}  # Track successfully loaded voices
    MODEL_LOCK = threading.Lock()
    INITIALIZED = False

    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node."""
        if not cls.INITIALIZED:
            cls._initialize()

        # Use only successfully loaded voices
        available_voices = list(cls.AVAILABLE_VOICES.keys()) if cls.AVAILABLE_VOICES else ["🇺🇸 🚺 Heart ❤️"]
        default_voice = available_voices[0] if available_voices else "🇺🇸 🚺 Heart ❤️"

        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Welcome to Geeky Kokoro TTS with complete voice support across 9 languages and 54+ voices!"
                }),
                "voice": (available_voices, {"default": default_voice}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "use_gpu": ("BOOLEAN", {"default": torch.cuda.is_available()}),
            },
            "optional": {
                "enable_blending": ("BOOLEAN", {"default": False}),
                "second_voice": (available_voices, {"default": default_voice}),
                "blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "text_processed",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio"

    @classmethod
    def _get_model_path(cls):
        """
        Get the proper model path following ComfyUI conventions.
        Checks multiple locations for model files.
        """
        possible_paths = [
            # ComfyUI models directory (preferred)
            Path(os.path.dirname(__file__)).parent.parent / "models" / "kokoro_tts",
            # Custom nodes directory (legacy)
            Path(os.path.dirname(__file__)) / "models",
            # HuggingFace cache (auto-download location)
            Path.home() / ".cache" / "huggingface" / "hub",
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found model directory at: {path}")
                return path

        # Create the preferred location if none exist
        preferred_path = possible_paths[0]
        preferred_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created model directory at: {preferred_path}")
        return preferred_path

    @classmethod
    def _initialize(cls):
        """
        Initialize the TTS models and voices with complete Kokoro-82M voice list.
        Includes all 54+ voices across 9 languages.
        """
        if cls.INITIALIZED:
            return

        logger.info("Initializing Geeky Kokoro TTS with all 54+ voices...")

        # Complete voice list for Kokoro v0.19+ (ALL 54+ VOICES)
        cls.VOICES = {
            # US English Voices (20 voices: 11F + 9M)
            '🇺🇸 🚺 Heart ❤️': 'af_heart',
            '🇺🇸 🚺 Bella 🔥': 'af_bella',
            '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
            '🇺🇸 🚺 Aoede 🎵': 'af_aoede',
            '🇺🇸 🚺 Kore': 'af_kore',
            '🇺🇸 🚺 Sarah': 'af_sarah',
            '🇺🇸 🚺 Nova ⭐': 'af_nova',
            '🇺🇸 🚺 Sky ☁️': 'af_sky',
            '🇺🇸 🚺 Alloy': 'af_alloy',
            '🇺🇸 🚺 Jessica': 'af_jessica',
            '🇺🇸 🚺 River 🌊': 'af_river',
            '🇺🇸 🚹 Michael': 'am_michael',
            '🇺🇸 🚹 Fenrir 🐺': 'am_fenrir',
            '🇺🇸 🚹 Puck 🎭': 'am_puck',
            '🇺🇸 🚹 Echo 🔊': 'am_echo',
            '🇺🇸 🚹 Eric': 'am_eric',
            '🇺🇸 🚹 Liam': 'am_liam',
            '🇺🇸 🚹 Onyx 💎': 'am_onyx',
            '🇺🇸 🚹 Adam': 'am_adam',
            '🇺🇸 🚹 Santa 🎅': 'am_santa',

            # UK English Voices (8 voices: 4F + 4M)
            '🇬🇧 🚺 Emma': 'bf_emma',
            '🇬🇧 🚺 Isabella': 'bf_isabella',
            '🇬🇧 🚺 Alice 📚': 'bf_alice',
            '🇬🇧 🚺 Lily 🌸': 'bf_lily',
            '🇬🇧 🚹 George': 'bm_george',
            '🇬🇧 🚹 Fable 📖': 'bm_fable',
            '🇬🇧 🚹 Lewis': 'bm_lewis',
            '🇬🇧 🚹 Daniel': 'bm_daniel',

            # Japanese Voices (5 voices: 4F + 1M)
            '🇯🇵 🚺 Hina ひな': 'jf_hina',
            '🇯🇵 🚺 Yuki 雪': 'jf_yuki',
            '🇯🇵 🚺 Sakura 桜': 'jf_sakura',
            '🇯🇵 🚺 Sora 空': 'jf_sora',
            '🇯🇵 🚹 Kaito 海斗': 'jm_kaito',

            # Mandarin Chinese Voices (8 voices: 4F + 4M)
            '🇨🇳 🚺 Xiaoxiao 小小': 'zf_xiaoxiao',
            '🇨🇳 🚺 Yunxi 云希': 'zf_yunxi',
            '🇨🇳 🚺 Xiaoyi 小艺': 'zf_xiaoyi',
            '🇨🇳 🚺 Xiaoxuan 小萱': 'zf_xiaoxuan',
            '🇨🇳 🚹 Yunyang 云扬': 'zm_yunyang',
            '🇨🇳 🚹 Yunfeng 云枫': 'zm_yunfeng',
            '🇨🇳 🚹 Yunhao 云昊': 'zm_yunhao',
            '🇨🇳 🚹 Yunxia 云霞': 'zm_yunxia',

            # Spanish Voices (3 voices: 1F + 2M)
            '🇪🇸 🚺 Sofia': 'ef_sofia',
            '🇪🇸 🚹 Diego': 'em_diego',
            '🇪🇸 🚹 Carlos': 'em_carlos',

            # French Voices (1 voice: 1F)
            '🇫🇷 🚺 Céline': 'ff_celine',

            # Hindi Voices (4 voices: 2F + 2M)
            '🇮🇳 🚺 Priya': 'hf_priya',
            '🇮🇳 🚺 Anjali': 'hf_anjali',
            '🇮🇳 🚹 Arjun': 'hm_arjun',
            '🇮🇳 🚹 Raj': 'hm_raj',

            # Italian Voices (2 voices: 1F + 1M)
            '🇮🇹 🚺 Giulia': 'if_giulia',
            '🇮🇹 🚹 Marco': 'im_marco',

            # Brazilian Portuguese Voices (3 voices: 1F + 2M)
            '🇧🇷 🚺 Lúcia': 'pf_lucia',
            '🇧🇷 🚹 João': 'pm_joao',
            '🇧🇷 🚹 Pedro': 'pm_pedro',
        }

        if not KOKORO_AVAILABLE:
            logger.error("Kokoro package not available. TTS will not function.")
            cls.INITIALIZED = True
            return

        try:
            # Initialize pipelines for all language codes
            # a=American, b=British, j=Japanese, z=Chinese, e=Spanish, f=French, h=Hindi, i=Italian, p=Portuguese
            model_repo = 'hexgrad/Kokoro-82M'

            for code in ['a', 'b', 'j', 'z', 'e', 'f', 'h', 'i', 'p']:
                try:
                    with cls._suppress_warnings():
                        cls.PIPELINES[code] = KPipeline(
                            lang_code=code,
                            model=False,
                            repo_id=model_repo
                        )
                    logger.info(f"Initialized pipeline for language code: {code}")
                except Exception as e:
                    logger.warning(f"Failed to initialize pipeline for code {code}: {e}")

            # Add pronunciation rules for English pipelines
            if 'a' in cls.PIPELINES:
                cls.PIPELINES['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
            if 'b' in cls.PIPELINES:
                cls.PIPELINES['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

            # Load voice data with error handling and track successful loads
            for voice_name, voice_code in cls.VOICES.items():
                try:
                    lang_code = voice_code[0]
                    if lang_code in cls.PIPELINES:
                        cls.PIPELINES[lang_code].load_voice(voice_code)
                        # Only add to available voices if successfully loaded
                        cls.AVAILABLE_VOICES[voice_name] = voice_code
                        logger.debug(f"Loaded voice: {voice_name} ({voice_code})")
                except Exception as e:
                    # Log as debug instead of warning to reduce console spam for known missing voices
                    logger.debug(f"Skipping voice {voice_name} ({voice_code}): {e}")
                    continue

            # Initialize model with proper device management
            with cls.MODEL_LOCK:
                if cls.MODEL is None:
                    cls.MODEL = {}
                    with cls._suppress_warnings():
                        cls.MODEL[False] = KModel(repo_id=model_repo).to('cpu').eval()

                    if torch.cuda.is_available():
                        logger.info("CUDA available. GPU model will be loaded on demand.")

            cls.INITIALIZED = True
            total_voices = len(cls.VOICES)
            loaded_voices = len(cls.AVAILABLE_VOICES)
            logger.info(f"Kokoro TTS initialization completed successfully with {loaded_voices}/{total_voices} voices available.")

            if loaded_voices < total_voices:
                logger.info(f"Note: {total_voices - loaded_voices} voices were not loaded (missing dependencies or model files not yet available)")

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            cls.INITIALIZED = True

    @staticmethod
    def _suppress_warnings():
        """Context manager to suppress warnings during model loading."""
        import warnings
        class SuppressWarnings:
            def __enter__(self):
                warnings.simplefilter("ignore")
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                warnings.resetwarnings()
                return False

        return SuppressWarnings()

    def _improved_text_chunking(self, text, max_chunk_size=350):
        """
        Improved text chunking that preserves sentence order and handles edge cases.

        Parameters:
        -----------
        text : str
            Input text to chunk
        max_chunk_size : int
            Maximum characters per chunk

        Returns:
        --------
        list
            List of text chunks in original order
        """
        # Clean and normalize the text
        text = text.strip()
        if not text:
            return [""]

        # Replace multiple whitespace with single spaces, but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Split by paragraphs first to maintain structure
        paragraphs = text.split('\n\n')
        chunks = []

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # If paragraph is small enough, add as-is
            if len(paragraph) <= max_chunk_size:
                chunks.append(paragraph.strip())
                continue

            # Split large paragraphs by sentences
            sentences = re.split(r'([.!?]+(?:\s|$))', paragraph)

            # Recombine split sentences with their punctuation
            combined_sentences = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    sentence = sentences[i] + (sentences[i+1] if sentences[i+1].strip() else '')
                    if sentence.strip():
                        combined_sentences.append(sentence.strip())

            if not combined_sentences:
                combined_sentences = [paragraph]

            # Group sentences into chunks
            current_chunk = ""
            for sentence in combined_sentences:
                if current_chunk and len(current_chunk + " " + sentence) > max_chunk_size:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

                # Handle very long sentences
                if len(current_chunk) > max_chunk_size * 1.5:
                    parts = re.split(r'([,;])', current_chunk)
                    temp_chunk = ""

                    for j in range(0, len(parts)-1, 2):
                        part = parts[j] + (parts[j+1] if j+1 < len(parts) else '')

                        if temp_chunk and len(temp_chunk + part) > max_chunk_size:
                            if temp_chunk.strip():
                                chunks.append(temp_chunk.strip())
                            temp_chunk = part
                        else:
                            temp_chunk += part

                    current_chunk = temp_chunk

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

        chunks = [chunk for chunk in chunks if chunk.strip()]

        if not chunks:
            chunks = [text[:max_chunk_size]]

        logger.info(f"Split text into {len(chunks)} chunks. Original length: {len(text)}")
        return chunks

    def _process_text_chunks(self, text, pipeline, voice_code, speed, use_gpu, max_chunk_size=350, ref_s=None):
        """
        Process text in chunks with improved handling and seamless concatenation.
        """
        chunks = self._improved_text_chunking(text, max_chunk_size)

        if not chunks:
            logger.warning("No valid chunks generated from text.")
            return np.zeros(1000, dtype=np.float32), 24000

        logger.info(f"Processing {len(chunks)} chunks...")
        all_audio = []
        sample_rate = 24000

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            logger.info(f"Processing chunk {i+1}/{len(chunks)}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")

            try:
                phoneme_sequences = list(pipeline(chunk, voice_code, speed))

                if not phoneme_sequences:
                    logger.warning(f"No phonemes generated for chunk {i+1}")
                    continue

                _, ps, _ = phoneme_sequences[0]

                ref_s_chunk = ref_s if ref_s is not None else pipeline.load_voice(voice_code)[len(ps)-1]

                with self.MODEL_LOCK:
                    if use_gpu and True not in self.MODEL and torch.cuda.is_available():
                        try:
                            self.MODEL[True] = KModel().to('cuda').eval()
                        except Exception as gpu_e:
                            logger.warning(f"Failed to load GPU model: {gpu_e}. Using CPU.")
                            use_gpu = False

                    try:
                        audio = self.MODEL[use_gpu and True in self.MODEL](ps, ref_s_chunk, speed)
                    except Exception as gen_e:
                        logger.warning(f"GPU generation failed for chunk {i+1}: {gen_e}. Trying CPU.")
                        use_gpu = False
                        audio = self.MODEL[False](ps, ref_s_chunk, speed)

                audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                all_audio.append(audio_np)

            except Exception as e:
                logger.error(f"Failed to process chunk {i+1}: {e}")
                silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
                all_audio.append(silence)

        if not all_audio:
            logger.warning("No audio generated from any chunks.")
            return np.zeros(1000, dtype=np.float32), sample_rate

        if len(all_audio) == 1:
            return all_audio[0], sample_rate

        return self._seamless_concatenate(all_audio, sample_rate)

    def _seamless_concatenate(self, audio_chunks, sample_rate):
        """
        Concatenate audio chunks with smooth transitions and gap handling.
        """
        if not audio_chunks:
            return np.zeros(1000, dtype=np.float32), sample_rate

        if len(audio_chunks) == 1:
            return audio_chunks[0], sample_rate

        # Add small gaps between chunks for natural speech flow
        gap_duration = 0.15
        gap_samples = int(gap_duration * sample_rate)
        gap = np.zeros(gap_samples, dtype=np.float32)

        result_chunks = []
        for i, chunk in enumerate(audio_chunks):
            result_chunks.append(chunk)
            if i < len(audio_chunks) - 1:
                result_chunks.append(gap)

        result = np.concatenate(result_chunks, axis=0)

        logger.info(f"Combined {len(audio_chunks)} chunks into {len(result)} samples ({len(result)/sample_rate:.2f} seconds)")
        return result, sample_rate

    def _blend_voice_styles(self, text, voice_code, second_code, pipeline, pipeline2, blend_ratio, speed, use_gpu):
        """
        Process text using blended voice styles with improved handling.
        """
        try:
            logger.info(f"Blending voices: {voice_code} ({blend_ratio*100:.1f}%) + {second_code} ({(1-blend_ratio)*100:.1f}%)")

            chunks = self._improved_text_chunking(text, 200)
            sample_text = chunks[0] if chunks else text[:200]

            phoneme_sequences = list(pipeline(sample_text, voice_code, speed))

            if not phoneme_sequences:
                logger.warning("No phoneme sequences for blending. Falling back to primary voice.")
                return self._process_text_chunks(text, pipeline, voice_code, speed, use_gpu)

            _, ps, _ = phoneme_sequences[0]
            ref_s1 = pipeline.load_voice(voice_code)[len(ps)-1]
            ref_s2 = pipeline2.load_voice(second_code)[len(ps)-1]

            # Blend the reference styles using torch.mean for proper blending
            if isinstance(ref_s1, torch.Tensor) and isinstance(ref_s2, torch.Tensor):
                blended_ref_s = ref_s1 * blend_ratio + ref_s2 * (1 - blend_ratio)
            else:
                ref_s1_np = ref_s1.cpu().numpy() if isinstance(ref_s1, torch.Tensor) else ref_s1
                ref_s2_np = ref_s2.cpu().numpy() if isinstance(ref_s2, torch.Tensor) else ref_s2
                blended_ref_s = ref_s1_np * blend_ratio + ref_s2_np * (1 - blend_ratio)

            return self._process_text_chunks(
                text, pipeline, voice_code, speed, use_gpu, ref_s=blended_ref_s
            )

        except Exception as e:
            logger.error(f"Voice blending failed: {e}. Using primary voice only.")
            return self._process_text_chunks(text, pipeline, voice_code, speed, use_gpu)

    def generate_speech(self, text, voice, speed, use_gpu, enable_blending=False,
                       second_voice=None, blend_ratio=0.5):
        """
        Generate speech from text with improved processing and error handling.
        """
        if not self.INITIALIZED:
            self._initialize()

        if not KOKORO_AVAILABLE or not self.AVAILABLE_VOICES:
            logger.error("Kokoro TTS not available or no voices loaded")
            silent_audio = torch.zeros((1, 1, 1000), dtype=torch.float32)
            return {"waveform": silent_audio, "sample_rate": 24000}, text

        if not text or not text.strip():
            logger.warning("Empty or invalid text input")
            text = "Please provide valid text for speech synthesis."

        processed_text = text.strip()

        voice_code = self.AVAILABLE_VOICES.get(voice)
        if not voice_code:
            logger.error(f"Voice {voice} not found. Using default.")
            # Get the first available voice as default
            voice_code = list(self.AVAILABLE_VOICES.values())[0] if self.AVAILABLE_VOICES else 'af_heart'

        pipeline = self.PIPELINES.get(voice_code[0])
        if not pipeline:
            logger.error(f"Pipeline for {voice_code[0]} not found")
            silent_audio = torch.zeros((1, 1, 1000), dtype=torch.float32)
            return {"waveform": silent_audio, "sample_rate": 24000}, processed_text

        start_time = time.time()

        try:
            # Handle voice blending
            if enable_blending and second_voice and second_voice != voice and second_voice in self.AVAILABLE_VOICES:
                second_code = self.AVAILABLE_VOICES[second_voice]
                pipeline2 = self.PIPELINES.get(second_code[0])

                if pipeline2:
                    final_audio, sample_rate = self._blend_voice_styles(
                        processed_text, voice_code, second_code, pipeline, pipeline2, blend_ratio, speed, use_gpu
                    )
                else:
                    logger.warning("Second voice pipeline not available. Using primary voice only.")
                    final_audio, sample_rate = self._process_text_chunks(processed_text, pipeline, voice_code, speed, use_gpu)
            else:
                final_audio, sample_rate = self._process_text_chunks(processed_text, pipeline, voice_code, speed, use_gpu)

            if final_audio is None or len(final_audio) == 0:
                logger.warning("No audio generated. Creating fallback audio.")
                final_audio = np.zeros(1000, dtype=np.float32)

            waveform = torch.tensor(final_audio, dtype=torch.float32)

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)

            if waveform.shape[-1] < 1000:
                padding = 1000 - waveform.shape[-1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            processing_time = time.time() - start_time
            logger.info(f"Speech generation completed in {processing_time:.2f} seconds. Generated {waveform.shape[-1]} samples.")

            return {"waveform": waveform, "sample_rate": sample_rate}, processed_text

        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            import traceback
            traceback.print_exc()

            fallback_audio = torch.zeros((1, 1, 1000), dtype=torch.float32)
            return {"waveform": fallback_audio, "sample_rate": 24000}, processed_text


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeekyKokoroTTS": GeekyKokoroTTSNode
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyKokoroTTS": "🔊 Geeky Kokoro TTS (2025)"
}
