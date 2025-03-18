"""
Main module for Geeky Kokoro TTS implementation in ComfyUI.
Provides text-to-speech functionality with voice customization.
"""
import os
import torch
import numpy as np
import soundfile as sf
import logging
import time
import threading
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Import kokoro components with informative error handling
try:
    from kokoro import KModel, KPipeline
    KOKORO_AVAILABLE = True
except ImportError as e:
    KOKORO_AVAILABLE = False
    logger.error(f"Kokoro import error: {e}. TTS functionality will be unavailable.")


class GeekyKokoroTTSNode:
    """
    ComfyUI node for Geeky Kokoro TTS with style-based voice blending and improved text processing.
    """
    
    # Class variables for model management
    MODEL = None
    PIPELINES = {}
    VOICES = {}
    MODEL_LOCK = threading.Lock()
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node."""
        if not cls.VOICES:
            cls._initialize()
            
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "Welcome to Geeky Kokoro TTS for ComfyUI. You can adjust voice parameters to customize the output."
                }),
                "voice": (list(cls.VOICES.keys()), {"default": "🇺🇸 🚺 Heart ❤️"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "use_gpu": ("BOOLEAN", {"default": torch.cuda.is_available()}),
            },
            "optional": {
                "enable_blending": ("BOOLEAN", {"default": False}),
                "second_voice": (list(cls.VOICES.keys()), {"default": "🇺🇸 🚺 Sarah"}),
                "blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "display": "slider"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "text_processed",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio"
    
    @classmethod
    def _initialize(cls):
        """
        Initialize the TTS models and voices.
        Called once before first use.
        """
        logger.info("Initializing Geeky Kokoro TTS...")
        
        # Define available voices
        cls.VOICES = {
            '🇺🇸 🚺 Heart ❤️': 'af_heart', '🇺🇸 🚺 Bella 🔥': 'af_bella',
            '🇺🇸 🚺 Nicole 🎧': 'af_nicole', '🇺🇸 🚺 Aoede': 'af_aoede',
            '🇺🇸 🚺 Kore': 'af_kore', '🇺🇸 🚺 Sarah': 'af_sarah',
            '🇺🇸 🚺 Nova': 'af_nova', '🇺🇸 🚺 Sky': 'af_sky',
            '🇺🇸 🚺 Alloy': 'af_alloy', '🇺🇸 🚺 Jessica': 'af_jessica',
            '🇺🇸 🚺 River': 'af_river', '🇺🇸 🚹 Michael': 'am_michael',
            '🇺🇸 🚹 Fenrir': 'am_fenrir', '🇺🇸 🚹 Puck': 'am_puck',
            '🇺🇸 🚹 Echo': 'am_echo', '🇺🇸 🚹 Eric': 'am_eric',
            '🇺🇸 🚹 Liam': 'am_liam', '🇺🇸 🚹 Onyx': 'am_onyx',
            '🇺🇸 🚹 Adam': 'am_adam', '🇬🇧 🚺 Emma': 'bf_emma',
            '🇬🇧 🚺 Isabella': 'bf_isabella', '🇬🇧 🚺 Alice': 'bf_alice',
            '🇬🇧 🚺 Lily': 'bf_lily', '🇬🇧 🚹 George': 'bm_george',
            '🇬🇧 🚹 Fable': 'bm_fable', '🇬🇧 🚹 Lewis': 'bm_lewis',
            '🇬🇧 🚹 Daniel': 'bm_daniel',
        }
        
        if not KOKORO_AVAILABLE:
            logger.error("Kokoro package not available. TTS will not function.")
            return
        
        try:
            # Initialize pipelines for US (a) and UK (b) English
            for code in ['a', 'b']:
                with cls._suppress_warnings():
                    cls.PIPELINES[code] = KPipeline(
                        lang_code=code, 
                        model=False,
                        repo_id='hexgrad/Kokoro-82M'
                    )
            
            # Add pronunciation rules
            cls.PIPELINES['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
            cls.PIPELINES['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
            
            # Load voice data
            for voice_code in cls.VOICES.values():
                cls.PIPELINES[voice_code[0]].load_voice(voice_code)
            
            # Initialize model
            with cls.MODEL_LOCK:
                if cls.MODEL is None:
                    cls.MODEL = {}
                    with cls._suppress_warnings():
                        cls.MODEL[False] = KModel(repo_id='hexgrad/Kokoro-82M').to('cpu').eval()
                    
                    if torch.cuda.is_available():
                        logger.info("CUDA available. GPU model will be loaded on demand.")
            
            logger.info("Initialization complete.")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def _suppress_warnings():
        """Context manager to suppress warnings."""
        import warnings
        class SuppressWarnings:
            def __enter__(self):
                warnings.simplefilter("ignore")
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                warnings.resetwarnings()
                return False
        
        return SuppressWarnings()
    
    def _process_text_chunks(self, text, pipeline, voice_code, speed, use_gpu, max_chunk_size=400, ref_s=None):
        """
        Process text in chunks with improved handling and crossfading.
        Optional ref_s for blended voices.
        
        Parameters:
        -----------
        text : str
            Input text to process
        pipeline : KPipeline
            TTS pipeline to use
        voice_code : str
            Voice identifier
        speed : float
            Playback speed
        use_gpu : bool
            Whether to use GPU
        max_chunk_size : int
            Maximum size of text chunks
        ref_s : numpy.ndarray or torch.Tensor
            Reference style for voice blending
            
        Returns:
        --------
        tuple
            Tuple of (audio_array, sample_rate)
        """
        # Preprocess text
        processed_text = text.replace('\n', ' ').strip()
        if not processed_text:
            logger.warning("Empty text after preprocessing.")
            return None, 24000
        
        # Split into chunks, preserving sentence boundaries
        chunks = self._split_into_chunks(processed_text, max_chunk_size)
        
        logger.info(f"Processing {len(chunks)} chunks...")
        all_audio = []
        sample_rate = 24000
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            logger.info(f"Chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            phoneme_sequences = list(pipeline(chunk, voice_code, speed))
            
            if not phoneme_sequences:
                logger.warning(f"No phonemes for chunk {i+1}")
                continue
            
            _, ps, _ = phoneme_sequences[0]
            
            # Use provided ref_s for blending, otherwise load from pipeline
            ref_s_chunk = ref_s if ref_s is not None else pipeline.load_voice(voice_code)[len(ps)-1]
            
            try:
                # Generate audio from phonemes
                with self.MODEL_LOCK:
                    audio = self.MODEL[use_gpu and True in self.MODEL](ps, ref_s_chunk, speed)
                
                audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                all_audio.append(audio_np)
            except Exception as e:
                logger.error(f"Chunk {i+1} failed: {e}")
                
                # Try with CPU if GPU fails
                if use_gpu:
                    try:
                        with self.MODEL_LOCK:
                            audio = self.MODEL[False](ps, ref_s_chunk, speed)
                        
                        audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                        all_audio.append(audio_np)
                    except Exception as e2:
                        logger.error(f"CPU fallback failed: {e2}")
        
        # Handle empty results
        if not all_audio:
            logger.warning("No audio generated from chunks.")
            return None, sample_rate
        
        # Combine chunks with crossfading
        if len(all_audio) == 1:
            return all_audio[0], sample_rate
        
        return self._crossfade_chunks(all_audio, sample_rate)
    
    def _split_into_chunks(self, text, max_chunk_size=400):
        """
        Split text into manageable chunks for processing.
        
        Parameters:
        -----------
        text : str
            Input text
        max_chunk_size : int
            Maximum size of chunks
            
        Returns:
        --------
        list
            List of text chunks
        """
        chunks = []
        sentences = text.split('.')
        current_chunk, current_size = [], 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence += '.'
            sentence_size = len(sentence)
            
            # Handle long sentences
            if sentence_size > max_chunk_size:
                # Split long sentences by words
                words = sentence.split()
                word_chunk, word_size = [], 0
                for word in words:
                    word_size_curr = len(word) + 1
                    if word_size + word_size_curr > max_chunk_size and word_chunk:
                        chunks.append(' '.join(word_chunk))
                        word_chunk, word_size = [], 0
                    word_chunk.append(word)
                    word_size += word_size_curr
                
                if word_chunk:
                    chunks.append(' '.join(word_chunk))
            # Add to current chunk or start new chunk
            elif current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_size = [sentence], sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        # Fallback for empty chunks
        if not chunks:
            chunks = [text]
            
        return chunks
    
    def _crossfade_chunks(self, audio_chunks, sample_rate):
        """
        Apply crossfading between audio chunks.
        
        Parameters:
        -----------
        audio_chunks : list
            List of audio arrays
        sample_rate : int
            Sample rate
            
        Returns:
        --------
        tuple
            Tuple of (combined_audio, sample_rate)
        """
        # Crossfade between chunks
        crossfade_ms = 30  # 30ms crossfade
        crossfade_samples = int(crossfade_ms / 1000 * sample_rate)
        
        # Calculate total length
        total_length = sum(len(a) for a in audio_chunks) - crossfade_samples * (len(audio_chunks) - 1)
        result = np.zeros(total_length, dtype=np.float32)
        pos = 0
        
        # Apply crossfade
        for i, chunk in enumerate(audio_chunks):
            chunk_len = len(chunk)
            
            if i == 0:
                # First chunk (no leading crossfade)
                result[:chunk_len] = chunk
                pos += chunk_len - crossfade_samples
            else:
                # Create crossfade ramps
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                # Apply crossfade
                result[pos:pos+crossfade_samples] = (
                    result[pos:pos+crossfade_samples] * fade_out +
                    chunk[:crossfade_samples] * fade_in
                )
                
                # Add remaining part of chunk
                if crossfade_samples < chunk_len:
                    result[pos+crossfade_samples:pos+chunk_len] = chunk[crossfade_samples:]
                
                pos += chunk_len - crossfade_samples
        
        return result, sample_rate
    
    def _blend_voice_styles(self, text, voice_code, second_code, pipeline, pipeline2, blend_ratio, speed, use_gpu):
        """
        Process text using blended voice styles.
        
        Parameters:
        -----------
        text : str
            Input text
        voice_code : str
            Primary voice code
        second_code : str
            Secondary voice code
        pipeline : KPipeline
            Primary pipeline
        pipeline2 : KPipeline
            Secondary pipeline
        blend_ratio : float
            Blend ratio (0-1)
        speed : float
            Playback speed
        use_gpu : bool
            Whether to use GPU
            
        Returns:
        --------
        tuple
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Process full text into chunks with blended styles
            processed_text = text.replace('\n', ' ').strip()
            phoneme_sequences = list(pipeline(processed_text, voice_code, speed))
            
            if not phoneme_sequences:
                logger.warning("No phoneme sequences generated.")
                return None, 24000
            
            # Get reference styles for both voices (using first chunk for consistency)
            _, ps, _ = phoneme_sequences[0]
            ref_s1 = pipeline.load_voice(voice_code)[len(ps)-1]
            ref_s2 = pipeline2.load_voice(second_code)[len(ps)-1]
            
            # Blend styles
            logger.info(f"Blending '{voice_code}' ({blend_ratio*100}%) and '{second_code}' ({(1-blend_ratio)*100}%)")
            
            if isinstance(ref_s1, torch.Tensor) and isinstance(ref_s2, torch.Tensor):
                blended_ref_s = ref_s1 * blend_ratio + ref_s2 * (1 - blend_ratio)
            else:
                blended_ref_s = np.add(ref_s1 * blend_ratio, ref_s2 * (1 - blend_ratio))
            
            # Process chunks with blended style
            return self._process_text_chunks(
                text, pipeline, voice_code, speed, use_gpu, ref_s=blended_ref_s
            )
            
        except Exception as e:
            logger.error(f"Voice blending error: {e}")
            return None, 24000
    
    def generate_speech(self, text, voice, speed, use_gpu, enable_blending=False, 
                       second_voice=None, blend_ratio=0.5):
        """
        Generate speech from text with optional voice blending.
        
        Parameters:
        -----------
        text : str
            Input text
        voice : str
            Voice name
        speed : float
            Playback speed
        use_gpu : bool
            Whether to use GPU
        enable_blending : bool
            Whether to enable voice blending
        second_voice : str
            Secondary voice for blending
        blend_ratio : float
            Blend ratio (0-1)
            
        Returns:
        --------
        tuple
            Tuple of (audio_dict, processed_text)
        """
        # Initialize if not already done
        if not self.VOICES:
            self._initialize()
        
        # Load GPU model if needed and available
        if use_gpu and True not in self.MODEL and torch.cuda.is_available():
            try:
                with self.MODEL_LOCK:
                    self.MODEL[True] = KModel().to('cuda').eval()
            except Exception as e:
                logger.error(f"GPU load failed: {e}. Using CPU.")
                use_gpu = False
        
        # Get voice code and pipeline
        voice_code = self.VOICES[voice]
        pipeline = self.PIPELINES[voice_code[0]]
        start_time = time.time()
        
        # Handle voice blending
        if enable_blending and second_voice and second_voice != voice:
            second_code = self.VOICES[second_voice]
            pipeline2 = self.PIPELINES[second_code[0]]
            
            # Process with blended styles
            final_audio, sample_rate = self._blend_voice_styles(
                text, voice_code, second_code, pipeline, pipeline2, blend_ratio, speed, use_gpu
            )
            
            # Fallback to primary voice if blending fails
            if final_audio is None:
                logger.warning("Blended audio generation failed, falling back to primary voice.")
                final_audio, sample_rate = self._process_text_chunks(text, pipeline, voice_code, speed, use_gpu)
                final_audio = final_audio if final_audio is not None else np.zeros(1000, dtype=np.float32)
        else:
            # Single voice, no blending
            final_audio, sample_rate = self._process_text_chunks(text, pipeline, voice_code, speed, use_gpu)
            final_audio = final_audio if final_audio is not None else np.zeros(1000, dtype=np.float32)
        
        # Convert to tensor format
        waveform = torch.tensor(final_audio, dtype=torch.float32).unsqueeze(0)
        
        # Ensure minimum length
        if waveform.shape[-1] < 1000:
            waveform = torch.nn.functional.pad(waveform, (0, 1000 - waveform.shape[-1]))
        
        logger.info(f"Speech generation completed in {time.time() - start_time:.2f} seconds")
        
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}, text


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeekyKokoroTTS": GeekyKokoroTTSNode
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyKokoroTTS": "🔊 Geeky Kokoro TTS"
}