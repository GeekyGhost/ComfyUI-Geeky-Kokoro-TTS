"""
Main module for Geeky Kokoro TTS implementation in ComfyUI.
Provides text-to-speech functionality with voice customization.
Updated for ComfyUI v3.49+ and Python 3.12 compatibility.
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
    ComfyUI node for Geeky Kokoro TTS with improved text processing and modern compatibility.
    Updated for Kokoro v0.19+ and ComfyUI v3.49+.
    """
    
    # Class variables for model management
    MODEL = None
    PIPELINES = {}
    VOICES = {}
    MODEL_LOCK = threading.Lock()
    INITIALIZED = False
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node."""
        if not cls.INITIALIZED:
            cls._initialize()
            
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "Welcome to the updated Geeky Kokoro TTS for ComfyUI. This version features improved text chunking and the latest Kokoro models."
                }),
                "voice": (list(cls.VOICES.keys()), {"default": "🇺🇸 🚺 Heart ❤️"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "use_gpu": ("BOOLEAN", {"default": torch.cuda.is_available()}),
            },
            "optional": {
                "enable_blending": ("BOOLEAN", {"default": False}),
                "second_voice": (list(cls.VOICES.keys()) if cls.VOICES else ["🇺🇸 🚺 Sarah"], {"default": "🇺🇸 🚺 Sarah"}),
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
        # Current ComfyUI model directory structure
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
        Initialize the TTS models and voices with modern Kokoro support.
        """
        if cls.INITIALIZED:
            return
            
        logger.info("Initializing Geeky Kokoro TTS with updated models...")
        
        # Updated voice list for Kokoro v0.19+
        cls.VOICES = {
            # US English Voices
            '🇺🇸 🚺 Heart ❤️': 'af_heart', '🇺🇸 🚺 Bella 🔥': 'af_bella',
            '🇺🇸 🚺 Nicole 🎧': 'af_nicole', '🇺🇸 🚺 Aoede': 'af_aoede',
            '🇺🇸 🚺 Kore': 'af_kore', '🇺🇸 🚺 Sarah': 'af_sarah',
            '🇺🇸 🚺 Nova': 'af_nova', '🇺🇸 🚺 Sky': 'af_sky',
            '🇺🇸 🚺 Alloy': 'af_alloy', '🇺🇸 🚺 Jessica': 'af_jessica',
            '🇺🇸 🚺 River': 'af_river', '🇺🇸 🚹 Michael': 'am_michael',
            '🇺🇸 🚹 Fenrir': 'am_fenrir', '🇺🇸 🚹 Puck': 'am_puck',
            '🇺🇸 🚹 Echo': 'am_echo', '🇺🇸 🚹 Eric': 'am_eric',
            '🇺🇸 🚹 Liam': 'am_liam', '🇺🇸 🚹 Onyx': 'am_onyx',
            '🇺🇸 🚹 Adam': 'am_adam',
            # UK English Voices  
            '🇬🇧 🚺 Emma': 'bf_emma', '🇬🇧 🚺 Isabella': 'bf_isabella',
            '🇬🇧 🚺 Alice': 'bf_alice', '🇬🇧 🚺 Lily': 'bf_lily',
            '🇬🇧 🚹 George': 'bm_george', '🇬🇧 🚹 Fable': 'bm_fable',
            '🇬🇧 🚹 Lewis': 'bm_lewis', '🇬🇧 🚹 Daniel': 'bm_daniel',
        }
        
        if not KOKORO_AVAILABLE:
            logger.error("Kokoro package not available. TTS will not function.")
            cls.INITIALIZED = True
            return
        
        try:
            # Initialize pipelines for US (a) and UK (b) English
            # Using updated model repo for better compatibility
            model_repo = 'hexgrad/Kokoro-82M'
            
            for code in ['a', 'b']:
                with cls._suppress_warnings():
                    cls.PIPELINES[code] = KPipeline(
                        lang_code=code, 
                        model=False,
                        repo_id=model_repo
                    )
            
            # Add pronunciation rules
            cls.PIPELINES['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
            cls.PIPELINES['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
            
            # Load voice data with error handling
            for voice_name, voice_code in cls.VOICES.items():
                try:
                    cls.PIPELINES[voice_code[0]].load_voice(voice_code)
                except Exception as e:
                    logger.warning(f"Failed to load voice {voice_name} ({voice_code}): {e}")
                    # Remove failed voices from the list
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
            logger.info("Kokoro TTS initialization completed successfully.")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            cls.INITIALIZED = True  # Mark as initialized even if failed to prevent repeated attempts
    
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
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        
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
            # Improved sentence detection that handles various punctuation
            sentences = re.split(r'([.!?]+(?:\s|$))', paragraph)
            
            # Recombine split sentences with their punctuation
            combined_sentences = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    sentence = sentences[i] + (sentences[i+1] if sentences[i+1].strip() else '')
                    if sentence.strip():
                        combined_sentences.append(sentence.strip())
            
            # If no proper sentences found, fall back to the original paragraph
            if not combined_sentences:
                combined_sentences = [paragraph]
            
            # Group sentences into chunks
            current_chunk = ""
            for sentence in combined_sentences:
                # Check if adding this sentence would exceed chunk size
                if current_chunk and len(current_chunk + " " + sentence) > max_chunk_size:
                    # Add current chunk if it has content
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                
                # Handle very long sentences
                if len(current_chunk) > max_chunk_size * 1.5:
                    # Split by clauses (commas, semicolons)
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
            
            # Add any remaining content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        # Final cleanup - remove empty chunks and ensure we have content
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        if not chunks:
            chunks = [text[:max_chunk_size]]  # Fallback to truncated original
        
        logger.info(f"Split text into {len(chunks)} chunks. Original length: {len(text)}")
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i+1}: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
        
        return chunks
    
    def _process_text_chunks(self, text, pipeline, voice_code, speed, use_gpu, max_chunk_size=350, ref_s=None):
        """
        Process text in chunks with improved handling and seamless concatenation.
        
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
        # Use improved chunking
        chunks = self._improved_text_chunking(text, max_chunk_size)
        
        if not chunks:
            logger.warning("No valid chunks generated from text.")
            return np.zeros(1000, dtype=np.float32), 24000
        
        logger.info(f"Processing {len(chunks)} chunks...")
        all_audio = []
        sample_rate = 24000
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")
            
            try:
                # Generate phoneme sequences
                phoneme_sequences = list(pipeline(chunk, voice_code, speed))
                
                if not phoneme_sequences:
                    logger.warning(f"No phonemes generated for chunk {i+1}")
                    continue
                
                _, ps, _ = phoneme_sequences[0]
                
                # Use provided ref_s for blending, otherwise load from pipeline
                ref_s_chunk = ref_s if ref_s is not None else pipeline.load_voice(voice_code)[len(ps)-1]
                
                # Generate audio from phonemes
                with self.MODEL_LOCK:
                    # Load GPU model if requested and not already loaded
                    if use_gpu and True not in self.MODEL and torch.cuda.is_available():
                        try:
                            self.MODEL[True] = KModel().to('cuda').eval()
                        except Exception as gpu_e:
                            logger.warning(f"Failed to load GPU model: {gpu_e}. Using CPU.")
                            use_gpu = False
                    
                    # Generate audio
                    try:
                        audio = self.MODEL[use_gpu and True in self.MODEL](ps, ref_s_chunk, speed)
                    except Exception as gen_e:
                        logger.warning(f"GPU generation failed for chunk {i+1}: {gen_e}. Trying CPU.")
                        use_gpu = False
                        audio = self.MODEL[False](ps, ref_s_chunk, speed)
                
                # Convert to numpy if needed
                audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                all_audio.append(audio_np)
                
            except Exception as e:
                logger.error(f"Failed to process chunk {i+1}: {e}")
                # Add silence to maintain timing
                silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)  # 0.5 second silence
                all_audio.append(silence)
        
        # Handle empty results
        if not all_audio:
            logger.warning("No audio generated from any chunks.")
            return np.zeros(1000, dtype=np.float32), sample_rate
        
        # Combine chunks with improved concatenation
        if len(all_audio) == 1:
            return all_audio[0], sample_rate
        
        return self._seamless_concatenate(all_audio, sample_rate)
    
    def _seamless_concatenate(self, audio_chunks, sample_rate):
        """
        Concatenate audio chunks with smooth transitions and gap handling.
        
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
        if not audio_chunks:
            return np.zeros(1000, dtype=np.float32), sample_rate
            
        if len(audio_chunks) == 1:
            return audio_chunks[0], sample_rate
        
        # Add small gaps between chunks for natural speech flow
        gap_duration = 0.15  # 150ms gap
        gap_samples = int(gap_duration * sample_rate)
        gap = np.zeros(gap_samples, dtype=np.float32)
        
        # Concatenate with gaps
        result_chunks = []
        for i, chunk in enumerate(audio_chunks):
            result_chunks.append(chunk)
            # Add gap between chunks (but not after the last one)
            if i < len(audio_chunks) - 1:
                result_chunks.append(gap)
        
        # Final concatenation
        result = np.concatenate(result_chunks, axis=0)
        
        logger.info(f"Combined {len(audio_chunks)} chunks into {len(result)} samples ({len(result)/sample_rate:.2f} seconds)")
        return result, sample_rate
    
    def _blend_voice_styles(self, text, voice_code, second_code, pipeline, pipeline2, blend_ratio, speed, use_gpu):
        """
        Process text using blended voice styles with improved handling.
        
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
            Blend ratio (0-1, where 1.0 = 100% primary voice)
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
            logger.info(f"Blending voices: {voice_code} ({blend_ratio*100:.1f}%) + {second_code} ({(1-blend_ratio)*100:.1f}%)")
            
            # Get a sample chunk to extract reference styles
            chunks = self._improved_text_chunking(text, 200)  # Smaller chunks for style extraction
            sample_text = chunks[0] if chunks else text[:200]
            
            # Process sample text to get phoneme sequences
            phoneme_sequences = list(pipeline(sample_text, voice_code, speed))
            
            if not phoneme_sequences:
                logger.warning("No phoneme sequences for blending. Falling back to primary voice.")
                return self._process_text_chunks(text, pipeline, voice_code, speed, use_gpu)
            
            # Get reference styles for both voices
            _, ps, _ = phoneme_sequences[0]
            ref_s1 = pipeline.load_voice(voice_code)[len(ps)-1]
            ref_s2 = pipeline2.load_voice(second_code)[len(ps)-1]
            
            # Blend the reference styles
            if isinstance(ref_s1, torch.Tensor) and isinstance(ref_s2, torch.Tensor):
                blended_ref_s = ref_s1 * blend_ratio + ref_s2 * (1 - blend_ratio)
            else:
                # Convert to numpy if needed and blend
                ref_s1_np = ref_s1.cpu().numpy() if isinstance(ref_s1, torch.Tensor) else ref_s1
                ref_s2_np = ref_s2.cpu().numpy() if isinstance(ref_s2, torch.Tensor) else ref_s2
                blended_ref_s = ref_s1_np * blend_ratio + ref_s2_np * (1 - blend_ratio)
            
            # Process with blended style
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
        if not self.INITIALIZED:
            self._initialize()
        
        if not KOKORO_AVAILABLE or not self.VOICES:
            logger.error("Kokoro TTS not available or no voices loaded")
            # Return silent audio as fallback
            silent_audio = torch.zeros((1, 1, 1000), dtype=torch.float32)
            return {"waveform": silent_audio, "sample_rate": 24000}, text
        
        # Validate and clean input text
        if not text or not text.strip():
            logger.warning("Empty or invalid text input")
            text = "Please provide valid text for speech synthesis."
        
        processed_text = text.strip()
        
        # Get voice code and pipeline
        voice_code = self.VOICES.get(voice)
        if not voice_code:
            logger.error(f"Voice {voice} not found. Using default.")
            voice_code = 'af_heart'
            
        pipeline = self.PIPELINES.get(voice_code[0])
        if not pipeline:
            logger.error(f"Pipeline for {voice_code[0]} not found")
            silent_audio = torch.zeros((1, 1, 1000), dtype=torch.float32)
            return {"waveform": silent_audio, "sample_rate": 24000}, processed_text
        
        start_time = time.time()
        
        try:
            # Handle voice blending
            if enable_blending and second_voice and second_voice != voice and second_voice in self.VOICES:
                second_code = self.VOICES[second_voice]
                pipeline2 = self.PIPELINES.get(second_code[0])
                
                if pipeline2:
                    # Process with blended styles
                    final_audio, sample_rate = self._blend_voice_styles(
                        processed_text, voice_code, second_code, pipeline, pipeline2, blend_ratio, speed, use_gpu
                    )
                else:
                    logger.warning("Second voice pipeline not available. Using primary voice only.")
                    final_audio, sample_rate = self._process_text_chunks(processed_text, pipeline, voice_code, speed, use_gpu)
            else:
                # Single voice processing
                final_audio, sample_rate = self._process_text_chunks(processed_text, pipeline, voice_code, speed, use_gpu)
            
            # Validate audio output
            if final_audio is None or len(final_audio) == 0:
                logger.warning("No audio generated. Creating fallback audio.")
                final_audio = np.zeros(1000, dtype=np.float32)
            
            # Convert to tensor format expected by ComfyUI
            waveform = torch.tensor(final_audio, dtype=torch.float32)
            
            # Ensure correct shape: (batch, channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            elif waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # (1, channels, samples)
            
            # Ensure minimum length for ComfyUI compatibility
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
            
            # Return fallback audio
            fallback_audio = torch.zeros((1, 1, 1000), dtype=torch.float32)
            return {"waveform": fallback_audio, "sample_rate": 24000}, processed_text


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeekyKokoroTTS": GeekyKokoroTTSNode
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyKokoroTTS": "🔊 Geeky Kokoro TTS (Updated)"
}