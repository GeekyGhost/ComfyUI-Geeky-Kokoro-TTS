"""
GeekyWhisperSpeech node for ComfyUI.
Provides reliable voice cloning with optimized performance and improved chunking.
Incorporates advanced techniques from the original WhisperSpeech implementation.
"""
import os
import torch
import numpy as np
import torchaudio
import tempfile
import shutil
import time
import threading
import re
import math
import logging
from os.path import expanduser
from pathlib import Path
from contextlib import contextmanager, nullcontext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GeekyWhisperSpeech")

# Add WhisperSpeech integration
try:
    from whisperspeech.pipeline import Pipeline
    WHISPER_AVAILABLE = True
    logger.info("WhisperSpeech imported successfully")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("WhisperSpeech not available. Will use fallback.")

# Disable Torch Dynamo to avoid Triton-related errors
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass


class GeekyWhisperSpeechNode:
    """Optimized WhisperSpeech node with reliable performance and improved chunking."""
    
    # Track WhisperSpeech pipeline for reuse
    WHISPER_PIPELINE = None
    EMBEDDING_CACHE = {}  # Cache for speaker embeddings
    PIPELINE_LOCK = threading.Lock()  # Lock for thread-safe pipeline access
    DEFAULT_SPEAKER = None  # Default speaker embedding for fallback
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "Welcome to Geeky WhisperSpeech for ComfyUI."
                }),
                "user_audio": ("AUDIO",),
                "enable_blending": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "second_voice": ("AUDIO",),
                "blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "output_volume": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0, "display": "slider"}),
                "use_gpu": ("BOOLEAN", {"default": True}),
                "cache_embeddings": ("BOOLEAN", {"default": True}),
                "smart_chunking": ("BOOLEAN", {"default": True}),
                "characters_per_second": ("FLOAT", {"default": 13.0, "min": 5.0, "max": 25.0, "step": 0.5}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
                "max_chunk_size": ("INT", {"default": 350, "min": 100, "max": 800, "step": 50})
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "text_processed",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio"
    
    @staticmethod
    def get_compute_device():
        """Intelligently select the best available compute device.
        Directly adapted from WhisperSpeech's inference.py
        """
        if torch.cuda.is_available() and (torch.version.cuda or torch.version.hip):
            return 'cuda'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    @staticmethod
    def inference_context():
        """Create appropriate context for inference based on available hardware.
        Adapted from WhisperSpeech's inference.py
        """
        if torch.cuda.is_available():
            return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        else:
            return nullcontext()
    
    def get_whisper_pipeline(self, use_gpu=True, optimize=True, torch_compile=False):
        """Get or initialize the WhisperSpeech pipeline with thread safety and optimizations."""
        if not WHISPER_AVAILABLE:
            return None
        
        with self.PIPELINE_LOCK:    
            if self.__class__.WHISPER_PIPELINE is None:
                try:
                    logger.info("Initializing WhisperSpeech pipeline...")
                    device = self.get_compute_device() if use_gpu else "cpu"
                    
                    # Initialize with optimization flags
                    self.__class__.WHISPER_PIPELINE = Pipeline(
                        torch_compile=torch_compile,
                        device=device,
                        optimize=optimize
                    )
                    
                    # Load default speaker if needed
                    if self.__class__.DEFAULT_SPEAKER is None:
                        self.__class__.DEFAULT_SPEAKER = self.__class__.WHISPER_PIPELINE.default_speaker
                    
                    logger.info(f"WhisperSpeech pipeline initialized successfully on {device}")
                except Exception as e:
                    logger.error(f"Error initializing WhisperSpeech pipeline: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
        
        return self.__class__.WHISPER_PIPELINE
    
    def _generate_cache_key(self, audio_dict):
        """Generate a unique cache key for an audio input."""
        if not isinstance(audio_dict, dict) or "waveform" not in audio_dict:
            return None
            
        waveform = audio_dict["waveform"]
        # Create a fingerprint based on the shape and simple statistics
        if isinstance(waveform, torch.Tensor):
            shape_str = "_".join(str(x) for x in waveform.shape)
            try:
                # Calculate basic statistics as a fingerprint
                if waveform.numel() > 0:
                    # Use just a few key statistics to identify the audio
                    min_val = float(torch.min(waveform))
                    max_val = float(torch.max(waveform))
                    mean_val = float(torch.mean(waveform))
                    std_val = float(torch.std(waveform))
                    # Add more stats for better uniqueness
                    rms_val = float(torch.sqrt(torch.mean(waveform**2)))
                    return f"audio_{shape_str}_{min_val:.4f}_{max_val:.4f}_{mean_val:.4f}_{std_val:.4f}_{rms_val:.4f}"
                else:
                    return f"audio_empty_{shape_str}"
            except:
                return f"audio_{shape_str}_stats_error"
        return None
    
    def extract_speaker_embedding(self, audio_dict, use_cache=True):
        """
        Extract voice embedding from audio with enhanced caching for performance.
        Uses the exact technique from the WhisperSpeech pipeline implementation.
        """
        pipe = self.get_whisper_pipeline()
        if pipe is None:
            return self.__class__.DEFAULT_SPEAKER if self.__class__.DEFAULT_SPEAKER is not None else None
            
        if audio_dict is None or not isinstance(audio_dict, dict):
            logger.warning("Invalid custom voice audio input, using default speaker")
            return self.__class__.DEFAULT_SPEAKER if self.__class__.DEFAULT_SPEAKER is not None else None
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._generate_cache_key(audio_dict)
            if cache_key and cache_key in self.__class__.EMBEDDING_CACHE:
                logger.info("Using cached voice embedding")
                return self.__class__.EMBEDDING_CACHE[cache_key]
            
        try:
            # Extract waveform and sample rate
            waveform = audio_dict.get("waveform", None)
            sample_rate = audio_dict.get("sample_rate", 24000)
            
            if waveform is None:
                logger.error("Audio has no waveform, using default speaker")
                return self.__class__.DEFAULT_SPEAKER if self.__class__.DEFAULT_SPEAKER is not None else None
                
            # Ensure proper shape for torchaudio.save
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Use a temp file that gets automatically cleaned up
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_path = temp_file.name
                
            try:
                # Save to temporary file
                torchaudio.save(audio_path, waveform.cpu(), sample_rate)
                
                # Extract speaker embedding using the pipeline's method
                start_time = time.time()
                
                # Get audio info for proper duration limiting
                audio_info = torchaudio.info(audio_path)
                actual_sample_rate = audio_info.sample_rate
                
                # Limit to first 30 seconds for embedding extraction (exactly as in pipeline.py)
                num_frames = actual_sample_rate * 30
                
                # Use SpeechBrain for consistent embedding extraction
                if not hasattr(pipe, 'encoder') or pipe.encoder is None:
                    device = pipe.device
                    # Handle MPS incompatibility as in the original code
                    if device == 'mps': device = 'cpu'
                    from speechbrain.pretrained import EncoderClassifier
                    pipe.encoder = EncoderClassifier.from_hparams(
                        "speechbrain/spkrec-ecapa-voxceleb",
                        savedir=expanduser("~/.cache/speechbrain/"),
                        run_opts={"device": device}
                    )
                
                # Load audio with proper frame limiting
                samples, sr = torchaudio.load(audio_path, num_frames=num_frames)
                samples = samples[:, :num_frames]
                samples = pipe.encoder.audio_normalizer(samples[0], sr)
                speaker_emb = pipe.encoder.encode_batch(samples.unsqueeze(0))
                speaker_emb = speaker_emb[0,0].to(pipe.device)
                
                logger.info(f"Voice embedding extracted in {time.time() - start_time:.2f} seconds")
                
                # Cache the result if enabled
                if use_cache and cache_key:
                    self.__class__.EMBEDDING_CACHE[cache_key] = speaker_emb
                    logger.info("Cached voice embedding for future use")
                
                return speaker_emb
            finally:
                # Clean up temp file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error extracting custom voice: {e}")
            import traceback
            traceback.print_exc()
            return self.__class__.DEFAULT_SPEAKER if self.__class__.DEFAULT_SPEAKER is not None else None

    def preprocess_text(self, text):
        """
        Preprocess text to handle common TTS issues like contractions and abbreviations.
        This improves pronunciation quality in the generated speech.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Common contractions that WhisperSpeech sometimes struggles with
        contractions = {
            r"won't": "will not",
            r"can't": "cannot",
            r"don't": "do not",
            r"doesn't": "does not",
            r"didn't": "did not",
            r"isn't": "is not",
            r"aren't": "are not",
            r"wasn't": "was not",
            r"weren't": "were not",
            r"haven't": "have not",
            r"hasn't": "has not",
            r"hadn't": "had not",
            r"it's": "it is",
            r"we've": "we have",
            r"they've": "they have",
            r"who've": "who have",
            r"shouldn't": "should not",
            r"wouldn't": "would not",
            r"couldn't": "could not",
            r"they're": "they are",
            r"we're": "we are",
            r"you're": "you are",
        }
        
        # Apply only the problematic contractions that cause issues
        for contraction, expansion in contractions.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        
        # Expand certain abbreviations and symbols
        text = re.sub(r'\bDr\.', 'Doctor', text)
        text = re.sub(r'\bMr\.', 'Mister', text)
        text = re.sub(r'\bMrs\.', 'Misses', text)
        text = re.sub(r'\bMs\.', 'Miss', text)
        text = re.sub(r'\bSt\.', 'Saint', text)
        text = re.sub(r'\bCo\.', 'Company', text)
        text = re.sub(r'\bJr\.', 'Junior', text)
        text = re.sub(r'\bSr\.', 'Senior', text)
        text = re.sub(r'\bvs\.', 'versus', text)
        text = re.sub(r'&', 'and', text)
        
        # Add slight pause after period/comma if not followed by space
        text = re.sub(r'\.([A-Za-z])', '. \1', text)
        text = re.sub(r',([A-Za-z])', ', \1', text)
        
        # Ensure proper spacing for punctuation 
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        return text
    
    def smart_split_text(self, text, max_length=350, overlap_chars=50):
        """
        Advanced text chunking that preserves natural language boundaries with context overlap.
        Now returns chunk metadata to prevent repetition of overlapping sections.
        """
        if len(text) <= max_length:
            return [{"text": text, "render_start": 0, "render_end": len(text)}]
        
        # First, normalize the text for better processing
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Define sentence boundary pattern - handle more edge cases
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        # If a sentence is very long, split it at punctuation or conjunctions
        processed_sentences = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            
            # Add period if missing sentence terminator
            if not s.endswith(('.', '!', '?')):
                s += '.'
                
            # Split very long sentences
            if len(s) > max_length * 0.8:
                # Try to split at commas, semicolons, or conjunctions
                subparts = re.split(r'(?<=[,;])\s+|(?<=\s(and|but|or|because|however|moreover))\s+', s)
                for part in subparts:
                    if part.strip():
                        processed_sentences.append(part.strip())
            else:
                processed_sentences.append(s)
        
        # Create chunks with context overlap, but track what should be rendered
        chunks = []
        current_chunk = ""
        current_length = 0
        absolute_position = 0  # Track position in original text
        chunk_start_position = 0  # Position where current chunk starts in original text
        
        for sentence in processed_sentences:
            sentence_len = len(sentence) + 1  # +1 for space
            
            # If adding this sentence would exceed max length and we already have content
            if current_length + sentence_len > max_length and current_chunk:
                # Add the current chunk to the list with absolute positions
                chunks.append({
                    "text": current_chunk.strip(),
                    "absolute_start": chunk_start_position,
                    "absolute_end": absolute_position - 1  # -1 for trailing space
                })
                
                # Calculate overlap for next chunk
                overlap_start = max(0, absolute_position - overlap_chars)
                
                # Extract overlap text from original text
                overlap_text = text[overlap_start:absolute_position]
                
                # Start a new chunk with the overlap
                current_chunk = overlap_text + " " + sentence + " "
                current_length = len(overlap_text) + sentence_len + 1
                
                # Update chunk start position, accounting for overlap
                chunk_start_position = overlap_start
            else:
                # Add to current chunk
                current_chunk += sentence + " "
                current_length += sentence_len
                
                # Set chunk_start_position only for the first sentence
                if current_length == sentence_len:
                    chunk_start_position = absolute_position
            
            # Update absolute position after adding the sentence
            absolute_position += sentence_len
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "absolute_start": chunk_start_position,
                "absolute_end": absolute_position - 1  # -1 for trailing space
            })
        
        # Now add render boundaries to avoid repetition
        chunks_with_boundaries = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk renders everything
                render_start = 0
            else:
                # Calculate where non-overlapping content starts
                # This is the key to avoiding repetition
                prev_end = chunks[i-1]["absolute_end"]
                current_start = chunk["absolute_start"]
                overlap_length = prev_end - current_start + 1
                if overlap_length > 0:
                    # Skip the overlapping text when rendering
                    render_start = overlap_length
                else:
                    # No overlap (gap in text - shouldn't happen with our algorithm)
                    render_start = 0
            
            # Add chunk with render boundaries
            chunks_with_boundaries.append({
                "text": chunk["text"],
                "render_start": render_start,
                "render_end": len(chunk["text"])
            })
        
        logger.info(f"Split text into {len(chunks_with_boundaries)} chunks with non-repeating boundaries")
        return chunks_with_boundaries
    
    def generate_speech_for_chunk(self, chunk_with_boundaries, voice_emb, pipe, device, cps=13.0, temperature=0.7, top_k=None):
        """
        Generate speech for a single text chunk with context, but only return the renderable portion.
        This prevents repeating overlapping text between chunks.
        """
        try:
            # Extract chunk text and render boundaries
            chunk_text = chunk_with_boundaries["text"]
            render_start = chunk_with_boundaries["render_start"]
            render_end = chunk_with_boundaries["render_end"]
            
            # Generate audio for the full chunk (including context)
            with self.inference_context():
                # Generate speech tokens with specified character per second rate
                speech_tokens = pipe.t2s.generate(chunk_text, cps=cps, show_progress_bar=False, T=temperature, top_k=top_k)[0]
                speech_tokens = speech_tokens[speech_tokens != 512]  # Remove padding
                
                # Try to move to device if possible
                if device == 'cuda' and torch.cuda.is_available():
                    try:
                        speech_tokens = speech_tokens.to(device)
                        voice_emb_gpu = voice_emb.to(device)
                    except Exception as e:
                        logger.warning(f"Unable to move to GPU: {e}")
                        voice_emb_gpu = voice_emb
                else:
                    voice_emb_gpu = voice_emb
                    
                # Generate audio tokens
                audio_tokens = pipe.s2a.generate(
                    speech_tokens,
                    speakers=voice_emb_gpu.unsqueeze(0),
                    show_progress_bar=False
                )
                
                # Decode to audio
                full_audio_wave = pipe.vocoder.decode(audio_tokens)
                
                # Ensure proper shape
                full_audio_wave = full_audio_wave.squeeze()  # Remove any singleton dimensions
                
                # Calculate sample positions based on character positions
                # This is an approximation - characters don't map linearly to audio samples
                # but it's close enough for our purposes
                total_chars = render_end - 0  # Full length of text
                total_samples = full_audio_wave.shape[-1]
                samples_per_char = total_samples / total_chars if total_chars > 0 else 0
                
                # Calculate sample positions for render boundaries
                render_start_sample = int(render_start * samples_per_char)
                render_end_sample = total_samples
                
                # Trim audio to only the renderable portion
                if render_start_sample >= total_samples:
                    logger.warning(f"Render start ({render_start_sample}) exceeds audio length ({total_samples}), returning full audio")
                    trimmed_audio = full_audio_wave
                else:
                    trimmed_audio = full_audio_wave[render_start_sample:render_end_sample]
                
                if render_start > 0:
                    logger.info(f"Trimmed {render_start} chars ({render_start_sample} samples) from chunk start to avoid repetition")
                
                return trimmed_audio
            
        except Exception as e:
            logger.error(f"Error generating speech for chunk: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_speech(self, text, user_audio, enable_blending=False, 
                      second_voice=None, blend_ratio=0.5, output_volume=0.0,
                      use_gpu=True, cache_embeddings=True, smart_chunking=True,
                      characters_per_second=13.0, temperature=0.7, max_chunk_size=350):
        """
        Enhanced speech generation with overlap handling to prevent repetition.
        """
        # Default output in case of errors
        default_output = {"waveform": torch.zeros((1, 1, 1000), dtype=torch.float32), "sample_rate": 24000}
        
        # Start timing for performance tracking
        total_start_time = time.time()
        
        # Get the properly configured WhisperSpeech pipeline
        pipe = self.get_whisper_pipeline(use_gpu)
        if pipe is None:
            logger.error("WhisperSpeech not available, returning default output")
            return default_output, text
        
        try:
            # Store original text for return value
            original_text = text
            
            # Preprocess text to improve pronunciation
            text = self.preprocess_text(text)
            
            # Extract voice from user audio (with caching if enabled)
            embed_start = time.time()
            user_voice_emb = self.extract_speaker_embedding(user_audio, use_cache=cache_embeddings)
            if user_voice_emb is None:
                logger.error("Failed to extract voice from user audio and no default speaker available")
                return default_output, text
            
            # Handle voice blending if enabled
            if enable_blending and second_voice is not None:
                second_voice_emb = self.extract_speaker_embedding(second_voice, use_cache=cache_embeddings)
                if second_voice_emb is not None:
                    logger.info(f"Blending voices with ratio {blend_ratio:.2f}")
                    # Make sure they're on the same device
                    device = user_voice_emb.device
                    second_voice_emb = second_voice_emb.to(device)
                    
                    # Linear blending of embeddings
                    voice_emb = user_voice_emb * blend_ratio + second_voice_emb * (1 - blend_ratio)
                else:
                    logger.warning("Failed to extract voice from second audio, using only user voice")
                    voice_emb = user_voice_emb
            else:
                voice_emb = user_voice_emb
            
            logger.info(f"Embedding preparation took {time.time() - embed_start:.2f} seconds")
            
            # Determine device for generation
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device} for generation")
            
            # Generate speech with improved chunking strategy
            gen_start = time.time()
            
            # Determine if we should use chunking based on text length
            use_chunking = smart_chunking and len(text) > max_chunk_size
            
            # Top-k parameter for sampling (prevents low-probability tokens)
            top_k = 50 if temperature > 0.5 else None
            
            if use_chunking:
                logger.info(f"Processing long text ({len(text)} chars) using smart chunking")
                
                # Split text into chunks with overlap boundaries
                chunks_with_boundaries = self.smart_split_text(text, max_length=max_chunk_size)
                
                # Process each chunk with consistent parameters
                audio_segments = []
                
                for i, chunk in enumerate(chunks_with_boundaries):
                    chunk_text_preview = chunk["text"][chunk["render_start"]:chunk["render_end"]][:30] + "..."
                    logger.info(f"Processing chunk {i+1}/{len(chunks_with_boundaries)}: {chunk_text_preview}")
                    chunk_start = time.time()
                    
                    # Generate speech for this chunk with boundaries to prevent repetition
                    audio_wave = self.generate_speech_for_chunk(
                        chunk, voice_emb, pipe, device, 
                        cps=characters_per_second,
                        temperature=temperature,
                        top_k=top_k
                    )
                    
                    if audio_wave is not None and audio_wave.numel() > 0:
                        audio_segments.append(audio_wave)
                        logger.info(f"Chunk {i+1} processed in {time.time() - chunk_start:.2f} seconds")
                    else:
                        logger.warning(f"Failed to process chunk {i+1} or empty audio returned, skipping")
                
                # Combine segments with simple concatenation (no crossfade needed for non-overlapping segments)
                if audio_segments:
                    # Simple concatenation is enough since we've already handled overlap
                    audio_wave = torch.cat(audio_segments, dim=0)
                else:
                    logger.error("No valid audio segments generated")
                    audio_wave = torch.zeros(24000, dtype=torch.float32, device=device)  # 1 second of silence
            else:
                logger.info(f"Processing text as a single unit (length: {len(text)} chars)")
                
                # Create a single chunk with full render boundaries
                single_chunk = {"text": text, "render_start": 0, "render_end": len(text)}
                
                # Generate in a single pass with specified parameters
                audio_wave = self.generate_speech_for_chunk(
                    single_chunk, voice_emb, pipe, device, 
                    cps=characters_per_second,
                    temperature=temperature,
                    top_k=top_k
                )
                
                if audio_wave is None:
                    logger.error("Failed to generate speech")
                    audio_wave = torch.zeros(24000, dtype=torch.float32, device=device)  # 1 second of silence
            
            logger.info(f"Speech generation took {time.time() - gen_start:.2f} seconds")
            
            # Apply volume adjustment if needed
            if output_volume != 0.0:
                # Convert dB to linear gain
                gain = 10.0 ** (output_volume / 20.0)
                audio_wave = audio_wave * gain
                
                # Apply soft clipping if needed to prevent distortion
                if torch.max(torch.abs(audio_wave)) > 0.95:
                    audio_wave = torch.tanh(audio_wave * 0.6) / 0.6
                    logger.info("Applied soft clipping to prevent distortion")
            
            # Apply final normalization to ensure good volume level
            max_amp = torch.max(torch.abs(audio_wave))
            if max_amp > 0:
                # Target -1 dB peak
                target_peak = 0.89
                if max_amp > target_peak:
                    audio_wave = audio_wave * (target_peak / max_amp)
            
            # Convert to format expected by ComfyUI - 3D tensor [1, 1, samples]
            if audio_wave.dim() == 0:  # Scalar tensor
                audio_wave = torch.zeros(24000, dtype=torch.float32, device=audio_wave.device)  # 1 second of silence
                
            if audio_wave.dim() == 1:
                audio_wave = audio_wave.unsqueeze(0).unsqueeze(0)
            elif audio_wave.dim() == 2:
                audio_wave = audio_wave.unsqueeze(0)
                
            # Move back to CPU if necessary
            if audio_wave.device.type == 'cuda':
                audio_wave = audio_wave.cpu()
                
            # Create output dictionary
            output_audio = {
                "waveform": audio_wave,
                "sample_rate": 24000  # WhisperSpeech uses 24kHz
            }
            
            logger.info(f"Total processing completed in {time.time() - total_start_time:.2f} seconds")
            return output_audio, original_text
            
        except Exception as e:
            logger.error(f"Error in speech generation: {e}")
            import traceback
            traceback.print_exc()
            return default_output, text

    def logits_to_probs(self, logits, temperature=1.0, top_k=None):
        """
        Convert logits to probabilities with temperature scaling and top-k filtering.
        Directly adapted from WhisperSpeech's inference.py for consistent token generation.
        """
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, torch.tensor(-float("Inf"), device=logits.device), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def multinomial_sample_one_no_sync(self, probs_sort):
        """
        Sample from multinomial distribution without CUDA synchronization.
        Directly adapted from WhisperSpeech's inference.py.
        """
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    def sample(self, logits, temperature=1.0, top_k=None):
        """
        Sample from logits with temperature and top-k filtering.
        Directly adapted from WhisperSpeech's inference.py.
        """
        probs = self.logits_to_probs(logits, temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeekyWhisperSpeech": GeekyWhisperSpeechNode
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyWhisperSpeech": "🔊 Geeky WhisperSpeech"
}