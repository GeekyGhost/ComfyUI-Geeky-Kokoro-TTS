"""
GeekyWhisperSpeech node for ComfyUI.
Provides reliable voice cloning with optimized performance and improved chunking.
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
from pathlib import Path

# Add WhisperSpeech integration
try:
    from whisperspeech.pipeline import Pipeline
    WHISPER_AVAILABLE = True
    print("WhisperSpeech imported successfully")
except ImportError:
    WHISPER_AVAILABLE = False
    print("WhisperSpeech not available. Will use fallback.")

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
                "use_gpu": ("BOOLEAN", {"default": torch.cuda.is_available()}),
                "cache_embeddings": ("BOOLEAN", {"default": True}),
                "smart_chunking": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "text_processed",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio"
    
    def get_whisper_pipeline(self, use_gpu=False):
        """Get or initialize the WhisperSpeech pipeline with thread safety."""
        if not WHISPER_AVAILABLE:
            return None
        
        with self.PIPELINE_LOCK:    
            if self.__class__.WHISPER_PIPELINE is None:
                try:
                    print("Initializing WhisperSpeech pipeline...")
                    # Initialize without torch_compile to avoid Triton errors
                    self.__class__.WHISPER_PIPELINE = Pipeline(torch_compile=False)
                    print("WhisperSpeech pipeline initialized successfully")
                except Exception as e:
                    print(f"Error initializing WhisperSpeech pipeline: {e}")
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
                    return f"audio_{shape_str}_{min_val:.4f}_{max_val:.4f}_{mean_val:.4f}"
                else:
                    return f"audio_empty_{shape_str}"
            except:
                return f"audio_{shape_str}_stats_error"
        return None
    
    def extract_custom_voice_from_audio(self, audio_dict, use_cache=True):
        """Extract voice embedding from audio with caching for performance."""
        pipe = self.get_whisper_pipeline()
        if pipe is None:
            return None
            
        if audio_dict is None or not isinstance(audio_dict, dict):
            print("Invalid custom voice audio input")
            return None
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._generate_cache_key(audio_dict)
            if cache_key and cache_key in self.__class__.EMBEDDING_CACHE:
                print(f"Using cached voice embedding")
                return self.__class__.EMBEDDING_CACHE[cache_key]
            
        try:
            # Extract waveform and sample rate
            waveform = audio_dict.get("waveform", None)
            sample_rate = audio_dict.get("sample_rate", 24000)
            
            if waveform is None:
                print("Custom voice audio has no waveform")
                return None
                
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
                
                # Extract speaker embedding
                start_time = time.time()
                speaker_emb = pipe.extract_spk_emb(audio_path)
                print(f"Voice embedding extracted in {time.time() - start_time:.2f} seconds")
                
                # Cache the result if enabled
                if use_cache and cache_key:
                    self.__class__.EMBEDDING_CACHE[cache_key] = speaker_emb
                    print("Cached voice embedding for future use")
                
                return speaker_emb
            finally:
                # Clean up temp file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error extracting custom voice: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def pad_short_text(self, text, min_length=10):
        """Pad very short texts to ensure they generate proper speech."""
        text = text.strip()
        if len(text) >= min_length:
            return text
            
        # For very short text, add padding to ensure proper generation
        if not text.endswith(('.', '!', '?')):
            text += '.'
            
        padding_needed = min_length - len(text)
        if padding_needed > 0:
            # Add padding that won't affect the meaning
            padding = " " + "." * (padding_needed - 1)
            print(f"Text is very short, adding minimal padding")
            return text + padding
        return text
    
    def smart_split_text(self, text, max_length=400, overlap_chars=40):
        """
        Split text into chunks with context overlap, preserving sentence structure.
        This approach ensures consistent pacing and tone between chunks.
        """
        if len(text) <= max_length:
            return [text]
        
        # First, normalize the text to simplify processing
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Use a more sophisticated approach to split by sentences
        # This regex finds sentence boundaries (., !, ? followed by a space or end of string)
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() + '. ' for s in sentences if s.strip()]
        
        # Now create chunks with context overlap
        chunks = []
        current_chunk = ""
        current_length = 0
        
        # Keep track of the last few sentences for overlap
        last_sentences = []
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed the max length and we already have content
            if current_length + sentence_len > max_length and current_chunk:
                # Add the current chunk to the list
                chunks.append(current_chunk.strip())
                
                # Create overlap from the last few sentences
                overlap_text = ""
                overlap_len = 0
                
                # Build overlap from last sentences, but don't exceed overlap_chars
                for prev_sentence in reversed(last_sentences):
                    if overlap_len + len(prev_sentence) <= overlap_chars:
                        overlap_text = prev_sentence + overlap_text
                        overlap_len += len(prev_sentence)
                    else:
                        # If we can't fit the whole sentence, take what we can
                        space_left = overlap_chars - overlap_len
                        if space_left > 20:  # Only take a partial sentence if we can get a meaningful chunk
                            words = prev_sentence.split()
                            partial = []
                            partial_len = 0
                            for word in reversed(words):
                                if partial_len + len(word) + 1 <= space_left:
                                    partial.insert(0, word)
                                    partial_len += len(word) + 1
                                else:
                                    break
                            if partial:
                                overlap_text = ' '.join(partial) + " " + overlap_text
                        break
                
                # Start a new chunk with the overlap text
                current_chunk = overlap_text + sentence
                current_length = len(current_chunk)
                
                # Reset the last sentences tracking with current sentence
                last_sentences = [sentence]
            else:
                # Add to current chunk
                current_chunk += sentence
                current_length += sentence_len
                
                # Keep track of last few sentences for potential overlap
                last_sentences.append(sentence)
                
                # Limit the number of sentences we track to avoid excessive memory use
                if len(last_sentences) > 5:
                    last_sentences.pop(0)
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Handle long single sentences that exceed the max_length
        if not chunks:
            # Break the text into chunks of max_length with overlap
            chunks = []
            for i in range(0, len(text), max_length - overlap_chars):
                if i > 0:
                    # Start overlap_chars earlier to create the overlap
                    start = max(0, i - overlap_chars)
                else:
                    start = 0
                end = min(i + max_length, len(text))
                chunks.append(text[start:end])
        
        print(f"Split text into {len(chunks)} chunks with context overlap")
        return chunks
    
    def crossfade_audio_segments(self, segments, crossfade_ms=150, sample_rate=24000):
        """
        Combine audio segments with smooth crossfading, similar to Kokoro's approach.
        Implements volume normalization and cubic easing for natural transitions.
        """
        if not segments:
            return torch.zeros(1, dtype=torch.float32)
        
        if len(segments) == 1:
            return segments[0]
        
        # Calculate crossfade samples
        crossfade_samples = int(crossfade_ms * sample_rate / 1000)
        
        # Function to normalize a segment to consistent volume
        def normalize_segment(segment):
            max_amplitude = torch.max(torch.abs(segment))
            if max_amplitude > 0:
                # Normalize to 0.9 to allow headroom for crossfading
                return segment * (0.9 / max_amplitude)
            return segment
        
        # Normalize all segments for consistent volume
        normalized_segments = [normalize_segment(seg) for seg in segments]
        
        # Calculate total output length
        total_length = sum(seg.shape[-1] for seg in normalized_segments) - crossfade_samples * (len(normalized_segments) - 1)
        
        # Create output tensor
        result = torch.zeros(total_length, dtype=normalized_segments[0].dtype, device=normalized_segments[0].device)
        position = 0
        
        # Generate cubic easing curves for crossfading (smoother than linear)
        def cubic_ease_out(t):
            return 1 - (1 - t)**3
            
        def cubic_ease_in(t):
            return t**3
        
        # Process each segment
        for i, segment in enumerate(normalized_segments):
            segment_len = segment.shape[-1]
            
            if i == 0:
                # First segment - no crossfade at the beginning
                end_pos = min(segment_len, result.shape[0])
                result[:end_pos] = segment[:end_pos]
                position = segment_len - crossfade_samples
            else:
                # Create smooth crossfade curves
                t = torch.linspace(0, 1, crossfade_samples, device=segment.device)
                fade_out = torch.tensor([1 - cubic_ease_in(x) for x in t], device=segment.device)
                fade_in = torch.tensor([cubic_ease_out(x) for x in t], device=segment.device)
                
                # Apply crossfade
                crossfade_end = min(position + crossfade_samples, result.shape[0])
                samples_to_blend = crossfade_end - position
                
                if samples_to_blend > 0 and samples_to_blend <= crossfade_samples and samples_to_blend <= segment.shape[0]:
                    result[position:crossfade_end] = (
                        result[position:crossfade_end] * fade_out[:samples_to_blend] +
                        segment[:samples_to_blend] * fade_in[:samples_to_blend]
                    )
                    
                    # Add rest of the segment after crossfade
                    remaining_start = position + samples_to_blend
                    remaining_length = segment_len - samples_to_blend
                    
                    if remaining_length > 0 and remaining_start < result.shape[0]:
                        end_pos = min(remaining_start + remaining_length, result.shape[0])
                        samples_to_copy = end_pos - remaining_start
                        result[remaining_start:end_pos] = segment[samples_to_blend:samples_to_blend+samples_to_copy]
                
                position += segment_len - crossfade_samples
        
        return result
    
    def generate_speech_for_chunk(self, text, voice_emb, pipe, device, cps=13.0):
        """Generate speech for a single text chunk with consistent parameters."""
        try:
            # Generate speech tokens
            speech_tokens = pipe.t2s.generate(text, cps=cps, show_progress_bar=False)[0]
            speech_tokens = speech_tokens[speech_tokens != 512]  # Remove padding
            
            # Try to move to device if possible
            if device == 'cuda' and torch.cuda.is_available():
                try:
                    speech_tokens = speech_tokens.to(device)
                    voice_emb_gpu = voice_emb.to(device)
                except Exception as e:
                    print(f"Warning: Unable to move to GPU: {e}")
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
            audio_wave = pipe.vocoder.decode(audio_tokens)
            
            # Ensure proper shape
            audio_wave = audio_wave.squeeze()  # Remove any singleton dimensions
            
            return audio_wave
            
        except Exception as e:
            print(f"Error generating speech for chunk: {e}")
            return None
    
    def generate_speech(self, text, user_audio, enable_blending=False, 
                      second_voice=None, blend_ratio=0.5, output_volume=0.0,
                      use_gpu=False, cache_embeddings=True, smart_chunking=True):
        """
        Generate speech using WhisperSpeech with optimized performance.
        """
        # Default output in case of errors
        default_output = {"waveform": torch.zeros((1, 1, 1000), dtype=torch.float32), "sample_rate": 24000}
        
        # Start timing
        total_start_time = time.time()
        
        # Get the WhisperSpeech pipeline
        pipe = self.get_whisper_pipeline(use_gpu)
        if pipe is None:
            print("WhisperSpeech not available, returning default output")
            return default_output, text
        
        try:
            # Pad very short texts to ensure proper generation
            original_text = text
            text = self.pad_short_text(text)
            
            # Extract voice from user audio (with caching if enabled)
            embed_start = time.time()
            user_voice_emb = self.extract_custom_voice_from_audio(user_audio, use_cache=cache_embeddings)
            if user_voice_emb is None:
                print("Failed to extract voice from user audio")
                return default_output, text
            
            # Handle voice blending if enabled
            if enable_blending and second_voice is not None:
                second_voice_emb = self.extract_custom_voice_from_audio(second_voice, use_cache=cache_embeddings)
                if second_voice_emb is not None:
                    print(f"Blending voices with ratio {blend_ratio:.2f}")
                    # Make sure they're on the same device
                    device = user_voice_emb.device
                    second_voice_emb = second_voice_emb.to(device)
                    
                    # Simple linear blending
                    voice_emb = user_voice_emb * blend_ratio + second_voice_emb * (1 - blend_ratio)
                else:
                    print("Failed to extract voice from second audio, using only user voice")
                    voice_emb = user_voice_emb
            else:
                voice_emb = user_voice_emb
            
            print(f"Embedding preparation took {time.time() - embed_start:.2f} seconds")
            
            # Move to appropriate device if GPU is enabled
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            
            # Generate speech - use different approach based on text length
            gen_start = time.time()
            
            # Determine if we should use chunking
            use_chunking = smart_chunking and len(text) > 500  # Only chunk long texts
            
            if use_chunking:
                print(f"Processing long text ({len(text)} chars) using smart chunking")
                
                # Split text into chunks with context overlap
                chunks = self.smart_split_text(text)
                
                # Process each chunk with consistent parameters
                audio_segments = []
                
                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:30]}...")
                    chunk_start = time.time()
                    
                    # Use consistent CPS value for all chunks
                    cps_value = 13.0
                    
                    # Generate speech for this chunk
                    audio_wave = self.generate_speech_for_chunk(chunk, voice_emb, pipe, device, cps=cps_value)
                    
                    if audio_wave is not None:
                        audio_segments.append(audio_wave)
                        print(f"Chunk {i+1} processed in {time.time() - chunk_start:.2f} seconds")
                    else:
                        print(f"Failed to process chunk {i+1}, skipping")
                
                # Combine segments with crossfading
                if audio_segments:
                    audio_wave = self.crossfade_audio_segments(audio_segments)
                else:
                    print("No valid audio segments generated")
                    audio_wave = torch.zeros(24000, dtype=torch.float32, device=device)  # 1 second of silence
            else:
                print(f"Processing text as a single unit (length: {len(text)} chars)")
                
                # Adjust CPS based on text length for optimal results
                cps = 13.0 if len(text) < 50 else 14.0
                
                # Generate in a single pass
                audio_wave = self.generate_speech_for_chunk(text, voice_emb, pipe, device, cps=cps)
                
                if audio_wave is None:
                    print("Failed to generate speech")
                    audio_wave = torch.zeros(24000, dtype=torch.float32, device=device)  # 1 second of silence
            
            print(f"Speech generation took {time.time() - gen_start:.2f} seconds")
            
            # Apply volume adjustment if needed
            if output_volume != 0.0:
                # Convert dB to linear gain
                gain = 10.0 ** (output_volume / 20.0)
                audio_wave = audio_wave * gain
                
                # Apply soft clipping if needed
                if torch.max(torch.abs(audio_wave)) > 0.95:
                    audio_wave = torch.tanh(audio_wave)
            
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
            
            print(f"Total processing completed in {time.time() - total_start_time:.2f} seconds")
            return output_audio, original_text
            
        except Exception as e:
            print(f"Error in speech generation: {e}")
            import traceback
            traceback.print_exc()
            return default_output, text


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeekyWhisperSpeech": GeekyWhisperSpeechNode
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyWhisperSpeech": "🔊 Geeky WhisperSpeech"
}