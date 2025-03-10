import folder_paths
import os
import torch
import numpy as np
import soundfile as sf
from kokoro import KModel, KPipeline
import warnings
import time
import threading

class GeekyKokoroTTSNode:
    """
    ComfyUI node for Geeky Kokoro TTS with style-based voice blending and improved text processing
    """
    
    MODEL = None
    PIPELINES = {}
    VOICES = {}
    MODEL_LOCK = threading.Lock()
    
    @classmethod
    def INPUT_TYPES(cls):
        if not cls.VOICES:
            cls._initialize()
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Welcome to Geeky Kokoro TTS for ComfyUI. You can adjust voice parameters to customize the output."}),
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
        print("Initializing Geeky Kokoro TTS...")
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
        
        for code in ['a', 'b']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls.PIPELINES[code] = KPipeline(lang_code=code, model=False, repo_id='hexgrad/Kokoro-82M')
        
        cls.PIPELINES['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
        cls.PIPELINES['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
        
        for voice_code in cls.VOICES.values():
            cls.PIPELINES[voice_code[0]].load_voice(voice_code)
        
        with cls.MODEL_LOCK:
            if cls.MODEL is None:
                cls.MODEL = {}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cls.MODEL[False] = KModel(repo_id='hexgrad/Kokoro-82M').to('cpu').eval()
                if torch.cuda.is_available():
                    print("CUDA available. GPU model on demand.")
        print("Initialization complete.")
    
    def _process_text_chunks(self, text, pipeline, voice_code, speed, use_gpu, max_chunk_size=400, ref_s=None):
        """
        Process text in chunks with improved handling and crossfading
        Optional ref_s for blended voices
        """
        processed_text = text.replace('\n', ' ').strip()
        if not processed_text:
            print("Warning: Empty text after preprocessing.")
            return None, 24000
        
        # Split into chunks, preserving sentence boundaries
        chunks = []
        sentences = processed_text.split('.')
        current_chunk, current_size = [], 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence += '.'
            sentence_size = len(sentence)
            
            if sentence_size > max_chunk_size:
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
            elif current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_size = [sentence], sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        if not chunks:
            chunks = [processed_text]
        
        print(f"Processing {len(chunks)} chunks...")
        all_audio = []
        sample_rate = 24000
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            print(f"Chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            phoneme_sequences = list(pipeline(chunk, voice_code, speed))
            if not phoneme_sequences:
                print(f"Warning: No phonemes for chunk {i+1}")
                continue
            
            _, ps, _ = phoneme_sequences[0]
            # Use provided ref_s for blending, otherwise load from pipeline
            ref_s_chunk = ref_s if ref_s is not None else pipeline.load_voice(voice_code)[len(ps)-1]
            try:
                with self.MODEL_LOCK:
                    audio = self.MODEL[use_gpu and True in self.MODEL](ps, ref_s_chunk, speed)
                audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                all_audio.append(audio_np)
            except Exception as e:
                print(f"Chunk {i+1} failed: {e}")
                if use_gpu:
                    try:
                        with self.MODEL_LOCK:
                            audio = self.MODEL[False](ps, ref_s_chunk, speed)
                        audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
                        all_audio.append(audio_np)
                    except Exception as e2:
                        print(f"CPU fallback failed: {e2}")
        
        if not all_audio:
            print("No audio generated from chunks.")
            return None, sample_rate
        
        if len(all_audio) == 1:
            return all_audio[0], sample_rate
        
        # Crossfade between chunks
        crossfade_ms = 30  # 30ms
        crossfade_samples = int(crossfade_ms / 1000 * sample_rate)
        total_length = sum(len(a) for a in all_audio) - crossfade_samples * (len(all_audio) - 1)
        result = np.zeros(total_length, dtype=np.float32)
        pos = 0
        
        for i, chunk in enumerate(all_audio):
            chunk_len = len(chunk)
            if i == 0:
                result[:chunk_len] = chunk
                pos += chunk_len - crossfade_samples
            else:
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                result[pos:pos+crossfade_samples] = (
                    result[pos:pos+crossfade_samples] * fade_out +
                    chunk[:crossfade_samples] * fade_in
                )
                if crossfade_samples < chunk_len:
                    result[pos+crossfade_samples:pos+chunk_len] = chunk[crossfade_samples:]
                pos += chunk_len - crossfade_samples
        
        return result, sample_rate
    
    def _blend_voice_styles(self, ps, ref_s1, ref_s2, blend_ratio, speed, use_gpu):
        """Blend reference styles and generate audio"""
        try:
            # Convert blend_ratio (0-1) to Kokoro-style weights (summing to 100)
            weight1 = blend_ratio * 100  # e.g., 0.6 -> 60
            weight2 = (1 - blend_ratio) * 100  # e.g., 0.4 -> 40
            
            # Assume ref_s is a numpy array or tensor; blend them
            if isinstance(ref_s1, torch.Tensor) and isinstance(ref_s2, torch.Tensor):
                blended_ref_s = ref_s1 * (weight1 / 100) + ref_s2 * (weight2 / 100)
            else:  # Assume numpy array if not tensor
                blended_ref_s = np.add(ref_s1 * (weight1 / 100), ref_s2 * (weight2 / 100))
            
            # Generate audio with blended style
            with self.MODEL_LOCK:
                audio = self.MODEL[use_gpu and True in self.MODEL](ps, blended_ref_s, speed)
            return audio
        except Exception as e:
            print(f"Style blending failed: {e}")
            return None
    
    def generate_speech(self, text, voice, speed, use_gpu, enable_blending=False, 
                       second_voice=None, blend_ratio=0.5):
        if not self.VOICES:
            self._initialize()
        
        if use_gpu and True not in self.MODEL and torch.cuda.is_available():
            try:
                with self.MODEL_LOCK:
                    self.MODEL[True] = KModel().to('cuda').eval()
            except Exception as e:
                print(f"GPU load failed: {e}. Using CPU.")
                use_gpu = False
        
        voice_code = self.VOICES[voice]
        pipeline = self.PIPELINES[voice_code[0]]
        start_time = time.time()
        
        if enable_blending and second_voice and second_voice != voice:
            second_code = self.VOICES[second_voice]
            pipeline2 = self.PIPELINES[second_code[0]]
            
            # Process full text into chunks with blended styles
            processed_text = text.replace('\n', ' ').strip()
            phoneme_sequences = list(pipeline(processed_text, voice_code, speed))
            if not phoneme_sequences:
                print("No phoneme sequences generated.")
                return {"waveform": torch.zeros((1, 1, 1000), dtype=torch.float32), "sample_rate": 24000}, text
            
            # Get reference styles for both voices (using first chunk for consistency)
            _, ps, _ = phoneme_sequences[0]
            ref_s1 = pipeline.load_voice(voice_code)[len(ps)-1]
            ref_s2 = pipeline2.load_voice(second_code)[len(ps)-1]
            
            # Blend styles once for all chunks
            print(f"Blending '{voice}' ({blend_ratio*100}%) and '{second_voice}' ({(1-blend_ratio)*100}%)")
            if isinstance(ref_s1, torch.Tensor) and isinstance(ref_s2, torch.Tensor):
                blended_ref_s = ref_s1 * (blend_ratio) + ref_s2 * (1 - blend_ratio)
            else:
                blended_ref_s = np.add(ref_s1 * blend_ratio, ref_s2 * (1 - blend_ratio))
            
            # Process chunks with blended style
            final_audio, sample_rate = self._process_text_chunks(
                text, pipeline, voice_code, speed, use_gpu, ref_s=blended_ref_s
            )
            
            if final_audio is None:
                print("Blended audio generation failed, falling back to primary voice.")
                final_audio, sample_rate = self._process_text_chunks(text, pipeline, voice_code, speed, use_gpu)
                final_audio = final_audio if final_audio is not None else np.zeros(1000, dtype=np.float32)
        else:
            # Single voice, no blending
            final_audio, sample_rate = self._process_text_chunks(text, pipeline, voice_code, speed, use_gpu)
            final_audio = final_audio if final_audio is not None else np.zeros(1000, dtype=np.float32)
        
        waveform = torch.tensor(final_audio, dtype=torch.float32).unsqueeze(0)
        if waveform.shape[-1] < 1000:
            waveform = torch.nn.functional.pad(waveform, (0, 1000 - waveform.shape[-1]))
        
        print(f"Completed in {time.time() - start_time:.2f} seconds")
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}, text

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeekyKokoroTTS": GeekyKokoroTTSNode
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyKokoroTTS": "🔊 Geeky Kokoro TTS"
}
