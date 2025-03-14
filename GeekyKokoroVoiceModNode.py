import torch
import numpy as np
import time
import os
import sys

# Import the voice profile utilities
from .voice_profiles_utils import VoiceProfileUtils, SCIPY_AVAILABLE, FALLBACKS_AVAILABLE

class GeekyKokoroAdvancedVoiceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "effect_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "output_volume": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0, "display": "slider"}),
            },
            "optional": {
                "voice_profile": (["None", "Child", "Teen", "Adult", "Elder", "Feminine", "Masculine", 
                                  "Robot", "Telephone", "Megaphone", "Radio", "Underwater", "Whisper", 
                                  "Demon", "Angel", "Alien", "Darth Vader", "Chipmunk", "Deep Voice", 
                                  "Fantasy Elf", "Orc", "Monster", "Ghost", "Radio Host", "TV Announcer", 
                                  "Movie Trailer", "Singer"], {"default": "None"}),
                "profile_intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "blend_method": (["Linear", "Spectral", "Crossfade"], {"default": "Linear"}),  # New dropdown
                "enable_pitch_formant": ("BOOLEAN", {"default": False}),
                "pitch_shift": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5, "display": "slider"}),
                "formant_shift": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1, "display": "slider"}),
                "auto_tune": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "enable_time": ("BOOLEAN", {"default": False}),
                "time_stretch": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "display": "slider"}),
                "vibrato": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "vibrato_speed": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 8.0, "step": 0.1}),
                "enable_spatial": ("BOOLEAN", {"default": False}),
                "reverb_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "reverb_room_size": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "enable_tone": ("BOOLEAN", {"default": False}),
                "bass_boost": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "mid_boost": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "treble_boost": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "harmonics": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "enable_effects": ("BOOLEAN", {"default": False}),
                "distortion": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "tremolo": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "bitcrush": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "noise_reduction": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "enable_dynamics": ("BOOLEAN", {"default": False}),
                "compression": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "limit_ceiling": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "warmth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_voice"
    CATEGORY = "audio"
    
    def debug_audio_stats(self, audio, name="audio"):
        """Prints debug information about audio array"""
        try:
            min_val = np.min(audio)
            max_val = np.max(audio)
            mean_val = np.mean(audio)
            rms = np.sqrt(np.mean(audio**2))
            has_nan = np.any(np.isnan(audio))
            has_inf = np.any(np.isinf(audio))
            print(f"Audio '{name}' stats: Shape: {audio.shape}, Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}, RMS: {rms:.4f}, NaN: {has_nan}, Inf: {has_inf}")
            if has_nan or has_inf or max_val > 10.0 or min_val < -10.0:
                print(f"  WARNING: Problematic values detected in {name}")
            return not (has_nan or has_inf)
        except Exception as e:
            print(f"Debug error: {e}")
            return False
    
    def soft_clip(self, audio, threshold=0.99):
        """Apply soft clipping using tanh to reduce distortion"""
        scale = 2.0  # Adjust this for softer/harder clipping (higher = softer)
        clipped = np.tanh(audio * scale / threshold) * threshold
        return clipped
    
    def linear_blend(self, original_audio, processed_audio, blend_factor):
        """Simple linear blending in the time domain"""
        return original_audio * (1 - blend_factor) + processed_audio * blend_factor

    def spectral_blend(self, original_audio, processed_audio, sr, blend_factor):
        """Spectral blending using STFT"""
        if not SCIPY_AVAILABLE or not FALLBACKS_AVAILABLE:
            print("Spectral blending requires scipy and audio_utils, falling back to linear")
            return self.linear_blend(original_audio, processed_audio, blend_factor)
        
        from .audio_utils import stft, istft
        
        n_fft = 2048
        hop_length = n_fft // 4
        
        S_orig = stft(original_audio, n_fft=n_fft, hop_length=hop_length)
        S_proc = stft(processed_audio, n_fft=n_fft, hop_length=hop_length)
        
        mag_orig = np.abs(S_orig)
        mag_proc = np.abs(S_proc)
        blended_mag = mag_orig * (1 - blend_factor) + mag_proc * blend_factor
        
        phase_orig = np.angle(S_orig)
        S_blended = blended_mag * np.exp(1j * phase_orig)
        
        blended_audio = istft(S_blended, hop_length=hop_length, length=len(original_audio))
        return VoiceProfileUtils.preserve_volume(original_audio, blended_audio)

    def crossfade_blend(self, original_audio, processed_audio, sr, blend_factor):
        """Crossfade blending with overlapping segments"""
        crossfade_ms = 50
        crossfade_samples = int(crossfade_ms / 1000 * sr)
        segment_length = crossfade_samples * 2
        num_segments = len(original_audio) // segment_length
        
        result = np.zeros_like(original_audio)
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            if end > len(original_audio):
                end = len(original_audio)
            
            orig_segment = original_audio[start:end]
            proc_segment = processed_audio[start:end]
            
            fade_length = min(crossfade_samples, len(orig_segment))
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            
            if i == 0:
                result[start:end] = orig_segment * (1 - blend_factor) + proc_segment * blend_factor
            else:
                overlap_start = start
                overlap_end = start + fade_length
                result[overlap_start:overlap_end] = (
                    result[overlap_start:overlap_end] * fade_out +
                    (orig_segment[:fade_length] * (1 - blend_factor) + proc_segment[:fade_length] * blend_factor) * fade_in
                )
                if end > overlap_end:
                    result[overlap_end:end] = orig_segment[fade_length:] * (1 - blend_factor) + proc_segment[fade_length:] * blend_factor
        
        # Handle remainder
        if num_segments * segment_length < len(original_audio):
            start = num_segments * segment_length
            result[start:] = original_audio[start:] * (1 - blend_factor) + processed_audio[start:] * blend_factor
        
        return VoiceProfileUtils.preserve_volume(original_audio, result)
    
    def process_voice(self, audio, effect_blend=1.0, output_volume=0.0,
                     voice_profile="None", profile_intensity=0.7, blend_method="Linear",
                     enable_pitch_formant=False, pitch_shift=0.0, formant_shift=0.0, auto_tune=0.0,
                     enable_time=False, time_stretch=1.0, vibrato=0.0, vibrato_speed=5.0,
                     enable_spatial=False, reverb_amount=0.0, reverb_room_size=0.5, 
                     echo_delay=0.0,
                     enable_tone=False, bass_boost=0.0, mid_boost=0.0, treble_boost=0.0, harmonics=0.0,
                     enable_effects=False, distortion=0.0, tremolo=0.0, bitcrush=0.0, noise_reduction=0.0,
                     enable_dynamics=False, compression=0.0, limit_ceiling=0.95, warmth=0.0):
        default_output = {"waveform": torch.zeros((1, 1, 1000), dtype=torch.float32), "sample_rate": 24000}
        try:
            if audio is None or not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
                print(f"Error: Invalid audio input: {type(audio)}")
                return (default_output,)
            
            start_time = time.time()
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if waveform.dim() != 3 or waveform.shape[0] != 1:
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0)
                elif waveform.dim() == 2:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform[0:1, 0:1, :]
            
            original_length = waveform.shape[-1]
            audio_data = waveform.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            self.debug_audio_stats(audio_data, "input_audio")
            
            original_audio = audio_data.copy()
            result = audio_data.copy()
            
            # Apply voice profile from the utility class
            if voice_profile != "None" and profile_intensity > 0:
                result = VoiceProfileUtils.apply_voice_profile(result, sample_rate, voice_profile, profile_intensity)
                self.debug_audio_stats(result, "after_voice_profile")
            else:
                # If no profile selected, use manual controls with utility methods
                if enable_pitch_formant:
                    if abs(pitch_shift) >= 0.1:
                        result = VoiceProfileUtils._apply_pitch_shift(result, sample_rate, pitch_shift)
                    if abs(formant_shift) >= 0.1:
                        result = VoiceProfileUtils._apply_formant_shift(result, sample_rate, formant_shift)
                    if auto_tune >= 0.1:
                        result = VoiceProfileUtils._apply_auto_tune(result, sample_rate, auto_tune)
                
                if enable_time:
                    if abs(time_stretch - 1.0) >= 0.01:
                        result = VoiceProfileUtils._apply_time_stretch(result, sample_rate, time_stretch)
                    if vibrato >= 0.1:
                        result = VoiceProfileUtils._apply_vibrato(result, sample_rate, vibrato, vibrato_speed)
                
                if enable_spatial:
                    if reverb_amount >= 0.1:
                        result = VoiceProfileUtils._apply_reverb(result, sample_rate, reverb_amount, reverb_room_size)
                    if echo_delay >= 0.1:
                        result = VoiceProfileUtils._apply_echo(result, sample_rate, echo_delay)
                
                if enable_tone:
                    if abs(bass_boost) >= 0.1 or abs(mid_boost) >= 0.1 or abs(treble_boost) >= 0.1:
                        result = VoiceProfileUtils._apply_eq(result, sample_rate, bass_boost, mid_boost, treble_boost)
                    if harmonics >= 0.1:
                        result = VoiceProfileUtils._apply_harmonics(result, sample_rate, harmonics)
                
                if enable_effects:
                    if distortion >= 0.1:
                        result = VoiceProfileUtils._apply_distortion(result, distortion)
                    if tremolo >= 0.1:
                        result = VoiceProfileUtils._apply_tremolo(result, tremolo, sample_rate)
                    if bitcrush >= 0.1:
                        result = VoiceProfileUtils._apply_bitcrush(result, bitcrush)
                    if noise_reduction >= 0.1:
                        result = VoiceProfileUtils._apply_noise_reduction(result, noise_reduction)
                
                if enable_dynamics:
                    if compression >= 0.1:
                        result = VoiceProfileUtils._apply_compression(result, compression, limit_ceiling)
                    if warmth >= 0.1:
                        result = VoiceProfileUtils._apply_warmth(result, warmth)
            
            # Apply selected blending method
            if effect_blend < 1.0:
                print(f"Using {blend_method} blending")
                if blend_method == "Spectral":
                    result = self.spectral_blend(original_audio, result, sample_rate, effect_blend)
                elif blend_method == "Crossfade":
                    result = self.crossfade_blend(original_audio, result, sample_rate, effect_blend)
                else:  # Default to Linear
                    result = self.linear_blend(original_audio, result, effect_blend)
                self.debug_audio_stats(result, f"after_{blend_method.lower()}_blend")
            
            # VOLUME CONTROL IMPLEMENTATION - WITH DECIBEL-BASED SCALING
            if output_volume <= -60.0:  # Treat -60 dB as mute
                result = np.zeros_like(result)
                print("Volume set to -60 dB or lower, audio muted")
            else:
                # Convert dB to linear gain (dB = 20 * log10(gain))
                gain = 10.0 ** (output_volume / 20.0)
                result = result * gain
                self.debug_audio_stats(result, "after_volume_gain")
                
                # Log the volume change for debugging
                initial_rms = np.sqrt(np.mean(original_audio**2)) if np.any(original_audio != 0) else 0.0001
                final_rms = np.sqrt(np.mean(result**2)) if np.any(result != 0) else 0.0001
                rms_ratio = final_rms / initial_rms if initial_rms > 0 else 0
                print(f"Volume control: {output_volume:+.1f} dB (Gain: {gain:.3f}x, RMS change: {initial_rms:.4f} → {final_rms:.4f}, ratio: {rms_ratio:.2f}x)")

            # Debug peak value
            max_val = np.max(np.abs(result))
            print(f"Peak amplitude after volume adjustment: {max_val:.4f}")
            
            # Check for NaN/Inf AFTER volume adjustment
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                print("WARNING: NaN or inf values detected, using original audio")
                gain = 10.0 ** (output_volume / 20.0)
                result = original_audio.copy() * gain
            
            # Length adjustment
            if len(result) != original_length:
                if len(result) > original_length:
                    result = result[:original_length]
                else:
                    result = np.pad(result, (0, original_length - len(result)), mode='constant')
                self.debug_audio_stats(result, "after_length_adjustment")
            
            # Soft clipping to reduce distortion
            max_amp = np.max(np.abs(result))
            if max_amp > 0.99:
                print(f"Applying soft clipping: peak {max_amp:.4f} limited to ±0.99")
                result = self.soft_clip(result, threshold=0.99)
                self.debug_audio_stats(result, "after_soft_clipping")
            
            # Convert to torch tensor for output
            processed_waveform = torch.tensor(result, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.debug_audio_stats(processed_waveform.numpy().squeeze(), "final_output")
            
            # Ensure correct shape
            if processed_waveform.shape != (1, 1, original_length):
                processed_waveform = processed_waveform[:, :, :original_length]
                if processed_waveform.shape[-1] < original_length:
                    processed_waveform = torch.nn.functional.pad(processed_waveform, (0, original_length - processed_waveform.shape[-1]))
            
            output = {"waveform": processed_waveform, "sample_rate": sample_rate}
            print(f"Voice processing completed in {time.time() - start_time:.2f} seconds")
            return (output,)
        except Exception as e:
            print(f"Voice processing critical error: {str(e)}")
            import traceback
            traceback.print_exc()
            return (default_output,)

NODE_CLASS_MAPPINGS = {
    "GeekyKokoroAdvancedVoice": GeekyKokoroAdvancedVoiceNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyKokoroAdvancedVoice": "🔊 Geeky Kokoro Advanced Voice"
}
