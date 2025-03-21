"""
Advanced voice modification node for ComfyUI's Geeky Kokoro TTS system.
This module provides voice transformation capabilities to modify TTS output.
"""
import torch
import numpy as np
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import the voice profile utilities
from .voice_profiles_utils import VoiceProfileUtils, SCIPY_AVAILABLE, FALLBACKS_AVAILABLE


class GeekyKokoroAdvancedVoiceNode:
    """
    ComfyUI node for applying advanced voice effects and transformations.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node."""
        return {
            "required": {
                "audio": ("AUDIO",),
                "effect_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "output_volume": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0, "display": "slider"}),
                "voice_profile": (["None", "Cinematic", "Monster", "Singer", "Robot", "Child", "Darth Vader", "Custom"], 
                                 {"default": "None"}),
                "profile_intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            },
            "optional": {
                "manual_mode": ("BOOLEAN", {"default": False}),
                "pitch_shift": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5, "display": "slider"}),
                "formant_shift": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1, "display": "slider"}),
                "reverb_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "echo_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "distortion": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "compression": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "eq_bass": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1, "display": "slider"}),
                "eq_mid": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1, "display": "slider"}),
                "eq_treble": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1, "display": "slider"}),
                "use_gpu": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_voice"
    CATEGORY = "audio"
    
    def debug_audio_stats(self, audio, name="audio"):
        """
        Print debug information about audio array.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal to analyze
        name : str
            Identifier for the audio in debug output
            
        Returns:
        --------
        bool
            True if audio contains valid values, False otherwise
        """
        try:
            min_val = np.min(audio)
            max_val = np.max(audio)
            mean_val = np.mean(audio)
            rms = np.sqrt(np.mean(audio**2))
            has_nan = np.any(np.isnan(audio))
            has_inf = np.any(np.isinf(audio))
            
            logger.debug(
                f"Audio '{name}' stats: Shape: {audio.shape}, "
                f"Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}, "
                f"RMS: {rms:.4f}, NaN: {has_nan}, Inf: {has_inf}"
            )
            
            if has_nan or has_inf or max_val > 10.0 or min_val < -10.0:
                logger.warning(f"Problematic values detected in {name}")
            
            return not (has_nan or has_inf)
        except Exception as e:
            logger.error(f"Debug error: {e}")
            return False
    
    def soft_clip(self, audio, threshold=0.99):
        """
        Apply soft clipping using tanh to reduce distortion.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal to clip
        threshold : float
            Clipping threshold
            
        Returns:
        --------
        numpy.ndarray
            Soft-clipped audio
        """
        scale = 2.0  # Adjust this for softer/harder clipping (higher = softer)
        clipped = np.tanh(audio * scale / threshold) * threshold
        return clipped
    
    def linear_blend(self, original_audio, processed_audio, blend_factor):
        """
        Simple linear blending in the time domain.
        
        Parameters:
        -----------
        original_audio : numpy.ndarray
            Original audio signal
        processed_audio : numpy.ndarray
            Processed audio signal
        blend_factor : float
            Blend factor (0.0 = original only, 1.0 = processed only)
            
        Returns:
        --------
        numpy.ndarray
            Blended audio
        """
        return original_audio * (1 - blend_factor) + processed_audio * blend_factor
    
    def process_voice(self, audio, effect_blend=1.0, output_volume=0.0,
                     voice_profile="None", profile_intensity=0.7, manual_mode=False,
                     pitch_shift=0.0, formant_shift=0.0, reverb_amount=0.0, echo_delay=0.0,
                     distortion=0.0, compression=0.0, eq_bass=0.0, eq_mid=0.0, eq_treble=0.0,
                     use_gpu=False):
        """
        Process audio with voice effects.
        
        Parameters:
        -----------
        audio : dict
            Input audio dictionary with 'waveform' and 'sample_rate' keys
        effect_blend : float
            Blend amount between original and processed audio
        output_volume : float
            Output volume adjustment in dB
        voice_profile : str
            Name of the voice profile to apply
        profile_intensity : float
            Intensity of the voice profile effect
        manual_mode : bool
            Whether to use manual parameter settings
        pitch_shift : float
            Pitch shift amount in semitones
        formant_shift : float
            Formant shift amount
        reverb_amount : float
            Reverb amount
        echo_delay : float
            Echo delay amount
        distortion : float
            Distortion amount
        compression : float
            Compression amount
        eq_bass : float
            Bass EQ adjustment
        eq_mid : float
            Mid EQ adjustment
        eq_treble : float
            Treble EQ adjustment
        use_gpu : bool
            Whether to use GPU for processing
            
        Returns:
        --------
        tuple
            Tuple containing output audio dictionary
        """
        # Default output in case of errors
        default_output = {"waveform": torch.zeros((1, 1, 1000), dtype=torch.float32), "sample_rate": 24000}
        
        try:
            # Validate input audio
            if audio is None or not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
                logger.error(f"Invalid audio input: {type(audio)}")
                return (default_output,)
            
            start_time = time.time()
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Normalize waveform shape
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
            
            # Make copies for processing
            original_audio = audio_data.copy()
            result = audio_data.copy()
            
            # Apply voice profile or manual parameters
            if voice_profile != "None" and voice_profile != "Custom" and profile_intensity > 0:
                result = VoiceProfileUtils.apply_voice_profile(
                    result, sample_rate, voice_profile, profile_intensity, use_gpu
                )
                self.debug_audio_stats(result, "after_voice_profile")
            elif manual_mode or voice_profile == "Custom":
                # Only apply effects that have non-zero values
                if abs(pitch_shift) >= 0.1:
                    result = VoiceProfileUtils._apply_pitch_shift(result, sample_rate, pitch_shift, use_gpu)
                
                if abs(formant_shift) >= 0.1:
                    result = VoiceProfileUtils._apply_formant_shift(result, sample_rate, formant_shift)
                
                if reverb_amount >= 0.1:
                    result = VoiceProfileUtils._apply_reverb(result, sample_rate, reverb_amount)
                
                if echo_delay >= 0.1:
                    result = VoiceProfileUtils._apply_echo(result, sample_rate, echo_delay)
                
                if distortion >= 0.1:
                    result = VoiceProfileUtils._apply_distortion(result, distortion)
                
                if compression >= 0.1:
                    result = VoiceProfileUtils._apply_compression(result, compression)
                
                if abs(eq_bass) >= 0.1 or abs(eq_mid) >= 0.1 or abs(eq_treble) >= 0.1:
                    result = VoiceProfileUtils._apply_eq(
                        result, sample_rate, eq_bass, eq_mid, eq_treble
                    )
            
            # Apply blending with original audio
            if effect_blend < 1.0:
                result = self.linear_blend(original_audio, result, effect_blend)
                self.debug_audio_stats(result, "after_linear_blend")
            
            # Apply volume adjustment
            if output_volume <= -60.0:  # Treat -60 dB as mute
                result = np.zeros_like(result)
                logger.info("Volume set to -60 dB or lower, audio muted")
            else:
                # Convert dB to linear gain
                gain = 10.0 ** (output_volume / 20.0)
                result = result * gain
                self.debug_audio_stats(result, "after_volume_gain")
            
            # Debug peak value
            max_val = np.max(np.abs(result))
            logger.debug(f"Peak amplitude after volume adjustment: {max_val:.4f}")
            
            # Check for NaN/Inf after volume adjustment
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                logger.warning("NaN or inf values detected, using original audio")
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
                logger.info(f"Applying soft clipping: peak {max_amp:.4f} limited to ±0.99")
                result = self.soft_clip(result, threshold=0.99)
                self.debug_audio_stats(result, "after_soft_clipping")
            
            # Convert to torch tensor for output
            processed_waveform = torch.tensor(result, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.debug_audio_stats(processed_waveform.numpy().squeeze(), "final_output")
            
            # Ensure correct shape
            if processed_waveform.shape != (1, 1, original_length):
                processed_waveform = processed_waveform[:, :, :original_length]
                if processed_waveform.shape[-1] < original_length:
                    processed_waveform = torch.nn.functional.pad(
                        processed_waveform, 
                        (0, original_length - processed_waveform.shape[-1])
                    )
            
            output = {"waveform": processed_waveform, "sample_rate": sample_rate}
            logger.info(f"Voice processing completed in {time.time() - start_time:.2f} seconds")
            return (output,)
            
        except Exception as e:
            logger.error(f"Voice processing critical error: {str(e)}")
            import traceback
            traceback.print_exc()
            return (default_output,)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeekyKokoroAdvancedVoice": GeekyKokoroAdvancedVoiceNode
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyKokoroAdvancedVoice": "🔊 Geeky Kokoro Advanced Voice"
}
