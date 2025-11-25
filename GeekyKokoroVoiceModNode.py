"""
Advanced voice modification node for ComfyUI's Geeky Kokoro TTS system.
This module provides professional voice transformation capabilities with guided morphing support.

2025 Edition - Enhanced with:
- Guided voice morphing using secondary audio file
- Autotune-style pitch correction
- Advanced spectral and formant morphing
- Professional-grade audio processing
"""
import torch
import numpy as np
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import the voice profile utilities
from .voice_profiles_utils import VoiceProfileUtils, SCIPY_AVAILABLE, FALLBACKS_AVAILABLE

# Import guided morphing utilities
try:
    from .guided_voice_morph import GuidedVoiceMorph
    from .audio_feature_extraction import AudioFeatureExtractor
    GUIDED_MORPH_AVAILABLE = True
    logger.info("✅ Guided voice morphing features loaded successfully")
except ImportError as e:
    GUIDED_MORPH_AVAILABLE = False
    logger.warning(f"⚠️  Guided morphing features not available: {e}")


class GeekyKokoroAdvancedVoiceNode:
    """
    ComfyUI node for applying advanced voice effects and transformations.
    Enhanced with guided voice morphing from secondary audio files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node."""

        # Enhanced voice profiles list
        voice_profiles = [
            "None", "Custom",
            "Cinematic", "Monster", "Singer", "Robot", "Child", "Darth Vader",
            "Alien", "Deep Voice", "Chipmunk", "Telephone", "Radio",
            "Cathedral", "Cave", "Metallic", "Whisper", "Shout"
        ]

        return {
            "required": {
                "audio": ("AUDIO",),
                "effect_blend": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),
                "output_volume": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0,
                    "display": "slider"
                }),
                "voice_profile": (voice_profiles, {"default": "None"}),
                "profile_intensity": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),
            },
            "optional": {
                # Guided morphing inputs
                "guide_audio": ("AUDIO",),
                "enable_guided_morph": ("BOOLEAN", {"default": False}),
                "pitch_morph_amount": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),
                "formant_morph_amount": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),
                "spectral_morph_amount": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),
                "amplitude_morph_amount": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),

                # Manual controls
                "manual_mode": ("BOOLEAN", {"default": False}),
                "pitch_shift": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.1,
                    "display": "slider"
                }),
                "formant_shift": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "display": "slider"
                }),

                # Enhanced audio effects
                "reverb_amount": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),
                "reverb_room_size": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),
                "echo_delay": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),
                "echo_feedback": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.9, "step": 0.01,
                    "display": "slider"
                }),
                "distortion": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),
                "compression": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider"
                }),

                # Enhanced EQ controls
                "eq_bass": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "display": "slider"
                }),
                "eq_mid": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "display": "slider"
                }),
                "eq_treble": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "display": "slider"
                }),

                # Advanced controls
                "time_stretch": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "display": "slider"
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "display": "slider"
                }),
                "warmth": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "display": "slider"
                }),

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
        """
        scale = 2.0
        clipped = np.tanh(audio * scale / threshold) * threshold
        return clipped

    def linear_blend(self, original_audio, processed_audio, blend_factor):
        """
        Simple linear blending in the time domain.
        """
        return original_audio * (1 - blend_factor) + processed_audio * blend_factor

    def extract_audio_data(self, audio_dict):
        """
        Extract numpy array from audio dictionary with proper format handling.
        """
        try:
            waveform = audio_dict["waveform"]
            sample_rate = audio_dict["sample_rate"]

            # Normalize waveform shape
            if waveform.dim() != 3 or waveform.shape[0] != 1:
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0)
                elif waveform.dim() == 2:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform[0:1, 0:1, :]

            audio_data = waveform.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

            # Ensure 1-D mono
            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=0).astype(np.float32, copy=False)
            elif audio_data.ndim > 2:
                audio_data = np.asarray(audio_data, dtype=np.float32).reshape(-1)

            return audio_data, sample_rate

        except Exception as e:
            logger.error(f"Error extracting audio data: {e}")
            return None, None

    def process_voice(self, audio, effect_blend=1.0, output_volume=0.0,
                      voice_profile="None", profile_intensity=0.7,
                      guide_audio=None, enable_guided_morph=False,
                      pitch_morph_amount=0.0, formant_morph_amount=0.0,
                      spectral_morph_amount=0.0, amplitude_morph_amount=0.0,
                      manual_mode=False, pitch_shift=0.0, formant_shift=0.0,
                      reverb_amount=0.0, reverb_room_size=0.5,
                      echo_delay=0.0, echo_feedback=0.3,
                      distortion=0.0, compression=0.0,
                      eq_bass=0.0, eq_mid=0.0, eq_treble=0.0,
                      time_stretch=1.0, brightness=0.0, warmth=0.0,
                      use_gpu=False):
        """
        Process audio with voice effects and optional guided morphing.
        """
        # Default output in case of errors
        default_output = {
            "waveform": torch.zeros((1, 1, 1000), dtype=torch.float32),
            "sample_rate": 24000
        }

        try:
            # Validate input audio
            if audio is None or not isinstance(audio, dict):
                logger.error(f"Invalid audio input: {type(audio)}")
                return (default_output,)

            start_time = time.time()

            # Extract audio data
            audio_data, sample_rate = self.extract_audio_data(audio)
            if audio_data is None:
                return (default_output,)

            original_length = len(audio_data)
            self.debug_audio_stats(audio_data, "input_audio")

            # Make copies for processing
            original_audio = audio_data.copy()
            result = audio_data.copy()

            # ===== GUIDED MORPHING (if enabled and available) =====
            if enable_guided_morph and guide_audio is not None and GUIDED_MORPH_AVAILABLE:
                logger.info("Applying guided voice morphing...")

                # Extract guide audio
                guide_data, guide_sr = self.extract_audio_data(guide_audio)

                if guide_data is not None:
                    try:
                        result = GuidedVoiceMorph.guided_voice_morph(
                            result, sample_rate,
                            guide_data, guide_sr,
                            pitch_morph=pitch_morph_amount,
                            formant_morph=formant_morph_amount,
                            spectral_morph=spectral_morph_amount,
                            amplitude_morph=amplitude_morph_amount
                        )
                        self.debug_audio_stats(result, "after_guided_morph")
                    except Exception as morph_error:
                        logger.error(f"Guided morphing failed: {morph_error}")
                else:
                    logger.warning("Could not extract guide audio data")

            # ===== VOICE PROFILE APPLICATION =====
            if voice_profile != "None" and voice_profile != "Custom" and profile_intensity > 0:
                logger.info(f"Applying voice profile: {voice_profile}")
                result = self._apply_enhanced_voice_profile(
                    result, sample_rate, voice_profile, profile_intensity, use_gpu
                )
                self.debug_audio_stats(result, "after_voice_profile")

            # ===== MANUAL/CUSTOM EFFECTS =====
            elif manual_mode or voice_profile == "Custom":
                logger.info("Applying manual/custom effects...")

                # Pitch shift
                if abs(pitch_shift) >= 0.1:
                    result = VoiceProfileUtils._apply_pitch_shift(
                        result, sample_rate, pitch_shift, use_gpu
                    )

                # Formant shift
                if abs(formant_shift) >= 0.1:
                    result = VoiceProfileUtils._apply_formant_shift(
                        result, sample_rate, formant_shift
                    )

                # Time stretch
                if abs(time_stretch - 1.0) >= 0.05:
                    result = VoiceProfileUtils._apply_time_stretch(
                        result, sample_rate, time_stretch
                    )

                # Reverb
                if reverb_amount >= 0.01:
                    result = VoiceProfileUtils._apply_reverb(
                        result, sample_rate, reverb_amount,
                        room_size=reverb_room_size, damping=0.5
                    )

                # Echo
                if echo_delay >= 0.01:
                    result = VoiceProfileUtils._apply_echo(
                        result, sample_rate, echo_delay, echo_feedback
                    )

                # Distortion
                if distortion >= 0.01:
                    result = VoiceProfileUtils._apply_distortion(result, distortion)

                # Compression
                if compression >= 0.01:
                    result = VoiceProfileUtils._apply_compression(result, compression)

                # EQ
                if abs(eq_bass) >= 0.05 or abs(eq_mid) >= 0.05 or abs(eq_treble) >= 0.05:
                    result = VoiceProfileUtils._apply_eq(
                        result, sample_rate, eq_bass, eq_mid, eq_treble
                    )

                # Brightness control (high shelf)
                if abs(brightness) >= 0.05:
                    result = self._apply_brightness(result, sample_rate, brightness)

                # Warmth control (low shelf)
                if abs(warmth) >= 0.05:
                    result = self._apply_warmth(result, sample_rate, warmth)

            # ===== EFFECT BLENDING =====
            if effect_blend < 1.0:
                result = self.linear_blend(original_audio, result, effect_blend)
                self.debug_audio_stats(result, "after_blend")

            # ===== VOLUME ADJUSTMENT =====
            if output_volume <= -60.0:
                result = np.zeros_like(result)
                logger.info("Volume set to -60 dB or lower, audio muted")
            else:
                gain = 10.0 ** (output_volume / 20.0)
                result = result * gain
                self.debug_audio_stats(result, "after_volume")

            # Check for NaN/Inf
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                logger.warning("NaN or inf values detected, using original audio")
                gain = 10.0 ** (output_volume / 20.0)
                result = original_audio.copy() * gain

            # Length adjustment
            cur_len = int(result.shape[-1])
            if cur_len != original_length:
                if cur_len > original_length:
                    result = result[:original_length]
                else:
                    result = np.pad(
                        result.astype(np.float32, copy=False),
                        (0, original_length - cur_len),
                        mode='constant'
                    )

            # Soft clipping
            max_amp = np.max(np.abs(result))
            if max_amp > 0.99:
                logger.info(f"Applying soft clipping: peak {max_amp:.4f}")
                result = self.soft_clip(result, threshold=0.99)

            # Convert to torch tensor
            processed_waveform = torch.tensor(result, dtype=torch.float32)
            processed_waveform = processed_waveform.unsqueeze(0).unsqueeze(0)

            # Ensure correct shape
            if processed_waveform.shape != (1, 1, original_length):
                processed_waveform = processed_waveform[:, :, :original_length]
                if processed_waveform.shape[-1] < original_length:
                    processed_waveform = torch.nn.functional.pad(
                        processed_waveform,
                        (0, original_length - processed_waveform.shape[-1])
                    )

            output = {"waveform": processed_waveform, "sample_rate": sample_rate}

            elapsed = time.time() - start_time
            logger.info(f"✅ Voice processing completed in {elapsed:.2f} seconds")

            return (output,)

        except Exception as e:
            logger.error(f"Voice processing critical error: {str(e)}")
            import traceback
            traceback.print_exc()
            return (default_output,)

    def _apply_enhanced_voice_profile(self, audio, sr, profile_name, intensity, use_gpu):
        """
        Apply enhanced voice profiles with additional presets.
        """
        # Extended voice profiles
        EXTENDED_PROFILES = {
            "Alien": {
                "pitch_shift": -8,
                "formant_shift": -3,
                "modulation": {"rate": 8, "depth": 0.4},
                "distortion": 0.5,
                "reverb": 0.4
            },
            "Deep Voice": {
                "pitch_shift": -5,
                "formant_shift": -2,
                "eq_bass": 0.6,
                "eq_mid": 0.1,
                "eq_treble": -0.3,
                "compression": 0.4
            },
            "Chipmunk": {
                "pitch_shift": 6,
                "formant_shift": 2.5,
                "time_stretch": 0.9,
                "eq_treble": 0.4
            },
            "Telephone": {
                "filter_band": [300, 3400],
                "distortion": 0.2,
                "compression": 0.6,
                "eq_mid": 0.3
            },
            "Radio": {
                "filter_band": [100, 5000],
                "distortion": 0.15,
                "compression": 0.7,
                "eq_mid": 0.4,
                "eq_treble": 0.2
            },
            "Cathedral": {
                "reverb": 0.9,
                "echo": {"amount": 0.4, "feedback": 0.6},
                "eq_bass": 0.3,
                "eq_mid": 0.1
            },
            "Cave": {
                "reverb": 0.8,
                "echo": {"amount": 0.6, "feedback": 0.5},
                "eq_bass": 0.4,
                "eq_treble": -0.3
            },
            "Metallic": {
                "modulation": {"rate": 100, "depth": 0.2},
                "distortion": 0.4,
                "filter_band": [800, 4000],
                "eq_mid": 0.5
            },
            "Whisper": {
                "pitch_shift": -1,
                "distortion": 0.6,
                "eq_bass": -0.5,
                "eq_treble": 0.6,
                "compression": 0.3
            },
            "Shout": {
                "pitch_shift": 2,
                "distortion": 0.3,
                "compression": 0.8,
                "eq_mid": 0.5,
                "eq_treble": 0.3
            }
        }

        # Check if it's an extended profile
        if profile_name in EXTENDED_PROFILES:
            profile = EXTENDED_PROFILES[profile_name]
            result = audio.copy()

            # Apply pitch shift
            if "pitch_shift" in profile:
                pitch_val = profile["pitch_shift"] * intensity
                if abs(pitch_val) >= 0.1:
                    result = VoiceProfileUtils._apply_pitch_shift(result, sr, pitch_val, use_gpu)

            # Apply formant shift
            if "formant_shift" in profile:
                formant_val = profile["formant_shift"] * intensity
                if abs(formant_val) >= 0.1:
                    result = VoiceProfileUtils._apply_formant_shift(result, sr, formant_val)

            # Apply time stretch
            if "time_stretch" in profile:
                time_val = 1.0 + (profile["time_stretch"] - 1.0) * intensity
                if abs(time_val - 1.0) >= 0.01:
                    result = VoiceProfileUtils._apply_time_stretch(result, sr, time_val)

            # Apply reverb
            if "reverb" in profile:
                reverb_val = profile["reverb"] * intensity
                if reverb_val >= 0.01:
                    result = VoiceProfileUtils._apply_reverb(result, sr, reverb_val)

            # Apply echo
            if "echo" in profile:
                echo_amount = profile["echo"]["amount"] * intensity if isinstance(profile["echo"], dict) else profile["echo"] * intensity
                echo_feedback = profile["echo"].get("feedback", 0.3) if isinstance(profile["echo"], dict) else 0.3
                if echo_amount >= 0.01:
                    result = VoiceProfileUtils._apply_echo(result, sr, echo_amount, echo_feedback)

            # Apply distortion
            if "distortion" in profile:
                dist_val = profile["distortion"] * intensity
                if dist_val >= 0.01:
                    result = VoiceProfileUtils._apply_distortion(result, dist_val)

            # Apply compression
            if "compression" in profile:
                comp_val = profile["compression"] * intensity
                if comp_val >= 0.01:
                    result = VoiceProfileUtils._apply_compression(result, comp_val)

            # Apply EQ
            if any(key in profile for key in ["eq_bass", "eq_mid", "eq_treble"]):
                bass_val = profile.get("eq_bass", 0) * intensity
                mid_val = profile.get("eq_mid", 0) * intensity
                treble_val = profile.get("eq_treble", 0) * intensity
                if abs(bass_val) >= 0.01 or abs(mid_val) >= 0.01 or abs(treble_val) >= 0.01:
                    result = VoiceProfileUtils._apply_eq(result, sr, bass_val, mid_val, treble_val)

            # Apply band-pass filter
            if "filter_band" in profile and SCIPY_AVAILABLE:
                import scipy.signal as signal
                low, high = profile["filter_band"]
                nyquist = sr / 2
                low_norm = min(max(low / nyquist, 0.01), 0.99)
                high_norm = min(max(high / nyquist, 0.01), 0.99)
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                result = signal.filtfilt(b, a, result)

            # Apply modulation
            if "modulation" in profile and isinstance(profile["modulation"], dict):
                mod_rate = profile["modulation"].get("rate", 5)
                mod_depth = profile["modulation"].get("depth", 0.3) * intensity
                if mod_depth >= 0.01:
                    t = np.arange(len(result)) / sr
                    modulation = 1.0 + mod_depth * np.sin(2 * np.pi * mod_rate * t)
                    result = result * modulation

            return VoiceProfileUtils.preserve_volume(audio, result)

        # Use existing profile from VoiceProfileUtils
        else:
            return VoiceProfileUtils.apply_voice_profile(
                audio, sr, profile_name, intensity, use_gpu
            )

    def _apply_brightness(self, audio, sr, amount):
        """
        Apply brightness control (high-frequency emphasis).
        """
        if abs(amount) < 0.01:
            return audio

        try:
            if SCIPY_AVAILABLE:
                import scipy.signal as signal
                nyquist = sr / 2
                cutoff = min(max(4000 / nyquist, 0.01), 0.99)

                # High-shelf filter
                b, a = signal.butter(2, cutoff, btype='high')
                filtered = signal.filtfilt(b, a, audio)

                # Mix with original
                gain = 1.0 + amount * 0.5
                result = audio + filtered * amount * gain

                return VoiceProfileUtils.preserve_volume(audio, result)
            else:
                # FFT-based fallback
                n = len(audio)
                fft = np.fft.rfft(audio)
                freq = np.fft.rfftfreq(n, 1/sr)

                # Emphasize high frequencies
                gain_mask = np.ones(len(freq))
                gain_mask[freq >= 4000] *= (1.0 + amount)

                fft_bright = fft * gain_mask
                result = np.fft.irfft(fft_bright, n)

                return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            logger.error(f"Brightness control error: {e}")
            return audio

    def _apply_warmth(self, audio, sr, amount):
        """
        Apply warmth control (low-frequency emphasis).
        """
        if abs(amount) < 0.01:
            return audio

        try:
            if SCIPY_AVAILABLE:
                import scipy.signal as signal
                nyquist = sr / 2
                cutoff = min(max(500 / nyquist, 0.01), 0.99)

                # Low-shelf filter
                b, a = signal.butter(2, cutoff, btype='low')
                filtered = signal.filtfilt(b, a, audio)

                # Mix with original
                gain = 1.0 + amount * 0.5
                result = audio + filtered * amount * gain

                return VoiceProfileUtils.preserve_volume(audio, result)
            else:
                # FFT-based fallback
                n = len(audio)
                fft = np.fft.rfft(audio)
                freq = np.fft.rfftfreq(n, 1/sr)

                # Emphasize low frequencies
                gain_mask = np.ones(len(freq))
                gain_mask[freq <= 500] *= (1.0 + amount)

                fft_warm = fft * gain_mask
                result = np.fft.irfft(fft_warm, n)

                return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            logger.error(f"Warmth control error: {e}")
            return audio


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeekyKokoroAdvancedVoice": GeekyKokoroAdvancedVoiceNode
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyKokoroAdvancedVoice": "🎛️ Geeky Kokoro Advanced Voice (2025)"
}
