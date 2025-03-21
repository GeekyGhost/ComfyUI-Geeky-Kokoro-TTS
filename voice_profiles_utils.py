"""
Utility functions for voice profiles and audio effects for the Geeky Kokoro TTS system.
This module handles advanced voice transformations and effects.
"""
import numpy as np
import torch
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import the required libraries with fallbacks
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available, using fallback implementations")

try:
    import resampy
    RESAMPY_AVAILABLE = True
except ImportError:
    RESAMPY_AVAILABLE = False
    logger.warning("resampy not available, using fallback implementations")

try:
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, using basic fallback implementations")

# Try to import fallback implementations
try:
    from .audio_utils import (
        simple_pitch_shift, 
        stft_phase_vocoder, 
        formant_shift_basic, 
        stft, 
        istft, 
        amplitude_to_db, 
        db_to_amplitude
    )
    FALLBACKS_AVAILABLE = True
except ImportError:
    FALLBACKS_AVAILABLE = False
    logger.warning("audio_utils.py not found, some effects may not work")


class VoiceProfileUtils:
    """
    A utility class for applying voice transformations and audio effects.
    """
    
    # Voice profiles parameters
    VOICE_PROFILES = {
        "Cinematic": {
            "pitch_shift": -3,
            "reverb": 0.6,
            "compression": 0.5,
            "eq_bass": 0.3,
            "eq_mid": 0.2,
            "eq_treble": 0
        },
        "Monster": {
            "pitch_shift": -6,
            "formant_shift": -1.5,
            "distortion": 0.4,
            "eq_bass": 0.5,
            "eq_mid": 0,
            "eq_treble": -0.2
        },
        "Singer": {
            "pitch_shift": 0,
            "compression": 0.6,
            "eq_bass": 0.1,
            "eq_mid": 0.3,
            "eq_treble": 0.2,
            "reverb": 0.3
        },
        "Robot": {
            "pitch_shift": -2,
            "filter_band": [500, 2000],
            "distortion": 0.3,
            "modulation": {"rate": 50, "depth": 0.3}
        },
        "Child": {
            "pitch_shift": 3,
            "formant_shift": 1.5,
            "time_stretch": 1.1,
            "eq_bass": -0.2,
            "eq_mid": 0.1,
            "eq_treble": 0.3
        },
        "Darth Vader": {
            "pitch_shift": -4,
            "speed": 1.25,
            "lowpass": 2500,
            "echo": {"amount": 0.3, "feedback": 0.4},
            "distortion": 0.3,
            "breath_rate": 0.4,
            "breath_depth": 0.15
        }
    }
    
    @staticmethod
    def preserve_volume(original, processed):
        """
        Preserve the original volume level in the processed audio.
        
        Parameters:
        -----------
        original : numpy.ndarray
            Original audio signal
        processed : numpy.ndarray
            Processed audio signal
            
        Returns:
        --------
        numpy.ndarray
            Volume-normalized processed audio
        """
        try:
            original_rms = np.sqrt(np.mean(original**2))
            processed_rms = np.sqrt(np.mean(processed**2))
            if processed_rms > 0 and original_rms > 0:
                gain = original_rms / processed_rms
                processed *= gain
            return processed
        except Exception as e:
            logger.error(f"Volume preservation error: {e}")
            return processed

    @staticmethod
    def _speed_change(audio, speed_factor):
        """
        Change the playback speed of audio without affecting pitch.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        speed_factor : float
            Speed factor (>1 = faster, <1 = slower)
            
        Returns:
        --------
        numpy.ndarray
            Speed-changed audio
        """
        if abs(speed_factor - 1.0) < 0.01:
            return audio
        try:
            indices = np.round(np.arange(0, len(audio), speed_factor))
            valid_indices = indices[indices < len(audio)].astype(int)
            return audio[valid_indices] if len(valid_indices) > 0 else audio
        except Exception as e:
            logger.error(f"Speed change error: {e}")
            return audio
    
    @staticmethod
    def _apply_pitch_shift(audio, sr, n_steps, use_gpu=False):
        """
        Apply pitch shifting to audio using the best available method.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        n_steps : float
            Number of steps to shift (positive = up, negative = down)
        use_gpu : bool
            Whether to use GPU acceleration if available
            
        Returns:
        --------
        numpy.ndarray
            Pitch-shifted audio
        """
        if abs(n_steps) < 0.1:
            return audio
        try:
            if LIBROSA_AVAILABLE and RESAMPY_AVAILABLE:
                audio_out = librosa.effects.pitch_shift(
                    audio, sr=sr, n_steps=n_steps,
                    bins_per_octave=24,
                    res_type='kaiser_best'
                )
            elif LIBROSA_AVAILABLE:
                audio_out = librosa.effects.pitch_shift(
                    audio, sr=sr, n_steps=n_steps,
                    bins_per_octave=12, res_type='fft'
                )
            elif FALLBACKS_AVAILABLE and SCIPY_AVAILABLE:
                audio_out = stft_phase_vocoder(audio, sr, n_steps)
            else:
                # Basic implementation using speed change and resampling
                rate = 2.0 ** (-n_steps / 12)
                changed = VoiceProfileUtils._speed_change(audio, 1.0/rate)
                indices = np.linspace(0, len(changed) - 1, len(audio))
                audio_out = np.interp(indices, np.arange(len(changed)), changed) if len(changed) > 0 else audio
            
            return VoiceProfileUtils.preserve_volume(audio, audio_out)
        except Exception as e:
            logger.error(f"Pitch shift error: {e}")
            return audio
    
    @staticmethod
    def _apply_formant_shift(audio, sr, shift_amount):
        """
        Apply formant shifting to audio using the best available method.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        shift_amount : float
            Shift amount (-5 to 5, negative = down, positive = up)
            
        Returns:
        --------
        numpy.ndarray
            Formant-shifted audio
        """
        if abs(shift_amount) < 0.1:
            return audio
        try:
            if LIBROSA_AVAILABLE:
                n_fft = 1024  # Reduced from 2048 for efficiency
                hop_length = n_fft // 4
                S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
                S_mag, S_phase = librosa.magphase(S)
                freq_bins = S_mag.shape[0]
                shift_bins = int(freq_bins * shift_amount * 0.02)
                S_mag_shifted = np.zeros_like(S_mag)
                
                if shift_bins > 0:
                    # Shift formants up
                    S_mag_shifted[shift_bins:, :] = S_mag[:-shift_bins, :]
                    # Smoothly blend the lowest frequencies
                    blend_region = min(shift_bins, 20)
                    for i in range(blend_region):
                        weight = i / blend_region
                        S_mag_shifted[i, :] = S_mag[i, :] * (1 - weight)
                elif shift_bins < 0:
                    # Shift formants down
                    shift_bins = abs(shift_bins)
                    S_mag_shifted[:-shift_bins, :] = S_mag[shift_bins:, :]
                    # Keep the very highest frequencies with reduced amplitude
                    S_mag_shifted[-shift_bins:, :] = S_mag[-shift_bins:, :] * 0.5
                else:
                    S_mag_shifted = S_mag
                
                S_shifted = S_mag_shifted * np.exp(1j * np.angle(S))
                y_shifted = librosa.istft(S_shifted, hop_length=hop_length, length=len(audio))
                return VoiceProfileUtils.preserve_volume(audio, y_shifted)
            elif FALLBACKS_AVAILABLE:
                return VoiceProfileUtils.preserve_volume(audio, formant_shift_basic(audio, sr, shift_amount))
            else:
                # Basic implementation for systems with minimal dependencies
                audio_copy = audio.copy()
                shifted = VoiceProfileUtils._apply_pitch_shift(audio_copy, sr, shift_amount * 3)
                time_stretched = VoiceProfileUtils._speed_change(shifted, 2 ** (shift_amount * 3 / 12))
                indices = np.linspace(0, len(time_stretched) - 1, len(audio))
                result = np.interp(indices, np.arange(len(time_stretched)), time_stretched)
                return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            logger.error(f"Formant shift error: {e}")
            return audio
    
    @staticmethod
    def _apply_time_stretch(audio, sr, rate):
        """
        Apply time stretching to audio while preserving pitch.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        rate : float
            Stretch factor (>1 = slower, <1 = faster)
            
        Returns:
        --------
        numpy.ndarray
            Time-stretched audio
        """
        if abs(rate - 1.0) < 0.01:
            return audio
        try:
            if LIBROSA_AVAILABLE:
                stretched = librosa.effects.time_stretch(audio, rate=rate)
                if len(stretched) > len(audio):
                    stretched = stretched[:len(audio)]
                elif len(stretched) < len(audio):
                    stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
            else:
                stretched = VoiceProfileUtils._speed_change(audio, 1.0/rate)
                indices = np.linspace(0, len(stretched) - 1, len(audio))
                stretched = np.interp(indices, np.arange(len(stretched)), stretched)
            return VoiceProfileUtils.preserve_volume(audio, stretched)
        except Exception as e:
            logger.error(f"Time stretch error: {e}")
            return audio
    
    @staticmethod
    def _apply_reverb(audio, sr, amount, room_size=0.5, damping=0.5):
        """
        Apply reverb effect to audio.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        amount : float
            Reverb amount (0.0 to 1.0)
        room_size : float
            Room size parameter (0.0 to 1.0)
        damping : float
            Damping parameter (0.0 to 1.0)
            
        Returns:
        --------
        numpy.ndarray
            Audio with reverb effect
        """
        if amount < 0.01:
            return audio
        try:
            reverb_length = int(sr * (0.1 + room_size * 1.5))
            decay_factor = 0.2 + damping * 0.7
            impulse = np.zeros(reverb_length)
            impulse[0] = 1.0
            
            # Create early reflections
            num_early = int(5 + room_size * 10)
            early_times = np.sort(np.random.randint(1, reverb_length // 3, num_early))
            early_amps = np.random.uniform(0.1, 0.4, num_early) * (1 - damping * 0.5)
            for time, amp in zip(early_times, early_amps):
                impulse[time] = amp
            
            # Add diffuse reverb tail
            for i in range(1, reverb_length):
                impulse[i] += np.random.randn() * np.exp(-i / (sr * decay_factor))
            
            impulse /= np.max(np.abs(impulse))
            
            # Apply convolution
            if SCIPY_AVAILABLE:
                reverb_signal = signal.fftconvolve(audio, impulse, mode='full')[:len(audio)]
            else:
                # Manual convolution (slower but works without scipy)
                reverb_signal = np.zeros_like(audio)
                for i in range(len(audio)):
                    for j in range(min(i + 1, reverb_length)):
                        if i - j >= 0:
                            reverb_signal[i] += audio[i - j] * impulse[j]
            
            output = (1 - amount) * audio + amount * reverb_signal
            return VoiceProfileUtils.preserve_volume(audio, output)
        except Exception as e:
            logger.error(f"Reverb effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_echo(audio, sr, amount, feedback=0.3):
        """
        Apply echo effect to audio.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        amount : float
            Echo amount (0.0 to 1.0)
        feedback : float
            Feedback parameter for multiple echoes (0.0 to 1.0)
            
        Returns:
        --------
        numpy.ndarray
            Audio with echo effect
        """
        if amount < 0.01:
            return audio
        try:
            delay_time = 0.1 + amount * 0.4
            delay_samples = int(sr * delay_time)
            
            # Handle short input signals
            if delay_samples >= len(audio):
                delay_samples = len(audio) // 2
                if delay_samples == 0:
                    return audio
            
            output = np.zeros_like(audio, dtype=np.float32)
            output[:] = audio[:]
            output[delay_samples:] += audio[:-delay_samples] * amount
            
            # Add multiple echo taps with feedback
            if feedback > 0.01:
                for i in range(1, 3):  # Limit to 3 echo taps
                    tap_delay = delay_samples * (i + 1)
                    tap_gain = amount * (feedback ** i)
                    
                    if tap_delay >= len(audio) or tap_gain < 0.01:
                        break
                    output[tap_delay:] += audio[:-tap_delay] * tap_gain
            
            return VoiceProfileUtils.preserve_volume(audio, output)
        except Exception as e:
            logger.error(f"Echo effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_eq(audio, sr, bass=0.0, mid=0.0, treble=0.0):
        """
        Apply 3-band equalizer to audio.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        bass : float
            Bass gain (-1.0 to 1.0)
        mid : float
            Mid gain (-1.0 to 1.0)
        treble : float
            Treble gain (-1.0 to 1.0)
            
        Returns:
        --------
        numpy.ndarray
            Equalized audio
        """
        if abs(bass) < 0.01 and abs(mid) < 0.01 and abs(treble) < 0.01:
            return audio
        try:
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                bass_norm = min(max(250 / nyquist, 0.001), 0.999)
                mid_low_norm = min(max(250 / nyquist, 0.001), 0.999)
                mid_high_norm = min(max(4000 / nyquist, 0.001), 0.999)
                treble_norm = min(max(4000 / nyquist, 0.001), 0.999)
                
                # Ensure valid band limits
                if mid_low_norm >= mid_high_norm:
                    mid_low_norm = mid_high_norm * 0.5
                
                # Create filters
                b_bass, a_bass = signal.butter(2, bass_norm, btype='low')
                b_mid, a_mid = signal.butter(2, [mid_low_norm, mid_high_norm], btype='band')
                b_treble, a_treble = signal.butter(2, treble_norm, btype='high')
                
                # Apply filters
                bass_filtered = signal.filtfilt(b_bass, a_bass, audio)
                mid_filtered = signal.filtfilt(b_mid, a_mid, audio)
                treble_filtered = signal.filtfilt(b_treble, a_treble, audio)
                
                # Mix with gains
                bass_gain = 10 ** (bass * 0.5)
                mid_gain = 10 ** (mid * 0.4)
                treble_gain = 10 ** (treble * 0.5)
                result = bass_filtered * bass_gain + mid_filtered * mid_gain + treble_filtered * treble_gain
            else:
                # FFT-based EQ (less efficient but works without scipy)
                n = len(audio)
                fft = np.fft.rfft(audio)
                freq = np.fft.rfftfreq(n, 1/sr)
                
                # Create frequency masks
                gain_mask = np.ones(len(freq))
                gain_mask[freq < 250] *= (1.0 + bass * 0.8)
                gain_mask[(freq >= 250) & (freq < 4000)] *= (1.0 + mid * 0.6)
                gain_mask[freq >= 4000] *= (1.0 + treble * 0.8)
                
                # Apply masks and convert back to time domain
                fft_eq = fft * gain_mask
                result = np.fft.irfft(fft_eq, n)
            
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            logger.error(f"EQ effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_distortion(audio, effect_strength):
        """
        Apply distortion effect to audio.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        effect_strength : float
            Distortion amount (0.0 to 1.0)
            
        Returns:
        --------
        numpy.ndarray
            Distorted audio
        """
        if effect_strength < 0.01:
            return audio
        try:
            drive = 1 + effect_strength * 15
            distorted = np.tanh(audio * drive) / np.tanh(drive)
            result = audio * (1 - effect_strength) + distorted * effect_strength
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            logger.error(f"Distortion error: {e}")
            return audio
    
    @staticmethod
    def _apply_compression(audio, amount, limit_ceiling=0.95):
        """
        Apply dynamic range compression to audio.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        amount : float
            Compression amount (0.0 to 1.0)
        limit_ceiling : float
            Maximum output level
            
        Returns:
        --------
        numpy.ndarray
            Compressed audio
        """
        if amount < 0.01:
            return audio
        try:
            # Calculate compression parameters
            ratio = 1 + 7 * amount
            rms = np.sqrt(np.mean(audio**2))
            db_rms = 20 * np.log10(max(rms, 1e-8))
            threshold_db = db_rms - 6 - amount * 12
            threshold = 10 ** (threshold_db / 20)
            
            # Time constants
            attack_samples = max(int((20 - amount * 15) * 24000 / 1000), 1)  # Assuming default SR = 24000
            release_samples = max(int((150 + amount * 350) * 24000 / 1000), 1)
            
            # Process sample by sample
            result = np.zeros_like(audio)
            gain = 1.0
            for i in range(len(audio)):
                abs_sample = abs(audio[i])
                if abs_sample > threshold:
                    above_threshold = abs_sample - threshold
                    compressed_above = above_threshold / ratio
                    target_gain = (threshold + compressed_above) / abs_sample
                else:
                    target_gain = 1.0
                
                # Apply time constants
                if target_gain < gain:
                    gain = gain + (target_gain - gain) / attack_samples  # Attack
                else:
                    gain = gain + (target_gain - gain) / release_samples  # Release
                
                result[i] = audio[i] * gain
            
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return audio
    
    @staticmethod
    def _apply_darth_vader(audio, sr, intensity=0.7):
        """
        Apply Darth Vader voice effect.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        intensity : float
            Effect intensity (0.0 to 1.0)
            
        Returns:
        --------
        numpy.ndarray
            Processed audio with Darth Vader effect
        """
        try:
            # Pitch down
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, -4 * intensity)
            
            # Slow down and resample
            slowed = VoiceProfileUtils._speed_change(pitched, 1.25)
            indices = np.linspace(0, len(slowed) - 1, len(audio))
            resampled = np.interp(indices, np.arange(len(slowed)), slowed) if len(slowed) > 0 else audio
            
            # Low-pass filter for helmet effect
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                cutoff = min(max(2500 / nyquist, 0.01), 0.99)
                b, a = signal.butter(2, cutoff, btype='low')
                filtered = signal.filtfilt(b, a, resampled)
            else:
                filtered = resampled
            
            # Add echo for spatial effect
            echo = VoiceProfileUtils._apply_echo(filtered, sr, 0.3 * intensity, 0.4)
            
            # Add distortion for vocal effect
            distorted = VoiceProfileUtils._apply_distortion(echo, 0.3 * intensity)
            
            # Add breathing modulation
            t = np.arange(len(distorted)) / sr
            breath = 1.0 + 0.15 * intensity * np.sin(2 * np.pi * 0.4 * t)
            result = distorted * breath
            
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            logger.error(f"Darth Vader effect error: {e}")
            return audio
    
    @staticmethod
    def apply_voice_profile(audio, sr, profile_name, intensity, use_gpu=False):
        """
        Apply a predefined voice profile to audio.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
        sr : int
            Sample rate
        profile_name : str
            Name of the voice profile to apply
        intensity : float
            Effect intensity (0.0 to 1.0)
        use_gpu : bool
            Whether to use GPU acceleration if available
            
        Returns:
        --------
        numpy.ndarray
            Processed audio with selected voice profile
        """
        if profile_name == "None" or intensity < 0.01:
            return audio
        
        try:
            # Create a copy to avoid modifying the original
            result = audio.copy()
            
            # Apply special case profiles
            if profile_name == "Darth Vader":
                return VoiceProfileUtils._apply_darth_vader(result, sr, intensity)
            
            # Get profile parameters if defined
            if profile_name in VoiceProfileUtils.VOICE_PROFILES:
                profile = VoiceProfileUtils.VOICE_PROFILES[profile_name]
                
                # Apply pitch shift if specified
                if "pitch_shift" in profile:
                    pitch_value = profile["pitch_shift"] * intensity
                    if abs(pitch_value) >= 0.1:
                        result = VoiceProfileUtils._apply_pitch_shift(result, sr, pitch_value, use_gpu)
                
                # Apply formant shift if specified
                if "formant_shift" in profile:
                    formant_value = profile["formant_shift"] * intensity
                    if abs(formant_value) >= 0.1:
                        result = VoiceProfileUtils._apply_formant_shift(result, sr, formant_value)
                
                # Apply time stretch if specified
                if "time_stretch" in profile:
                    time_value = 1.0 + (profile["time_stretch"] - 1.0) * intensity
                    if abs(time_value - 1.0) >= 0.01:
                        result = VoiceProfileUtils._apply_time_stretch(result, sr, time_value)
                
                # Apply reverb if specified
                if "reverb" in profile:
                    reverb_value = profile["reverb"] * intensity
                    if reverb_value >= 0.01:
                        result = VoiceProfileUtils._apply_reverb(result, sr, reverb_value)
                
                # Apply echo if specified
                if "echo" in profile:
                    echo_amount = profile["echo"]["amount"] * intensity if isinstance(profile["echo"], dict) else profile["echo"] * intensity
                    echo_feedback = profile["echo"].get("feedback", 0.3) if isinstance(profile["echo"], dict) else 0.3
                    if echo_amount >= 0.01:
                        result = VoiceProfileUtils._apply_echo(result, sr, echo_amount, echo_feedback)
                
                # Apply distortion if specified
                if "distortion" in profile:
                    distortion_value = profile["distortion"] * intensity
                    if distortion_value >= 0.01:
                        result = VoiceProfileUtils._apply_distortion(result, distortion_value)
                
                # Apply compression if specified
                if "compression" in profile:
                    compression_value = profile["compression"] * intensity
                    if compression_value >= 0.01:
                        result = VoiceProfileUtils._apply_compression(result, compression_value)
                
                # Apply EQ if specified
                if any(key in profile for key in ["eq_bass", "eq_mid", "eq_treble"]):
                    bass_value = profile.get("eq_bass", 0) * intensity
                    mid_value = profile.get("eq_mid", 0) * intensity
                    treble_value = profile.get("eq_treble", 0) * intensity
                    if abs(bass_value) >= 0.01 or abs(mid_value) >= 0.01 or abs(treble_value) >= 0.01:
                        result = VoiceProfileUtils._apply_eq(result, sr, bass_value, mid_value, treble_value)
                
                # Apply modulation if specified
                if "modulation" in profile and isinstance(profile["modulation"], dict):
                    mod_rate = profile["modulation"].get("rate", 5)
                    mod_depth = profile["modulation"].get("depth", 0.3) * intensity
                    if mod_depth >= 0.01:
                        t = np.arange(len(result)) / sr
                        modulation = 1.0 + mod_depth * np.sin(2 * np.pi * mod_rate * t)
                        result = result * modulation
                
            # Simple fallback profiles if not explicitly defined
            elif profile_name == "Cinematic":
                result = VoiceProfileUtils._apply_pitch_shift(result, sr, -3 * intensity, use_gpu)
                result = VoiceProfileUtils._apply_reverb(result, sr, 0.6 * intensity)
                result = VoiceProfileUtils._apply_compression(result, 0.5 * intensity)
                result = VoiceProfileUtils._apply_eq(result, sr, 0.3 * intensity, 0.2 * intensity, 0)
            elif profile_name == "Monster":
                result = VoiceProfileUtils._apply_pitch_shift(result, sr, -6 * intensity, use_gpu)
                result = VoiceProfileUtils._apply_formant_shift(result, sr, -1.5 * intensity)
                result = VoiceProfileUtils._apply_distortion(result, 0.4 * intensity)
                result = VoiceProfileUtils._apply_eq(result, sr, 0.5 * intensity, 0, -0.2 * intensity)
            elif profile_name == "Singer":
                result = VoiceProfileUtils._apply_compression(result, 0.6 * intensity)
                result = VoiceProfileUtils._apply_eq(result, sr, 0.1 * intensity, 0.3 * intensity, 0.2 * intensity)
                result = VoiceProfileUtils._apply_reverb(result, sr, 0.3 * intensity)
            elif profile_name == "Robot":
                result = VoiceProfileUtils._apply_pitch_shift(result, sr, -2 * intensity, use_gpu)
                if SCIPY_AVAILABLE:
                    nyquist = sr / 2
                    low_freq = min(max(500 / nyquist, 0.01), 0.99)
                    high_freq = min(max(2000 / nyquist, 0.01), 0.99)
                    b, a = signal.butter(2, [low_freq, high_freq], btype='band')
                    result = signal.filtfilt(b, a, result)
                result = VoiceProfileUtils._apply_distortion(result, 0.3 * intensity)
                t = np.arange(len(result)) / sr
                modulation = 1.0 + 0.3 * intensity * np.sin(2 * np.pi * 50 * t)
                result = result * modulation
            elif profile_name == "Child":
                result = VoiceProfileUtils._apply_pitch_shift(result, sr, 3 * intensity, use_gpu)
                result = VoiceProfileUtils._apply_formant_shift(result, sr, 1.5 * intensity)
                result = VoiceProfileUtils._apply_time_stretch(result, sr, 1.1)
                result = VoiceProfileUtils._apply_eq(result, sr, -0.2 * intensity, 0.1 * intensity, 0.3 * intensity)
            
            # Preserve the overall volume
            return VoiceProfileUtils.preserve_volume(audio, result)
        
        except Exception as e:
            logger.error(f"Voice profile error for {profile_name}: {e}")
            return audio
