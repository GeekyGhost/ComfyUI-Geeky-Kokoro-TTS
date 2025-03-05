import torch
import numpy as np
import soundfile as sf
import threading
import time
import os
import sys
from functools import lru_cache

# Try to import audio processing libraries, with fallbacks
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available, using fallback implementations")

try:
    import resampy
    RESAMPY_AVAILABLE = True
except ImportError:
    RESAMPY_AVAILABLE = False
    print("Warning: resampy not available, using fallback implementations")

try:
    import scipy.signal as signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using basic fallback implementations")

# Import our fallback implementations
try:
    from .audio_utils import (simple_pitch_shift, stft_phase_vocoder, formant_shift_basic, 
                             stft, istft, amplitude_to_db, db_to_amplitude)
    FALLBACKS_AVAILABLE = True
except ImportError:
    FALLBACKS_AVAILABLE = False
    print("Warning: audio_utils.py not found, some effects may not work")

class GeekyKokoroAdvancedVoiceNode:
    """
    Advanced ComfyUI node for professional-quality voice modification with a comprehensive
    suite of effects, voice morphing capabilities, and improved algorithms.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "effect_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            },
            "optional": {
                # Voice Character Controls
                "voice_morph": (["None", "Child", "Teen", "Adult", "Elder", 
                                 "Feminine", "Masculine", "Neutral", "Androgynous"], {"default": "None"}),
                "morph_intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                
                # Pitch and Formant Controls
                "enable_pitch_formant": ("BOOLEAN", {"default": False}),
                "pitch_shift": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5, "display": "slider"}),
                "formant_shift": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1, "display": "slider"}),
                "auto_tune": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                
                # Time Controls
                "enable_time": ("BOOLEAN", {"default": False}),
                "time_stretch": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "display": "slider"}),
                "vibrato": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "vibrato_speed": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 8.0, "step": 0.1}),
                
                # Spatial Controls
                "enable_spatial": ("BOOLEAN", {"default": False}),
                "reverb_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "reverb_room_size": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reverb_damping": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "echo_delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "echo_feedback": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.9, "step": 0.01}),
                
                # Tone Controls
                "enable_tone": ("BOOLEAN", {"default": False}),
                "bass_boost": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "mid_boost": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "treble_boost": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "harmonics": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                
                # Effects Controls
                "enable_effects": ("BOOLEAN", {"default": False}),
                "distortion": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "tremolo": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "bitcrush": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "noise_reduction": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                
                # Compression and Limiting
                "enable_dynamics": ("BOOLEAN", {"default": False}),
                "compression": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "limit_ceiling": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "warmth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                
                # Character Effects
                "character_effect": (["None", "Robot", "Telephone", "Megaphone", "Radio", "Underwater", 
                                      "Cosmic", "Whisper", "ASMR", "Demon", "Angel", "Alien", "8-bit Game"], 
                                     {"default": "None"}),
                "character_intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                
                # Preset System
                "preset": (["None", "Chipmunk", "Deep Voice", "Helium", "Phone Call", "Child Voice", 
                            "Elder Voice", "Robot Voice", "Ethereal", "Monster", "Ghost", "Fantasy Elf", 
                            "Dwarf", "Orc", "Celestial", "Demonic", "Radio Host", "Podcast", "Movie Trailer", 
                            "Storyteller", "Singer"], 
                           {"default": "None"}),
                "preset_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_voice"
    CATEGORY = "audio"
    
    # ============ PITCH AND FORMANT PROCESSING ============
    
    @staticmethod
    def _apply_pitch_shift(audio, sr, n_steps, quality='high'):
        """Enhanced pitch shifting with fallbacks for when resampy isn't available"""
        if abs(n_steps) < 0.1:
            return audio
        
        try:
            # Check if we have librosa with resampy
            if LIBROSA_AVAILABLE and RESAMPY_AVAILABLE:
                # High quality pitch shifting with resampy
                try:
                    audio_out = librosa.effects.pitch_shift(
                        audio, 
                        sr=sr, 
                        n_steps=n_steps, 
                        bins_per_octave=24 if quality == 'high' else 12,
                        res_type='kaiser_best' if quality == 'high' else 'kaiser_fast'
                    )
                    return audio_out
                except Exception as e:
                    print(f"Librosa+resampy pitch shift error: {e}")
                    # Continue to fallbacks
            
            # Check if we have librosa without resampy
            if LIBROSA_AVAILABLE:
                try:
                    # Force librosa to use its internal implementation
                    audio_out = librosa.effects.pitch_shift(
                        audio,
                        sr=sr,
                        n_steps=n_steps,
                        bins_per_octave=12,
                        res_type='fft'  # Use FFT-based resampling
                    )
                    return audio_out
                except Exception as e:
                    print(f"Librosa-only pitch shift error: {e}")
                    # Continue to fallbacks
            
            # Check if we have scipy and our custom phase vocoder
            if FALLBACKS_AVAILABLE and SCIPY_AVAILABLE:
                try:
                    # Use our custom phase vocoder implementation
                    return stft_phase_vocoder(audio, sr, n_steps)
                except Exception as e:
                    print(f"STFT phase vocoder pitch shift error: {e}")
                    # Continue to simplest fallback
            
            # Simplest fallback using basic resampling
            if FALLBACKS_AVAILABLE:
                try:
                    return simple_pitch_shift(audio, sr, n_steps)
                except Exception as e:
                    print(f"Simple pitch shift error: {e}")
                    # Last resort: return original audio
                    return audio
            
            # If fallbacks aren't available or all else fails
            print("No pitch shifting implementations available")
            return audio
            
        except Exception as e:
            print(f"Pitch shift error: {e}")
            return audio
    
    @staticmethod
    def _apply_formant_shift(audio, sr, shift_amount):
        """Improved formant shifting with fallbacks"""
        if abs(shift_amount) < 0.1:
            return audio
        
        try:
            # Check if we have librosa
            if LIBROSA_AVAILABLE:
                try:
                    # Extract spectral envelope using higher resolution STFT
                    n_fft = 2048
                    hop_length = n_fft // 4
                    
                    # Compute STFT
                    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
                    S_mag, S_phase = librosa.magphase(S)
                    
                    # Apply formant shift
                    freq_bins = S_mag.shape[0]
                    shift_bins = int(freq_bins * shift_amount * 0.1)
                    
                    # Initialize shifted magnitude spectrogram
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
                    
                    # Preserve original phase information for more natural sound
                    S_shifted = S_mag_shifted * np.exp(1j * np.angle(S))
                    
                    # Convert back to time domain
                    y_shifted = librosa.istft(S_shifted, hop_length=hop_length, length=len(audio))
                    
                    return y_shifted
                except Exception as e:
                    print(f"Librosa formant shift error: {e}")
                    # Continue to fallbacks
            
            # Check if we have our custom formant shifting implementation
            if FALLBACKS_AVAILABLE:
                try:
                    return formant_shift_basic(audio, sr, shift_amount)
                except Exception as e:
                    print(f"Basic formant shift error: {e}")
                    return audio
            
            # If all else fails
            print("No formant shifting implementations available")
            return audio
            
        except Exception as e:
            print(f"Formant shift error: {e}")
            return audio
    
    @staticmethod
    def _apply_auto_tune(audio, sr, strength):
        """Apply auto-tune effect with fallbacks"""
        if strength < 0.01:
            return audio
            
        try:
            # Most basic version that works without dependencies
            # Segment-based pitch quantization
            if not LIBROSA_AVAILABLE:
                print("Auto-tune requires librosa, using minimal effect")
                # Apply a very slight chorus effect as a minimal substitute
                delay_samples = int(sr * 0.02)  # 20ms delay
                if delay_samples >= len(audio):
                    return audio
                    
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples]
                
                # Mix original and delayed with strength factor
                result = audio * (1 - strength * 0.3) + delayed * (strength * 0.3)
                return result
            
            # Librosa-based pitch detection and correction
            try:
                # Extract pitch
                hop_length = 512
                fmin = 65.0  # C2 approx
                fmax = 2093.0  # C7 approx
                
                # Get pitch contour using safer calls
                try:
                    f0, voiced_flag, _ = librosa.pyin(
                        audio, 
                        fmin=fmin,
                        fmax=fmax,
                        sr=sr,
                        hop_length=hop_length
                    )
                except Exception as e:
                    print(f"Pitch extraction error: {e}")
                    return audio
                
                # Only process if we have valid pitch information
                if f0 is None or np.all(np.isnan(f0)):
                    return audio
                    
                # Replace NaN values with 0 for unvoiced frames
                f0 = np.nan_to_num(f0)
                
                # Quantize to semitones
                tuned_f0 = f0.copy()
                
                # Only tune voiced frames
                voiced_mask = ~np.isnan(f0)
                if np.any(voiced_mask):
                    try:
                        # Convert Hz to midi notes
                        midi_notes = librosa.hz_to_midi(f0[voiced_mask])
                        
                        # Quantize midi notes to the nearest semitone
                        quantized_notes = np.round(midi_notes)
                        
                        # Convert back to Hz
                        tuned_hz = librosa.midi_to_hz(quantized_notes)
                        
                        # Apply strength factor to blend between original and tuned pitch
                        tuned_f0[voiced_mask] = f0[voiced_mask] * (1 - strength) + tuned_hz * strength
                    except Exception as e:
                        print(f"Auto-tune conversion error: {e}")
                        return audio
                
                # Simplified segment-based implementation for compatibility
                output = np.zeros_like(audio)
                segment_length = hop_length * 2
                num_segments = len(audio) // segment_length
                
                for i in range(num_segments):
                    start = i * segment_length
                    end = start + segment_length
                    center_idx = (i * 2) + 1  # Corresponding f0 index
                    
                    if center_idx < len(f0) and center_idx < len(tuned_f0):
                        # Get original and tuned pitch at this point
                        orig_pitch = f0[center_idx]
                        tuned_pitch = tuned_f0[center_idx]
                        
                        # Only process if we have valid pitch values
                        if orig_pitch > 0 and tuned_pitch > 0:
                            # Calculate shift in semitones
                            shift_amount = 12 * np.log2(tuned_pitch / orig_pitch)
                            
                            # Apply pitch shift to this segment
                            if abs(shift_amount) > 0.01:
                                try:
                                    segment = audio[start:end]
                                    shifted_segment = GeekyKokoroAdvancedVoiceNode._apply_pitch_shift(
                                        segment, sr, shift_amount
                                    )
                                    output[start:end] = shifted_segment
                                except Exception:
                                    output[start:end] = audio[start:end]
                            else:
                                output[start:end] = audio[start:end]
                        else:
                            output[start:end] = audio[start:end]
                    else:
                        output[start:end] = audio[start:end]
                
                # Handle any remaining samples
                if len(audio) > num_segments * segment_length:
                    output[num_segments * segment_length:] = audio[num_segments * segment_length:]
                    
                return output
            except Exception as e:
                print(f"Auto-tune error: {e}")
                return audio
                
        except Exception as e:
            print(f"Auto-tune error (non-critical): {e}")
            return audio
    
    # ============ TIME-BASED EFFECTS ============
    
    @staticmethod
    def _apply_time_stretch(audio, sr, rate):
        """Apply time stretching with fallbacks"""
        if abs(rate - 1.0) < 0.01:
            return audio
            
        try:
            # Check if we have librosa
            if LIBROSA_AVAILABLE:
                try:
                    # Use librosa's time_stretch
                    stretched = librosa.effects.time_stretch(audio, rate=rate)
                    
                    # Handle length differences
                    if len(stretched) > len(audio):
                        stretched = stretched[:len(audio)]
                    elif len(stretched) < len(audio):
                        stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
                        
                    return stretched
                except Exception as e:
                    print(f"Librosa time stretch error: {e}")
                    # Continue to fallbacks
            
            # Fallback implementation using resampling
            # We'll use a simple method that works with just NumPy
            indices = np.arange(0, len(audio), rate)
            indices = indices[indices < len(audio)]
            stretched = np.interp(indices, np.arange(len(audio)), audio)
            
            # Resample back to original length if needed
            if len(stretched) != len(audio):
                indices = np.linspace(0, len(stretched) - 1, len(audio))
                stretched = np.interp(indices, np.arange(len(stretched)), stretched)
            
            return stretched
                
        except Exception as e:
            print(f"Time stretch error: {e}")
            return audio
    
    @staticmethod
    def _apply_vibrato(audio, sr, depth, speed=5.0):
        """Apply vibrato effect with fallbacks"""
        if depth < 0.01:
            return audio
            
        try:
            # Create time array
            hop_length = 512
            time_arr = np.arange(len(audio)) / sr
            
            # Create modulation signal - sine wave with given speed (Hz)
            mod_signal = depth * np.sin(2 * np.pi * speed * time_arr)
            
            # Convert modulation to pitch shift amounts (in semitones)
            max_semitones = 1.0  # Maximum semitone shift
            pitch_shifts = mod_signal * max_semitones
            
            # Apply variable pitch shifting through the audio
            # We'll process in short segments
            segment_length = hop_length * 4
            num_segments = len(audio) // segment_length
            
            result = np.zeros_like(audio)
            
            # Process each segment with a different pitch shift
            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length
                
                # Use the pitch shift at the center of this segment
                center_idx = start + segment_length // 2
                if center_idx >= len(pitch_shifts):
                    center_idx = len(pitch_shifts) - 1
                    
                shift_amount = pitch_shifts[center_idx]
                
                segment = audio[start:end]
                
                # Skip very small segments
                if len(segment) < hop_length:
                    result[start:end] = segment
                    continue
                    
                # Apply pitch shift to this segment
                try:
                    shifted_segment = GeekyKokoroAdvancedVoiceNode._apply_pitch_shift(
                        segment, sr, shift_amount
                    )
                    
                    # Handle potential length mismatch
                    if len(shifted_segment) > len(segment):
                        shifted_segment = shifted_segment[:len(segment)]
                    elif len(shifted_segment) < len(segment):
                        shifted_segment = np.pad(shifted_segment, (0, len(segment) - len(shifted_segment)), mode='constant')
                    
                    result[start:end] = shifted_segment
                except Exception:
                    # If pitch shift fails, use original segment
                    result[start:end] = segment
                
            # Handle the remaining samples
            if len(audio) > num_segments * segment_length:
                start = num_segments * segment_length
                result[start:] = audio[start:]
                
            return result
        except Exception as e:
            print(f"Vibrato error: {e}")
            return audio
    
    # ============ SPATIAL EFFECTS ============
    
    @staticmethod
    def _apply_reverb(audio, sr, amount, room_size=0.5, damping=0.5):
        """Enhanced reverb with room size and damping controls"""
        if amount < 0.01:
            return audio
            
        try:
            # Calculate impulse response based on parameters
            reverb_length = int(sr * (0.1 + room_size * 1.5))  # 0.1s to 1.6s
            decay_factor = 0.2 + damping * 0.7  # 0.2 to 0.9
            
            # Create impulse response
            impulse = np.zeros(reverb_length)
            impulse[0] = 1.0  # Delta function at the start
            
            # Early reflections (more for larger rooms)
            num_early = int(5 + room_size * 10)
            early_times = np.sort(np.random.randint(1, reverb_length // 3, num_early))
            early_amps = np.random.uniform(0.1, 0.4, num_early) * (1 - damping * 0.5)
            for time, amp in zip(early_times, early_amps):
                impulse[time] = amp
            
            # Exponential decay for the tail
            for i in range(1, reverb_length):
                impulse[i] += np.random.randn() * np.exp(-i / (sr * decay_factor))
            
            # Normalize the impulse response
            impulse /= np.max(np.abs(impulse))
            
            # Apply the reverb using convolution
            if SCIPY_AVAILABLE:
                try:
                    reverb_signal = signal.convolve(audio, impulse, mode='full')[:len(audio)]
                except Exception as e:
                    print(f"Reverb convolution error: {e}")
                    # Fallback method
                    reverb_signal = np.zeros_like(audio)
                    reverb_signal[0] = audio[0]
                    for i in range(1, len(audio)):
                        reverb_signal[i] = audio[i]
                        # Add delayed and decayed versions of earlier samples
                        for j in range(1, min(i, 5000)):
                            reverb_signal[i] += audio[i-j] * impulse[min(j, len(impulse)-1)]
            else:
                # Basic convolution without scipy
                reverb_signal = np.zeros_like(audio)
                for i in range(len(audio)):
                    for j in range(min(i + 1, reverb_length)):
                        if i - j >= 0:
                            reverb_signal[i] += audio[i - j] * impulse[j]
            
            # Mix original and reverberant signal based on amount
            output = (1 - amount) * audio + amount * reverb_signal
            
            return output
        except Exception as e:
            print(f"Reverb effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_echo(audio, sr, amount, feedback=0.3):
        """Add echo effect with feedback control"""
        if amount < 0.01:
            return audio
            
        try:
            # Calculate delay in samples
            delay_time = 0.1 + amount * 0.4  # 0.1s to 0.5s
            delay_samples = int(sr * delay_time)
            
            # Create the delayed signal
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples]
            
            # Apply feedback
            result = audio.copy()
            
            # Apply multiple echo taps with decreasing volume
            echo_signal = delayed * amount
            result += echo_signal
            
            # Add feedback delays
            for i in range(1, 5):  # Up to 5 feedback taps
                tap_gain = amount * (feedback ** i)
                if tap_gain < 0.01:
                    break
                    
                # Shift the echo further in time
                tap_delay = delay_samples * i
                if tap_delay >= len(audio):
                    break
                    
                tap_signal = np.zeros_like(audio)
                tap_signal[tap_delay:] = audio[:-tap_delay]
                
                # Add this tap to the result
                result += tap_signal * tap_gain
            
            # Prevent clipping
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result /= max_val
                
            return result
        except Exception as e:
            print(f"Echo effect error: {e}")
            return audio
    
    # ============ TONE SHAPING ============
    
    @staticmethod
    def _apply_eq(audio, sr, bass=0.0, mid=0.0, treble=0.0):
        """Apply 3-band equalizer with bass, mid, and treble controls"""
        if abs(bass) < 0.01 and abs(mid) < 0.01 and abs(treble) < 0.01:
            return audio
            
        try:
            # Define the frequency ranges - ensure values are normalized correctly
            bass_cutoff = 250    # Hz
            mid_center = 1000    # Hz
            treble_cutoff = 4000 # Hz
            
            if SCIPY_AVAILABLE:
                # Convert to normalized frequencies (must be 0 < norm < 1)
                nyquist = sr / 2
                bass_norm = min(max(bass_cutoff / nyquist, 0.001), 0.999)
                mid_low_norm = min(max(bass_cutoff / nyquist, 0.001), 0.999)
                mid_high_norm = min(max(treble_cutoff / nyquist, 0.001), 0.999)
                treble_norm = min(max(treble_cutoff / nyquist, 0.001), 0.999)
                
                # Ensure mid_low_norm < mid_high_norm
                if mid_low_norm >= mid_high_norm:
                    mid_low_norm = mid_high_norm * 0.5
                
                # Process the audio in three bands
                # Low-pass for bass
                try:
                    b_bass, a_bass = signal.butter(2, bass_norm, btype='low')
                    bass_filtered = signal.lfilter(b_bass, a_bass, audio)
                except Exception as e:
                    print(f"Bass filter error: {e}")
                    bass_filtered = audio
                
                # Band-pass for mids
                try:
                    b_mid, a_mid = signal.butter(2, [mid_low_norm, mid_high_norm], btype='band')
                    mid_filtered = signal.lfilter(b_mid, a_mid, audio)
                except Exception as e:
                    print(f"Mid filter error: {e}")
                    mid_filtered = audio
                
                # High-pass for treble
                try:
                    b_treble, a_treble = signal.butter(2, treble_norm, btype='high')
                    treble_filtered = signal.lfilter(b_treble, a_treble, audio)
                except Exception as e:
                    print(f"Treble filter error: {e}")
                    treble_filtered = audio
                
                # Apply gains to each band
                # Convert from -1..1 to gain factors (0.5..2.0)
                bass_gain = 10 ** (bass * 0.3)    # -6dB to +6dB
                mid_gain = 10 ** (mid * 0.2)      # -4dB to +4dB
                treble_gain = 10 ** (treble * 0.25) # -5dB to +5dB
                
                # Mix the outputs based on gains
                result = (bass_filtered * bass_gain + 
                         mid_filtered * mid_gain + 
                         treble_filtered * treble_gain) / 3
                
                return result
            else:
                # Very basic EQ without scipy using FFT-based filtering
                # We'll do a simple frequency domain modification
                n = len(audio)
                fft = np.fft.rfft(audio)
                freq = np.fft.rfftfreq(n, 1/sr)
                
                # Create gain mask based on frequency bands
                gain_mask = np.ones(len(freq))
                
                # Apply bass boost/cut
                bass_mask = (freq < bass_cutoff)
                gain_mask[bass_mask] *= (1.0 + bass * 0.5)
                
                # Apply mid boost/cut
                mid_mask = (freq >= bass_cutoff) & (freq < treble_cutoff)
                gain_mask[mid_mask] *= (1.0 + mid * 0.3)
                
                # Apply treble boost/cut
                treble_mask = (freq >= treble_cutoff)
                gain_mask[treble_mask] *= (1.0 + treble * 0.4)
                
                # Apply gain mask to FFT
                fft_eq = fft * gain_mask
                
                # Inverse FFT
                audio_eq = np.fft.irfft(fft_eq, n)
                
                return audio_eq
                     
        except Exception as e:
            print(f"EQ effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_harmonics(audio, sr, amount):
        """Add harmonics to enhance the voice"""
        if amount < 0.01:
            return audio
            
        try:
            # Simple harmonics generation without relying on pitch detection
            # Generate harmonic signals using waveshaping
            
            # Waveshaping for 2nd harmonic - squared function
            h2 = np.sign(audio) * (audio ** 2)
            
            # Waveshaping for 3rd harmonic - cubed function 
            h3 = audio ** 3
            
            # Normalize harmonics
            if np.max(np.abs(h2)) > 0:
                h2 = h2 / np.max(np.abs(h2))
            
            if np.max(np.abs(h3)) > 0:
                h3 = h3 / np.max(np.abs(h3))
            
            # Mix with original based on amount
            harmonic_mix = audio.copy()
            harmonic_mix += h2 * amount * 0.3  # Add 2nd harmonic
            harmonic_mix += h3 * amount * 0.15  # Add 3rd harmonic
            
            # Normalize the result
            if np.max(np.abs(harmonic_mix)) > 0:
                harmonic_mix = harmonic_mix / np.max(np.abs(harmonic_mix)) * np.max(np.abs(audio))
            
            # Mix with original based on amount
            result = audio * (1 - amount * 0.4) + harmonic_mix * amount * 0.4
            
            return result
        except Exception as e:
            print(f"Harmonics effect error: {e}")
            return audio
    
    # ============ EFFECTS ============
    
    @staticmethod
    def _apply_distortion(audio, effect_strength):
        """Improved distortion with smoother curve"""
        if effect_strength < 0.01:
            return audio
            
        try:
            # Create a more musical distortion curve
            # Mix between different distortion types
            drive = 1 + effect_strength * 15  # 1 to 16
            
            # Soft clipping (tanh)
            soft_clip = np.tanh(audio * drive)
            
            # Hard clipping
            threshold = 1 / drive
            hard_clip = np.clip(audio * drive, -threshold, threshold)
            
            # Tube-like asymmetric distortion
            tube_clip = np.copy(audio)
            positive_mask = tube_clip > 0
            tube_clip[positive_mask] = 1.0 - np.exp(-tube_clip[positive_mask] * drive)
            tube_clip[~positive_mask] = -1.0 + np.exp(tube_clip[~positive_mask] * drive)
            
            # Mix the distortion types based on intensity
            if effect_strength < 0.3:
                # Subtle distortion - mostly soft clip
                dist_mix = soft_clip
            elif effect_strength < 0.7:
                # Medium distortion - mix of soft and tube
                mix_ratio = (effect_strength - 0.3) / 0.4
                dist_mix = soft_clip * (1 - mix_ratio) + tube_clip * mix_ratio
            else:
                # Heavy distortion - mix in some hard clipping
                mix_ratio = (effect_strength - 0.7) / 0.3
                dist_mix = tube_clip * (1 - mix_ratio) + hard_clip * mix_ratio
            
            # Normalize the output
            if np.max(np.abs(dist_mix)) > 0:
                dist_mix = dist_mix / np.max(np.abs(dist_mix)) * 0.95
                
            # Mix with original for smoother transition
            result = audio * (1 - effect_strength) + dist_mix * effect_strength
            
            return result
        except Exception as e:
            print(f"Distortion error: {e}")
            return audio
    
    @staticmethod
    def _apply_tremolo(audio, effect_strength, sr, speed=5.0):
        """Apply tremolo (amplitude modulation)"""
        if effect_strength < 0.01:
            return audio
            
        try:
            # Create the tremolo wave (sine wave amplitude modulation)
            t = np.arange(len(audio)) / sr
            mod_depth = effect_strength * 0.9  # 0 to 0.9
            
            # Create modulation signal - sine wave
            tremolo_wave = 1.0 - mod_depth + mod_depth * np.sin(2 * np.pi * speed * t)
            
            # Apply the tremolo
            tremolo_audio = audio * tremolo_wave
            
            return tremolo_audio
        except Exception as e:
            print(f"Tremolo error: {e}")
            return audio
    
    @staticmethod
    def _apply_bitcrush(audio, intensity):
        """Apply bit-depth reduction for lo-fi effect"""
        if intensity < 0.01:
            return audio
            
        try:
            # Map intensity to bit depth (16 bits down to 2 bits)
            min_bits = 2
            max_bits = 16
            bits = max_bits - intensity * (max_bits - min_bits)
            bits = int(bits)
            
            # Apply bit depth reduction
            # This simulates reducing the bit depth by quantizing the amplitude
            steps = 2**bits
            audio_crushed = np.round(audio * (steps/2)) / (steps/2)
            
            # Mix with original based on intensity
            result = audio * (1 - intensity) + audio_crushed * intensity
            
            return result
        except Exception as e:
            print(f"Bitcrush error: {e}")
            return audio
    
    @staticmethod
    def _apply_noise_reduction(audio, amount):
        """Apply simple noise reduction with fallbacks"""
        if amount < 0.01:
            return audio
            
        try:
            # Check if we have librosa for spectral processing
            if LIBROSA_AVAILABLE:
                try:
                    # Simple spectral gating noise reduction
                    # Convert to spectrum
                    S = librosa.stft(audio)
                    S_mag, S_phase = librosa.magphase(S)
                    
                    # Estimate noise floor
                    # Assume the quietest 5% of the spectrum is noise
                    mag_db = librosa.amplitude_to_db(S_mag)
                    noise_threshold_db = np.percentile(mag_db, 5)
                    
                    # Set threshold based on amount
                    threshold_db = noise_threshold_db + 20 * (1 - amount)  # Lower the threshold as amount increases
                    
                    # Apply spectral gating
                    mask = mag_db > threshold_db
                    
                    # Smooth the mask to prevent musical noise
                    mask_smooth = np.zeros_like(mask, dtype=float)
                    for i in range(mask.shape[1]):
                        if SCIPY_AVAILABLE:
                            try:
                                mask_smooth[:, i] = signal.medfilt(mask[:, i].astype(float), 3)
                            except Exception:
                                mask_smooth[:, i] = mask[:, i].astype(float)
                        else:
                            mask_smooth[:, i] = mask[:, i].astype(float)
                        
                    # Create soft mask
                    gain_db = (mag_db - threshold_db)
                    gain_db = np.maximum(-80, gain_db)  # Limit to -80dB
                    gain_db = gain_db * amount  # Apply amount
                    gain_linear = librosa.db_to_amplitude(gain_db)
                    
                    # Apply the soft mask
                    S_mag_clean = S_mag * gain_linear
                    
                    # Reconstruct the audio
                    S_clean = S_mag_clean * S_phase
                    audio_clean = librosa.istft(S_clean, length=len(audio))
                    
                    return audio_clean
                except Exception as e:
                    print(f"Spectral processing error: {e}")
                    # Fall back to simple envelope-based noise reduction
            
            # Simple envelope-based noise reduction as fallback
            # Calculate signal envelope
            abs_audio = np.abs(audio)
            
            # Smooth the envelope
            window_size = min(int(sr / 50), len(abs_audio) // 10)  # ~20ms window
            if window_size < 1:
                window_size = 1
                
            envelope = np.zeros_like(abs_audio)
            for i in range(len(abs_audio)):
                start = max(0, i - window_size)
                end = min(len(abs_audio), i + window_size)
                envelope[i] = np.max(abs_audio[start:end])
            
            # Calculate noise floor as the 10th percentile
            noise_floor = np.percentile(envelope, 10)
            
            # Apply soft noise gate
            gain = 1.0 - amount * (noise_floor / (envelope + 1e-10))
            gain = np.clip(gain, 0.0, 1.0)
            
            # Apply gain
            result = audio * gain
            
            return result
                
        except Exception as e:
            print(f"Noise reduction error: {e}")
            return audio
    
    # ============ DYNAMICS PROCESSING ============
    
    @staticmethod
    def _apply_compression(audio, amount, limit_ceiling=0.95):
        """Apply compression with variable ratio and smart makeup gain"""
        if amount < 0.01:
            return audio
            
        try:
            # Compression parameters based on amount
            # Map amount to ratio (1:1 to 8:1)
            ratio = 1 + 7 * amount
            
            # Threshold depends on the audio level
            rms = np.sqrt(np.mean(audio**2))
            db_rms = 20 * np.log10(max(rms, 1e-8))
            
            # Set threshold based on RMS level and amount
            threshold_db = db_rms - 6 - amount * 12  # -6dB to -18dB below RMS
            threshold = 10 ** (threshold_db / 20)
            
            # Attack and release times (ms)
            attack_ms = 20 - amount * 15  # 20ms to 5ms
            release_ms = 150 + amount * 350  # 150ms to 500ms
            
            # Convert to samples
            attack_samples = int(attack_ms * sr / 1000)
            release_samples = int(release_ms * sr / 1000)
            
            # Ensure we don't divide by zero
            if attack_samples < 1:
                attack_samples = 1
            if release_samples < 1:
                release_samples = 1
            
            # Apply compression
            result = np.zeros_like(audio)
            
            # Start with gain at 1.0
            gain = 1.0
            
            # Process sample by sample
            for i in range(len(audio)):
                # Calculate the absolute value of the current sample
                abs_sample = abs(audio[i])
                
                # Check if we need to compress
                if abs_sample > threshold:
                    # Calculate target gain
                    # Above threshold, apply ratio
                    above_threshold = abs_sample - threshold
                    compressed_above = above_threshold / ratio
                    target_gain = (threshold + compressed_above) / abs_sample
                else:
                    # Below threshold, no compression
                    target_gain = 1.0
                
                # Apply attack/release
                if target_gain < gain:
                    # Attack phase - gain is decreasing
                    gain = gain + (target_gain - gain) / attack_samples
                else:
                    # Release phase - gain is increasing
                    gain = gain + (target_gain - gain) / release_samples
                
                # Apply the gain
                result[i] = audio[i] * gain
            
            # Apply makeup gain to bring back overall level
            # Calculate RMS before and after compression
            rms_before = np.sqrt(np.mean(audio**2))
            rms_after = np.sqrt(np.mean(result**2))
            
            # Calculate makeup gain in dB
            if rms_after > 0:
                makeup_db = 20 * np.log10(max(rms_before, 1e-8)) - 20 * np.log10(max(rms_after, 1e-8))
                makeup_gain = 10 ** (makeup_db / 20)
            else:
                makeup_gain = 1.0
            
            # Apply makeup gain
            result *= makeup_gain
            
            # Apply ceiling limiting
            if np.max(np.abs(result)) > limit_ceiling:
                result = result / np.max(np.abs(result)) * limit_ceiling
                
            return result
        except Exception as e:
            print(f"Compression error: {e}")
            return audio
    
    @staticmethod
    def _apply_warmth(audio, amount):
        """Add analog warmth effect"""
        if amount < 0.01:
            return audio
            
        try:
            # Apply subtle even harmonic distortion for warmth
            warmth = audio.copy()
            
            # Create even harmonics (similar to tube saturation)
            # Second harmonic distortion
            h2 = audio**2 * np.sign(audio)
            h2 = h2 / np.max(np.abs(h2)) if np.max(np.abs(h2)) > 0 else h2
            
            # Fourth harmonic
            h4 = audio**4 * np.sign(audio)
            h4 = h4 / np.max(np.abs(h4)) if np.max(np.abs(h4)) > 0 else h4
            
            # Mix in the harmonics based on amount
            warmth = audio * (1 - amount * 0.4) + h2 * (amount * 0.35) + h4 * (amount * 0.05)
            
            # Add subtle low-frequency enhancement using a basic low-pass filter
            if SCIPY_AVAILABLE:
                try:
                    # Low shelf EQ boost for warmth using normal filter
                    nyquist = sr / 2
                    low_freq = min(200 / nyquist, 0.9)  # Ensure it's < 1.0
                    
                    # Apply standard lowpass filter with boost
                    b, a = signal.butter(2, low_freq, btype='low')
                    low_end = signal.lfilter(b, a, warmth)
                    
                    # Mix boosted low end with original
                    boost_amount = 0.2 * amount
                    warmth = warmth * (1 - boost_amount) + low_end * (1 + boost_amount)
                except Exception as e:
                    print(f"Warmth EQ error: {e}")
            else:
                # Simple FFT-based low boost
                n = len(audio)
                fft = np.fft.rfft(warmth)
                freq = np.fft.rfftfreq(n, 1/sr)
                
                # Boost frequencies below 200Hz
                boost_mask = freq < 200
                boost_amount = 1.0 + amount * 0.3
                fft[boost_mask] *= boost_amount
                
                # Inverse FFT
                warmth = np.fft.irfft(fft, n)
            
            # Apply very gentle soft clipping to add saturation
            clip_amount = 0.8 + 0.2 * (1 - amount)  # 0.8 to 1.0
            warmth = np.tanh(warmth / clip_amount) * clip_amount
            
            return warmth
        except Exception as e:
            print(f"Warmth effect error: {e}")
            return audio
    
    # ============ CHARACTER EFFECTS ============
    
    @staticmethod
    def _apply_character_effect(audio, sr, effect_name, intensity):
        """Apply special vocal character effects with fallbacks"""
        if effect_name == "None" or intensity < 0.01:
            return audio
            
        try:
            result = audio.copy()
            
            if effect_name == "Robot":
                # Robot voice effect - simplified implementation
                # Apply a bandpass filter for a "robotic" sound
                if SCIPY_AVAILABLE:
                    try:
                        nyquist = sr / 2
                        low_freq = 500 / nyquist
                        high_freq = 2000 / nyquist
                        
                        # Ensure frequencies are normalized correctly
                        low_freq = min(max(low_freq, 0.01), 0.99)
                        high_freq = min(max(high_freq, 0.01), 0.99)
                        
                        # Make sure low_freq < high_freq
                        if low_freq >= high_freq:
                            low_freq = high_freq * 0.5
                        
                        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
                        robot = signal.lfilter(b, a, audio)
                    except Exception as e:
                        print(f"Robot filter error: {e}")
                        # Fallback to FFT-based filter
                        n = len(audio)
                        fft = np.fft.rfft(audio)
                        freq = np.fft.rfftfreq(n, 1/sr)
                        
                        mask = (freq >= 500) & (freq <= 2000)
                        fft_filtered = np.zeros_like(fft)
                        fft_filtered[mask] = fft[mask]
                        
                        robot = np.fft.irfft(fft_filtered, n)
                else:
                    # Fallback to FFT-based filter
                    n = len(audio)
                    fft = np.fft.rfft(audio)
                    freq = np.fft.rfftfreq(n, 1/sr)
                    
                    mask = (freq >= 500) & (freq <= 2000)
                    fft_filtered = np.zeros_like(fft)
                    fft_filtered[mask] = fft[mask]
                    
                    robot = np.fft.irfft(fft_filtered, n)
                
                # Add amplitude modulation for a "robotic" buzziness
                t = np.arange(len(audio)) / sr
                buzz_freq = 50  # Hz
                buzz_depth = 0.3 * intensity
                buzz = 1.0 - buzz_depth + buzz_depth * np.sin(2 * np.pi * buzz_freq * t)
                
                robot = robot * buzz
                
                # Mix with original based on intensity
                result = audio * (1 - intensity) + robot * intensity
                
            elif effect_name == "Telephone":
                # Telephone effect - simplified bandpass filter
                if SCIPY_AVAILABLE:
                    try:
                        nyquist = sr / 2
                        low_freq = 300 / nyquist
                        high_freq = 3400 / nyquist
                        
                        # Ensure frequencies are normalized correctly
                        low_freq = min(max(low_freq, 0.01), 0.99)
                        high_freq = min(max(high_freq, 0.01), 0.99)
                        
                        # Make sure low_freq < high_freq
                        if low_freq >= high_freq:
                            low_freq = high_freq * 0.5
                        
                        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
                        telephone = signal.lfilter(b, a, audio)
                    except Exception as e:
                        print(f"Telephone filter error: {e}")
                        # Fallback to FFT-based filter
                        n = len(audio)
                        fft = np.fft.rfft(audio)
                        freq = np.fft.rfftfreq(n, 1/sr)
                        
                        mask = (freq >= 300) & (freq <= 3400)
                        fft_filtered = np.zeros_like(fft)
                        fft_filtered[mask] = fft[mask]
                        
                        telephone = np.fft.irfft(fft_filtered, n)
                else:
                    # Fallback to FFT-based filter
                    n = len(audio)
                    fft = np.fft.rfft(audio)
                    freq = np.fft.rfftfreq(n, 1/sr)
                    
                    mask = (freq >= 300) & (freq <= 3400)
                    fft_filtered = np.zeros_like(fft)
                    fft_filtered[mask] = fft[mask]
                    
                    telephone = np.fft.irfft(fft_filtered, n)
                
                # Add mild distortion
                telephone = np.tanh(telephone * 2) / 2
                
                # Add telephone line noise
                noise = np.random.normal(0, 0.01, len(audio)) * intensity
                telephone = telephone + noise
                
                # Mix with original based on intensity
                result = audio * (1 - intensity) + telephone * intensity
                
            elif effect_name == "Megaphone":
                # Megaphone effect - simplified implementation
                if SCIPY_AVAILABLE:
                    try:
                        nyquist = sr / 2
                        low_freq = 500 / nyquist
                        high_freq = 4000 / nyquist
                        
                        # Ensure frequencies are normalized correctly
                        low_freq = min(max(low_freq, 0.01), 0.99)
                        high_freq = min(max(high_freq, 0.01), 0.99)
                        
                        # Make sure low_freq < high_freq
                        if low_freq >= high_freq:
                            low_freq = high_freq * 0.5
                        
                        b, a = signal.butter(2, [low_freq, high_freq], btype='band')
                        megaphone = signal.lfilter(b, a, audio)
                    except Exception as e:
                        print(f"Megaphone filter error: {e}")
                        # Fallback to FFT-based filter
                        n = len(audio)
                        fft = np.fft.rfft(audio)
                        freq = np.fft.rfftfreq(n, 1/sr)
                        
                        mask = (freq >= 500) & (freq <= 4000)
                        fft_filtered = np.zeros_like(fft)
                        fft_filtered[mask] = fft[mask]
                        
                        megaphone = np.fft.irfft(fft_filtered, n)
                else:
                    # Fallback to FFT-based filter
                    n = len(audio)
                    fft = np.fft.rfft(audio)
                    freq = np.fft.rfftfreq(n, 1/sr)
                    
                    mask = (freq >= 500) & (freq <= 4000)
                    fft_filtered = np.zeros_like(fft)
                    fft_filtered[mask] = fft[mask]
                    
                    megaphone = np.fft.irfft(fft_filtered, n)
                
                # Add distortion
                megaphone = np.clip(megaphone * 2.5, -0.9, 0.9)
                
                # Mix with original based on intensity
                result = audio * (1 - intensity) + megaphone * intensity
                
            elif effect_name == "Radio":
                # Old radio effect - simplified implementation
                if SCIPY_AVAILABLE:
                    try:
                        nyquist = sr / 2
                        low_freq = 400 / nyquist
                        high_freq = 3000 / nyquist
                        
                        # Ensure frequencies are normalized correctly
                        low_freq = min(max(low_freq, 0.01), 0.99)
                        high_freq = min(max(high_freq, 0.01), 0.99)
                        
                        # Make sure low_freq < high_freq
                        if low_freq >= high_freq:
                            low_freq = high_freq * 0.5
                        
                        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
                        radio = signal.lfilter(b, a, audio)
                    except Exception as e:
                        print(f"Radio filter error: {e}")
                        # Fallback to FFT-based filter
                        n = len(audio)
                        fft = np.fft.rfft(audio)
                        freq = np.fft.rfftfreq(n, 1/sr)
                        
                        mask = (freq >= 400) & (freq <= 3000)
                        fft_filtered = np.zeros_like(fft)
                        fft_filtered[mask] = fft[mask]
                        
                        radio = np.fft.irfft(fft_filtered, n)
                else:
                    # Fallback to FFT-based filter
                    n = len(audio)
                    fft = np.fft.rfft(audio)
                    freq = np.fft.rfftfreq(n, 1/sr)
                    
                    mask = (freq >= 400) & (freq <= 3000)
                    fft_filtered = np.zeros_like(fft)
                    fft_filtered[mask] = fft[mask]
                    
                    radio = np.fft.irfft(fft_filtered, n)
                
                # Add mild distortion
                radio = np.tanh(radio * 1.5) / 1.5
                
                # Add radio static noise
                noise_level = 0.03 * intensity
                static = np.random.normal(0, noise_level, len(audio))
                
                # Modulate static amplitude
                static_env = np.abs(radio) + 0.1
                static *= static_env
                
                radio = radio + static
                
                # Add amplitude modulation for radio fade effect
                t = np.arange(len(audio)) / sr
                am_depth = 0.2 * intensity
                am_freq = 0.5  # Hz
                am = 1.0 - am_depth + am_depth * np.sin(2 * np.pi * am_freq * t)
                
                radio *= am
                
                # Mix with original based on intensity
                result = audio * (1 - intensity) + radio * intensity
                
            elif effect_name == "Underwater":
                # Underwater effect - simplified implementation
                if SCIPY_AVAILABLE:
                    try:
                        nyquist = sr / 2
                        cutoff = 800 / nyquist
                        
                        # Ensure frequency is normalized correctly
                        cutoff = min(max(cutoff, 0.01), 0.99)
                        
                        b, a = signal.butter(4, cutoff, btype='low')
                        underwater = signal.lfilter(b, a, audio)
                    except Exception as e:
                        print(f"Underwater filter error: {e}")
                        # Fallback to FFT-based filter
                        n = len(audio)
                        fft = np.fft.rfft(audio)
                        freq = np.fft.rfftfreq(n, 1/sr)
                        
                        mask = freq <= 800
                        fft_filtered = np.zeros_like(fft)
                        fft_filtered[mask] = fft[mask]
                        
                        underwater = np.fft.irfft(fft_filtered, n)
                else:
                    # Fallback to FFT-based filter
                    n = len(audio)
                    fft = np.fft.rfft(audio)
                    freq = np.fft.rfftfreq(n, 1/sr)
                    
                    mask = freq <= 800
                    fft_filtered = np.zeros_like(fft)
                    fft_filtered[mask] = fft[mask]
                    
                    underwater = np.fft.irfft(fft_filtered, n)
                
                # Apply subtle tremolo
                t = np.arange(len(audio)) / sr
                trem_depth = 0.3 * intensity
                trem_freq = 0.8  # Hz
                trem = 1.0 - trem_depth + trem_depth * np.sin(2 * np.pi * trem_freq * t)
                
                underwater_effect = underwater * trem
                
                # Mix with original based on intensity
                result = audio * (1 - intensity) + underwater_effect * intensity
                
            elif effect_name == "Whisper":
                # Whisper effect - simplified implementation
                # Reduce volume and add noise
                whisper = audio * 0.6
                
                # Reduce low frequencies
                if SCIPY_AVAILABLE:
                    try:
                        nyquist = sr / 2
                        cutoff = 600 / nyquist
                        
                        # Ensure frequency is normalized correctly
                        cutoff = min(max(cutoff, 0.01), 0.99)
                        
                        b, a = signal.butter(4, cutoff, btype='high')
                        whisper = signal.lfilter(b, a, whisper)
                    except Exception as e:
                        print(f"Whisper filter error: {e}")
                        # Fallback to FFT-based filter
                        n = len(audio)
                        fft = np.fft.rfft(whisper)
                        freq = np.fft.rfftfreq(n, 1/sr)
                        
                        mask = freq >= 600
                        fft_filtered = np.zeros_like(fft)
                        fft_filtered[mask] = fft[mask]
                        
                        whisper = np.fft.irfft(fft_filtered, n)
                else:
                    # Fallback to FFT-based filter
                    n = len(audio)
                    fft = np.fft.rfft(whisper)
                    freq = np.fft.rfftfreq(n, 1/sr)
                    
                    mask = freq >= 600
                    fft_filtered = np.zeros_like(fft)
                    fft_filtered[mask] = fft[mask]
                    
                    whisper = np.fft.irfft(fft_filtered, n)
                
                # Add breath noise
                noise_level = 0.15 * intensity
                breath = np.random.normal(0, noise_level, len(audio))
                
                # Shape noise to follow original envelope
                envelope = np.abs(whisper)
                window_size = min(int(sr * 0.01), len(envelope) // 10)
                if window_size < 1:
                    window_size = 1
                smoothed_envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
                breath *= smoothed_envelope
                
                whisper = whisper + breath
                
                # Add slight distortion to simulate air turbulence
                whisper = np.clip(whisper * 1.5, -0.9, 0.9)
                
                # Mix with original based on intensity
                result = audio * (1 - intensity) + whisper * intensity
                
            elif effect_name == "Demon":
                # Demon voice effect - simplified implementation
                # Pitch shift down using our robust method
                try:
                    demon = GeekyKokoroAdvancedVoiceNode._apply_pitch_shift(
                        audio, sr, n_steps=-5 * intensity
                    )
                except Exception as e:
                    print(f"Demon pitch shift error: {e}")
                    demon = audio
                
                # Add distortion
                demon = np.tanh(demon * (1 + 2 * intensity)) / (1 + intensity)
                
                # Add low frequency rumble
                t = np.arange(len(audio)) / sr
                rumble_freq = 40  # Hz
                rumble = np.sin(2 * np.pi * rumble_freq * t) * 0.2 * intensity
                
                demon = demon + rumble
                
                # Mix with original based on intensity
                result = audio * (1 - intensity) + demon * intensity
                
            elif effect_name == "8-bit Game":
                # 8-bit video game sound effect - simplified implementation
                # Reduce bit depth
                bit_depth = 3 + (1 - intensity) * 5  # 3 to 8 bits
                steps = 2 ** int(bit_depth)
                game = np.round(audio * (steps/2)) / (steps/2)
                
                # Reduce sample rate (without actually changing SR)
                sr_reduction = 0.1 + (1 - intensity) * 0.4  # 0.1x to 0.5x original SR
                downsample_factor = int(1 / sr_reduction)
                
                # Protect against very small downsample factors
                if downsample_factor < 1:
                    downsample_factor = 1
                
                # Downsample and then upsample to create the effect
                downsampled = np.zeros(len(audio))
                for i in range(0, len(audio), downsample_factor):
                    if i < len(audio):
                        value = game[i]
                        for j in range(min(downsample_factor, len(audio) - i)):
                            downsampled[i + j] = value
                
                game = downsampled
                
                # Mix with original based on intensity
                result = audio * (1 - intensity) + game * intensity
            
            # Normalize output
            if np.max(np.abs(result)) > 0:
                result = result / np.max(np.abs(result)) * 0.95
                
            return result
        except Exception as e:
            print(f"Character effect error: {e}")
            return audio
    
    # ============ VOICE MORPHING ============
    
    @staticmethod
    def _apply_voice_morph(audio, sr, morph_type, intensity):
        """Apply voice morphing with fallbacks"""
        if morph_type == "None" or intensity < 0.01:
            return audio
            
        try:
            result = audio.copy()
            
            # Setup morph parameters based on type
            if morph_type == "Child":
                # Child voice transformation
                pitch_shift = 3 * intensity
                formant_shift = 2 * intensity
                brightness = 0.3 * intensity
                breathiness = 0.2 * intensity
                
            elif morph_type == "Teen":
                # Teen voice transformation
                pitch_shift = 1.5 * intensity
                formant_shift = 1.0 * intensity
                brightness = 0.2 * intensity
                breathiness = 0.1 * intensity
                
            elif morph_type == "Adult":
                # Adult voice (neutral transformation - slight enhancement)
                pitch_shift = 0
                formant_shift = 0
                brightness = 0.1 * intensity
                breathiness = 0
                
            elif morph_type == "Elder":
                # Elder voice transformation
                pitch_shift = -1 * intensity
                formant_shift = -0.5 * intensity
                breathiness = 0.3 * intensity
                brightness = -0.2 * intensity
                
            elif morph_type == "Feminine":
                # Feminine voice transformation
                pitch_shift = 2 * intensity
                formant_shift = 1.5 * intensity
                brightness = 0.2 * intensity
                breathiness = 0.2 * intensity
                
            elif morph_type == "Masculine":
                # Masculine voice transformation
                pitch_shift = -2 * intensity
                formant_shift = -1.5 * intensity
                brightness = -0.1 * intensity
                breathiness = -0.1 * intensity
                
            elif morph_type == "Neutral":
                # Gender-neutral voice - simplified
                pitch_shift = 0  # Start from neutral point without analysis
                formant_shift = 0
                brightness = 0
                breathiness = 0.1 * intensity
                
            elif morph_type == "Androgynous":
                # Androgynous voice - simplified
                pitch_shift = 0  # Similar to neutral without analysis
                formant_shift = 0
                brightness = 0.1 * intensity
                breathiness = 0.15 * intensity
            else:
                # Default/fallback
                pitch_shift = 0
                formant_shift = 0
                brightness = 0
                breathiness = 0
            
            # Apply the morph effects in sequence
            
            # 1. Apply pitch shifting
            if abs(pitch_shift) >= 0.1:
                try:
                    result = GeekyKokoroAdvancedVoiceNode._apply_pitch_shift(
                        result, sr, n_steps=pitch_shift, quality='high'
                    )
                except Exception as e:
                    print(f"Voice morph pitch shift error: {e}")
            
            # 2. Apply formant shifting
            if abs(formant_shift) >= 0.1:
                try:
                    result = GeekyKokoroAdvancedVoiceNode._apply_formant_shift(
                        result, sr, shift_amount=formant_shift
                    )
                except Exception as e:
                    print(f"Voice morph formant shift error: {e}")
            
            # 3. Apply brightness adjustment (EQ)
            if abs(brightness) >= 0.05:
                try:
                    # Brightness is primarily controlled by high frequencies
                    # Use our simplified EQ
                    bass = 0  # Neutral bass
                    mid = brightness * 0.5  # Slight mid adjustment
                    treble = brightness  # Main brightness control
                    
                    result = GeekyKokoroAdvancedVoiceNode._apply_eq(
                        result, sr, bass=bass, mid=mid, treble=treble
                    )
                except Exception as e:
                    print(f"Voice morph EQ error: {e}")
            
            # 4. Apply breathiness if needed
            if breathiness >= 0.05:
                try:
                    # Create breath noise
                    noise = np.random.normal(0, 0.1, len(audio))
                    
                    # Shape noise to follow vocal envelope
                    envelope = np.abs(result)
                    
                    # Use a simpler smoothing method
                    window_size = min(int(sr * 0.01), len(envelope) // 10)
                    if window_size < 1:
                        window_size = 1
                    smoothed_envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
                    shaped_noise = noise * smoothed_envelope
                    
                    # Apply highpass filter for breath frequencies if scipy is available
                    if SCIPY_AVAILABLE:
                        try:
                            nyquist = sr / 2
                            cutoff_high = min(3000 / nyquist, 0.9)  # Ensure < 1.0
                            
                            b, a = signal.butter(2, cutoff_high, btype='high')
                            breath_noise = signal.lfilter(b, a, shaped_noise)
                            
                            # Add to result based on breathiness parameter
                            result = result * (1 - breathiness * 0.3) + breath_noise * breathiness
                        except Exception as e:
                            print(f"Voice morph breathiness filter error: {e}")
                            # If filter fails, just use shaped noise
                            result = result * (1 - breathiness * 0.2) + shaped_noise * breathiness * 0.2
                    else:
                        # FFT-based high-pass filter
                        n = len(audio)
                        fft = np.fft.rfft(shaped_noise)
                        freq = np.fft.rfftfreq(n, 1/sr)
                        
                        mask = freq >= 3000
                        fft_filtered = np.zeros_like(fft)
                        fft_filtered[mask] = fft[mask]
                        
                        breath_noise = np.fft.irfft(fft_filtered, n)
                        
                        # Add to result
                        result = result * (1 - breathiness * 0.3) + breath_noise * breathiness
                    
                except Exception as e:
                    print(f"Voice morph breathiness error: {e}")
            
            return result
        except Exception as e:
            print(f"Voice morphing error: {e}")
            return audio
    
    # ============ PRESETS ============
    
    @staticmethod
    def _apply_preset(audio, sr, preset_name, strength):
        """Apply pre-defined effect combinations as presets"""
        if preset_name == "None" or strength < 0.01:
            return audio
            
        try:
            # Preset definitions with normalized parameters
            presets = {
                "Chipmunk": {
                    "pitch_shift": 6,
                    "formant_shift": 1.5
                },
                "Deep Voice": {
                    "pitch_shift": -5,
                    "formant_shift": -1.2,
                    "bass_boost": 0.4
                },
                "Helium": {
                    "pitch_shift": 8,
                    "formant_shift": 2.0,
                    "treble_boost": 0.5
                },
                "Phone Call": {
                    "character_effect": "Telephone",
                    "character_intensity": 1.0
                },
                "Child Voice": {
                    "voice_morph": "Child",
                    "morph_intensity": 1.0
                },
                "Elder Voice": {
                    "voice_morph": "Elder",
                    "morph_intensity": 1.0
                },
                "Robot Voice": {
                    "character_effect": "Robot",
                    "character_intensity": 1.0
                },
                "Ethereal": {
                    "pitch_shift": 2,
                    "reverb_amount": 0.8,
                    "reverb_room_size": 0.9,
                    "vibrato": 0.3
                },
                "Monster": {
                    "pitch_shift": -6,
                    "formant_shift": -1.5,
                    "distortion": 0.4,
                    "bass_boost": 0.5
                },
                "Ghost": {
                    "pitch_shift": 1,
                    "reverb_amount": 0.7,
                    "formant_shift": 0.5,
                    "tremolo": 0.3
                },
                "Fantasy Elf": {
                    "pitch_shift": 3,
                    "formant_shift": 1.2,
                    "reverb_amount": 0.4,
                    "harmonics": 0.6
                },
                "Dwarf": {
                    "pitch_shift": -3,
                    "formant_shift": -0.8,
                    "bass_boost": 0.6,
                    "warmth": 0.5
                },
                "Orc": {
                    "pitch_shift": -4,
                    "formant_shift": -1.0,
                    "distortion": 0.3,
                    "bass_boost": 0.7
                },
                "Celestial": {
                    "character_effect": "Angel",
                    "character_intensity": 1.0
                },
                "Demonic": {
                    "character_effect": "Demon",
                    "character_intensity": 1.0
                },
                "Radio Host": {
                    "compression": 0.7,
                    "warmth": 0.4,
                    "mid_boost": 0.3,
                    "bass_boost": 0.2
                },
                "Podcast": {
                    "compression": 0.6,
                    "noise_reduction": 0.5,
                    "mid_boost": 0.2,
                    "warmth": 0.3
                },
                "Movie Trailer": {
                    "pitch_shift": -1,
                    "compression": 0.8,
                    "bass_boost": 0.5,
                    "reverb_amount": 0.3
                },
                "Storyteller": {
                    "warmth": 0.5,
                    "reverb_amount": 0.2,
                    "compression": 0.4,
                    "bass_boost": 0.1
                },
                "Singer": {
                    "auto_tune": 0.5,
                    "reverb_amount": 0.3,
                    "compression": 0.6,
                    "harmonics": 0.4
                }
            }
            
            if preset_name not in presets:
                return audio
                
            params = presets[preset_name]
            result = audio.copy()
            
            # Apply each effect in the preset, scaled by strength
            for effect, value in params.items():
                # Scale the effect by the strength parameter
                scaled_value = value * strength
                
                # Apply the appropriate effect based on the parameter name
                if effect == "pitch_shift" and abs(scaled_value) >= 0.1:
                    result = GeekyKokoroAdvancedVoiceNode._apply_pitch_shift(
                        result, sr, n_steps=scaled_value
                    )
                elif effect == "formant_shift" and abs(scaled_value) >= 0.1:
                    result = GeekyKokoroAdvancedVoiceNode._apply_formant_shift(
                        result, sr, shift_amount=scaled_value
                    )
                elif effect == "reverb_amount" and scaled_value >= 0.1:
                    room_size = params.get("reverb_room_size", 0.5) * strength
                    damping = params.get("reverb_damping", 0.5) * strength
                    result = GeekyKokoroAdvancedVoiceNode._apply_reverb(
                        result, sr, amount=scaled_value, room_size=room_size, damping=damping
                    )
                elif effect == "distortion" and scaled_value >= 0.1:
                    result = GeekyKokoroAdvancedVoiceNode._apply_distortion(
                        result, effect_strength=scaled_value
                    )
                elif effect == "tremolo" and scaled_value >= 0.1:
                    result = GeekyKokoroAdvancedVoiceNode._apply_tremolo(
                        result, effect_strength=scaled_value, sr=sr
                    )
                elif effect == "bass_boost" and abs(scaled_value) >= 0.1:
                    mid = params.get("mid_boost", 0) * strength
                    treble = params.get("treble_boost", 0) * strength
                    result = GeekyKokoroAdvancedVoiceNode._apply_eq(
                        result, sr, bass=scaled_value, mid=mid, treble=treble
                    )
                elif effect == "harmonics" and scaled_value >= 0.1:
                    result = GeekyKokoroAdvancedVoiceNode._apply_harmonics(
                        result, sr, amount=scaled_value
                    )
                elif effect == "compression" and scaled_value >= 0.1:
                    ceiling = 0.95
                    result = GeekyKokoroAdvancedVoiceNode._apply_compression(
                        result, amount=scaled_value, limit_ceiling=ceiling
                    )
                elif effect == "auto_tune" and scaled_value >= 0.1:
                    result = GeekyKokoroAdvancedVoiceNode._apply_auto_tune(
                        result, sr, strength=scaled_value
                    )
                elif effect == "warmth" and scaled_value >= 0.1:
                    result = GeekyKokoroAdvancedVoiceNode._apply_warmth(
                        result, amount=scaled_value
                    )
                elif effect == "vibrato" and scaled_value >= 0.1:
                    speed = 5.0  # Default speed
                    result = GeekyKokoroAdvancedVoiceNode._apply_vibrato(
                        result, sr, depth=scaled_value, speed=speed
                    )
                elif effect == "noise_reduction" and scaled_value >= 0.1:
                    result = GeekyKokoroAdvancedVoiceNode._apply_noise_reduction(
                        result, amount=scaled_value
                    )
                elif effect == "character_effect" and scaled_value >= 0.1:
                    intensity = params.get("character_intensity", 1.0) * strength
                    result = GeekyKokoroAdvancedVoiceNode._apply_character_effect(
                        result, sr, effect_name=value, intensity=intensity
                    )
                elif effect == "voice_morph" and scaled_value >= 0.1:
                    intensity = params.get("morph_intensity", 1.0) * strength
                    result = GeekyKokoroAdvancedVoiceNode._apply_voice_morph(
                        result, sr, morph_type=value, intensity=intensity
                    )
            
            return result
        except Exception as e:
            print(f"Preset application error: {e}")
            return audio
    
    # ============ MAIN PROCESSING FUNCTION ============
    
    def process_voice(self, audio, effect_blend=1.0, 
                     # Voice Morphing
                     voice_morph="None", morph_intensity=0.7,
                     # Pitch and Formant
                     enable_pitch_formant=False, pitch_shift=0.0, formant_shift=0.0, auto_tune=0.0,
                     # Time Controls
                     enable_time=False, time_stretch=1.0, vibrato=0.0, vibrato_speed=5.0,
                     # Spatial
                     enable_spatial=False, reverb_amount=0.0, reverb_room_size=0.5, reverb_damping=0.5, 
                     echo_delay=0.0, echo_feedback=0.3,
                     # Tone Controls
                     enable_tone=False, bass_boost=0.0, mid_boost=0.0, treble_boost=0.0, harmonics=0.0,
                     # Effects
                     enable_effects=False, distortion=0.0, tremolo=0.0, bitcrush=0.0, noise_reduction=0.0,
                     # Dynamics
                     enable_dynamics=False, compression=0.0, limit_ceiling=0.95, warmth=0.0,
                     # Character Effects
                     character_effect="None", character_intensity=0.7,
                     # Presets
                     preset="None", preset_strength=0.7):
        """Apply comprehensive voice processing with effect groups and improved algorithms"""
        # Default output in case of total failure
        default_output = {"waveform": torch.zeros((1, 1, 1000), dtype=torch.float32), "sample_rate": 24000}
        
        try:
            # Early validation of input
            if audio is None:
                print("Error: Input audio is None")
                return (default_output,)
            
            if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
                print(f"Error: Invalid audio input format: {type(audio)}")
                return (default_output,)
            
            # Start timing for performance monitoring
            start_time = time.time()
            
            # Get input audio
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            sr = sample_rate  # For convenience in effect functions
            
            # Debug input
            print(f"Input waveform shape: {waveform.shape}, sample_rate: {sample_rate}")
            
            # Handle tensor shape robustly
            if waveform.dim() != 3 or waveform.shape[0] != 1:
                print(f"Unexpected waveform shape {waveform.shape}, adjusting to [1, 1, samples]")
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0)
                elif waveform.dim() == 2:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform[0:1, 0:1, :]  # Force to [1, 1, samples]
            
            # Convert to numpy for processing, preserving original length
            original_length = waveform.shape[-1]
            audio_data = waveform.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            
            # Create a copy of the original for blending
            original_audio = audio_data.copy()
            result = audio_data.copy()
            
            # Apply preset first if specified (this overrides individual settings)
            if preset != "None" and preset_strength > 0:
                print(f"Applying preset: {preset} with strength {preset_strength}")
                result = self._apply_preset(result, sample_rate, preset, preset_strength)
            else:
                # Apply voice morphing if enabled
                if voice_morph != "None" and morph_intensity > 0:
                    print(f"Applying voice morph: {voice_morph} with intensity {morph_intensity}")
                    result = self._apply_voice_morph(result, sample_rate, voice_morph, morph_intensity)
                
                # Apply pitch and formant modifications if enabled
                if enable_pitch_formant:
                    if abs(pitch_shift) >= 0.1:
                        print(f"Applying pitch shift: {pitch_shift}")
                        result = self._apply_pitch_shift(result, sample_rate, pitch_shift)
                    
                    if abs(formant_shift) >= 0.1:
                        print(f"Applying formant shift: {formant_shift}")
                        result = self._apply_formant_shift(result, sample_rate, formant_shift)
                    
                    if auto_tune >= 0.1:
                        print(f"Applying auto-tune: {auto_tune}")
                        result = self._apply_auto_tune(result, sample_rate, auto_tune)
                
                # Apply time-based effects if enabled
                if enable_time:
                    if abs(time_stretch - 1.0) >= 0.01:
                        print(f"Applying time stretch: {time_stretch}")
                        result = self._apply_time_stretch(result, sample_rate, time_stretch)
                    
                    if vibrato >= 0.1:
                        print(f"Applying vibrato: {vibrato} with speed {vibrato_speed}")
                        result = self._apply_vibrato(result, sample_rate, vibrato, vibrato_speed)
                
                # Apply spatial effects if enabled
                if enable_spatial:
                    if reverb_amount >= 0.1:
                        print(f"Applying reverb: {reverb_amount}")
                        result = self._apply_reverb(result, sample_rate, reverb_amount, 
                                                  reverb_room_size, reverb_damping)
                    
                    if echo_delay >= 0.1:
                        print(f"Applying echo: {echo_delay}")
                        result = self._apply_echo(result, sample_rate, echo_delay, echo_feedback)
                
                # Apply tone shaping if enabled
                if enable_tone:
                    if abs(bass_boost) >= 0.1 or abs(mid_boost) >= 0.1 or abs(treble_boost) >= 0.1:
                        print(f"Applying EQ: bass={bass_boost}, mid={mid_boost}, treble={treble_boost}")
                        result = self._apply_eq(result, sample_rate, bass_boost, mid_boost, treble_boost)
                    
                    if harmonics >= 0.1:
                        print(f"Applying harmonics: {harmonics}")
                        result = self._apply_harmonics(result, sample_rate, harmonics)
                
                # Apply effects if enabled
                if enable_effects:
                    if distortion >= 0.1:
                        print(f"Applying distortion: {distortion}")
                        result = self._apply_distortion(result, distortion)
                    
                    if tremolo >= 0.1:
                        print(f"Applying tremolo: {tremolo}")
                        result = self._apply_tremolo(result, tremolo, sample_rate)
                    
                    if bitcrush >= 0.1:
                        print(f"Applying bitcrush: {bitcrush}")
                        result = self._apply_bitcrush(result, bitcrush)
                    
                    if noise_reduction >= 0.1:
                        print(f"Applying noise reduction: {noise_reduction}")
                        result = self._apply_noise_reduction(result, noise_reduction)
                
                # Apply dynamics processing if enabled
                if enable_dynamics:
                    if compression >= 0.1:
                        print(f"Applying compression: {compression}")
                        result = self._apply_compression(result, compression, limit_ceiling)
                    
                    if warmth >= 0.1:
                        print(f"Applying warmth: {warmth}")
                        result = self._apply_warmth(result, warmth)
                
                # Apply character effect if specified
                if character_effect != "None" and character_intensity > 0:
                    print(f"Applying character effect: {character_effect}")
                    result = self._apply_character_effect(result, sample_rate, 
                                                        character_effect, character_intensity)
            
            # Apply overall effect blend (mix with original)
            if effect_blend < 1.0:
                print(f"Blending with original: {effect_blend * 100}% effect, {(1 - effect_blend) * 100}% original")
                result = original_audio * (1 - effect_blend) + result * effect_blend
            
            # Ensure output length matches input
            if len(result) != original_length:
                print(f"Output length mismatch: {len(result)} vs {original_length}, adjusting...")
                if len(result) > original_length:
                    result = result[:original_length]
                else:
                    result = np.pad(result, (0, original_length - len(result)), mode='constant')
            
            # Final normalization to prevent clipping
            max_amp = np.max(np.abs(result))
            if max_amp > 0.99:
                print(f"Normalizing output (peak: {max_amp})")
                result = result / max_amp * 0.99
            
            # Convert back to tensor
            processed_waveform = torch.tensor(result, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Final shape validation
            if processed_waveform.shape != (1, 1, original_length):
                print(f"Final shape mismatch: {processed_waveform.shape}, forcing to [1, 1, {original_length}]")
                processed_waveform = processed_waveform[:, :, :original_length]
                if processed_waveform.shape[-1] < original_length:
                    processed_waveform = torch.nn.functional.pad(
                        processed_waveform, (0, original_length - processed_waveform.shape[-1])
                    )
            
            # Create output dictionary
            output = {"waveform": processed_waveform, "sample_rate": sample_rate}
            
            # Report processing time
            process_time = time.time() - start_time
            print(f"Voice processing completed in {process_time:.2f} seconds")
            
            # Return as tuple to match RETURN_TYPES
            return (output,)
            
        except Exception as e:
            print(f"Voice processing critical error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return default output as tuple
            return (default_output,)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeekyKokoroAdvancedVoice": GeekyKokoroAdvancedVoiceNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyKokoroAdvancedVoice": "🔊 Geeky Kokoro Advanced Voice"
}