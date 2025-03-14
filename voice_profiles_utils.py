
import numpy as np
import torch
import time

# Import the required libraries with fallbacks
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

# Try to import fallback implementations
try:
    from .audio_utils import (simple_pitch_shift, stft_phase_vocoder, formant_shift_basic, 
                             stft, istft, amplitude_to_db, db_to_amplitude)
    FALLBACKS_AVAILABLE = True
except ImportError:
    FALLBACKS_AVAILABLE = False
    print("Warning: audio_utils.py not found, some effects may not work")

class VoiceProfileUtils:
    @staticmethod
    def preserve_volume(original, processed):
        try:
            original_rms = np.sqrt(np.mean(original**2))
            processed_rms = np.sqrt(np.mean(processed**2))
            if processed_rms > 0 and original_rms > 0:
                gain = original_rms / processed_rms
                processed *= gain
            return processed
        except Exception as e:
            print(f"Volume preservation error: {e}")
            return processed

    # Pitch and Formant Processing
    @staticmethod
    def _improved_speed_change(audio, speed_factor):
        if abs(speed_factor - 1.0) < 0.01:
            return audio
        try:
            indices = np.round(np.arange(0, len(audio), speed_factor))
            valid_indices = indices[indices < len(audio)].astype(int)
            return audio[valid_indices] if len(valid_indices) > 0 else audio
        except Exception as e:
            print(f"Speed change error: {e}")
            return audio
    
    @staticmethod
    def _improved_simple_pitch_shift(audio, sr, n_steps, bins_per_octave=12):
        if abs(n_steps) < 0.1:
            return audio
        try:
            rate = 2.0 ** (-n_steps / bins_per_octave)
            changed = VoiceProfileUtils._improved_speed_change(audio, 1.0/rate)
            indices = np.linspace(0, len(changed) - 1, len(audio))
            return np.interp(indices, np.arange(len(changed)), changed) if len(changed) > 0 else audio
        except Exception as e:
            print(f"Simple pitch shift error: {e}")
            return audio
    
    @staticmethod
    def _apply_pitch_shift(audio, sr, n_steps, quality='high'):
        if abs(n_steps) < 0.1:
            return audio
        try:
            if LIBROSA_AVAILABLE and RESAMPY_AVAILABLE:
                audio_out = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps,
                                                      bins_per_octave=24 if quality == 'high' else 12,
                                                      res_type='kaiser_best' if quality == 'high' else 'kaiser_fast')
            elif LIBROSA_AVAILABLE:
                audio_out = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps,
                                                      bins_per_octave=12, res_type='fft')
            elif FALLBACKS_AVAILABLE and SCIPY_AVAILABLE:
                audio_out = stft_phase_vocoder(audio, sr, n_steps)
            else:
                audio_out = VoiceProfileUtils._improved_simple_pitch_shift(audio, sr, n_steps)
            return VoiceProfileUtils.preserve_volume(audio, audio_out)
        except Exception as e:
            print(f"Pitch shift error: {e}")
            return audio
    
    @staticmethod
    def _apply_formant_shift(audio, sr, shift_amount):
        if abs(shift_amount) < 0.1:
            return audio
        try:
            if LIBROSA_AVAILABLE:
                n_fft = 2048
                hop_length = n_fft // 4
                S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
                S_mag, S_phase = librosa.magphase(S)
                freq_bins = S_mag.shape[0]
                shift_bins = int(freq_bins * shift_amount * 0.1)
                S_mag_shifted = np.zeros_like(S_mag)
                if shift_bins > 0:
                    S_mag_shifted[shift_bins:, :] = S_mag[:-shift_bins, :]
                    blend_region = min(shift_bins, 20)
                    for i in range(blend_region):
                        weight = i / blend_region
                        S_mag_shifted[i, :] = S_mag[i, :] * (1 - weight)
                elif shift_bins < 0:
                    shift_bins = abs(shift_bins)
                    S_mag_shifted[:-shift_bins, :] = S_mag[shift_bins:, :]
                    S_mag_shifted[-shift_bins:, :] = S_mag[-shift_bins:, :] * 0.5
                else:
                    S_mag_shifted = S_mag
                S_shifted = S_mag_shifted * np.exp(1j * np.angle(S))
                y_shifted = librosa.istft(S_shifted, hop_length=hop_length, length=len(audio))
                return VoiceProfileUtils.preserve_volume(audio, y_shifted)
            elif FALLBACKS_AVAILABLE:
                return VoiceProfileUtils.preserve_volume(audio, formant_shift_basic(audio, sr, shift_amount))
            else:
                audio_copy = audio.copy()
                shifted = VoiceProfileUtils._improved_simple_pitch_shift(audio_copy, sr, shift_amount * 3)
                time_stretched = VoiceProfileUtils._improved_speed_change(shifted, 2 ** (shift_amount * 3 / 12))
                indices = np.linspace(0, len(time_stretched) - 1, len(audio))
                result = np.interp(indices, np.arange(len(time_stretched)), time_stretched)
                return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Formant shift error: {e}")
            return audio
    
    @staticmethod
    def _apply_auto_tune(audio, sr, strength):
        if strength < 0.01:
            return audio
        try:
            if not LIBROSA_AVAILABLE:
                delay_samples = int(sr * 0.02)
                if delay_samples >= len(audio):
                    return audio
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples]
                result = audio * (1 - strength * 0.3) + delayed * (strength * 0.3)
            else:
                hop_length = 512
                f0, voiced_flag, _ = librosa.pyin(audio, fmin=65.0, fmax=2093.0, sr=sr, hop_length=hop_length)
                if f0 is None or np.all(np.isnan(f0)):
                    return audio
                f0 = np.nan_to_num(f0)
                tuned_f0 = f0.copy()
                voiced_mask = ~np.isnan(f0)
                if np.any(voiced_mask):
                    midi_notes = librosa.hz_to_midi(f0[voiced_mask])
                    quantized_notes = np.round(midi_notes)
                    tuned_hz = librosa.midi_to_hz(quantized_notes)
                    tuned_f0[voiced_mask] = f0[voiced_mask] * (1 - strength) + tuned_hz * strength
                output = np.zeros_like(audio)
                segment_length = hop_length * 2
                num_segments = len(audio) // segment_length
                for i in range(num_segments):
                    start = i * segment_length
                    end = start + segment_length
                    center_idx = (i * 2) + 1
                    if center_idx < len(f0) and center_idx < len(tuned_f0):
                        orig_pitch = f0[center_idx]
                        tuned_pitch = tuned_f0[center_idx]
                        if orig_pitch > 0 and tuned_pitch > 0:
                            shift_amount = 12 * np.log2(tuned_pitch / orig_pitch)
                            if abs(shift_amount) > 0.01:
                                segment = audio[start:end]
                                shifted_segment = VoiceProfileUtils._apply_pitch_shift(segment, sr, shift_amount)
                                if len(shifted_segment) > len(segment):
                                    shifted_segment = shifted_segment[:len(segment)]
                                elif len(shifted_segment) < len(segment):
                                    shifted_segment = np.pad(shifted_segment, (0, len(segment) - len(shifted_segment)), mode='constant')
                                output[start:end] = shifted_segment
                            else:
                                output[start:end] = audio[start:end]
                        else:
                            output[start:end] = audio[start:end]
                    else:
                        output[start:end] = audio[start:end]
                if len(audio) > num_segments * segment_length:
                    output[num_segments * segment_length:] = audio[num_segments * segment_length:]
                result = output
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Auto-tune error: {e}")
            return audio
    
    # Time-Based Effects
    @staticmethod
    def _apply_time_stretch(audio, sr, rate):
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
                stretched = VoiceProfileUtils._improved_speed_change(audio, 1.0/rate)
                indices = np.linspace(0, len(stretched) - 1, len(audio))
                stretched = np.interp(indices, np.arange(len(stretched)), stretched)
            return VoiceProfileUtils.preserve_volume(audio, stretched)
        except Exception as e:
            print(f"Time stretch error: {e}")
            return audio
    
    @staticmethod
    def _apply_vibrato(audio, sr, depth, speed=5.0):
        if depth < 0.01:
            return audio
        try:
            hop_length = 512
            time_arr = np.arange(len(audio)) / sr
            mod_signal = depth * np.sin(2 * np.pi * speed * time_arr)
            pitch_shifts = mod_signal * 1.0
            segment_length = hop_length * 4
            num_segments = len(audio) // segment_length
            result = np.zeros_like(audio)
            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length
                center_idx = start + segment_length // 2
                if center_idx >= len(pitch_shifts):
                    center_idx = len(pitch_shifts) - 1
                shift_amount = pitch_shifts[center_idx]
                segment = audio[start:end]
                if len(segment) < hop_length:
                    result[start:end] = segment
                    continue
                shifted_segment = VoiceProfileUtils._improved_simple_pitch_shift(segment, sr, shift_amount)
                if len(shifted_segment) > len(segment):
                    shifted_segment = shifted_segment[:len(segment)]
                elif len(shifted_segment) < len(segment):
                    shifted_segment = np.pad(shifted_segment, (0, len(segment) - len(shifted_segment)), mode='constant')
                result[start:end] = shifted_segment
            if len(audio) > num_segments * segment_length:
                result[num_segments * segment_length:] = audio[num_segments * segment_length:]
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Vibrato error: {e}")
            return audio
    
    # Spatial Effects
    @staticmethod
    def _apply_reverb(audio, sr, amount, room_size=0.5, damping=0.5):
        if amount < 0.01:
            return audio
        try:
            reverb_length = int(sr * (0.1 + room_size * 1.5))
            decay_factor = 0.2 + damping * 0.7
            impulse = np.zeros(reverb_length)
            impulse[0] = 1.0
            num_early = int(5 + room_size * 10)
            early_times = np.sort(np.random.randint(1, reverb_length // 3, num_early))
            early_amps = np.random.uniform(0.1, 0.4, num_early) * (1 - damping * 0.5)
            for time, amp in zip(early_times, early_amps):
                impulse[time] = amp
            for i in range(1, reverb_length):
                impulse[i] += np.random.randn() * np.exp(-i / (sr * decay_factor))
            impulse /= np.max(np.abs(impulse))
            if SCIPY_AVAILABLE:
                reverb_signal = signal.fftconvolve(audio, impulse, mode='full')[:len(audio)]
            else:
                reverb_signal = np.zeros_like(audio)
                for i in range(len(audio)):
                    for j in range(min(i + 1, reverb_length)):
                        if i - j >= 0:
                            reverb_signal[i] += audio[i - j] * impulse[j]
            output = (1 - amount) * audio + amount * reverb_signal
            return VoiceProfileUtils.preserve_volume(audio, output)
        except Exception as e:
            print(f"Reverb effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_echo(audio, sr, amount, feedback=0.3):
        if amount < 0.01:
            return audio
        try:
            delay_time = 0.1 + amount * 0.4
            delay_samples = int(sr * delay_time)
            if delay_samples >= len(audio):
                delay_samples = len(audio) // 2
                if delay_samples == 0:
                    return audio
            output = np.zeros_like(audio, dtype=np.float32)
            output[:] = audio[:]
            output[delay_samples:] += audio[:-delay_samples] * amount
            if feedback > 0.01:
                for i in range(1, 3):
                    tap_delay = delay_samples * (i + 1)
                    tap_gain = amount * (feedback ** i)
                    if tap_delay >= len(audio) or tap_gain < 0.01:
                        break
                    output[tap_delay:] += audio[:-tap_delay] * tap_gain
            return VoiceProfileUtils.preserve_volume(audio, output)
        except Exception as e:
            print(f"Echo effect error: {e}")
            return audio
    
    # Tone Shaping
    @staticmethod
    def _apply_eq(audio, sr, bass=0.0, mid=0.0, treble=0.0):
        if abs(bass) < 0.01 and abs(mid) < 0.01 and abs(treble) < 0.01:
            return audio
        try:
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                bass_norm = min(max(250 / nyquist, 0.001), 0.999)
                mid_low_norm = min(max(250 / nyquist, 0.001), 0.999)
                mid_high_norm = min(max(4000 / nyquist, 0.001), 0.999)
                treble_norm = min(max(4000 / nyquist, 0.001), 0.999)
                if mid_low_norm >= mid_high_norm:
                    mid_low_norm = mid_high_norm * 0.5
                b_bass, a_bass = signal.butter(2, bass_norm, btype='low')
                b_mid, a_mid = signal.butter(2, [mid_low_norm, mid_high_norm], btype='band')
                b_treble, a_treble = signal.butter(2, treble_norm, btype='high')
                bass_filtered = signal.filtfilt(b_bass, a_bass, audio)
                mid_filtered = signal.filtfilt(b_mid, a_mid, audio)
                treble_filtered = signal.filtfilt(b_treble, a_treble, audio)
                bass_gain = 10 ** (bass * 0.5)
                mid_gain = 10 ** (mid * 0.4)
                treble_gain = 10 ** (treble * 0.5)
                result = bass_filtered * bass_gain + mid_filtered * mid_gain + treble_filtered * treble_gain
            else:
                n = len(audio)
                fft = np.fft.rfft(audio)
                freq = np.fft.rfftfreq(n, 1/sr)
                gain_mask = np.ones(len(freq))
                gain_mask[freq < 250] *= (1.0 + bass * 0.8)
                gain_mask[(freq >= 250) & (freq < 4000)] *= (1.0 + mid * 0.6)
                gain_mask[freq >= 4000] *= (1.0 + treble * 0.8)
                fft_eq = fft * gain_mask
                result = np.fft.irfft(fft_eq, n)
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"EQ effect error: {e}")
            return audio

    @staticmethod
    def _apply_harmonics(audio, sr, amount):
        if amount < 0.01:
            return audio
        try:
            h2 = np.sign(audio) * (audio ** 2)
            h3 = audio ** 3
            h2 = h2 / np.max(np.abs(h2)) if np.max(np.abs(h2)) > 0 else h2
            h3 = h3 / np.max(np.abs(h3)) if np.max(np.abs(h3)) > 0 else h3
            harmonic_mix = audio + h2 * amount * 0.5 + h3 * amount * 0.25
            result = audio * (1 - amount * 0.5) + harmonic_mix * amount * 0.5
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Harmonics effect error: {e}")
            return audio
    
    # Effects
    @staticmethod
    def _apply_distortion(audio, effect_strength):
        if effect_strength < 0.01:
            return audio
        try:
            drive = 1 + effect_strength * 15
            distorted = np.tanh(audio * drive) / np.tanh(drive)
            result = audio * (1 - effect_strength) + distorted * effect_strength
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Distortion error: {e}")
            return audio
    
    @staticmethod
    def _apply_tremolo(audio, effect_strength, sr, speed=5.0):
        if effect_strength < 0.01:
            return audio
        try:
            t = np.arange(len(audio)) / sr
            mod_depth = effect_strength * 0.9
            tremolo_wave = 1.0 - mod_depth + mod_depth * np.sin(2 * np.pi * speed * t)
            result = audio * tremolo_wave
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Tremolo error: {e}")
            return audio
    
    @staticmethod
    def _apply_bitcrush(audio, intensity):
        if intensity < 0.01:
            return audio
        try:
            bits = int(16 - intensity * (16 - 2))
            steps = 2**bits
            audio_crushed = np.round(audio * (steps/2)) / (steps/2)
            result = audio * (1 - intensity) + audio_crushed * intensity
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Bitcrush error: {e}")
            return audio
    
    @staticmethod
    def _apply_noise_reduction(audio, amount):
        if amount < 0.01:
            return audio
        try:
            abs_audio = np.abs(audio)
            window_size = min(int(sr / 50), len(abs_audio) // 10) or 1
            envelope = np.zeros_like(abs_audio)
            for i in range(len(abs_audio)):
                start = max(0, i - window_size)
                end = min(len(abs_audio), i + window_size)
                envelope[i] = np.max(abs_audio[start:end])
            noise_floor = np.percentile(envelope, 10)
            gain = 1.0 - amount * (noise_floor / (envelope + 1e-10))
            gain = np.clip(gain, 0.0, 1.0)
            result = audio * gain
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Noise reduction error: {e}")
            return audio
    
    # Dynamics Processing
    @staticmethod
    def _apply_compression(audio, amount, limit_ceiling=0.95):
        if amount < 0.01:
            return audio
        try:
            ratio = 1 + 7 * amount
            rms = np.sqrt(np.mean(audio**2))
            db_rms = 20 * np.log10(max(rms, 1e-8))
            threshold_db = db_rms - 6 - amount * 12
            threshold = 10 ** (threshold_db / 20)
            attack_samples = max(int((20 - amount * 15) * sr / 1000), 1)
            release_samples = max(int((150 + amount * 350) * sr / 1000), 1)
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
                if target_gain < gain:
                    gain = gain + (target_gain - gain) / attack_samples
                else:
                    gain = gain + (target_gain - gain) / release_samples
                result[i] = audio[i] * gain
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Compression error: {e}")
            return audio
    
    @staticmethod
    def _apply_warmth(audio, amount):
        if amount < 0.01:
            return audio
        try:
            warmth = audio.copy()
            h2 = audio**2 * np.sign(audio)
            h4 = audio**4 * np.sign(audio)
            h2 = h2 / np.max(np.abs(h2)) if np.max(np.abs(h2)) > 0 else h2
            h4 = h4 / np.max(np.abs(h4)) if np.max(np.abs(h4)) > 0 else h4
            warmth = audio * (1 - amount * 0.4) + h2 * (amount * 0.35) + h4 * (amount * 0.05)
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                low_freq = min(200 / nyquist, 0.9)
                b, a = signal.butter(2, low_freq, btype='low')
                low_end = signal.filtfilt(b, a, warmth)
                boost_amount = 0.2 * amount
                warmth = warmth * (1 - boost_amount) + low_end * (1 + boost_amount)
            else:
                n = len(audio)
                fft = np.fft.rfft(warmth)
                freq = np.fft.rfftfreq(n, 1/sr)
                boost_mask = freq < 200
                fft[boost_mask] *= (1.0 + amount * 0.3)
                warmth = np.fft.irfft(fft, n)
            clip_amount = 0.8 + 0.2 * (1 - amount)
            warmth = np.tanh(warmth / clip_amount) * clip_amount
            return VoiceProfileUtils.preserve_volume(audio, warmth)
        except Exception as e:
            print(f"Warmth effect error: {e}")
            return audio
            
    # Character & Voice Profile Effects
    @staticmethod
    def _apply_robot(audio, sr, intensity=0.7):
        try:
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                low_freq = min(max(500 / nyquist, 0.01), 0.99)
                high_freq = min(max(2000 / nyquist, 0.01), 0.99)
                b, a = signal.butter(2, [low_freq, high_freq], btype='band')
                filtered = signal.filtfilt(b, a, audio)
            else:
                filtered = audio
            t = np.arange(len(audio)) / sr
            buzz = 1.0 + 0.5 * intensity * np.sin(2 * np.pi * 50 * t)
            robot = filtered * buzz
            robot = VoiceProfileUtils._apply_distortion(robot, 0.3 * intensity)
            robot = VoiceProfileUtils._apply_auto_tune(robot, sr, 0.2 * intensity)
            return VoiceProfileUtils.preserve_volume(audio, robot)
        except Exception as e:
            print(f"Robot effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_telephone(audio, sr, intensity=0.7):
        try:
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                low_freq = min(max(300 / nyquist, 0.01), 0.99)
                high_freq = min(max(3400 / nyquist, 0.01), 0.99)
                b, a = signal.butter(2, [low_freq, high_freq], btype='band')
                filtered = signal.filtfilt(b, a, audio)
            else:
                filtered = audio
            distorted = VoiceProfileUtils._apply_distortion(filtered, 0.2 * intensity)
            noise_level = 0.02 * intensity
            noise = np.random.normal(0, noise_level, len(distorted))
            result = distorted + noise
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Telephone effect error: {e}")
            return audio

    @staticmethod
    def _apply_megaphone(audio, sr, intensity=0.7):
        try:
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                low_freq = min(max(500 / nyquist, 0.01), 0.99)
                high_freq = min(max(4000 / nyquist, 0.01), 0.99)
                b, a = signal.butter(2, [low_freq, high_freq], btype='band')
                filtered = signal.filtfilt(b, a, audio)
            else:
                filtered = audio
            compressed = VoiceProfileUtils._apply_compression(filtered, 0.5 * intensity)
            distorted = np.clip(compressed * 3.0, -0.9, 0.9)
            reverb = VoiceProfileUtils._apply_reverb(distorted, sr, 0.2 * intensity, 0.3, 0.7)
            return VoiceProfileUtils.preserve_volume(audio, reverb)
        except Exception as e:
            print(f"Megaphone effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_radio(audio, sr, intensity=0.7):
        try:
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                low_freq = min(max(400 / nyquist, 0.01), 0.99)
                high_freq = min(max(3000 / nyquist, 0.01), 0.99)
                b, a = signal.butter(2, [low_freq, high_freq], btype='band')
                filtered = signal.filtfilt(b, a, audio)
            else:
                filtered = audio
            distorted = np.tanh(filtered * 1.5) / 1.5
            t = np.arange(len(audio)) / sr
            am = 1.0 + 0.4 * intensity * np.sin(2 * np.pi * 0.5 * t)
            radio = distorted * am
            noise_level = 0.05 * intensity
            static = np.random.normal(0, noise_level, len(audio))
            result = radio + static
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Radio effect error: {e}")
            return audio
    
    @staticmethod
    def apply_voice_profile(audio, sr, profile_name, intensity):
        """Main entry point for applying voice profiles"""
        if profile_name == "None" or intensity < 0.01:
            return audio
        try:
            result = audio.copy()
            
            # Age-based voices
            if profile_name == "Child":
                result = VoiceProfileUtils._apply_child_voice(audio, sr, intensity)
            elif profile_name == "Teen":
                result = VoiceProfileUtils._apply_teen_voice(audio, sr, intensity)
            elif profile_name == "Adult":
                result = VoiceProfileUtils._apply_adult_voice(audio, sr, intensity)
            elif profile_name == "Elder":
                result = VoiceProfileUtils._apply_elder_voice(audio, sr, intensity)
            
            # Gender-based voices
            elif profile_name == "Feminine":
                result = VoiceProfileUtils._apply_feminine_voice(audio, sr, intensity)
            elif profile_name == "Masculine":
                result = VoiceProfileUtils._apply_masculine_voice(audio, sr, intensity)
            
            # Character effects 
            elif profile_name == "Robot":
                result = VoiceProfileUtils._apply_robot(audio, sr, intensity)
            elif profile_name == "Telephone":
                result = VoiceProfileUtils._apply_telephone(audio, sr, intensity)
            elif profile_name == "Megaphone":
                result = VoiceProfileUtils._apply_megaphone(audio, sr, intensity)
            elif profile_name == "Radio":
                result = VoiceProfileUtils._apply_radio(audio, sr, intensity)
            elif profile_name == "Underwater":
                result = VoiceProfileUtils._apply_underwater(audio, sr, intensity)
            elif profile_name == "Whisper":
                result = VoiceProfileUtils._apply_whisper(audio, sr, intensity)
            elif profile_name == "Demon":
                result = VoiceProfileUtils._apply_demon(audio, sr, intensity)
            elif profile_name == "Angel":
                result = VoiceProfileUtils._apply_angel(audio, sr, intensity)
            elif profile_name == "Alien":
                result = VoiceProfileUtils._apply_alien(audio, sr, intensity)
            elif profile_name == "Darth Vader":
                result = VoiceProfileUtils._apply_darth_vader(audio, sr, intensity)
            
            # Presets
            elif profile_name == "Chipmunk":
                result = VoiceProfileUtils._apply_chipmunk(audio, sr, intensity)
            elif profile_name == "Deep Voice":
                result = VoiceProfileUtils._apply_deep_voice(audio, sr, intensity)
            elif profile_name == "Fantasy Elf":
                result = VoiceProfileUtils._apply_fantasy_elf(audio, sr, intensity)
            elif profile_name == "Orc":
                result = VoiceProfileUtils._apply_orc(audio, sr, intensity)
            elif profile_name == "Monster":
                result = VoiceProfileUtils._apply_monster(audio, sr, intensity)
            elif profile_name == "Ghost":
                result = VoiceProfileUtils._apply_ghost(audio, sr, intensity)
            elif profile_name == "Radio Host":
                result = VoiceProfileUtils._apply_radio_host(audio, sr, intensity)
            elif profile_name == "TV Announcer":
                result = VoiceProfileUtils._apply_tv_announcer(audio, sr, intensity)
            elif profile_name == "Movie Trailer":
                result = VoiceProfileUtils._apply_movie_trailer(audio, sr, intensity)
            elif profile_name == "Singer":
                result = VoiceProfileUtils._apply_singer(audio, sr, intensity)
            
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Voice profile error for {profile_name}: {e}")
            return audio
        
    # Character Voice Implementations    
    @staticmethod
    def _apply_underwater(audio, sr, intensity=0.7):
        try:
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                cutoff = min(max(800 / nyquist, 0.01), 0.99)
                b, a = signal.butter(2, cutoff, btype='low')
                underwater = signal.filtfilt(b, a, audio)
            else:
                underwater = audio
            t = np.arange(len(audio)) / sr
            trem = 1.0 + 0.4 * intensity * np.sin(2 * np.pi * 0.8 * t)
            underwater = underwater * trem
            underwater = VoiceProfileUtils._apply_vibrato(underwater, sr, 0.2 * intensity, 0.5)
            return VoiceProfileUtils.preserve_volume(audio, underwater)
        except Exception as e:
            print(f"Underwater effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_whisper(audio, sr, intensity=0.7):
        try:
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                cutoff = min(max(600 / nyquist, 0.01), 0.99)
                b, a = signal.butter(2, cutoff, btype='high')
                whisper = signal.filtfilt(b, a, audio)
            else:
                whisper = audio
            noise_level = 0.2 * intensity
            breath = np.random.normal(0, noise_level, len(audio))
            envelope = np.abs(whisper)
            window_size = min(int(sr * 0.01), len(envelope) // 10) or 1
            smoothed_envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
            breath *= smoothed_envelope
            result = whisper + breath
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Whisper effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_demon(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, -6 * intensity)
            distorted = VoiceProfileUtils._apply_distortion(pitched, 0.5 * intensity)
            t = np.arange(len(audio)) / sr
            rumble = np.sin(2 * np.pi * 40 * t) * 0.4 * intensity
            result = distorted + rumble
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Demon effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_angel(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, 3 * intensity)
            harmonic = VoiceProfileUtils._apply_harmonics(pitched, sr, 0.5 * intensity)
            reverb = VoiceProfileUtils._apply_reverb(harmonic, sr, 0.7 * intensity, 0.9, 0.3)
            return VoiceProfileUtils.preserve_volume(audio, reverb)
        except Exception as e:
            print(f"Angel effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_alien(audio, sr, intensity=0.7):
        try:
            vibrato = VoiceProfileUtils._apply_vibrato(audio, sr, 0.4 * intensity, 4.0)
            pitched = VoiceProfileUtils._apply_pitch_shift(vibrato, sr, 1.5 * intensity)
            reverb = VoiceProfileUtils._apply_reverb(pitched, sr, 0.5 * intensity, 0.7, 0.5)
            return VoiceProfileUtils.preserve_volume(audio, reverb)
        except Exception as e:
            print(f"Alien effect error: {e}")
            return audio
            
    @staticmethod
    def _apply_darth_vader(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, -4 * intensity)
            slowed = VoiceProfileUtils._improved_speed_change(pitched, 1.25)
            indices = np.linspace(0, len(slowed) - 1, len(audio))
            resampled = np.interp(indices, np.arange(len(slowed)), slowed) if len(slowed) > 0 else audio
            if SCIPY_AVAILABLE:
                nyquist = sr / 2
                cutoff = min(max(2500 / nyquist, 0.01), 0.99)
                b, a = signal.butter(2, cutoff, btype='low')
                filtered = signal.filtfilt(b, a, resampled)
            else:
                filtered = resampled
            echo = VoiceProfileUtils._apply_echo(filtered, sr, 0.3 * intensity, 0.4)
            distorted = VoiceProfileUtils._apply_distortion(echo, 0.3 * intensity)
            t = np.arange(len(distorted)) / sr
            breath = 1.0 + 0.15 * intensity * np.sin(2 * np.pi * 0.4 * t)
            result = distorted * breath
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Darth Vader effect error: {e}")
            return audio
            
    @staticmethod
    def _apply_chipmunk(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, 6 * intensity)
            formanted = VoiceProfileUtils._apply_formant_shift(pitched, sr, 1.5 * intensity)
            return VoiceProfileUtils.preserve_volume(audio, formanted)
        except Exception as e:
            print(f"Chipmunk effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_deep_voice(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, -5 * intensity)
            formanted = VoiceProfileUtils._apply_formant_shift(pitched, sr, -1.2 * intensity)
            eq = VoiceProfileUtils._apply_eq(formanted, sr, 0.4 * intensity, 0, 0)
            return VoiceProfileUtils.preserve_volume(audio, eq)
        except Exception as e:
            print(f"Deep voice effect error: {e}")
            return audio
            
    @staticmethod
    def _apply_child_voice(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, 3 * intensity)
            formanted = VoiceProfileUtils._apply_formant_shift(pitched, sr, 2 * intensity)
            bright = VoiceProfileUtils._apply_eq(formanted, sr, 0, 0.3 * intensity, 0.2 * intensity)
            noise = np.random.normal(0, 0.1 * intensity, len(audio))
            envelope = np.abs(bright)
            window_size = min(int(sr * 0.01), len(envelope) // 10) or 1
            smoothed_envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
            breath_noise = noise * smoothed_envelope * 0.2
            result = bright + breath_noise
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Child voice effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_elder_voice(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, -1 * intensity)
            formanted = VoiceProfileUtils._apply_formant_shift(pitched, sr, -0.5 * intensity)
            vibrato = VoiceProfileUtils._apply_vibrato(formanted, sr, 0.3 * intensity, 3.0)
            eq = VoiceProfileUtils._apply_eq(vibrato, sr, 0.3 * intensity, -0.2 * intensity, -0.3 * intensity)
            return VoiceProfileUtils.preserve_volume(audio, eq)
        except Exception as e:
            print(f"Elder voice effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_teen_voice(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, 1.5 * intensity)
            formanted = VoiceProfileUtils._apply_formant_shift(pitched, sr, 1.0 * intensity)
            eq = VoiceProfileUtils._apply_eq(formanted, sr, 0, 0.2 * intensity, 0.1 * intensity)
            return VoiceProfileUtils.preserve_volume(audio, eq)
        except Exception as e:
            print(f"Teen voice effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_adult_voice(audio, sr, intensity=0.7):
        try:
            compression = VoiceProfileUtils._apply_compression(audio, 0.1 * intensity)
            eq = VoiceProfileUtils._apply_eq(compression, sr, 0, 0.1 * intensity, 0)
            return VoiceProfileUtils.preserve_volume(audio, eq)
        except Exception as e:
            print(f"Adult voice effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_feminine_voice(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, 2 * intensity)
            formanted = VoiceProfileUtils._apply_formant_shift(pitched, sr, 1.5 * intensity)
            bright = VoiceProfileUtils._apply_eq(formanted, sr, 0, 0.2 * intensity, 0.2 * intensity)
            
            # Add breathiness for feminine voice
            noise = np.random.normal(0, 0.1 * intensity, len(audio))
            envelope = np.abs(bright)
            window_size = min(int(sr * 0.01), len(envelope) // 10) or 1
            smoothed_envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
            breath_noise = noise * smoothed_envelope * 0.2
            
            result = bright + breath_noise
            return VoiceProfileUtils.preserve_volume(audio, result)
        except Exception as e:
            print(f"Feminine voice effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_masculine_voice(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, -2 * intensity)
            formanted = VoiceProfileUtils._apply_formant_shift(pitched, sr, -1.5 * intensity)
            eq = VoiceProfileUtils._apply_eq(formanted, sr, bass=0.1 * intensity, mid=-0.1 * intensity, treble=-0.1 * intensity)
            warmth = VoiceProfileUtils._apply_warmth(eq, 0.1 * intensity)
            return VoiceProfileUtils.preserve_volume(audio, warmth)
        except Exception as e:
            print(f"Masculine voice effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_movie_trailer(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, -1 * intensity)
            eq = VoiceProfileUtils._apply_eq(pitched, sr, 0.5 * intensity, 0, 0)
            compression = VoiceProfileUtils._apply_compression(eq, 0.8 * intensity)
            reverb = VoiceProfileUtils._apply_reverb(compression, sr, 0.3 * intensity)
            echo = VoiceProfileUtils._apply_echo(reverb, sr, 0.2 * intensity)
            return VoiceProfileUtils.preserve_volume(audio, echo)
        except Exception as e:
            print(f"Movie trailer effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_singer(audio, sr, intensity=0.7):
        try:
            auto_tune = VoiceProfileUtils._apply_auto_tune(audio, sr, 0.5 * intensity)
            reverb = VoiceProfileUtils._apply_reverb(auto_tune, sr, 0.3 * intensity)
            compression = VoiceProfileUtils._apply_compression(reverb, 0.6 * intensity)
            harmonics = VoiceProfileUtils._apply_harmonics(compression, sr, 0.4 * intensity)
            vibrato = VoiceProfileUtils._apply_vibrato(harmonics, sr, 0.2 * intensity)
            return VoiceProfileUtils.preserve_volume(audio, vibrato)
        except Exception as e:
            print(f"Singer effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_tv_announcer(audio, sr, intensity=0.7):
        try:
            compression = VoiceProfileUtils._apply_compression(audio, 0.8 * intensity)
            eq = VoiceProfileUtils._apply_eq(compression, sr, 0.3 * intensity, 0.5 * intensity, 0)
            warmth = VoiceProfileUtils._apply_warmth(eq, 0.3 * intensity)
            return VoiceProfileUtils.preserve_volume(audio, warmth)
        except Exception as e:
            print(f"TV Announcer effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_fantasy_elf(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, 3 * intensity)
            formant = VoiceProfileUtils._apply_formant_shift(pitched, sr, 1.2 * intensity)
            reverb = VoiceProfileUtils._apply_reverb(formant, sr, 0.4 * intensity)
            harmonic = VoiceProfileUtils._apply_harmonics(reverb, sr, 0.6 * intensity)
            vibrato = VoiceProfileUtils._apply_vibrato(harmonic, sr, 0.3 * intensity)
            return VoiceProfileUtils.preserve_volume(audio, vibrato)
        except Exception as e:
            print(f"Fantasy Elf effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_orc(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, -4 * intensity)
            formant = VoiceProfileUtils._apply_formant_shift(pitched, sr, -1.0 * intensity)
            distorted = VoiceProfileUtils._apply_distortion(formant, 0.4 * intensity)
            eq = VoiceProfileUtils._apply_eq(distorted, sr, 0.7 * intensity, 0, 0)
            return VoiceProfileUtils.preserve_volume(audio, eq)
        except Exception as e:
            print(f"Orc effect error: {e}")
            return audio
            
    @staticmethod
    def _apply_monster(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, -6 * intensity)
            formant = VoiceProfileUtils._apply_formant_shift(pitched, sr, -1.5 * intensity)
            distorted = VoiceProfileUtils._apply_distortion(formant, 0.6 * intensity)
            eq = VoiceProfileUtils._apply_eq(distorted, sr, 0.5 * intensity, 0, 0)
            return VoiceProfileUtils.preserve_volume(audio, eq)
        except Exception as e:
            print(f"Monster effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_ghost(audio, sr, intensity=0.7):
        try:
            pitched = VoiceProfileUtils._apply_pitch_shift(audio, sr, 1 * intensity)
            reverb = VoiceProfileUtils._apply_reverb(pitched, sr, 0.7 * intensity)
            formant = VoiceProfileUtils._apply_formant_shift(reverb, sr, 0.5 * intensity)
            tremolo = VoiceProfileUtils._apply_tremolo(formant, 0.3 * intensity, sr)
            vibrato = VoiceProfileUtils._apply_vibrato(tremolo, sr, 0.2 * intensity)
            return VoiceProfileUtils.preserve_volume(audio, vibrato)
        except Exception as e:
            print(f"Ghost effect error: {e}")
            return audio
    
    @staticmethod
    def _apply_radio_host(audio, sr, intensity=0.7):
        try:
            compression = VoiceProfileUtils._apply_compression(audio, 0.7 * intensity)
            warmth = VoiceProfileUtils._apply_warmth(compression, 0.4 * intensity)
            eq = VoiceProfileUtils._apply_eq(warmth, sr, 0.2 * intensity, 0.3 * intensity, 0)
            return VoiceProfileUtils.preserve_volume(audio, eq)
        except Exception as e:
            print(f"Radio Host effect error: {e}")
            return audio