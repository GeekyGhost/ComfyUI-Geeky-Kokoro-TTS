"""
Advanced audio feature extraction for professional voice morphing and analysis.
This module provides tools for extracting pitch, formants, spectral envelope, and other features.
"""
import numpy as np
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Import required libraries with fallbacks
try:
    import scipy.signal as signal
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, some features will be limited")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available, using fallback implementations")


class AudioFeatureExtractor:
    """
    Professional-grade audio feature extraction for voice morphing and analysis.
    """

    @staticmethod
    def extract_pitch_contour(audio: np.ndarray, sr: int,
                             fmin: float = 50, fmax: float = 500,
                             frame_length: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch contour from audio using autocorrelation or librosa.

        Parameters:
        -----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        fmin : float
            Minimum frequency to consider
        fmax : float
            Maximum frequency to consider
        frame_length : int
            Frame length for analysis

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Pitch values in Hz and time points
        """
        try:
            if LIBROSA_AVAILABLE:
                # Use librosa's pyin for more accurate pitch tracking
                f0 = librosa.pyin(
                    audio,
                    fmin=fmin,
                    fmax=fmax,
                    sr=sr,
                    frame_length=frame_length
                )[0]
                # Remove NaN values by interpolation
                nans = np.isnan(f0)
                if not nans.all():
                    x = np.arange(len(f0))
                    f0[nans] = np.interp(x[nans], x[~nans], f0[~nans])
                else:
                    f0 = np.full_like(f0, 150.0)  # Default to 150 Hz

                times = librosa.frames_to_time(
                    np.arange(len(f0)),
                    sr=sr,
                    hop_length=frame_length // 4
                )
            else:
                # Fallback: Use autocorrelation-based pitch detection
                hop_length = frame_length // 4
                num_frames = 1 + (len(audio) - frame_length) // hop_length
                f0 = np.zeros(num_frames)

                for i in range(num_frames):
                    start = i * hop_length
                    end = start + frame_length
                    frame = audio[start:end]

                    # Autocorrelation
                    corr = np.correlate(frame, frame, mode='full')
                    corr = corr[len(corr)//2:]

                    # Find peaks in autocorrelation
                    min_lag = int(sr / fmax)
                    max_lag = int(sr / fmin)

                    if max_lag < len(corr):
                        search_range = corr[min_lag:max_lag]
                        if len(search_range) > 0:
                            peak = np.argmax(search_range) + min_lag
                            if peak > 0:
                                f0[i] = sr / peak
                            else:
                                f0[i] = 150.0
                        else:
                            f0[i] = 150.0
                    else:
                        f0[i] = 150.0

                times = np.arange(num_frames) * hop_length / sr

            logger.debug(f"Extracted pitch contour: {len(f0)} frames")
            return f0, times

        except Exception as e:
            logger.error(f"Pitch extraction error: {e}")
            # Return flat pitch contour as fallback
            num_frames = max(1, len(audio) // (frame_length // 4))
            return np.full(num_frames, 150.0), np.linspace(0, len(audio)/sr, num_frames)

    @staticmethod
    def extract_formants(audio: np.ndarray, sr: int,
                        num_formants: int = 4) -> np.ndarray:
        """
        Extract formant frequencies from audio using LPC (Linear Predictive Coding).

        Parameters:
        -----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        num_formants : int
            Number of formants to extract

        Returns:
        --------
        np.ndarray
            Formant frequencies [num_frames, num_formants]
        """
        try:
            if not SCIPY_AVAILABLE:
                logger.warning("scipy not available, returning default formants")
                num_frames = max(1, len(audio) // 512)
                # Default formant values for neutral voice
                defaults = np.array([700, 1220, 2600, 3500])[:num_formants]
                return np.tile(defaults, (num_frames, 1))

            frame_length = 1024
            hop_length = frame_length // 2
            num_frames = 1 + (len(audio) - frame_length) // hop_length

            formants = np.zeros((num_frames, num_formants))
            lpc_order = 2 + sr // 1000  # Rule of thumb for LPC order

            for i in range(num_frames):
                start = i * hop_length
                end = start + frame_length
                frame = audio[start:end]

                # Pre-emphasis
                emphasized = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])

                # Apply window
                windowed = emphasized * np.hamming(len(emphasized))

                # Compute LPC coefficients
                try:
                    # Autocorrelation method
                    r = np.correlate(windowed, windowed, mode='full')
                    r = r[len(r)//2:]
                    r = r[:lpc_order + 1]

                    # Levinson-Durbin recursion
                    a = np.zeros(lpc_order + 1)
                    a[0] = 1.0
                    e = r[0]

                    for k in range(1, lpc_order + 1):
                        alpha = -np.sum(a[:k] * r[k:0:-1]) / e
                        a_new = np.zeros(k + 1)
                        a_new[0] = 1.0
                        a_new[1:k] = a[1:k] + alpha * a[k-1:0:-1]
                        a_new[k] = alpha
                        a = a_new
                        e = e * (1 - alpha * alpha)

                    # Find roots of LPC polynomial
                    roots = np.roots(a)

                    # Convert roots to frequencies
                    angles = np.angle(roots)
                    freqs = angles * (sr / (2 * np.pi))

                    # Keep only positive frequencies
                    freqs = freqs[freqs > 0]
                    freqs = np.sort(freqs)

                    # Extract formants
                    if len(freqs) >= num_formants:
                        formants[i, :] = freqs[:num_formants]
                    else:
                        # Pad with default values
                        defaults = np.array([700, 1220, 2600, 3500])[:num_formants]
                        formants[i, :len(freqs)] = freqs
                        formants[i, len(freqs):] = defaults[len(freqs):]

                except Exception as lpc_error:
                    # Use default formants for this frame
                    defaults = np.array([700, 1220, 2600, 3500])[:num_formants]
                    formants[i, :] = defaults

            logger.debug(f"Extracted {num_formants} formants over {num_frames} frames")
            return formants

        except Exception as e:
            logger.error(f"Formant extraction error: {e}")
            num_frames = max(1, len(audio) // 512)
            defaults = np.array([700, 1220, 2600, 3500])[:num_formants]
            return np.tile(defaults, (num_frames, 1))

    @staticmethod
    def extract_spectral_envelope(audio: np.ndarray, sr: int,
                                  n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract spectral envelope using cepstral smoothing.

        Parameters:
        -----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        n_fft : int
            FFT size

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Spectral envelope [num_frames, n_fft//2+1] and frequencies
        """
        try:
            hop_length = n_fft // 4

            if LIBROSA_AVAILABLE:
                # Compute STFT
                D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
                mag = np.abs(D)

                # Smooth using cepstral analysis
                envelope = np.zeros_like(mag)
                quefrency_limit = int(sr / 50)  # Limit to remove fine structure

                for i in range(mag.shape[1]):
                    spectrum = mag[:, i]
                    log_spectrum = np.log(spectrum + 1e-10)

                    # Cepstrum
                    cepstrum = np.fft.ifft(log_spectrum).real

                    # Lifter (smooth by keeping only low quefrencies)
                    liftered = np.copy(cepstrum)
                    if quefrency_limit < len(liftered):
                        liftered[quefrency_limit:-quefrency_limit] = 0

                    # Back to frequency domain
                    smoothed = np.exp(np.fft.fft(liftered).real[:len(spectrum)])
                    envelope[:, i] = smoothed

                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            else:
                # Fallback implementation
                num_frames = 1 + (len(audio) - n_fft) // hop_length
                envelope = np.zeros((n_fft // 2 + 1, num_frames))

                for i in range(num_frames):
                    start = i * hop_length
                    end = start + n_fft
                    frame = audio[start:end]

                    # Apply window
                    windowed = frame * np.hanning(len(frame))

                    # FFT
                    spectrum = np.abs(np.fft.rfft(windowed))
                    envelope[:, i] = spectrum

                freqs = np.fft.rfftfreq(n_fft, 1/sr)

            logger.debug(f"Extracted spectral envelope: {envelope.shape}")
            return envelope, freqs

        except Exception as e:
            logger.error(f"Spectral envelope extraction error: {e}")
            num_frames = max(1, len(audio) // (n_fft // 4))
            envelope = np.ones((n_fft // 2 + 1, num_frames))
            freqs = np.fft.rfftfreq(n_fft, 1/sr)
            return envelope, freqs

    @staticmethod
    def extract_amplitude_envelope(audio: np.ndarray, sr: int,
                                   frame_length: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract amplitude envelope from audio.

        Parameters:
        -----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        frame_length : int
            Frame length for analysis

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Amplitude envelope and time points
        """
        try:
            hop_length = frame_length // 4

            if LIBROSA_AVAILABLE:
                # Use RMS energy
                rms = librosa.feature.rms(
                    y=audio,
                    frame_length=frame_length,
                    hop_length=hop_length
                )[0]
                times = librosa.frames_to_time(
                    np.arange(len(rms)),
                    sr=sr,
                    hop_length=hop_length
                )
            else:
                # Fallback: Manual RMS calculation
                num_frames = 1 + (len(audio) - frame_length) // hop_length
                rms = np.zeros(num_frames)

                for i in range(num_frames):
                    start = i * hop_length
                    end = start + frame_length
                    frame = audio[start:end]
                    rms[i] = np.sqrt(np.mean(frame ** 2))

                times = np.arange(num_frames) * hop_length / sr

            logger.debug(f"Extracted amplitude envelope: {len(rms)} frames")
            return rms, times

        except Exception as e:
            logger.error(f"Amplitude envelope extraction error: {e}")
            num_frames = max(1, len(audio) // (frame_length // 4))
            return np.ones(num_frames), np.linspace(0, len(audio)/sr, num_frames)

    @staticmethod
    def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC (Mel-frequency cepstral coefficients) for timbre analysis.

        Parameters:
        -----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        n_mfcc : int
            Number of MFCCs to extract

        Returns:
        --------
        np.ndarray
            MFCC features [n_mfcc, num_frames]
        """
        try:
            if LIBROSA_AVAILABLE:
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            else:
                # Simplified MFCC without librosa (less accurate but functional)
                n_fft = 2048
                hop_length = n_fft // 4
                num_frames = 1 + (len(audio) - n_fft) // hop_length
                mfcc = np.zeros((n_mfcc, num_frames))

                for i in range(num_frames):
                    start = i * hop_length
                    end = start + n_fft
                    frame = audio[start:end]

                    # Apply window and FFT
                    windowed = frame * np.hanning(len(frame))
                    spectrum = np.abs(np.fft.rfft(windowed))

                    # Simplified mel scaling (not true mel filterbank)
                    log_spectrum = np.log(spectrum + 1e-10)

                    # DCT to get cepstral coefficients
                    from scipy.fftpack import dct
                    cepstrum = dct(log_spectrum, type=2, norm='ortho')
                    mfcc[:, i] = cepstrum[:n_mfcc]

            logger.debug(f"Extracted {n_mfcc} MFCCs over {mfcc.shape[1]} frames")
            return mfcc

        except Exception as e:
            logger.error(f"MFCC extraction error: {e}")
            num_frames = max(1, len(audio) // 512)
            return np.zeros((n_mfcc, num_frames))

    @staticmethod
    def extract_all_features(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract all audio features for comprehensive analysis.

        Parameters:
        -----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary containing all extracted features
        """
        try:
            logger.info("Extracting comprehensive audio features...")

            features = {}

            # Pitch
            f0, f0_times = AudioFeatureExtractor.extract_pitch_contour(audio, sr)
            features['pitch'] = f0
            features['pitch_times'] = f0_times

            # Formants
            formants = AudioFeatureExtractor.extract_formants(audio, sr)
            features['formants'] = formants

            # Spectral envelope
            spec_env, spec_freqs = AudioFeatureExtractor.extract_spectral_envelope(audio, sr)
            features['spectral_envelope'] = spec_env
            features['spectral_freqs'] = spec_freqs

            # Amplitude envelope
            amp_env, amp_times = AudioFeatureExtractor.extract_amplitude_envelope(audio, sr)
            features['amplitude_envelope'] = amp_env
            features['amplitude_times'] = amp_times

            # MFCCs for timbre
            mfcc = AudioFeatureExtractor.extract_mfcc(audio, sr)
            features['mfcc'] = mfcc

            logger.info(f"Extracted {len(features)} feature types successfully")
            return features

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}
