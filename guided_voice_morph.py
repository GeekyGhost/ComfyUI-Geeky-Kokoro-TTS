"""
Professional guided voice morphing module for ComfyUI Geeky Kokoro TTS.
Uses extracted audio features from a guide audio to morph the input voice.
"""
import numpy as np
import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Import required libraries with fallbacks
try:
    import scipy.signal as signal
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, some morphing features will be limited")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available, using fallback implementations")

# Import local modules
try:
    from .audio_feature_extraction import AudioFeatureExtractor
    from .audio_utils import stft, istft
    from .voice_profiles_utils import VoiceProfileUtils
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    logger.warning("Local utility modules not fully available")


class GuidedVoiceMorph:
    """
    Professional voice morphing guided by reference audio.
    Enables autotune-like effects and comprehensive voice transformation.
    """

    @staticmethod
    def dynamic_time_warp(source_features: np.ndarray,
                         target_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Dynamic Time Warping to align two feature sequences.

        Parameters:
        -----------
        source_features : np.ndarray
            Source feature sequence [num_frames_source, feature_dim]
        target_features : np.ndarray
            Target feature sequence [num_frames_target, feature_dim]

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Alignment path (source_indices, target_indices)
        """
        try:
            # Ensure 2D arrays
            if source_features.ndim == 1:
                source_features = source_features.reshape(-1, 1)
            if target_features.ndim == 1:
                target_features = target_features.reshape(-1, 1)

            n, m = len(source_features), len(target_features)

            # Initialize cost matrix
            cost = np.full((n + 1, m + 1), np.inf)
            cost[0, 0] = 0

            # Compute cost matrix
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    dist = np.linalg.norm(source_features[i-1] - target_features[j-1])
                    cost[i, j] = dist + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])

            # Backtrack to find optimal path
            i, j = n, m
            path_source, path_target = [i-1], [j-1]

            while i > 1 or j > 1:
                if i == 1:
                    j -= 1
                elif j == 1:
                    i -= 1
                else:
                    min_cost = min(cost[i-1, j-1], cost[i-1, j], cost[i, j-1])
                    if cost[i-1, j-1] == min_cost:
                        i, j = i-1, j-1
                    elif cost[i-1, j] == min_cost:
                        i = i-1
                    else:
                        j = j-1

                path_source.append(i-1)
                path_target.append(j-1)

            path_source.reverse()
            path_target.reverse()

            logger.debug(f"DTW alignment: {len(path_source)} points")
            return np.array(path_source), np.array(path_target)

        except Exception as e:
            logger.error(f"DTW error: {e}")
            # Return linear alignment as fallback
            n, m = len(source_features), len(target_features)
            source_indices = np.linspace(0, n-1, min(n, m)).astype(int)
            target_indices = np.linspace(0, m-1, min(n, m)).astype(int)
            return source_indices, target_indices

    @staticmethod
    def morph_pitch_to_guide(audio: np.ndarray, sr: int,
                            guide_pitch: np.ndarray,
                            source_pitch: Optional[np.ndarray] = None,
                            morph_amount: float = 1.0) -> np.ndarray:
        """
        Morph audio pitch to match guide audio pitch contour.

        Parameters:
        -----------
        audio : np.ndarray
            Source audio signal
        sr : int
            Sample rate
        guide_pitch : np.ndarray
            Target pitch contour in Hz
        source_pitch : np.ndarray, optional
            Source pitch contour (will be extracted if None)
        morph_amount : float
            Amount of morphing (0.0 = no change, 1.0 = full morph)

        Returns:
        --------
        np.ndarray
            Pitch-morphed audio
        """
        try:
            if morph_amount < 0.01:
                return audio

            # Extract source pitch if not provided
            if source_pitch is None:
                if UTILS_AVAILABLE:
                    source_pitch, _ = AudioFeatureExtractor.extract_pitch_contour(audio, sr)
                else:
                    logger.warning("Cannot extract pitch, returning original audio")
                    return audio

            # Align guide pitch to source pitch length using DTW
            if len(guide_pitch) != len(source_pitch):
                source_idx, guide_idx = GuidedVoiceMorph.dynamic_time_warp(
                    source_pitch.reshape(-1, 1),
                    guide_pitch.reshape(-1, 1)
                )
                aligned_guide = guide_pitch[guide_idx]
            else:
                aligned_guide = guide_pitch

            # Ensure same length
            min_len = min(len(source_pitch), len(aligned_guide))
            source_pitch = source_pitch[:min_len]
            aligned_guide = aligned_guide[:min_len]

            # Compute pitch shift ratios for each frame
            pitch_ratios = aligned_guide / (source_pitch + 1e-10)

            # Smooth pitch ratios to avoid artifacts
            if SCIPY_AVAILABLE:
                from scipy.ndimage import gaussian_filter1d
                pitch_ratios = gaussian_filter1d(pitch_ratios, sigma=2)

            # Apply morphing amount
            pitch_ratios = 1.0 + (pitch_ratios - 1.0) * morph_amount

            # Clamp to reasonable range
            pitch_ratios = np.clip(pitch_ratios, 0.5, 2.0)

            # Apply time-varying pitch shift
            result = GuidedVoiceMorph._apply_time_varying_pitch_shift(
                audio, sr, pitch_ratios
            )

            logger.info(f"Pitch morphing applied with {morph_amount:.2f} intensity")
            return result

        except Exception as e:
            logger.error(f"Pitch morphing error: {e}")
            return audio

    @staticmethod
    def _apply_time_varying_pitch_shift(audio: np.ndarray, sr: int,
                                       pitch_ratios: np.ndarray,
                                       frame_length: int = 2048) -> np.ndarray:
        """
        Apply time-varying pitch shift to audio.

        Parameters:
        -----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        pitch_ratios : np.ndarray
            Pitch shift ratios for each frame
        frame_length : int
            Frame length for processing

        Returns:
        --------
        np.ndarray
            Pitch-shifted audio
        """
        try:
            hop_length = frame_length // 4

            if LIBROSA_AVAILABLE:
                # Use phase vocoder for time-varying pitch shift
                D = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)

                # Interpolate pitch ratios to match STFT frames
                num_frames = D.shape[1]
                if len(pitch_ratios) != num_frames:
                    x_old = np.linspace(0, 1, len(pitch_ratios))
                    x_new = np.linspace(0, 1, num_frames)
                    if SCIPY_AVAILABLE:
                        interp_func = interp1d(x_old, pitch_ratios, kind='linear',
                                             fill_value='extrapolate')
                        pitch_ratios_interp = interp_func(x_new)
                    else:
                        pitch_ratios_interp = np.interp(x_new, x_old, pitch_ratios)
                else:
                    pitch_ratios_interp = pitch_ratios

                # Apply pitch shifting frame by frame
                D_shifted = np.zeros_like(D)
                for i in range(num_frames):
                    ratio = pitch_ratios_interp[i]
                    # Shift frequencies by the ratio
                    for freq_bin in range(D.shape[0]):
                        new_bin = int(freq_bin * ratio)
                        if 0 <= new_bin < D.shape[0]:
                            D_shifted[new_bin, i] += D[freq_bin, i]

                # Inverse STFT
                result = librosa.istft(D_shifted, hop_length=hop_length, length=len(audio))

            else:
                # Fallback: Apply average pitch shift
                avg_ratio = np.mean(pitch_ratios)
                semitones = 12 * np.log2(avg_ratio)

                if UTILS_AVAILABLE:
                    result = VoiceProfileUtils._apply_pitch_shift(audio, sr, semitones)
                else:
                    # Very basic fallback
                    result = audio

            return result

        except Exception as e:
            logger.error(f"Time-varying pitch shift error: {e}")
            return audio

    @staticmethod
    def morph_formants_to_guide(audio: np.ndarray, sr: int,
                               guide_formants: np.ndarray,
                               source_formants: Optional[np.ndarray] = None,
                               morph_amount: float = 1.0) -> np.ndarray:
        """
        Morph audio formants to match guide audio formants.

        Parameters:
        -----------
        audio : np.ndarray
            Source audio signal
        sr : int
            Sample rate
        guide_formants : np.ndarray
            Target formant frequencies [num_frames, num_formants]
        source_formants : np.ndarray, optional
            Source formant frequencies
        morph_amount : float
            Amount of morphing (0.0 = no change, 1.0 = full morph)

        Returns:
        --------
        np.ndarray
            Formant-morphed audio
        """
        try:
            if morph_amount < 0.01:
                return audio

            # Extract source formants if not provided
            if source_formants is None:
                if UTILS_AVAILABLE:
                    source_formants = AudioFeatureExtractor.extract_formants(audio, sr)
                else:
                    logger.warning("Cannot extract formants, returning original audio")
                    return audio

            # Align guide formants to source formants length
            if len(guide_formants) != len(source_formants):
                if len(guide_formants) > len(source_formants):
                    # Downsample guide
                    indices = np.linspace(0, len(guide_formants)-1, len(source_formants))
                    indices = indices.astype(int)
                    aligned_guide = guide_formants[indices]
                else:
                    # Upsample guide
                    indices = np.linspace(0, len(guide_formants)-1, len(source_formants))
                    if SCIPY_AVAILABLE:
                        aligned_guide = np.zeros((len(source_formants), guide_formants.shape[1]))
                        for i in range(guide_formants.shape[1]):
                            interp_func = interp1d(
                                np.arange(len(guide_formants)),
                                guide_formants[:, i],
                                kind='linear',
                                fill_value='extrapolate'
                            )
                            aligned_guide[:, i] = interp_func(indices)
                    else:
                        aligned_guide = np.zeros((len(source_formants), guide_formants.shape[1]))
                        for i in range(guide_formants.shape[1]):
                            aligned_guide[:, i] = np.interp(
                                indices,
                                np.arange(len(guide_formants)),
                                guide_formants[:, i]
                            )
            else:
                aligned_guide = guide_formants

            # Compute average formant shift
            # Take median ratio across formants and frames
            formant_ratios = aligned_guide / (source_formants + 1e-10)
            avg_ratio = np.median(formant_ratios)

            # Convert to formant shift parameter (rough approximation)
            formant_shift = np.log2(avg_ratio) * 2 * morph_amount

            # Clamp to reasonable range
            formant_shift = np.clip(formant_shift, -5, 5)

            # Apply formant shift
            if UTILS_AVAILABLE:
                result = VoiceProfileUtils._apply_formant_shift(audio, sr, formant_shift)
            else:
                result = audio

            logger.info(f"Formant morphing applied: shift {formant_shift:.2f}")
            return result

        except Exception as e:
            logger.error(f"Formant morphing error: {e}")
            return audio

    @staticmethod
    def morph_spectral_envelope(audio: np.ndarray, sr: int,
                               guide_envelope: np.ndarray,
                               source_envelope: Optional[np.ndarray] = None,
                               morph_amount: float = 1.0) -> np.ndarray:
        """
        Morph spectral envelope to match guide audio.

        Parameters:
        -----------
        audio : np.ndarray
            Source audio signal
        sr : int
            Sample rate
        guide_envelope : np.ndarray
            Target spectral envelope [freq_bins, num_frames]
        source_envelope : np.ndarray, optional
            Source spectral envelope
        morph_amount : float
            Amount of morphing (0.0 = no change, 1.0 = full morph)

        Returns:
        --------
        np.ndarray
            Spectrally-morphed audio
        """
        try:
            if morph_amount < 0.01:
                return audio

            n_fft = 2048
            hop_length = n_fft // 4

            # Extract source envelope if not provided
            if source_envelope is None:
                if UTILS_AVAILABLE:
                    source_envelope, _ = AudioFeatureExtractor.extract_spectral_envelope(
                        audio, sr, n_fft
                    )
                else:
                    logger.warning("Cannot extract spectral envelope")
                    return audio

            # Compute STFT of audio
            if LIBROSA_AVAILABLE:
                D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            elif UTILS_AVAILABLE:
                D = stft(audio, n_fft=n_fft, hop_length=hop_length)
            else:
                return audio

            mag = np.abs(D)
            phase = np.angle(D)

            # Align guide envelope to source envelope shape
            if guide_envelope.shape != source_envelope.shape:
                # Resize using interpolation
                aligned_guide = np.zeros_like(source_envelope)
                for i in range(source_envelope.shape[0]):
                    src_idx = int(i * guide_envelope.shape[0] / source_envelope.shape[0])
                    src_idx = min(src_idx, guide_envelope.shape[0] - 1)
                    for j in range(source_envelope.shape[1]):
                        tgt_idx = int(j * guide_envelope.shape[1] / source_envelope.shape[1])
                        tgt_idx = min(tgt_idx, guide_envelope.shape[1] - 1)
                        aligned_guide[i, j] = guide_envelope[src_idx, tgt_idx]
            else:
                aligned_guide = guide_envelope

            # Align to magnitude shape
            if aligned_guide.shape != mag.shape:
                temp = np.zeros_like(mag)
                min_freq = min(aligned_guide.shape[0], mag.shape[0])
                min_time = min(aligned_guide.shape[1], mag.shape[1])
                temp[:min_freq, :min_time] = aligned_guide[:min_freq, :min_time]
                aligned_guide = temp

            # Compute envelope ratio
            envelope_ratio = aligned_guide / (source_envelope + 1e-10)

            # Smooth the ratio
            if SCIPY_AVAILABLE:
                from scipy.ndimage import gaussian_filter
                envelope_ratio = gaussian_filter(envelope_ratio, sigma=[2, 1])

            # Apply morphing
            morph_ratio = 1.0 + (envelope_ratio - 1.0) * morph_amount

            # Apply to magnitude
            morphed_mag = mag * morph_ratio

            # Reconstruct
            D_morphed = morphed_mag * np.exp(1j * phase)

            if LIBROSA_AVAILABLE:
                result = librosa.istft(D_morphed, hop_length=hop_length, length=len(audio))
            elif UTILS_AVAILABLE:
                result = istft(D_morphed, hop_length=hop_length, length=len(audio))
            else:
                result = audio

            logger.info(f"Spectral envelope morphing applied with {morph_amount:.2f} intensity")
            return result

        except Exception as e:
            logger.error(f"Spectral envelope morphing error: {e}")
            return audio

    @staticmethod
    def morph_amplitude_envelope(audio: np.ndarray, sr: int,
                                guide_amplitude: np.ndarray,
                                source_amplitude: Optional[np.ndarray] = None,
                                morph_amount: float = 1.0) -> np.ndarray:
        """
        Morph amplitude envelope to match guide audio dynamics.

        Parameters:
        -----------
        audio : np.ndarray
            Source audio signal
        sr : int
            Sample rate
        guide_amplitude : np.ndarray
            Target amplitude envelope
        source_amplitude : np.ndarray, optional
            Source amplitude envelope
        morph_amount : float
            Amount of morphing (0.0 = no change, 1.0 = full morph)

        Returns:
        --------
        np.ndarray
            Amplitude-morphed audio
        """
        try:
            if morph_amount < 0.01:
                return audio

            frame_length = 2048
            hop_length = frame_length // 4

            # Extract source amplitude if not provided
            if source_amplitude is None:
                if UTILS_AVAILABLE:
                    source_amplitude, _ = AudioFeatureExtractor.extract_amplitude_envelope(
                        audio, sr, frame_length
                    )
                else:
                    logger.warning("Cannot extract amplitude envelope")
                    return audio

            # Align guide amplitude to source amplitude length
            if len(guide_amplitude) != len(source_amplitude):
                x_old = np.linspace(0, 1, len(guide_amplitude))
                x_new = np.linspace(0, 1, len(source_amplitude))
                if SCIPY_AVAILABLE:
                    interp_func = interp1d(x_old, guide_amplitude, kind='linear',
                                         fill_value='extrapolate')
                    aligned_guide = interp_func(x_new)
                else:
                    aligned_guide = np.interp(x_new, x_old, guide_amplitude)
            else:
                aligned_guide = guide_amplitude

            # Compute amplitude ratio
            amp_ratio = aligned_guide / (source_amplitude + 1e-10)

            # Smooth ratio
            if SCIPY_AVAILABLE:
                from scipy.ndimage import gaussian_filter1d
                amp_ratio = gaussian_filter1d(amp_ratio, sigma=2)

            # Apply morphing amount
            amp_ratio = 1.0 + (amp_ratio - 1.0) * morph_amount

            # Clamp to reasonable range
            amp_ratio = np.clip(amp_ratio, 0.1, 10.0)

            # Expand ratio to audio sample length
            ratio_expanded = np.interp(
                np.arange(len(audio)),
                np.linspace(0, len(audio), len(amp_ratio)),
                amp_ratio
            )

            # Apply amplitude scaling
            result = audio * ratio_expanded

            logger.info(f"Amplitude envelope morphing applied with {morph_amount:.2f} intensity")
            return result

        except Exception as e:
            logger.error(f"Amplitude envelope morphing error: {e}")
            return audio

    @staticmethod
    def guided_voice_morph(audio: np.ndarray, sr: int,
                          guide_audio: np.ndarray, guide_sr: int,
                          pitch_morph: float = 0.0,
                          formant_morph: float = 0.0,
                          spectral_morph: float = 0.0,
                          amplitude_morph: float = 0.0) -> np.ndarray:
        """
        Apply comprehensive guided voice morphing using guide audio features.

        Parameters:
        -----------
        audio : np.ndarray
            Source audio signal
        sr : int
            Sample rate of source audio
        guide_audio : np.ndarray
            Guide audio signal
        guide_sr : int
            Sample rate of guide audio
        pitch_morph : float
            Pitch morphing amount (0.0 to 1.0)
        formant_morph : float
            Formant morphing amount (0.0 to 1.0)
        spectral_morph : float
            Spectral morphing amount (0.0 to 1.0)
        amplitude_morph : float
            Amplitude morphing amount (0.0 to 1.0)

        Returns:
        --------
        np.ndarray
            Morphed audio signal
        """
        try:
            logger.info("Starting guided voice morphing...")

            # Resample guide audio if needed
            if guide_sr != sr:
                if LIBROSA_AVAILABLE:
                    guide_audio = librosa.resample(guide_audio, orig_sr=guide_sr, target_sr=sr)
                else:
                    logger.warning("Cannot resample guide audio, using original")

            result = audio.copy()

            # Extract all features from guide audio
            if UTILS_AVAILABLE:
                guide_features = AudioFeatureExtractor.extract_all_features(guide_audio, sr)
            else:
                logger.error("Feature extraction not available")
                return audio

            # Apply morphing in order
            if pitch_morph > 0.01 and 'pitch' in guide_features:
                logger.info(f"Applying pitch morphing: {pitch_morph:.2f}")
                result = GuidedVoiceMorph.morph_pitch_to_guide(
                    result, sr, guide_features['pitch'], morph_amount=pitch_morph
                )

            if formant_morph > 0.01 and 'formants' in guide_features:
                logger.info(f"Applying formant morphing: {formant_morph:.2f}")
                result = GuidedVoiceMorph.morph_formants_to_guide(
                    result, sr, guide_features['formants'], morph_amount=formant_morph
                )

            if spectral_morph > 0.01 and 'spectral_envelope' in guide_features:
                logger.info(f"Applying spectral morphing: {spectral_morph:.2f}")
                result = GuidedVoiceMorph.morph_spectral_envelope(
                    result, sr, guide_features['spectral_envelope'], morph_amount=spectral_morph
                )

            if amplitude_morph > 0.01 and 'amplitude_envelope' in guide_features:
                logger.info(f"Applying amplitude morphing: {amplitude_morph:.2f}")
                result = GuidedVoiceMorph.morph_amplitude_envelope(
                    result, sr, guide_features['amplitude_envelope'], morph_amount=amplitude_morph
                )

            # Preserve volume
            if UTILS_AVAILABLE:
                result = VoiceProfileUtils.preserve_volume(audio, result)

            logger.info("Guided voice morphing completed successfully")
            return result

        except Exception as e:
            logger.error(f"Guided voice morphing error: {e}")
            import traceback
            traceback.print_exc()
            return audio
