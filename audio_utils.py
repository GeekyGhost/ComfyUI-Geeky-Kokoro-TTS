"""
Fallback audio processing utilities for when resampy and other optional dependencies aren't available.
These implementations are designed to work with standard libraries like NumPy and SciPy.
"""
import numpy as np
import scipy.signal as signal
import warnings
from functools import lru_cache

# ============== PITCH SHIFTING FUNCTIONS ==============

def simple_pitch_shift(audio, sr, n_steps, bins_per_octave=12):
    """
    Basic pitch shifting using resampling technique, no resampy required
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    sr : int
        Sample rate
    n_steps : float
        Number of steps to shift (positive = up, negative = down)
    bins_per_octave : int
        Number of bins per octave
        
    Returns:
    --------
    numpy.ndarray
        Pitch-shifted audio
    """
    if abs(n_steps) < 0.01:
        return audio
        
    # Convert steps to rate
    rate = 2.0 ** (-n_steps / bins_per_octave)
    
    # First adjust the speed (stretch in time domain)
    # Slower playback = lower pitch, faster playback = higher pitch
    indices = np.arange(0, len(audio), rate)
    indices = indices[indices < len(audio)]
    stretched = np.interp(indices, np.arange(len(audio)), audio)
    
    # Then resample back to original length
    time_stretched = len(stretched)
    indices = np.linspace(0, time_stretched - 1, len(audio))
    result = np.interp(indices, np.arange(time_stretched), stretched)
    
    return result

def scipy_resample_fixed_length(audio, factor):
    """
    Resample audio using scipy.signal.resample with fixed output length
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    factor : float
        Resampling factor (>1 = upsample, <1 = downsample)
        
    Returns:
    --------
    numpy.ndarray
        Resampled audio
    """
    output_size = int(len(audio) / factor)
    return signal.resample(audio, output_size)

def stft_phase_vocoder(audio, sr, n_steps, bins_per_octave=12):
    """
    Phase vocoder pitch shifting using STFT, more advanced than simple resampling
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    sr : int
        Sample rate
    n_steps : float
        Number of steps to shift (positive = up, negative = down)
    bins_per_octave : int
        Number of bins per octave
        
    Returns:
    --------
    numpy.ndarray
        Pitch-shifted audio
    """
    if abs(n_steps) < 0.01:
        return audio
        
    # Convert steps to rate
    rate = 2.0 ** (-n_steps / bins_per_octave)
    
    # STFT parameters
    n_fft = 2048
    hop_length = n_fft // 4
    
    # Compute STFT
    D = stft(audio, n_fft=n_fft, hop_length=hop_length)
    
    # Create new spectrogram with adjusted phase progression
    time_steps = D.shape[1]
    new_time_steps = int(time_steps / rate)
    
    # Phase advance
    phase_adv = np.linspace(0, np.pi * rate, D.shape[0])[:, np.newaxis]
    
    # Time-stretch the magnitude and adjust phase
    if rate < 1:  # Upsampling in frequency (downsampling in time)
        # Stretch the spectrogram in time (fewer columns)
        D_stretch = np.zeros((D.shape[0], new_time_steps), dtype=complex)
        
        # Interpolate magnitude
        mag = np.abs(D)
        phase = np.angle(D)
        
        # Time points for original and stretched
        time_orig = np.arange(time_steps)
        time_new = np.linspace(0, time_steps - 1, new_time_steps)
        
        for f in range(D.shape[0]):
            mag_interp = np.interp(time_new, time_orig, mag[f, :])
            phase_interp = np.interp(time_new, time_orig, phase[f, :])
            
            # Adjust phase progression for pitch shift
            phase_interp += phase_adv[f] * np.arange(new_time_steps)
            
            # Combine magnitude and adjusted phase
            D_stretch[f, :] = mag_interp * np.exp(1j * phase_interp)
    else:  # Downsampling in frequency (upsampling in time)
        # Stretch the spectrogram in time (more columns)
        D_stretch = np.zeros((D.shape[0], new_time_steps), dtype=complex)
        
        # Interpolate magnitude and phase
        mag = np.abs(D)
        phase = np.angle(D)
        
        # Time points for original and stretched
        time_orig = np.arange(time_steps)
        time_new = np.linspace(0, time_steps - 1, new_time_steps)
        
        for f in range(D.shape[0]):
            mag_interp = np.interp(time_new, time_orig, mag[f, :])
            phase_interp = np.interp(time_new, time_orig, phase[f, :])
            
            # Adjust phase progression for pitch shift
            phase_interp += phase_adv[f] * np.arange(new_time_steps)
            
            # Combine magnitude and adjusted phase
            D_stretch[f, :] = mag_interp * np.exp(1j * phase_interp)
    
    # Invert STFT
    y_shift = istft(D_stretch, hop_length=hop_length, length=len(audio))
    
    return y_shift

# ============== STFT/ISTFT FUNCTIONS ==============

def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann'):
    """
    Short-time Fourier transform using scipy for when librosa is not available
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
        
    # Create window
    if window == 'hann':
        fft_window = signal.windows.hann(win_length, sym=True)
    else:
        fft_window = signal.get_window(window, win_length, fftbins=True)
        
    # Pad the window to n_fft if needed
    if win_length < n_fft:
        padding = np.zeros(n_fft - win_length)
        fft_window = np.concatenate([fft_window, padding])
        
    # Number of frames
    n_frames = 1 + (len(y) - n_fft) // hop_length
    
    # Pre-allocate output
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
    
    # Compute STFT
    for i in range(n_frames):
        frame = y[i * hop_length:i * hop_length + n_fft]
        
        # Apply window
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
            
        windowed = frame * fft_window
        
        # FFT
        spectrum = np.fft.rfft(windowed)
        
        # Store
        stft_matrix[:, i] = spectrum
        
    return stft_matrix

def istft(stft_matrix, hop_length=None, win_length=None, window='hann', length=None):
    """
    Inverse short-time Fourier transform using scipy for when librosa is not available
    """
    n_fft = 2 * (stft_matrix.shape[0] - 1)
    
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    
    # Create window
    if window == 'hann':
        ifft_window = signal.windows.hann(win_length, sym=True)
    else:
        ifft_window = signal.get_window(window, win_length, fftbins=True)
    
    # Pad the window to n_fft if needed
    if win_length < n_fft:
        padding = np.zeros(n_fft - win_length)
        ifft_window = np.concatenate([ifft_window, padding])
    
    # Number of frames
    n_frames = stft_matrix.shape[1]
    
    # Expected output length
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    
    # Pre-allocate output
    y = np.zeros(expected_signal_len)
    
    # Pre-allocate window buffer
    window_buffer = np.zeros(expected_signal_len)
    
    # Compute ISTFT
    for i in range(n_frames):
        # IFFT
        ytmp = np.fft.irfft(stft_matrix[:, i])
        
        # Apply window
        ytmp = ytmp * ifft_window
        
        # Overlap-add
        start = i * hop_length
        end = start + n_fft
        y[start:end] += ytmp
        window_buffer[start:end] += ifft_window
    
    # Normalize by window overlap
    # Avoid division by zero
    window_buffer = np.maximum(window_buffer, 1e-10)
    y = y / window_buffer
    
    # Trim to original length if provided
    if length is not None:
        y = y[:length]
    
    return y

# ============== FORMANT SHIFTING FUNCTIONS ==============

def formant_shift_basic(audio, sr, shift_amount):
    """
    Basic formant shifting using spectral envelope manipulation
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    sr : int
        Sample rate
    shift_amount : float
        Shift amount (-1 to 1, negative = down, positive = up)
        
    Returns:
    --------
    numpy.ndarray
        Formant-shifted audio
    """
    if abs(shift_amount) < 0.1:
        return audio
    
    # STFT parameters
    n_fft = 2048
    hop_length = n_fft // 4
    
    # Compute STFT
    S = stft(audio, n_fft=n_fft, hop_length=hop_length)
    
    # Separate magnitude and phase
    mag = np.abs(S)
    phase = np.angle(S)
    
    # Shift formants by manipulating the magnitude spectrogram
    freq_bins = mag.shape[0]
    shift_bins = int(freq_bins * shift_amount * 0.1)
    
    # Initialize shifted magnitude spectrogram
    mag_shifted = np.zeros_like(mag)
    
    if shift_bins > 0:
        # Shift formants up
        mag_shifted[shift_bins:, :] = mag[:-shift_bins, :]
        # Smoothly blend the lowest frequencies
        blend_region = min(shift_bins, 20)
        for i in range(blend_region):
            weight = i / blend_region
            mag_shifted[i, :] = mag[i, :] * (1 - weight)
    elif shift_bins < 0:
        # Shift formants down
        shift_bins = abs(shift_bins)
        mag_shifted[:-shift_bins, :] = mag[shift_bins:, :]
        # Keep the very highest frequencies with reduced amplitude
        mag_shifted[-shift_bins:, :] = mag[-shift_bins:, :] * 0.5
    else:
        mag_shifted = mag
    
    # Recombine with original phase
    S_shifted = mag_shifted * np.exp(1j * phase)
    
    # ISTFT
    y_shifted = istft(S_shifted, hop_length=hop_length, length=len(audio))
    
    return y_shifted

# ============== UTILITY FUNCTIONS ==============

def amplitude_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """
    Convert amplitude spectrogram to dB-scaled spectrogram
    """
    magnitude = np.abs(S)
    
    ref_value = np.abs(ref)
    
    log_spec = 20.0 * np.log10(np.maximum(amin, magnitude) / ref_value)
    
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
        
    return log_spec

def db_to_amplitude(db, ref=1.0):
    """
    Convert dB-scaled spectrogram to amplitude spectrogram
    """
    return ref * 10.0 ** (db / 20.0)

def detect_available_libraries():
    """
    Detect which audio processing libraries are available
    """
    available_libs = {
        'librosa': False,
        'resampy': False,
        'soundfile': False,
        'numpy': True,  # NumPy is a requirement
        'scipy': True   # SciPy is a requirement
    }
    
    try:
        import librosa
        available_libs['librosa'] = True
    except ImportError:
        pass
    
    try:
        import resampy
        available_libs['resampy'] = True
    except ImportError:
        pass
        
    try:
        import soundfile
        available_libs['soundfile'] = True
    except ImportError:
        pass
    
    return available_libs

def get_optimal_pitch_shift_method():
    """
    Determine the best available pitch shifting method
    
    Returns:
    --------
    str
        Name of the best available method:
        - 'librosa_resampy': Best quality, requires librosa and resampy
        - 'stft_phase_vocoder': Good quality, no resampy required
        - 'simple': Basic quality, minimal dependencies
    """
    libs = detect_available_libraries()
    
    if libs['librosa'] and libs['resampy']:
        return 'librosa_resampy'
    elif libs['librosa']:
        return 'librosa_no_resampy'
    elif libs['scipy']:
        return 'stft_phase_vocoder'
    else:
        return 'simple'

def get_optimal_formant_shift_method():
    """
    Determine the best available formant shifting method
    
    Returns:
    --------
    str
        Name of the best available method:
        - 'librosa': Best quality, requires librosa
        - 'basic': Basic quality, minimal dependencies
    """
    libs = detect_available_libraries()
    
    if libs['librosa']:
        return 'librosa'
    else:
        return 'basic'