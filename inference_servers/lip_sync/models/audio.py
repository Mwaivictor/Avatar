"""
Audio processing utilities for Wav2Lip.

Handles mel spectrogram computation matching the Wav2Lip training pipeline:
  - 16kHz sample rate
  - 80 mel filter banks
  - 800-sample FFT window (50ms)
  - 200-sample hop length (12.5ms)
"""

import numpy as np
from scipy.signal import get_window
from scipy.fft import fft


# ━━━━━━━━━━━━━━━━━━━━ Mel Spectrogram ━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _create_mel_filterbank(sample_rate, n_fft, n_mels=80, fmin=55.0, fmax=7600.0):
    """Create mel-scale triangular filterbank matrix."""
    fmax = min(fmax, sample_rate / 2.0)
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]

        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    return filterbank


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 800,
    hop_length: int = 200,
    n_mels: int = 80,
    fmin: float = 55.0,
    fmax: float = 7600.0,
) -> np.ndarray:
    """
    Compute mel spectrogram from audio waveform.

    Args:
        audio: Float32 mono audio (-1 to 1).
        sample_rate: Audio sample rate.
        n_fft: FFT window size.
        hop_length: Hop between frames.
        n_mels: Number of mel filter banks.
        fmin: Minimum frequency for mel scale.
        fmax: Maximum frequency for mel scale.

    Returns:
        Mel spectrogram of shape (n_mels, T).
    """
    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # STFT
    window = get_window("hann", n_fft, fftbins=True)
    n_frames = 1 + (len(audio) - n_fft) // hop_length

    if n_frames < 1:
        # Pad short audio
        audio = np.pad(audio, (0, n_fft - len(audio) + hop_length))
        n_frames = 1

    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for t in range(n_frames):
        start = t * hop_length
        frame = audio[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * window
        spectrum = np.fft.rfft(windowed)
        stft_matrix[:, t] = spectrum

    # Power spectrogram
    power_spec = np.abs(stft_matrix) ** 2

    # Mel filterbank
    mel_basis = _create_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    mel_spec = np.dot(mel_basis, power_spec)

    # Log compression
    mel_spec = np.log(np.maximum(mel_spec, 1e-5))

    return mel_spec.astype(np.float32)


def mel_for_wav2lip(audio: np.ndarray, sample_rate: int = 16000,
                    num_frames: int = 16) -> np.ndarray:
    """
    Compute mel spectrogram and reshape for Wav2Lip input.

    Returns:
        Array of shape (1, 80, num_frames) suitable for the audio encoder.
    """
    mel = compute_mel_spectrogram(audio, sample_rate)

    # Pad or truncate to target frame count
    if mel.shape[1] < num_frames:
        mel = np.pad(mel, ((0, 0), (0, num_frames - mel.shape[1])))
    else:
        mel = mel[:, :num_frames]

    return mel.reshape(1, 80, num_frames)
