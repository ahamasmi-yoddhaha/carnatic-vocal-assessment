import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from typing import Dict, Tuple

# Standard note frequencies (Hz) for octave 3 & 4 with A4 = 440 Hz
SCALE_TO_HZ = {
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56,
    "E3": 164.81, "F3": 174.61, "F#3": 185.00, "G3": 196.00,
    "G#3": 207.65, "A3": 220.00, "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00,
    "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88
}


def estimate_pitch_acf(frame: np.ndarray, sr: int, min_f0: float = 80.0, max_f0: float = 650.0) -> Tuple[float, float]:
    """
    Normalized Autocorrelation (ACF) pitch and confidence estimator.
    """
    min_lag = int(sr / max_f0)
    max_lag = int(sr / min_f0)
    
    if len(frame) < max_lag:
        return 0.0, 0.0
        
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(frame)-1:]
    
    energy = corr[0]
    if energy <= 1e-8:
        return 0.0, 0.0
        
    norm_corr = corr / energy
    search_window = norm_corr[min_lag:min(max_lag, len(norm_corr))]
    
    if len(search_window) == 0:
        return 0.0, 0.0
        
    peak_offset = np.argmax(search_window)
    peak_val = search_window[peak_offset]
    peak_lag = peak_offset + min_lag
    
    # Voiced threshold
    if peak_val > 0.35 and peak_lag > 0:
        f0 = sr / peak_lag
        return float(f0), float(peak_val)
    else:
        return 0.0, float(max(0.0, peak_val))


def extract_features_from_audio(
    audio_path: str,
    s_scale: str,
    frame_size_sec: float = 0.1,
    hop_size_sec: float = 0.05
) -> pd.DataFrame:
    """
    Extracts time-domain, spectral, and relative pitch features from an audio file.
    """
    sr, y = wavfile.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32) / 32768.0

    tonic_hz = SCALE_TO_HZ.get(s_scale, 220.0)  # Default A3 = 220 Hz
    frame_len = int(frame_size_sec * sr)
    hop_len = int(hop_size_sec * sr)

    n_frames = (len(y) - frame_len) // hop_len + 1
    if n_frames <= 0:
        return pd.DataFrame()

    shape = (n_frames, frame_len)
    strides = (y.strides[0] * hop_len, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)

    # 1. Time-Domain Features
    rms = np.sqrt(np.mean(frames**2, axis=1))
    delta_rms = np.concatenate([[0.0], np.diff(rms)])
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    # 2. Spectral Features
    window = np.hanning(frame_len)
    fft_mag = np.abs(np.fft.rfft(frames * window, axis=1))
    power_spec = fft_mag ** 2
    total_power = np.sum(power_spec, axis=1) + 1e-10

    freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)
    spectral_centroid = np.sum(fft_mag * freqs, axis=1) / (np.sum(fft_mag, axis=1) + 1e-10)

    # Spectral Rolloff (85% energy)
    cum_power = np.cumsum(power_spec, axis=1)
    rolloff_threshold = 0.85 * total_power[:, np.newaxis]
    rolloff_idx = np.argmax(cum_power >= rolloff_threshold, axis=1)
    spectral_rolloff = freqs[rolloff_idx]

    # Spectral Flatness (geometric mean / arithmetic mean of power spectrum)
    log_power = np.log(power_spec + 1e-10)
    geom_mean = np.exp(np.mean(log_power, axis=1))
    arith_mean = np.mean(power_spec, axis=1) + 1e-10
    spectral_flatness = geom_mean / arith_mean

    # Band Energies
    low_band_mask = freqs < 1000
    mid_band_mask = (freqs >= 1000) & (freqs < 3000)
    high_band_mask = freqs >= 3000

    energy_low = np.sum(power_spec[:, low_band_mask], axis=1) / total_power
    energy_mid = np.sum(power_spec[:, mid_band_mask], axis=1) / total_power
    energy_high = np.sum(power_spec[:, high_band_mask], axis=1) / total_power

    # 3. Pitch & Carnatic Relative Cents
    f0_list = []
    conf_list = []
    for f in frames:
        f0, conf = estimate_pitch_acf(f, sr)
        f0_list.append(f0)
        conf_list.append(conf)

    f0_arr = np.array(f0_list, dtype=np.float32)
    conf_arr = np.array(conf_list, dtype=np.float32)
    is_voiced = (f0_arr > 0).astype(np.int32)

    # Relative cents relative to tonic Sa
    cents = np.zeros_like(f0_arr)
    valid_f0 = f0_arr > 0
    cents[valid_f0] = 1200.0 * np.log2(f0_arr[valid_f0] / tonic_hz)

    # Position in 12-semitone octave (0 to 1200)
    cents_octave = np.zeros_like(cents)
    cents_octave[valid_f0] = np.mod(cents[valid_f0], 1200.0)

    # Deviation from nearest semitone (-50 to +50 cents)
    cents_dev = np.zeros_like(cents)
    cents_dev[valid_f0] = np.mod(cents[valid_f0] + 50.0, 100.0) - 50.0
    abs_cents_dev = np.abs(cents_dev)

    # Pitch stability (rolling standard deviation over 5 frames)
    cents_series = pd.Series(cents)
    pitch_stability = cents_series.rolling(window=5, min_periods=1).std().fillna(0.0).values
    delta_f0 = np.concatenate([[0.0], np.diff(f0_arr)])

    df = pd.DataFrame({
        "frame_idx": np.arange(n_frames),
        "rms": np.round(rms, 5),
        "delta_rms": np.round(delta_rms, 5),
        "zcr": np.round(zcr, 4),
        "spectral_centroid": np.round(spectral_centroid, 1),
        "spectral_rolloff": np.round(spectral_rolloff, 1),
        "spectral_flatness": np.round(spectral_flatness, 5),
        "energy_low": np.round(energy_low, 4),
        "energy_mid": np.round(energy_mid, 4),
        "energy_high": np.round(energy_high, 4),
        "f0": np.round(f0_arr, 2),
        "delta_f0": np.round(delta_f0, 2),
        "pitch_confidence": np.round(conf_arr, 4),
        "is_voiced": is_voiced,
        "cents": np.round(cents, 2),
        "cents_octave": np.round(cents_octave, 2),
        "cents_dev": np.round(cents_dev, 2),
        "abs_cents_dev": np.round(abs_cents_dev, 2),
        "pitch_stability": np.round(pitch_stability, 2)
    })

    return df

