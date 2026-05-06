"""
Preprocessing functions for whale sound analysis.
Each function takes raw .wav bytes and returns processed .wav bytes.
The dashboard.py will call `run_preprocessing_step` with the appropriate step name and parameters.
"""
from __future__ import annotations
import io
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, medfilt, sosfiltfilt

# --- INTERNAL UTILITIES ---

def _decode_wav_bytes(wav_bytes: bytes):
    """Decodes raw bytes into a numpy array."""
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
    return y.astype(np.float32), int(sr)

def _encode_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """Encodes numpy array to WAV bytes. PCM_16 is used to reduce data size."""
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV", subtype='PCM_16')
    buffer.seek(0)
    return buffer.getvalue()

# --- FILTERING LOGIC ---

def apply_whale_filters(y: np.ndarray, sr: int) -> np.ndarray:
    """Cleans audio by removing noise and applying a bandpass filter."""
    y = np.nan_to_num(y)
    
    # 1. Spectral Subtraction (Noise reduction)
    stft = librosa.stft(y)
    mag, phase = librosa.magphase(stft)
    if mag.shape[1] > 0:
        # Use a low percentile over all frames rather than the first frames.
        noise_est = np.percentile(mag, 20, axis=1, keepdims=True)
    else:
        noise_est = np.zeros((mag.shape[0], 1), dtype=mag.dtype)

    # Keep subtraction conservative to avoid removing true calls.
    subtraction_strength = 0.9
    floor_ratio = 0.08
    mag_clean = np.maximum(mag - subtraction_strength * noise_est, floor_ratio * mag)
    y = librosa.istft(mag_clean * phase)
    
    # 2. Band-pass (Standard whale frequency range)
    nyq = 0.5 * sr
    low_hz, high_hz = 30.0, 2000.0
    low = max(0.001, low_hz / nyq)
    high = min(0.999, high_hz / nyq)
    if low < high:
        sos = butter(5, [low, high], btype='band', output='sos')
        y = sosfiltfilt(sos, y)
    
    # 3. Cleanup & Normalization
    y = medfilt(y, kernel_size=3)
    y = np.nan_to_num(y.astype(np.float32))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        y = librosa.util.normalize(y)
    return y.astype(np.float32)

# --- PREPROCESSING STEPS REGISTRY ---

def filter_raw(wav_bytes: bytes) -> bytes:
    """Returns the original audio unchanged."""
    return wav_bytes

def filter_whale_clean(wav_bytes: bytes) -> bytes:
    """Full cleaning pipeline."""
    y, sr = _decode_wav_bytes(wav_bytes)
    
    # CRITICAL: Mac Socket Safety
    # If the file is huge, the socket will throw OSError [Errno 22].
    # We limit the dashboard preview to the first 60 seconds.
    if len(y) > sr * 60:
        y = y[:sr * 60]
        
    y_filtered = apply_whale_filters(y, sr)
    return _encode_wav_bytes(y_filtered, sr)

def filter_lowpass(wav_bytes: bytes, cutoff_hz: float = 1000.0) -> bytes:
    """Applies a Butterworth low-pass filter at the given cutoff frequency."""
    y, sr = _decode_wav_bytes(wav_bytes)
    nyq = 0.5 * sr
    cutoff = min(max(float(cutoff_hz) / nyq, 0.001), 0.999)
    sos = butter(8, cutoff, btype="low", output="sos")
    y = sosfiltfilt(sos, y)
    y = librosa.util.normalize(np.nan_to_num(y.astype(np.float32)))
    return _encode_wav_bytes(y, sr)


def filter_highpass(wav_bytes: bytes, cutoff_hz: float = 100.0) -> bytes:
    """Applies a Butterworth high-pass filter at the given cutoff frequency."""
    y, sr = _decode_wav_bytes(wav_bytes)
    nyq = 0.5 * sr
    cutoff = min(max(float(cutoff_hz) / nyq, 0.001), 0.999)
    sos = butter(8, cutoff, btype="high", output="sos")
    y = sosfiltfilt(sos, y)
    y = librosa.util.normalize(np.nan_to_num(y.astype(np.float32)))
    return _encode_wav_bytes(y, sr)


def filter_crop(wav_bytes: bytes, start_s: float = 0.0, end_s: float | None = None) -> bytes:
    """Crops the audio to the given time range (in seconds)."""
    y, sr = _decode_wav_bytes(wav_bytes)
    i0 = max(0, int(float(start_s) * sr))
    i1 = int(float(end_s) * sr) if end_s is not None else len(y)
    i1 = min(i1, len(y))
    return _encode_wav_bytes(y[i0:i1], sr)


# This dictionary is read by dashboard.py to generate buttons
PREPROCESSING_STEPS = {
    "raw": filter_raw,
    "whale_clean": filter_whale_clean,
    "lowpass": filter_lowpass,
    "highpass": filter_highpass,
}

# --- MANDATORY DASHBOARD INTERFACE ---

def run_preprocessing_step(wav_bytes: bytes, step_name: str, params: dict | None = None) -> bytes:
    """
    This is the specific function dashboard.py line 28 is looking for.
    It must take bytes and return bytes.
    """
    # Look up the function in the dictionary, default to filter_raw if not found
    func = PREPROCESSING_STEPS.get(step_name, filter_raw)

    # Execute the function and return the resulting bytes
    if params:
        return func(wav_bytes, **params)
    return func(wav_bytes)
