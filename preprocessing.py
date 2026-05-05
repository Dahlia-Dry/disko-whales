from __future__ import annotations
import io
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, medfilt

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
    noise_est = np.mean(mag[:, :min(mag.shape[1], 10)], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_est, 0.0)
    y = librosa.istft(mag_clean * phase)
    
    # 2. Band-pass (Standard whale frequency range)
    nyq = 0.5 * sr
    # Low cutoff: 30 Hz | High cutoff: 2,000 Hz (2 kHz)
    low, high = max(0.001, 30 / nyq), min(0.999, 2000 / nyq)
    b, a = butter(5, [low, high], btype='band')
    y = lfilter(b, a, y)
    
    # 3. Cleanup & Normalization
    y = medfilt(y, kernel_size=3)
    y = librosa.util.normalize(np.nan_to_num(y))
    return y

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

# This dictionary is read by dashboard.py to generate buttons
PREPROCESSING_STEPS = {
    "raw": filter_raw,
    "whale_clean": filter_whale_clean,
}

# --- MANDATORY DASHBOARD INTERFACE ---

def run_preprocessing_step(wav_bytes: bytes, step_name: str) -> bytes:
    """
    This is the specific function dashboard.py line 28 is looking for.
    It must take bytes and return bytes.
    """
    # Look up the function in the dictionary, default to filter_raw if not found
    func = PREPROCESSING_STEPS.get(step_name, filter_raw)
    
    # Execute the function and return the resulting bytes
    return func(wav_bytes)
