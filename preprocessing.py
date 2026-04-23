from __future__ import annotations
import io
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, medfilt

def _decode_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Safely decodes bytes to numpy."""
    try:
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception:
        # Fallback to silence if the file is corrupted
        return np.zeros(100, dtype=np.float32), 44100

def _encode_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """Encodes numpy to bytes. Fixes Mac socket limits with PCM_16."""
    buffer = io.BytesIO()
    # PCM_16 cuts file size in half, preventing socket overload
    sf.write(buffer, y, sr, format="WAV", subtype='PCM_16')
    buffer.seek(0) # CRITICAL for Flask
    return buffer.getvalue()

def apply_filters(y: np.ndarray, sr: int) -> np.ndarray:
    """Standard cleaning pipeline for Disko Bay audio."""
    y = np.nan_to_num(y)
    
    # 1. Spectral Subtraction
    stft = librosa.stft(y)
    mag, phase = librosa.magphase(stft)
    noise_est = np.mean(mag[:, :min(mag.shape[1], 10)], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_est, 0.0)
    y = librosa.istft(mag_clean * phase)
    
    # 2. Band-pass (10Hz to 48kHz)
    nyq = 0.5 * sr
    low, high = max(0.001, 10 / nyq), min(0.999, 48000 / nyq)
    b, a = butter(5, [low, high], btype='band')
    y = lfilter(b, a, y)
    
    # 3. Median Filter & Normalization
    y = medfilt(y, kernel_size=3)
    return librosa.util.normalize(np.nan_to_num(y))

# --- PREPROCESSING STEPS ---

def filter_raw(wav_bytes: bytes) -> bytes:
    """Must return strictly bytes."""
    return wav_bytes

def filter_whale_clean(wav_bytes: bytes) -> bytes:
    """Must return strictly bytes."""
    y, sr = _decode_wav_bytes(wav_bytes)
    
    # SAFETY NET: Limit to 60 seconds to prevent file size crashes
    y = y[:sr * 60] 
    
    y_filtered = apply_filters(y, sr)
    return _encode_wav_bytes(y_filtered, sr)

# --- DASHBOARD REGISTRY ---
# dashboard.py reads this dictionary to create the buttons
PREPROCESSING_STEPS = {
    "raw": filter_raw,
    "whale_clean": filter_whale_clean,
}

# If your dashboard explicitly calls this wrapper function instead of the dict:
def run_preprocessing_step(wav_bytes: bytes, step_name: str) -> bytes:
    """
    STRICT CONTRACT: Input bytes, Output bytes.
    No tuples, no JSON strings.
    """
    func = PREPROCESSING_STEPS.get(step_name, filter_raw)
    return func(wav_bytes)
