import io
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, medfilt

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def _decode_wav_bytes(wav_bytes: bytes):
    # Safely load the audio
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
    return y, sr

def _encode_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    # Safely write the audio back to bytes
    buffer = io.BytesIO()
    # PCM_16 keeps the file size small to prevent socket crashes
    sf.write(buffer, y, sr, format="WAV", subtype='PCM_16')
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------------------------------------
# PREPROCESSING STEPS (Must return exactly 'bytes')
# ---------------------------------------------------------
def filter_raw(wav_bytes: bytes) -> bytes:
    """Returns the original file unchanged."""
    return wav_bytes

def filter_whale_clean(wav_bytes: bytes) -> bytes:
    """Cleans the audio and strictly returns bytes."""
    y, sr = _decode_wav_bytes(wav_bytes)
    
    # 1. Limit to 60 seconds to prevent file-size crashes on Mac sockets
    if len(y) > sr * 60:
        y = y[:sr * 60]
        
    y = np.nan_to_num(y)

    # 2. Spectral Subtraction (Remove background hiss)
    stft = librosa.stft(y)
    mag, phase = librosa.magphase(stft)
    noise_est = np.mean(mag[:, :min(mag.shape[1], 10)], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_est, 0.0)
    y = librosa.istft(mag_clean * phase)

    # 3. Band-pass (10Hz to 48kHz)
    nyq = 0.5 * sr
    low, high = max(0.001, 10 / nyq), min(0.999, 48000 / nyq)
    b, a = butter(5, [low, high], btype='band')
    y = lfilter(b, a, y)

    # 4. Normalize
    y = medfilt(y, kernel_size=3)
    y = librosa.util.normalize(np.nan_to_num(y))
    
    # STRICT RETURN: Only bytes
    return _encode_wav_bytes(y, sr)

# ---------------------------------------------------------
# DASHBOARD REGISTRY
# ---------------------------------------------------------
PREPROCESSING_STEPS = {
    "raw": filter_raw,
    "whale_clean": filter_whale_clean,
}
