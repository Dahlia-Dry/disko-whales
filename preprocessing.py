from __future__ import annotations
import io
import json
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, medfilt

# --- RESEARCH-BASED RULES ---
SPECIES_RULES = {
    "fin": [(13, 40), (85, 140)],
    "humpback": [(40, 6000)],
    "beluga": [(2000, 5900)],
    "narwhal": [(300, 18000), (47000, 49000)] 
}

# --- INTERNAL UTILITIES ---

def _decode_wav_bytes(wav_bytes: bytes):
    """Safely decodes bytes to numpy."""
    try:
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception:
        return np.array([0.0], dtype=np.float32), 44100

def _encode_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """Encodes numpy to PCM_16 bytes. Fixes Errno 22 with seek(0)."""
    buffer = io.BytesIO()
    # PCM_16 significantly reduces data size to prevent Mac socket errors
    sf.write(buffer, y, sr, format="WAV", subtype='PCM_16')
    buffer.seek(0) # CRITICAL: Reset pointer so the dashboard can read the data
    return buffer.getvalue()

# --- PREPROCESSING STEPS ---

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

def filter_raw(wav_bytes: bytes) -> bytes:
    """Return input unchanged."""
    return wav_bytes

def filter_whale_clean(wav_bytes: bytes) -> bytes:
    """Preprocessing step that only cleans the audio."""
    y, sr = _decode_wav_bytes(wav_bytes)
    y_filtered = apply_filters(y, sr)
    return _encode_wav_bytes(y_filtered, sr)

# --- THE "HACKED" CLASSIFICATION STEP ---

def perform_classification(y: np.ndarray, sr: int) -> dict:
    """Generates the metadata dict you requested."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    
    total_events = len(peaks)
    counts = {s: 0 for s in SPECIES_RULES.keys()}
    
    for p in peaks:
        start = max(0, int(librosa.frames_to_samples(p) - sr*0.05))
        end = min(len(y), int(librosa.frames_to_samples(p) + sr*0.05))
        segment = y[start:end]
        
        if len(segment) > 10:
            fft_res = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1/sr)
            peak_f = freqs[np.argmax(fft_res)]
            
            for species, bands in SPECIES_RULES.items():
                if any(low <= peak_f <= high for low, high in bands):
                    counts[species] += 1
                    break

    return {
        "event": list(counts.keys()),
        "probability": [round(counts[s] / total_events, 3) if total_events > 0 else 0.0 for s in counts.keys()],
        "predicted": [1 if counts[s] > 0 else 0 for s in counts.keys()]
    }

# --- REGISTRY & INTERFACE ---

PREPROCESSING_STEPS = {
    "raw": filter_raw,
    "clean_audio": filter_whale_clean,
    "whale_classification": filter_whale_clean, # We use the same filter function
}

def run_preprocessing_step(wav_bytes: bytes, step_name: str):
    """
    Main interface for dashboard.py.
    Returns (processed_wav_bytes, label).
    """
    # 1. Get the audio processing function
    func = PREPROCESSING_STEPS.get(step_name, filter_raw)
    processed_wav = func(wav_bytes)
    
    # 2. If the user clicked 'whale_classification', calculate the metadata
    if step_name == "whale_classification":
        # We re-decode to analyze the filtered audio
        y, sr = _decode_wav_bytes(processed_wav)
        metadata = perform_classification(y, sr)
        # Return the JSON string as the label for the dashboard
        return processed_wav, json.dumps(metadata)
    
    # Default label logic for other steps
    label = {
        "raw": "Raw audio",
        "clean_audio": "Applied Cleaning Filters"
    }.get(step_name, f"Applied step: {step_name}")
    
    return processed_wav, label
