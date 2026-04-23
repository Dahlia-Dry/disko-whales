from __future__ import annotations
import io
import json
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, medfilt

# --- RULES ---
SPECIES_RULES = {
    "Fin Whale": [(13, 40), (85, 140)],
    "Humpback": [(40, 6000)],
    "Beluga": [(2000, 5900)],
    "Narwhal": [(300, 18000), (47000, 49000)] 
}

def _decode_wav_bytes(wav_bytes: bytes):
    try:
        # We load at original SR for classification accuracy...
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception:
        return np.zeros(100, dtype=np.float32), 44100

def _encode_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """Socket-Safe Encoder."""
    # REDUCTION 1: Downsample to 22050Hz for the dashboard preview. 
    # This reduces data size by 4x if the original was 96kHz.
    if sr > 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050
        
    buffer = io.BytesIO()
    # REDUCTION 2: Force PCM_16 (Standard 16-bit)
    sf.write(buffer, y, sr, format="WAV", subtype='PCM_16')
    buffer.seek(0)
    # REDUCTION 3: Cast to strict bytes object
    return bytes(buffer.getvalue())

def apply_whale_filters(y, sr):
    y = np.nan_to_num(y)
    # Spectral Subtraction
    stft = librosa.stft(y)
    mag, phase = librosa.magphase(stft)
    noise_est = np.mean(mag[:, :min(mag.shape[1], 10)], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_est, 0.0)
    y = librosa.istft(mag_clean * phase)
    # Bandpass
    nyq = 0.5 * sr
    low, high = max(0.001, 10 / nyq), min(0.999, 48000 / nyq)
    b, a = butter(5, [low, high], btype='band')
    y = lfilter(b, a, y)
    # Median
    y = medfilt(y, kernel_size=3)
    return librosa.util.normalize(np.nan_to_num(y))

def run_classification_logic(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    total = len(peaks)
    counts = {s: 0 for s in SPECIES_RULES.keys()}
    for p in peaks:
        start = max(0, int(librosa.frames_to_samples(p) - sr*0.05))
        end = min(len(y), int(librosa.frames_to_samples(p) + sr*0.05))
        segment = y[start:end]
        if len(segment) > 10:
            fft_res = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1/sr)
            peak_f = freqs[np.argmax(fft_res)]
            for s, bands in SPECIES_RULES.items():
                if any(l <= peak_f <= h for l, h in bands):
                    counts[s] += 1
                    break
    return {
        "event": list(counts.keys()),
        "probability": [round(counts[s]/total, 2) if total > 0 else 0 for s in counts.keys()],
        "predicted": [1 if counts[s] > 0 else 0 for s in counts.keys()]
    }

# --- PRIMARY STEPS ---

def filter_raw(wav_bytes: bytes) -> bytes:
    return wav_bytes

def filter_whale_classification(wav_bytes: bytes) -> tuple[bytes, str]:
    y, sr = _decode_wav_bytes(wav_bytes)
    
    # REDUCTION 4: Limit preview to 30 seconds.
    # Dashboard users don't need 10 minutes of audio at once.
    y = y[:sr * 30] 
    
    y_filt = apply_whale_filters(y, sr)
    meta = run_classification_logic(y_filt, sr)
    
    # Return downsampled/trimmed bytes and the metadata JSON
    return _encode_wav_bytes(y_filt, sr), json.dumps(meta)

# --- REGISTRY ---

PREPROCESSING_STEPS = {
    "raw": filter_raw,
    "whale_classification": filter_whale_classification,
}

def run_preprocessing_step(wav_bytes: bytes, step_name: str):
    """Called by dashboard.py"""
    if step_name == "whale_classification":
        # Returns (bytes, json_string)
        return filter_whale_classification(wav_bytes)
    
    # Handle 'raw' or other simple steps
    func = PREPROCESSING_STEPS.get(step_name, filter_raw)
    res = func(wav_bytes)
    return res, "Raw audio"
