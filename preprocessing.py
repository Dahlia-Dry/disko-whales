from __future__ import annotations
import io
import json
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import butter, lfilter, medfilt

# --- CONSTANTS & RULES ---
SPECIES_RULES = {
    "fin": [(13, 40), (85, 140)],
    "humpback": [(40, 6000)],
    "beluga": [(2000, 5900)],
    "narwhal": [(300, 18000), (47000, 49000)] 
}

def _decode_wav_bytes(wav_bytes: bytes):
    """Decodes bytes safely. Adds a check for empty inputs."""
    try:
        # Using a context manager for the BytesIO object
        with io.BytesIO(wav_bytes) as bio:
            y, sr = librosa.load(bio, sr=None, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception as e:
        print(f"CRITICAL: Decode failed: {e}")
        # Return a tiny silent array to prevent script crash
        return np.zeros(100, dtype=np.float32), 44100

def _encode_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """Encodes to bytes. Forces PCM_16 to reduce data size for Mac sockets."""
    buffer = io.BytesIO()
    # PCM_16 is half the size of float32, reducing Errno 22 risk
    sf.write(buffer, y, sr, format="WAV", subtype='PCM_16')
    buffer.seek(0) 
    return buffer.getvalue()

# --- FILTERING HELPERS ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_whale_filters(y, sr):
    y = np.nan_to_num(y)
    
    # Spectral Subtraction
    stft = librosa.stft(y)
    mag, phase = librosa.magphase(stft)
    noise_est = np.mean(mag[:, :min(mag.shape[1], 10)], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_est, 0.0)
    y = librosa.istft(mag_clean * phase)
    
    # Dynamic Band-pass
    try:
        b, a = butter_bandpass(10, 48000, sr)
        y = lfilter(b, a, y)
    except ValueError:
        b, a = butter_bandpass(10, (sr/2)-1, sr)
        y = lfilter(b, a, y)
        
    y = medfilt(y, kernel_size=3)
    # Clip to prevent extreme peaks from breaking normalization
    y = np.clip(y, -1.0, 1.0)
    y = librosa.util.normalize(np.nan_to_num(y))
    return y

# --- CLASSIFICATION LOGIC ---

def analyze_and_classify(y, sr):
    # If file is empty or too quiet, return empty results
    if len(y) < 100 or np.max(np.abs(y)) < 1e-4:
        return {"event": list(SPECIES_RULES.keys()), "probability": [0,0,0,0], "predicted": [False]*4}

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
        "probability": [round(counts[s] / total_events, 4) if total_events > 0 else 0 for s in counts.keys()],
        "predicted": [counts[s] > 0 for s in counts.keys()]
    }

# --- DASHBOARD PIPELINE ---

def whale_preprocessing_pipeline(wav_bytes: bytes) -> tuple[bytes, str]:
    y, sr = _decode_wav_bytes(wav_bytes)
    
    # PROTECTION: If audio is longer than 5 minutes, crop it for the dashboard preview.
    # This prevents the "OSError [Errno 22]" by keeping the response size manageable.
    max_duration_sec = 300 
    if len(y) > sr * max_duration_sec:
        y = y[:sr * max_duration_sec]

    # 1. Apply Filters
    y_filtered = apply_whale_filters(y, sr)
    
    # 2. Extract Metadata
    metadata = analyze_and_classify(y_filtered, sr)
    
    # 3. Encode
    processed_bytes = _encode_wav_bytes(y_filtered, sr)
    
    return processed_bytes, json.dumps(metadata)

# --- REGISTRY ---

def filter_raw(wav_bytes: bytes) -> bytes:
    return wav_bytes

PREPROCESSING_STEPS = {
    "raw": filter_raw,
    "whale_classification": whale_preprocessing_pipeline,
}

def run_preprocessing_step(wav_bytes: bytes, step_name: str):
    """Main dashboard entry point."""
    if step_name == "whale_classification":
        processed, label = whale_preprocessing_pipeline(wav_bytes)
        return processed, label
    else:
        func = PREPROCESSING_STEPS.get(step_name, filter_raw)
        processed = func(wav_bytes)
        # Ensure we return a string for the label
        label = {"raw": "Raw audio"}.get(step_name, f"Applied step: {step_name}")
        return processed, label
