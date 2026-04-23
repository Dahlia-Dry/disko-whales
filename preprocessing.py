from __future__ import annotations
import io
import json
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import butter, lfilter, medfilt

# --- CONSTANTS & RULES ---
# These bands match your updated research data
SPECIES_RULES = {
    "fin": [(13, 40), (85, 140)],
    "humpback": [(40, 6000)],
    "beluga": [(2000, 5900)],
    "narwhal": [(300, 18000), (47000, 49000)] 
}

def _decode_wav_bytes(wav_bytes: bytes):
    """Decodes raw bytes into a numpy array."""
    try:
        # Load audio; sr=None preserves the original sample rate
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception as e:
        print(f"Error decoding audio: {e}")
        return np.array([0.0], dtype=np.float32), 44100

def _encode_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """Encodes numpy array back to WAV bytes with buffer reset."""
    buffer = io.BytesIO()
    # Using PCM_16 for better compatibility and smaller size
    sf.write(buffer, y, sr, format="WAV", subtype='PCM_16')
    buffer.seek(0)  # CRITICAL: Reset pointer to start to avoid OSError [Errno 22]
    return buffer.getvalue()

# --- FILTERING HELPERS ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Generates coefficients for the Butterworth filter."""
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_whale_filters(y, sr):
    """The multi-stage filtering pipeline."""
    # 1. Non-finite fix (NaN/Inf to 0)
    y = np.nan_to_num(y)
    
    # 2. Spectral Subtraction (Noise Fingerprinting)
    stft = librosa.stft(y)
    mag, phase = librosa.magphase(stft)
    noise_est = np.mean(mag[:, :min(mag.shape[1], 10)], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_est, 0.0)
    y = librosa.istft(mag_clean * phase)
    
    # 3. Dynamic Band-pass (10Hz to 48kHz)
    try:
        b, a = butter_bandpass(10, 48000, sr)
        y = lfilter(b, a, y)
    except ValueError:
        # Fallback if SR is too low for 48kHz
        b, a = butter_bandpass(10, (sr/2)-1, sr)
        y = lfilter(b, a, y)
        
    # 4. Median Filter & Normalization
    y = medfilt(y, kernel_size=3)
    y = librosa.util.normalize(np.nan_to_num(y))
    return y

# --- CLASSIFICATION LOGIC ---

def analyze_and_classify(y, sr):
    """Analyzes events and returns metadata for the dashboard."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    
    total_events = len(peaks)
    counts = {s: 0 for s in SPECIES_RULES.keys()}
    
    for p in peaks:
        # 100ms window around the peak
        start = max(0, int(librosa.frames_to_samples(p) - sr*0.05))
        end = min(len(y), int(librosa.frames_to_samples(p) + sr*0.05))
        segment = y[start:end]
        
        if len(segment) > 10:
            # FFT to find dominant frequency
            fft_res = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1/sr)
            peak_f = freqs[np.argmax(fft_res)]
            
            # Check bands
            for species, bands in SPECIES_RULES.items():
                if any(low <= peak_f <= high for low, high in bands):
                    counts[species] += 1
                    break

    # Construct the JSON-compatible response
    results = {
        "event": list(counts.keys()),
        "probability": [round(counts[s] / total_events, 3) if total_events > 0 else 0 for s in counts.keys()],
        "predicted": [counts[s] > 0 for s in counts.keys()]
    }
    return results

# --- PREPROCESSING STEPS REGISTRY ---

def filter_raw(wav_bytes: bytes) -> bytes:
    """Return raw bytes unchanged."""
    return wav_bytes

def whale_preprocessing_pipeline(wav_bytes: bytes) -> tuple[bytes, str]:
    """Decodes, filters, and analyzes whale sounds in one pass."""
    y, sr = _decode_wav_bytes(wav_bytes)
    
    # 1. Preprocessing (Filters)
    y_filtered = apply_whale_filters(y, sr)
    
    # 2. Classification (Metadata)
    metadata = analyze_and_classify(y_filtered, sr)
    
    # 3. Return bytes and JSON string
    processed_bytes = _encode_wav_bytes(y_filtered, sr)
    return processed_bytes, json.dumps(metadata)

PREPROCESSING_STEPS = {
    "raw": filter_raw,
    "whale_classification": whale_preprocessing_pipeline,
}

def run_preprocessing_step(wav_bytes: bytes, step_name: str):
    """
    Main entry point for dashboard.py. 
    Returns (processed_wav_bytes, label_string).
    """
    if step_name == "whale_classification":
        processed, label = whale_preprocessing_pipeline(wav_bytes)
        return processed, label
    else:
        func = PREPROCESSING_STEPS.get(step_name, filter_raw)
        processed = func(wav_bytes)
        label = {"raw": "Raw audio"}.get(step_name, f"Applied step: {step_name}")
        return processed, label
