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
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
    return y.astype(np.float32), int(sr)

def _encode_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    return buffer.getvalue()

# --- FILTERING HELPERS ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = max(0.001, lowcut / nyq), min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_whale_filters(y, sr):
    # 1. Non-finite fix
    y = np.nan_to_num(y)
    
    # 2. Spectral Subtraction
    stft = librosa.stft(y)
    mag, phase = librosa.magphase(stft)
    noise_est = np.mean(mag[:, :min(mag.shape[1], 10)], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_est, 0.0)
    y = librosa.istft(mag_clean * phase)
    
    # 3. Band-pass (10Hz to 48kHz or Nyquist)
    try:
        b, a = butter_bandpass(10, 48000, sr)
        y = lfilter(b, a, y)
    except ValueError:
        b, a = butter_bandpass(10, (sr/2)-1, sr)
        y = lfilter(b, a, y)
        
    # 4. Median Filter & Normalization
    y = medfilt(y, kernel_size=3)
    y = librosa.util.normalize(np.nan_to_num(y))
    return y

# --- CLASSIFICATION LOGIC ---

def analyze_and_classify(y, sr):
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

    # Format the specific output requested
    results = {
        "event": list(counts.keys()),
        "probability": [round(counts[s] / total_events, 3) if total_events > 0 else 0 for s in counts.keys()],
        "predicted": [counts[s] > 0 for s in counts.keys()]
    }
    return results

# --- PREPROCESSING STEPS REGISTRY ---

def filter_raw(wav_bytes: bytes) -> bytes:
    return wav_bytes

def whale_preprocessing_pipeline(wav_bytes: bytes) -> tuple[bytes, dict]:
    """Decodes, filters, and analyzes whale sounds in one pass."""
    y, sr = _decode_wav_bytes(wav_bytes)
    
    # Apply Preprocessing
    y_filtered = apply_whale_filters(y, sr)
    
    # Apply Classification
    metadata = analyze_and_classify(y_filtered, sr)
    
    # Return encoded audio and the metadata dictionary
    processed_bytes = _encode_wav_bytes(y_filtered, sr)
    return processed_bytes, metadata

PREPROCESSING_STEPS = {
    "raw": filter_raw,
    "whale_classification": whale_preprocessing_pipeline,
}

def run_preprocessing_step(wav_bytes: bytes, step_name: str):
    """Run a named preprocessing step and return (processed_wav_bytes, label)."""
    if step_name == "whale_classification":
        processed, metadata = whale_preprocessing_pipeline(wav_bytes)
        # We pass the JSON metadata as a string in the label field
        return processed, json.dumps(metadata)
    else:
        func = PREPROCESSING_STEPS.get(step_name, filter_raw)
        processed = func(wav_bytes)
        label = {"raw": "Raw audio"}.get(step_name, f"Applied step: {step_name}")
        return processed, label
		
