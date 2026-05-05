import io
import librosa
import numpy as np

# If you use TensorFlow, uncomment these lines. 
# NOTE: We keep imports inside the functions to prevent Mac semaphore leaks!
# import tensorflow as tf
# import tensorflow_hub as hub

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def _decode_wav_bytes(wav_bytes: bytes):
    """Safely decode the incoming bytes from the dashboard into numpy arrays."""
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
    return y, sr

# ---------------------------------------------------------
# MODEL IMPLEMENTATIONS
# ---------------------------------------------------------
def run_mock_whale_model(wav_bytes: bytes) -> dict:
    """
    A fake model that safely tests your dashboard without memory leaks.
    It generates dynamic probabilities based on the audio's RMS energy.
    """
    y, sr = _decode_wav_bytes(wav_bytes)
    
    # Calculate some basic feature to make the "fake" results change per file
    rms = np.mean(librosa.feature.rms(y=y))
    
    labels = ["Humpback Whale", "Blue Whale", "Orca", "Bowhead Whale", "Vessel Noise", "Ocean Hiss"]
    
    # Generate random probabilities that sum to 1.0
    np.random.seed(int(rms * 1000000) % 10000) # Pseudo-random based on audio
    probs = np.random.dirichlet(np.ones(len(labels)), size=1)[0]
    
    # Sort from highest probability to lowest
    sorted_indices = np.argsort(probs)[::-1]
    
    events = [labels[i] for i in sorted_indices]
    probabilities = [float(probs[i]) for i in sorted_indices]
    
    # Mark the highest probability as "Predicted: True"
    predicted = [False] * len(events)
    predicted[0] = True 
    
    # The dictionary format MUST match this exactly for dashboard.py line 407
    return {
        "event": events,
        "probability": probabilities,
        "predicted": predicted
    }

def run_real_tf_model(wav_bytes: bytes) -> dict:
    """
    TEMPLATE FOR YOUR REAL MODEL.
    Load your TensorFlow/Keras model inside the function to prevent Mac socket crashes.
    """
    # 1. Decode audio
    y, sr = _decode_wav_bytes(wav_bytes)
    
    # 2. Resample if your model requires a specific rate (e.g., YAMNet needs 16kHz)
    target_sr = 16000
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # --- INSERT YOUR TF LOGIC HERE ---
    # Example:
    # model = tf.keras.models.load_model("my_whale_model.h5")
    # spectrogram = make_spectrogram_for_model(y)
    # predictions = model.predict(np.expand_dims(spectrogram, axis=0))[0]
    # ---------------------------------
    
    # Placeholder return so it doesn't crash if you accidentally click it
    return {
        "event": ["Implementation Pending"],
        "probability": [1.0],
        "predicted": [True]
    }

# ---------------------------------------------------------
# DASHBOARD REGISTRY
# ---------------------------------------------------------
# The keys in this dictionary generate the buttons in the dashboard.
CLASSIFICATION_MODELS = {
    "mock_whale_net": "Mock Whale Model (Fast)",
    "real_tf_model": "TensorFlow Whale Net",
}

# ---------------------------------------------------------
# MANDATORY DASHBOARD INTERFACE
# ---------------------------------------------------------
def run_classification_model(wav_bytes: bytes, model_name: str) -> dict:
    """
    This is the specific function dashboard.py line 403 is looking for.
    It takes the bytes from the selected segment and routes it to the right model.
    """
    if model_name == "mock_whale_net":
        return run_mock_whale_model(wav_bytes)
    elif model_name == "real_tf_model":
        return run_real_tf_model(wav_bytes)
    else:
        # Fallback if a model isn't found
        return {
            "event": ["Unknown Model"],
            "probability": [0.0],
            "predicted": [False]
        }
