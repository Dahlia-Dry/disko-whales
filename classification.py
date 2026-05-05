"""Classification model interface for dashboard.py.

Contract:
- Input: .wav file bytes
- Output: dict with EXACT keys:
    {
        'event': [...],
        'probability': [...],
        'predicted': [...],
    }

This module is intentionally structured like preprocessing.py:
- One function per model
- A registry mapping model name -> function
- One dispatcher (`run_classification_model`) used by dashboard.py
"""

from __future__ import annotations

from functools import lru_cache

import tensorflow as tf
import tensorflow_hub as hub

MODEL_URL = "https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2"

# Mapping from Google model's internal short codes to human-readable common names.
# Source: https://www.kaggle.com/models/google/multispecies-whale (Outputs section)
_GOOGLE_CLASS_COMMON_NAMES: dict[str, str] = {
    "Oo":           "Orca",
    "Mn":           "Humpback whale",
    "Eg":           "Right whale (Atlantic)",
    "Be":           "Bryde's whale",
    "Upcall":       "Right whale (Pacific, upcall)",
    "Bp":           "Fin whale",
    "Call":         "Orca call",
    "Gunshot":      "Right whale (Pacific, gunshot)",
    "Echolocation": "Orca echolocation",
    "Bm":           "Blue whale",
    "Whistle":      "Orca whistle",
    "Ba":           "Minke whale",
}


def _empty_output() -> dict:
    return {
        "event": [],
        "probability": [],
        "predicted": [],
    }


def model_placeholder(wav_bytes: bytes) -> dict:
    """Default placeholder model that returns no detections."""
    _ = wav_bytes
    return _empty_output()


@lru_cache(maxsize=1)
def _load_google_model():
    model = hub.load(MODEL_URL)
    metadata = model.metadata()
    class_names = metadata["class_names"].numpy()
    class_names = [
        c.decode("utf-8") if isinstance(c, bytes) else str(c)
        for c in class_names
    ]
    return model, class_names


def google_model_simple(wav_bytes: bytes) -> dict:
    """Simple Google example model 
    https://www.kaggle.com/models/google/multispecies-whale
    """
    if not wav_bytes:
        return _empty_output()

    model, class_names = _load_google_model()

    waveform, sample_rate = tf.audio.decode_wav(tf.convert_to_tensor(wav_bytes))

    # Convert to mono if needed so model input matches [samples, 1].
    if waveform.shape.rank == 2 and waveform.shape[-1] > 1:
        waveform = tf.reduce_mean(waveform, axis=1, keepdims=True)

    batch = tf.expand_dims(waveform, 0)
    spectrogram = model.front_end(batch)
    context_windows = tf.signal.frame(
        tf.squeeze(spectrogram, 0),
        frame_length=128,
        frame_step=64,
        axis=-2,
    )

    logits = model.logits(context_windows)
    probabilities = tf.nn.sigmoid(logits)
    mean_probabilities = tf.reduce_mean(probabilities, axis=0).numpy().tolist()

    # Use a simple 0.5 cutoff for 0/1 predicted labels.
    predicted = [1 if float(p) >= 0.5 else 0 for p in mean_probabilities]

    common_names = [
        _GOOGLE_CLASS_COMMON_NAMES.get(c, c) for c in class_names
    ]

    _ = sample_rate
    return {
        "event": common_names,
        "probability": [float(p) for p in mean_probabilities],
        "predicted": predicted,
    }


# Registry: each entry is either a plain callable (legacy) or a dict with keys:
#   "fn"    – callable matching the model contract
#   "about" – one-sentence description shown as a tooltip in the dashboard
#
# To add a new model:
#   "my_model": {
#       "fn": model_my_classifier,
#       "about": "Short description of the model.",
#   },
CLASSIFICATION_MODELS: dict[str, dict | callable] = {
    "google_model_simple": {
        "fn": google_model_simple,
        "about": (
            "Google Multispecies Whale Detector (EfficientNet-B0). "
            "Scores 12 classes across 7 species and 5 call types on 5-second "
            "24 kHz context windows. "
            "Source: kaggle.com/models/google/multispecies-whale"
        ),
    },
}


def get_classification_model_about(model_name: str) -> str:
    """Return the about string for a registered model, or empty string if not set."""
    entry = CLASSIFICATION_MODELS.get(model_name)
    if isinstance(entry, dict):
        return entry.get("about", "")
    return ""


def run_classification_model(wav_bytes: bytes, model_name: str = "placeholder") -> dict:
    """Run the selected classification model on wav bytes.

    Parameters
    ----------
    wav_bytes:
        Raw WAV bytes from the uploaded (or preprocessed) signal.
    model_name:
        Optional model selector for future expansion.

    Returns
    -------
    dict
        Dict with keys: event, probability, predicted.
        'predicted' should contain 0/1 values.
    """
    entry = CLASSIFICATION_MODELS.get(model_name)
    if entry is None:
        model_func = model_placeholder
    elif isinstance(entry, dict):
        model_func = entry.get("fn", model_placeholder)
    else:
        model_func = entry  # legacy plain-callable entry
    result = model_func(wav_bytes)

    # Defensive normalization so dashboard always receives the expected schema.
    if not isinstance(result, dict):
        return _empty_output()

    return {
        "event": list(result.get("event", [])),
        "probability": list(result.get("probability", [])),
        "predicted": list(result.get("predicted", [])),
    }
