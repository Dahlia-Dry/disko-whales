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

    _ = sample_rate
    return {
        "event": class_names,
        "probability": [float(p) for p in mean_probabilities],
        "predicted": predicted,
    }


# Add new classification models here.
CLASSIFICATION_MODELS = {
    "google_model_simple": google_model_simple,
}


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
    model_func = CLASSIFICATION_MODELS.get(model_name, model_placeholder)
    result = model_func(wav_bytes)

    # Defensive normalization so dashboard always receives the expected schema.
    if not isinstance(result, dict):
        return _empty_output()

    return {
        "event": list(result.get("event", [])),
        "probability": list(result.get("probability", [])),
        "predicted": list(result.get("predicted", [])),
    }
