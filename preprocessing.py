Preprocessing interface for dashboard.py.

Contract:
- Input: .wav file bytes
- Output: .wav file bytes

To add a new preprocessing step:
1. Create a function with signature: `func(wav_bytes: bytes) -> bytes`
2. Register it in PREPROCESSING_STEPS

from __future__ import annotations

import io

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


def _decode_wav_bytes(wav_bytes: bytes):
	y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
	return y.astype(np.float32), int(sr)


def _encode_wav_bytes(y: np.ndarray, sr: int) -> bytes:
	buffer = io.BytesIO()
	sf.write(buffer, y, sr, format="WAV")
	return buffer.getvalue()


def filter_raw(wav_bytes: bytes) -> bytes:
	"""Return input unchanged."""
	return wav_bytes

# Add new preprocessing steps here.
PREPROCESSING_STEPS = {
	"raw": filter_raw,
}


def run_preprocessing_step(wav_bytes: bytes, step_name: str):
	"""Run a named preprocessing step and return (processed_wav_bytes, label)."""
	func = PREPROCESSING_STEPS.get(step_name, filter_raw)
	processed = func(wav_bytes)
	label = {
		"raw": "Raw audio",
	}.get(step_name, f"Applied step: {step_name}")
	return processed, label
