# Disko Audio Explorer
Tilde Marie Reinhardt (s260388), Dahlia Louise Dry (s250127), Francesca Di Mella, Daniel Jason Cecil Phillips (s184796)

- dashboard.py: GUI which pulls together all the separate analysis components
- preprocessing.py: preprocessing methods
- classification.py: classification model methods
- disko_sound.py: class for feature extraction from .wav files & spectrogram plotting

![Demo Video](https://youtu.be/QBNhz_jX0JY)

## Quick Start
### Install required packages
```bash
pip install -r requirements.txt
```
### Run the dashboard
```bash
python dashboard.py
```
### Use the dashboard
Go to http://127.0.0.1:8050/ in browser

## Dashboard Features
- Upload `.wav` recordings and play them back directly in the browser.
- Inspect the current file with a linked spectrogram and waveform view.
- Trim the uploaded file by box-selecting a time range on the trim waveform and applying the crop.
- Apply preprocessing filters from the `Preprocessing` section. The built-in options currently include raw audio, low-pass, and high-pass filtering, as well as a custom filtering routine called Whale Clean (background subtraction -> bandpass filter -> normalization + median filtering), .
- Adjust low-pass and high-pass cutoff values directly in the dashboard when those filters are selected.
- Select any time range on the main spectrogram or waveform to run feature extraction on that segment.
- View acoustic measurements for the selected region, including duration, dominant frequency, spectral centroid, spectral rolloff, bandwidth, RMS energy, and frequency range.
- Run classification models on the current selection and review per-class probabilities in the `Detection & Classification` table.
- Mark detection rows as validated with the checkbox column. Validation decisions are written to `exports/validation_log.json`.
- Compare the uploaded audio against known example recordings in the `Disko Whale Catalog` section.

### Exports
- `Export Spectrogram PNG` downloads a PNG of the current processed spectrogram, including the active selection highlight if one exists.
- `Export Detection CSV` downloads the current detection table as CSV and also updates `exports/detection_results.csv`.
- Exported filenames include the uploaded filename stem and the active preprocessing filter name.
- Validation state is included in exported detection rows through the `user_validated` column.

## Add/edit A Preprocessing Step
All preprocessing is done through `preprocessing.py`.

Required input:
- Input: `.wav` bytes
- Output: `.wav` bytes 

### 1) Add the function in preprocessing.py
Example:
```python
def filter_my_step(wav_bytes: bytes) -> bytes:
	# decode wav_bytes -> process audio -> encode wav_bytes
	return processed_wav_bytes
```

### 2) Register the function in PREPROCESSING_STEPS
In `preprocessing.py`, add:
```python
PREPROCESSING_STEPS = {
	"raw": filter_raw,
	"my_step": filter_my_step,
}
```

### 3) Use it in the dashboard
`dashboard.py` automatically reads model names from `PREPROCESSING_STEPS` and renders them as buttons in the `Preprocessing` section.

Note: `Revert To Original` always restores the originally uploaded `.wav`.

## Add A Classification Model
Classification is done through `classification.py`.

Required function contract:
- Input: `.wav` file bytes
- Output: dict exactly like:
```python
{
	"event": [...],
	"probability": [...],
	"predicted": [...],
}
```
Rules:
- `event`: class labels (for example, `"humpback whale call"`)
- `probability`: float score per event
- `predicted`: 0/1 per event (based on model cutoff)

### 1) Add a model function in classification.py
Example:
```python
def model_my_classifier(wav_bytes: bytes) -> dict:
	# run model inference on wav_bytes
	return {
		"event": ["humpback whale call", "fin whale call"],
		"probability": [0.82, 0.14],
		"predicted": [1, 0],
	}
```
### 2) Register it in CLASSIFICATION_MODELS
In `classification.py`, register the model with a short `about` description (shown as a tooltip when hovering the button in the dashboard):
```python
CLASSIFICATION_MODELS = {
	"google_model_simple": {          # existing entry
		"fn": google_model_simple,
		"about": "Source: https://www.kaggle.com/models/google/multispecies-whale.",
	},
	"my_classifier": {
		"fn": model_my_classifier,
		"about": "My custom classifier trained on Disko Bay recordings.",
	},
}
```

### 3) Use it in the dashboard
`dashboard.py` automatically reads model names from `CLASSIFICATION_MODELS` and renders them as buttons in the `Detection & Classification` section.

### Prediction Validation
`dashboard.py` already converts this dict into the classification table and appends a `Validated` checkbox column for human 0/1 review.

## Troubleshooting
Requires python < 3.14 for tensorflow to work