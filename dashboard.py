"""
Disko Audio Explorer — Dash dashboard
  • Upload a .wav file
  • Play it back in the browser
  • View the spectrogram
  • Box-select a time range on the spectrogram → run Disko_Sound analysis

Author: @Dahlia Dry, April 2026
"""

import base64
import io
import json
import os
import tempfile
import uuid

import dash
from dash import Input, Output, State, ALL, dash_table, dcc, html, Patch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np
import librosa
import pandas as pd
import soundfile as sf
from classification import CLASSIFICATION_MODELS, run_classification_model, get_classification_model_about
from disko_sound import Disko_Sound
from preprocessing import PREPROCESSING_STEPS, filter_crop, run_preprocessing_step

#SETTINGS
CACHE_DIR = os.path.join(tempfile.gettempdir(), "disko_whales_cache")
MAX_PLOT_DURATION_SECONDS = 15*60
PLOT_TARGET_SR = 4000
DEFAULT_LOWPASS_CUTOFF_HZ = 1000
DEFAULT_HIGHPASS_CUTOFF_HZ = 100
CLIP_ENDS = 0

# ── Whale catalog data ─────────────────────────────────────────────────────────

WHALE_DATA = {
    "Narwhal": {
        "call_types": "clicks, whistles, pulses, knocks",
        "frequency_range": "generally 300 Hz–150 kHz",
        "typical_duration": "clicks last <0.1 s, whistles and pulses last 0.1–2 s",
        "example_files": [
            "known_samples/narwhal/Narwhal.wav",
            "known_samples/narwhal/narwhal-voicesofthesea.wav",
        ],
    },
    "Fin whale": {
        "call_types": "pulses, downsweeps, songs",
        "frequency_range": "typically 15–40 Hz, sometimes up to 100 Hz",
        "typical_duration": "songs can last minutes–hours, pulses last 0.5–2 s",
        "example_files": [
            "known_samples/fin/fin.wav",
            "known_samples/fin/finWhale.wav",
        ],
    },
    "Humpback whale": {
        "call_types": "structured songs, moans, grunts, squeaks",
        "frequency_range": "generally 20 Hz to 10 kHz, with most energy in 100 Hz to 4 kHz",
        "typical_duration": "songs can last 10–30 minutes, individual calls typically 0.5–5 s",
        "example_files": [
            "known_samples/humpback/humpback_bubblenetFeeding.wav",
            "known_samples/humpback/humpback_socialSounds.wav",
            "known_samples/humpback/humpbackvoicesofthesea1.wav",
            "known_samples/humpback/humpback-whale-song-skj-bay_MarianneRasmussen.wav",
        ],
    },
    "Beluga whale": {
        "call_types": "clicks, whistles, chirps, pulses/bursts",
        "frequency_range": "generally 1–120 kHz, with whistles often in 1–20 kHz range",
        "typical_duration": "clicks last <0.1 s, whistles and calls last <5 s",
        "example_files": [
            "known_samples/beluga/beluga_clicks.wav",
            "known_samples/beluga/beluga_socialSounds.wav",
            "known_samples/beluga/beluga-voicesofthesea.wav",
            "known_samples/beluga/beluga.wav",
        ],
    },
}

# ── App ────────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="Disko Audio Explorer", suppress_callback_exceptions=True)

UPLOAD_STYLE = {
    "width": "100%",
    "height": "80px",
    "lineHeight": "80px",
    "borderWidth": "2px",
    "borderStyle": "dashed",
    "borderRadius": "8px",
    "textAlign": "center",
    "cursor": "pointer",
    "marginBottom": "16px",
    "color": "#555",
}

SELECTOR_BUTTON_STYLE = {
    "padding": "8px 12px",
    "border": "1px solid #ccc",
    "borderRadius": "6px",
    "cursor": "pointer",
}


def humanize_name(name: str):
    return name.replace("_", " ").replace("-", " ").title()


def make_audio_player(wav_bytes: bytes, filename: str, sr: int) -> html.Div:
    """Create an audio player from WAV bytes with base64 data URI."""
    duration_seconds = duration_from_wav_bytes(wav_bytes)
    data_uri = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode('utf-8')}"
    return html.Div(
        [
            html.P(
                f"▶  {filename}   |   {sr:,} Hz   |   {duration_seconds:.2f} s",
                style={"margin": "8px 0 4px", "fontSize": "13px", "color": "#444"},
            ),
            html.Audio(
                src=data_uri,
                controls=True,
                style={"width": "100%"},
            ),
        ],
        style={"marginBottom": "8px"},
    )


def render_selector_buttons(option_names, selected_value, button_type, extra_buttons=None):
    buttons = []
    for option_name in option_names:
        is_active = option_name == selected_value
        buttons.append(
            html.Button(
                humanize_name(option_name),
                id={"type": button_type, "index": option_name},
                n_clicks=0,
                style={
                    **SELECTOR_BUTTON_STYLE,
                    "backgroundColor": "#2f6fdd" if is_active else "#ffffff",
                    "color": "#ffffff" if is_active else "#333333",
                },
            )
        )
    if extra_buttons:
        buttons.extend(extra_buttons)
    return buttons

app.layout = html.Div(
    style={
        "fontFamily": "'Segoe UI', Arial, sans-serif",
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "28px 24px",
        "backgroundColor": "#fafafa",
    },
    children=[
        html.H1("🐋 Disko Audio Explorer", style={"marginBottom": "4px"}),
        html.P(
            "Upload a .wav file to visualize and interactively analyze whale vocalizations.",
            style={"color": "#555", "marginTop": 0},
        ),
        # ── Upload ─────────────────────────────────────────────────────────────
        dcc.Upload(
            id="upload-audio",
            children=html.Div(["Drag & Drop or ", html.A("Select a .wav file")]),
            style=UPLOAD_STYLE,
            accept=".wav",
        ),
        # ── Audio player ────────────────────────────────────────────────────────
        html.Div(id="audio-player"),
        # ── Trim Audio (Waveform-based) ─────────────────────────────────────────
        html.Div(
            id="trim-panel",
            style={"display": "none"},
            children=[
                html.H3("Trim Audio", style={"marginBottom": "6px"}),
                html.P(
                    "Use Box Select tool (□) on the waveform below to select a region to keep. Click Apply Trim to crop and discard the rest.",
                    style={"color": "#666", "fontSize": "13px", "marginTop": 0},
                ),
                html.Div(
                    html.P("Trim Waveform", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "4px", "color": "#666"}),
                    style={"marginBottom": "4px"},
                ),
                dcc.Graph(
                    id="trim-waveform",
                    figure=go.Figure(
                        layout=go.Layout(
                            height=160,
                            xaxis={"title": "Time (s)"},
                            yaxis={"title": "Amplitude"},
                            paper_bgcolor="#fafafa",
                            plot_bgcolor="#f5f8ff",
                        )
                    ),
                    config={
                        "modeBarButtonsToAdd": ["select2d"],
                        "displayModeBar": True,
                        "scrollZoom": True,
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "Apply Trim from Selection",
                            id="trim-apply-btn",
                            n_clicks=0,
                            style={**SELECTOR_BUTTON_STYLE, "backgroundColor": "#2f6fdd", "color": "#fff"},
                        ),
                    ],
                    style={"display": "flex", "gap": "8px", "marginTop": "8px"},
                ),
                html.Div(id="trim-status", style={"color": "#444", "fontSize": "13px", "marginTop": "6px"}),
            ],
        ),
        # ── Preprocessing ───────────────────────────────────────────────────────
        html.Div(
            [
                html.H3("Preprocessing", style={"marginBottom": "8px"}),
                html.P(
                    "Choose a filter. Changing the filter recomputes the spectrogram, analysis, and classification.",
                    style={"color": "#666", "fontSize": "13px", "marginTop": 0},
                ),
                html.Div(
                    render_selector_buttons(
                        PREPROCESSING_STEPS.keys(),
                        "raw",
                        "preprocessing-step-btn",
                        extra_buttons=[
                            html.Button(
                                "Revert To Original",
                                id="filter-revert",
                                n_clicks=0,
                                style={**SELECTOR_BUTTON_STYLE, "backgroundColor": "#ffffff", "color": "#333333"},
                            )
                        ],
                    ),
                    id="preprocessing-buttons",
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "8px"},
                ),
                html.Div(
                    [
                        html.Span("Low pass cutoff:", style={"fontSize": "13px", "color": "#444"}),
                        dcc.Input(
                            id="lowpass-cutoff-hz",
                            type="number",
                            value=DEFAULT_LOWPASS_CUTOFF_HZ,
                            min=1,
                            max=24000,
                            step=1,
                            debounce=True,
                            style={"width": "90px", "padding": "4px 8px", "border": "1px solid #ccc", "borderRadius": "4px", "fontSize": "13px"},
                        ),
                        html.Span("Hz", style={"fontSize": "13px", "color": "#444"}),
                    ],
                    id="lowpass-controls",
                    style={"display": "none", "alignItems": "center", "gap": "8px", "marginBottom": "8px"},
                ),
                html.Div(
                    [
                        html.Span("High pass cutoff:", style={"fontSize": "13px", "color": "#444"}),
                        dcc.Input(
                            id="highpass-cutoff-hz",
                            type="number",
                            value=DEFAULT_HIGHPASS_CUTOFF_HZ,
                            min=1,
                            max=24000,
                            step=1,
                            debounce=True,
                            style={"width": "90px", "padding": "4px 8px", "border": "1px solid #ccc", "borderRadius": "4px", "fontSize": "13px"},
                        ),
                        html.Span("Hz", style={"fontSize": "13px", "color": "#444"}),
                    ],
                    id="highpass-controls",
                    style={"display": "none", "alignItems": "center", "gap": "8px", "marginBottom": "8px"},
                ),
                html.Div(id="preprocessing-status", style={"color": "#444", "fontSize": "13px"}),
            ],
            style={"marginTop": "24px"},
        ),
        # ── Spectrogram + Waveform ─────────────────────────────────────────────
        html.Div(
            [
                html.H3("Spectrogram + Waveform", style={"marginBottom": "4px"}),
                html.P(
                    "Use the Box Select tool (□) in the toolbar on either the spectrogram or the waveform to pick a time region for analysis.",
                    style={"color": "#666", "fontSize": "13px", "marginTop": 0},
                ),
            ],
            style={"marginTop": "24px"},
        ),
        dcc.Graph(
            id="spectrogram",
            figure=go.Figure(
                layout=go.Layout(
                    height=420,
                    xaxis={"title": "Time (s)"},
                    yaxis={"title": "Frequency (Hz)"},
                    paper_bgcolor="#fafafa",
                    plot_bgcolor="#111",
                )
            ),
            config={
                "modeBarButtonsToAdd": ["select2d"],
                "displayModeBar": True,
                "scrollZoom": True,
            },
        ),
        html.Div(
            html.P("Waveform", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "4px", "color": "#666"}),
            style={"marginTop": "12px"},
        ),
        dcc.Graph(
            id="waveform",
            figure=go.Figure(
                layout=go.Layout(
                    height=160,
                    xaxis={"title": "Time (s)"},
                    yaxis={"title": "Amplitude"},
                    paper_bgcolor="#fafafa",
                    plot_bgcolor="#f5f8ff",
                )
            ),
            config={
                "modeBarButtonsToAdd": ["select2d"],
                "displayModeBar": True,
                "scrollZoom": True,
            },
        ),
        # ── Analysis panel ─────────────────────────────────────────────────────
        html.Div(id="analysis-status", style={"marginTop": "20px", "color": "#666", "fontSize": "13px"}),
        html.Div(id="analysis-panel", style={"marginTop": "28px"}),
        html.Div(id="detection-status", style={"marginTop": "20px", "color": "#666", "fontSize": "13px"}),
        html.Div(id="classification-panel", style={"marginTop": "28px"}),
        # ── Disko Whale Catalog ─────────────────────────────────────────────────
        html.Div(
            [
                html.H3("Disko Whale Catalog", style={"marginBottom": "4px"}),
                html.P(
                    "Use this catalog to compare your file against known species examples and assist with manual validation of the model predictions above.",
                    style={"color": "#666", "fontSize": "13px", "marginTop": 0},
                ),
                dcc.Dropdown(
                    id="catalog-species-dropdown",
                    options=[{"label": k, "value": k} for k in WHALE_DATA],
                    placeholder="Select a species to compare…",
                    clearable=True,
                    style={"maxWidth": "320px", "marginBottom": "12px"},
                ),
                html.Div(id="catalog-info-panel"),
            ],
            style={"marginTop": "36px"},
        ),
        html.Div(
            [
                html.H3("Export Results", style={"marginBottom": "8px"}),
                html.P(
                    "Export the current spectrogram as PNG and the current detection table as CSV.",
                    style={"color": "#555", "marginTop": 0},
                ),
                html.Div(
                    [
                        html.Button("Export Spectrogram PNG", id="export-spectrogram-btn", n_clicks=0, style={"padding": "10px 14px"}),
                        html.Button("Export Detection CSV", id="export-csv-btn", n_clicks=0, style={"padding": "10px 14px"}),
                    ],
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "8px"},
                ),
                html.Div(id="export-status", style={"color": "#444", "fontSize": "13px"}),
                dcc.Download(id="download-spectrogram"),
                dcc.Download(id="download-csv"),
            ],
            style={"marginTop": "28px", "marginBottom": "28px"},
        ),
        # ── Hidden stores ──────────────────────────────────────────────────────
        dcc.Store(id="audio-store"),
        dcc.Store(id="processed-audio-store"),
        dcc.Store(id="selection-store"),
        dcc.Store(id="trim-selection-store"),
        dcc.Store(id="filter-store", data={"filter_name": "raw", "label": "Raw audio"}),
        dcc.Store(id="classification-model-store", data=None),
        dcc.Store(id="detection-store"),
        dcc.Store(id="validation-store", data={}),
    ],
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def encode_wav_bytes(y: np.ndarray, sr: int):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def write_wav_bytes_to_cache(wav_bytes: bytes, prefix: str) -> str:
    ensure_cache_dir()
    filename = f"{prefix}-{uuid.uuid4().hex}.wav"
    path = os.path.join(CACHE_DIR, filename)
    with open(path, "wb") as handle:
        handle.write(wav_bytes)
    return path


def read_wav_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def load_audio_for_plot(path: str):
    """Load a capped-duration, downsampled signal for UI plotting."""
    if not path or not os.path.exists(path):
        return None, None
    y, sr = librosa.load(path, sr=PLOT_TARGET_SR, mono=True, duration=MAX_PLOT_DURATION_SECONDS)
    return y.astype(np.float32), int(sr)


def load_audio_full(path: str):
    """Load full-resolution signal for analysis/export/classification."""
    if not path or not os.path.exists(path):
        return None, None
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float32), int(sr)


def duration_from_wav_bytes(wav_bytes: bytes) -> float:
    info = sf.info(io.BytesIO(wav_bytes))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def slugify_filename(value: str | None):
    if not value:
        return "audio"
    stem = os.path.splitext(os.path.basename(value))[0]
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in stem)
    return safe.strip("-") or "audio"


def save_validation_log(records: list[dict]):
    os.makedirs("exports", exist_ok=True)
    with open("exports/validation_log.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def build_export_rows(detection_rows: list[dict] | None, validation_store: dict | None):
    export_rows = []
    validation_store = validation_store or {}
    for row in detection_rows or []:
        export_row = dict(row)
        row_key = row.get("event")
        export_row["user_validated"] = bool(validation_store.get(row_key, False))
        export_rows.append(export_row)
    return export_rows


def get_selected_segment(y: np.ndarray, sr: int, selected_data: dict | None):
    if y is None or sr is None:
        return None

    if not selected_data or "range" not in selected_data or "x" not in selected_data["range"]:
        return {
            "segment": y,
            "t0": 0.0,
            "t1": len(y) / sr,
            "duration": len(y) / sr,
            "used_full_signal": True,
        }

    t0, t1 = selected_data["range"]["x"]
    t0, t1 = sorted((float(t0), float(t1)))
    i0 = max(0, int(t0 * sr))
    i1 = min(len(y), int(t1 * sr))
    seg = y[i0:i1]

    return {
        "segment": seg,
        "t0": t0,
        "t1": t1,
        "duration": max(0.0, t1 - t0),
        "used_full_signal": False,
    }


def get_auto_spectrogram_ceiling(freqs: np.ndarray, spectrogram_db: np.ndarray) -> float:
    """Estimate a useful upper frequency bound from cumulative spectral energy."""
    if freqs.size == 0 or spectrogram_db.size == 0:
        return 0.0

    # Convert dB values to linear power so energy can be accumulated by frequency.
    power = np.power(10.0, spectrogram_db / 10.0)
    per_freq_energy = np.mean(power, axis=1)
    total_energy = float(np.sum(per_freq_energy))
    if total_energy <= 0:
        return float(freqs[-1])

    cumulative = np.cumsum(per_freq_energy) / total_energy
    idx = int(np.searchsorted(cumulative, 0.995, side="left"))
    idx = max(0, min(idx, len(freqs) - 1))

    nyquist = float(freqs[-1])
    padded = float(freqs[idx]) * 1.15
    min_ceiling = min(300.0, nyquist)
    return min(nyquist, max(min_ceiling, padded))

def render_spectrogram_png(y: np.ndarray, sr: int, selected_data: dict | None):
    n_fft = 2048
    hop = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    times = librosa.frames_to_time(np.arange(spectrogram_db.shape[1]), sr=sr, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    max_freq = get_auto_spectrogram_ceiling(freqs, spectrogram_db)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=160)
    mesh = ax.pcolormesh(times, freqs, spectrogram_db, shading="auto", cmap="viridis", vmin=-80, vmax=0)
    ax.set_ylim(0, max_freq)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Processed Spectrogram")

    if selected_data and "range" in selected_data and "x" in selected_data["range"]:
        t0, t1 = sorted((float(selected_data["range"]["x"][0]), float(selected_data["range"]["x"][1])))
        ax.axvspan(t0, t1, color="white", alpha=0.15)

    fig.colorbar(mesh, ax=ax, label="dB")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def make_waveform_figure(y: np.ndarray, sr: int) -> go.Figure:
    if len(y) > 4000:
        indices = np.linspace(0, len(y) - 1, 4000).astype(int)
        y_plot = y[indices]
        times = np.linspace(0, len(y) / sr, num=4000)
    else:
        y_plot = y
        times = np.arange(len(y)) / sr

    fig = go.Figure(
        go.Scatter(
            x=times,
            y=y_plot,
            mode="lines",
            line=dict(color="#2f6fdd", width=1),
            hovertemplate="Time: %{x:.3f} s<br>Amplitude: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=160,
        margin=dict(l=60, r=20, t=10, b=40),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        dragmode="select",
        paper_bgcolor="#fafafa",
        plot_bgcolor="#f5f8ff",
        xaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
    )
    return fig


def make_spectrogram_figure(y: np.ndarray, sr: int):
    n_fft = 2048
    hop = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    times = librosa.frames_to_time(
        np.arange(S_db.shape[1]), sr=sr, hop_length=hop
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    max_freq = get_auto_spectrogram_ceiling(freqs, S_db)

    fig = go.Figure(
        go.Heatmap(
            z=S_db,
            x=times,
            y=freqs,
            colorscale="Viridis",
            colorbar=dict(title="dB", thickness=14),
            zmin=-80,
            zmax=0,
            hovertemplate="Time: %{x:.2f} s<br>Freq: %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis=dict(range=[0, max_freq]),
        dragmode="select",
        height=420,
        margin=dict(l=60, r=20, t=20, b=60),
        paper_bgcolor="#fafafa",
        plot_bgcolor="#111",
    )
    return fig


def build_catalog_audio_block(y: np.ndarray, sr: int, audio_src: str, label: str) -> html.Div:
    """Waveform + spectrogram + audio player card for catalog comparison."""
    wave_fig = make_waveform_figure(y, sr)
    wave_fig.update_layout(height=160, margin=dict(l=50, r=10, t=10, b=40))
    spec_fig = make_spectrogram_figure(y, sr)
    spec_fig.update_layout(height=260, margin=dict(l=60, r=20, t=10, b=50))
    return html.Div(
        style={
            "backgroundColor": "#ffffff",
            "border": "1px solid #e0e0e0",
            "borderRadius": "10px",
            "padding": "16px",
            "minWidth": 0,
        },
        children=[
            html.H4(label, style={"marginTop": 0, "marginBottom": "12px", "color": "#1a1a2e", "fontSize": "14px"}),
            html.P("Waveform", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "4px", "color": "#666"}),
            dcc.Graph(figure=wave_fig, config={"displayModeBar": False}),
            html.P("Spectrogram", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "4px", "marginTop": "12px", "color": "#666"}),
            dcc.Graph(figure=spec_fig, config={"displayModeBar": False}),
            html.Audio(src=audio_src, controls=True, style={"width": "100%", "marginTop": "10px"}),
        ],
    )


def render_detection_panel(detection_rows, validation_store=None, selected_model=None):
    detection_rows = detection_rows or []
    validation_store = validation_store or {}
    sorted_rows = sorted(detection_rows, key=lambda row: row["probability"], reverse=True)
    top_row = sorted_rows[0] if sorted_rows else None

    model_about_map = {
        name: get_classification_model_about(name)
        for name in CLASSIFICATION_MODELS
    }
    model_buttons = render_selector_buttons(
        CLASSIFICATION_MODELS.keys(),
        selected_model,
        "classification-model-btn",
    )

    table_rows = []
    for row in sorted_rows:
        checkbox_value = ["validated"] if validation_store.get(row["event"], False) else []
        table_rows.append(
            html.Tr(
                [
                    html.Td(row["event"], style={"padding": "8px 10px", "borderBottom": "1px solid #ddd"}),
                    html.Td(f"{row['probability']:.4f}", style={"padding": "8px 10px", "borderBottom": "1px solid #ddd"}),
                    html.Td(
                        "Yes" if row.get("predicted", False) else "No",
                        style={
                            "padding": "8px 10px",
                            "borderBottom": "1px solid #ddd",
                            "fontWeight": "bold" if row.get("predicted", False) else "normal",
                        },
                    ),
                    html.Td(
                        dcc.Checklist(
                            id={"type": "validation-checkbox", "index": row["event"]},
                            options=[{"label": "", "value": "validated"}],
                            value=checkbox_value,
                            style={"margin": 0},
                            inputStyle={"marginRight": "0"},
                        ),
                        style={"padding": "8px 10px", "borderBottom": "1px solid #ddd", "textAlign": "center"},
                    ),
                ]
            )
        )

    subtitle = "Choose a classification model to run detection on the current selection."
    if selected_model and not top_row:
        subtitle = f"Model '{humanize_name(selected_model)}' ran, but produced no rows for the current selection."
    if top_row:
        subtitle = f"Top predicted class: {top_row['event']} ({top_row['probability']:.4f})."

    return html.Div(
        [
            html.H3("Detection & Classification", style={"marginBottom": "8px"}),
            html.P("Available classification models:", style={"color": "#444", "marginBottom": "6px"}),
            html.Div(model_buttons, style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "6px"}),
            html.Div(
                model_about_map.get(selected_model, "") if selected_model else "",
                style={
                    "fontSize": "13px",
                    "color": "#555",
                    "backgroundColor": "#f0f4ff",
                    "border": "1px solid #ccd6f6",
                    "borderRadius": "6px",
                    "padding": "8px 12px",
                    "marginBottom": "10px",
                    "display": "block" if (selected_model and model_about_map.get(selected_model)) else "none",
                },
            ),
            html.P(
                subtitle,
                style={"color": "#444", "marginTop": 0},
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Class", style={"textAlign": "left", "padding": "10px", "backgroundColor": "#e8e8e8"}),
                                    html.Th("Probability", style={"textAlign": "left", "padding": "10px", "backgroundColor": "#e8e8e8"}),
                                    html.Th("Predicted", style={"textAlign": "left", "padding": "10px", "backgroundColor": "#e8e8e8"}),
                                    html.Th("Validated", style={"textAlign": "center", "padding": "10px", "backgroundColor": "#e8e8e8"}),
                                ]
                            )
                        ),
                        html.Tbody(table_rows),
                    ],
                    style={"width": "100%", "borderCollapse": "collapse", "backgroundColor": "#fff"},
                ),
                style={"overflowX": "auto", "border": "1px solid #e0e0e0"},
            ),
        ]
    )


# ── Callbacks ──────────────────────────────────────────────────────────────────


@app.callback(
    Output("catalog-info-panel", "children"),
    Input("catalog-species-dropdown", "value"),
    State("processed-audio-store", "data"),
    State("audio-store", "data"),
)
def update_catalog_panel(species, processed_store, audio_store):
    if not species:
        return html.P("Select a species above to view its call characteristics and compare with known examples.", style={"color": "#888", "fontSize": "13px"})
    info = WHALE_DATA[species]
    rows = [
        ("Call types", info["call_types"]),
        ("Frequency range", info["frequency_range"]),
        ("Typical duration", info["typical_duration"]),
    ]
    info_table = html.Div(
        html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Property", style={"textAlign": "left", "padding": "8px 12px", "backgroundColor": "#e8e8e8", "width": "28%"}),
                    html.Th("Value", style={"textAlign": "left", "padding": "8px 12px", "backgroundColor": "#e8e8e8"}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(k, style={"padding": "7px 12px", "borderBottom": "1px solid #e0e0e0", "fontWeight": "500", "fontSize": "13px"}),
                        html.Td(v, style={"padding": "7px 12px", "borderBottom": "1px solid #e0e0e0", "fontSize": "13px"}),
                    ])
                    for k, v in rows
                ]),
            ],
            style={"width": "100%", "borderCollapse": "collapse", "backgroundColor": "#fff"},
        ),
        style={"border": "1px solid #e0e0e0", "borderRadius": "6px", "overflow": "hidden", "maxWidth": "720px", "marginBottom": "20px"},
    )

    blocks = []

    # Current file under inspection (if uploaded)
    if processed_store:
        y_up, sr_up = load_audio_for_plot((processed_store or {}).get("processed_wav_path"))
        if y_up is not None:
            buf = io.BytesIO()
            sf.write(buf, y_up, sr_up, format="WAV")
            audio_src = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
            filename = (audio_store or {}).get("filename", "uploaded file")
            blocks.append(build_catalog_audio_block(y_up, sr_up, audio_src, f"🔍 {filename}"))

    # Known example files for the selected species
    for file_path in info.get("example_files", []):
        if not os.path.exists(file_path):
            continue
        try:
            y_ex, sr_ex = librosa.load(file_path, sr=None, mono=True)
            y_ex = y_ex.astype(np.float32)
            buf = io.BytesIO()
            sf.write(buf, y_ex, sr_ex, format="WAV")
            audio_src = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
            label = os.path.splitext(os.path.basename(file_path))[0]
            blocks.append(build_catalog_audio_block(y_ex, sr_ex, audio_src, label))
        except Exception as exc:
            blocks.append(html.Div(
                html.P(f"Could not load {os.path.basename(file_path)}: {exc}", style={"color": "#c00", "fontSize": "13px"}),
                style={"padding": "12px"},
            ))

    grid = html.Div(
        blocks,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fill, minmax(360px, 1fr))",
            "gap": "20px",
        },
    )

    return html.Div([info_table, grid])


@app.callback(
    Output("preprocessing-buttons", "children"),
    Input("filter-store", "data"),
)
def update_preprocessing_buttons(filter_store):
    selected_filter = (filter_store or {}).get("filter_name", "raw")
    return render_selector_buttons(
        PREPROCESSING_STEPS.keys(),
        selected_filter,
        "preprocessing-step-btn",
        extra_buttons=[
            html.Button(
                "Revert To Original",
                id="filter-revert",
                n_clicks=0,
                style={**SELECTOR_BUTTON_STYLE, "backgroundColor": "#ffffff", "color": "#333333"},
            )
        ],
    )


@app.callback(
    Output("audio-player", "children"),
    Output("audio-store", "data"),
    Output("processed-audio-store", "data"),
    Output("preprocessing-status", "children"),
    Output("filter-store", "data", allow_duplicate=True),
    Output("classification-model-store", "data", allow_duplicate=True),
    Output("selection-store", "data", allow_duplicate=True),
    Output("validation-store", "data", allow_duplicate=True),
    Output("export-status", "children"),
    Output("trim-panel", "style"),
    Input("upload-audio", "contents"),
    State("upload-audio", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if contents is None:
        return (
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update,
        )

    _header, b64 = contents.split(",", 1)
    uploaded_raw = base64.b64decode(b64)
    raw = uploaded_raw

    # Auto-trim upload edges before any first waveform rendering if configured.
    clip_seconds = max(0.0, float(CLIP_ENDS))
    clip_status = ""
    if clip_seconds > 0:
        uploaded_duration = duration_from_wav_bytes(uploaded_raw)
        if uploaded_duration > 2.0 * clip_seconds:
            raw = filter_crop(
                uploaded_raw,
                start_s=clip_seconds,
                end_s=uploaded_duration - clip_seconds,
            )
            clip_status = f"Auto-clipped {clip_seconds:.2f}s from start and end."
        else:
            clip_status = (
                f"Auto-clip skipped because clip_ends={clip_seconds:.2f}s is too large "
                f"for a {uploaded_duration:.2f}s file."
            )

    wav_info = sf.info(io.BytesIO(raw))
    sr = int(wav_info.samplerate)
    original_wav_path = write_wav_bytes_to_cache(raw, "original")
    processed_wav_path = write_wav_bytes_to_cache(raw, "processed")
    duration_seconds = duration_from_wav_bytes(raw)

    audio_player = make_audio_player(raw, filename, sr)

    store = {
        "filename": filename,
        "original_wav_path": original_wav_path,
    }
    processed_store = {
        "processed_wav_path": processed_wav_path,
        "duration": duration_seconds,
    }
    preprocessing_label = "Raw audio"
    if clip_status:
        preprocessing_label = f"Raw audio | {clip_status}"

    return audio_player, store, processed_store, preprocessing_label, {"filter_name": "raw", "label": "Raw audio"}, None, None, {}, "Exports will reflect the current processed audio and validations.", {"display": "block"}


@app.callback(
    Output("trim-waveform", "figure"),
    Input("processed-audio-store", "data"),
    prevent_initial_call=True,
)
def update_trim_waveform(processed_store):
    """Update the trim waveform whenever processed audio changes."""
    y, sr = load_audio_for_plot((processed_store or {}).get("processed_wav_path"))
    if y is None:
        return dash.no_update
    return make_waveform_figure(y, sr)


@app.callback(
    Output("trim-selection-store", "data"),
    Input("trim-waveform", "selectedData"),
    prevent_initial_call=True,
)
def persist_trim_selection(selected_data):
    """Store the trim waveform selection."""
    return selected_data or None


@app.callback(
    Output("audio-player", "children", allow_duplicate=True),
    Output("audio-store", "data", allow_duplicate=True),
    Output("processed-audio-store", "data", allow_duplicate=True),
    Output("filter-store", "data", allow_duplicate=True),
    Output("selection-store", "data", allow_duplicate=True),
    Output("trim-status", "children"),
    Input("trim-apply-btn", "n_clicks"),
    State("audio-store", "data"),
    State("trim-selection-store", "data"),
    prevent_initial_call=True,
)
def apply_trim(_clicks, audio_store, trim_selection):
    if not audio_store:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Upload audio first."
    original_wav_path = audio_store.get("original_wav_path")
    if not original_wav_path or not os.path.exists(original_wav_path):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Original file not found."

    if not trim_selection or "range" not in (trim_selection or {}) or "x" not in (trim_selection.get("range") or {}):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, "No selection made. Use Box Select tool on the waveform."

    start_s, end_s = trim_selection["range"]["x"]
    start_s, end_s = sorted((float(start_s), float(end_s)))

    original_wav_bytes = read_wav_bytes(original_wav_path)
    cropped_bytes = filter_crop(original_wav_bytes, start_s=start_s, end_s=end_s)
    new_original_path = write_wav_bytes_to_cache(cropped_bytes, "original")
    new_processed_path = write_wav_bytes_to_cache(cropped_bytes, "processed")
    new_duration = duration_from_wav_bytes(cropped_bytes)

    # Load sample rate from original to display in audio player
    y, sr = load_audio_full(new_original_path)
    filename = audio_store.get("filename", "trimmed_audio.wav")
    new_audio_player = make_audio_player(cropped_bytes, filename, sr)

    new_audio_store = {**audio_store, "original_wav_path": new_original_path}
    new_processed_store = {"processed_wav_path": new_processed_path, "duration": new_duration}
    status = f"Trimmed to {start_s:.2f}–{end_s:.2f} s ({new_duration:.2f} s). Re-upload to restore original."
    return new_audio_player, new_audio_store, new_processed_store, {"filter_name": "raw", "label": "Raw audio"}, None, status


@app.callback(
    Output("processed-audio-store", "data", allow_duplicate=True),
    Output("preprocessing-status", "children", allow_duplicate=True),
    Output("filter-store", "data", allow_duplicate=True),
    Input({"type": "preprocessing-step-btn", "index": ALL}, "n_clicks"),
    Input("filter-revert", "n_clicks"),
    State("audio-store", "data"),
    State("lowpass-cutoff-hz", "value"),
    State("highpass-cutoff-hz", "value"),
    prevent_initial_call=True,
)
def update_preprocessing(_step_clicks, _revert_clicks, store, lowpass_hz, highpass_hz):
    if not store:
        return dash.no_update, "Upload audio first.", dash.no_update

    triggered = dash.ctx.triggered_id

    original_wav_path = (store or {}).get("original_wav_path")
    if not original_wav_path or not os.path.exists(original_wav_path):
        return dash.no_update, "Original upload could not be found. Please upload again.", dash.no_update
    original_wav_bytes = read_wav_bytes(original_wav_path)

    if triggered == "filter-revert":
        reverted_path = write_wav_bytes_to_cache(original_wav_bytes, "processed")
        return {
            "processed_wav_path": reverted_path,
            "duration": duration_from_wav_bytes(original_wav_bytes),
        }, "Reverted to original uploaded WAV", {"filter_name": "raw", "label": "Raw audio"}

    filter_name = "raw"
    if isinstance(triggered, dict) and triggered.get("type") == "preprocessing-step-btn":
        filter_name = triggered.get("index", "raw")

    params = None
    label_override = humanize_name(filter_name)
    if filter_name == "lowpass":
        cutoff = float(lowpass_hz or DEFAULT_LOWPASS_CUTOFF_HZ)
        params = {"cutoff_hz": cutoff}
        label_override = f"Low pass @ {int(cutoff)} Hz"
    elif filter_name == "highpass":
        cutoff = float(highpass_hz or DEFAULT_HIGHPASS_CUTOFF_HZ)
        params = {"cutoff_hz": cutoff}
        label_override = f"High pass @ {int(cutoff)} Hz"

    result = run_preprocessing_step(original_wav_bytes, filter_name, params=params)
    if isinstance(result, tuple):
        processed_wav_bytes, label = result
    else:
        processed_wav_bytes = result
        label = label_override
    processed_path = write_wav_bytes_to_cache(processed_wav_bytes, "processed")
    return {
        "processed_wav_path": processed_path,
        "duration": duration_from_wav_bytes(processed_wav_bytes),
    }, label, {"filter_name": filter_name, "label": label}


@app.callback(
    Output("lowpass-controls", "style"),
    Output("highpass-controls", "style"),
    Input("filter-store", "data"),
)
def toggle_freq_controls(filter_store):
    filter_name = (filter_store or {}).get("filter_name", "raw")
    show = {"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "8px"}
    hide = {"display": "none"}
    if filter_name == "lowpass":
        return show, hide
    if filter_name == "highpass":
        return hide, show
    return hide, hide


@app.callback(
    Output("processed-audio-store", "data", allow_duplicate=True),
    Output("preprocessing-status", "children", allow_duplicate=True),
    Output("filter-store", "data", allow_duplicate=True),
    Input("lowpass-cutoff-hz", "value"),
    Input("highpass-cutoff-hz", "value"),
    State("filter-store", "data"),
    State("audio-store", "data"),
    prevent_initial_call=True,
)
def update_preprocessing_on_freq_change(lowpass_hz, highpass_hz, filter_store, audio_store):
    filter_name = (filter_store or {}).get("filter_name", "raw")
    if filter_name not in ("lowpass", "highpass"):
        return dash.no_update, dash.no_update, dash.no_update

    original_wav_path = (audio_store or {}).get("original_wav_path")
    if not original_wav_path or not os.path.exists(original_wav_path):
        return dash.no_update, "Upload audio first.", dash.no_update
    original_wav_bytes = read_wav_bytes(original_wav_path)

    if filter_name == "lowpass":
        cutoff = float(lowpass_hz or DEFAULT_LOWPASS_CUTOFF_HZ)
        label = f"Low pass @ {int(cutoff)} Hz"
    else:
        cutoff = float(highpass_hz or DEFAULT_HIGHPASS_CUTOFF_HZ)
        label = f"High pass @ {int(cutoff)} Hz"

    result = run_preprocessing_step(original_wav_bytes, filter_name, params={"cutoff_hz": cutoff})
    processed_path = write_wav_bytes_to_cache(result, "processed")
    return (
        {"processed_wav_path": processed_path, "duration": duration_from_wav_bytes(result)},
        label,
        {"filter_name": filter_name, "label": label},
    )


@app.callback(
    Output("spectrogram", "figure"),
    Input("processed-audio-store", "data"),
    prevent_initial_call=True,
)
def update_spectrogram(processed_store):
    y, sr = load_audio_for_plot((processed_store or {}).get("processed_wav_path"))
    if y is None:
        return dash.no_update
    return make_spectrogram_figure(y, sr)


@app.callback(
    Output("waveform", "figure"),
    Input("processed-audio-store", "data"),
    prevent_initial_call=True,
)
def update_waveform(processed_store):
    y, sr = load_audio_for_plot((processed_store or {}).get("processed_wav_path"))
    if y is None:
        return dash.no_update
    return make_waveform_figure(y, sr)


@app.callback(
    Output("spectrogram", "figure", allow_duplicate=True),
    Output("waveform", "figure", allow_duplicate=True),
    Input("spectrogram", "relayoutData"),
    Input("waveform", "relayoutData"),
    prevent_initial_call=True,
)
def sync_xaxis(spec_relayout, wave_relayout):
    def get_x_update(relayout_data):
        """Return (range_or_None, is_reset) from a relayoutData dict."""
        if not relayout_data:
            return None, False
        if relayout_data.get("xaxis.autorange") or relayout_data.get("autosize"):
            return None, True
        if "xaxis.range[0]" in relayout_data:
            return [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]], False
        if "xaxis.range" in relayout_data:
            return relayout_data["xaxis.range"], False
        return None, False

    triggered = dash.ctx.triggered_id
    if triggered == "spectrogram":
        x_range, is_reset = get_x_update(spec_relayout)
    else:
        x_range, is_reset = get_x_update(wave_relayout)

    if not is_reset and x_range is None:
        return dash.no_update, dash.no_update

    patch = Patch()
    if is_reset:
        patch["layout"]["xaxis"]["autorange"] = True
    else:
        patch["layout"]["xaxis"]["range"] = x_range
        patch["layout"]["xaxis"]["autorange"] = False

    if triggered == "spectrogram":
        return dash.no_update, patch
    return patch, dash.no_update


@app.callback(
    Output("selection-store", "data"),
    Input("spectrogram", "selectedData"),
    Input("waveform", "selectedData"),
    prevent_initial_call=True,
)
def persist_selection(spec_selected, wave_selected):
    triggered = dash.ctx.triggered_id
    if triggered == "waveform":
        return wave_selected
    return spec_selected


@app.callback(
    Output("analysis-panel", "children"),
    Input("processed-audio-store", "data"),
    Input("selection-store", "data"),
    running=[
        (Output("analysis-status", "children"), "Analysis running... please wait.", "Analysis ready."),
    ],
    prevent_initial_call=True,
)
def on_selection(processed_store, selected_data):
    processed_wav_path = (processed_store or {}).get("processed_wav_path")
    y_plot, sr_plot = load_audio_for_plot(processed_wav_path)
    y_full, sr_full = load_audio_full(processed_wav_path)
    if y_plot is None or sr_plot is None or y_full is None or sr_full is None:
        return None

    # Use the same display signal as the main spectrogram so both views match visually.
    segment_info_plot = get_selected_segment(y_plot, sr_plot, selected_data)
    seg_plot = segment_info_plot["segment"]

    # Keep feature extraction on full-resolution audio.
    segment_info_full = get_selected_segment(y_full, sr_full, selected_data)
    seg_full = segment_info_full["segment"]

    if len(seg_full) < int(sr_full * 0.05):
        return html.P("Selected segment too short for analysis.", style={"color": "#c00"})

    # Write segment to a temp file so Disko_Sound can load it
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        sf.write(tmp_path, seg_full, sr_full)
        ds = Disko_Sound(tmp_path)

        features = {
            "duration (s)": ds.get_call_duration(),
            "dominant frequency (Hz)": ds.get_dominant_frequency(),
            "spectral centroid (Hz)": ds.get_spectral_centroid(),
            "spectral rolloff 95% (Hz)": ds.get_spectral_rolloff(),
            "bandwidth (Hz)": ds.get_bandwidth(),
            "RMS energy": ds.get_rms_energy(),
        }

        freq_min, freq_max = ds.get_frequency_range()
        features["freq range min (Hz)"] = freq_min
        features["freq range max (Hz)"] = freq_max
    finally:
        os.unlink(tmp_path)

    rows = [{"Feature": k, "Value": f"{v:.4g}"} for k, v in features.items()]

    seg_spec_fig = make_spectrogram_figure(seg_plot, sr_plot)
    seg_spec_fig.update_layout(
        height=260,
        margin=dict(l=60, r=20, t=10, b=50),
        title=None,
        dragmode="zoom",
    )

    seg_wave_fig = make_waveform_figure(seg_plot, sr_plot)
    seg_wave_fig.update_layout(
        height=140,
        margin=dict(l=60, r=20, t=4, b=40),
        dragmode="zoom",
    )

    heading = "Analysis — full signal"
    if not segment_info_plot["used_full_signal"]:
        heading = (
            f"Analysis — {segment_info_plot['t0']:.2f} s → {segment_info_plot['t1']:.2f} s "
            f"({segment_info_plot['duration']:.2f} s)"
        )

    return html.Div(
        [
            html.H3(heading, style={"marginBottom": "8px"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "minmax(360px, 420px) minmax(420px, 1fr)", "gap": "24px", "alignItems": "start"},
                children=[
                    html.Div(
                        dash_table.DataTable(
                            data=rows,
                            columns=[
                                {"name": "Feature", "id": "Feature"},
                                {"name": "Value", "id": "Value"},
                            ],
                            style_cell={
                                "textAlign": "left",
                                "padding": "6px 14px",
                                "fontSize": "13px",
                            },
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor": "#e8e8e8",
                            },
                            style_table={"minWidth": "360px"},
                        ),
                        style={"minWidth": 0},
                    ),
                    html.Div(
                        [
                            html.P("Spectrogram", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "4px", "color": "#666"}),
                            dcc.Graph(figure=seg_spec_fig, config={"displayModeBar": False}),
                            html.P("Waveform", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "4px", "marginTop": "8px", "color": "#666"}),
                            dcc.Graph(figure=seg_wave_fig, config={"displayModeBar": False}),
                        ],
                        style={"minWidth": 0},
                    ),
                ],
            ),
        ]
    )


@app.callback(
    Output("classification-model-store", "data"),
    Input({"type": "classification-model-btn", "index": ALL}, "n_clicks"),
    State("classification-model-store", "data"),
    prevent_initial_call=True,
)
def update_selected_classification_model(_clicks, current_model):
    triggered = dash.ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get("type") == "classification-model-btn":
        return triggered.get("index", current_model)
    return current_model


@app.callback(
    Output("detection-store", "data"),
    Output("classification-panel", "children"),
    Input("classification-model-store", "data"),
    State("processed-audio-store", "data"),
    State("selection-store", "data"),
    State("validation-store", "data"),
    running=[
        (Output("detection-status", "children"), "Detection and classification running... please wait.", "Detection ready."),
    ],
    prevent_initial_call=True,
)
def run_detection_placeholder_callback(selected_model, processed_store, selected_data, validation_store):
    if not selected_model:
        return [], render_detection_panel([], validation_store, selected_model)

    y, sr = load_audio_full((processed_store or {}).get("processed_wav_path"))
    if y is None or sr is None:
        return [], render_detection_panel([], validation_store, selected_model)

    segment_info = get_selected_segment(y, sr, selected_data)
    seg = segment_info["segment"]
    if len(seg) < int(sr * 0.05):
        return [], render_detection_panel([], validation_store, selected_model)

    wav_bytes = encode_wav_bytes(seg, sr)

    # Add or swap classification models in classification.py (run_classification_model).
    model_output = run_classification_model(wav_bytes, model_name=selected_model)

    detection_rows = []
    for event, probability, predicted in zip(
        model_output.get("event", []),
        model_output.get("probability", []),
        model_output.get("predicted", []),
    ):
        detection_rows.append(
            {
                "event": str(event),
                "probability": float(probability),
                "predicted": bool(predicted),
            }
        )

    panel = render_detection_panel(detection_rows, validation_store, selected_model)
    return detection_rows, panel


@app.callback(
    Output("validation-store", "data"),
    Output("export-status", "children", allow_duplicate=True),
    Input({"type": "validation-checkbox", "index": ALL}, "value"),
    State({"type": "validation-checkbox", "index": ALL}, "id"),
    State("detection-store", "data"),
    prevent_initial_call=True,
)
def persist_validations(values, ids, detection_rows):
    validation_map = {}
    for value, item_id in zip(values, ids):
        validation_map[item_id["index"]] = "validated" in value

    export_rows = build_export_rows(detection_rows, validation_map)
    save_validation_log(export_rows)
    validated_count = sum(validation_map.values())
    return validation_map, f"Saved {validated_count} validation decisions to exports/validation_log.json"


@app.callback(
    Output("download-spectrogram", "data"),
    Output("export-status", "children", allow_duplicate=True),
    Input("export-spectrogram-btn", "n_clicks"),
    State("processed-audio-store", "data"),
    State("selection-store", "data"),
    State("audio-store", "data"),
    State("filter-store", "data"),
    prevent_initial_call=True,
)
def export_spectrogram(_n_clicks, processed_store, selected_data, audio_store, filter_store):
    y, sr = load_audio_full((processed_store or {}).get("processed_wav_path"))
    if y is None or sr is None:
        return dash.no_update, "Upload audio before exporting the spectrogram."

    png_bytes = render_spectrogram_png(y, sr, selected_data)
    filename_root = slugify_filename((audio_store or {}).get("filename"))
    filter_name = (filter_store or {}).get("filter_name", "raw")
    filename = f"{filename_root}-{filter_name}-spectrogram.png"
    return dcc.send_bytes(png_bytes, filename), f"Exported {filename}"


@app.callback(
    Output("download-csv", "data"),
    Output("export-status", "children", allow_duplicate=True),
    Input("export-csv-btn", "n_clicks"),
    State("detection-store", "data"),
    State("validation-store", "data"),
    State("audio-store", "data"),
    State("filter-store", "data"),
    prevent_initial_call=True,
)
def export_detection_csv(_n_clicks, detection_rows, validation_store, audio_store, filter_store):
    export_rows = build_export_rows(detection_rows, validation_store)
    if not export_rows:
        return dash.no_update, "No detection rows available to export yet."

    export_df = pd.DataFrame(export_rows)
    os.makedirs("exports", exist_ok=True)
    export_df.to_csv("exports/detection_results.csv", index=False)

    filename_root = slugify_filename((audio_store or {}).get("filename"))
    filter_name = (filter_store or {}).get("filter_name", "raw")
    filename = f"{filename_root}-{filter_name}-detection-results.csv"
    return dcc.send_data_frame(export_df.to_csv, filename, index=False), f"Exported {filename} and updated exports/detection_results.csv"


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
