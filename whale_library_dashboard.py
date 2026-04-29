"""
Whale call library dashboard
• Go to http://127.0.0.1:8051/ in browser after running the code
• Choose a whale species from the dropdown menu
• Upload an unknown .wav file
• Choose a preprocessing filter
• See side-by-side comparison of waveform, spectrogram, and call info
• Manual validation time! :) Weehoo!
"""

import base64
import io
import os
import tempfile

import dash
from dash import html, dcc, Input, Output, State, ALL
import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf

# ── Import preprocessing ──────────────────────────────────────────────────────
try:
    from preprocessing import PREPROCESSING_STEPS, run_preprocessing_step
    PREPROCESSING_AVAILABLE = True
except Exception:
    PREPROCESSING_AVAILABLE = False
    PREPROCESSING_STEPS = {"raw": None}

    def run_preprocessing_step(wav_bytes: bytes, step_name: str):
        return wav_bytes, "Raw audio (preprocessing unavailable)"

# ── Import feature extraction ─────────────────────────────────────────────────
try:
    from disko_sound import Disko_Sound
    DISKO_AVAILABLE = True
except Exception:
    DISKO_AVAILABLE = False

# ── Import model ──────────────────────────────────────────────────────────────
try:
    from classification import run_classification_model
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

    # This is my "dummy model" in case TensorFlow doesn't run. It only shows a value of 0.25.
    def run_classification_model(wav_bytes, model_name=None):
        return {
            "event": ["Narwhal", "Fin whale", "Humpback whale", "Beluga whale"],
            "probability": [0.25, 0.25, 0.25, 0.25],
            "predicted": [0, 0, 0, 0],
        }

# ── Whale data ────────────────────────────────────────────────────────────────
whale_data = {
    "Narwhal": {
        "call_types": "...",
        "frequency_range": "... – ...",
        "typical_duration": "... – ...",
        "example_files": [
            "known_samples/narwhal/Narwhal.wav",
            "known_samples/narwhal/narwhal-voicesofthesea.wav",
        ],
    },
    "Fin whale": {
        "call_types": "...",
        "frequency_range": "... – ...",
        "typical_duration": "... – ...",
        "example_files": [
            "known_samples/fin/fin.wav",
            "known_samples/fin/finWhale.wav",
        ],
    },
    "Humpback whale": {
        "call_types": "...",
        "frequency_range": "... – ...",
        "typical_duration": "... – ...",
        "example_files": [
            "known_samples/humpback/humpback_bubblenetFeeding.wav",
            "known_samples/humpback/humpback_socialSounds.wav",
            "known_samples/humpback/humpbackvoicesofthesea1.wav",
            "known_samples/humpback/humpback-whale-song-skj-bay_MarianneRasmussen.wav",
        ],
    },
    "Beluga whale": {
        "call_types": "...",
        "frequency_range": "... – ...",
        "typical_duration": "... – ...",
        "example_files": [
            "known_samples/beluga/beluga_clicks.wav",
            "known_samples/beluga/beluga_socialSounds.wav",
            "known_samples/beluga/beluga-voicesofthesea.wav",
            "known_samples/beluga/beluga.wav",
        ],
    },
}

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Whale call library")

# ── Style constants ───────────────────────────────────────────────────────────
COLORS = {
    "bg": "#fafafa",
    "surface": "#ffffff",
    "border": "#e0e0e0",
    "text": "#1a1a2e",
    "muted": "#666",
    "accent": "#2f6fdd",
    "accent_light": "#e8f0fb",
}

SELECTOR_BTN_BASE = {
    "padding": "7px 14px",
    "border": "1px solid #ccc",
    "borderRadius": "6px",
    "cursor": "pointer",
    "fontSize": "13px",
    "fontFamily": "inherit",
}

TABLE_CELL = {
    "padding": "7px 12px",
    "borderBottom": f"1px solid {COLORS['border']}",
    "fontSize": "13px",
}

TABLE_HEADER = {
    **TABLE_CELL,
    "fontWeight": "600",
    "backgroundColor": "#f0f0f0",
    "borderBottom": f"1px solid {COLORS['border']}",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def decode_wav_bytes(wav_bytes: bytes):
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
    return y.astype(np.float32), int(sr)


def encode_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def load_audio(file_path: str):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


def safe_preprocess(wav_bytes: bytes, filter_name: str):
    result = run_preprocessing_step(wav_bytes, filter_name)
    if isinstance(result, tuple):
        return result[0]
    return result


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
        margin=dict(l=50, r=10, t=10, b=40),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor="#f5f8ff",
        xaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
        yaxis=dict(showgrid=True, gridcolor="#e8e8e8"),
    )
    return fig


def make_spectrogram_figure(y: np.ndarray, sr: int) -> go.Figure:
    n_fft = 2048
    hop = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    max_freq = min(int(freqs[-1]), 8000)

    fig = go.Figure(
        go.Heatmap(
            z=S_db,
            x=times,
            y=freqs,
            colorscale="Viridis",
            colorbar=dict(title="dB", thickness=12, len=0.8),
            zmin=-80,
            zmax=0,
            hovertemplate="Time: %{x:.2f} s<br>Freq: %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=60, r=20, t=10, b=50),
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis=dict(range=[0, max_freq]),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor="#111",
    )
    return fig


def extract_features(y: np.ndarray, sr: int) -> dict | None:
    if not DISKO_AVAILABLE:
        return None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        sf.write(tmp_path, y, sr)
        ds = Disko_Sound(tmp_path)
        freq_min, freq_max = ds.get_frequency_range()
        features = {
            "dominant_freq": ds.get_dominant_frequency(),
            "spectral_centroid": ds.get_spectral_centroid(),
            "spectral_rolloff": ds.get_spectral_rolloff(),
            "bandwidth": ds.get_bandwidth(),
            "freq_min": freq_min,
            "freq_max": freq_max,
            "rms_energy": ds.get_rms_energy(),
            "duration": ds.get_call_duration(),
            "zero_crossing_rate": ds.get_zero_crossing_rate(),
            "snr_db": ds.get_signal_to_noise_ratio(),
        }
        os.unlink(tmp_path)
        return features
    except Exception as e:
        return {"_error": str(e)}


def render_species_info_table(whale_info: dict, species_name: str) -> html.Div:
    """Full-width table showing species-level call info. Rendered once above the comparison."""
    rows_data = [
        ("Call types", whale_info.get("call_types", "—")),
        ("Typical frequency range", whale_info.get("frequency_range", "—")),
        ("Typical call duration", whale_info.get("typical_duration", "—")),
    ]

    table_rows = [
        html.Tr([
            html.Td(k, style={**TABLE_CELL, "width": "30%", "fontWeight": "500"}),
            html.Td(v, style=TABLE_CELL),
        ])
        for k, v in rows_data
    ]

    return html.Div(
        [
            html.H5(
                f"🐋 {species_name} — call information",
                style={"marginBottom": "8px", "color": COLORS["text"], "fontWeight": "600"},
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Property", style={**TABLE_HEADER, "width": "30%"}),
                            html.Th("Value", style=TABLE_HEADER),
                        ])),
                        html.Tbody(table_rows),
                    ],
                    style={"width": "100%", "borderCollapse": "collapse", "backgroundColor": COLORS["surface"]},
                ),
                style={"border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "overflow": "hidden"},
            ),
        ],
        style={"marginBottom": "24px"},
    )


def render_call_table(features: dict | None, label: str) -> html.Div:
    """Per-file measured acoustic parameters from Disko_Sound."""
    title = html.H5(
        "📊 Call parameters",
        style={"marginBottom": "8px", "color": COLORS["text"], "fontWeight": "600"},
    )

    if features is None:
        return html.Div([title, html.P("Feature extraction unavailable (disko_sound not loaded).", style={"color": COLORS["muted"], "fontSize": "13px"})])

    if "_error" in features:
        return html.Div([title, html.P(f"Extraction error: {features['_error']}", style={"color": "#c00", "fontSize": "13px"})])

    rows_data = [
        ("Dominant frequency", f"{features['dominant_freq']:.1f} Hz"),
        ("Spectral centroid", f"{features['spectral_centroid']:.1f} Hz"),
        ("Spectral rolloff (95%)", f"{features['spectral_rolloff']:.1f} Hz"),
        ("Bandwidth", f"{features['bandwidth']:.1f} Hz"),
        ("Freq range min", f"{features['freq_min']:.1f} Hz"),
        ("Freq range max", f"{features['freq_max']:.1f} Hz"),
        ("RMS energy", f"{features['rms_energy']:.6f}"),
        ("Measured duration", f"{features['duration']:.3f} s"),
        ("Zero crossing rate", f"{features['zero_crossing_rate']:.5f}"),
        ("Est. SNR", f"{features['snr_db']:.2f} dB"),
    ]

    table_rows = [
        html.Tr([
            html.Td(k, style=TABLE_CELL),
            html.Td(v, style={**TABLE_CELL, "fontWeight": "500", "color": COLORS["accent"]}),
        ])
        for k, v in rows_data
    ]

    return html.Div(
        [
            title,
            html.Div(
                html.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Parameter", style=TABLE_HEADER),
                            html.Th("Value", style=TABLE_HEADER),
                        ])),
                        html.Tbody(table_rows),
                    ],
                    style={"width": "100%", "borderCollapse": "collapse", "backgroundColor": COLORS["surface"]},
                ),
                style={"border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "overflow": "hidden"},
            ),
        ]
    )


def render_model_output(result: dict, selected_whale: str) -> html.Div:
    rows = []
    for e, p in zip(result["event"], result["probability"]):
        is_match = selected_whale and e.lower() == selected_whale.lower()
        rows.append(
            html.Tr([
                html.Td(e, style={**TABLE_CELL, "fontWeight": "bold" if is_match else "normal", "color": "green" if is_match else COLORS["text"]}),
                html.Td(f"{p:.4f}", style=TABLE_CELL),
            ])
        )
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Class", style=TABLE_HEADER),
                    html.Th("Probability", style=TABLE_HEADER),
                ])),
                html.Tbody(rows),
            ],
            style={"width": "100%", "borderCollapse": "collapse", "backgroundColor": COLORS["surface"]},
        ),
        style={"border": f"1px solid {COLORS['border']}", "borderRadius": "6px", "overflow": "hidden"},
    )


def render_preprocessing_buttons(selected: str) -> list:
    buttons = []
    for name in PREPROCESSING_STEPS.keys():
        is_active = name == selected
        buttons.append(
            html.Button(
                humanize(name),
                id={"type": "preproc-btn", "index": name},
                n_clicks=0,
                style={
                    **SELECTOR_BTN_BASE,
                    "backgroundColor": COLORS["accent"] if is_active else COLORS["surface"],
                    "color": "#fff" if is_active else "#333",
                },
            )
        )
    buttons.append(
        html.Button(
            "Revert To Original",
            id="preproc-revert",
            n_clicks=0,
            style={**SELECTOR_BTN_BASE, "backgroundColor": COLORS["surface"], "color": "#333"},
        )
    )
    return buttons


def build_audio_block(y: np.ndarray, sr: int, wav_bytes: bytes, audio_src: str,
                      label: str, selected_whale: str) -> html.Div:
    """Build a single audio column: waveform + spectrogram + audio player + call params + classification."""
    waveform_fig = make_waveform_figure(y, sr)
    spectrogram_fig = make_spectrogram_figure(y, sr)
    features = extract_features(y, sr)
    model_result = run_classification_model(wav_bytes)

    return html.Div(
        style={
            "backgroundColor": COLORS["surface"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "10px",
            "padding": "16px",
        },
        children=[
            html.H4(label, style={"marginTop": 0, "marginBottom": "12px", "color": COLORS["text"]}),

            # Waveform
            html.P("Waveform", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "4px", "color": COLORS["muted"]}),
            dcc.Graph(figure=waveform_fig, config={"displayModeBar": False}),

            # Spectrogram
            html.P("Spectrogram", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "4px", "marginTop": "12px", "color": COLORS["muted"]}),
            dcc.Graph(figure=spectrogram_fig, config={"displayModeBar": False}),

            # Audio player
            html.Audio(src=audio_src, controls=True, style={"width": "100%", "marginTop": "10px", "marginBottom": "16px"}),

            # Measured call parameters from Disko_Sound
            render_call_table(features, label),
            html.Div(style={"height": "12px"}),

            # Classification model output
            html.H5("Classification", style={"marginBottom": "8px", "fontWeight": "600"}),
            render_model_output(model_result, selected_whale),
        ],
    )


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={
        "maxWidth": "1200px",
        "margin": "auto",
        "padding": "28px 24px",
        "backgroundColor": COLORS["bg"],
        "fontFamily": "'Segoe UI', Arial, sans-serif",
    },
    children=[
        html.H1("🐋 Whale Call Library", style={"marginBottom": "4px"}),
        html.P(
            "Select a species, upload an unknown file, and compare them side by side!",
            style={"color": COLORS["muted"], "marginTop": 0},
        ),

        # Species selector
        html.Div(
            [
                html.Label("Species", style={"fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                dcc.Dropdown(
                    id="whale-dropdown",
                    options=[{"label": k, "value": k} for k in whale_data],
                    value="Narwhal",
                    clearable=False,
                    style={"maxWidth": "320px"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),

        # Upload
        dcc.Upload(
            id="upload-audio",
            children=html.Div(["Drag and drop or ", html.A("select a .wav file")]),
            style={
                "width": "100%",
                "height": "70px",
                "lineHeight": "70px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "8px",
                "textAlign": "center",
                "cursor": "pointer",
                "marginBottom": "16px",
                "color": COLORS["muted"],
            },
            accept=".wav",
        ),

        # Preprocessing panel
        html.Div(
            [
                html.H3("Preprocessing", style={"marginBottom": "6px"}),
                html.P(
                    "Choose a preprocessing filter to apply to the audio files. Applies to both the known and unknown audio.",
                    style={"color": COLORS["muted"], "fontSize": "13px", "marginTop": 0},
                ),
                html.Div(
                    id="preproc-buttons",
                    children=render_preprocessing_buttons("raw"),
                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "8px"},
                ),
                html.Div(id="preproc-status", style={"color": COLORS["muted"], "fontSize": "13px"}),
            ],
            style={"marginBottom": "24px"},
        ),

        # Stores
        dcc.Store(id="uploaded-audio-store"),
        dcc.Store(id="filter-store", data={"filter_name": "raw"}),

        # Main content (species info table + comparison grid)
        html.Div(id="whale-content"),
    ],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("uploaded-audio-store", "data"),
    Input("upload-audio", "contents"),
    prevent_initial_call=True,
)
def store_upload(contents):
    return contents


@app.callback(
    Output("preproc-buttons", "children"),
    Output("filter-store", "data"),
    Output("preproc-status", "children"),
    Input({"type": "preproc-btn", "index": ALL}, "n_clicks"),
    Input("preproc-revert", "n_clicks"),
    State("filter-store", "data"),
    prevent_initial_call=True,
)
def update_filter(_step_clicks, _revert_clicks, current_filter):
    triggered = dash.ctx.triggered_id

    if triggered == "preproc-revert":
        new_filter = "raw"
        status = "Reverted to original audio."
    elif isinstance(triggered, dict) and triggered.get("type") == "preproc-btn":
        new_filter = triggered.get("index", "raw")
        status = f"Filter applied: {humanize(new_filter)}"
    else:
        new_filter = (current_filter or {}).get("filter_name", "raw")
        status = dash.no_update

    return render_preprocessing_buttons(new_filter), {"filter_name": new_filter}, status


@app.callback(
    Output("whale-content", "children"),
    Input("whale-dropdown", "value"),
    Input("uploaded-audio-store", "data"),
    Input("filter-store", "data"),
)
def update_content(selected, uploaded_contents, filter_store):
    data = whale_data[selected]
    filter_name = (filter_store or {}).get("filter_name", "raw")

    # Full-width species info table at the top — shown once, not per file
    layout = [
        html.H2(selected, style={"marginBottom": "12px"}),
        render_species_info_table(data, selected),
        html.Hr(),
        html.H3("Side-by-side comparison", style={"marginBottom": "16px"}),
    ]

    # ── Uploaded unknown block ─────────────────────────────────
    uploaded_block = None

    if uploaded_contents:
        try:
            _header, b64 = uploaded_contents.split(",", 1)
            raw_bytes = base64.b64decode(b64)

            processed_bytes = safe_preprocess(raw_bytes, filter_name)
            y_u, sr_u = decode_wav_bytes(processed_bytes)

            player_buf = io.BytesIO()
            sf.write(player_buf, y_u, sr_u, format="WAV")
            player_b64 = base64.b64encode(player_buf.getvalue()).decode()
            audio_src = f"data:audio/wav;base64,{player_b64}"

            uploaded_block = build_audio_block(
                y_u, sr_u, processed_bytes, audio_src,
                "Unknown file", selected,
            )
        except Exception as e:
            uploaded_block = html.Div(
                [html.H4("Unknown file"), html.P(f"Error processing upload: {e}", style={"color": "#c00"})],
                style={"backgroundColor": COLORS["surface"], "border": f"1px solid {COLORS['border']}", "borderRadius": "10px", "padding": "16px"},
            )

    # ── Known example blocks ───────────────────────────────────
    for file_path in data["example_files"]:
        try:
            y, sr = load_audio(file_path)
            raw_bytes = encode_wav_bytes(y, sr)

            processed_bytes = safe_preprocess(raw_bytes, filter_name)
            y_p, sr_p = decode_wav_bytes(processed_bytes)

            player_b64 = base64.b64encode(processed_bytes).decode()
            audio_src = f"data:audio/wav;base64,{player_b64}"

            known_block = build_audio_block(
                y_p, sr_p, processed_bytes, audio_src,
                f"Known — {os.path.basename(file_path)}", selected,
            )

            if uploaded_block:
                row = html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "20px",
                        "marginBottom": "32px",
                    },
                    children=[known_block, uploaded_block],
                )
            else:
                row = html.Div(known_block, style={"marginBottom": "32px"})

            layout.append(row)

        except Exception as e:
            layout.append(html.P(f"Error loading {file_path}: {e}", style={"color": "#c00"}))

    return layout


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8051)