"""
Whale Sound Analyzer — Dash dashboard
  • Upload a .wav file
  • Play it back in the browser
  • View the spectrogram
  • Box-select a time range on the spectrogram → run Disko_Sound analysis
"""

import base64
import io
import json
import os
import tempfile

import dash
from dash import Input, Output, State, ALL, dash_table, dcc, html
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np
import librosa
import pandas as pd
import soundfile as sf

from classification import CLASSIFICATION_MODELS, run_classification_model
from disko_sound import Disko_Sound
from preprocessing import PREPROCESSING_STEPS, run_preprocessing_step

# ── App ────────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="Whale Sound Analyzer")

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
                html.Div(id="preprocessing-status", style={"color": "#444", "fontSize": "13px"}),
            ],
            style={"marginTop": "24px"},
        ),
        # ── Spectrogram ────────────────────────────────────────────────────────
        html.Div(
            [
                html.H3("Spectrogram", style={"marginBottom": "4px"}),
                html.P(
                    "Use the Box Select tool (□) in the toolbar to pick a time region for analysis.",
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
        # ── Analysis panel ─────────────────────────────────────────────────────
        html.Div(id="analysis-status", style={"marginTop": "20px", "color": "#666", "fontSize": "13px"}),
        html.Div(id="analysis-panel", style={"marginTop": "28px"}),
        html.Div(id="detection-status", style={"marginTop": "20px", "color": "#666", "fontSize": "13px"}),
        html.Div(id="classification-panel", style={"marginTop": "28px"}),
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
        dcc.Store(id="filter-store", data={"filter_name": "raw", "label": "Raw audio"}),
        dcc.Store(id="classification-model-store", data=None),
        dcc.Store(id="detection-store"),
        dcc.Store(id="validation-store", data={}),
    ],
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def decode_audio(contents: str):
    """dcc.Upload content string → (y float32 array, sr int, raw bytes)."""
    _header, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    y, sr = librosa.load(buf, sr=None, mono=True)
    return y, sr, raw


def decode_wav_bytes(wav_bytes: bytes):
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
    return y.astype(np.float32), int(sr)


def encode_wav_bytes(y: np.ndarray, sr: int):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def serialize_audio(y: np.ndarray, sr: int):
    return {"y": y.tolist(), "sr": int(sr), "duration": float(len(y) / sr)}


def deserialize_audio(store: dict):
    if not store:
        return None, None
    return np.asarray(store["y"], dtype=np.float32), int(store["sr"])


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

def render_spectrogram_png(y: np.ndarray, sr: int, selected_data: dict | None):
    n_fft = 2048
    hop = 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    times = librosa.frames_to_time(np.arange(spectrogram_db.shape[1]), sr=sr, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=160)
    mesh = ax.pcolormesh(times, freqs, spectrogram_db, shading="auto", cmap="viridis", vmin=-80, vmax=0)
    max_freq = min(int(freqs[-1]), 8000)
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


def make_spectrogram_figure(y: np.ndarray, sr: int):
    n_fft = 2048
    hop = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    times = librosa.frames_to_time(
        np.arange(S_db.shape[1]), sr=sr, hop_length=hop
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    max_freq = min(int(freqs[-1]), 8000)

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


def render_detection_panel(detection_rows, validation_store=None, selected_model=None):
    detection_rows = detection_rows or []
    validation_store = validation_store or {}
    sorted_rows = sorted(detection_rows, key=lambda row: row["probability"], reverse=True)
    top_row = sorted_rows[0] if sorted_rows else None

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

    subtitle = "Choose a classification model below to run detection on the current selection."
    if selected_model and not top_row:
        subtitle = f"Model '{humanize_name(selected_model)}' ran, but produced no rows for the current selection."
    if top_row:
        subtitle = f"Top predicted class: {top_row['event']} ({top_row['probability']:.4f})."

    return html.Div(
        [
            html.H3("Detection & Classification", style={"marginBottom": "8px"}),
            html.P("Available classification models:", style={"color": "#444", "marginBottom": "6px"}),
            html.Div(model_buttons, style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"}),
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
    Input("upload-audio", "contents"),
    State("upload-audio", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if contents is None:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    y, sr, raw = decode_audio(contents)

    audio_player = html.Div(
        [
            html.P(
                f"▶  {filename}   |   {sr:,} Hz   |   {len(y)/sr:.2f} s",
                style={"margin": "8px 0 4px", "fontSize": "13px", "color": "#444"},
            ),
            html.Audio(
                src=contents,
                controls=True,
                style={"width": "100%"},
            ),
        ],
        style={"marginBottom": "8px"},
    )

    store = {
        "contents": contents,
        "sr": int(sr),
        "filename": filename,
        "original_wav_b64": base64.b64encode(raw).decode("utf-8"),
    }
    processed_store = serialize_audio(y, sr)
    return audio_player, store, processed_store, "Raw audio", {"filter_name": "raw", "label": "Raw audio"}, None, None, {}, "Exports will reflect the current processed audio and validations."


@app.callback(
    Output("processed-audio-store", "data", allow_duplicate=True),
    Output("preprocessing-status", "children", allow_duplicate=True),
    Output("filter-store", "data", allow_duplicate=True),
    Input({"type": "preprocessing-step-btn", "index": ALL}, "n_clicks"),
    Input("filter-revert", "n_clicks"),
    State("audio-store", "data"),
    prevent_initial_call=True,
)
def update_preprocessing(_step_clicks, _revert_clicks, store):
    if not store:
        return dash.no_update, "Upload audio first.", dash.no_update

    triggered = dash.ctx.triggered_id

    original_wav_bytes = base64.b64decode(store["original_wav_b64"])

    if triggered == "filter-revert":
        y, sr = decode_wav_bytes(original_wav_bytes)
        return serialize_audio(y, sr), "Reverted to original uploaded WAV", {"filter_name": "raw", "label": "Raw audio"}

    filter_name = "raw"
    if isinstance(triggered, dict) and triggered.get("type") == "preprocessing-step-btn":
        filter_name = triggered.get("index", "raw")

    processed_wav_bytes, label = run_preprocessing_step(original_wav_bytes, filter_name)
    y, sr = decode_wav_bytes(processed_wav_bytes)
    return serialize_audio(y, sr), label, {"filter_name": filter_name, "label": label}


@app.callback(
    Output("spectrogram", "figure"),
    Input("processed-audio-store", "data"),
    prevent_initial_call=True,
)
def update_spectrogram(processed_store):
    y, sr = deserialize_audio(processed_store)
    if y is None:
        return dash.no_update
    return make_spectrogram_figure(y, sr)


@app.callback(
    Output("selection-store", "data"),
    Input("spectrogram", "selectedData"),
    prevent_initial_call=True,
)
def persist_selection(selected_data):
    return selected_data


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
    y, sr = deserialize_audio(processed_store)
    if y is None or sr is None:
        return None

    segment_info = get_selected_segment(y, sr, selected_data)
    seg = segment_info["segment"]

    if len(seg) < int(sr * 0.05):
        return html.P("Selected segment too short for analysis.", style={"color": "#c00"})

    # Write segment to a temp file so Disko_Sound can load it
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        sf.write(tmp_path, seg, sr)
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

    seg_fig = make_spectrogram_figure(seg, sr)
    seg_fig.update_layout(
        height=320,
        margin=dict(l=60, r=20, t=10, b=50),
        title=None,
        dragmode="zoom",
    )

    heading = "Analysis — full signal"
    if not segment_info["used_full_signal"]:
        heading = (
            f"Analysis — {segment_info['t0']:.2f} s → {segment_info['t1']:.2f} s "
            f"({segment_info['duration']:.2f} s)"
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
                        dcc.Graph(figure=seg_fig, config={"displayModeBar": False}),
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

    y, sr = deserialize_audio(processed_store)
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
    y, sr = deserialize_audio(processed_store)
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
