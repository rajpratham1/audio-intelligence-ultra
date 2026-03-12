import os
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import whisper
from transformers import pipeline


APP_TITLE = "Audio Intelligence Ultra"
WHISPER_MODEL_NAME = "base"
SENTIMENT_MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
TARGET_SAMPLE_RATE = 16000


def load_models() -> tuple[Any, Any]:
    print("Initializing AI models...")
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_NAME,
    )
    return whisper_model, sentiment_analyzer


WHISPER_MODEL, SENTIMENT_ANALYZER = load_models()


def normalize_audio(audio_array: np.ndarray) -> np.ndarray:
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    if np.issubdtype(audio_array.dtype, np.integer):
        max_value = np.iinfo(audio_array.dtype).max
        audio_array = audio_array.astype(np.float32) / max_value
    else:
        audio_array = audio_array.astype(np.float32)

    max_amplitude = np.max(np.abs(audio_array)) if audio_array.size else 0.0
    if max_amplitude > 1.0:
        audio_array = audio_array / max_amplitude

    return np.clip(audio_array, -1.0, 1.0)


def resample_audio(audio_array: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    if original_rate == target_rate or audio_array.size == 0:
        return audio_array.astype(np.float32)

    duration = audio_array.shape[0] / float(original_rate)
    target_length = max(int(round(duration * target_rate)), 1)

    original_positions = np.linspace(0.0, duration, num=audio_array.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)

    return np.interp(target_positions, original_positions, audio_array).astype(np.float32)


def prepare_audio(audio_input: tuple[int, np.ndarray] | None) -> tuple[np.ndarray | None, float]:
    if audio_input is None:
        return None, 0.0

    sample_rate, audio_array = audio_input
    if audio_array is None or len(audio_array) == 0:
        return None, 0.0

    audio_array = normalize_audio(np.asarray(audio_array))
    duration_seconds = float(audio_array.shape[0]) / float(sample_rate)
    audio_array = resample_audio(audio_array, sample_rate, TARGET_SAMPLE_RATE)

    return audio_array, duration_seconds


def create_sentiment_plot(label: str, score: float) -> go.Figure:
    positive = label.upper() == "POSITIVE"
    color = "#818cf8" if positive else "#f87171"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            title={"text": f"Sentiment: {label.title()}", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": color},
                "bgcolor": "#111827",
                "borderwidth": 2,
                "bordercolor": "#374151",
                "steps": [
                    {"range": [0, 40], "color": "#3f1d1d"},
                    {"range": [40, 60], "color": "#3f3720"},
                    {"range": [60, 100], "color": "#1f3b2d"},
                ],
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=24, r=24, t=60, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def build_sentiment_highlight(label: str, score: float) -> list[tuple[str, str]]:
    summary = f"{label.title()} ({score:.2f}%)"
    category = "positive" if label.upper() == "POSITIVE" else "negative"
    return [(summary, category)]


def process_audio(audio_input: tuple[int, np.ndarray] | None, task_type: str):
    if audio_input is None:
        empty_plot = create_sentiment_plot("Neutral", 0.0)
        return "Upload an audio file to begin.", "0", "0.0", "", [("No analysis yet", None)], empty_plot

    audio_array, duration_seconds = prepare_audio(audio_input)
    if audio_array is None:
        empty_plot = create_sentiment_plot("Neutral", 0.0)
        return "The uploaded audio could not be read.", "0", "0.0", "", [("No analysis yet", None)], empty_plot

    task = "translate" if task_type == "Translate to English" else "transcribe"

    try:
        result = WHISPER_MODEL.transcribe(audio_array, task=task, fp16=False, language=None)
    except Exception as exc:
        empty_plot = create_sentiment_plot("Neutral", 0.0)
        return (
            f"Audio processing failed: {exc}",
            "0",
            "0.0",
            "",
            [("Analysis failed", "negative")],
            empty_plot,
        )

    transcript = result.get("text", "").strip()
    if not transcript:
        empty_plot = create_sentiment_plot("Neutral", 0.0)
        return "No speech detected in the audio.", "0", "0.0", "", [("No speech detected", None)], empty_plot

    word_count = len(transcript.split())
    wpm = round(word_count / (duration_seconds / 60.0), 1) if duration_seconds > 0 else 0.0

    sentiment_result = SENTIMENT_ANALYZER(transcript[:512])[0]
    sentiment_label = sentiment_result["label"]
    sentiment_score = round(float(sentiment_result["score"]) * 100.0, 2)

    return (
        transcript,
        str(word_count),
        str(wpm),
        f"{duration_seconds:.2f} sec",
        build_sentiment_highlight(sentiment_label, sentiment_score),
        create_sentiment_plot(sentiment_label, sentiment_score),
    )


CUSTOM_CSS = """
body {
    background:
        radial-gradient(circle at top left, rgba(99, 102, 241, 0.22), transparent 30%),
        linear-gradient(180deg, #020617 0%, #0f172a 55%, #111827 100%);
}
.gradio-container {
    max-width: 1200px !important;
}
.panel-card {
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.72);
    backdrop-filter: blur(12px);
}
"""


theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
)


with gr.Blocks(theme=theme, css=CUSTOM_CSS, title=APP_TITLE) as app:
    gr.Markdown(
        """
        # Audio Intelligence Ultra
        Upload an audio file, then run either transcription or translation to English.
        The app also calculates word count, speaking speed, and transcript sentiment.
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel-card"):
            audio_input = gr.Audio(
                label="Upload Audio File",
                type="numpy",
            )
            task_radio = gr.Radio(
                ["Transcribe", "Translate to English"],
                value="Transcribe",
                label="Task",
            )
            analyze_button = gr.Button("Run Analysis", variant="primary")

        with gr.Column(scale=2, elem_classes="panel-card"):
            with gr.Row():
                words_output = gr.Textbox(label="Words", interactive=False)
                wpm_output = gr.Textbox(label="Speed (WPM)", interactive=False)
                duration_output = gr.Textbox(label="Duration", interactive=False)

            transcript_output = gr.Textbox(
                label="Full Transcript",
                lines=12,
                max_lines=18,
                interactive=False,
                show_copy_button=True,
            )

            with gr.Row():
                sentiment_output = gr.HighlightedText(
                    label="Sentiment",
                    color_map={"positive": "#14532d", "negative": "#7f1d1d"},
                    interactive=False,
                )
                gauge_output = gr.Plot(label="Sentiment Gauge")

    analyze_button.click(
        fn=process_audio,
        inputs=[audio_input, task_radio],
        outputs=[
            transcript_output,
            words_output,
            wpm_output,
            duration_output,
            sentiment_output,
            gauge_output,
        ],
    )


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("PORT", "7860"))
    app.launch(server_name=server_name, server_port=server_port)
