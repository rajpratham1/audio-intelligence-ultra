from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import whisper
from transformers import pipeline


print("Initializing Whisper, sentiment, and summarization models...")
whisper_model = whisper.load_model("base")
sentiment_analyzer = pipeline("sentiment-analysis")

try:
    summarizer = pipeline("text2text-generation", model="t5-small")
except Exception as exc:
    print(f"Warning: summarization model failed to load: {exc}")
    summarizer = None


def create_sentiment_plot(label: str, score: float):
    color = "#6366f1" if label == "POSITIVE" else "#ef4444"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Sentiment: {label}", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": color},
                "bgcolor": "#1f2937",
                "borderwidth": 2,
                "bordercolor": "#374151",
            },
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def export_transcript(text: str) -> str:
    file_path = Path("transcript_export.txt")
    file_path.write_text(text, encoding="utf-8")
    return str(file_path)


def process_audio(audio_path, task_type):
    if not audio_path:
        return "No audio provided.", None, [], "0 words", "0 WPM", "No summary available.", None

    task = "translate" if task_type == "Translate to English" else "transcribe"

    try:
        result = whisper_model.transcribe(audio_path, task=task, fp16=False)
    except Exception as exc:
        message = f"Audio processing failed: {exc}"
        return message, None, [], "0 words", "0 WPM", "No summary available.", None

    text = result.get("text", "").strip()
    segments = result.get("segments") or []
    duration = segments[-1].get("end", 0) if segments else 0
    word_count = len(text.split())
    wpm = round((word_count / (duration / 60)), 1) if duration > 0 else 0

    sentiment = sentiment_analyzer(text[:512] or "No transcript available.")[0]
    label = sentiment["label"]
    score = round(sentiment["score"] * 100, 2)
    plot = create_sentiment_plot(label, score)

    summary = "Summary model not available."
    if summarizer:
        if word_count > 20:
            try:
                summary = summarizer(
                    f"summarize: {text[:512]}",
                    max_length=60,
                    min_length=10,
                    do_sample=False,
                )[0]["generated_text"]
            except Exception:
                summary = "Summary generation failed."
        else:
            summary = "Text too short to summarize."

    file_out = export_transcript(text)
    highlighted = [(label, label)]
    return text, plot, highlighted, f"{word_count} words", f"{wpm} WPM", summary, file_out


custom_css = """
.gradio-container { background: #0f172a !important; padding: 10px !important; }
.stat-box { border: 1px solid #374151; background: #1f2937; padding: 10px; border-radius: 8px; font-size: 14px; }
#submit_btn { margin-top: 10px; }
@media (max-width: 600px) {
    .stat-box { font-size: 12px; }
    h1 { font-size: 24px !important; }
}
"""


with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo"), css=custom_css) as app:
    gr.Markdown("# Audio Intelligence Ultra")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            audio_input = gr.Audio(type="filepath", label="Audio Source")
            task_radio = gr.Radio(
                ["Transcribe (Original)", "Translate to English"],
                value="Transcribe (Original)",
                label="Task",
            )
            submit_btn = gr.Button("Run Analysis", variant="primary", elem_id="submit_btn")
            download_out = gr.File(label="Download Transcript")

        with gr.Column(scale=1, min_width=300):
            with gr.Row():
                word_stat = gr.Textbox(label="Words", elem_classes="stat-box")
                wpm_stat = gr.Textbox(label="Speed", elem_classes="stat-box")

            with gr.Tabs():
                with gr.TabItem("Text"):
                    text_output = gr.Textbox(label="Full Transcript", lines=6)
                with gr.TabItem("AI Summary"):
                    summary_output = gr.Textbox(label="Key Points", lines=4)
                with gr.TabItem("Analytics"):
                    sentiment_label = gr.HighlightedText(label="Sentiment")
                    plot_output = gr.Plot()

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, task_radio],
        outputs=[text_output, plot_output, sentiment_label, word_stat, wpm_stat, summary_output, download_out],
    )


if __name__ == "__main__":
    app.launch(debug=True)
