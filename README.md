# Audio Intelligence Ultra

This repository now includes a runnable Gradio application in [`app.py`](/c:/Users/rajpr/OneDrive/Desktop/t/app.py), a dependency list in [`requirements.txt`](/c:/Users/rajpr/OneDrive/Desktop/t/requirements.txt), and the original prototype notebook in [`Untitled0.ipynb`](/c:/Users/rajpr/OneDrive/Desktop/t/Untitled0.ipynb).

The notebook was the original source of the project logic. Its transcription, translation, summarization, sentiment, and export flow has now been extracted into `app.py` so the project can be run as a normal Python application outside Jupyter.

## Project Components

- [`app.py`](/c:/Users/rajpr/OneDrive/Desktop/t/app.py): standalone Gradio application for audio transcription and analysis
- [`requirements.txt`](/c:/Users/rajpr/OneDrive/Desktop/t/requirements.txt): Python dependencies needed to run the app
- [`Untitled0.ipynb`](/c:/Users/rajpr/OneDrive/Desktop/t/Untitled0.ipynb): original notebook prototype, still useful for reference and experimentation

## What The App Does

The extracted app implements an end-to-end audio intelligence workflow:

1. Installs the required Python packages inside the notebook environment.
2. Loads the Whisper `base` model for speech-to-text.
3. Loads a Hugging Face sentiment-analysis pipeline.
4. Tries to load a `t5-small` text-generation pipeline for summarization.
5. Accepts an audio file path through a Gradio interface.
6. Runs either transcription in the original language or translation to English.
7. Calculates word count and speaking speed in words per minute.
8. Scores transcript sentiment and renders it in a Plotly gauge.
9. Generates a short summary when the transcript is long enough.
10. Exports the transcript to `transcript_export.txt` for download.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

After startup, open the local Gradio URL shown in the terminal.

## Notebook Structure

The notebook has 8 code cells with the following purpose:

1. Dependency installation for `openai-whisper`, `transformers`, `torch`, `nltk`, and `pyarrow`.
2. A small proof-of-concept showing Whisper import and sentiment analysis on sample text.
3. Additional installation for `gradio` and `plotly`.
4. The main application code:
   - loads models
   - defines `create_sentiment_plot(...)`
   - defines `export_transcript(...)`
   - defines `process_audio(audio_path, task_type)`
   - builds the Gradio Blocks UI
   - launches the app with `debug=True` and `share=True`
5. Downloads a sample WAV file for testing.
6. Uses `inspect.getsource(...)` to assemble and save a standalone `app.py`.
7. Initializes Git and pushes the generated app to GitHub.
8. Performs a manual test of `process_audio(...)`.

## Main Processing Logic

The core function in the notebook is `process_audio(audio_path, task_type)`.

Its behavior is:

- Returns a fallback response when no audio is provided.
- Chooses Whisper task mode:
  - `transcribe` for original-language transcription
  - `translate` for English translation
- Extracts transcript text from Whisper output.
- Estimates audio duration from the last segment end time.
- Computes:
  - transcript word count
  - speaking speed in WPM
- Runs sentiment analysis on the first 512 characters of the transcript.
- Builds a Plotly gauge visualization for the sentiment score.
- Attempts summarization when:
  - the summarizer loaded successfully
  - the transcript has more than 20 words
- Writes the transcript to `transcript_export.txt`.

The function returns:

- transcript text
- sentiment plot
- highlighted sentiment label data
- word-count text
- WPM text
- summary text
- downloadable transcript file path

## Gradio Interface

The notebook defines a mobile-friendly Gradio interface with:

- an audio upload/input component using `type="filepath"`
- a task selector:
  - `Transcribe (Original)`
  - `Translate to English`
- a run-analysis button
- a transcript download file output
- word-count and speed stats
- a transcript tab
- a summary tab
- an analytics tab for sentiment label and Plotly chart

The interface styling uses custom CSS with a dark background and responsive sizing for smaller screens.

## Dependencies Seen In The Notebook

The notebook explicitly installs or imports:

- `openai-whisper`
- `transformers`
- `torch`
- `nltk`
- `pyarrow`
- `gradio`
- `plotly`
- `whisper`
- `os`
- `inspect`

Depending on the runtime environment, Whisper may also require `ffmpeg` to be available on the system when working with file-based audio input.

## Files Created By The App Or Notebook

When run successfully, the project may create:

- `transcript_export.txt`
- `sample_audio.wav`

The repository now includes [`app.py`](/c:/Users/rajpr/OneDrive/Desktop/t/app.py) and [`requirements.txt`](/c:/Users/rajpr/OneDrive/Desktop/t/requirements.txt).

## Important Notes

- The notebook contains a hardcoded GitHub personal access token in one cell. This is a serious secret-management problem and that token should be revoked immediately if it is still valid.
- The Git cell also force-pushes to GitHub after reinitializing `.git`, which is risky and should not be reused as-is.
- The notebook still mixes prototyping, deployment, testing, and Git operations in one file. The extracted `app.py` is the cleaner entry point and should be preferred for normal use.

## How To Use The Notebook

1. Open [`Untitled0.ipynb`](/c:/Users/rajpr/OneDrive/Desktop/t/Untitled0.ipynb) in Jupyter or Google Colab.
2. Run the installation cells first.
3. Run the main application cell to load models and start the Gradio UI.
4. Upload or record audio.
5. Choose transcription or translation.
6. Review transcript, summary, sentiment, word count, and WPM.
7. Download the exported transcript if needed.

## Current Status

This repository now has a runnable Python app plus the original notebook prototype. The main remaining cleanup item is removing sensitive or risky content from the notebook, especially the embedded GitHub token and force-push workflow.
