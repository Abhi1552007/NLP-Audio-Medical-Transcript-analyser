# Audio Medical Transcript Analyser

A Flask web app that turns a recorded doctor–patient consultation into a structured clinical note.

```
audio upload ──► Whisper speech-to-text ──► BioBERT clinical NER ──► structured extraction ──► FLAN-T5 summary
```

## Features

- **Speech-to-text** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`small` model, int8 on CPU, falling back to float32 or the OpenAI `whisper` package if needed)
- **Clinical entity extraction** with BioBERT (`dmis-lab/biobert-base-cased-v1.1`, overridable via the `BIOBERT_MODEL` environment variable), plus rule-based extraction of complaints, medications and findings
- **Clinical summarisation** with `google/flan-t5-large`
- A simple web UI: upload audio, view the transcript, extracted entities and summary

Everything runs locally on CPU, so no patient audio leaves the machine.

## Project structure

| Path | Role |
|---|---|
| `src/app.py` | Flask app and routes (`/`, `/analyze`) |
| `src/audio_stt.py` | Robust Whisper loading and transcription |
| `src/ner_biobert.py`, `src/extractor_biobert.py` | BioBERT NER and structured extraction |
| `src/clinical_ner.py` | Rule/dictionary-based clinical NER |
| `src/summarizer_clinical.py` | FLAN-T5 summariser |
| `scripts/download_summarizer.py` | Pre-downloads the summarisation model |
| `templates/` | Upload and result pages |

## Running it

```bash
pip install flask faster-whisper transformers torch
python scripts/download_summarizer.py   # optional: cache the model up front
python src/app.py --port 5001
```

Then open http://localhost:5001 and upload an audio file. The first run downloads the Whisper, BioBERT and FLAN-T5 models.

> Related: [Med_Transcript_Analyzer](https://github.com/Abhi1552007/Med_Transcript_Analyzer) is the earlier text-only version of this pipeline, built on the MTSamples dataset.

*This is a research prototype and is not intended for clinical use.*
