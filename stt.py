"""
stt.py - speech-to-text for the Audio Medical Transcript Analyser.

Prefers faster-whisper (fast on CPU) and falls back to the original
openai-whisper package if faster-whisper is missing or fails to load.
Loaded models are cached so the (slow) model load happens only once per run.

AI disclosure (CS50 final project policy): Claude (Anthropic) helped refactor
this module, fixing the tuple returned by faster-whisper's transcribe() and
adding model caching. The design and original code are my own.
"""

import os

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

_model = None      # cached model object
_backend = None    # "faster" or "openai"


def _load_model():
    """Load (once) and return (backend_name, model)."""
    global _model, _backend
    if _model is not None:
        return _backend, _model

    errors = []
    try:
        from faster_whisper import WhisperModel
        try:
            _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
        except ValueError:
            # Some CPUs do not support int8; float32 always works
            _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="float32")
        _backend = "faster"
        return _backend, _model
    except Exception as e:
        errors.append(f"faster-whisper: {e}")

    try:
        import whisper
        _model = whisper.load_model(WHISPER_MODEL_SIZE)
        _backend = "openai"
        return _backend, _model
    except Exception as e:
        errors.append(f"openai-whisper: {e}")

    raise RuntimeError(
        "Could not load a Whisper model (" + "; ".join(errors) + "). "
        "Install one with `pip install faster-whisper` and make sure ffmpeg is on your PATH."
    )


def transcribe_audio(path, language="en", beam_size=1):
    """
    Transcribe the audio file at `path`.

    Returns (transcript, segments) where segments is a list of
    {"start": float, "end": float, "text": str}.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    backend, model = _load_model()
    segments = []

    if backend == "faster":
        # transcribe() returns a lazy generator of segments plus an info object
        raw_segments, _info = model.transcribe(path, language=language, beam_size=beam_size)
        for seg in raw_segments:
            text = (seg.text or "").strip()
            if text:
                segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    else:
        result = model.transcribe(path, language=language)
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                segments.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": text})

    transcript = "\n".join(s["text"] for s in segments)
    return transcript, segments
