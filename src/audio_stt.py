"""
src/audio_stt.py

Robust transcription wrapper that:
 - prefers faster-whisper if available,
 - falls back to openai/whisper package if faster-whisper constructor is incompatible,
 - gives clear, actionable error messages if neither works.

Function:
    transcribe_audio(path: str, model_size="small", language="en", device="cpu") -> (transcript, segments)

Segments is a list of dicts: {"start": float, "end": float, "text": str}
"""

from pathlib import Path
from typing import Tuple, List, Dict
import os
import inspect

# Try to import faster-whisper first (fast, recommended)
_fast_whisper_available = False
_whisper_package_available = False
try:
    # This import will succeed for normal faster-whisper installs
    from faster_whisper import WhisperModel as _FW_WhisperModel  # type: ignore
    _fast_whisper_available = True
except Exception:
    _FW_WhisperModel = None

# fallback: the "whisper" package (OpenAI) if installed
try:
    import whisper as _openai_whisper  # type: ignore
    _whisper_package_available = True
except Exception:
    _openai_whisper = None

def _call_faster_whisper(model_size: str, audio_path: str, language: str, device: str, cache_dir: str, beam_size: int):
    """
    Try to create a faster-whisper model in a flexible way and transcribe.
    This tries a few constructor signatures because versions vary.
    """
    if _FW_WhisperModel is None:
        raise RuntimeError("faster-whisper not available")

    # default cache dir
    os.makedirs(cache_dir, exist_ok=True)

    # Try the most common constructor first
    # Most common: WhisperModel(model_size, device=device, compute_type='int8', cache_dir=cache_dir)
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

    # Try a few constructor signatures using introspection
    constructor = _FW_WhisperModel
    sig = None
    try:
        sig = inspect.signature(constructor)
    except Exception:
        sig = None

    instantiated = None
    last_exc = None

    # Helper attempts with different arg combinations
    attempts = [
        {"args": (model_size,), "kwargs": {"device": device, "compute_type": compute_type, "cache_dir": cache_dir}},
        {"args": (model_size,), "kwargs": {"device": device, "compute_type": compute_type}},
        {"args": (), "kwargs": {"model_path": str(model_size), "device": device, "compute_type": compute_type}},
        {"args": (), "kwargs": {"model_path": str(cache_dir), "device": device, "compute_type": compute_type}},
        {"args": (str(model_size),), "kwargs": {"device": device}},
    ]

    for attempt in attempts:
        try:
            instantiated = constructor(*attempt["args"], **attempt["kwargs"])
            break
        except TypeError as e:
            last_exc = e
        except Exception as e:
            last_exc = e

    if instantiated is None:
        # raise a detailed error that will be caught by caller
        raise RuntimeError(
            "Failed to instantiate faster-whisper WhisperModel with common signatures. "
            f"Last error: {last_exc}. You likely have an incompatible faster-whisper / ctranslate2 version. "
            "Try upgrading faster-whisper: `pip install --upgrade faster-whisper ctranslate2` (or see README)."
        )

    # transcribe; the object returned by faster-whisper iterates segments
    segments = []
    parts: List[str] = []
    try:
        # newer faster-whisper may require different call style (transcribe returns generator)
        for segment in instantiated.transcribe(str(audio_path), language=language, beam_size=beam_size):
            text = getattr(segment, "text", str(segment)).strip()
            start = float(getattr(segment, "start", 0.0))
            end = float(getattr(segment, "end", 0.0))
            segments.append({"start": start, "end": end, "text": text})
            parts.append(text)
    except TypeError as e:
        # some builds return tuples/dicts — handle gracefully
        raise RuntimeError(f"faster-whisper transcribe call failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Transcription using faster-whisper failed: {e}")

    transcript = "\n".join(parts).strip()
    return transcript, segments

def _call_openai_whisper(model_size: str, audio_path: str, language: str):
    """
    Use the original 'whisper' package (openai/whisper). This will download models to ~/.cache/whisper by default.
    """
    if _openai_whisper is None:
        raise RuntimeError("openai whisper package not available")
    try:
        model = _openai_whisper.load_model(model_size)
    except Exception as e:
        raise RuntimeError(f"whisper.load_model failed: {e}")
    try:
        result = model.transcribe(str(audio_path), language=language)
        text = result.get("text", "").strip()
        # build segments if available
        segments = []
        segments_raw = result.get("segments") or []
        for s in segments_raw:
            segments.append({"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": s.get("text","").strip()})
        return text, segments
    except Exception as e:
        raise RuntimeError(f"whisper.transcribe failed: {e}")

def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    language: str = "en",
    beam_size: int = 5,
    cache_dir: str = None,
    device: str = "cpu",
) -> Tuple[str, List[Dict]]:
    """
    Robust transcription entrypoint.

    - audio_path: path to file
    - model_size: e.g. "tiny","base","small","medium","large" (or local model path)
    - device: "cpu" or "cuda"
    - cache_dir: where to cache faster-whisper models (default: models/whisper)
    """
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if cache_dir is None:
        cache_dir = os.environ.get("WHISPER_CACHE", "models/whisper")

    # 1) Try faster-whisper first (recommended)
    if _fast_whisper_available:
        try:
            return _call_faster_whisper(model_size=model_size, audio_path=audio_path, language=language, device=device, cache_dir=cache_dir, beam_size=beam_size)
        except Exception as e:
            # don't fail immediately — try openai whisper next and return helpful message if both fail
            faster_err = e
        else:
            faster_err = None
    else:
        faster_err = None

    # 2) Try openai/whisper package as fallback
    if _whisper_package_available:
        try:
            return _call_openai_whisper(model_size=model_size, audio_path=audio_path, language=language)
        except Exception as e:
            whisper_err = e
        else:
            whisper_err = None
    else:
        whisper_err = None

    # 3) If we reach here, both failed
    msg = "Failed to transcribe audio. Attempted faster-whisper and openai/whisper.\n"
    if faster_err:
        msg += f"faster-whisper error: {faster_err}\n"
    if whisper_err:
        msg += f"openai whisper error: {whisper_err}\n"

    msg += (
        "Suggested fixes:\n"
        " 1) Ensure faster-whisper is installed and updated: pip install --upgrade faster-whisper ctranslate2\n"
        " 2) Or install OpenAI whisper: pip install -U openai-whisper\n"
        " 3) Ensure ffmpeg is installed and on PATH (brew install ffmpeg on macOS)\n"
        "After installing, re-run the app.\n"
    )
    raise RuntimeError(msg)
