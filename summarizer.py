"""
summarizer.py - optional abstractive summary with FLAN-T5.

The model is large (about 3 GB for flan-t5-large), so it is loaded lazily and
only when the user ticks "AI summary" on the upload form. The model name can
be changed with the SUMMARY_MODEL environment variable (flan-t5-base is a
lighter choice for laptops).

AI disclosure (CS50 final project policy): Claude (Anthropic) helped
rewrite this module. The original idea and first version are my own.
"""

import os

MODEL_NAME = os.environ.get("SUMMARY_MODEL", "google/flan-t5-base")

_pipe = None


def generate_clinical_summary(transcript, note):
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("text2text-generation", model=MODEL_NAME)

    symptoms = ", ".join(s["term"] for s in note["symptoms"]) or "none found"
    medications = ", ".join(note["medications"]) or "none mentioned"
    # FLAN-T5 has a 512-token input window, so long transcripts are trimmed
    prompt = (
        "Summarize this doctor-patient conversation as a short clinical note.\n"
        f"Symptoms: {symptoms}. Medications: {medications}.\n"
        f"Conversation: {transcript[:1500]}"
    )
    return _pipe(prompt, max_new_tokens=160)[0]["generated_text"].strip()
