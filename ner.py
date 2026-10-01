"""
ner.py - optional transformer-based named-entity recognition.

Uses a Hugging Face token-classification model. The model name comes from the
BIOBERT_MODEL environment variable. The plain BioBERT base checkpoint
(dmis-lab/biobert-base-cased-v1.1) has no trained NER head, so its labels come
out as meaningless LABEL_0 / LABEL_1. In that case run_ner() returns None and
the caller falls back to the rule-based extractor in rules.py. Set
BIOBERT_MODEL to a model fine-tuned for clinical NER (for example
d4data/biomedical-ner-all) to use real model predictions.

AI disclosure: Claude (Anthropic) helped add the LABEL_x detection and the
opt-in behaviour.
"""

import os

MODEL_NAME = os.environ.get("BIOBERT_MODEL", "").strip()

_pipe = None


def get_ner_pipeline():
    """Load and cache the NER pipeline. Returns None if no model is configured."""
    global _pipe
    if not MODEL_NAME:
        return None
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline("token-classification", model=MODEL_NAME,
                         aggregation_strategy="simple", device=-1)
    return _pipe


def run_ner(text):
    """
    Return a list of entity dicts ({entity_group, score, word, start, end}),
    or None when no usable model is available.
    """
    pipe = get_ner_pipeline()
    if pipe is None or not text.strip():
        return None
    entities = pipe(text)
    # An untrained classification head only ever produces generic labels
    if entities and all(str(e.get("entity_group", "")).startswith("LABEL_") for e in entities):
        return None
    return entities
