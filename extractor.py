"""
extractor.py - turn a raw transcript into a structured clinical note.

Combines rule-based dictionary matching (rules.py) with optional transformer
NER (ner.py), then derives the chief complaint, relevant history, tests that
were mentioned and any follow-up instructions.

AI disclosure (CS50 final project policy): Claude (Anthropic) helped
rewrite this module. The original idea and first version are my own.
"""

import re

from ner import run_ner
from rules import SYMPTOMS, CONDITIONS, extract_simple_entities

TEST_KEYWORDS = ["ECG", "EKG", "echocardiogram", "troponin", "chest x-ray", "x-ray",
                 "CT scan", "MRI", "CBC", "blood test", "ultrasound", "BNP", "D-dimer"]

FOLLOW_UP_PATTERNS = [
    r"follow[ -]?up in [^.?!]+",
    r"come back in [^.?!]+",
    r"return (?:to the clinic )?in [^.?!]+",
    r"see you (?:again )?in [^.?!]+",
]


def split_sentences(text):
    """Split a transcript into trimmed sentences."""
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+|\n+", text) if s.strip()]


def find_chief_complaint(sentences):
    """First sentence that mentions any known symptom phrase, else the first sentence."""
    phrases = [p for ps in SYMPTOMS.values() for p in ps]
    for s in sentences:
        low = s.lower()
        if any(p in low for p in phrases):
            return s
    return sentences[0] if sentences else ""


def find_history(sentences):
    """Sentences that mention a past condition or the word 'history'."""
    hits = [s for s in sentences
            if "history" in s.lower() or any(c in s.lower() for c in CONDITIONS)]
    return " ".join(hits[:4])


def extract_clinical_structured(transcript):
    text = transcript or ""
    sentences = split_sentences(text)

    found = extract_simple_entities(text)
    symptoms = found["symptoms"]
    medications = list(found["medications"])
    conditions = list(found["conditions"])

    # Merge in model entities when a real NER model is configured
    entities = run_ner(text)
    if entities:
        known = {s["term"] for s in symptoms}
        for e in entities:
            group = str(e.get("entity_group", "")).lower()
            word = e.get("word", "").strip()
            score = round(float(e.get("score", 0)), 2)
            if len(word) < 3:
                continue
            if "sign" in group or "symptom" in group:
                if word.lower() not in known:
                    symptoms.append({"term": word.lower(), "confidence": score, "icd10": None})
                    known.add(word.lower())
            elif ("medication" in group or "drug" in group or "chemical" in group) and word.lower() not in medications:
                medications.append(word.lower())
            elif "disease" in group and word.lower() not in conditions:
                conditions.append(word.lower())

    tests = [t for t in TEST_KEYWORDS if re.search(r"\b" + re.escape(t) + r"\b", text, re.I)]

    follow_ups = []
    for pattern in FOLLOW_UP_PATTERNS:
        follow_ups += [m.group(0) for m in re.finditer(pattern, text, re.I)]

    return {
        "chief_complaint": find_chief_complaint(sentences),
        "history": find_history(sentences),
        "symptoms": symptoms,
        "medications": medications,
        "conditions": conditions,
        "tests_mentioned": tests,
        "follow_ups": follow_ups,
        "used_model_ner": bool(entities),
    }
