"""
rules.py - dictionary / rule-based clinical knowledge.

Two jobs:
 1. extract_simple_entities(): find symptoms, medications and known
    conditions in a transcript by whole-word dictionary matching.
 2. infer_differential(), suggest_medications(), suggest_scans(),
    generate_plan(): tiny hand-written heuristics that turn symptoms into
    a ranked list of *possible* diagnoses and follow-up ideas.

These heuristics are for demonstration only. They are NOT clinical decision
support and must never be used to treat a real patient.

AI disclosure (CS50 final project policy): Claude (Anthropic) helped
rewrite this module. The original idea and first version are my own.
"""

import re

# Each canonical symptom maps to the phrases that patients actually say
SYMPTOMS = {
    "chest pain": ["chest pain", "pain in my chest", "chest hurts", "chest tightness"],
    "shortness of breath": ["shortness of breath", "short of breath", "breathless", "can't breathe", "hard to breathe"],
    "cough": ["cough", "coughing"],
    "fever": ["fever", "temperature", "feverish"],
    "nausea": ["nausea", "nauseous", "feel sick"],
    "vomiting": ["vomiting", "vomit", "throwing up", "threw up"],
    "diarrhea": ["diarrhea", "diarrhoea", "loose stools"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "light-headed"],
    "fatigue": ["fatigue", "tired", "exhausted"],
    "palpitations": ["palpitations", "heart racing", "heart pounding"],
    "headache": ["headache", "head hurts", "migraine"],
    "wheeze": ["wheeze", "wheezing"],
    "sore throat": ["sore throat", "throat hurts"],
}

ICD10 = {
    "chest pain": "R07.9", "shortness of breath": "R06.02", "cough": "R05.9",
    "fever": "R50.9", "nausea": "R11.0", "vomiting": "R11.10", "diarrhea": "R19.7",
    "dizziness": "R42", "fatigue": "R53.83", "palpitations": "R00.2",
    "headache": "R51.9", "wheeze": "R06.2", "sore throat": "J02.9",
}

MEDICATIONS = ["aspirin", "ibuprofen", "paracetamol", "acetaminophen", "omeprazole",
               "prednisone", "metformin", "lisinopril", "amlodipine", "amoxicillin",
               "insulin", "simvastatin", "atorvastatin", "salbutamol", "albuterol"]

CONDITIONS = ["hypertension", "high blood pressure", "diabetes", "asthma", "copd",
              "heart failure", "heart attack", "stent", "stroke", "pneumonia"]

SYMPTOM_TO_DX = {
    "chest pain": ["acute coronary syndrome", "pericarditis", "musculoskeletal chest pain", "pulmonary embolism"],
    "shortness of breath": ["congestive heart failure", "pneumonia", "asthma", "copd", "pulmonary embolism"],
    "fever": ["viral infection", "bacterial infection", "influenza"],
    "cough": ["bronchitis", "pneumonia", "upper respiratory tract infection"],
    "vomiting": ["gastroenteritis", "food poisoning"],
    "nausea": ["gastroenteritis", "medication side effect"],
    "diarrhea": ["gastroenteritis", "food poisoning"],
    "headache": ["migraine", "tension headache"],
    "palpitations": ["arrhythmia", "anxiety", "hyperthyroidism"],
    "dizziness": ["orthostatic hypotension", "arrhythmia", "vestibular disorder"],
    "wheeze": ["asthma", "copd", "bronchospasm"],
    "sore throat": ["viral pharyngitis", "streptococcal pharyngitis"],
}

DX_TO_MEDS = {
    "acute coronary syndrome": ["aspirin (antiplatelet) with immediate emergency evaluation"],
    "congestive heart failure": ["diuretic for symptom relief", "ACE inhibitor / ARB long term"],
    "pneumonia": ["empiric antibiotics guided by severity"],
    "asthma": ["short-acting bronchodilator (e.g. salbutamol)"],
    "gastroenteritis": ["oral rehydration; antiemetic if needed"],
    "migraine": ["NSAID or triptan depending on history"],
    "arrhythmia": ["ECG and cardiology referral before treatment"],
    "pulmonary embolism": ["urgent evaluation before anticoagulation"],
    "bronchitis": ["supportive care; antibiotics usually not needed"],
}

DX_TO_SCANS = {
    "acute coronary syndrome": ["ECG", "Troponin", "Chest X-ray"],
    "congestive heart failure": ["ECG", "Echocardiogram", "BNP", "Chest X-ray"],
    "pulmonary embolism": ["D-dimer", "CT pulmonary angiogram"],
    "pneumonia": ["Chest X-ray", "CBC"],
    "arrhythmia": ["ECG", "Holter monitor"],
    "streptococcal pharyngitis": ["Rapid strep test"],
}

URGENT = {"acute coronary syndrome", "pulmonary embolism"}


def _contains(text, phrase):
    """Whole-word, case-insensitive phrase search."""
    return re.search(r"\b" + re.escape(phrase) + r"\b", text, re.I) is not None


def extract_simple_entities(text):
    """Return {"symptoms": [...], "medications": [...], "conditions": [...]}."""
    symptoms = []
    for canonical, phrases in SYMPTOMS.items():
        if any(_contains(text, p) for p in phrases):
            symptoms.append({"term": canonical, "confidence": None, "icd10": ICD10.get(canonical)})
    medications = [m for m in MEDICATIONS if _contains(text, m)]
    conditions = [c for c in CONDITIONS if _contains(text, c)]
    return {"symptoms": symptoms, "medications": medications, "conditions": conditions}


def infer_differential(symptom_terms, history_text=""):
    """
    Rank possible diagnoses. A diagnosis scores higher when it appears early in
    a symptom's list and gets a bonus for every additional symptom supporting it.
    Returns [{"dx": str, "score": float}, ...] sorted best first.
    """
    scores = {}
    support = {}
    for term in symptom_terms:
        for i, dx in enumerate(SYMPTOM_TO_DX.get(term, [])):
            scores[dx] = max(scores.get(dx, 0.0), 0.8 - i * 0.15)
            support[dx] = support.get(dx, 0) + 1
    for dx, n in support.items():
        scores[dx] = min(0.99, scores[dx] + 0.1 * (n - 1))

    history = (history_text or "").lower()
    if any(w in history for w in ("heart attack", "myocardial infarction", "stent")):
        scores["acute coronary syndrome"] = max(scores.get("acute coronary syndrome", 0), 0.85)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"dx": dx, "score": round(s, 2)} for dx, s in ranked]


def _collect(diagnoses, table):
    out = []
    for d in diagnoses:
        for item in table.get(d["dx"], []):
            if item not in out:
                out.append(item)
    return out


def suggest_medications(diagnoses):
    return _collect(diagnoses[:3], DX_TO_MEDS)


def suggest_scans(diagnoses):
    return _collect(diagnoses[:3], DX_TO_SCANS)


def generate_plan(diagnoses):
    """One short paragraph: urgency, tests, treatment classes, red flags."""
    if not diagnoses:
        return ("No clear pattern found in the conversation. Recommend vitals, a focused "
                "examination and review if symptoms persist.")
    parts = []
    if any(d["dx"] in URGENT for d in diagnoses[:2]):
        parts.append("Urgent emergency evaluation is recommended to rule out life-threatening causes.")
    else:
        parts.append("Arrange focused outpatient investigations and treat symptoms meanwhile.")
    scans = suggest_scans(diagnoses)
    if scans:
        parts.append("Suggested tests: " + ", ".join(scans) + ".")
    meds = suggest_medications(diagnoses)
    if meds:
        parts.append("Treatment classes to consider: " + "; ".join(meds) + ".")
    parts.append("Red flags: worsening breathlessness, fainting, very high fever or severe chest pain need immediate care.")
    return " ".join(parts)
