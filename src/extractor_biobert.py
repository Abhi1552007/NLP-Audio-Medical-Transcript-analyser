# src/extractor_biobert.py
import re
from ner_biobert import run_ner, get_ner_pipeline
from clinical_ner import extract_simple_entities, normalize_entity_name
from typing import List, Dict

# List of scan/test keywords to detect
SCAN_KEYWORDS = ["ecg","echocardiogram","troponin","chest x-ray","chest xray","x-ray","xray","ct","ct scan","mri","cbc","bnP","bnf","bn p","ct head","ultrasound"]

def normalize_label(label:str) -> str:
    """Map model-specific entity_group labels to our labels."""
    l = label.lower()
    if l in ("drug","chemical","med","medication","pharm","drug_name"):
        return "MEDICATION"
    if l in ("disease","disease_disorder","diagnosis","diagnoses","problem","condition","symptom"):
        return "DIAGNOSIS"
    if l in ("symptom","sign"):
        return "SYMPTOM"
    return label.upper()

def safe_run_ner(text:str):
    try:
        ents = run_ner(text)
        return ents
    except Exception:
        # fallback to simple rule-based extractor
        return None

def extract_clinical_structured(transcript: str) -> Dict:
    text = transcript or ""
    # Chief complaint heuristics
    cc = None
    m = re.search(r'what brought you in (today\?)?\s*(.+)', text, re.I)
    if m:
        cc = m.group(2).strip().splitlines()[0]
    if not cc:
        # pick first patient utterance / first line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            cc = lines[0]
    # HPI: look for "history" block
    hpi = None
    m2 = re.search(r'(history of present illness|history of present|hpi|history:)\s*[:\-]?\s*(.+?)(?:\n\n|\Z)', text, re.I | re.S)
    if m2:
        hpi = m2.group(2).strip()
    if not hpi:
        # fallback: join first 3 non-empty lines
        parts = [ln.strip() for ln in text.splitlines() if ln.strip()]
        hpi = " ".join(parts[0:3]) if parts else ""

    # Try BioBERT NER
    ner_output = safe_run_ner(text)
    symptoms=[]
    medications=[]
    diagnoses=[]
    scans=[]
    followups=[]

    if ner_output is None:
        # fallback to clinical_ner rule-based
        ents = extract_simple_entities(text)
        for e in ents:
            if e.label == "SYMPTOM":
                norm = normalize_entity_name(e.text, "SYMPTOM")
                symptoms.append({"term": e.text, "confidence": round(e.confidence,3), "icd10": norm.get("code")})
            elif e.label == "MEDICATION":
                norm = normalize_entity_name(e.text, "MEDICATION")
                medications.append({"name": e.text, "score": round(e.confidence,3), "rxnorm": norm.get("code")})
            elif e.label == "DIAGNOSIS":
                diagnoses.append(e.text)
    else:
        # ner_output is list of dicts from HF pipeline
        for ent in ner_output:
            group = ent.get("entity_group") or ent.get("entity") or ent.get("label") or ""
            label = normalize_label(group)
            word = ent.get("word") or ent.get("entity_group") or ent.get("entity")
            score = float(ent.get("score",0.0))
            start = ent.get("start"); end = ent.get("end")
            if label == "MEDICATION":
                medications.append({"name": word, "score": round(score,3), "rxnorm": None})
            elif label == "DIAGNOSIS":
                diagnoses.append(word)
            else:
                # try to detect if the entity is a symptom by simple keywords
                low = word.lower()
                if any(k in low for k in ["pain","cough","fever","dizzy","breath","nausea","vomit","weak","fatigue","shortness"]):
                    norm = normalize_entity_name(word, "SYMPTOM")
                    symptoms.append({"term": word, "confidence": round(score,3), "icd10": norm.get("code")})
                else:
                    # treat unknown as diagnosis if score high
                    if score>0.85:
                        diagnoses.append(word)

    # detect scans/tests via keyword search (also from transcript)
    lowtext = text.lower()
    for kw in SCAN_KEYWORDS:
        if kw.lower() in lowtext and kw not in scans:
            scans.append(kw)

    # follow-up detection: look for "follow up" or "return in"
    follow_patterns = [r'follow up in (\d+)', r'follow up', r'return in (\d+)', r'return to clinic', r'urgent review', r'urgent referral']
    for p in follow_patterns:
        if re.search(p, lowtext):
            followups.append(p)

    # confidence: average of medication scores and symptom confidences if available
    scores=[]
    for s in symptoms:
        if s.get("confidence"): scores.append(s["confidence"])
    for m in medications:
        if m.get("score"): scores.append(m["score"])
    conf = round(sum(scores)/len(scores),3) if scores else 0.5

    out = {
        "chief_complaint": cc,
        "history_of_present_illness": hpi,
        "symptoms": symptoms,
        "medications": medications,
        "diagnoses": list(dict.fromkeys(diagnoses)),
        "recommended_scans": scans,
        "follow_up": followups,
        "confidence_score": conf
    }
    return out
