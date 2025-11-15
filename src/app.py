import os
import uuid
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from typing import Tuple, List, Dict, Any
from pathlib import Path

# faster-whisper
from faster_whisper import WhisperModel

# your extractor (unchanged)
from extractor_biobert import extract_clinical_structured

# ---- config ----
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a"}
WHISPER_MODEL_SIZE = "small"   # small = good tradeoff on CPU
DEFAULT_BEAM = 1               # smaller -> faster; increase for better quality
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# ---- cached whisper model handle ----
_WHISPER = None

def load_whisper():
    """
    Load and cache faster-whisper model. Tries int8 then float32.
    """
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER
    try:
        _WHISPER = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    except Exception:
        _WHISPER = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="float32")
    return _WHISPER

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(path: str, language: str = "en", beam_size:int = DEFAULT_BEAM) -> Tuple[str, List[Dict[str,Any]]]:
    """
    Fast transcription using faster-whisper.
    Returns full transcript and list of segments.
    """
    model = load_whisper()
    segments_out = []
    parts = []

    # correct unpacking: segments generator + info
    segments, info = model.transcribe(str(path), language=language, beam_size=beam_size)
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        parts.append(text)
        segments_out.append({
            "start": float(getattr(seg, "start", 0.0)),
            "end": float(getattr(seg, "end", 0.0)),
            "text": text
        })
    full_transcript = "\n".join(parts).strip()
    return full_transcript, segments_out

# --- small medical knowledge heuristics (rule-based)
# These are simple heuristics intended to give useful suggested diagnoses,
# medication classes, scans and a plan. They are NOT clinical decision support.

_SYMPTOM_TO_DX = {
    # symptom keywords -> probable diagnoses (ordered)
    "chest pain": ["acute coronary syndrome", "pericarditis", "musculoskeletal chest pain", "pulmonary embolism"],
    "shortness of breath": ["congestive heart failure", "pneumonia", "asthma", "copd", "pulmonary embolism"],
    "fever": ["viral infection", "bacterial infection", "influenza"],
    "cough": ["bronchitis", "pneumonia", "upper respiratory tract infection"],
    "vomit": ["gastroenteritis", "food poisoning"],
    "nausea": ["gastroenteritis", "medication side effect"],
    "diarrhea": ["gastroenteritis", "food poisoning"],
    "headache": ["migraine", "tension headache", "intracranial pathology (rare)"],
    "palpitations": ["arrhythmia", "anxiety", "hyperthyroidism"],
    "dizziness": ["orthostatic hypotension", "arrhythmia", "vestibular disorder"],
    "breathless": ["congestive heart failure", "pneumonia", "asthma"],
    "wheeze": ["asthma", "copd", "bronchospasm"]
}

_DX_TO_MEDS = {
    # diagnosis (keywords) -> typical medication classes (suggestions, NOT prescriptions)
    "acute coronary syndrome": ["aspirin (antiplatelet) — suggest immediate ED evaluation"],
    "congestive heart failure": ["diuretics (e.g., furosemide) — symptomatic relief", "ACE inhibitor / ARB — long term"],
    "pneumonia": ["empiric antibiotics (guided by severity)"],
    "asthma": ["short-acting bronchodilator (e.g., salbutamol)"],
    "gastroenteritis": ["oral rehydration, antiemetic if needed"],
    "migraine": ["analgesic (NSAID) or triptan depending on history"],
    "arrhythmia": ["refer for ECG and cardiology evaluation"],
    "pulmonary embolism": ["urgent evaluation (D-dimer, CT pulmonary angiogram)"],
    "bronchitis": ["supportive care; antibiotics not routinely unless bacterial"]
}

_DX_TO_SCANS = {
    "acute coronary syndrome": ["ECG", "Troponin", "Chest X-ray"],
    "congestive heart failure": ["ECG", "Echocardiogram", "BNP", "Chest X-ray"],
    "pulmonary embolism": ["CT pulmonary angiogram", "D-dimer"],
    "pneumonia": ["Chest X-ray"],
    "stroke": ["CT head", "MRI brain"],
    "gastroenteritis": ["usually none; consider stool studies if prolonged"],
    "migraine": ["none unless red flags — consider neuroimaging if focal signs"]
}

def infer_differential(symptoms: List[Dict[str,Any]], history_text: str) -> List[Dict[str,Any]]:
    """
    Take normalized symptoms list (list of dicts with 'term') and return
    ranked differential diagnoses with simple scores (0-1).
    """
    # accumulate candidates
    candidates = {}
    symptom_terms = [s.get("term","").lower() for s in symptoms if s.get("term")]
    for term in symptom_terms:
        for key, dlist in _SYMPTOM_TO_DX.items():
            if key in term or term in key or key in term:
                for i,dx in enumerate(dlist):
                    # score: higher for earlier entries; increment for multiple matching symptoms
                    score = 0.8 - i*0.15
                    candidates[dx] = max(candidates.get(dx, 0.0), score)
    # If history contains high-risk words, boost certain dx
    hist = (history_text or "").lower()
    if "heart attack" in hist or "myocardial infarction" in hist or "stent" in hist:
        candidates["congestive heart failure"] = max(candidates.get("congestive heart failure",0), 0.75)
        candidates["acute coronary syndrome"] = max(candidates.get("acute coronary syndrome",0), 0.85)
    # normalize to sorted list
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [{"dx": dx, "score": round(float(score), 3)} for dx, score in ranked]

def suggest_medications(diagnoses: List[Dict[str,Any]]) -> List[str]:
    meds = []
    for d in diagnoses:
        name = d.get("dx","").lower()
        for k, v in _DX_TO_MEDS.items():
            if k in name:
                for med in v:
                    if med not in meds:
                        meds.append(med)
    return meds

def suggest_scans(diagnoses: List[Dict[str,Any]]) -> List[str]:
    scans = []
    for d in diagnoses:
        name = d.get("dx","").lower()
        for k, v in _DX_TO_SCANS.items():
            if k in name:
                for t in v:
                    if t not in scans:
                        scans.append(t)
    return scans

def generate_plan(diagnoses: List[Dict[str,Any]], symptoms: List[Dict[str,Any]]) -> str:
    """
    Short actionable plan text: urgent vs outpatient, suggested tests, red flags.
    """
    if not diagnoses:
        return "No clear likely diagnosis from the notes — recommend basic vitals, focused exam, and conservative management or further testing if symptoms persist."

    top = diagnoses[0]["dx"].lower()
    plan_parts = []

    # urgent flags
    urgent = ["acute coronary syndrome", "pulmonary embolism", "stroke"]
    if any(u in d["dx"].lower() for d in diagnoses[:2] for u in urgent):
        plan_parts.append("Urgent evaluation in Emergency Department is recommended for potential life-threatening causes.")
    else:
        plan_parts.append("Arrange focused outpatient investigations and treat symptomatically while pending results.")

    # suggested scans/tests
    scans = suggest_scans(diagnoses)
    if scans:
        plan_parts.append("Suggested tests: " + ", ".join(scans) + ".")

    # meds (non-prescriptive classes)
    meds = suggest_medications(diagnoses)
    if meds:
        plan_parts.append("Consider (non-prescriptive) treatments: " + "; ".join(meds) + ".")

    # red flags
    plan_parts.append("Watch for red flags: worsening breathlessness, syncope, very high fever, severe chest pain — seek immediate care if these occur.")
    return " ".join(plan_parts)

# ---------------------------------------------------------
# UI helpers: normalization and concise summary builder
# ---------------------------------------------------------

def normalize_structured(structured: Any) -> Dict[str, Any]:
    """
    Convert extractor output into consistent dict:
    {chief_complaint, symptoms(list of {term,confidence,icd10}), medical_history, diagnosis, scans(list), followups}
    """
    out = {
        "chief_complaint": "",
        "symptoms": [],
        "medical_history": "",
        "diagnosis": "",
        "scans": [],
        "followups": ""
    }
    if not structured:
        return out

    # helper to pull attribute/key
    def _get(k):
        try:
            if isinstance(structured, dict):
                return structured.get(k, None)
            return getattr(structured, k, None)
        except Exception:
            return None

    cc = _get("chief_complaint") or _get("chiefComplaint") or _get("complaint") or ""
    mh = _get("medical_history") or _get("medicalHistory") or _get("history") or ""
    diag = _get("diagnosis") or _get("diagnoses") or _get("possible_diagnoses") or ""
    scans = _get("scans") or _get("recommended_scans") or _get("tests") or []
    followups = _get("followups") or _get("follow_up") or _get("plan") or ""

    raw_sym = _get("symptoms") or _get("symptoms_identified") or _get("symptoms_list") or []

    normalized_symptoms = []
    if isinstance(raw_sym, str):
        for s in [x.strip() for x in raw_sym.split(",") if x.strip()]:
            normalized_symptoms.append({"term": s, "confidence": None, "icd10": None})
    elif isinstance(raw_sym, list):
        for item in raw_sym:
            if isinstance(item, dict):
                term = item.get("term") or item.get("name") or item.get("label") or ""
                conf = item.get("confidence") or item.get("score") or item.get("prob") or None
                icd = item.get("icd10") or (item.get("norm_code") and item.get("norm_code").get("code")) or None
                normalized_symptoms.append({"term": term, "confidence": conf, "icd10": icd})
            else:
                normalized_symptoms.append({"term": str(item), "confidence": None, "icd10": None})
    else:
        txt = str(raw_sym)
        if txt:
            normalized_symptoms.append({"term": txt, "confidence": None, "icd10": None})

    scans_list = []
    if isinstance(scans, str):
        scans_list = [s.strip() for s in scans.split(",") if s.strip()]
    elif isinstance(scans, list):
        scans_list = [str(s) for s in scans if s]
    else:
        if scans:
            scans_list = [str(scans)]

    out["chief_complaint"] = cc
    out["medical_history"] = mh
    out["diagnosis"] = diag
    out["scans"] = scans_list
    out["followups"] = followups
    out["symptoms"] = normalized_symptoms
    return out

def build_concise_summary(out: Dict[str, Any], top_dxs: List[Dict[str,Any]]) -> str:
    """
    Short summary paragraph (1-2 sentences) that is clinician-friendly.
    """
    parts = []
    if out.get("chief_complaint"):
        parts.append(f"Chief complaint: {out['chief_complaint']}.")
    if out.get("symptoms"):
        terms = ", ".join([s["term"] for s in out["symptoms"]][:6])
        parts.append(f"Symptoms: {terms}.")
    if top_dxs:
        parts.append("Top suspected: " + ", ".join([d["dx"] for d in top_dxs[:3]]) + ".")
    if out.get("scans"):
        parts.append("Consider: " + ", ".join(out["scans"][:4]) + ".")
    return " ".join(parts) or "No summary generated."

# ---- routes ----

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return "No audio uploaded", 400
    file = request.files["audio"]
    if file.filename == "" or not allowed_file(file.filename):
        return "Invalid audio file", 400

    fname = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    save_path = os.path.join(UPLOAD_FOLDER, fname)
    file.save(save_path)

    # Fast transcription (beam_size small for speed)
    transcript, segments = transcribe_audio(save_path, beam_size=DEFAULT_BEAM)

    # run existing extractor (unchanged)
    raw_struct = extract_clinical_structured(transcript)

    # normalize output
    out = normalize_structured(raw_struct)

    # infer differential diagnoses
    top_dxs = infer_differential(out["symptoms"], out["medical_history"])

    # suggest meds and scans and plan
    meds = suggest_medications(top_dxs)
    scans = suggest_scans(top_dxs)
    plan = generate_plan(top_dxs, out["symptoms"])

    # build concise summary to show top
    summary_text = build_concise_summary(out, top_dxs)

    # pass everything to template
    return render_template("result.html",
                           transcript=transcript,
                           segments=segments,
                           out=out,
                           top_dxs=top_dxs,
                           suggested_meds=meds,
                           suggested_scans=scans,
                           plan=plan,
                           summary=summary_text,
                           audio_name=fname)

# ---- main ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    print(f"Starting server on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
