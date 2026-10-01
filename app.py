"""
app.py - Flask web application for the Audio Medical Transcript Analyser.

Routes
  GET  /                      upload form (audio file or pasted transcript)
  POST /analyze               transcribe + analyse, save to SQLite, redirect to result
  GET  /analysis/<id>         show one saved analysis
  POST /analysis/<id>/delete  delete a saved analysis
  GET  /history               list / search past analyses + most common diagnoses

AI disclosure (CS50 final project policy): Claude (Anthropic) helped me
restructure this file, add the SQLite history feature and the pasted-transcript
option, and fix bugs in field names between the extractor and the templates.
"""

import json
import os
import sqlite3
import uuid

from flask import Flask, flash, g, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from extractor import extract_clinical_structured
from rules import generate_plan, infer_differential, suggest_medications, suggest_scans

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.environ.get("DATABASE", os.path.join(BASE_DIR, "analyses.db"))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------- database helpers ----------

def get_db():
    """Open one SQLite connection per request (stored on flask.g)."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with open(os.path.join(BASE_DIR, "schema.sql")) as f:
        conn = sqlite3.connect(DATABASE)
        conn.executescript(f.read())
        conn.close()


init_db()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------- analysis pipeline ----------

def analyse_transcript(transcript, want_summary):
    """Run extraction + heuristics on a transcript. Returns a dict of results."""
    note = extract_clinical_structured(transcript)
    terms = [s["term"] for s in note["symptoms"]]
    diagnoses = infer_differential(terms, note["history"])

    ai_summary = None
    if want_summary:
        try:
            from summarizer import generate_clinical_summary
            ai_summary = generate_clinical_summary(transcript, note)
        except Exception as e:   # model missing / out of memory: keep going without it
            ai_summary = f"(AI summary unavailable: {e})"

    return {"note": note, "diagnoses": diagnoses,
            "plan": generate_plan(diagnoses), "ai_summary": ai_summary}


# ---------- routes ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    title = request.form.get("title", "").strip() or "Untitled consultation"
    pasted = request.form.get("transcript", "").strip()
    file = request.files.get("audio")
    want_summary = request.form.get("ai_summary") == "on"
    duration = None

    if file and file.filename:
        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload mp3, wav, m4a, ogg or flac.", "danger")
            return redirect(url_for("index"))
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        file.save(path)
        try:
            from stt import transcribe_audio
            transcript, segments = transcribe_audio(path)
            duration = segments[-1]["end"] if segments else 0.0
        except Exception as e:
            flash(f"Transcription failed: {e}", "danger")
            return redirect(url_for("index"))
        finally:
            # Privacy: never keep patient audio on disk after transcription
            os.remove(path)
    elif pasted:
        transcript, filename = pasted, "(pasted text)"
    else:
        flash("Please choose an audio file or paste a transcript.", "warning")
        return redirect(url_for("index"))

    if not transcript:
        flash("No speech was detected in that recording.", "warning")
        return redirect(url_for("index"))

    result = analyse_transcript(transcript, want_summary)

    db = get_db()
    cur = db.execute(
        "INSERT INTO analyses (title, filename, duration, transcript, note_json, plan, ai_summary) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (title, filename, duration, transcript, json.dumps(result["note"]),
         result["plan"], result["ai_summary"]))
    analysis_id = cur.lastrowid
    db.executemany("INSERT INTO diagnoses (analysis_id, dx, score) VALUES (?, ?, ?)",
                   [(analysis_id, d["dx"], d["score"]) for d in result["diagnoses"]])
    db.commit()
    return redirect(url_for("show_analysis", analysis_id=analysis_id))


@app.route("/analysis/<int:analysis_id>")
def show_analysis(analysis_id):
    db = get_db()
    row = db.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if row is None:
        flash("That analysis does not exist.", "warning")
        return redirect(url_for("history"))
    diagnoses = [dict(r) for r in db.execute(
        "SELECT dx, score FROM diagnoses WHERE analysis_id = ? ORDER BY score DESC", (analysis_id,))]
    note = json.loads(row["note_json"])
    return render_template("result.html", a=row, note=note, diagnoses=diagnoses,
                           meds=suggest_medications(diagnoses), scans=suggest_scans(diagnoses))


@app.route("/analysis/<int:analysis_id>/delete", methods=["POST"])
def delete_analysis(analysis_id):
    db = get_db()
    db.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    db.commit()
    flash("Analysis deleted.", "success")
    return redirect(url_for("history"))


@app.route("/history")
def history():
    q = request.args.get("q", "").strip()
    db = get_db()
    if q:
        like = f"%{q}%"
        rows = db.execute(
            "SELECT DISTINCT a.id, a.title, a.created_at, a.filename FROM analyses a "
            "LEFT JOIN diagnoses d ON d.analysis_id = a.id "
            "WHERE a.title LIKE ? OR a.transcript LIKE ? OR d.dx LIKE ? "
            "ORDER BY a.created_at DESC", (like, like, like)).fetchall()
    else:
        rows = db.execute("SELECT id, title, created_at, filename FROM analyses "
                          "ORDER BY created_at DESC").fetchall()

    # Top suspected diagnosis per analysis, for the table
    top = {r["analysis_id"]: r["dx"] for r in db.execute(
        "SELECT analysis_id, dx FROM diagnoses d WHERE score = "
        "(SELECT MAX(score) FROM diagnoses WHERE analysis_id = d.analysis_id)")}

    common = db.execute(
        "SELECT dx, COUNT(*) AS n FROM diagnoses GROUP BY dx ORDER BY n DESC, dx LIMIT 5").fetchall()
    return render_template("history.html", rows=rows, top=top, common=common, q=q)


@app.errorhandler(413)
def too_large(_e):
    flash("That file is larger than the 50 MB limit.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
