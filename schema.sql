-- Database schema for the Audio Medical Transcript Analyser.
-- Created automatically by app.py on first run (see init_db()).

CREATE TABLE IF NOT EXISTS analyses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    filename      TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    duration      REAL,
    transcript    TEXT NOT NULL,
    note_json     TEXT NOT NULL,   -- structured note from extractor.py
    plan          TEXT,
    ai_summary    TEXT
);

-- One row per suspected diagnosis so we can query across analyses
CREATE TABLE IF NOT EXISTS diagnoses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id   INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    dx            TEXT NOT NULL,
    score         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_diagnoses_analysis ON diagnoses(analysis_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_dx ON diagnoses(dx);
