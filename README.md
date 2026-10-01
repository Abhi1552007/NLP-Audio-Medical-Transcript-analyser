# Audio Medical Transcript Analyser
#### Video Demo: <URL HERE>
#### Description:

Audio Medical Transcript Analyser is a Flask web application that turns a recorded doctor–patient consultation into a structured clinical note. A user uploads an audio file (or pastes a transcript), and the app transcribes the speech with OpenAI's Whisper model. It then pulls out the chief complaint, relevant past history, symptoms (with ICD-10 codes), medications, tests and follow-up instructions, and shows a ranked list of possible diagnoses with a suggested plan. Every analysis is saved in a SQLite database, so earlier consultations can be searched, reopened, printed or deleted from a History page.

I built this because doctors spend a large part of every consultation typing notes instead of looking at the patient. Automatic note-taking ("ambient scribing") is a real and growing field, and I wanted to see how far I could get with free, open models that run entirely on a laptop CPU. Because nothing is sent to an external API, no patient audio ever leaves the machine, and the app deletes each uploaded recording as soon as it has been transcribed.

**Important:** this is a learning project. The diagnosis and treatment suggestions come from a small hand-written rule table. They are not medical advice, and the app must never be used on real patients.

## How to run it

```bash
pip install -r requirements.txt     # also needs ffmpeg installed for audio decoding
python app.py                       # then open http://localhost:5001
```

The first audio upload downloads the Whisper `small` model (about 500 MB). If you just want to try the app quickly, paste the example conversation from `samples/chest_pain.txt` into the text box. That path needs only Flask.

## How it works

1. **Speech to text.** The audio is transcribed with `faster-whisper`, a fast re-implementation of Whisper, using 8-bit weights on the CPU.
2. **Extraction.** The transcript is split into sentences. A dictionary of symptom phrases maps everyday wording ("short of breath", "threw up", "lightheaded") to a canonical symptom and its ICD-10 code. Medications, past conditions, tests and follow-up phrases ("follow up in two days") are found with whole-word regular expressions.
3. **Reasoning.** Each symptom points to a list of possible diagnoses, ordered from most to least likely. A diagnosis scores higher when it appears earlier in a list, and it gains a bonus for every additional symptom that supports it. A history of a heart attack or stent raises the score for acute coronary syndrome. From the top diagnoses the app derives suggested tests, treatment classes, and a plan that flags urgent cases.
4. **Storage.** The transcript, the structured note (stored as JSON) and the plan go into an `analyses` table. Each suspected diagnosis goes into its own row in a `diagnoses` table, which lets the History page search by diagnosis and count the most frequent suspicions with a `GROUP BY` query.
5. **Optional AI summary.** If the user ticks the checkbox, a FLAN-T5 model writes a short free-text summary as well.

## Files

- **`app.py`** is the Flask application. It opens one SQLite connection per request and creates the tables from `schema.sql` on start-up. It defines five routes: the upload form (`/`), the analysis pipeline (`/analyze`), viewing a saved result (`/analysis/<id>`), deleting one, and the searchable `/history` page. It also handles error cases (no input, wrong file type, files over 50 MB, failed transcription, silent recordings) with flashed messages rather than crashing.
- **`stt.py`** loads Whisper once and caches it, because loading takes several seconds. It tries `faster-whisper` first and falls back to the original `openai-whisper` package. It returns the full transcript plus time-stamped segments, which I use to show the recording length.
- **`extractor.py`** converts the raw transcript into the structured note: chief complaint, history, symptoms, medications, conditions, tests mentioned and follow-ups.
- **`rules.py`** holds all the medical knowledge tables (symptom synonyms, ICD-10 codes, symptom→diagnosis, diagnosis→tests and treatments) and the scoring and plan functions. I kept the data and logic here so the rules can be extended without touching the web code.
- **`ner.py`** is an optional transformer-based named-entity recogniser, enabled by setting the `BIOBERT_MODEL` environment variable to a clinical NER model.
- **`summarizer.py`** is the optional FLAN-T5 summary, loaded only when requested.
- **`schema.sql`** defines the two tables and their indexes.
- **`templates/`** contains `layout.html`, the shared navigation, flash messages and disclaimer that every page extends with Jinja. It also has `index.html` (upload form), `result.html` (the clinical note, with score bars and a print-friendly style) and `history.html` (search table and statistics).
- **`static/styles.css`** holds styling, including a `@media print` block so a note prints cleanly. **`static/script.js`** stops empty submissions and shows a spinner, because transcription can take a minute and the page would otherwise look frozen.
- **`samples/chest_pain.txt`** is an example consultation for testing.

## Design decisions

**Rules instead of a "smart" model for diagnoses.** My first version used the general BioBERT checkpoint for entity recognition. I later realised that this checkpoint has no trained NER head, so its labels (`LABEL_0`, `LABEL_1`) were essentially random. I replaced that with transparent dictionary rules that I can read, test and explain. The transformer path is still there, but it is opt-in and is ignored automatically if the model only outputs generic labels. For a medical tool, I think predictable behaviour matters more than looking clever.

**Saving results in SQL.** Originally each result disappeared once the page was closed. Storing analyses in SQLite made the app useful over time. I store the note as JSON because its shape may change as I add fields. Diagnoses get their own table because I wanted to query across consultations.

**Privacy.** Uploaded audio is deleted in a `finally` block straight after transcription, even if transcription fails. Only the text is kept.

**Pasted transcripts.** Supporting pasted text means the app can be demonstrated and tested without downloading any models. It is also useful when a transcript already exists.

**Lazy model loading.** The heavy models are imported only when they are first needed, so the web server starts instantly.

## Limitations and future work

The rule tables cover only about a dozen common symptoms, and the app cannot tell the doctor's voice from the patient's. Next steps would be speaker diarisation, negation handling (so that "no fever" is not counted as fever), user accounts, and exporting notes as PDF.

## Use of AI tools

As CS50 allows for final projects, I used Claude (Anthropic) as a helper. It helped me restructure the code into modules, add the SQLite history feature, and fix bugs. Each file where it contributed says so in a comment at the top.
