from transformers import pipeline

# Load once
_SUMMARIZER = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=512,
    min_length=50
)

def generate_clinical_summary(transcript, extracted):
    prompt = f"""
You are a clinical expert. Based on the transcript and extracted items,
produce a concise medical summary.

Transcript:
{transcript}

Extracted symptoms: {extracted['symptoms']}
Extracted medications: {extracted['medications']}
Vitals found: {extracted['vitals']}

Create sections:

1. Chief complaint (1 sentence)
2. HPI (3–5 sentences)
3. Symptoms (bullet points)
4. Possible diagnoses (5 ranked)
5. Recommended tests / scans (bullet points)
6. Medications (only if mentioned)
7. Plan & follow-up (clinical, concise)

Output clearly and medically accurate.

"""

    result = _SUMMARIZER(prompt)[0]["generated_text"]
    return result
