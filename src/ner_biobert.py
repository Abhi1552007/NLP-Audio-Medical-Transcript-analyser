# src/ner_biobert.py
from transformers import pipeline
import os
from pathlib import Path

# Default model: Biobert base (you may predownload into models/biobert)
DEFAULT_MODEL = os.environ.get("BIOBERT_MODEL", "dmis-lab/biobert-base-cased-v1.1")

_NER_PIPE = None
def get_ner_pipeline():
    global _NER_PIPE
    if _NER_PIPE: return _NER_PIPE
    local = Path("models/biobert")
    model_ref = str(local) if local.exists() else DEFAULT_MODEL
    # Pipeline for token-classification; aggregation="simple" gives merged entities
    _NER_PIPE = pipeline("token-classification", model=model_ref, tokenizer=model_ref, aggregation_strategy="simple", device=-1)
    return _NER_PIPE

def run_ner(text: str):
    """
    Returns list of entities: {'entity_group': 'DISEASE'/'CHEMICAL'/'DRUG'/..., 'score':float, 'word':str, 'start':int,'end':int}
    Mapping of entity_group depends on model. We'll normalize in extractor.
    """
    pipe = get_ner_pipeline()
    return pipe(text)
