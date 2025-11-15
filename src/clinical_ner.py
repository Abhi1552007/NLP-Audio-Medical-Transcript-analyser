# src/clinical_ner.py
import re
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Entity:
    text:str
    label:str
    start:int=0
    end:int=0
    confidence:float=0.7
    norm_code:Dict[str,str]=field(default_factory=dict)

SYMPTOMS = ["chest pain","shortness of breath","cough","fever","nausea","vomiting","dizziness","fatigue","palpitations","headache","wheeze"]
MEDICATIONS = ["aspirin","ibuprofen","omeprazole","prednisone","metformin","lisinopril","amoxicillin","insulin","simvastatin"]
DIAGNOSES = ["hypertension","diabetes","pneumonia","heart failure","congestive heart failure","copd"]

def _find(text, words, label):
    res=[]
    low=text.lower()
    for w in words:
        idx=low.find(w)
        if idx!=-1:
            res.append(Entity(text=text[idx:idx+len(w)], label=label, start=idx, end=idx+len(w), confidence=0.75))
    return res

def extract_simple_entities(text:str):
    ents=[]
    ents+=_find(text, SYMPTOMS, "SYMPTOM")
    ents+=_find(text, MEDICATIONS, "MEDICATION")
    ents+=_find(text, DIAGNOSES, "DIAGNOSIS")
    return ents

ICD10={"chest pain":"R07.9","shortness of breath":"R06.02","fever":"R50.9","dizziness":"R42"}
RXNORM={"aspirin":"1191","ibuprofen":"5640","omeprazole":"7632"}

def normalize_entity_name(name:str,label:str):
    key=name.lower().strip()
    if label=="SYMPTOM": return {"code":ICD10.get(key)}
    if label=="MEDICATION": return {"code":RXNORM.get(key)}
    return {}
