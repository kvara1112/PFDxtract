import streamlit as st
import pandas as pd
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoTokenizer, AutoModel
import joblib
from huggingface_hub import hf_hub_download

nltk.download("punkt")

st.set_page_config(page_title="PubMedBERT Theme Annotation", layout="wide")

# -------------------------------
# Initialize model and tokenizer
# -------------------------------

MODEL_NAME = "kvara03/pubmedbert_theme_classifier_iteration4"

# Load tokenizer + model directly from HF
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
lemmatizer = WordNetLemmatizer()

# Load classification pipeline
clf = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME)

# Download label encoder from HF and load it
label_path = hf_hub_download(
    repo_id=MODEL_NAME,
    filename="label_encoder.pkl"
)

le = joblib.load(label_path)

# -------------------------------
# Negation detection
# -------------------------------
EXPLICIT_NEGATIONS = ["no", "not", "never", "none", "without", "cannot", "can't", "didn't", "doesn't"]
IMPLICIT_NEGATIONS = ["lack", "absence", "fail", "missing", "decline"]

def contains_negation(sentence: str) -> bool:
    words = re.findall(r"\b\w+\b", sentence.lower())
    lemmas = [lemmatizer.lemmatize(w) for w in words]
    return any(w in EXPLICIT_NEGATIONS or w in IMPLICIT_NEGATIONS for w in lemmas)

def find_negated_sentences_in_text(text: str):
    sentences = nltk.sent_tokenize(text)
    return [s for s in sentences if contains_negation(s)]

# -------------------------------
# Pretrained annotator
# -------------------------------
def pretrained_annotator(negated_sentences, report_name, confidence):
    hits = []
    for sentence in negated_sentences:
        if not sentence.strip():
            continue
        try:
            pred = clf(sentence, truncation=True, max_length=256)[0]
            label_id = int(pred["label"].replace("LABEL_", "")) if "LABEL_" in pred["label"] else int(pred["label"])
            theme = le.inverse_transform([label_id])[0]
            score = pred["score"]
            if score >= confidence:
                hits.append({
                    "framework": "PubMed-Theme",
                    "theme": theme,
                    "confidence": score,
                    #"combined_score": score,
                    "matched_keywords": [],
                    "matched_sentences": [sentence],
                })
        except Exception as e:
            print(f"⚠️ Error processing sentence in {report_name}: {e}")
    return hits

# -------------------------------
# Processing function
# -------------------------------
def process_selected_reports(df, text_column, confidenceScore):
    final_rows = []
    for idx, row in df.iterrows():
        text = str(row[text_column])
        report_name = row.get("Title", f"Report_{idx}")
        negated_sentences = find_negated_sentences_in_text(text)
        theme_hits = pretrained_annotator(negated_sentences, report_name, confidenceScore)

        for hit in theme_hits:
            final_rows.append({
                "Record ID": idx,
                "Title": report_name,
                "Framework": hit["framework"],
                "Theme": hit["theme"],
                "Confidence": hit["confidence"],
                #"Combined Score": hit["combined_score"],
                #"Matched Keywords": ", ".join(hit["matched_keywords"]),
                "coroner_name": row.get("coroner_name", ""),
                "coroner_area": row.get("coroner_area", ""),
                "year": row.get("year", ""),
                "date_of_report": row.get("date_of_report", ""),
                "Matched Sentences": " | ".join(hit["matched_sentences"]),
            })
    return pd.DataFrame(final_rows)