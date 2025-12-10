import streamlit as st
import pandas as pd
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoTokenizer, AutoModel
import joblib
from huggingface_hub import hf_hub_download
import html

nltk.download("punkt")

st.set_page_config(page_title="PubMedBERT Theme Annotation", layout="wide")


# Initialize model and tokenizer


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


# Negation detection

EXPLICIT_NEGATIONS = ["no", "not", "never", "none", "without", "cannot", "can't", "didn't", "doesn't"]
IMPLICIT_NEGATIONS = ["lack", "absence", "fail", "missing", "decline"]

def contains_negation(sentence: str) -> bool:
    words = re.findall(r"\b\w+\b", sentence.lower())
    lemmas = [lemmatizer.lemmatize(w) for w in words]
    return any(w in EXPLICIT_NEGATIONS or w in IMPLICIT_NEGATIONS for w in lemmas)

def find_negated_sentences_in_text(text: str):
    sentences = nltk.sent_tokenize(text)
    return [s for s in sentences if contains_negation(s)]


# Pretrained annotator

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


# Processing function

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
                "Full Text": text,
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
THEME_COLORS = {
        "Situational- Team Factors": "#F54927",
        "Situational- Individual Staff Factors": "#F56F27", 
        "Situational- Task Characteristics": "#F5273C",
        "Situational- Patient Factors": "#273CF5",
        "Local Working Conditions- Workload and Staffing Issues": "#D6F527",
        "Local Working Conditions- Supervision and Leadership": "#95F527",
        "Local Working Conditions- Drugs, Equipment and Supplies": "#27556C",
        "Local Working Conditions- Lines of Responsibility": "#780BF4",
        "Local Working Conditions- Management of Staff and Staffing Levels": "#B727F5",
        "Organisational Factors- Physical Environment": "#2795F5",
        "Organisational Factors- Support from other departments": "#38F527",
        "Organisational Factors- Care Planning": "#F5B727",
        "Organisational Factors- Staff Training and Education": "#C809BB",
        "Organisational Factors- Policies and Procedures": "#700505",
        "Organisational Factors- Escalation/referral factor": "#F40BC1",
        "External Factors- Design of Equipment, Supplies and Drugs": "#275EF1",
        "External Factors- National Policies": "#09C83C",
        "Communication and Culture- Safety Culture": "",
        "Communication and Culture- Verbal and Written Communication": "#505287",
        "Human Error- Slips or Lapses": "#E06F1F",
        "Human Error- Violations": "#47E6B9"
    }

def generate_html_report(results_df: pd.DataFrame, text_column = "Full Text")-> str:

    html_out = "<html><head><meta charset='UTF-8'><title>Annotated Theme Report</title></head><body>"
    html_out += "<h1>Annotated Theme Report</h1>"

    # Legend Table
    html_out += "<h2>Theme Legend</h2><table border='1' cellpadding='6'>"
    html_out += "<tr><th>Theme</th><th>Color</th></tr>"

    # Create a lowercase key version of your theme colors
    THEME_COLORS_LOWER = {k.lower(): v for k, v in THEME_COLORS.items()}


    for theme, color in THEME_COLORS.items():
        html_out += f"<tr><td>{theme}</td><td style='background:{color};'>&nbsp;&nbsp;&nbsp;</td></tr>"

    html_out += "</table><hr>"

    #PROCESS EACH REPORT GROUP 
    for title, group in results_df.groupby("Title"):

        html_out += f"<h2>{html.escape(str(title))}</h2>"

        # Extract the full text
        full_text = group["Full Text"].iloc[0]

        
        # Build list of (sentence, color) pairs
        
        sentence_color_pairs = []
        for _, row in group.iterrows():
            theme = str(row["Theme"]).strip()
            theme_key = theme.lower()
            color = THEME_COLORS_LOWER.get(theme_key, "#f0f0f0")  # fallback color


            for sent in row["Matched Sentences"].split(" | "):
                sent_clean = sent.strip()
                if sent_clean:
                    sentence_color_pairs.append((sent_clean, color))

        
        # Highlight sentences inside the paragraph
        
        highlighted_text = full_text
        for sent, color in sentence_color_pairs:
            highlighted_text = re.sub(
                re.escape(sent),
                rf"<span style='background:{color}; padding:2px; border-radius:3px;'>{sent}</span>",
                highlighted_text,
                count=1
            )

    
        # Output highlighted content block
        
        html_out += f"<p><strong>Content:</strong><br>{highlighted_text}</p>"

        
        # Output each theme block
        
        for _, row in group.iterrows():
            theme = str(row["Theme"]).strip()
            theme_key = theme.lower()
            color = THEME_COLORS_LOWER.get(theme_key, "#f0f0f0")

            html_out += f"""
                <div style="border-left: 5px solid {color};
                            padding: 10px; margin: 12px 0;
                            background:{color}33; border-radius:6px;">
                    <p><strong>Theme:</strong> {html.escape(str(theme))}</p>
                    <p><strong>Framework:</strong> {html.escape(str(row['Framework']))}</p>
                    <p><strong>Confidence:</strong> {row['Confidence']:.4f}</p>
                    <h4>Matched Sentences:</h4>
                    <ul>
            """

            matched_sentences = row["Matched Sentences"].split(" | ")
            for s in matched_sentences:
                html_out += (
                    f"<li style='background:{color}; padding:4px; "
                    f"border-radius:4px;'>{html.escape(s)}</li>"
                )

            html_out += "</ul></div>"

        html_out += "<hr>"

    html_out += "</body></html>"
    return html_out
