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
import datetime
nltk.download("punkt")

st.set_page_config(page_title="PubMedBERT Theme Annotation", layout="wide")


# Initialize model and tokenizer


MODEL_NAME = "kvara03/pubmedbert_theme_classifier_iteration5"

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
def _get_confidence_label(score):
    """Convert numerical score to confidence label"""
    if score >= 0.7:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"
    
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
                    "framework": "Extended Yorkshire Contributory",
                    "theme": theme,
                    "confidence": score,
                    "level": _get_confidence_label(score),
                    #"combined_score": score,
                    "matched_keywords": [],
                    "matched_sentences": [sentence],
                })
        except Exception as e:
            print(f"⚠️ Error processing sentence in {report_name}: {e}")
    return hits


# Processing function

    
def process_selected_reports(df, text_column, confidenceScore):
   
    print(df)
    df.columns = df.columns.str.strip()
    
    final_rows = []
    skipped_rows = []

    final_rows = []
    for idx, row in df.iterrows():
        text = str(row[text_column]).strip() if pd.notna(row[text_column]) and str(row[text_column]).strip() else None
        print(text)

        report_name = row.get("Title", f"Report_{idx}")
        print(report_name)
        print(text)
        negated_sentences = find_negated_sentences_in_text(text)
        #print(negated_sentences)
        theme_hits = pretrained_annotator(negated_sentences, report_name, confidenceScore)

        for hit in theme_hits:
            final_rows.append({
                "Record ID": idx,
                "Title": report_name,
                "Full Text": text,
                "Framework": hit["framework"],
                "Theme": hit["theme"],
                "Confidence Score": hit["confidence"],
                "Confidence": hit["level"] ,
                #"Matched Keywords": ", ".join(hit["matched_keywords"]),
                "coroner_name": row.get("coroner_name", ""),
                "coroner_area": row.get("coroner_area", ""),
                "year": row.get("year", ""),
                "date_of_report": row.get("date_of_report", ""),
                "Matched Sentences": " | ".join(hit["matched_sentences"]),
            })

    return pd.DataFrame(final_rows)
THEME_COLORS = {
        "Situational- Team Factors": "#FB7459",
        "Situational- Individual Staff Factors": "#A78B7C", 
        "Situational- Task Characteristics": "#F86A78",
        "Situational- Patient Factors": "#717EF6",
        "Local Working Conditions- Workload and Staffing Issues": "#E0F46D",
        "Local Working Conditions- Supervision and Leadership": "#ACF35B",
        "Local Working Conditions- Drugs, Equipment and Supplies": "#86CDF1",
        "Local Working Conditions- Lines of Responsibility": "#B077F2",
        "Local Working Conditions- Management of Staff and Staffing Levels": "#C874EC",
        "Organisational Factors- Physical Environment": "#7CBEF8",
        "Organisational Factors- Support from other departments": "#84EF7A",
        "Organisational Factors- Care Planning": "#F3C14D",
        "Organisational Factors- Staff Training and Education": "#F660EC",
        "Organisational Factors- Policies and Procedures": "#F97878",
        "Organisational Factors- Escalation/referral factor": "#EF7FD7",
        "External Factors- Design of Equipment, Supplies and Drugs": "#96B1FD",
        "External Factors- National Policies": "#83F3A1",
        "Communication and Culture- Safety Culture": "#F171CF",
        "Communication and Culture- Verbal and Written Communication": "#B0B1E7",
        "Human Error- Slips or Lapses": "#F9CEAF",
        "Human Error- Violations": "#66E7C2"
    }

def generate_html_report(results_df: pd.DataFrame, text_column = "Full Text")-> str:

    html_out = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Annotated Theme Report</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                margin: 0;
                padding: 20px;
                color: #333;
                background-color: #f9f9f9;
            }}
            h1 {{ 
                color: #2c3e50; 
                border-bottom: 3px solid #3498db; 
                padding-bottom: 10px; 
                margin-top: 30px;
                font-weight: 600;
            }}
            h2 {{ 
                color: #2c3e50; 
                margin-top: 30px; 
                border-bottom: 2px solid #bdc3c7; 
                padding-bottom: 5px; 
                font-weight: 600;
            }}
            h3 {{
                color: #34495e;
                font-weight: 600;
                margin-top: 20px;
            }}
            .record-container {{ 
                margin-bottom: 40px; 
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                padding: 20px;
                page-break-after: always; 
            }}
            .highlighted-text {{ 
                margin: 15px 0; 
                padding: 15px; 
                border-radius: 4px;
                border: 1px solid #ddd; 
                background-color: #fff; 
                line-height: 1.7;
            }}
            .theme-info {{ margin: 15px 0; }}
            .theme-info table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin-top: 15px;
                border-radius: 4px;
                overflow: hidden;
            }}
            .theme-info th, .theme-info td {{ 
                border: 1px solid #ddd; 
                padding: 12px; 
                text-align: left; 
            }}
            .theme-info th {{ 
                background-color: #3498db; 
                color: white;
                font-weight: 600;
            }}
            .theme-info tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .report-header {{
                background-color: #3498db;
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            .legend-container {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                padding: 15px;
                margin-bottom: 20px;
            }}
            .legend-title {{
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .theme-color-box {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 1px solid #999;
            }}
            @media print {{
                .record-container {{ page-break-after: always; }}
                body {{ background-color: white; }}
            }}
        </style>
    </head>
    <body>

    <div class="report-header">
        <h1>Annotated Theme Report</h1>
        <p>Generated on {datetime.datetime.now().strftime("%d %B %Y, %H:%M")}</p>
    </div>
    """

    # --------------------------------------------------
    # LEGEND
    # --------------------------------------------------

    html_out += """
    <div class="legend-container">
        <div class="legend-title">Extended Yorkshire Contributory Factors Framework</div>
        <table class="theme-info">
            <tr>
                <th>Theme</th>
                <th>Color</th>
            </tr>
    """

    THEME_COLORS_LOWER = {k.lower(): v for k, v in THEME_COLORS.items()}

    for theme, color in THEME_COLORS.items():
        html_out += f"""
            <tr>
                <td>{html.escape(theme)}</td>
                <td><div class="theme-color-box" style="background-color:{color};"></div></td>
            </tr>
        """

    html_out += "</table></div>"

    # --------------------------------------------------
    # PROCESS EACH DOCUMENT
    # --------------------------------------------------

    for title, group in results_df.groupby("Title"):

        html_out += f"""
        <div class="record-container">
            <h2>{html.escape(str(title))}</h2>
        """

        full_text = group["Full Text"].iloc[0]

        sentence_color_pairs = []
        for _, row in group.iterrows():
            theme = str(row["Theme"]).strip().lower()
            color = THEME_COLORS_LOWER.get(theme, "#f0f0f0")

            for sent in str(row["Matched Sentences"]).split(" | "):
                if sent.strip():
                    sentence_color_pairs.append((sent.strip(), color))

        highlighted_text = full_text
        for sent, color in sentence_color_pairs:
            highlighted_text = re.sub(
                re.escape(sent),
                rf"<span style='background-color:{color}; padding:3px 5px; border-radius:4px;'>{html.escape(sent)}</span>",
                highlighted_text,
                count=1
            )

        html_out += f"""
            <div class="highlighted-text">
                <h3>Annotated Content</h3>
                <p>{highlighted_text}</p>
            </div>
        """

        for _, row in group.iterrows():
            color = THEME_COLORS_LOWER.get(str(row["Theme"]).lower(), "#f0f0f0")

            html_out += f"""
            <div class="theme-info">
                <h3>{html.escape(row["Theme"])}</h3>
                <table>
                    <tr>
                        <th>Framework</th>
                        <th>Confidence Score</th>
                    </tr>
                    <tr>
                        <td>{html.escape(row["Framework"])}</td>
                        <td>{row["Confidence Score"]:.4f}</td>
                    </tr>
                </table>

                <h4>Matched Sentences</h4>
                <ul>
            """

            for s in str(row["Matched Sentences"]).split(" | "):
                html_out += f"""
                    <li padding:6px; margin-bottom:6px; border-radius:4px;">
                        {html.escape(s)}
                    </li>
                """

            html_out += "</ul></div>"

        html_out += "</div>"

    # --------------------------------------------------
    # END HTML
    # --------------------------------------------------

    html_out += """
    </body>
    </html>
    """
    return html_out
