import re
import logging
import unicodedata
from datetime import datetime
import pandas as pd
import nltk # type: ignore

# Configure logging (can be centralized in the main app file later)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)

def clean_text(text: str) -> str:
    """Clean text while preserving structure and metadata formatting"""
    if not text:
        return ""

    try:
        text = str(text)
        text = unicodedata.normalize("NFKD", text)

        replacements = {
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            "â€¦": "...",
            'â€"': "-",
            "â€¢": "•",
            "Â": " ",
            "\u200b": "",
            "\uf0b7": "",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2022": "•",
        }

        for encoded, replacement in replacements.items():
            text = text.replace(encoded, replacement)

        text = re.sub(r"<[^>]+>", "", text)
        text = "".join(
            char if char.isprintable() or char == "\n" else " " for char in text
        )
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n+", "\n", text)

        return text.strip()

    except Exception as e:
        logging.error(f"Error in clean_text: {e}")
        return ""

def clean_text_for_modeling(text: str) -> str:
    """Clean text with enhanced noise removal for modeling purposes"""
    if not isinstance(text, str):
        return ""

    try:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "", text)
        text = re.sub(
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)
        text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "", text)
        text = re.sub(
            r"\b(?:ref|reference|case)(?:\s+no)?\.?\s*[-:\s]?\s*\w+[-\d]+\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(regulation|paragraph|section|subsection|article)\s+\d+\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        legal_terms = r"\b(coroner|inquest|hearing|evidence|witness|statement|report|dated|signed)\b"
        text = re.sub(legal_terms, "", text, flags=re.IGNORECASE)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = " ".join(word for word in text.split() if len(word) > 2)
        cleaned_text = text.strip()
        return cleaned_text if len(cleaned_text.split()) >= 3 else ""
    except Exception as e:
        logging.error(f"Error in text cleaning for modeling: {e}")
        return ""

def extract_concern_text(content: str) -> str:
    """Extract complete concern text from PFD report content with robust section handling"""
    if pd.isna(content) or not isinstance(content, str):
        return ""

    concern_identifiers = [
        "CORONER'S CONCERNS", "MATTERS OF CONCERN", "The MATTERS OF CONCERN",
        "CORONER'S CONCERNS are", "MATTERS OF CONCERN are", "The MATTERS OF CONCERN are",
        "HEALTHCARE SAFETY CONCERNS", "SAFETY CONCERNS", "PATIENT SAFETY ISSUES",
        "HSIB FINDINGS", "INVESTIGATION FINDINGS", "THE CORONER'S MATTER OF CONCERN",
        "CONCERNS AND RECOMMENDATIONS", "CONCERNS IDENTIFIED", "TheMATTERS OF CONCERN"
    ]
    content_norm = ' '.join(content.split())
    content_lower = content_norm.lower()
    start_idx = -1
    for identifier in concern_identifiers:
        identifier_lower = identifier.lower()
        pos = content_lower.find(identifier_lower)
        if pos != -1:
            start_idx = pos + len(identifier)
            if start_idx < len(content_norm) and content_norm[start_idx] == ":":
                start_idx += 1
            break
    if start_idx == -1:
        return ""

    end_markers = [
        "ACTION SHOULD BE TAKEN", "CONCLUSIONS", "YOUR RESPONSE", "COPIES",
        "SIGNED:", "DATED THIS", "NEXT STEPS", "YOU ARE UNDER A DUTY",
    ]
    end_idx = len(content_norm)
    for marker in end_markers:
        marker_pos = content_lower.find(marker.lower(), start_idx)
        if marker_pos != -1 and marker_pos < end_idx:
            end_idx = marker_pos
    concerns_text = content_norm[start_idx:end_idx].strip()
    last_period = concerns_text.rfind('.')
    if last_period != -1:
        concerns_text = concerns_text[:last_period + 1]
    return concerns_text

def get_pfd_categories() -> list[str]:
    """Get all available PFD report categories"""
    return [
        "Accident at Work and Health and Safety related deaths",
        "Alcohol drug and medication related deaths",
        "Care Home Health related deaths", "Child Death from 2015",
        "Community health care and emergency services related deaths",
        "Emergency services related deaths 2019 onwards",
        "Hospital Death Clinical Procedures and medical management related deaths",
        "Mental Health related deaths", "Other related deaths", "Police related deaths",
        "Product related deaths", "Railway related deaths", "Road Highways Safety related deaths",
        "Service Personnel related deaths", "State Custody related deaths", "Suicide from 2015",
        "Wales prevention of future deaths reports 2019 onwards",
    ]

def extract_metadata(content: str) -> dict:
    """Extract structured metadata from report content."""
    metadata = {
        "date_of_report": None, "ref": None, "deceased_name": None,
        "coroner_name": None, "coroner_area": None, "categories": [],
    }
    if not content:
        return metadata
    try:
        date_patterns = [
            r"Date of report:?\s*(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})",
            r"Date of report:?\s*(\d{1,2}/\d{1,2}/\d{4})",
            r"DATED this (\d{1,2}(?:st|nd|rd|th)?\s+day of [A-Za-z]+\s+\d{4})",
            r"Date:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, content, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(1)
                try:
                    if "/" in date_str:
                        date_obj = datetime.strptime(date_str, "%d/%m/%Y")
                    else:
                        date_str = re.sub(r"(?<=\d)(st|nd|rd|th)", "", date_str)
                        date_str = re.sub(r"day of ", "", date_str)
                        try:
                            date_obj = datetime.strptime(date_str, "%d %B %Y")
                        except ValueError:
                            date_obj = datetime.strptime(date_str, "%d %b %Y")
                    metadata["date_of_report"] = date_obj.strftime("%d/%m/%Y")
                    break
                except ValueError as e:
                    logging.warning(f"Invalid date format: {date_str} - {e}")

        ref_match = re.search(r"Ref(?:erence)?:?\s*([-\d]+)", content)
        if ref_match:
            metadata["ref"] = ref_match.group(1).strip()

        name_match = re.search(r"Deceased name:?\s*([^\n]+)", content)
        if name_match:
            metadata["deceased_name"] = clean_text(name_match.group(1)).strip()

        coroner_match = re.search(r"Coroner(?:\'?s)? name:?\s*([^\n]+)", content)
        if coroner_match:
            metadata["coroner_name"] = clean_text(coroner_match.group(1)).strip()

        area_match = re.search(r"Coroner(?:\'?s)? Area:?\s*([^\n]+)", content)
        if area_match:
            metadata["coroner_area"] = clean_text(area_match.group(1)).strip()

        cat_match = re.search(
            r"Category:?\s*(.+?)(?=This report is being sent to:|$)",
            content, re.IGNORECASE | re.DOTALL,
        )
        if cat_match:
            category_text = cat_match.group(1).strip()
            category_text = re.sub(r"\s*[,;]\s*", "|", category_text)
            category_text = re.sub(r"[•·⋅‣⁃▪▫–—-]\s*", "|", category_text)
            category_text = re.sub(r"\s{2,}", "|", category_text)
            category_text = re.sub(r"\n+", "|", category_text)
            categories = category_text.split("|")
            cleaned_categories = []
            standard_categories_map = {cat.lower(): cat for cat in get_pfd_categories()}
            for cat in categories:
                cleaned_cat = clean_text(cat).strip()
                cleaned_cat = re.sub(r"&nbsp;", "", cleaned_cat)
                cleaned_cat = re.sub(r"\s*This report.*$", "", cleaned_cat, flags=re.IGNORECASE)
                cleaned_cat = re.sub(r"[|,;]", "", cleaned_cat)
                if cleaned_cat and not re.match(r"^[\s|,;]+$", cleaned_cat):
                    cat_lower = cleaned_cat.lower()
                    if cat_lower in standard_categories_map:
                        cleaned_cat = standard_categories_map[cat_lower]
                    else:
                        for std_lower, std_original in standard_categories_map.items():
                            if cat_lower in std_lower or std_lower in cat_lower:
                                cleaned_cat = std_original
                                break
                    cleaned_categories.append(cleaned_cat)
            seen = set()
            metadata["categories"] = [
                x for x in cleaned_categories if not (x.lower() in seen or seen.add(x.lower()))
            ]
        return metadata
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
        return metadata

def process_scraped_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean scraped data with metadata and concern extraction."""
    try:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df_copy = df.copy()
        if "Content" in df_copy.columns:
            processed_rows = []
            for _, row in df_copy.iterrows():
                processed_row = row.to_dict()
                content = str(row.get("Content", ""))
                metadata_extracted = extract_metadata(content)
                processed_row["Extracted_Concerns"] = extract_concern_text(content)
                processed_row.update(metadata_extracted)
                processed_rows.append(processed_row)
            result = pd.DataFrame(processed_rows)
        else:
            result = df_copy.copy()

        if "date_of_report" in result.columns:
            def parse_date(date_str):
                if pd.isna(date_str): return pd.NaT
                date_str = str(date_str).strip()
                if re.match(r"\d{1,2}/\d{1,2}/\d{4}", date_str):
                    return pd.to_datetime(date_str, format="%d/%m/%Y", errors='coerce')
                date_str = re.sub(r"(\d)(st|nd|rd|th)", r"\1", date_str)
                formats = ["%Y-%m-%d", "%d-%m-%Y", "%d %B %Y", "%d %b %Y"]
                for fmt in formats:
                    try: return pd.to_datetime(date_str, format=fmt)
                    except ValueError: continue
                try: return pd.to_datetime(date_str, errors='coerce')
                except: return pd.NaT
            result["date_of_report"] = result["date_of_report"].apply(parse_date)
        return result
    except Exception as e:
        logging.error(f"Error in process_scraped_data: {e}")
        return df # Return original on error

def is_response_document(row: pd.Series) -> bool:
    """Check if a document is a response based on its metadata and content."""
    try:
        for i in range(1, 10):  # Check PDF_1 to PDF_9
            pdf_name = str(row.get(f"PDF_{i}_Name", "")).lower()
            if "response" in pdf_name or "reply" in pdf_name:
                return True
            pdf_type = str(row.get(f"PDF_{i}_Type", "")).lower() # Check type too
            if pdf_type == "response":
                return True

        title = str(row.get("Title", "")).lower()
        if any(word in title for word in ["response", "reply", "answered"]):
            return True

        content = str(row.get("Content", "")).lower()
        return any(
            phrase in content for phrase in [
                "in response to", "responding to", "reply to", "response to",
                "following the regulation 28", "following receipt of the regulation 28"
            ]
        )
    except Exception as e:
        logging.error(f"Error checking response type: {e}")
        return False

def format_date_uk(date_obj):
    """Convert datetime object to UK date format string"""
    if pd.isna(date_obj): return ""
    try:
        if isinstance(date_obj, str):
            date_obj = pd.to_datetime(date_obj, errors='coerce')
        if pd.isna(date_obj): return "" # if conversion failed
        return date_obj.strftime("%d/%m/%Y")
    except:
        return str(date_obj) # fallback to string representation

def create_document_identifier(row: pd.Series) -> str:
    """Create a unique identifier for a document."""
    title = str(row.get("Title", "")).strip()
    ref = str(row.get("ref", "")).strip()
    deceased = str(row.get("deceased_name", "")).strip()
    return f"{title}_{ref}_{deceased}"

def deduplicate_documents(data: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate documents while preserving unique entries."""
    if 'Title' not in data.columns or 'ref' not in data.columns: # Ensure necessary columns exist
        return data
    data["doc_id"] = data.apply(create_document_identifier, axis=1)
    deduped_data = data.drop_duplicates(subset=["doc_id"], keep="first")
    return deduped_data.drop(columns=["doc_id"])

def truncate_text(text, max_length=30):
    if not text or len(text) <= max_length:
        return text
    if ":" in text: # Special handling for "Framework: Theme"
        parts = text.split(":", 1)
        framework = parts[0].strip()
        theme_part = parts[1].strip()
        if len(theme_part) > max_length - len(framework) - 2:
             # Try to break theme part
            words = theme_part.split()
            processed_theme = []
            current_line = []
            current_len = 0
            for word in words:
                if current_len + len(word) + (1 if current_line else 0) <= max_length - len(framework) -2:
                    current_line.append(word)
                    current_len += len(word) + (1 if current_line else 0)
                else:
                    processed_theme.append(" ".join(current_line))
                    current_line = [word]
                    current_len = len(word)
            if current_line: processed_theme.append(" ".join(current_line))
            
            if len(processed_theme) > 2 : return f"{framework}: {processed_theme[0]}<br>{processed_theme[1]}..."
            return f"{framework}: {('<br>').join(processed_theme)}"
        return f"{framework}: {theme_part}"

    words = text.split()
    lines = []
    current_line = []
    current_len = 0
    for word in words:
        if current_len + len(word) + (1 if current_line else 0) <= max_length:
            current_line.append(word)
            current_len += len(word) + (1 if current_line else 0)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
    if current_line: lines.append(" ".join(current_line))
    if len(lines) > 2 : return f"{lines[0]}<br>{lines[1]}..."
    return "<br>".join(lines)

improved_truncate_text = truncate_text # Alias for now, can be differentiated later if needed

def perform_advanced_keyword_search(text: str, search_query: str) -> bool:
    """Perform advanced keyword search with AND/OR operators."""
    if not text or not search_query: return False
    text_lower = str(text).lower()
    query_lower = search_query.lower()
    if " and " in query_lower:
        keywords = [k.strip() for k in query_lower.split(" and ")]
        return all(keyword in text_lower for keyword in keywords if keyword)
    elif " or " in query_lower:
        keywords = [k.strip() for k in query_lower.split(" or ")]
        return any(keyword in text_lower for keyword in keywords if keyword)
    else:
        return query_lower.strip() in text_lower

def initialize_nltk():
    """Initialize required NLTK resources with error handling."""
    try:
        resources = ["punkt", "stopwords", "averaged_perceptron_tagger"]
        for resource in resources:
            try:
                if resource == "punkt": nltk.data.find("tokenizers/punkt")
                elif resource == "stopwords": nltk.data.find("corpora/stopwords")
                elif resource == "averaged_perceptron_tagger": nltk.data.find("taggers/averaged_perceptron_tagger")
            except LookupError:
                nltk.download(resource, quiet=True)
    except Exception as e:
        logging.error(f"Error initializing NLTK resources: {e}")
        # Consider raising an error or handling it gracefully in the app
        # For now, just log it. The app might still function for some parts.

def filter_by_categories(df: pd.DataFrame, selected_categories: list[str]) -> pd.DataFrame:
    """Filter DataFrame by categories, handling both list and string formats in the 'categories' column."""
    if not selected_categories: return df
    if "categories" not in df.columns: return df

    selected_categories_lower = [str(sc).lower().strip() for sc in selected_categories]

    def check_match(row_categories):
        if pd.isna(row_categories): return False
        
        current_cats_lower = []
        if isinstance(row_categories, list):
            current_cats_lower = [str(c).lower().strip() for c in row_categories if c]
        elif isinstance(row_categories, str):
            current_cats_lower = [c.strip() for c in str(row_categories).lower().split(',') if c.strip()]
        
        return any(sc in current_cats_lower for sc in selected_categories_lower)

    return df[df["categories"].apply(check_match)]

def filter_by_areas(df: pd.DataFrame, selected_areas: list[str]) -> pd.DataFrame:
    if not selected_areas: return df
    if "coroner_area" not in df.columns: return df
    
    selected_areas_lower = [str(area).lower().strip() for area in selected_areas]
    
    return df[df["coroner_area"].fillna("").str.lower().str.strip().apply(
        lambda x: any(sa == x for sa in selected_areas_lower) # Exact match after normalization
    )]

def filter_by_coroner_names(df: pd.DataFrame, selected_names: list[str]) -> pd.DataFrame:
    if not selected_names: return df
    if "coroner_name" not in df.columns: return df

    selected_names_lower = [str(name).lower().strip() for name in selected_names]
    
    return df[df["coroner_name"].fillna("").str.lower().str.strip().apply(
        lambda x: any(sn == x for sn in selected_names_lower) # Exact match after normalization
    )]

def filter_by_document_type(df: pd.DataFrame, doc_types: list[str]) -> pd.DataFrame:
    """Filter DataFrame based on document types (Report or Response)."""
    if not doc_types: return df
    
    is_response_series = df.apply(is_response_document, axis=1)
    
    if "Report" in doc_types and "Response" not in doc_types:
        return df[~is_response_series]
    elif "Response" in doc_types and "Report" not in doc_types:
        return df[is_response_series]
    # If both are selected or neither (though 'if not doc_types' handles neither), return all
    return df

def combine_document_text(row: pd.Series) -> str:
    """Combine all text content from a document row for analysis."""
    text_parts = []
    if pd.notna(row.get("Title")): text_parts.append(str(row["Title"]))
    if pd.notna(row.get("Content")): text_parts.append(str(row["Content"]))
    
    # Iterate through potential PDF content columns
    for i in range(1, 10): # Assuming up to PDF_9_Content
        pdf_col_name = f"PDF_{i}_Content"
        if pdf_col_name in row.index and pd.notna(row.get(pdf_col_name)):
            text_parts.append(str(row[pdf_col_name]))
            
    return " ".join(text_parts)
