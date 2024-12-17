import pandas as pd
import re
import unicodedata
import logging
from typing import Dict, List

def clean_text(text: str) -> str:
    """Clean text while preserving structure and metadata formatting"""
    if not text:
        return ""
    
    try:
        text = str(text)
        text = unicodedata.normalize('NFKD', text)
        
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€¦': '...',
            'â€"': '-',
            'â€¢': '•',
            'Â': '',
            '\u200b': '',
            '\uf0b7': ''
        }
        
        for encoded, replacement in replacements.items():
            text = text.replace(encoded, replacement)
        
        text = re.sub(r'<[^>]+>', '', text)
        
        key_fields = [
            'Date of report:',
            'Ref:',
            'Deceased name:',
            'Coroner name:',
            'Coroners name:',
            'Coroner Area:',
            'Coroners Area:',
            'Category:',
            'This report is being sent to:'
        ]
        
        for field in key_fields:
            text = text.replace(field, f'\n{field}')
        
        text = ''.join(char if char.isprintable() or char == '\n' else ' ' for char in text)
        
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                if any(field in line for field in key_fields):
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        lines.append(f"{parts[0]}: {parts[1].strip()}")
                else:
                    lines.append(' '.join(line.split()))
        
        text = '\n'.join(lines)
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('"', '"').replace('"', '"')
        
        return text.strip()
    
    except Exception as e:
        logging.error(f"Error in clean_text: {e}")
        return ""

def extract_metadata(content: str) -> Dict:
    """Extract structured metadata from report content"""
    metadata = {
        'date_of_report': None,
        'ref': None,
        'deceased_name': None,
        'coroner_name': None,
        'coroner_area': None,
        'categories': []
    }
    
    try:
        # Extract date
        date_match = re.search(r'Date of report:\s*(\d{1,2}/\d{1,2}/\d{4})', content)
        if date_match:
            metadata['date_of_report'] = date_match.group(1)
        
        # Extract reference number
        ref_match = re.search(r'Ref:\s*([\d-]+)', content)
        if ref_match:
            metadata['ref'] = ref_match.group(1)
        
        # Extract deceased name
        name_match = re.search(r'Deceased name:\s*([^\n]+)', content)
        if name_match:
            metadata['deceased_name'] = name_match.group(1).strip()
        
        # Extract coroner details
        coroner_match = re.search(r'Coroner(?:s)? name:\s*([^\n]+)', content)
        if coroner_match:
            metadata['coroner_name'] = coroner_match.group(1).strip()
        
        area_match = re.search(r'Coroner(?:s)? Area:\s*([^\n]+)', content)
        if area_match:
            metadata['coroner_area'] = area_match.group(1).strip()
        
        # Extract categories
        cat_match = re.search(r'Category:\s*([^\n]+)', content)
        if cat_match:
            categories = cat_match.group(1).split('|')
            metadata['categories'] = [cat.strip() for cat in categories]
        
        return metadata
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
        return metadata

def process_scraped_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean scraped data"""
    try:
        df = df.copy()
        
        # Clean PDF content
        pdf_cols = [col for col in df.columns if col.endswith('_Content')]
        for col in pdf_cols:
            try:
                df[col] = df[col].fillna("").astype(str)
                df[col] = df[col].apply(clean_text)
            except Exception as e:
                logging.error(f"Error processing column {col}: {e}")
        
        # Extract metadata
        try:
            metadata = df['Content'].fillna("").apply(extract_metadata)
            metadata_df = pd.DataFrame(metadata.tolist())
            
            # Combine with original data
            result = pd.concat([df, metadata_df], axis=1)
            
            return result
        except Exception as e:
            logging.error(f"Error extracting metadata: {e}")
            return df
            
    except Exception as e:
        logging.error(f"Error in process_scraped_data: {e}")
        return df
