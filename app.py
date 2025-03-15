import streamlit as st
import pyLDAvis
import pyLDAvis.sklearn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import requests
from bs4 import BeautifulSoup 
import time
import urllib3
import io
import pdfplumber
import logging
import os
import zipfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import networkx as nx
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from bs4 import BeautifulSoup, Tag
import json  # Added for JSON export functionality
# Initialize NLTK resources
import nltk 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import random
import string
import traceback
from datetime import datetime
from openpyxl.utils import get_column_letter
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
from typing import Union

# Initialize session state variables
if 'stop_scraping' not in st.session_state:
    st.session_state.stop_scraping = False

if 'reports' not in st.session_state:
    st.session_state.reports = []

class BM25Vectorizer(BaseEstimator, TransformerMixin):
    """BM25 vectorizer implementation"""
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0
    ):
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
    def fit(self, raw_documents: List[str], y=None):
        X = self.count_vectorizer.fit_transform(raw_documents)
        
        # Calculate document lengths
        self.doc_lengths = np.array(X.sum(axis=1)).flatten()
        self.avg_doc_length = np.mean(self.doc_lengths)
        
        # Calculate IDF scores
        n_samples = X.shape[0]
        df = np.bincount(X.indices, minlength=X.shape[1])
        df = np.maximum(df, 1)
        self.idf = np.log((n_samples - df + 0.5) / (df + 0.5) + 1.0)
        
        return self
        
    def transform(self, raw_documents: List[str]) -> sp.csr_matrix:
        X = self.count_vectorizer.transform(raw_documents)
        doc_lengths = np.array(X.sum(axis=1)).flatten()
        
        X = sp.csr_matrix(X)
        
        # Calculate BM25 scores
        for i in range(X.shape[0]):
            start_idx = X.indptr[i]
            end_idx = X.indptr[i + 1]
            
            freqs = X.data[start_idx:end_idx]
            length_norm = 1 - self.b + self.b * doc_lengths[i] / self.avg_doc_length
            
            # BM25 formula
            X.data[start_idx:end_idx] = (
                ((self.k1 + 1) * freqs) / 
                (self.k1 * length_norm + freqs)
            ) * self.idf[X.indices[start_idx:end_idx]]
        
        return X
        
    def get_feature_names_out(self):
        return self.count_vectorizer.get_feature_names_out()

class WeightedTfidfVectorizer(BaseEstimator, TransformerMixin):
    """TF-IDF vectorizer with configurable weighting schemes"""
    def __init__(
        self,
        tf_scheme: str = 'raw',
        idf_scheme: str = 'smooth',
        norm: Optional[str] = 'l2',
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0
    ):
        self.tf_scheme = tf_scheme
        self.idf_scheme = idf_scheme
        self.norm = norm
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
    
    def _compute_tf(self, X: sp.csr_matrix) -> sp.csr_matrix:
        if self.tf_scheme == 'raw':
            return X
        elif self.tf_scheme == 'log':
            X.data = np.log1p(X.data)
        elif self.tf_scheme == 'binary':
            X.data = np.ones_like(X.data)
        elif self.tf_scheme == 'augmented':
            max_tf = X.max(axis=1).toarray().flatten()
            max_tf[max_tf == 0] = 1
            for i in range(X.shape[0]):
                start = X.indptr[i]
                end = X.indptr[i + 1]
                X.data[start:end] = 0.5 + 0.5 * (X.data[start:end] / max_tf[i])
        return X
    
    def _compute_idf(self, X: sp.csr_matrix) -> np.ndarray:
        n_samples = X.shape[0]
        df = np.bincount(X.indices, minlength=X.shape[1])
        df = np.maximum(df, 1)
        
        if self.idf_scheme == 'smooth':
            return np.log((n_samples + 1) / (df + 1)) + 1
        elif self.idf_scheme == 'standard':
            return np.log(n_samples / df) + 1
        elif self.idf_scheme == 'probabilistic':
            return np.log((n_samples - df) / df)
    
    def fit(self, raw_documents: List[str], y=None):
        X = self.count_vectorizer.fit_transform(raw_documents)
        self.idf_ = self._compute_idf(X)
        return self
    
    def transform(self, raw_documents: List[str]) -> sp.csr_matrix:
        X = self.count_vectorizer.transform(raw_documents)
        X = self._compute_tf(X)
        X = X.multiply(self.idf_)
        
        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)
        
        return X
    
    def get_feature_names_out(self):
        return self.count_vectorizer.get_feature_names_out()

def get_vectorizer(
    vectorizer_type: str,
    max_features: int,
    min_df: float,
    max_df: float,
    **kwargs
) -> Union[TfidfVectorizer, BM25Vectorizer, WeightedTfidfVectorizer]:
    """Create and configure the specified vectorizer type"""
    
    if vectorizer_type == 'tfidf':
        return TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
    elif vectorizer_type == 'bm25':
        return BM25Vectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            k1=kwargs.get('k1', 1.5),
            b=kwargs.get('b', 0.75)
        )
    elif vectorizer_type == 'weighted':
        return WeightedTfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            tf_scheme=kwargs.get('tf_scheme', 'raw'),
            idf_scheme=kwargs.get('idf_scheme', 'smooth')
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
        
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global headers for all requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Referer': 'https://judiciary.uk/'
}

# Core utility functions
def make_request(url: str, retries: int = 3, delay: int = 2) -> Optional[requests.Response]:
    """Make HTTP request with retries and delay"""
    for attempt in range(retries):
        try:
            time.sleep(delay)
            response = requests.get(url, headers=HEADERS, verify=False, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"Request failed: {str(e)}")
                raise e
            time.sleep(delay * (attempt + 1))
    return None


def combine_document_text(row: pd.Series) -> str:
    """Combine all text content from a document"""
    text_parts = []
    
    # Add title and content
    if pd.notna(row.get('Title')):
        text_parts.append(str(row['Title']))
    if pd.notna(row.get('Content')):
        text_parts.append(str(row['Content']))
    
    # Add PDF contents
    pdf_columns = [col for col in row.index if col.startswith('PDF_') and col.endswith('_Content')]
    for pdf_col in pdf_columns:
        if pd.notna(row.get(pdf_col)):
            text_parts.append(str(row[pdf_col]))
    
    return ' '.join(text_parts)
    

def clean_text_for_modeling(text: str) -> str:
    """Clean text with enhanced noise removal"""
    if not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses and phone numbers
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove dates and times
        text = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text)
        text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', text)
        
        # Remove specific document-related terms
        text = re.sub(r'\b(?:ref|reference|case)(?:\s+no)?\.?\s*[-:\s]?\s*\w+[-\d]+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(regulation|paragraph|section|subsection|article)\s+\d+\b', '', text, flags=re.IGNORECASE)
        
        # Remove common legal document terms
        legal_terms = r'\b(coroner|inquest|hearing|evidence|witness|statement|report|dated|signed)\b'
        text = re.sub(legal_terms, '', text, flags=re.IGNORECASE)
        
        # Remove special characters and multiple spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words
        text = ' '.join(word for word in text.split() if len(word) > 2)
        
        # Ensure minimum content length
        cleaned_text = text.strip()
        return cleaned_text if len(cleaned_text.split()) >= 3 else ""
    
    except Exception as e:
        logging.error(f"Error in text cleaning: {e}")
        return ""

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
            'Â': ' ',
            '\u200b': '',
            '\uf0b7': '',
            '\u2019': "'",
            '\u201c': '"',
            '\u201d': '"',
            '\u2013': '-',
            '\u2022': '•'
        }
        
        for encoded, replacement in replacements.items():
            text = text.replace(encoded, replacement)
        
        text = re.sub(r'<[^>]+>', '', text)
        text = ''.join(char if char.isprintable() or char == '\n' else ' ' for char in text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    except Exception as e:
        logging.error(f"Error in clean_text: {e}")
        return ""


def extract_metadata(content: str) -> dict:
    """
    Extract structured metadata from report content with improved category handling.
    
    Args:
        content (str): Raw report content
        
    Returns:
        dict: Extracted metadata including date, reference, names, categories, etc.
    """
    metadata = {
        'date_of_report': None,
        'ref': None,
        'deceased_name': None,
        'coroner_name': None,
        'coroner_area': None,
        'categories': []
    }
    
    if not content:
        return metadata
        
    try:
        # Extract date patterns
        date_patterns = [
            r'Date of report:?\s*(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})',
            r'Date of report:?\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'DATED this (\d{1,2}(?:st|nd|rd|th)?\s+day of [A-Za-z]+\s+\d{4})',
            r'Date:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, content, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(1)
                try:
                    if '/' in date_str:
                        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                    else:
                        date_str = re.sub(r'(?<=\d)(st|nd|rd|th)', '', date_str)
                        date_str = re.sub(r'day of ', '', date_str)
                        try:
                            date_obj = datetime.strptime(date_str, '%d %B %Y')
                        except ValueError:
                            date_obj = datetime.strptime(date_str, '%d %b %Y')
                    
                    metadata['date_of_report'] = date_obj.strftime('%d/%m/%Y')
                    break
                except ValueError as e:
                    logging.warning(f"Invalid date format found: {date_str} - {e}")
        
        # Extract reference number
        ref_match = re.search(r'Ref(?:erence)?:?\s*([-\d]+)', content)
        if ref_match:
            metadata['ref'] = ref_match.group(1).strip()
        
        # Extract deceased name
        name_match = re.search(r'Deceased name:?\s*([^\n]+)', content)
        if name_match:
            metadata['deceased_name'] = clean_text(name_match.group(1)).strip()
        
        # Extract coroner details
        coroner_match = re.search(r'Coroner(?:\'?s)? name:?\s*([^\n]+)', content)
        if coroner_match:
            metadata['coroner_name'] = clean_text(coroner_match.group(1)).strip()
        
        area_match = re.search(r'Coroner(?:\'?s)? Area:?\s*([^\n]+)', content)
        if area_match:
            metadata['coroner_area'] = clean_text(area_match.group(1)).strip()
        
        # Extract categories with enhanced handling
        cat_match = re.search(r'Category:?\s*(.+?)(?=This report is being sent to:|$)', content, re.IGNORECASE | re.DOTALL)
        if cat_match:
            category_text = cat_match.group(1).strip()
            
            # Normalize all possible separators to pipe
            category_text = re.sub(r'\s*[,;]\s*', '|', category_text)
            category_text = re.sub(r'[•·⋅‣⁃▪▫–—-]\s*', '|', category_text)
            category_text = re.sub(r'\s{2,}', '|', category_text)
            category_text = re.sub(r'\n+', '|', category_text)
            
            # Split into individual categories
            categories = category_text.split('|')
            cleaned_categories = []
            
            # Get standard categories for matching
            standard_categories = {cat.lower(): cat for cat in get_pfd_categories()}
            
            for cat in categories:
                # Clean and normalize the category
                cleaned_cat = clean_text(cat).strip()
                cleaned_cat = re.sub(r'&nbsp;', '', cleaned_cat)
                cleaned_cat = re.sub(r'\s*This report.*$', '', cleaned_cat, flags=re.IGNORECASE)
                cleaned_cat = re.sub(r'[|,;]', '', cleaned_cat)
                
                # Only process non-empty categories
                if cleaned_cat and not re.match(r'^[\s|,;]+$', cleaned_cat):
                    # Try to match with standard categories
                    cat_lower = cleaned_cat.lower()
                    
                    # Check for exact match first
                    if cat_lower in standard_categories:
                        cleaned_cat = standard_categories[cat_lower]
                    else:
                        # Try partial matching
                        for std_lower, std_original in standard_categories.items():
                            if cat_lower in std_lower or std_lower in cat_lower:
                                cleaned_cat = std_original
                                break
                    
                    cleaned_categories.append(cleaned_cat)
            
            # Remove duplicates while preserving order
            seen = set()
            metadata['categories'] = [x for x in cleaned_categories 
                                    if not (x.lower() in seen or seen.add(x.lower()))]
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
        return metadata

def get_pfd_categories() -> List[str]:
    """Get all available PFD report categories"""
    return [
        "Accident at Work and Health and Safety related deaths",
        "Alcohol drug and medication related deaths",
        "Care Home Health related deaths",
        "Child Death from 2015",
        "Community health care and emergency services related deaths",
        "Emergency services related deaths 2019 onwards",
        "Hospital Death Clinical Procedures and medical management related deaths",
        "Mental Health related deaths",
        "Other related deaths",
        "Police related deaths",
        "Product related deaths",
        "Railway related deaths",
        "Road Highways Safety related deaths",
        "Service Personnel related deaths",
        "State Custody related deaths", 
        "Suicide from 2015",
        "Wales prevention of future deaths reports 2019 onwards"
    ]
    
# PDF handling functions
def save_pdf(pdf_url: str, base_dir: str = 'pdfs') -> Tuple[Optional[str], Optional[str]]:
    """Download and save PDF, return local path and filename"""
    try:
        os.makedirs(base_dir, exist_ok=True)
        
        response = make_request(pdf_url)
        if not response:
            return None, None
        
        filename = os.path.basename(pdf_url)
        filename = re.sub(r'[^\w\-_\. ]', '_', filename)
        local_path = os.path.join(base_dir, filename)
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        return local_path, filename
    
    except Exception as e:
        logging.error(f"Error saving PDF {pdf_url}: {e}")
        return None, None

def extract_pdf_content(pdf_path: str, chunk_size: int = 10) -> str:
    """Extract text from PDF file with memory management"""
    try:
        filename = os.path.basename(pdf_path)
        text_chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(0, len(pdf.pages), chunk_size):
                chunk = pdf.pages[i:i+chunk_size]
                chunk_text = "\n\n".join([page.extract_text() or "" for page in chunk])
                text_chunks.append(chunk_text)
                
        full_content = f"PDF FILENAME: {filename}\n\n{''.join(text_chunks)}"
        return clean_text(full_content)
        
    except Exception as e:
        logging.error(f"Error extracting PDF text from {pdf_path}: {e}")
        return ""


def get_report_content(url: str) -> Optional[Dict]:
    """Get full content from report page with improved PDF and response handling"""
    try:
        logging.info(f"Fetching content from: {url}")
        response = make_request(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find('div', class_='flow') or soup.find('article', class_='single__post')
        
        if not content:
            logging.warning(f"No content found at {url}")
            return None
        
        # Extract main report content
        paragraphs = content.find_all(['p', 'table'])
        webpage_text = '\n\n'.join(p.get_text(strip=True, separator=' ') for p in paragraphs)
        
        pdf_contents = []
        pdf_paths = []
        pdf_names = []
        pdf_types = []  # Track if PDF is main report or response
        
        # Find all PDF links with improved classification
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf$'))
        
        for pdf_link in pdf_links:
            pdf_url = pdf_link['href']
            pdf_text = pdf_link.get_text(strip=True).lower()
            
            # Determine PDF type
            is_response = any(word in pdf_text.lower() for word in ['response', 'reply'])
            pdf_type = 'response' if is_response else 'report'
            
            if not pdf_url.startswith(('http://', 'https://')):
                pdf_url = f"https://www.judiciary.uk{pdf_url}" if not pdf_url.startswith('/') else f"https://www.judiciary.uk/{pdf_url}"
            
            pdf_path, pdf_name = save_pdf(pdf_url)
            
            if pdf_path:
                pdf_content = extract_pdf_content(pdf_path)
                pdf_contents.append(pdf_content)
                pdf_paths.append(pdf_path)
                pdf_names.append(pdf_name)
                pdf_types.append(pdf_type)
        
        return {
            'content': clean_text(webpage_text),
            'pdf_contents': pdf_contents,
            'pdf_paths': pdf_paths,
            'pdf_names': pdf_names,
            'pdf_types': pdf_types
        }
        
    except Exception as e:
        logging.error(f"Error getting report content: {e}")
        return None

def scrape_page(url: str) -> List[Dict]:
    """Scrape a single page with improved PDF handling"""
    reports = []
    try:
        response = make_request(url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results_list = soup.find('ul', class_='search__list')
        
        if not results_list:
            logging.warning(f"No results list found on page: {url}")
            return []
        
        cards = results_list.find_all('div', class_='card')
        
        for card in cards:
            try:
                title_elem = card.find('h3', class_='card__title')
                if not title_elem:
                    continue
                    
                title_link = title_elem.find('a')
                if not title_link:
                    continue
                
                title = clean_text(title_link.text)
                card_url = title_link['href']
                
                if not card_url.startswith(('http://', 'https://')):
                    card_url = f"https://www.judiciary.uk{card_url}"
                
                logging.info(f"Processing report: {title}")
                content_data = get_report_content(card_url)
                
                if content_data:
                    report = {
                        'Title': title,
                        'URL': card_url,
                        'Content': content_data['content']
                    }
                    
                    # Add PDF details with type classification
                    for i, (name, content, path, pdf_type) in enumerate(zip(
                        content_data['pdf_names'],
                        content_data['pdf_contents'],
                        content_data['pdf_paths'],
                        content_data['pdf_types']
                    ), 1):
                        report[f'PDF_{i}_Name'] = name
                        report[f'PDF_{i}_Content'] = content
                        report[f'PDF_{i}_Path'] = path
                        report[f'PDF_{i}_Type'] = pdf_type
                    
                    reports.append(report)
                    logging.info(f"Successfully processed: {title}")
                
            except Exception as e:
                logging.error(f"Error processing card: {e}")
                continue
        
        return reports
        
    except Exception as e:
        logging.error(f"Error fetching page {url}: {e}")
        return []

def get_total_pages(url: str) -> Tuple[int, int]:
    """
    Get total number of pages and total results count
    
    Returns:
        Tuple[int, int]: (total_pages, total_results)
    """
    try:
        response = make_request(url)
        if not response:
            logging.error(f"No response from URL: {url}")
            return 0, 0
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # First check for total results count
        total_results = 0
        results_header = soup.find('div', class_='search__header')
        if results_header:
            results_text = results_header.get_text()
            match = re.search(r'found (\d+) results?', results_text, re.IGNORECASE)
            if match:
                total_results = int(match.group(1))
                total_pages = (total_results + 9) // 10  # 10 results per page
                return total_pages, total_results
        
        # If no results header, check pagination
        pagination = soup.find('nav', class_='navigation pagination')
        if pagination:
            page_numbers = pagination.find_all('a', class_='page-numbers')
            numbers = [int(p.text.strip()) for p in page_numbers if p.text.strip().isdigit()]
            if numbers:
                return max(numbers), len(numbers) * 10  # Approximate result count
        
        # If no pagination but results exist
        results = soup.find('ul', class_='search__list')
        if results and results.find_all('div', class_='card'):
            cards = results.find_all('div', class_='card')
            return 1, len(cards)
            
        return 0, 0
        
    except Exception as e:
        logging.error(f"Error in get_total_pages: {str(e)}")
        return 0, 0

def process_scraped_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean scraped data with metadata extraction"""
    try:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        # Create a copy
        df = df.copy()
        
        # Extract metadata from Content field if it exists
        if 'Content' in df.columns:
            # Process each row
            processed_rows = []
            for _, row in df.iterrows():
                # Start with original row data
                processed_row = row.to_dict()
                
                # Extract metadata using existing function
                content = str(row.get('Content', ''))
                metadata = extract_metadata(content)
                
                # Update row with metadata
                processed_row.update(metadata)
                processed_rows.append(processed_row)
            
            # Create new DataFrame from processed rows
            result = pd.DataFrame(processed_rows)
        else:
            result = df.copy()
        
        # Convert date_of_report to datetime with UK format handling
        if 'date_of_report' in result.columns:
            def parse_date(date_str):
                if pd.isna(date_str):
                    return pd.NaT
                
                date_str = str(date_str).strip()
                
                # If already in DD/MM/YYYY format
                if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
                    return pd.to_datetime(date_str, format='%d/%m/%Y')
                
                # Remove ordinal indicators
                date_str = re.sub(r'(\d)(st|nd|rd|th)', r'\1', date_str)
                
                # Try different formats
                formats = [
                    '%Y-%m-%d',
                    '%d-%m-%Y', 
                    '%d %B %Y',
                    '%d %b %Y'
                ]
                
                for fmt in formats:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except ValueError:
                        continue
                
                try:
                    return pd.to_datetime(date_str)
                except:
                    return pd.NaT
            
            result['date_of_report'] = result['date_of_report'].apply(parse_date)
        
        return result
            
    except Exception as e:
        logging.error(f"Error in process_scraped_data: {e}")
        return df

                            
def get_category_slug(category: str) -> str:
    """Generate proper category slug for the website's URL structure"""
    if not category:
        return None
        
    # Create a slug exactly matching the website's format
    slug = category.lower()\
        .replace(' ', '-')\
        .replace('&', 'and')\
        .replace('--', '-')\
        .strip('-')
    
    logging.info(f"Generated category slug: {slug} from category: {category}")
    return slug
