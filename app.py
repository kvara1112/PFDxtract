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



def scrape_pfd_reports(
    keyword: Optional[str] = None,
    category: Optional[str] = None,
    order: str = "relevance",
    start_page: int = 1,
    end_page: Optional[int] = None
) -> List[Dict]:
    """
    Scrape PFD reports with enhanced progress tracking and proper pagination
    """
    all_reports = []
    base_url = "https://www.judiciary.uk/"
    
    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        report_count_text = st.empty()
        
        # Validate and prepare category
        category_slug = None
        if category:
            category_slug = category.lower()\
                .replace(' ', '-')\
                .replace('&', 'and')\
                .replace('--', '-')\
                .strip('-')
            logging.info(f"Using category: {category}, slug: {category_slug}")
        
        # Construct initial search URL
        base_search_url = construct_search_url(
            base_url=base_url,
            keyword=keyword,
            category=category,
            category_slug=category_slug
        )
        
        st.info(f"Searching at: {base_search_url}")
        
        # Get total pages and results count
        total_pages, total_results = get_total_pages(base_search_url)
        
        if total_results == 0:
            st.warning("No results found matching your search criteria")
            return []
            
        st.info(f"Found {total_results} matching reports across {total_pages} pages")
        
        # Apply page range limits
        start_page = max(1, start_page)  # Ensure start_page is at least 1
        if end_page is None:
            end_page = total_pages
        else:
            end_page = min(end_page, total_pages)  # Ensure end_page doesn't exceed total_pages
        
        if start_page > end_page:
            st.warning(f"Invalid page range: {start_page} to {end_page}")
            return []
            
        st.info(f"Scraping pages {start_page} to {end_page}")
        
        # Process each page in the specified range
        for current_page in range(start_page, end_page + 1):
            try:
                # Check if scraping should be stopped
                if hasattr(st.session_state, 'stop_scraping') and st.session_state.stop_scraping:
                    st.warning("Scraping stopped by user")
                    break
                
                # Update progress
                progress = (current_page - start_page) / (end_page - start_page + 1)
                progress_bar.progress(progress)
                status_text.text(f"Processing page {current_page} of {end_page} (out of {total_pages} total pages)")
                
                # Construct current page URL
                page_url = construct_search_url(
                    base_url=base_url,
                    keyword=keyword,
                    category=category,
                    category_slug=category_slug,
                    page=current_page
                )
                
                # Scrape current page
                page_reports = scrape_page(page_url)
                
                if page_reports:
                    # Deduplicate based on title and URL
                    existing_reports = {(r['Title'], r['URL']) for r in all_reports}
                    new_reports = [r for r in page_reports if (r['Title'], r['URL']) not in existing_reports]
                    
                    all_reports.extend(new_reports)
                    report_count_text.text(f"Retrieved {len(all_reports)} unique reports so far...")
                
                # Add delay between pages
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Error processing page {current_page}: {e}")
                st.warning(f"Error on page {current_page}. Continuing with next page...")
                continue
        
        # Sort results if specified
        if order != "relevance":
            all_reports = sort_reports(all_reports, order)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        report_count_text.empty()
        
        if all_reports:
            st.success(f"Successfully scraped {len(all_reports)} unique reports")
        else:
            st.warning("No reports were successfully retrieved")
        
        return all_reports
        
    except Exception as e:
        logging.error(f"Error in scrape_pfd_reports: {e}")
        st.error(f"An error occurred while scraping reports: {e}")
        return []


def construct_search_url(base_url: str, keyword: Optional[str] = None, 
                      category: Optional[str] = None, 
                      category_slug: Optional[str] = None, 
                      page: Optional[int] = None) -> str:
    """Constructs proper search URL with pagination"""
    # Start with base search URL
    url = f"{base_url}?s=&post_type=pfd"
    
    # Add category filter
    if category and category_slug:
        url += f"&pfd_report_type={category_slug}"
    
    # Add keyword search
    if keyword:
        url = f"{base_url}?s={keyword}&post_type=pfd"
        if category and category_slug:
            url += f"&pfd_report_type={category_slug}"
    
    # Add pagination
    if page and page > 1:
        url += f"&paged={page}"  # Changed from &page= to &paged= for proper pagination

    return url
def generate_lsa_reports(df_with_assignments, doc_stats, topic_stats, analysis_stats, document_summaries):
    """Generate statistical reports for LSA-based analysis"""
    reports = {}
    
    # Extract SVD model if available
    svd_model = analysis_stats.get('svd')
    
    # 1. Component analysis report
    if svd_model is not None:
        component_analysis = pd.DataFrame({
            'Component': np.arange(1, analysis_stats['n_components'] + 1),
            'Explained Variance Ratio': svd_model.explained_variance_ratio_,
            'Cumulative Explained Variance': np.cumsum(svd_model.explained_variance_ratio_)
        })
    else:
        # Create simpler report if SVD model not available
        component_analysis = pd.DataFrame({
            'Component': [analysis_stats['n_components']],
            'Cumulative Explained Variance': [analysis_stats['explained_variance']]
        })
    
    reports['component_analysis'] = component_analysis
    
    # 2. Cluster summary
    cluster_summary = pd.DataFrame([
        {
            'Cluster': row['cluster_id'],
            'Documents': row['size'],
            'Percentage': row['size'] / len(df_with_assignments) * 100,
            'Top Terms': ', '.join([term[0] for term in row['top_terms_by_frequency'][:10]]),
            'Distinctive Terms': ', '.join([term[0] for term in row['top_terms_by_distinctiveness'][:5]]),
            'Summary': f"Cluster {row['cluster_id']}: " + 
                      f"{row['size']} documents ({row['size'] / len(df_with_assignments) * 100:.1f}%) " +
                      f"focused on {', '.join([term[0] for term in row['top_terms_by_frequency'][:5]])}"
        }
        for _, row in topic_stats.iterrows()
    ])
    reports['cluster_summary'] = cluster_summary
    
    # 3. Document statistics
    reports['document_statistics'] = doc_stats
    
    # 4. Term importance by cluster
    term_importance = []
    for _, row in topic_stats.iterrows():
        for term, freq, dist in row['top_terms_by_frequency'][:20]:
            term_importance.append({
                'Cluster': row['cluster_id'],
                'Term': term,
                'Within-Cluster Frequency': freq,
                'Distinctiveness': dist,
                'Relative Importance': freq * dist
            })
    reports['term_importance'] = pd.DataFrame(term_importance)
    
    # 5. Top distinctive terms by cluster
    distinctive_terms = []
    for _, row in topic_stats.iterrows():
        for term, freq, dist in sorted(row['top_terms_by_distinctiveness'], key=lambda x: x[2], reverse=True)[:20]:
            if freq > 0:
                distinctive_terms.append({
                    'Cluster': row['cluster_id'],
                    'Term': term,
                    'Frequency': freq,
                    'Distinctiveness Score': dist
                })
    reports['distinctive_terms'] = pd.DataFrame(distinctive_terms)
    
    # 6. Cluster similarity matrix
    centroids = np.vstack([row['centroid'] for _, row in topic_stats.iterrows()])
    similarity_matrix = cosine_similarity(centroids)
    reports['cluster_similarity'] = pd.DataFrame(
        similarity_matrix,
        index=[f'Cluster {i}' for i in range(len(centroids))],
        columns=[f'Cluster {i}' for i in range(len(centroids))]
    )
    
    # 7. Document summaries (if available)
    if not document_summaries.empty:
        reports['document_summaries'] = document_summaries
    
    # 8. Analysis parameters
    reports['analysis_parameters'] = pd.DataFrame([{
        'Parameter': 'Analysis Type',
        'Value': 'LSA (Latent Semantic Analysis)'
    }, {
        'Parameter': 'Number of Components',
        'Value': analysis_stats['n_components']
    }, {
        'Parameter': 'Explained Variance',
        'Value': f"{analysis_stats['explained_variance']:.2%}"
    }, {
        'Parameter': 'Number of Clusters',
        'Value': analysis_stats['n_clusters']
    }, {
        'Parameter': 'Silhouette Score',
        'Value': f"{analysis_stats['silhouette_score']:.3f}"
    }, {
        'Parameter': 'Document Count',
        'Value': len(df_with_assignments)
    }])
    
    return reports

def generate_lda_reports(df_with_assignments, doc_stats, topic_stats, analysis_stats, document_summaries):
    """Generate statistical reports for LDA-based analysis"""
    reports = {}
    
    # 1. Topic summary
    topic_summary = pd.DataFrame([
        {
            'Topic': row['topic_id'],
            'Documents': len(row['representative_docs']),
            'Prevalence': row['prevalence'] * 100,
            'Top Terms': ', '.join([term[0] for term in row['top_terms'][:10]]),
            'Avg Weight': row['avg_weight'],
            'Distinctiveness': row['distinctiveness'],
            'Summary': f"Topic {row['topic_id']}: " + 
                      f"Prevalence {row['prevalence']*100:.1f}%, " +
                      f"focused on {', '.join([term[0] for term in row['top_terms'][:5]])}"
        }
        for _, row in topic_stats.iterrows()
    ])
    reports['topic_summary'] = topic_summary
    
    # 2. Document-topic statistics
    reports['document_statistics'] = doc_stats
    
    # 3. Term weights by topic
    term_weights = []
    for _, row in topic_stats.iterrows():
        for term, weight in row['top_terms'][:20]:
            term_weights.append({
                'Topic': row['topic_id'],
                'Term': term,
                'Weight': weight,
                'Topic Prevalence': row['prevalence'],
                'Scaled Importance': weight * row['prevalence']
            })
    reports['term_weights'] = pd.DataFrame(term_weights)
    
    # 4. Topic-term matrix (for visualization)
    # Create a matrix of top terms for each topic
    topic_term_matrix = {}
    for _, row in topic_stats.iterrows():
        topic_id = row['topic_id']
        for term, weight in row['top_terms'][:10]:
            if term not in topic_term_matrix:
                topic_term_matrix[term] = {}
            topic_term_matrix[term][f'Topic {topic_id}'] = weight
    
    # Convert to DataFrame
    topic_term_df = pd.DataFrame(topic_term_matrix).fillna(0).T
    reports['topic_term_matrix'] = topic_term_df
    
    # 5. Document summaries (if available)
    if not document_summaries.empty:
        reports['document_summaries'] = document_summaries
    
    # 6. Topic-document matrix
    # Create a matrix of top documents for each topic
    topic_doc_matrix = {}
    for _, row in topic_stats.iterrows():
        topic_id = row['topic_id']
        for doc_idx, weight in row['representative_docs']:
            doc_title = df_with_assignments.iloc[doc_idx].get('Title', f"Document {doc_idx}")
            if doc_title not in topic_doc_matrix:
                topic_doc_matrix[doc_title] = {}
            topic_doc_matrix[doc_title][f'Topic {topic_id}'] = weight
    
    # Convert to DataFrame
    topic_doc_df = pd.DataFrame(topic_doc_matrix).fillna(0).T
    reports['topic_document_matrix'] = topic_doc_df
    
    # 7. Topic correlation matrix
    # Calculate correlations between topic distributions
    topic_correlations = np.zeros((len(topic_stats), len(topic_stats)))
    for i, row_i in enumerate(topic_stats.iterrows()):
        for j, row_j in enumerate(topic_stats.iterrows()):
            # Extract terms and weights
            terms_i = {term: weight for term, weight in row_i[1]['top_terms']}
            terms_j = {term: weight for term, weight in row_j[1]['top_terms']}
            
            # Find common terms
            common_terms = set(terms_i.keys()) & set(terms_j.keys())
            if common_terms:
                # Calculate correlation based on common term weights
                weights_i = [terms_i[term] for term in common_terms]
                weights_j = [terms_j[term] for term in common_terms]
                correlation = np.corrcoef(weights_i, weights_j)[0, 1]
                topic_correlations[i, j] = correlation
    
    # Create correlation matrix DataFrame
    topic_correlation_df = pd.DataFrame(
        topic_correlations,
        index=[f'Topic {i}' for i in range(len(topic_stats))],
        columns=[f'Topic {i}' for i in range(len(topic_stats))]
    )
    reports['topic_correlations'] = topic_correlation_df
    
    # 8. Analysis parameters
    reports['analysis_parameters'] = pd.DataFrame([{
        'Parameter': 'Analysis Type',
        'Value': 'LDA (Latent Dirichlet Allocation)'
    }, {
        'Parameter': 'Number of Topics',
        'Value': analysis_stats['n_topics']
    }, {
        'Parameter': 'Topic Coherence',
        'Value': f"{analysis_stats['topic_coherence']:.3f}"
    }, {
        'Parameter': 'Perplexity',
        'Value': f"{analysis_stats['perplexity']:.2f}"
    }, {
        'Parameter': 'Document Count',
        'Value': len(df_with_assignments)
    }])
    
    return reports

def display_advanced_analysis_results():
    """Display the results of advanced document analysis"""
    if 'advanced_analysis_results' not in st.session_state:
        st.warning("No analysis results available. Please run the analysis first.")
        return
    
    results = st.session_state.advanced_analysis_results
    reports = st.session_state.advanced_analysis_reports
    
    # Determine analysis type for display
    analysis_type = results['analysis_type']
    
    # Create appropriate tabs based on analysis type
    if analysis_type == 'lsa':
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Cluster Summary", 
            "Term Analysis", 
            "Document Clusters",
            "Similarity Analysis",
            "Document Summaries"
        ])
        
        with tab1:
            st.subheader("Cluster Summary")
            st.dataframe(
                reports['cluster_summary'],
                column_config={
                    'Cluster': st.column_config.NumberColumn('Cluster ID'),
                    'Documents': st.column_config.NumberColumn('Document Count'),
                    'Percentage': st.column_config.NumberColumn('% of Corpus', format="%.1f%%"),
                    'Summary': st.column_config.TextColumn('Description')
                },
                hide_index=True
            )
            
            # Cluster distribution chart
            st.subheader("Cluster Size Distribution")
            cluster_counts = results['df_with_assignments']['cluster'].value_counts().sort_index()
            
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                labels={'x': 'Cluster', 'y': 'Number of Documents'},
                title='Document Distribution by Cluster'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Key Terms by Cluster")
            
            # Create cluster selector
            clusters = sorted(reports['term_importance']['Cluster'].unique())
            selected_cluster = st.selectbox(
                "Select cluster to view terms:",
                options=clusters
            )
            
            # Display top terms for selected cluster
            top_terms = reports['term_importance'][reports['term_importance']['Cluster'] == selected_cluster]
            top_terms = top_terms.sort_values('Relative Importance', ascending=False).head(20)
            
            st.subheader(f"Top Terms for Cluster {selected_cluster}")
            st.dataframe(
                top_terms[['Term', 'Within-Cluster Frequency', 'Distinctiveness', 'Relative Importance']],
                column_config={
                    'Term': st.column_config.TextColumn('Term'),
                    'Within-Cluster Frequency': st.column_config.NumberColumn('Frequency', format="%.3f"),
                    'Distinctiveness': st.column_config.NumberColumn('Distinctiveness', format="%.3f"),
                    'Relative Importance': st.column_config.NumberColumn('Importance', format="%.3f")
                },
                hide_index=True
            )
            
            # Display term importance chart
            fig = px.bar(
                top_terms.head(15),
                x='Term',
                y='Relative Importance',
                color='Distinctiveness',
                color_continuous_scale='Viridis',
                title=f'Top 15 Terms for Cluster {selected_cluster}'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display distinctive terms
            st.subheader(f"Most Distinctive Terms for Cluster {selected_cluster}")
            distinctive_terms = reports['distinctive_terms'][reports['distinctive_terms']['Cluster'] == selected_cluster]
            distinctive_terms = distinctive_terms.sort_values('Distinctiveness Score', ascending=False).head(15)
            
            st.dataframe(
                distinctive_terms[['Term', 'Frequency', 'Distinctiveness Score']],
                column_config={
                    'Term': st.column_config.TextColumn('Term'),
                    'Frequency': st.column_config.NumberColumn('Frequency', format="%.3f"),
                    'Distinctiveness Score': st.column_config.NumberColumn('Distinctiveness', format="%.3f")
                },
                hide_index=True
            )
            
            # Display distinctive terms chart
            fig = px.bar(
                distinctive_terms,
                x='Term',
                y='Distinctiveness Score',
                color='Frequency',
                color_continuous_scale='Inferno',
                title=f'Most Distinctive Terms for Cluster {selected_cluster}'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Documents by Cluster")
            
            # Create cluster selector
            clusters = sorted(results['df_with_assignments']['cluster'].unique())
            selected_cluster = st.selectbox(
                "Select cluster to view documents:",
                options=clusters,
                key="document_cluster_selector"
            )
            
            # Get documents in selected cluster
            cluster_docs = results['df_with_assignments'][results['df_with_assignments']['cluster'] == selected_cluster]
            
            # Get document statistics
            doc_stats = results['doc_stats'][results['doc_stats']['cluster'] == selected_cluster]
            doc_stats = doc_stats.sort_values('similarity_to_centroid', ascending=False)
            
            # Add statistics to document display
            if len(cluster_docs) > 0 and len(doc_stats) > 0:
                # Create mapping from document_id to statistics
                stats_map = {row['document_id']: row for _, row in doc_stats.iterrows()}
                
                # Create display dataframe
                display_docs = []
                for i, (_, doc) in enumerate(cluster_docs.iterrows()):
                    if i in stats_map:
                        stats = stats_map[i]
                        
                        # Get document preview
                        content = str(doc[results['content_column']])
                        preview = content[:300] + "..." if len(content) > 300 else content
                        
                        display_docs.append({
                            'Document ID': i,
                            'Title': doc.get('Title', f"Document {i}"),
                            'Similarity': stats['similarity_to_centroid'],
                            'Distinctiveness': stats['distinctiveness'],
                            'Preview': preview
                        })
                
                display_df = pd.DataFrame(display_docs)
                
                # Show top documents in cluster
                st.subheader(f"Most Representative Documents in Cluster {selected_cluster}")
                st.dataframe(
                    display_df.sort_values('Similarity', ascending=False).head(10),
                    column_config={
                        'Document ID': st.column_config.NumberColumn('Doc ID'),
                        'Title': st.column_config.TextColumn('Title'),
                        'Similarity': st.column_config.NumberColumn('Similarity to Centroid', format="%.3f"),
                        'Distinctiveness': st.column_config.NumberColumn('Distinctiveness', format="%.3f"),
                        'Preview': st.column_config.TextColumn('Document Preview')
                    },
                    hide_index=True
                )
                
                # Show borderline documents
                st.subheader(f"Borderline Documents in Cluster {selected_cluster}")
                st.dataframe(
                    display_df.sort_values('Distinctiveness').head(5),
                    column_config={
                        'Document ID': st.column_config.NumberColumn('Doc ID'),
                        'Title': st.column_config.TextColumn('Title'),
                        'Similarity': st.column_config.NumberColumn('Similarity to Centroid', format="%.3f"),
                        'Distinctiveness': st.column_config.NumberColumn('Distinctiveness', format="%.3f"),
                        'Preview': st.column_config.TextColumn('Document Preview')
                    },
                    hide_index=True
                )
            else:
                st.warning(f"No documents found in cluster {selected_cluster}")
        
        with tab4:
            st.subheader("Cluster Similarity Analysis")
            
            # Display cluster similarity heatmap
            cluster_similarity = reports['cluster_similarity']
            
            fig = px.imshow(
                cluster_similarity.values,
                labels=dict(x="Cluster", y="Cluster", color="Similarity"),
                x=cluster_similarity.columns,
                y=cluster_similarity.index,
                color_continuous_scale="Viridis",
                aspect="auto"
            )
            
            fig.update_layout(
                title="Cluster Similarity Matrix",
                xaxis_title="Cluster",
                yaxis_title="Cluster",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display similarity interpretation
            st.markdown("""
            ### Similarity Interpretation
            
            - **High similarity (> 0.7)** indicates clusters with overlapping themes
            - **Medium similarity (0.3 - 0.7)** indicates related but distinct clusters
            - **Low similarity (< 0.3)** indicates very different content areas
            
            Clusters with high similarity might be candidates for merging in future analyses.
            """)
        
        with tab5:
            st.subheader("Document Summaries")
            
            # Check if document summaries are available
            if 'document_summaries' in reports and not reports['document_summaries'].empty:
                summaries = reports['document_summaries']
                
                # Group by cluster
                clusters = sorted(summaries['topic_or_cluster'].unique())
                selected_cluster = st.selectbox(
                    "Select cluster to view summaries:",
                    options=clusters,
                    key="summary_cluster_selector"
                )
                
                # Display summaries for selected cluster
                cluster_summaries = summaries[summaries['topic_or_cluster'] == selected_cluster]
                
                for _, summary in cluster_summaries.iterrows():
                    with st.expander(f"{summary['document_title']}"):
                        st.markdown(f"**Document ID**: {summary['document_id']}")
                        st.markdown(f"**Summary**: {summary['summary']}")
                        st.markdown(f"**Reduction**: {summary['summary_length']} / {summary['full_length']} characters ({(summary['summary_length']/summary['full_length']*100):.1f}%)")
                
            else:
                st.info("No document summaries were generated. Run the analysis with 'Generate Document Summaries' option enabled.")
    
    else:  # LDA analysis
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Topic Summary", 
            "Term Analysis", 
            "Document Topics",
            "Topic Correlations",
            "Document Summaries"
        ])
        
        with tab1:
            st.subheader("Topic Summary")
            st.dataframe(
                reports['topic_summary'],
                column_config={
                    'Topic': st.column_config.NumberColumn('Topic ID'),
                    'Documents': st.column_config.NumberColumn('Primary Documents'),
                    'Prevalence': st.column_config.NumberColumn('Prevalence %', format="%.1f%%"),
                    'Avg Weight': st.column_config.NumberColumn('Average Weight', format="%.3f"),
                    'Distinctiveness': st.column_config.NumberColumn('Distinctiveness', format="%.3f"),
                    'Summary': st.column_config.TextColumn('Description')
                },
                hide_index=True
            )
            
            # Topic prevalence chart
            st.subheader("Topic Prevalence")
            topic_prev = reports['topic_summary'][['Topic', 'Prevalence']].sort_values('Topic')
            
            fig = px.bar(
                topic_prev,
                x='Topic',
                y='Prevalence',
                labels={'x': 'Topic', 'y': 'Prevalence (%)'},
                title='Topic Distribution'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Key Terms by Topic")
            
            # Create topic selector
            topics = sorted(reports['term_weights']['Topic'].unique())
            selected_topic = st.selectbox(
                "Select topic to view terms:",
                options=topics
            )
            
            # Display top terms for selected topic
            top_terms = reports['term_weights'][reports['term_weights']['Topic'] == selected_topic]
            top_terms = top_terms.sort_values('Weight', ascending=False).head(20)
            
            st.subheader(f"Top Terms for Topic {selected_topic}")
            st.dataframe(
                top_terms[['Term', 'Weight', 'Scaled Importance']],
                column_config={
                    'Term': st.column_config.TextColumn('Term'),
                    'Weight': st.column_config.NumberColumn('Weight', format="%.4f"),
                    'Scaled Importance': st.column_config.NumberColumn('Scaled Importance', format="%.4f")
                },
                hide_index=True
            )
            
            # Display term importance chart
            fig = px.bar(
                top_terms.head(15),
                x='Term',
                y='Weight',
                color='Scaled Importance',
                color_continuous_scale='Viridis',
                title=f'Top 15 Terms for Topic {selected_topic}'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display term heatmap across topics
            st.subheader("Topic-Term Heatmap")
            topic_term = reports['topic_term_matrix']
            
            # Select top terms for visualization
            top_overall_terms = top_terms['Term'].head(10).tolist()
            term_subset = topic_term.loc[top_overall_terms] if all(t in topic_term.index for t in top_overall_terms) else topic_term
            
            fig = px.imshow(
                term_subset,
                labels=dict(x="Topic", y="Term", color="Weight"),
                x=term_subset.columns,
                y=term_subset.index,
                color_continuous_scale="Viridis",
                aspect="auto"
            )
            
            fig.update_layout(
                title=f"Term Weights Across Topics (Top Terms from Topic {selected_topic})",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Documents by Topic")
            
            # Create topic selector
            topics = sorted(results['df_with_assignments']['topic'].unique())
            selected_topic = st.selectbox(
                "Select topic to view documents:",
                options=topics,
                key="document_topic_selector"
            )
            
            # Get documents in selected topic
            topic_docs = results['df_with_assignments'][results['df_with_assignments']['topic'] == selected_topic]
            
            # Get document statistics
            doc_stats = results['doc_stats'][results['doc_stats']['primary_topic'] == selected_topic]
            doc_stats = doc_stats.sort_values('primary_topic_weight', ascending=False)
            
            # Add statistics to document display
            if len(topic_docs) > 0 and len(doc_stats) > 0:
                # Create mapping from document_id to statistics
                stats_map = {row['document_id']: row for _, row in doc_stats.iterrows()}
                
                # Create display dataframe
                display_docs = []
                for i, (idx, doc) in enumerate(topic_docs.iterrows()):
                    if idx in stats_map:
                        stats = stats_map[idx]
                        
                        # Get document preview
                        content = str(doc[results['content_column']])
                        preview = content[:300] + "..." if len(content) > 300 else content
                        
                        display_docs.append({
                            'Document ID': idx,
                            'Title': doc.get('Title', f"Document {idx}"),
                            'Topic Weight': stats['primary_topic_weight'],
                            'Topic Clarity': stats['topic_clarity'],
                            'Secondary Topic': stats['secondary_topic'],
                            'Preview': preview
                        })
                
                display_df = pd.DataFrame(display_docs)
                
                # Show top documents in topic
                st.subheader(f"Most Representative Documents for Topic {selected_topic}")
                st.dataframe(
                    display_df.sort_values('Topic Weight', ascending=False).head(10),
                    column_config={
                        'Document ID': st.column_config.NumberColumn('Doc ID'),
                        'Title': st.column_config.TextColumn('Title'),
                        'Topic Weight': st.column_config.NumberColumn('Topic Weight', format="%.3f"),
                        'Topic Clarity': st.column_config.NumberColumn('Topic Clarity', format="%.3f"),
                        'Secondary Topic': st.column_config.NumberColumn('Secondary Topic'),
                        'Preview': st.column_config.TextColumn('Document Preview')
                    },
                    hide_index=True
                )
                
                # Show ambiguous documents
                st.subheader(f"Documents with Mixed Topic Assignment")
                st.dataframe(
                    display_df.sort_values('Topic Clarity').head(5),
                    column_config={
                        'Document ID': st.column_config.NumberColumn('Doc ID'),
                        'Title': st.column_config.TextColumn('Title'),
                        'Topic Weight': st.column_config.NumberColumn('Topic Weight', format="%.3f"),
                        'Topic Clarity': st.column_config.NumberColumn('Topic Clarity', format="%.3f"),
                        'Secondary Topic': st.column_config.NumberColumn('Secondary Topic'),
                        'Preview': st.column_config.TextColumn('Document Preview')
                    },
                    hide_index=True
                )
            else:
                st.warning(f"No documents found with primary topic {selected_topic}")
        
        with tab4:
            st.subheader("Topic Correlation Analysis")
            
            # Display topic correlation heatmap
            topic_corr = reports['topic_correlations']
            
            fig = px.imshow(
                topic_corr.values,
                labels=dict(x="Topic", y="Topic", color="Correlation"),
                x=topic_corr.columns,
                y=topic_corr.index,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                aspect="auto"
            )
            
            fig.update_layout(
                title="Topic Correlation Matrix",
                xaxis_title="Topic",
                yaxis_title="Topic",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display correlation interpretation
            st.markdown("""
            ### Correlation Interpretation
            
            - **Positive correlation (blue)** indicates topics that often appear together or have similar term distributions
            - **Negative correlation (red)** indicates topics that rarely co-occur or have very different term distributions
            - **Near-zero correlation (white)** indicates independent, unrelated topics
            
            Highly correlated topics may represent aspects of the same broader theme.
            """)
        
        with tab5:
            st.subheader("Document Summaries")
            
            # Check if document summaries are available
            if 'document_summaries' in reports and not reports['document_summaries'].empty:
                summaries = reports['document_summaries']
                
                # Group by topic
                topics = sorted(summaries['topic_or_cluster'].unique())
                selected_topic = st.selectbox(
                    "Select topic to view summaries:",
                    options=topics,
                    key="summary_topic_selector"
                )
                
                # Display summaries for selected topic
                topic_summaries = summaries[summaries['topic_or_cluster'] == selected_topic]
                
                for _, summary in topic_summaries.iterrows():
                    with st.expander(f"{summary['document_title']}"):
                        st.markdown(f"**Document ID**: {summary['document_id']}")
                        st.markdown(f"**Summary**: {summary['summary']}")
                        st.markdown(f"**Reduction**: {summary['summary_length']} / {summary['full_length']} characters ({(summary['summary_length']/summary['full_length']*100):.1f}%)")
                
            else:
                st.info("No document summaries were generated. Run the analysis with 'Generate Document Summaries' option enabled.")
    
    # Export options
    st.markdown("---")
    st.subheader("Export Results")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("Export to Excel"):
            try:
                # Create output buffer
                output = io.BytesIO()
                
                # Create Excel file with multiple sheets
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Write all reports to Excel with appropriate sheet names
                    for report_name, report_df in reports.items():
                        # Adjust sheet name if needed (Excel has a 31 character limit)
                        sheet_name = report_name[:31]
                        report_df.to_excel(writer, sheet_name=sheet_name, index=('correlation' in report_name or 'similarity' in report_name or 'matrix' in report_name))
                
                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{analysis_type}_analysis_results_{timestamp}.xlsx"
                
                st.download_button(
                    "📥 Download Excel Report",
                    data=output.getvalue(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error exporting to Excel: {str(e)}")
    
    with export_col2:
        if st.button("Export Detailed CSV Reports"):
            try:
                # Create a zip file containing all reports as CSVs
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for report_name, report_df in reports.items():
                        # Convert DataFrame to CSV
                        csv_buffer = io.StringIO()
                        report_df.to_csv(csv_buffer, index=('correlation' in report_name or 'similarity' in report_name or 'matrix' in report_name))
                        zip_file.writestr(f"{report_name}.csv", csv_buffer.getvalue())
                
                    # Add analysis parameters as JSON
                    param_dict = {
                        'analysis_type': analysis_type,
                        **{k: v for k, v in results['analysis_stats'].items() if not isinstance(v, (np.ndarray, pd.DataFrame)) and k not in ['svd', 'lda', 'vectorizer']}
                    }
                    zip_file.writestr("analysis_parameters.json", json.dumps(param_dict, indent=2))
                
                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{analysis_type}_analysis_results_{timestamp}.zip"
                
                zip_buffer.seek(0)
                st.download_button(
                    "📥 Download CSV Reports",
                    data=zip_buffer.getvalue(),
                    file_name=filename,
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"Error exporting to CSV: {str(e)}")
    
    # Additional government reporting options
    st.subheader("Government Reporting")
    gov_col1, gov_col2 = st.columns(2)
    
    with gov_col1:
        if st.button("Generate Summary Report"):
            try:
                # Create a summary report in Markdown format
                markdown_buffer = io.StringIO()
                
                # Add title and date
                markdown_buffer.write(f"# Document Analysis Summary Report\n\n")
                markdown_buffer.write(f"Date: {datetime.now().strftime('%d %B %Y')}\n\n")
                
                # Add analysis type and parameters
                markdown_buffer.write(f"## Analysis Type\n\n")
                if analysis_type == 'lsa':
                    markdown_buffer.write(f"Latent Semantic Analysis (LSA) with hierarchical clustering\n\n")
                    markdown_buffer.write(f"* Number of components: {results['analysis_stats']['n_components']}\n")
                    markdown_buffer.write(f"* Explained variance: {results['analysis_stats']['explained_variance']:.2%}\n")
                    markdown_buffer.write(f"* Number of clusters: {results['analysis_stats']['n_clusters']}\n")
                    markdown_buffer.write(f"* Silhouette score: {results['analysis_stats']['silhouette_score']:.3f}\n\n")
                else:
                    markdown_buffer.write(f"Latent Dirichlet Allocation (LDA) topic modeling\n\n")
                    markdown_buffer.write(f"* Number of topics: {results['analysis_stats']['n_topics']}\n")
                    markdown_buffer.write(f"* Topic coherence: {results['analysis_stats']['topic_coherence']:.3f}\n")
                    markdown_buffer.write(f"* Perplexity: {results['analysis_stats']['perplexity']:.2f}\n\n")
                
                # Add dataset information
                markdown_buffer.write(f"## Dataset Information\n\n")
                markdown_buffer.write(f"* Number of documents: {len(results['df_with_assignments'])}\n\n")
                
                # Add summary of topics/clusters
                if analysis_type == 'lsa':
                    markdown_buffer.write(f"## Cluster Summary\n\n")
                    for _, row in reports['cluster_summary'].iterrows():
                        markdown_buffer.write(f"### Cluster {int(row['Cluster'])}\n\n")
                        markdown_buffer.write(f"* Documents: {int(row['Documents'])} ({row['Percentage']:.1f}%)\n")
                        markdown_buffer.write(f"* Key terms: {row['Top Terms']}\n")
                        markdown_buffer.write(f"* Distinctive terms: {row['Distinctive Terms']}\n\n")
                else:
                    markdown_buffer.write(f"## Topic Summary\n\n")
                    for _, row in reports['topic_summary'].iterrows():
                        markdown_buffer.write(f"### Topic {int(row['Topic'])}\n\n")
                        markdown_buffer.write(f"* Prevalence: {row['Prevalence']:.1f}%\n")
                        markdown_buffer.write(f"* Key terms: {row['Top Terms']}\n")
                        markdown_buffer.write(f"* Distinctiveness: {row['Distinctiveness']:.3f}\n\n")
                
                # Add document summaries if available
                if 'document_summaries' in reports and not reports['document_summaries'].empty:
                    markdown_buffer.write(f"## Document Summaries\n\n")
                    for i, (group_id, group_summaries) in enumerate(reports['document_summaries'].groupby('topic_or_cluster')):
                        if i >= 3:  # Limit to first 3 groups for brevity
                            break
                        
                        if analysis_type == 'lsa':
                            markdown_buffer.write(f"### Cluster {int(group_id)} Summaries\n\n")
                        else:
                            markdown_buffer.write(f"### Topic {int(group_id)} Summaries\n\n")
                        
                        for j, summary in group_summaries.iterrows():
                            if j >= 3:  # Limit to first 3 summaries per group
                                break
                            markdown_buffer.write(f"#### {summary['document_title']}\n\n")
                            markdown_buffer.write(f"{summary['summary']}\n\n")
                
                # Add conclusion
                markdown_buffer.write(f"## Conclusion\n\n")
                if analysis_type == 'lsa':
                    markdown_buffer.write(f"The LSA analysis identified {results['analysis_stats']['n_clusters']} distinct document clusters with a silhouette score of {results['analysis_stats']['silhouette_score']:.3f}, indicating {'good' if results['analysis_stats']['silhouette_score'] > 0.5 else 'moderate'} cluster separation.\n\n")
                else:
                    markdown_buffer.write(f"The LDA topic modeling identified {results['analysis_stats']['n_topics']} distinct topics with a coherence score of {results['analysis_stats']['topic_coherence']:.3f}, indicating {'good' if results['analysis_stats']['topic_coherence'] > 0.5 else 'moderate'} topic coherence.\n\n")
                
                # Download button for Markdown report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{analysis_type}_summary_report_{timestamp}.md"
                
                st.download_button(
                    "📥 Download Summary Report",
                    data=markdown_buffer.getvalue(),
                    file_name=filename,
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Error generating summary report: {str(e)}")
    
    with gov_col2:
        if st.button("Generate Presentation"):
            try:
                # Create a simple HTML presentation
                html_buffer = io.StringIO()
                
                # Write HTML header
                html_buffer.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Document Analysis Presentation</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
                        .slide { height: 100vh; padding: 2em; page-break-after: always; }
                        h1 { color: #2c3e50; }
                        h2 { color: #3498db; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                        tr:nth-child(even) { background-color: #f9f9f9; }
                        .chart { height: 400px; border: 1px solid #ddd; margin: 20px 0; }
                        @media print { .slide { page-break-after: always; } }
                    </style>
                </head>
                <body>
                """)
                
                # Title slide
                html_buffer.write(f"""
                <div class="slide">
                    <h1>Document Analysis Report</h1>
                    <h2>{analysis_type.upper()} Analysis Results</h2>
                    <p>Date: {datetime.now().strftime('%d %B %Y')}</p>
                    <p>Total Documents: {len(results['df_with_assignments'])}</p>
                </div>
                """)
                
                # Analysis overview slide
                html_buffer.write(f"""
                <div class="slide">
                    <h1>Analysis Overview</h1>
                """)
                
                if analysis_type == 'lsa':
                    html_buffer.write(f"""
                    <h2>Latent Semantic Analysis (LSA)</h2>
                    <ul>
                        <li>Number of components: {results['analysis_stats']['n_components']}</li>
                        <li>Explained variance: {results['analysis_stats']['explained_variance']:.2%}</li>
                        <li>Number of clusters: {results['analysis_stats']['n_clusters']}</li>
                        <li>Silhouette score: {results['analysis_stats']['silhouette_score']:.3f}</li>
                    </ul>
                    <div class="chart"><p>Chart placeholder: Cluster distribution</p></div>
                    """)
                else:
                    html_buffer.write(f"""
                    <h2>Latent Dirichlet Allocation (LDA)</h2>
                    <ul>
                        <li>Number of topics: {results['analysis_stats']['n_topics']}</li>
                        <li>Topic coherence: {results['analysis_stats']['topic_coherence']:.3f}</li>
                        <li>Perplexity: {results['analysis_stats']['perplexity']:.2f}</li>
                    </ul>
                    <div class="chart"><p>Chart placeholder: Topic distribution</p></div>
                    """)
                
                html_buffer.write("""
                </div>
                """)
                
                # Group summary slide
                html_buffer.write(f"""
                <div class="slide">
                    <h1>{"Cluster" if analysis_type == 'lsa' else "Topic"} Summary</h1>
                    <table>
                        <tr>
                            <th>{"Cluster" if analysis_type == 'lsa' else "Topic"} ID</th>
                            <th>Size</th>
                            <th>{"%" if analysis_type == 'lsa' else "Prevalence"}</th>
                            <th>Key Terms</th>
                        </tr>
                """)
                
                summary_df = reports['cluster_summary'] if analysis_type == 'lsa' else reports['topic_summary']
                for _, row in summary_df.iterrows():
                    group_id = int(row['Cluster'] if analysis_type == 'lsa' else row['Topic'])
                    size = int(row['Documents'])
                    percentage = row['Percentage'] if analysis_type == 'lsa' else row['Prevalence']
                    terms = row['Top Terms'].split(', ')[:5]
                    
                    html_buffer.write(f"""
                    <tr>
                        <td>{group_id}</td>
                        <td>{size}</td>
                        <td>{percentage:.1f}%</td>
                        <td>{', '.join(terms)}</td>
                    </tr>
                    """)
                
                html_buffer.write("""
                    </table>
                </div>
                """)
                
                # Sample documents slide
                html_buffer.write(f"""
                <div class="slide">
                    <h1>Sample Documents</h1>
                """)
                
                if 'document_summaries' in reports and not reports['document_summaries'].empty:
                    summaries = reports['document_summaries']
                    group_id = summaries['topic_or_cluster'].iloc[0]
                    group_summaries = summaries[summaries['topic_or_cluster'] == group_id].head(3)
                    
                    html_buffer.write(f"""
                    <h2>{"Cluster" if analysis_type == 'lsa' else "Topic"} {int(group_id)}</h2>
                    """)
                    
                    for _, summary in group_summaries.iterrows():
                        html_buffer.write(f"""
                        <h3>{summary['document_title']}</h3>
                        <p>{summary['summary']}</p>
                        <hr>
                        """)
                else:
                    html_buffer.write("""
                    <p>No document summaries available.</p>
                    """)
                
                html_buffer.write("""
                </div>
                """)
                
                # Conclusion slide
                html_buffer.write(f"""
                <div class="slide">
                    <h1>Conclusions</h1>
                    <ul>
                """)
                
                if analysis_type == 'lsa':
                    html_buffer.write(f"""
                        <li>LSA analysis identified {results['analysis_stats']['n_clusters']} distinct document clusters</li>
                        <li>Silhouette score of {results['analysis_stats']['silhouette_score']:.3f} indicates {'good' if results['analysis_stats']['silhouette_score'] > 0.5 else 'moderate'} cluster separation</li>
                        <li>Analysis preserved {results['analysis_stats']['explained_variance']:.2%} of the original variance</li>
                        <li>The largest cluster contains {reports['cluster_summary']['Documents'].max()} documents ({reports['cluster_summary']['Percentage'].max():.1f}% of corpus)</li>
                    """)
                else:
                    html_buffer.write(f"""
                        <li>LDA modeling identified {results['analysis_stats']['n_topics']} distinct topics</li>
                        <li>Topic coherence of {results['analysis_stats']['topic_coherence']:.3f} indicates {'good' if results['analysis_stats']['topic_coherence'] > 0.5 else 'moderate'} topic definition</li>
                        <li>The most prevalent topic represents {reports['topic_summary']['Prevalence'].max():.1f}% of the corpus</li>
                        <li>Topic correlation analysis shows {'strong' if reports['topic_correlations'].values.max() > 0.7 else 'moderate' if reports['topic_correlations'].values.max() > 0.3 else 'minimal'} relationships between topics</li>
                    """)
                
                html_buffer.write("""
                    </ul>
                    <h2>Next Steps</h2>
                    <ul>
                        <li>Review individual document assignments for quality assurance</li>
                        <li>Consider further analysis of key themes identified</li>
                        <li>Use results to inform policy development and resource allocation</li>
                    </ul>
                </div>
                """)
                
                # Close HTML
                html_buffer.write("""
                </body>
                </html>
                """)
                
                # Download button for HTML presentation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{analysis_type}_presentation_{timestamp}.html"
                
                st.download_button(
                    "📥 Download Presentation",
                    data=html_buffer.getvalue(),
                    file_name=filename,
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Error generating presentation: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Critical Error")
        st.error(str(e))
        logging.critical(f"Application crash: {e}", exc_info=True)
