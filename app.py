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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score  # Added for semantic clustering
import networkx as nx
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from bs4 import BeautifulSoup, Tag
import json  # Added for JSON export functionality
from rank_bm25 import BM25Okapi  # Added for BM25 similarity

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
    """Clean text for modeling purposes with improved thematic focus"""
    if not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove common names and titles
        text = re.sub(r'\b(mr|mrs|ms|dr|prof|sir|lord|lady)\b\.?\s+\w+', '', text, flags=re.IGNORECASE)
        
        # Remove common location identifiers
        text = re.sub(r'\b(street|road|avenue|lane|drive|court|way|place|square|hospital|centre|center)\b', '', text, flags=re.IGNORECASE)
        
        # Remove UK city and county names (common ones)
        locations = r'\b(london|manchester|birmingham|liverpool|leeds|sheffield|bristol|newcastle|nottingham|cardiff|belfast|glasgow|edinburgh|surrey|kent|essex|sussex|yorkshire|lancashire)\b'
        text = re.sub(locations, '', text, flags=re.IGNORECASE)
        
        # Remove dates and times
        text = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text)
        
        # Remove phone numbers and postal codes
        text = re.sub(r'\b\d{3,4}[-\s]?\d{3,4}\b', '', text)
        text = re.sub(r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b', '', text, flags=re.IGNORECASE)
        
        # Remove reference numbers and case numbers
        text = re.sub(r'\b(?:ref|reference|case)(?:\s+no)?\.?\s*[-:\s]?\s*\w+[-\d]+\b', '', text, flags=re.IGNORECASE)
        
        # Remove any numbers or words containing numbers
        text = re.sub(r'\b\w*\d+\w*\b', '', text)
        
        # Remove specific document-related terms
        text = re.sub(r'\b(regulation|paragraph|section|subsection|article)\s+\d+\b', '', text, flags=re.IGNORECASE)
        
        # Remove common legal document terms
        text = re.sub(r'\b(coroner|inquest|hearing|evidence|witness|statement|report|dated|signed)\b', '', text, flags=re.IGNORECASE)
        
        # Remove single characters
        text = re.sub(r'\b[a-z]\b', '', text)
        
        # Remove special characters and multiple spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words (likely to be noise)
        text = ' '.join(word for word in text.split() if len(word) > 2)
        
        return text.strip()
    
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
    max_pages: Optional[int] = None
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
        
        # Apply max pages limit if specified
        if max_pages:
            total_pages = min(total_pages, max_pages)
            st.info(f"Limiting search to first {total_pages} pages")
        
        # Process each page
        for current_page in range(1, total_pages + 1):
            try:
                # Check if scraping should be stopped
                if hasattr(st.session_state, 'stop_scraping') and st.session_state.stop_scraping:
                    st.warning("Scraping stopped by user")
                    break
                
                # Update progress
                progress = (current_page - 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Processing page {current_page} of {total_pages}")
                
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


def render_scraping_tab():
    """Render the scraping tab with a clean 2x2 filter layout"""
    st.header("Scrape PFD Reports")

    # Initialize default values if not in session state
    if 'init_done' not in st.session_state:
        st.session_state.init_done = True
        st.session_state['search_keyword_default'] = "report"
        st.session_state['category_default'] = ""
        st.session_state['order_default'] = "relevance"
        st.session_state['max_pages_default'] = 0
    
    if 'scraped_data' in st.session_state and st.session_state.scraped_data is not None:
        st.success(f"Found {len(st.session_state.scraped_data)} reports")
        
        st.subheader("Results")
        st.dataframe(
            st.session_state.scraped_data,
            column_config={
                "URL": st.column_config.LinkColumn("Report Link"),
                "date_of_report": st.column_config.DateColumn("Date of Report", format="DD/MM/YYYY"),
                "categories": st.column_config.ListColumn("Categories")
            },
            hide_index=True
        )
        
        show_export_options(st.session_state.scraped_data, "scraped")

    # Create the search form with 2x2 layout
    with st.form("scraping_form"):
        # Create two rows with two columns each
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        # First row
        with row1_col1:
            search_keyword = st.text_input(
                "Search keywords:",
                value=st.session_state.get('search_keyword_default', "report"),
                key='search_keyword',
                help="Do not leave empty, use 'report' or another search term"
            )

        with row1_col2:
            category = st.selectbox(
                "PFD Report type:", 
                [""] + get_pfd_categories(), 
                index=0,
                key='category',
                format_func=lambda x: x if x else "Select a category"
            )

        # Second row
        with row2_col1:
            order = st.selectbox(
                "Sort by:", 
                ["relevance", "desc", "asc"],
                index=0,
                key='order',
                format_func=lambda x: {
                    "relevance": "Relevance",
                    "desc": "Newest first",
                    "asc": "Oldest first"
                }[x]
            )

        with row2_col2:
            max_pages = st.number_input(
                "Maximum pages to scrape:",
                min_value=0,
                value=st.session_state.get('max_pages_default', 0),
                key='max_pages',
                help="Enter 0 for all pages"
            )

        # Action buttons in a row
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            submitted = st.form_submit_button("Search Reports")
        with button_col2:
            stop_scraping = st.form_submit_button("Stop Scraping")
    
    # Handle stop scraping
    if stop_scraping:
        st.session_state.stop_scraping = True
        st.warning("Scraping will be stopped after the current page completes...")
        return
    
    if submitted:
        try:
            # Store search parameters in session state
            st.session_state.last_search_params = {
                'keyword': search_keyword,
                'category': category,
                'order': order
            }
            
            # Initialize stop_scraping flag
            st.session_state.stop_scraping = False

            # Set max pages
            max_pages_val = None if max_pages == 0 else max_pages
            
            # Perform scraping
            reports = scrape_pfd_reports(
                keyword=search_keyword,
                category=category if category else None,
                order=order,
                max_pages=max_pages_val
            )
            
            if reports:
                # Process the data
                df = pd.DataFrame(reports)
                df = process_scraped_data(df)
                
                # Store in session state
                st.session_state.scraped_data = df
                st.session_state.data_source = 'scraped'
                st.session_state.current_data = df
                
                # Trigger a rerun to refresh the page
                st.rerun()
            else:
                st.warning("No reports found matching your search criteria")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Scraping error: {e}")
            return False                           

def sort_reports(reports: List[Dict], order: str) -> List[Dict]:
    """Sort reports based on specified order"""
    if order == "date_desc":
        return sorted(reports, 
                     key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d"), 
                     reverse=True)
    elif order == "date_asc":
        return sorted(reports, 
                     key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d"))
    return reports

    

def plot_category_distribution(df: pd.DataFrame) -> None:
    """Plot category distribution"""
    all_cats = []
    for cats in df['categories'].dropna():
        if isinstance(cats, list):
            all_cats.extend(cats)
    
    cat_counts = pd.Series(all_cats).value_counts()
    
    fig = px.bar(
        x=cat_counts.index,
        y=cat_counts.values,
        title='Category Distribution',
        labels={'x': 'Category', 'y': 'Count'}
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Number of Reports",
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_coroner_areas(df: pd.DataFrame) -> None:
    """Plot coroner areas distribution"""
    area_counts = df['coroner_area'].value_counts().head(20)
    
    fig = px.bar(
        x=area_counts.index,
        y=area_counts.values,
        title='Top 20 Coroner Areas',
        labels={'x': 'Area', 'y': 'Count'}
    )
    
    fig.update_layout(
        xaxis_title="Coroner Area",
        yaxis_title="Number of Reports",
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def analyze_data_quality(df: pd.DataFrame) -> None:
    """Analyze and display data quality metrics for PFD reports"""
    
    # Calculate completeness metrics
    total_records = len(df)
    
    def calculate_completeness(field):
        if field not in df.columns:
            return 0
        non_empty = df[field].notna()
        if field == 'categories':
            non_empty = df[field].apply(lambda x: isinstance(x, list) and len(x) > 0)
        return (non_empty.sum() / total_records) * 100
    
    completeness_metrics = {
        'Title': calculate_completeness('Title'),
        'Content': calculate_completeness('Content'),
        'Date of Report': calculate_completeness('date_of_report'),
        'Deceased Name': calculate_completeness('deceased_name'),
        'Coroner Name': calculate_completeness('coroner_name'),
        'Coroner Area': calculate_completeness('coroner_area'),
        'Categories': calculate_completeness('categories')
    }
    
    # Calculate consistency metrics
    consistency_metrics = {
        'Title Format': (df['Title'].str.len() >= 10).mean() * 100,
        'Content Length': (df['Content'].str.len() >= 100).mean() * 100,
        'Date Format': df['date_of_report'].notna().mean() * 100,
        'Categories Format': df['categories'].apply(lambda x: isinstance(x, list)).mean() * 100
    }
    
    # Calculate PDF metrics
    pdf_columns = [col for col in df.columns if col.startswith('PDF_') and col.endswith('_Path')]
    reports_with_pdf = df[pdf_columns].notna().any(axis=1).sum()
    reports_with_multiple_pdfs = (df[pdf_columns].notna().sum(axis=1) > 1).sum()
    
    pdf_metrics = {
        'Reports with PDFs': (reports_with_pdf / total_records) * 100,
        'Reports with Multiple PDFs': (reports_with_multiple_pdfs / total_records) * 100
    }
    
    # Display metrics using Streamlit
    st.subheader("Data Quality Analysis")
    
    # Completeness
    st.markdown("### Field Completeness")
    completeness_df = pd.DataFrame(list(completeness_metrics.items()), 
                                 columns=['Field', 'Completeness %'])
    fig_completeness = px.bar(completeness_df, x='Field', y='Completeness %',
                             title='Field Completeness Analysis')
    fig_completeness.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_completeness, use_container_width=True)
    
    # Consistency
    st.markdown("### Data Consistency")
    consistency_df = pd.DataFrame(list(consistency_metrics.items()),
                                columns=['Metric', 'Consistency %'])
    fig_consistency = px.bar(consistency_df, x='Metric', y='Consistency %',
                            title='Data Consistency Analysis')
    fig_consistency.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_consistency, use_container_width=True)
    
    # PDF Analysis
    st.markdown("### PDF Attachment Analysis")
    pdf_df = pd.DataFrame(list(pdf_metrics.items()),
                         columns=['Metric', 'Percentage'])
    fig_pdf = px.bar(pdf_df, x='Metric', y='Percentage',
                     title='PDF Coverage Analysis')
    fig_pdf.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_pdf, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average Completeness",
            f"{np.mean(list(completeness_metrics.values())):.1f}%"
        )
        
    with col2:
        st.metric(
            "Average Consistency",
            f"{np.mean(list(consistency_metrics.values())):.1f}%"
        )
        
    with col3:
        st.metric(
            "PDF Coverage",
            f"{pdf_metrics['Reports with PDFs']:.1f}%"
        )
    
    # Detailed quality issues
    st.markdown("### Detailed Quality Issues")
    
    issues = []
    
    # Check for missing crucial fields
    for field, completeness in completeness_metrics.items():
        if completeness < 95:  # Less than 95% complete
            issues.append(f"- {field} is {completeness:.1f}% complete ({total_records - int(completeness * total_records / 100)} records missing)")
    
    # Check for consistency issues
    for metric, consistency in consistency_metrics.items():
        if consistency < 90:  # Less than 90% consistent
            issues.append(f"- {metric} shows {consistency:.1f}% consistency")
    
    # Check PDF coverage
    if pdf_metrics['Reports with PDFs'] < 90:
        issues.append(f"- {100 - pdf_metrics['Reports with PDFs']:.1f}% of reports lack PDF attachments")
    
    if issues:
        st.markdown("The following quality issues were identified:")
        for issue in issues:
            st.markdown(issue)
    else:
        st.success("No significant quality issues found in the dataset.")
            
def display_topic_network(lda, feature_names):
    """Display word similarity network with interactive filters"""
    #st.markdown("### Word Similarity Network")
    st.markdown("This network shows relationships between words based on their co-occurrence in documents.")
    
    # Store base network data in session state if not already present
    if 'network_data' not in st.session_state:
        # Get word counts across all documents
        word_counts = lda.components_.sum(axis=0)
        top_word_indices = word_counts.argsort()[:-100-1:-1]  # Store more words initially
        
        # Create word co-occurrence matrix
        word_vectors = normalize(lda.components_.T[top_word_indices])
        word_similarities = cosine_similarity(word_vectors)
        
        st.session_state.network_data = {
            'word_counts': word_counts,
            'top_word_indices': top_word_indices,
            'word_similarities': word_similarities,
            'feature_names': feature_names
        }
    
    # Network filters with keys to prevent rerun
    col1, col2, col3 = st.columns(3)
    with col1:
        min_similarity = st.slider(
            "Minimum Similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Higher values show stronger connections only",
            key="network_min_similarity"
        )
    with col2:
        max_words = st.slider(
            "Number of Words",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Number of most frequent words to show",
            key="network_max_words"
        )
    with col3:
        min_connections = st.slider(
            "Minimum Connections",
            min_value=1,
            max_value=10,
            value=5,
            help="Minimum number of connections per word",
            key="network_min_connections"
        )

    # Create network graph based on current filters
    G = nx.Graph()
    
    # Get stored data
    word_counts = st.session_state.network_data['word_counts']
    word_similarities = st.session_state.network_data['word_similarities']
    top_word_indices = st.session_state.network_data['top_word_indices'][:max_words]
    feature_names = st.session_state.network_data['feature_names']
    
    # Add nodes
    for idx, word_idx in enumerate(top_word_indices):
        G.add_node(idx, name=feature_names[word_idx], freq=float(word_counts[word_idx]))
    
    # Add edges based on current similarity threshold
    for i in range(len(top_word_indices)):
        for j in range(i+1, len(top_word_indices)):
            similarity = word_similarities[i, j]
            if similarity > min_similarity:
                G.add_edge(i, j, weight=float(similarity))
    
    # Filter nodes by minimum connections
    nodes_to_remove = []
    for node in G.nodes():
        if G.degree(node) < min_connections:
            nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)
    
    if len(G.nodes()) == 0:
        st.warning("No nodes match the current filter criteria. Try adjusting the filters.")
        return
    
    # Create visualization
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    
    # Create edge traces with varying thickness and color based on weight
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(
                width=weight * 3,
                color=f'rgba(100,100,100,{weight})'
            ),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create node trace with size based on frequency
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        freq = G.nodes[node]['freq']
        name = G.nodes[node]['name']
        connections = G.degree(node)
        node_text.append(f"{name}<br>Frequency: {freq:.0f}<br>Connections: {connections}")
        node_size.append(np.sqrt(freq) * 10)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=node_size,
            line=dict(width=1),
            color='lightblue',
            sizemode='area'
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=f"Word Network ({len(G.nodes())} words, {len(G.edges())} connections)",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add network statistics
    st.markdown("### Network Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Words", len(G.nodes()))
    with col2:
        st.metric("Number of Connections", len(G.edges()))
    with col3:
        if len(G.nodes()) > 0:
            density = 2 * len(G.edges()) / (len(G.nodes()) * (len(G.nodes()) - 1))
            st.metric("Network Density", f"{density:.2%}")


def get_top_words(model, feature_names, topic_idx, n_words=10):
    """Get top words for a given topic"""
    return [feature_names[i] for i in model.components_[topic_idx].argsort()[:-n_words-1:-1]]

def render_file_upload():
    """Render file upload section"""
    st.header("Upload Existing Data")
    
    # Generate unique key for the file uploader
    upload_key = f"file_uploader_{int(time.time() * 1000)}"
    
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file", 
        type=['csv', 'xlsx'],
        key=upload_key
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Process uploaded data
            df = process_scraped_data(df)
            
            # Clear any existing data first
            st.session_state.current_data = None
            st.session_state.scraped_data = None
            st.session_state.uploaded_data = None
            st.session_state.data_source = None
            
            # Then set new data
            st.session_state.uploaded_data = df.copy()
            st.session_state.data_source = 'uploaded'
            st.session_state.current_data = df.copy()
            
            st.success("File uploaded and processed successfully!")
            
            # Show the uploaded data
            st.subheader("Uploaded Data Preview")
            st.dataframe(
                df,
                column_config={
                    "URL": st.column_config.LinkColumn("Report Link"),
                    "date_of_report": st.column_config.DateColumn("Date of Report"),
                    "categories": st.column_config.ListColumn("Categories")
                },
                hide_index=True
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            logging.error(f"File upload error: {e}", exc_info=True)
            return False
    
    return False

def initialize_session_state():
    """Initialize all required session state variables"""
    # Initialize basic state variables if they don't exist
    if not hasattr(st.session_state, 'initialized'):
        # Clear all existing session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Set new session state variables
        st.session_state.data_source = None
        st.session_state.current_data = None
        st.session_state.scraped_data = None
        st.session_state.uploaded_data = None
        st.session_state.topic_model = None
        st.session_state.cleanup_done = False
        st.session_state.last_scrape_time = None
        st.session_state.last_upload_time = None
        st.session_state.analysis_filters = {
            'date_range': None,
            'selected_categories': None,
            'selected_areas': None
        }
        st.session_state.topic_model_settings = {
            'num_topics': 5,
            'max_features': 1000,
            'similarity_threshold': 0.3
        }
        st.session_state.initialized = True
    
    # Perform PDF cleanup if not done
    if not st.session_state.cleanup_done:
        try:
            pdf_dir = 'pdfs'
            os.makedirs(pdf_dir, exist_ok=True)
            
            current_time = time.time()
            cleanup_count = 0
            
            for file in os.listdir(pdf_dir):
                file_path = os.path.join(pdf_dir, file)
                try:
                    if os.path.isfile(file_path):
                        if os.stat(file_path).st_mtime < current_time - 86400:
                            os.remove(file_path)
                            cleanup_count += 1
                except Exception as e:
                    logging.warning(f"Error cleaning up file {file_path}: {e}")
                    continue
            
            if cleanup_count > 0:
                logging.info(f"Cleaned up {cleanup_count} old PDF files")
        except Exception as e:
            logging.error(f"Error during PDF cleanup: {e}")
        finally:
            st.session_state.cleanup_done = True
            
def validate_data(data: pd.DataFrame, purpose: str = "analysis") -> Tuple[bool, str]:
    """
    Validate data for different purposes
    
    Args:
        data: DataFrame to validate
        purpose: Purpose of validation ('analysis' or 'topic_modeling')
        
    Returns:
        tuple: (is_valid, message)
    """
    if data is None:
        return False, "No data available. Please scrape or upload data first."
    
    if not isinstance(data, pd.DataFrame):
        return False, "Invalid data format. Expected pandas DataFrame."
    
    if len(data) == 0:
        return False, "Dataset is empty."
        
    if purpose == "analysis":
        required_columns = ['date_of_report', 'categories', 'coroner_area']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
            
    elif purpose == "topic_modeling":
        if 'Content' not in data.columns:
            return False, "Missing required column: Content"
            
        valid_docs = data['Content'].dropna().str.strip().str.len() > 0
        if valid_docs.sum() < 2:
            return False, "Not enough valid documents found. Please ensure you have documents with text content."
            
    # Add type checking for critical columns
    if 'date_of_report' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date_of_report']):
        try:
            pd.to_datetime(data['date_of_report'])
        except Exception:
            return False, "Invalid date format in date_of_report column."
            
    if 'categories' in data.columns:
        if not data['categories'].apply(lambda x: isinstance(x, (list, type(None)))).all():
            return False, "Categories must be stored as lists or None values."
    
    return True, "Data is valid"
    
def is_response(row: pd.Series) -> bool:
    """
    Check if a report is a response document based on its metadata and content
    
    Args:
        row: DataFrame row containing report data
        
    Returns:
        bool: True if document is a response, False otherwise
    """
    try:
        # Check PDF names for response indicators
        pdf_response = False
        for i in range(1, 10):  # Check PDF_1 to PDF_9
            pdf_name = str(row.get(f'PDF_{i}_Name', '')).lower()
            if 'response' in pdf_name or 'reply' in pdf_name:
                pdf_response = True
                break
        
        # Check title for response indicators
        title = str(row.get('Title', '')).lower()
        title_response = any(word in title for word in ['response', 'reply', 'answered'])
        
        # Check content for response indicators
        content = str(row.get('Content', '')).lower()
        content_response = any(phrase in content for phrase in [
            'in response to',
            'responding to',
            'reply to',
            'response to',
            'following the regulation 28'
        ])
        
        return pdf_response or title_response or content_response
        
    except Exception as e:
        logging.error(f"Error checking response type: {e}")
        return False

def plot_timeline(df: pd.DataFrame) -> None:
    """Plot timeline of reports with improved formatting"""
    timeline_data = df.groupby(
        pd.Grouper(key='date_of_report', freq='M')
    ).size().reset_index()
    timeline_data.columns = ['Date', 'Count']
    
    fig = px.line(timeline_data, x='Date', y='Count',
                  title='Reports Timeline',
                  labels={'Count': 'Number of Reports'})
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Reports",
        hovermode='x unified',
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,  # Integer steps
            rangemode='nonnegative'  # Ensure y-axis starts at 0 or above
        ),
        xaxis=dict(
            tickformat='%B %Y',  # Month Year format
            tickangle=45
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_monthly_distribution(df: pd.DataFrame) -> None:
    """Plot monthly distribution with improved formatting"""
    # Format dates as Month Year
    df['month_year'] = df['date_of_report'].dt.strftime('%B %Y')
    monthly_counts = df['month_year'].value_counts().sort_index()
    
    fig = px.bar(
        x=monthly_counts.index,
        y=monthly_counts.values,
        labels={'x': 'Month', 'y': 'Number of Reports'},
        title='Monthly Distribution of Reports'
    )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Reports",
        xaxis_tickangle=45,
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,  # Integer steps
            rangemode='nonnegative'
        ),
        bargap=0.2
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_yearly_comparison(df: pd.DataFrame) -> None:
    """Plot year-over-year comparison with improved formatting"""
    yearly_counts = df['date_of_report'].dt.year.value_counts().sort_index()
    
    fig = px.line(
        x=yearly_counts.index.astype(int),  # Convert to integer years
        y=yearly_counts.values,
        markers=True,
        labels={'x': 'Year', 'y': 'Number of Reports'},
        title='Year-over-Year Report Volumes'
    )
    
    # Calculate appropriate y-axis maximum
    max_count = yearly_counts.max()
    y_max = max_count + (1 if max_count < 10 else 2)  # Add some padding
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Reports",
        xaxis=dict(
            tickmode='linear',
            tick0=yearly_counts.index.min(),
            dtick=1,  # Show every year
            tickformat='d'  # Format as integer
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,  # Integer steps
            range=[0, y_max]
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def export_to_excel(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to Excel bytes with proper formatting
    """
    try:
        if df is None or len(df) == 0:
            raise ValueError("No data available to export")
            
        # Create clean copy for export
        df_export = df.copy()
        
        # Format dates to UK format
        if 'date_of_report' in df_export.columns:
            df_export['date_of_report'] = df_export['date_of_report'].dt.strftime('%d/%m/%Y')
            
        # Handle list columns (like categories)
        for col in df_export.columns:
            if df_export[col].dtype == 'object':
                df_export[col] = df_export[col].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x) if pd.notna(x) else ''
                )
        
        # Create output buffer
        output = io.BytesIO()
        
        # Write to Excel
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='Reports', index=False)
            
            # Get the worksheet
            worksheet = writer.sheets['Reports']
            
            # Auto-adjust column widths
            for idx, col in enumerate(df_export.columns, 1):
                max_length = max(
                    df_export[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                adjusted_width = min(max_length + 2, 50)
                column_letter = get_column_letter(idx)  # Use openpyxl's get_column_letter
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Add filters to header row
            worksheet.auto_filter.ref = worksheet.dimensions
            
            # Freeze the header row
            worksheet.freeze_panes = 'A2'
        
        # Get the bytes value
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        logging.error(f"Error exporting to Excel: {e}", exc_info=True)
        raise Exception(f"Failed to export data to Excel: {str(e)}")


def show_export_options(df: pd.DataFrame, prefix: str):
    """Show export options for the data with descriptive filename and unique keys"""
    try:
        st.subheader("Export Options")
        
        # Generate timestamp and random suffix for unique keys
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        unique_id = f"{timestamp}_{random_suffix}"
        filename = f"pfd_reports_{prefix}_{timestamp}"
        
        col1, col2 = st.columns(2)
        
        # CSV Export
        with col1:
            try:
                # Create export copy with formatted dates
                df_csv = df.copy()
                if 'date_of_report' in df_csv.columns:
                    df_csv['date_of_report'] = df_csv['date_of_report'].dt.strftime('%d/%m/%Y')
                
                csv = df_csv.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Reports (CSV)",
                    csv,
                    f"{filename}.csv",
                    "text/csv",
                    key=f"download_csv_{prefix}_{unique_id}"
                )
            except Exception as e:
                st.error(f"Error preparing CSV export: {str(e)}")
        
        # Excel Export
        with col2:
            try:
                excel_data = export_to_excel(df)
                st.download_button(
                    "📥 Download Reports (Excel)",
                    excel_data,
                    f"{filename}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_excel_{prefix}_{unique_id}"
                )
            except Exception as e:
                st.error(f"Error preparing Excel export: {str(e)}")
        
        # PDF Download
        if any(col.startswith('PDF_') and col.endswith('_Path') for col in df.columns):
            st.subheader("Download PDFs")
            if st.button(f"Download all PDFs", key=f"pdf_button_{prefix}_{unique_id}"):
                with st.spinner("Preparing PDF download..."):
                    try:
                        pdf_zip_path = f"{filename}_pdfs.zip"
                        
                        with zipfile.ZipFile(pdf_zip_path, 'w') as zipf:
                            pdf_columns = [col for col in df.columns if col.startswith('PDF_') and col.endswith('_Path')]
                            added_files = set()
                            
                            for col in pdf_columns:
                                paths = df[col].dropna()
                                for pdf_path in paths:
                                    if pdf_path and os.path.exists(pdf_path) and pdf_path not in added_files:
                                        zipf.write(pdf_path, os.path.basename(pdf_path))
                                        added_files.add(pdf_path)
                        
                        with open(pdf_zip_path, 'rb') as f:
                            st.download_button(
                                "📦 Download All PDFs (ZIP)",
                                f.read(),
                                pdf_zip_path,
                                "application/zip",
                                key=f"download_pdfs_zip_{prefix}_{unique_id}"
                            )
                        
                        # Cleanup zip file
                        os.remove(pdf_zip_path)
                    except Exception as e:
                        st.error(f"Error preparing PDF download: {str(e)}")
                        
    except Exception as e:
        st.error(f"Error setting up export options: {str(e)}")
        logging.error(f"Export options error: {e}", exc_info=True)
        
def render_analysis_tab(data: pd.DataFrame = None):
    """Render the analysis tab with improved filters, file upload functionality, and analysis sections"""
    st.header("Reports Analysis")
    
    # Add file upload section at the top
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file", 
        type=['csv', 'xlsx'],
        help="Upload previously exported data"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Process uploaded data
            data = process_scraped_data(data)
            st.success("File uploaded and processed successfully!")
            
            # Update session state
            st.session_state.uploaded_data = data.copy()
            st.session_state.data_source = 'uploaded'
            st.session_state.current_data = data.copy()
        
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            logging.error(f"File upload error: {e}", exc_info=True)
            return
    
    # Use either uploaded data or passed data
    if data is None:
        data = st.session_state.get('current_data')
    
    if data is None or len(data) == 0:
        st.warning("No data available. Please upload a file or scrape reports first.")
        return
        
    try:
        # Get date range for the data
        min_date = data['date_of_report'].min().date()
        max_date = data['date_of_report'].max().date()
        
        # Filters sidebar
        with st.sidebar:
            st.header("Filters")
            
            # Date Range
            with st.expander("📅 Date Range", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "From",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="start_date_filter",
                        format="DD/MM/YYYY"
                    )
                with col2:
                    end_date = st.date_input(
                        "To",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="end_date_filter",
                        format="DD/MM/YYYY"
                    )
            
            # Document Type Filter
            doc_type = st.multiselect(
                "Document Type",
                ["Report", "Response"],
                default=["Report", "Response"],
                key="doc_type_filter",
                help="Filter by document type"
            )
            
            # Reference Number
            ref_numbers = sorted(data['ref'].dropna().unique())
            selected_refs = st.multiselect(
                "Reference Numbers",
                options=ref_numbers,
                key="ref_filter"
            )
            
            # Deceased Name
            deceased_search = st.text_input(
                "Deceased Name",
                key="deceased_filter",
                help="Enter partial or full name"
            )
            
            # Coroner Name
            coroner_names = sorted(data['coroner_name'].dropna().unique())
            selected_coroners = st.multiselect(
                "Coroner Names",
                options=coroner_names,
                key="coroner_filter"
            )
            
            # Coroner Area
            coroner_areas = sorted(data['coroner_area'].dropna().unique())
            selected_areas = st.multiselect(
                "Coroner Areas",
                options=coroner_areas,
                key="areas_filter"
            )
            
            # Categories
            all_categories = set()
            for cats in data['categories'].dropna():
                if isinstance(cats, list):
                    all_categories.update(cats)
            selected_categories = st.multiselect(
                "Categories",
                options=sorted(all_categories),
                key="categories_filter"
            )
            
            # Reset Filters Button
            if st.button("🔄 Reset Filters"):
                for key in st.session_state:
                    if key.endswith('_filter'):
                        del st.session_state[key]
                st.rerun()

        # Apply filters
        filtered_df = data.copy()

        # Date filter
        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['date_of_report'].dt.date >= start_date) &
                (filtered_df['date_of_report'].dt.date <= end_date)
            ]

        # Document type filter
        if doc_type:
            filtered_df = filtered_df[filtered_df.apply(is_response, axis=1)]

        # Reference number filter
        if selected_refs:
            filtered_df = filtered_df[filtered_df['ref'].isin(selected_refs)]

        if deceased_search:
            filtered_df = filtered_df[
                filtered_df['deceased_name'].fillna('').str.contains(
                    deceased_search, 
                    case=False, 
                    na=False
                )
            ]

        if selected_coroners:
            filtered_df = filtered_df[filtered_df['coroner_name'].isin(selected_coroners)]

        if selected_areas:
            filtered_df = filtered_df[filtered_df['coroner_area'].isin(selected_areas)]

        if selected_categories:
            filtered_df = filtered_df[
                filtered_df['categories'].apply(
                    lambda x: bool(x) and any(cat in x for cat in selected_categories)
                )
            ]

        # Show active filters
        active_filters = []
        if start_date != min_date or end_date != max_date:
            active_filters.append(f"Date: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
        if doc_type and doc_type != ["Report", "Response"]:
            active_filters.append(f"Document Types: {', '.join(doc_type)}")
        if selected_refs:
            active_filters.append(f"References: {', '.join(selected_refs)}")
        if deceased_search:
            active_filters.append(f"Deceased name contains: {deceased_search}")
        if selected_coroners:
            active_filters.append(f"Coroners: {', '.join(selected_coroners)}")
        if selected_areas:
            active_filters.append(f"Areas: {', '.join(selected_areas)}")
        if selected_categories:
            active_filters.append(f"Categories: {', '.join(selected_categories)}")

        if active_filters:
            st.info("Active filters:\n" + "\n".join(f"• {filter_}" for filter_ in active_filters))

        # Display results
        st.subheader("Results")
        st.write(f"Showing {len(filtered_df)} of {len(data)} reports")

        if len(filtered_df) > 0:
            # Display the dataframe
            st.dataframe(
                filtered_df,
                column_config={
                    "URL": st.column_config.LinkColumn("Report Link"),
                    "date_of_report": st.column_config.DateColumn(
                        "Date of Report",
                        format="DD/MM/YYYY"
                    ),
                    "categories": st.column_config.ListColumn("Categories"),
                    "Document Type": st.column_config.TextColumn(
                        "Document Type",
                        help="Type of document based on PDF filename"
                    )
                },
                hide_index=True
            )

            # Create tabs for different analyses
            st.markdown("---")
            quality_tab, temporal_tab, distribution_tab = st.tabs([
                "📊 Data Quality Analysis",
                "📅 Temporal Analysis",
                "📍 Distribution Analysis"
            ])

            # Data Quality Analysis Tab
            with quality_tab:
                analyze_data_quality(filtered_df)

            # Temporal Analysis Tab
            with temporal_tab:
                # Timeline of reports
                st.subheader("Reports Timeline")
                plot_timeline(filtered_df)
                
                # Monthly distribution
                st.subheader("Monthly Distribution")
                plot_monthly_distribution(filtered_df)
                
                # Year-over-year comparison
                st.subheader("Year-over-Year Comparison")
                plot_yearly_comparison(filtered_df)
                
                # Seasonal patterns
                st.subheader("Seasonal Patterns")
                seasonal_counts = filtered_df['date_of_report'].dt.month.value_counts().sort_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                fig = px.line(
                    x=month_names,
                    y=[seasonal_counts.get(i, 0) for i in range(1, 13)],
                    markers=True,
                    labels={'x': 'Month', 'y': 'Number of Reports'},
                    title='Seasonal Distribution of Reports'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Distribution Analysis Tab
            with distribution_tab:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Reports by Category")
                    plot_category_distribution(filtered_df)
                with col2:
                    st.subheader("Reports by Coroner Area")
                    plot_coroner_areas(filtered_df)

            # Export options
            st.markdown("---")
            st.subheader("Export Options")
            col1, col2 = st.columns(2)
            
            # CSV Export
            with col1:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "filtered_reports.csv",
                    "text/csv"
                )
            
            # Excel Export
            with col2:
                excel_data = export_to_excel(filtered_df)
                st.download_button(
                    "📥 Download Results (Excel)",
                    excel_data,
                    "filtered_reports.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("No data matches the selected filters.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Analysis error: {e}", exc_info=True)

def extract_advanced_topics(
    data: pd.DataFrame, 
    num_topics: int = 5, 
    max_features: int = 1000, 
    min_df: int = 2, 
    n_iterations: int = 20,
    min_similarity: float = 0.9
) -> Tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    """
    Advanced topic modeling with comprehensive preprocessing and error handling
    
    Args:
        data (pd.DataFrame): Input DataFrame containing documents
        num_topics (int): Number of topics to extract
        max_features (int): Maximum number of features to use
        min_df (int): Minimum document frequency for terms
        n_iterations (int): Maximum number of iterations for LDA
        min_similarity (float): Minimum similarity threshold for the word similarity network
    
    Returns:
        Tuple containing LDA model, vectorizer, and document-topic distribution
    """
    try:
        # Extensive logging
        logging.info(f"Starting topic modeling with {len(data)} documents")
        logging.info(f"Parameters: topics={num_topics}, max_features={max_features}, min_df={min_df}, min_similarity={min_similarity}")

        # Validate input data
        if data is None or len(data) == 0:
            raise ValueError("No data provided for topic modeling")

        # Remove duplicate documents based on content
        def prepare_document(doc: str) -> str:
            """Clean and prepare individual documents"""
            if pd.isna(doc):
                return None
            
            # Aggressive text cleaning
            cleaned_doc = clean_text_for_modeling(str(doc))
            
            # Minimum length check
            return cleaned_doc if len(cleaned_doc.split()) > 3 else None

        # Process documents
        documents = data['Content'].apply(prepare_document).dropna().unique().tolist()
        
        logging.info(f"Processed {len(documents)} unique valid documents")

        # Validate document count
        if len(documents) < num_topics:
            adjusted_topics = max(2, len(documents) // 2)
            logging.warning(f"Not enough documents for {num_topics} topics. Adjusting to {adjusted_topics}")
            num_topics = adjusted_topics

        # Vectorization with robust settings
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min(min_df, max(2, len(documents) // 10)),  # Adaptive min_df
            max_df=0.95,
            stop_words='english'
        )

        # Create document-term matrix
        dtm = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        logging.info(f"Document-term matrix shape: {dtm.shape}")
        logging.info(f"Number of features: {len(feature_names)}")

        # LDA with robust parameters
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method='online',
            learning_offset=50.,
            max_iter=n_iterations,
            doc_topic_prior=None,  # Let scikit-learn auto-estimate
            topic_word_prior=None  # Let scikit-learn auto-estimate
        )

        # Fit LDA model
        doc_topics = lda_model.fit_transform(dtm)
        
        # Add logging of results
        logging.info("Topic modeling completed successfully")
        logging.info(f"Document-topic matrix shape: {doc_topics.shape}")

        return lda_model, vectorizer, doc_topics
        
    except Exception as e:
        logging.error(f"Topic modeling failed: {e}", exc_info=True)
        raise

def is_response(row: pd.Series) -> bool:
    """
    Check if a document is a response based on its metadata and content
    """
    try:
        # Check PDF types first (most reliable)
        for i in range(1, 5):  # Check PDF_1 to PDF_4
            pdf_type = str(row.get(f'PDF_{i}_Type', '')).lower()
            if pdf_type == 'response':
                return True
        
        # Check PDF names as backup
        for i in range(1, 5):
            pdf_name = str(row.get(f'PDF_{i}_Name', '')).lower()
            if 'response' in pdf_name or 'reply' in pdf_name:
                return True
        
        # Check title and content as final fallback
        title = str(row.get('Title', '')).lower()
        if any(word in title for word in ['response', 'reply', 'answered']):
            return True
            
        content = str(row.get('Content', '')).lower()
        return any(phrase in content for phrase in [
            'in response to',
            'responding to',
            'reply to',
            'response to',
            'following the regulation 28',
            'following receipt of the regulation 28'
        ])
        
    except Exception as e:
        logging.error(f"Error checking response type: {e}")
        return False

def normalize_category(category: str) -> str:
    """Normalize category string for consistent matching"""
    if not category:
        return ""
    # Convert to lowercase and remove extra whitespace
    normalized = " ".join(str(category).lower().split())
    # Remove common separators and special characters
    normalized = re.sub(r'[,;|•·⋅‣⁃▪▫–—-]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def match_category(category: str, standard_categories: List[str]) -> Optional[str]:
    """Match a category against standard categories with fuzzy matching"""
    if not category:
        return None
    
    normalized_category = normalize_category(category)
    normalized_standards = {normalize_category(cat): cat for cat in standard_categories}
    
    # Try exact match first
    if normalized_category in normalized_standards:
        return normalized_standards[normalized_category]
    
    # Try partial matching
    for norm_std, original_std in normalized_standards.items():
        if normalized_category in norm_std or norm_std in normalized_category:
            return original_std
    
    return category  # Return original if no match found

def extract_categories(category_text: str, standard_categories: List[str]) -> List[str]:
    """Extract and normalize categories from raw text"""
    if not category_text:
        return []
    
    # Replace common separators with a standard one
    category_text = re.sub(r'\s*[|,;]\s*', '|', category_text)
    category_text = re.sub(r'[•·⋅‣⁃▪▫–—-]\s*', '|', category_text)
    category_text = re.sub(r'\s{2,}', '|', category_text)
    category_text = re.sub(r'\n+', '|', category_text)
    
    # Split and clean categories
    raw_categories = category_text.split('|')
    cleaned_categories = []
    
    for cat in raw_categories:
        cleaned_cat = clean_text(cat).strip()
        if cleaned_cat and not re.match(r'^[\s|,;]+$', cleaned_cat):
            matched_cat = match_category(cleaned_cat, standard_categories)
            if matched_cat:
                cleaned_categories.append(matched_cat)
    
    # Remove duplicates while preserving order
    seen = set()
    return [x for x in cleaned_categories if not (normalize_category(x) in seen or seen.add(normalize_category(x)))]
    

def filter_by_categories(df: pd.DataFrame, selected_categories: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame by categories with fuzzy matching
    
    Args:
        df: DataFrame containing 'categories' column
        selected_categories: List of categories to filter by
    
    Returns:
        Filtered DataFrame
    """
    if not selected_categories:
        return df
    
    def has_matching_category(row_categories):
        if not isinstance(row_categories, list):
            return False
        
        # Normalize categories for comparison
        row_cats_norm = [cat.lower().strip() for cat in row_categories if cat]
        selected_cats_norm = [cat.lower().strip() for cat in selected_categories if cat]
        
        for row_cat in row_cats_norm:
            for selected_cat in selected_cats_norm:
                # Check for partial matches in either direction
                if row_cat in selected_cat or selected_cat in row_cat:
                    return True
        return False
    
    return df[df['categories'].apply(has_matching_category)]






def filter_by_document_type(df: pd.DataFrame, doc_types: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame based on document types
    """
    if not doc_types:
        return df
    
    filtered_df = df.copy()
    is_response_mask = filtered_df.apply(is_response, axis=1)
    
    if len(doc_types) == 1:
        if "Response" in doc_types:
            return filtered_df[is_response_mask]
        elif "Report" in doc_types:
            return filtered_df[~is_response_mask]
    
    return filtered_df

def extract_topic_insights(lda_model, vectorizer, doc_topics, data: pd.DataFrame):
    """Extract insights from topic modeling results with improved error handling"""
    try:
        # Get feature names and initialize results
        feature_names = vectorizer.get_feature_names_out()
        topics_data = []
        
        # Ensure we have valid data
        valid_data = data[data['Content'].notna()].copy()
        if len(valid_data) == 0:
            raise ValueError("No valid documents found in dataset")

        # Calculate document frequencies with error handling
        doc_freq = {}
        for doc in valid_data['Content']:
            try:
                words = set(clean_text_for_modeling(str(doc)).split())
                for word in words:
                    doc_freq[word] = doc_freq.get(word, 0) + 1
            except Exception as e:
                logging.warning(f"Error processing document: {str(e)}")
                continue

        # Process each topic
        for idx, topic in enumerate(lda_model.components_):
            try:
                # Get top words
                top_word_indices = topic.argsort()[:-50-1:-1]
                topic_words = []
                
                for i in top_word_indices:
                    word = feature_names[i]
                    if len(word) > 1:
                        weight = float(topic[i])
                        topic_words.append({
                            'word': word,
                            'weight': weight,
                            'count': doc_freq.get(word, 0),
                            'documents': doc_freq.get(word, 0)
                        })

                # Get representative documents
                doc_scores = doc_topics[:, idx]
                top_doc_indices = doc_scores.argsort()[:-11:-1]
                
                related_docs = []
                for doc_idx in top_doc_indices:
                    if doc_scores[doc_idx] > 0.01:  # At least 1% relevance
                        if doc_idx < len(valid_data):
                            doc_row = valid_data.iloc[doc_idx]
                            doc_content = str(doc_row.get('Content', ''))
                            
                            related_docs.append({
                                'title': doc_row.get('Title', ''),
                                'date': doc_row.get('date_of_report', ''),
                                'relevance': float(doc_scores[doc_idx]),
                                'summary': doc_content[:300] + '...' if len(doc_content) > 300 else doc_content
                            })

                # Generate topic description
                meaningful_words = [word['word'] for word in topic_words[:5]]
                label = ' & '.join(meaningful_words[:3]).title()
                
                topic_data = {
                    'id': idx,
                    'label': label,
                    'description': f"Topic frequently mentions: {', '.join(meaningful_words)}",
                    'words': topic_words,
                    'representativeDocs': related_docs,
                    'prevalence': round((doc_scores > 0.05).mean() * 100, 1)
                }
                
                topics_data.append(topic_data)
                
            except Exception as e:
                logging.error(f"Error processing topic {idx}: {str(e)}")
                continue
        
        if not topics_data:
            raise ValueError("No valid topics could be extracted")
            
        return topics_data
        
    except Exception as e:
        logging.error(f"Error extracting topic insights: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract topic insights: {str(e)}")
        


def display_topic_analysis(topics_data):
    """Display topic analysis results"""
    for topic in topics_data:
        st.markdown(f"## Topic {topic['id'] + 1}: {topic['label']}")
        st.markdown(f"**Prevalence:** {topic['prevalence']}% of documents")
        st.markdown(f"**Description:** {topic['description']}")
        
        # Display key terms
        st.markdown("### Key Terms")
        terms_data = pd.DataFrame(topic['words'])
        if not terms_data.empty:
            st.dataframe(
                terms_data,
                column_config={
                    'word': st.column_config.TextColumn('Term'),
                    'weight': st.column_config.NumberColumn(
                        'Weight',
                        format="%.4f"
                    ),
                    'count': st.column_config.NumberColumn('Document Count'),
                },
                hide_index=True
            )
        
        # Display representative documents
        st.markdown("### Representative Documents")
        for doc in topic['representativeDocs']:
            with st.expander(f"{doc['title']} (Relevance: {doc['relevance']:.2%})"):
                st.markdown(f"**Date:** {doc['date']}")
                st.markdown(doc['summary'])
        
        st.markdown("---")
       
# Initialize NLTK resources
def initialize_nltk():
    """Initialize required NLTK resources with error handling"""
    try:
        resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                if resource == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif resource == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif resource == 'averaged_perceptron_tagger':
                    nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download(resource)
    except Exception as e:
        logging.error(f"Error initializing NLTK resources: {e}")
        raise

def create_document_identifier(row: pd.Series) -> str:
    """Create a unique identifier for a document based on its title and reference number"""
    title = str(row.get('Title', '')).strip()
    ref = str(row.get('ref', '')).strip()
    deceased = str(row.get('deceased_name', '')).strip()
    
    # Combine multiple fields to create a unique identifier
    identifier = f"{title}_{ref}_{deceased}"
    return identifier

def deduplicate_documents(data: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate documents while preserving unique entries"""
    # Create unique identifiers
    data['doc_id'] = data.apply(create_document_identifier, axis=1)
    
    # Keep first occurrence of each unique document
    deduped_data = data.drop_duplicates(subset=['doc_id'])
    
    # Drop the temporary identifier column
    deduped_data = deduped_data.drop(columns=['doc_id'])
    
    return deduped_data

def perform_semantic_clustering(data: pd.DataFrame, method: str = 'tfidf',  
                             min_cluster_size: int = 2, max_features: int = 5000, 
                             min_df: float = 0.01, max_df: float = 0.95) -> Dict:
    """
    Perform semantic clustering with BM25 or TF-IDF options 
    """
    try:
        # Initialize NLTK resources
        initialize_nltk()
        
        # Validate and process input data
        if 'Content' not in data.columns:
            raise ValueError("Input data must contain 'Content' column")
            
        processed_texts = data['Content'].apply(clean_text_for_modeling)
        valid_mask = processed_texts.notna() & (processed_texts != '')
        processed_texts = processed_texts[valid_mask]
        
        if len(processed_texts) == 0:
            raise ValueError("No valid text content found after preprocessing")
        
        # Keep the original data for display
        display_data = data[valid_mask].copy()
            
        # Calculate optimal parameters based on dataset size
        n_docs = len(processed_texts)
        min_clusters = max(2, min(3, n_docs // 10))
        max_clusters = max(3, min(5, n_docs // 7))
        
        # Adjust feature extraction parameters
        max_features = min(5000, n_docs * 200)
        min_df = max(2, int(0.05 * n_docs))  # At least 5% of documents
        max_df = min(0.95, 1.0 - (min_cluster_size / n_docs))  # Leave room for unique terms
        
        if method == 'tfidf':
            # Vectorization with TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,  
                max_df=max_df,
                stop_words='english',
                ngram_range=(1, 2),
                norm='l2'
            )
            doc_vectors = vectorizer.fit_transform(processed_texts)



        elif method == 'bm25':
            # Vectorization with BM25
            corpus = [text.split() for text in processed_texts]
            bm25 = BM25Okapi(corpus)
            
            # Get BM25 scores for each document
            doc_vectors = []
            for doc in corpus:
                scores = bm25.get_scores(doc)
                doc_vectors.append(scores)
            
            # Convert list of scores to dense numpy array
            doc_vectors = np.array(doc_vectors)



        else:
            raise ValueError(f"Invalid method: {method}")
        
        # Try different clustering configurations
        best_score = -1
        best_labels = None
        best_n_clusters = min_clusters
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                # Try hierarchical clustering
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='euclidean',
                    linkage='ward'
                )
                
                labels = clustering.fit_predict(doc_vectors)
                
                # Verify cluster sizes
                cluster_sizes = np.bincount(labels)
                if min(cluster_sizes) < min_cluster_size:
                    continue
                
                # Check cluster balance
                size_ratio = min(cluster_sizes) / max(cluster_sizes)
                if size_ratio < 0.2:  # Require more balanced clusters
                    continue
                
                # Calculate clustering quality
                score = silhouette_score(doc_vectors, labels, metric='euclidean')
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_n_clusters = n_clusters
                    
            except Exception as e:
                logging.warning(f"Clustering attempt failed for k={n_clusters}: {str(e)}")
                continue

        # If no good clustering found, try alternative approach
        if best_labels is None:
            best_n_clusters = 2  # Minimum viable clustering
            clustering = AgglomerativeClustering(
                n_clusters=best_n_clusters,
                metric='euclidean',
                linkage='ward'
            )
            best_labels = clustering.fit_predict(doc_vectors)

        # Calculate similarities based on chosen method
        if method == 'tfidf':
            similarity_matrix = cosine_similarity(doc_vectors)
        else:  
            similarity_matrix = cosine_similarity(doc_vectors)

        # Extract cluster information
        clusters = []
        for cluster_id in range(best_n_clusters):
            cluster_indices = np.where(best_labels == cluster_id)[0]
            
            # Skip if cluster is too small
            if len(cluster_indices) < min_cluster_size:
                continue
                
            # Get important terms 
            if method == 'tfidf':
                centroid_vector = np.mean(doc_vectors[cluster_indices].toarray(), axis=0)
                top_term_indices = np.argsort(centroid_vector)[::-1][:20]
                top_terms = [{'term': vectorizer.get_feature_names_out()[i], 
                             'score': centroid_vector[i]}
                             for i in top_term_indices]
            else: 
                top_terms = [] # BM25 doesn't provide term scores
                
            # Get representative documents
            doc_similarities = []
            for idx in cluster_indices:
                doc_vector = doc_vectors[idx]
                sim_to_centroid = cosine_similarity([doc_vector], 
                                                    [np.mean(doc_vectors[cluster_indices], axis=0)])[0][0]
                
                doc_info = {
                    'title': display_data.iloc[idx]['Title'],
                    'date': display_data.iloc[idx]['date_of_report'],
                    'similarity': float(sim_to_centroid),
                    'summary': display_data.iloc[idx]['Content'][:500]
                }
                doc_similarities.append((idx, sim_to_centroid, doc_info))

            # Sort by similarity and get representative docs
            doc_similarities.sort(key=lambda x: x[1], reverse=True)
            representative_docs = [item[2] for item in doc_similarities]

            # Calculate cluster cohesion
            cluster_similarities = similarity_matrix[cluster_indices][:, cluster_indices]
            cohesion = float(np.mean(cluster_similarities))

            clusters.append({
                'id': len(clusters),
                'size': len(cluster_indices),
                'cohesion': cohesion,
                'terms': top_terms,
                'documents': representative_docs
            })

        return {
            'n_clusters': len(clusters),
            'total_documents': len(processed_texts),
            'silhouette_score': float(best_score),
            'clusters': clusters
        }
        
    except Exception as e:
        logging.error(f"Error in semantic clustering: {e}", exc_info=True)
        raise ValueError(f"Clustering failed: {str(e)}")


def render_topic_summary_tab(data: pd.DataFrame) -> None:
    """Main topic analysis and summary tab rendering function"""
    st.header("Topic Analysis & Summaries")
    st.markdown("""
    This analysis identifies key themes and patterns in the report contents, automatically clustering similar documents
    and generating summaries for each thematic group.
    """)

    # Show previous results if available
    if 'topic_model' in st.session_state and st.session_state.topic_model is not None:
        st.sidebar.success("Previous analysis results available")
        if st.sidebar.button("View Previous Results"):
            render_summary_tab(st.session_state.topic_model, data)
            return

    with st.sidebar:
        st.header("Analysis Parameters")
        
        # Similarity method selection
        method = st.selectbox("Similarity Method", ['tfidf', 'bm25'])
        
        # Basic clustering parameters
        min_cluster_size = st.slider(
            "Minimum Group Size ❓", 
            2, 5, 2,
            help="Minimum number of documents needed to form a thematic group"
        )
        
        # Document frequency parameters
        total_docs = len(data)
        min_docs = st.slider(
            "Minimum Document Frequency ❓", 
            2, max(2, total_docs//2), 5,
            help="How many documents a term must appear in"
        )
        min_df = min_docs / total_docs
        
        max_docs = st.slider(
            "Maximum Document Frequency ❓", 
            min_docs, total_docs, int(total_docs * 0.9),
            help="Maximum number of documents a term can appear in"
        )
        max_df = max_docs / total_docs

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "From",
            value=data['date_of_report'].min().date(),
            min_value=data['date_of_report'].min().date(),
            max_value=data['date_of_report'].max().date(),
            key="analysis_start_date",
            format="DD/MM/YYYY"
        )

    with col2:
        end_date = st.date_input(
            "To",
            value=data['date_of_report'].max().date(),
            min_value=data['date_of_report'].min().date(),
            max_value=data['date_of_report'].max().date(),
            key="analysis_end_date",
            format="DD/MM/YYYY"
        )

    # Category selection
    all_categories = set()
    for cats in data['categories'].dropna():
        if isinstance(cats, list):
            all_categories.update(cats)
    
    categories = st.multiselect(
        "Categories ❓",
        options=sorted(all_categories),
        key="analysis_categories",
        help="Specific categories to analyze"
    )

    # Analysis button
    analyze_clicked = st.button(
        "🔍 Analyze Documents",
        type="primary",
        use_container_width=True,
        help="Start the analysis with current settings"
    )

    if analyze_clicked:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize
            progress_bar.progress(0.1)
            status_text.text("Initializing analysis...")
            initialize_nltk()

            # Filter data
            filtered_df = data.copy()
            progress_bar.progress(0.2)
            status_text.text("Applying filters...")
            
            # Apply date filter
            filtered_df = filtered_df[
                (filtered_df['date_of_report'].dt.date >= start_date) &
                (filtered_df['date_of_report'].dt.date <= end_date)
            ]
            
            # Apply category filter
            if categories:
                filtered_df = filter_by_categories(filtered_df, categories)
            
            progress_bar.progress(0.3)
            status_text.text("Preprocessing documents...")
            
            # Remove empty content
            filtered_df = filtered_df[filtered_df['Content'].notna() & (filtered_df['Content'].str.strip() != '')]
            
            if len(filtered_df) < min_cluster_size:
                progress_bar.empty()
                status_text.empty()
                st.warning(f"Not enough documents match the criteria. Found {len(filtered_df)}, need at least {min_cluster_size}.")
                return
            
            # Process content
            progress_bar.progress(0.4)
            status_text.text("Processing document content...")
            
            # Prepare the DataFrame with only required columns
            processed_df = pd.DataFrame({
                'Content': filtered_df['Content'],
                'Title': filtered_df['Title'],
                'date_of_report': filtered_df['date_of_report'],
                'URL': filtered_df['URL'],
                'categories': filtered_df['categories']
            })
            
            progress_bar.progress(0.6)
            status_text.text("Performing clustering analysis...")
            
            # Perform clustering
            cluster_results = perform_semantic_clustering(
                processed_df,
                method=method,
                min_cluster_size=min_cluster_size,
                max_features=5000,
                min_df=min_df,
                max_df=max_df
            )
            
            progress_bar.progress(0.8)
            status_text.text("Analyzing cluster results...")
            
            # Store results
            st.session_state.topic_model = cluster_results
            
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            render_summary_tab(cluster_results, processed_df)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Analysis error: {str(e)}")
            logging.error(f"Analysis error: {e}", exc_info=True)
        
def format_date_uk(date_obj):
    """Convert datetime object to UK date format string"""
    if pd.isna(date_obj):
        return ''
    try:
        if isinstance(date_obj, str):
            # Try to parse string to datetime first
            date_obj = pd.to_datetime(date_obj)
        return date_obj.strftime('%d/%m/%Y')
    except:
        return str(date_obj)

def generate_extractive_summary(documents, max_length=500):
    """Generate extractive summary from cluster documents with traceability"""
    try:
        # Combine all document texts with source tracking
        all_sentences = []
        for doc in documents:
            sentences = sent_tokenize(doc['summary'])
            for sent in sentences:
                all_sentences.append({
                    'text': sent,
                    'source': doc['title'],
                    'date': format_date_uk(doc['date'])  # Format date here
                })
        
        # Calculate sentence importance using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([s['text'] for s in all_sentences])
        
        # Calculate sentence scores
        sentence_scores = []
        for idx, sentence in enumerate(all_sentences):
            score = np.mean(tfidf_matrix[idx].toarray())
            sentence_scores.append((score, sentence))
        
        # Sort by importance and select top sentences
        sentence_scores.sort(reverse=True)
        summary_length = 0
        summary_sentences = []
        
        for score, sentence in sentence_scores:
            if summary_length + len(sentence['text']) <= max_length:
                summary_sentences.append({
                    'text': sentence['text'],
                    'source': sentence['source'],
                    'date': sentence['date'],
                    'score': float(score)
                })
                summary_length += len(sentence['text'])
            else:
                break
                
        return summary_sentences
        
    except Exception as e:
        logging.error(f"Error in extractive summarization: {e}")
        return []

def generate_abstractive_summary(cluster_terms, documents, max_length=500):
    """Generate abstractive summary from cluster information with improved date handling"""
    try:
        # Extract key themes from terms
        top_themes = [term['term'] for term in cluster_terms[:5]]
        
        # Get document dates and format them with proper sorting
        dates = []
        for doc in documents:
            try:
                if doc['date']:
                    date_obj = pd.to_datetime(doc['date'])
                    dates.append(date_obj)
            except:
                continue
        
        if dates:
            start_date = min(dates).strftime('%d/%m/%Y')
            end_date = max(dates).strftime('%d/%m/%Y')
            date_range = f"from {start_date} to {end_date}"
        else:
            date_range = ""
        
        # Extract key themes with better formatting
        main_themes = ', '.join(top_themes[:-1])
        if main_themes:
            themes_text = f"{main_themes} and {top_themes[-1]}"
        else:
            themes_text = top_themes[0] if top_themes else ""
        
        # Build better structured summary
        summary = f"This cluster contains {len(documents)} documents "
        if date_range:
            summary += f"{date_range} "
        summary += f"focused on {themes_text}. "
        
        # Add key patterns with improved statistics
        term_patterns = []
        for term in cluster_terms[5:8]:  # Get next 3 terms after main themes
            if term['cluster_frequency'] > 0:
                freq = term['cluster_frequency'] * 100
                # Add context based on frequency
                if freq > 75:
                    context = "very commonly"
                elif freq > 50:
                    context = "frequently"
                elif freq > 25:
                    context = "sometimes"
                else:
                    context = "occasionally"
                term_patterns.append(
                    f"{term['term']} ({context} appearing in {freq:.0f}% of documents)"
                )
        
        if term_patterns:
            summary += f"Common patterns include {', '.join(term_patterns)}. "
        
        # Add cluster distinctiveness if available
        if any(term['total_frequency'] < 0.5 for term in cluster_terms[:5]):
            distinctive_terms = [
                term['term'] for term in cluster_terms[:5] 
                if term['total_frequency'] < 0.5
            ]
            if distinctive_terms:
                summary += f"This cluster is particularly distinctive in its discussion of {', '.join(distinctive_terms)}."
        
        # Truncate to max length while preserving complete sentences
        if len(summary) > max_length:
            summary = summary[:max_length]
            last_period = summary.rfind('.')
            if last_period > 0:
                summary = summary[:last_period + 1]
            
        return summary
        
    except Exception as e:
        logging.error(f"Error in abstractive summarization: {e}")
        return "Error generating summary"

def get_optimal_clustering_params(num_docs: int) -> Dict[str, int]:
    """Calculate optimal clustering parameters based on dataset size"""
    
    # Base parameters
    params = {
        'min_cluster_size': 2,  # Minimum starting point
        'max_features': 5000,   # Maximum vocabulary size
        'min_docs': 2,         # Minimum document frequency
        'max_docs': None       # Maximum document frequency (will be calculated)
    }
    
    # Adjust minimum cluster size based on dataset size
    if num_docs < 10:
        params['min_cluster_size'] = 2
    elif num_docs < 20:
        params['min_cluster_size'] = 3
    elif num_docs < 50:
        params['min_cluster_size'] = 4
    else:
        params['min_cluster_size'] = 5
        
    # Adjust document frequency bounds
    params['min_docs'] = max(2, int(num_docs * 0.05))  # At least 5% of documents
    params['max_docs'] = min(
        int(num_docs * 0.95),  # No more than 95% of documents
        num_docs - params['min_cluster_size']  # Leave room for at least one cluster
    )
    
    # Adjust feature count based on dataset size
    if num_docs < 20:
        params['max_features'] = 2000
    elif num_docs < 50:
        params['max_features'] = 3000
    elif num_docs < 100:
        params['max_features'] = 4000
    else:
        params['max_features'] = 5000
        
    return params
    
def display_cluster_analysis(cluster_results: Dict) -> None:
    """Display comprehensive cluster analysis results"""
    try:
        st.subheader("Document Clustering Analysis")
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Clusters", cluster_results['n_clusters'])
        with col2:
            st.metric("Total Documents", cluster_results['total_documents'])
        with col3:
            st.metric("Clustering Quality", 
                     f"{cluster_results['silhouette_score']:.3f}",
                     help="Silhouette score (ranges from -1 to 1, higher is better)")
        
        # Display each cluster
        for cluster in cluster_results['clusters']:
            with st.expander(f"Cluster {cluster['id']+1} ({cluster['size']} documents)", 
                           expanded=True):
                
                # Cluster metrics
                st.markdown(f"**Cohesion Score**: {cluster['cohesion']:.3f}")
                
                # Terms analysis
                st.markdown("#### Key Terms")
                terms_df = pd.DataFrame(cluster['terms'])
                
                # Create term importance visualization
                fig = px.bar(
                    terms_df.head(10),
                    x='relevance',
                    y='term',
                    orientation='h',
                    title='Top Terms by Relevance',
                    labels={'relevance': 'Relevance Score', 'term': 'Term'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Representative documents with formatted dates
                st.markdown("#### Representative Documents")
                for doc in cluster['documents']:
                    st.markdown(f"**{doc['title']}** (Similarity: {doc['similarity']:.2f})")
                    st.markdown(f"**Date**: {format_date_uk(doc['date'])}")
                    st.markdown(f"**Summary**: {doc['summary'][:300]}...")
                    st.markdown("---")  # Separator between documents

    except Exception as e:
        st.error(f"Error displaying cluster analysis: {str(e)}")
        logging.error(f"Display error: {str(e)}", exc_info=True)



def export_cluster_results(cluster_results: Dict) -> bytes:
    """Export cluster results with proper timestamp handling"""
    output = io.BytesIO()
    
    # Prepare export data with timestamp conversion
    export_data = {
        'metadata': {
            'total_documents': cluster_results['total_documents'],
            'number_of_clusters': cluster_results['n_clusters'],
            'silhouette_score': cluster_results['silhouette_score'],
        },
        'clusters': []
    }
    
    # Convert cluster data
    for cluster in cluster_results['clusters']:
        # Create a copy of cluster with converted documents
        cluster_export = cluster.copy()
        for doc in cluster_export['documents']:
            # Ensure date is a string
            doc['date'] = str(doc['date'])
        
        export_data['clusters'].append(cluster_export)
    
    # Write JSON to BytesIO
    json.dump(export_data, io.TextIOWrapper(output, encoding='utf-8'), indent=2)
    output.seek(0)
    
    return output.getvalue()

def validate_data_state():
    """Check if valid data exists in session state"""
    return ('current_data' in st.session_state and 
            st.session_state.current_data is not None and 
            not st.session_state.current_data.empty)

def validate_model_state():
    """Check if valid topic model exists in session state"""
    return ('topic_model' in st.session_state and 
            st.session_state.topic_model is not None)

def handle_no_data_state(section):
    """Handle state when no data is available"""
    st.warning("No data available. Please scrape reports or upload a file first.")
    uploaded_file = st.file_uploader(
        "Upload existing data file",
        type=['csv', 'xlsx'],
        key=f"{section}_uploader"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            df = process_scraped_data(df)
            st.session_state.current_data = df
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def handle_no_model_state():
    """Handle state when no topic model is available"""
    st.warning("Please run the clustering analysis first to view summaries.")
    if st.button("Go to Topic Modeling"):
        st.session_state.current_tab = "🔬 Topic Modeling"
        st.rerun()

def handle_error(error):
    """Handle application errors"""
    st.error("An error occurred")
    st.error(str(error))
    logging.error(f"Application error: {error}", exc_info=True)
    
    with st.expander("Error Details"):
        st.code(traceback.format_exc())
    
    st.warning("Recovery options:")
    st.markdown("""
    1. Clear data and restart
    2. Upload different data
    3. Check filter settings
    """)

def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center'>
        <p>Built with Streamlit • Data from UK Judiciary</p>
        </div>""",
        unsafe_allow_html=True
    )

def render_topic_summary_tab(data: pd.DataFrame) -> None:
    """Main topic analysis and summary tab rendering function"""
    st.header("Topic Analysis & Summaries")
    st.markdown("""
    This analysis identifies key themes and patterns in the report contents, automatically clustering similar documents
    and generating summaries for each thematic group.
    """)

    # Show previous results if available
    if 'topic_model' in st.session_state and st.session_state.topic_model is not None:
        st.sidebar.success("Previous analysis results available")
        if st.sidebar.button("View Previous Results"):
            render_summary_tab(st.session_state.topic_model, data)
            return

    with st.sidebar:
        st.header("Analysis Parameters")
        
        # Similarity method selection
        method = st.selectbox("Similarity Method", ['tfidf', 'bm25'])
        
        # Basic clustering parameters
        min_cluster_size = st.slider(
            "Minimum Group Size ❓", 
            2, 5, 2,
            help="Minimum number of documents needed to form a thematic group"
        )
        
        # Document frequency parameters
        total_docs = len(data)
        min_docs = st.slider(
            "Minimum Document Frequency ❓", 
            2, max(2, total_docs//2), 5,
            help="How many documents a term must appear in"
        )
        min_df = min_docs / total_docs
        
        max_docs = st.slider(
            "Maximum Document Frequency ❓", 
            min_docs, total_docs, int(total_docs * 0.9),
            help="Maximum number of documents a term can appear in"
        )
        max_df = max_docs / total_docs

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "From",
            value=data['date_of_report'].min().date(),
            min_value=data['date_of_report'].min().date(),
            max_value=data['date_of_report'].max().date(),
            key="analysis_start_date",
            format="DD/MM/YYYY"
        )

    with col2:
        end_date = st.date_input(
            "To",
            value=data['date_of_report'].max().date(),
            min_value=data['date_of_report'].min().date(),
            max_value=data['date_of_report'].max().date(),
            key="analysis_end_date",
            format="DD/MM/YYYY"
        )

    # Category selection
    all_categories = set()
    for cats in data['categories'].dropna():
        if isinstance(cats, list):
            all_categories.update(cats)
    
    categories = st.multiselect(
        "Categories ❓",
        options=sorted(all_categories),
        key="analysis_categories",
        help="Specific categories to analyze"
    )

    # Analysis button
    analyze_clicked = st.button(
        "🔍 Analyze Documents",
        type="primary",
        use_container_width=True,
        help="Start the analysis with current settings"
    )

    if analyze_clicked:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize
            progress_bar.progress(0.1)
            status_text.text("Initializing analysis...")
            initialize_nltk()

            # Filter data
            filtered_df = data.copy()
            progress_bar.progress(0.2)
            status_text.text("Applying filters...")
            
            # Apply date filter
            filtered_df = filtered_df[
                (filtered_df['date_of_report'].dt.date >= start_date) &
                (filtered_df['date_of_report'].dt.date <= end_date)
            ]
            
            # Apply category filter
            if categories:
                filtered_df = filter_by_categories(filtered_df, categories)
            
            progress_bar.progress(0.3)
            status_text.text("Preprocessing documents...")
            
            # Remove empty content
            filtered_df = filtered_df[filtered_df['Content'].notna() & (filtered_df['Content'].str.strip() != '')]
            
            if len(filtered_df) < min_cluster_size:
                progress_bar.empty()
                status_text.empty()
                st.warning(f"Not enough documents match the criteria. Found {len(filtered_df)}, need at least {min_cluster_size}.")
                return
            
            # Process content
            progress_bar.progress(0.4)
            status_text.text("Processing document content...")
            
            # Prepare the DataFrame with only required columns
            processed_df = pd.DataFrame({
                'Content': filtered_df['Content'],
                'Title': filtered_df['Title'],
                'date_of_report': filtered_df['date_of_report'],
                'URL': filtered_df['URL'],
                'categories': filtered_df['categories']
            })
            
            progress_bar.progress(0.6)
            status_text.text("Performing clustering analysis...")
            
            # Perform clustering
            cluster_results = perform_semantic_clustering(
                processed_df,
                method=method,
                min_cluster_size=min_cluster_size,
                max_features=5000,
                min_df=min_df,
                max_df=max_df
            )
            
            progress_bar.progress(0.8)
            status_text.text("Analyzing cluster results...")
            
            # Store results
            st.session_state.topic_model = cluster_results
            
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            render_summary_tab(cluster_results, processed_df)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Analysis error: {str(e)}")
            logging.error(f"Analysis error: {e}", exc_info=True)
            

def render_summary_tab(cluster_results: Dict, original_data: pd.DataFrame) -> None:
    """Render cluster summaries and records with flexible column handling"""
    if not cluster_results or 'clusters' not in cluster_results:
        st.warning("No cluster results available.")
        return
    
    st.write(f"Found {cluster_results['total_documents']} total documents in {cluster_results['n_clusters']} clusters")
    
    for cluster in cluster_results['clusters']:
        st.markdown(f"### Cluster {cluster['id']+1} ({cluster['size']} documents)")
        
        # Overview
        st.markdown("#### Overview") 
        abstractive_summary = generate_abstractive_summary(
            cluster['terms'],
            cluster['documents']
        )
        st.write(abstractive_summary)
        
        # Key terms table
        st.markdown("#### Key Terms")
        terms_df = pd.DataFrame([
            {'Term': term['term'], 
             'Frequency': f"{term['cluster_frequency']*100:.0f}%"}
            for term in cluster['terms'][:10]
        ])
        st.dataframe(terms_df, hide_index=True)
        
        # Records
        st.markdown("#### Records")
        st.success(f"Showing {len(cluster['documents'])} matching documents")
        
        # Get the full records from original data
        doc_titles = [doc.get('title', '') for doc in cluster['documents']]
        cluster_docs = original_data[original_data['Title'].isin(doc_titles)].copy()
        
        # Sort to match the original order
        title_to_position = {title: i for i, title in enumerate(doc_titles)}
        cluster_docs['sort_order'] = cluster_docs['Title'].map(title_to_position)
        cluster_docs = cluster_docs.sort_values('sort_order').drop('sort_order', axis=1)
        
        # Determine available columns
        available_columns = []
        column_config = {}
        
        # Always include URL and Title if available
        if 'URL' in cluster_docs.columns:
            available_columns.append('URL')
            column_config['URL'] = st.column_config.LinkColumn("Report Link")
        
        if 'Title' in cluster_docs.columns:
            available_columns.append('Title')
            column_config['Title'] = st.column_config.TextColumn("Title")
        
        # Add date if available
        if 'date_of_report' in cluster_docs.columns:
            available_columns.append('date_of_report')
            column_config['date_of_report'] = st.column_config.DateColumn(
                "Date of Report",
                format="DD/MM/YYYY"
            )
        
        # Add optional columns if available
        optional_columns = ['ref', 'deceased_name', 'coroner_name', 'coroner_area', 'categories']
        for col in optional_columns:
            if col in cluster_docs.columns:
                available_columns.append(col)
                if col == 'categories':
                    column_config[col] = st.column_config.ListColumn("Categories")
                else:
                    column_config[col] = st.column_config.TextColumn(col.replace('_', ' ').title())
        
        # Display the dataframe with available columns
        if available_columns:
            st.dataframe(
                cluster_docs[available_columns],
                column_config=column_config,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No displayable columns found in the data")
        
        st.markdown("---")
        
def main():
    """Updated main application entry point."""
    initialize_session_state()
    
    st.title("UK Judiciary PFD Reports Analysis")
    st.markdown("""
    This application analyzes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    You can scrape new reports, analyze existing data, and explore thematic patterns.
    """)
    
    # Updated tab selection without topic modeling tab
    current_tab = st.radio(
        "Select section:",
        [
            "🔍 Scrape Reports",
            "📊 Analysis",
            "📝 Topic Analysis & Summaries"
        ],
        label_visibility="collapsed",
        horizontal=True,
        key="main_tab_selector"
    )
    
    st.markdown("---")
    
    try:
        if current_tab == "🔍 Scrape Reports":
            render_scraping_tab()
            
        elif current_tab == "📊 Analysis":
            if not validate_data_state():
                handle_no_data_state("analysis")
            else:
                render_analysis_tab(st.session_state.current_data)
        
        elif current_tab == "📝 Topic Analysis & Summaries":
            if not validate_data_state():
                handle_no_data_state("topic_summary")
            else:
                render_topic_summary_tab(st.session_state.current_data)
        
        # Sidebar data management
        with st.sidebar:
            st.header("Data Management")
            
            if hasattr(st.session_state, 'data_source'):
                st.info(f"Current data: {st.session_state.data_source}")
            
            if st.button("Clear All Data"):
                for key in ['current_data', 'scraped_data', 'uploaded_data', 
                          'topic_model', 'data_source']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("All data cleared")
                st.experimental_rerun()
        
        render_footer()
        
    except Exception as e:
        handle_error(e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Critical Error")
        st.error(str(e))
        logging.critical(f"Application crash: {e}", exc_info=True)
