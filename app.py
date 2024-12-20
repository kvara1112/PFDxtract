import streamlit as st
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from bs4 import BeautifulSoup, Tag

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

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

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
    
    # Simply combine all text fields
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
    """Clean text with better preprocessing"""
    if not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove any numbers or words containing numbers
        text = re.sub(r'\b\w*\d+\w*\b', '', text)
        
        # Remove single characters
        text = re.sub(r'\b[a-z]\b', '', text)
        
        # Remove special characters and multiple spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    except Exception as e:
        logging.error(f"Error in text cleaning: {e}")
        return ""

def extract_topics_lda(df: pd.DataFrame, num_topics: int = 5, max_features: int = 1000) -> Tuple[LatentDirichletAllocation, TfidfVectorizer, np.ndarray]:
    """Extract topics with improved preprocessing"""
    try:
        # Combine and clean texts
        texts = []
        for _, row in df.iterrows():
            combined_text = combine_document_text(row)
            cleaned_text = clean_text_for_modeling(combined_text)
            if cleaned_text and len(cleaned_text.split()) > 3:  # Ensure meaningful content
                texts.append(cleaned_text)
        
        # Configure vectorizer with stricter parameters
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,  # Only keep terms appearing in at least 2 documents
            max_df=0.95,  # Remove terms appearing in >95% of documents
            stop_words='english',
            token_pattern=r'(?u)\b[a-z]{2,}\b'  # Only words with 2+ letters
        )
        
        # Create document-term matrix
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Configure LDA
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            n_jobs=-1,
            max_iter=20,
            learning_method='batch',
            doc_topic_prior=0.1,
            topic_word_prior=0.01
        )
        
        # Fit model
        doc_topic_dist = lda_model.fit_transform(tfidf_matrix)
        
        # Normalize the components
        for idx in range(len(lda_model.components_)):
            lda_model.components_[idx] = lda_model.components_[idx] / lda_model.components_[idx].sum()
        
        return lda_model, vectorizer, doc_topic_dist
        
    except Exception as e:
        st.error(f"Error in topic extraction: {str(e)}")
        raise e



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
    """Extract structured metadata from report content"""
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
            r'Date of report:\s*(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})',
            r'Date of report:\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'DATED this (\d{1,2}(?:st|nd|rd|th)?\s+day of [A-Za-z]+\s+\d{4})',
            r'Date:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})'
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
        
        # Extract categories
        cat_match = re.search(r'Category:?\s*([^\n]+)', content)
        if cat_match:
            categories = cat_match.group(1).split('|')
            metadata['categories'] = [clean_text(cat).strip() for cat in categories if clean_text(cat).strip()]
        
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
    """Get full content from report page with multiple PDF handling"""
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
        
        paragraphs = content.find_all(['p', 'table'])
        webpage_text = '\n\n'.join(p.get_text(strip=True, separator=' ') for p in paragraphs)
        
        pdf_contents = []
        pdf_paths = []
        pdf_names = []
        
        pdf_links = (
            soup.find_all('a', class_='related-content__link', href=re.compile(r'\.pdf$')) or
            soup.find_all('a', href=re.compile(r'\.pdf$'))
        )
        
        for pdf_link in pdf_links:
            pdf_url = pdf_link['href']
            
            if not pdf_url.startswith(('http://', 'https://')):
                pdf_url = f"https://www.judiciary.uk{pdf_url}" if not pdf_url.startswith('/') else f"https://www.judiciary.uk/{pdf_url}"
            
            pdf_path, pdf_name = save_pdf(pdf_url)
            
            if pdf_path:
                pdf_content = extract_pdf_content(pdf_path)
                pdf_contents.append(pdf_content)
                pdf_paths.append(pdf_path)
                pdf_names.append(pdf_name)
        
        return {
            'content': clean_text(webpage_text),
            'pdf_contents': pdf_contents,
            'pdf_paths': pdf_paths,
            'pdf_names': pdf_names
        }
        
    except Exception as e:
        logging.error(f"Error getting report content: {e}")
        return None

# Scraping functions
def scrape_page(url: str) -> List[Dict]:
    """Scrape a single page of search results"""
    try:
        response = make_request(url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results_list = soup.find('ul', class_='search__list')
        
        if not results_list:
            logging.warning(f"No results list found on page: {url}")
            return []
        
        reports = []
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
                    
                    # Add PDF details
                    for i, (name, content, path) in enumerate(zip(
                        content_data['pdf_names'],
                        content_data['pdf_contents'],
                        content_data['pdf_paths']
                    ), 1):
                        report[f'PDF_{i}_Name'] = name
                        report[f'PDF_{i}_Content'] = content
                        report[f'PDF_{i}_Path'] = path
                    
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
    """Process and clean scraped data with improved metadata extraction"""
    try:
        # Create a copy
        df = df.copy()
        
        # Extract metadata from Content field if it exists
        if 'Content' in df.columns:
            # Process each row
            processed_rows = []
            for _, row in df.iterrows():
                # Start with the original row data
                processed_row = row.to_dict()
                
                # Extract metadata using the existing function
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
                
                # Try different formats but always convert to datetime
                formats = [
                    '%Y-%m-%d',
                    '%d-%m-%Y',
                    '%d %B %Y',
                    '%d %b %Y'
                ]
                
                for fmt in formats:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                
                try:
                    # Last resort - try pandas default parser
                    return pd.to_datetime(date_str)
                except:
                    return pd.NaT
            
            # Convert dates to datetime objects
            result['date_of_report'] = result['date_of_report'].apply(parse_date)
        
        return result
            
    except Exception as e:
        logging.error(f"Error in process_scraped_data: {e}")
        return df

def construct_search_url(base_url: str, keyword: Optional[str] = None, 
                        category: Optional[str] = None, 
                        category_slug: Optional[str] = None, 
                        page: Optional[int] = None) -> str:
    """Constructs search URL with proper parameter handling"""
    # Clean inputs
    keyword = keyword.strip() if keyword else None
    category = category.strip() if category else None
    
    # Build query parameters
    query_params = []
    
    # Always add post_type for filtered searches
    if category or keyword:
        query_params.append("post_type=pfd")
    
    # Add category filter if present
    if category and category_slug:
        query_params.append(f"pfd_report_type={category_slug}")
    
    # Add keyword filter if present
    if keyword:
        query_params.append(f"s={keyword}")
    
    # Construct base URL
    if query_params:
        query_string = "&".join(query_params)
        url = f"{base_url}?{query_string}"
    else:
        url = f"{base_url}prevention-of-future-death-reports/"
    
    # Add pagination if needed
    if page and page > 1:
        if query_params:
            url = f"{url}&page={page}"
        else:
            url = f"{url}page/{page}/"
    
    return url


def scrape_pfd_reports(
    keyword: Optional[str] = None,
    category: Optional[str] = None,
    date_after: Optional[str] = None,
    date_before: Optional[str] = None,
    order: str = "relevance",
    max_pages: Optional[int] = None
) -> List[Dict]:
    """
    Scrape PFD reports with comprehensive filtering and improved error handling
    
    Args:
        keyword: Search term
        category: Report category
        date_after: Start date filter
        date_before: End date filter
        order: Sort order
        max_pages: Maximum number of pages to scrape
        
    Returns:
        List of dictionaries containing report data
    """
    all_reports = []
    base_url = "https://www.judiciary.uk/"
    
    try:
        # Validate and prepare category
        category_slug = None
        if category:
            matching_categories = [
                cat for cat in get_pfd_categories() 
                if cat.lower() == category.lower()
            ]
            
            if not matching_categories:
                st.error(f"No matching category found for: {category}")
                return []
            
            category = matching_categories[0]
            category_slug = category.lower().replace(' ', '-').replace('&', 'and')
        
        # Construct initial search URL
        base_search_url = construct_search_url(
            base_url=base_url,
            keyword=keyword,
            category=category,
            category_slug=category_slug
        )
        
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
        progress_bar = st.progress(0)
        for current_page in range(1, total_pages + 1):
            try:
                # Construct page URL using the same URL constructor
                page_url = construct_search_url(
                    base_url=base_url,
                    keyword=keyword,
                    category=category,
                    category_slug=category_slug,
                    page=current_page
                )
                
                st.write(f"Processing page {current_page} of {total_pages}")
                
                # Scrape current page
                page_reports = scrape_page(page_url)
                
                # Check if page has results
                if not page_reports:
                    st.warning(f"No results found on page {current_page}. Stopping search.")
                    break
                
                # Apply date filters if specified
                if date_after or date_before:
                    page_reports = filter_reports_by_date(
                        page_reports, date_after, date_before
                    )
                
                all_reports.extend(page_reports)
                
                # Update progress
                progress = int((current_page / total_pages) * 100)
                progress_bar.progress(progress)
                
                # Add delay between pages
                time.sleep(2)
                
            except Exception as page_error:
                logging.error(f"Error processing page {current_page}: {page_error}")
                st.warning(f"Error on page {current_page}. Continuing with next page...")
                continue
        
        progress_bar.progress(100)
        st.success(f"Successfully scraped {len(all_reports)} reports")
        
        # Sort results if specified
        if order != "relevance":
            all_reports = sort_reports(all_reports, order)
        
        return all_reports
        
    except Exception as e:
        logging.error(f"Error in scrape_pfd_reports: {e}")
        st.error(f"An error occurred while scraping reports: {e}")
        return all_reports

def filter_reports_by_date(reports: List[Dict], 
                         date_after: Optional[str], 
                         date_before: Optional[str]) -> List[Dict]:
    """Filter reports by date range"""
    filtered_reports = reports.copy()
    
    if date_after:
        after_date = datetime.strptime(date_after, "%Y-%m-%d")
        filtered_reports = [
            report for report in filtered_reports 
            if datetime.strptime(report['date'], "%Y-%m-%d") >= after_date
        ]
    
    if date_before:
        before_date = datetime.strptime(date_before, "%Y-%m-%d")
        filtered_reports = [
            report for report in filtered_reports 
            if datetime.strptime(report['date'], "%Y-%m-%d") <= before_date
        ]
    
    return filtered_reports

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

        
def process_scraped_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean scraped data with improved metadata extraction"""
    try:
        # Create a copy
        df = df.copy()
        
        # Extract metadata from Content field if it exists
        if 'Content' in df.columns:
            # Process each row
            processed_rows = []
            for _, row in df.iterrows():
                # Start with the original row data
                processed_row = row.to_dict()
                
                # Extract metadata from Content
                content = str(row.get('Content', ''))
                
                # Extract date
                date_match = re.search(r'Date of report:\s*(\d{1,2}(?:/|-)\d{1,2}(?:/|-)\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})', content)
                if date_match:
                    processed_row['date_of_report'] = date_match.group(1)
                
                # Rest of metadata extraction remains the same
                ref_match = re.search(r'Ref(?:erence)?:?\s*([-\d]+)', content)
                if ref_match:
                    processed_row['ref'] = ref_match.group(1)
                
                name_match = re.search(r'Deceased name:?\s*([^\n]+)', content)
                if name_match:
                    processed_row['deceased_name'] = name_match.group(1).strip()
                
                coroner_match = re.search(r'Coroner(?:\'?s)? name:?\s*([^\n]+)', content)
                if coroner_match:
                    processed_row['coroner_name'] = coroner_match.group(1).strip()
                
                area_match = re.search(r'Coroner(?:\'?s)? Area:?\s*([^\n]+)', content)
                if area_match:
                    processed_row['coroner_area'] = area_match.group(1).strip()
                
                cat_match = re.search(r'Category:?\s*([^\n]+)', content)
                if cat_match:
                    categories = cat_match.group(1).split('|')
                    processed_row['categories'] = [cat.strip() for cat in categories if cat.strip()]
                else:
                    processed_row['categories'] = []
                
                processed_rows.append(processed_row)
            
            # Create new DataFrame from processed rows
            result = pd.DataFrame(processed_rows)
        else:
            result = df.copy()
        
        # Convert date_of_report to datetime with improved handling
        if 'date_of_report' in result.columns:
            def parse_date(date_str):
                if pd.isna(date_str):
                    return pd.NaT
                
                date_str = str(date_str).strip()
                
                # Try different date formats
                formats = [
                    '%d/%m/%Y',
                    '%Y-%m-%d',
                    '%d-%m-%Y',
                    '%d %B %Y',
                    '%d %b %Y'
                ]
                
                # Remove ordinal indicators
                date_str = re.sub(r'(\d)(st|nd|rd|th)', r'\1', date_str)
                
                for fmt in formats:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                
                # If all formats fail, try pandas default parser
                try:
                    return pd.to_datetime(date_str)
                except:
                    return pd.NaT
            
            result['date_of_report'] = result['date_of_report'].apply(parse_date)
        
        return result
            
    except Exception as e:
        logging.error(f"Error in process_scraped_data: {e}")
        return df


def plot_timeline(df: pd.DataFrame) -> None:
    """Plot timeline of reports"""
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
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

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


def render_scraping_tab():
    """Render the scraping tab"""
    st.header("Scrape PFD Reports")
    
    if 'scraped_data' in st.session_state and st.session_state.scraped_data is not None:
        # Show existing results if available
        st.success(f"Found {len(st.session_state.scraped_data)} reports")
        
        # Display results table with UK date format
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
        
        # Show export options
        show_export_options(st.session_state.scraped_data, "scraped")
    
    with st.form("scraping_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            search_keyword = st.text_input("Search keywords (do not leave empty use reports as a general term or another search term):", value="reports")
            category = st.selectbox("PFD Report type:", 
                [""] + get_pfd_categories(), 
                format_func=lambda x: x if x else "Select a category")
            
            order = st.selectbox("Sort by:", [
                "relevance",
                "desc",
                "asc"
            ], format_func=lambda x: {
                "relevance": "Relevance",
                "desc": "Newest first",
                "asc": "Oldest first"
            }[x])
        
        with col2:
            date_after = st.date_input(
                "Published after:",
                None,
                format="DD/MM/YYYY"
            )
            
            date_before = st.date_input(
                "Published before:",
                None,
                format="DD/MM/YYYY"
            )
            
            max_pages = st.number_input(
                "Maximum pages to scrape (0 for all):", 
                min_value=0, 
                value=0
            )
        
        submitted = st.form_submit_button("Search Reports")
    
    if submitted:
        try:
            # Validate date range if both dates are provided
            if date_after and date_before and date_after > date_before:
                st.error("Start date must be before or equal to end date.")
                return

            # Store search parameters in session state
            st.session_state.last_search_params = {
                'keyword': search_keyword,
                'category': category,
                'date_after': date_after.strftime('%d/%m/%Y') if date_after else None,
                'date_before': date_before.strftime('%d/%m/%Y') if date_before else None,
                'order': order
            }

            # Convert dates to required format
            date_after_str = date_after.strftime('%d/%m/%Y') if date_after else None
            date_before_str = date_before.strftime('%d/%m/%Y') if date_before else None
            
            # Set max pages
            max_pages_val = None if max_pages == 0 else max_pages
            
            # Perform scraping
            reports = scrape_pfd_reports(
                keyword=search_keyword,
                category=category if category else None,
                date_after=date_after_str,
                date_before=date_before_str,
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
            
def show_export_options(df: pd.DataFrame, prefix: str):
    """Show export options for the data with descriptive filename"""
    st.subheader("Export Options")
    
    # Generate descriptive filename components
    filename_parts = []
    
    # Add search parameters from session state if available
    if hasattr(st.session_state, 'last_search_params'):
        params = st.session_state.last_search_params
        
        # Add keyword if present
        if params.get('keyword'):
            filename_parts.append(f"kw_{params['keyword'].replace(' ', '_')}")
        
        # Add category if present
        if params.get('category'):
            filename_parts.append(f"cat_{params['category'].replace(' ', '_').lower()}")
        
        # Add date range if present
        if params.get('date_after'):
            filename_parts.append(f"after_{params['date_after']}")
        if params.get('date_before'):
            filename_parts.append(f"before_{params['date_before']}")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine filename parts
    filename_base = "_".join(filename_parts) if filename_parts else "pfd_reports"
    filename = f"{filename_base}_{prefix}_{timestamp}"
    
    col1, col2 = st.columns(2)
    
    # CSV Export
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Reports (CSV)",
            csv,
            f"{filename}.csv",
            "text/csv",
            key=f"download_csv_{prefix}_{timestamp}"
        )
    
    # Excel Export
    with col2:
        excel_data = export_to_excel(df)
        st.download_button(
            "📥 Download Reports (Excel)",
            excel_data,
            f"{filename}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_excel_{prefix}_{timestamp}"
        )
    
    # PDF Download
    st.subheader("Download PDFs")
    if st.button(f"Download all PDFs_{timestamp}", key=f"pdf_button_{prefix}_{timestamp}"):
        with st.spinner("Preparing PDF download..."):
            pdf_zip_path = f"{filename}_pdfs.zip"
            
            with zipfile.ZipFile(pdf_zip_path, 'w') as zipf:
                unique_pdfs = set()
                pdf_columns = [col for col in df.columns if col.startswith('PDF_') and col.endswith('_Path')]
                
                for col in pdf_columns:
                    paths = df[col].dropna()
                    unique_pdfs.update(paths)
                
                for pdf_path in unique_pdfs:
                    if pdf_path and os.path.exists(pdf_path):
                        zipf.write(pdf_path, os.path.basename(pdf_path))
            
            with open(pdf_zip_path, 'rb') as f:
                st.download_button(
                    "📦 Download All PDFs (ZIP)",
                    f.read(),
                    pdf_zip_path,
                    "application/zip",
                    key=f"download_pdfs_zip_{prefix}_{timestamp}"
                )
            
            # Cleanup zip file
            os.remove(pdf_zip_path)
    
def render_analysis_tab(data: pd.DataFrame):
    """Render the analysis tab with upload option"""
    st.header("Reports Analysis")
    
    # Add option to clear current data and upload new file
    if st.session_state.current_data is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            total_reports = len(st.session_state.current_data) if isinstance(st.session_state.current_data, pd.DataFrame) else 0
            data_source = st.session_state.data_source or "unknown source"
            st.info(f"Currently analyzing {total_reports} reports from {data_source}")
        with col2:
            if st.button("Clear Current Data"):
                st.session_state.current_data = None
                st.session_state.data_source = None
                st.session_state.scraped_data = None
                st.session_state.uploaded_data = None
                st.rerun()
    
    # Show file upload if no data or if data was cleared
    if st.session_state.current_data is None:
        upload_col1, upload_col2 = st.columns([3, 1])
        with upload_col1:
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel file", 
                type=['csv', 'xlsx']
            )
        
        if uploaded_file is not None:
            try:
                # Read the file based on extension
                if uploaded_file.name.lower().endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file type")
                    return
                
                # Required columns
                required_columns = [
                    'Title', 'URL', 'Content', 
                    'date_of_report', 'categories', 'coroner_area'
                ]
                
                # Check for missing columns
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.write("Available columns:", list(df.columns))
                    return
                
                # Process the data
                processed_df = process_scraped_data(df)
                
                # Update session state
                st.session_state.uploaded_data = processed_df.copy()
                st.session_state.current_data = processed_df.copy()
                st.session_state.data_source = 'uploaded'
                
                st.success(f"File uploaded successfully! Total reports: {len(processed_df)}")
                st.rerun()
                
            except Exception as read_error:
                st.error(f"Error reading file: {read_error}")
                logging.error(f"File read error: {read_error}", exc_info=True)
                return
        return
    
    # If we have data, validate it before proceeding
    try:
        is_valid, message = validate_data(data, "analysis")
        if not is_valid:
            st.error(message)
            return
            
        # Get date range for the data
        min_date = data['date_of_report'].min().date()
        max_date = data['date_of_report'].max().date()
        
        # Sidebar for filtering
        with st.sidebar:
            st.header("Analysis Filters")
            
            # Date range filter
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="date_range_filter"
            )
            
            # Category filter
            all_categories = set()
            for cats in data['categories'].dropna():
                if isinstance(cats, list):
                    all_categories.update(cats)
            
            selected_categories = st.multiselect(
                "Categories",
                options=sorted(all_categories),
                key="categories_filter"
            )
            
            # Coroner area filter
            coroner_areas = sorted(data['coroner_area'].dropna().unique())
            selected_areas = st.multiselect(
                "Coroner Areas",
                options=coroner_areas,
                key="areas_filter"
            )
        
        # Apply filters
        filtered_df = data.copy()
        
        # Date filter
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['date_of_report'].dt.date >= date_range[0]) &
                (filtered_df['date_of_report'].dt.date <= date_range[1])
            ]
        
        # Category filter
        if selected_categories:
            filtered_df = filtered_df[
                filtered_df['categories'].apply(
                    lambda x: bool(x) and any(cat in x for cat in selected_categories)
                )
            ]
        
        # Area filter
        if selected_areas:
            filtered_df = filtered_df[filtered_df['coroner_area'].isin(selected_areas)]
        
        # Show filter status
        active_filters = []
        if len(date_range) == 2 and (date_range[0] != min_date or date_range[1] != max_date):
            active_filters.append(f"Date range: {date_range[0]} to {date_range[1]}")
        if selected_categories:
            active_filters.append(f"Categories: {', '.join(selected_categories)}")
        if selected_areas:
            active_filters.append(f"Areas: {', '.join(selected_areas)}")
            
        if active_filters:
            st.info(f"Active filters: {' • '.join(active_filters)}")
        
        if len(filtered_df) == 0:
            st.warning("No data matches the selected filters.")
            return
            
        # Display filtered results count
        st.write(f"Showing {len(filtered_df)} of {len(data)} reports")
        
        # Overview metrics
        st.subheader("Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reports", len(filtered_df))
        with col2:
            st.metric("Unique Coroner Areas", filtered_df['coroner_area'].nunique())
        with col3:
            categories_count = sum(len(cats) if isinstance(cats, list) else 0 
                                 for cats in filtered_df['categories'].dropna())
            st.metric("Total Category Tags", categories_count)
        with col4:
            date_range_days = (filtered_df['date_of_report'].max() - filtered_df['date_of_report'].min()).days
            avg_reports_month = len(filtered_df) / (date_range_days / 30) if date_range_days > 0 else len(filtered_df)
            st.metric("Avg Reports/Month", f"{avg_reports_month:.1f}")
        
        # Visualizations
        st.subheader("Visualizations")
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Timeline",
            "Categories",
            "Coroner Areas",
            "Data Quality"
        ])
        
        with viz_tab1:
            try:
                plot_timeline(filtered_df)
            except Exception as e:
                st.error(f"Error creating timeline plot: {str(e)}")
        
        with viz_tab2:
            try:
                plot_category_distribution(filtered_df)
            except Exception as e:
                st.error(f"Error creating category distribution plot: {str(e)}")
        
        with viz_tab3:
            try:
                plot_coroner_areas(filtered_df)
            except Exception as e:
                st.error(f"Error creating coroner areas plot: {str(e)}")
                
        with viz_tab4:
            try:
                analyze_data_quality(filtered_df)
            except Exception as e:
                st.error(f"Error in data quality analysis: {str(e)}")
                logging.error(f"Data quality analysis error: {e}", exc_info=True)

    except Exception as e:
        st.error(f"An error occurred in the analysis tab: {str(e)}")
        logging.error(f"Analysis error: {e}", exc_info=True)


def export_to_excel(df: pd.DataFrame) -> bytes:
    """Handle Excel export with proper buffer management and error handling"""
    excel_buffer = io.BytesIO()
    
    try:
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        excel_data = excel_buffer.getvalue()
        return excel_data
        
    except Exception as e:
        logging.error(f"Error exporting to Excel: {e}")
        raise e
        
    finally:
        excel_buffer.close()


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

def generate_topic_label(topic_words):
    """Generate a meaningful label for a topic based on its top words"""
    return " & ".join([word for word, _ in topic_words[:3]]).title()

def format_topic_data(lda_model, vectorizer, doc_topics, df):
    """Format topic modeling results with clean display"""
    feature_names = vectorizer.get_feature_names_out()
    topics_data = []
    
    # Calculate document frequencies
    doc_freq = {}
    for doc in df['Content'].fillna(''):
        words = set(clean_text_for_modeling(doc).split())
        for word in words:
            doc_freq[word] = doc_freq.get(word, 0) + 1

    for idx, topic in enumerate(lda_model.components_):
        # Get top words with normalized weights
        top_word_indices = topic.argsort()[:-50-1:-1]
        topic_words = []
        
        for i in top_word_indices:
            word = feature_names[i]
            if len(word) > 1:  # Only include words longer than 1 character
                weight = float(topic[i])  # Keep as decimal for later processing
                count = sum(1 for doc in df['Content'].fillna('') 
                           if word in clean_text_for_modeling(doc).split())
                topic_words.append({
                    'word': word,
                    'weight': weight,
                    'count': count,
                    'documents': doc_freq.get(word, 0)
                })

        # Get related documents
        doc_scores = doc_topics[:, idx]
        doc_scores = doc_scores / doc_scores.sum() if doc_scores.sum() > 0 else doc_scores
        related_docs = []
        
        for doc_idx in doc_scores.argsort()[:-11:-1]:
            if doc_scores[doc_idx] > 0.01:  # At least 1% relevance
                doc = df.iloc[doc_idx]
                
                # Get other topics
                other_topics = []
                doc_topic_dist = doc_topics[doc_idx]
                doc_topic_dist = doc_topic_dist / doc_topic_dist.sum() if doc_topic_dist.sum() > 0 else doc_topic_dist
                
                for other_idx, score in enumerate(doc_topic_dist):
                    if other_idx != idx and score > 0.01:
                        other_label = ' & '.join([
                            feature_names[i] for i in 
                            lda_model.components_[other_idx].argsort()[:-4:-1]
                        ]).title()
                        other_topics.append({
                            'label': other_label,
                            'score': float(score)
                        })
                
                content = str(doc.get('Content', ''))
                related_docs.append({
                    'title': doc.get('Title', ''),
                    'date': doc.get('date_of_report', '').strftime('%Y-%m-%d') if pd.notna(doc.get('date_of_report')) else '',
                    'summary': content[:300] + '...' if len(content) > 300 else content,
                    'topicRelevance': float(doc_scores[doc_idx]),
                    'coroner': doc.get('coroner_name', ''),
                    'area': doc.get('coroner_area', ''),
                    'otherTopics': other_topics
                })

        # Calculate topic prevalence
        topic_prevalence = (doc_scores > 0.05).mean() * 100
        
        # Create topic label from top meaningful words
        meaningful_words = [
            feature_names[i] for i in top_word_indices[:5]
            if len(feature_names[i]) > 1
        ][:3]
        label = ' & '.join(meaningful_words).title()
        
        topics_data.append({
            'id': idx,
            'label': label,
            'description': f"Topic frequently mentions: {', '.join(meaningful_words[:5])}",
            'words': topic_words,
            'relatedReports': related_docs,
            'prevalence': round(topic_prevalence, 1),
            'trend': {
                'direction': 'stable',
                'percentage': 0,
                'previousMonths': [topic_prevalence] * 4
            }
        })
    
    return topics_data


def extract_key_points(text, point_type='findings'):
    """Extract key findings or recommendations from text"""
    text = text.lower()
    points = []
    
    # Define markers for findings and recommendations
    finding_markers = ['found:', 'finding:', 'findings:', 'identified:', 'noted:', 'concerns:']
    recommendation_markers = ['recommend:', 'recommendation:', 'recommendations:', 'action:', 'actions:']
    
    markers = finding_markers if point_type == 'findings' else recommendation_markers
    
    # Split text into sentences
    sentences = text.split('.')
    
    in_section = False
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Check if we're entering a relevant section
        if any(marker in sentence for marker in markers):
            in_section = True
            # Extract point after the marker
            for marker in markers:
                if marker in sentence:
                    point = sentence.split(marker)[-1].strip()
                    if point:
                        points.append(point.capitalize())
        
        # Add points from continuing lines in a section
        elif in_section and sentence:
            if len(sentence.split()) > 3:  # Only add if sentence is substantial
                points.append(sentence.capitalize())
        
        # Exit section if we hit another major section
        elif in_section and any(marker in sentence for marker in finding_markers + recommendation_markers):
            in_section = False
    
    # Clean up and limit points
    points = [p for p in points if len(p.split()) > 3][:5]  # Limit to 5 main points
    return points

def generate_topic_description(topic_words, topic_docs):
    """Generate a meaningful description for a topic"""
    words = [word for word, _ in topic_words[:5]]
    description = f"Topics related to {', '.join(words[:-1])} and {words[-1]}"
    
    if topic_docs:
        # Add context from documents if available
        common_areas = Counter(doc.get('area', '') for doc in topic_docs 
                             if doc.get('area')).most_common(3)
        if common_areas:
            areas = ', '.join(area for area, _ in common_areas if area)
            if areas:
                description += f". Frequently reported in {areas}"
    
    return description

def render_topic_modeling_tab(data: pd.DataFrame):
    """Render the topic modeling analysis tab"""
    st.header("Topic Modeling Analysis")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Topic Modeling Options")
        
        num_topics = st.slider(
            "Number of Topics",
            min_value=2,
            max_value=20,
            value=5,
            help="Select number of topics to extract"
        )
        
        max_features = st.slider(
            "Maximum Features",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Maximum number of words to include in analysis"
        )
        
        min_term_freq = st.number_input(
            "Minimum Term Frequency",
            min_value=1,
            value=2,
            help="Minimum number of occurrences required for a term"
        )
    
    # Run topic modeling
    if st.button("Extract Topics"):
        try:
            with st.spinner("Extracting topics from documents..."):
                # Extract topics
                result = extract_topics_lda(data, num_topics=num_topics, max_features=max_features)
                if not result:
                    st.error("Topic extraction failed. Please check your data.")
                    return
                
                lda_model, vectorizer, doc_topics = result
                
                # Format results for visualization
                topics_data = format_topic_data(
                    lda_model, 
                    vectorizer, 
                    doc_topics, 
                    data
                )
                
                # Display topics
                for topic in topics_data:
                    st.markdown(f"## Topic: {topic['label']} ({topic['prevalence']}% of reports)")
                    st.markdown(f"**Description:** {topic['description']}")
                    
                    # Key terms section
                    st.markdown("### Key Terms")
                    term_count = st.slider(
                        "Number of terms to show", 
                        5, 50, 10, 
                        key=f"terms_{topic['id']}"
                    )
                    
                    # Create term frequency table
                    term_data = pd.DataFrame(topic['words'][:term_count])
                    term_data['relevance'] = term_data['weight'].apply(lambda x: round(x * 100, 2))
                    
                    st.dataframe(
                        term_data,
                        column_config={
                            'word': st.column_config.TextColumn(
                                'Term',
                                width='medium'
                            ),
                            'count': st.column_config.NumberColumn(
                                'Occurrences',
                                width='small'
                            ),
                            'documents': st.column_config.NumberColumn(
                                'Documents',
                                help='Number of documents containing this term',
                                width='small'
                            ),
                            'relevance': st.column_config.ProgressColumn(
                                'Relevance',
                                help='Percentage relevance to the topic',
                                format="%.2f%%",
                                min_value=0,
                                max_value=100,
                                width='medium'
                            )
                        },
                        hide_index=True
                    )
                    
                    # Related reports section
                    st.markdown("### Related Reports")
                    report_tab1, report_tab2 = st.tabs(["Summary View", "Detailed View"])
                    
                    with report_tab1:
                        # Create a summary table
                        summary_data = [{
                            'Title': r['title'],
                            'Date': r['date'],
                            'Area': r['area'],
                            'Relevance': round(r['topicRelevance'] * 100, 2)
                        } for r in topic['relatedReports']]
                        
                        st.dataframe(
                            pd.DataFrame(summary_data),
                            column_config={
                                'Title': st.column_config.TextColumn('Title', width='large'),
                                'Date': st.column_config.TextColumn('Date', width='small'),
                                'Area': st.column_config.TextColumn('Area', width='medium'),
                                'Relevance': st.column_config.ProgressColumn(
                                    'Topic Relevance',
                                    format="%.2f%%",
                                    min_value=0,
                                    max_value=100,
                                    width='medium'
                                )
                            },
                            hide_index=True
                        )
                    
                    with report_tab2:
                        for report in topic['relatedReports']:
                            st.markdown(f"#### {report['title']}")
                            col1, col2, col3 = st.columns([2,2,1])
                            with col1:
                                st.markdown(f"**Date:** {report['date']}")
                            with col2:
                                st.markdown(f"**Coroner:** {report['coroner']}")
                            with col3:
                                st.markdown(f"**Area:** {report['area']}")
                            
                            st.markdown("**Summary:**")
                            st.markdown(report['summary'])
                            
                            if report['otherTopics']:
                                st.markdown("**Related Topics:**")
                                for other_topic in report['otherTopics']:
                                    st.markdown(
                                        f"- {other_topic['label']}: "
                                        f"{other_topic['score']*100:.2f}%"
                                    )
                            st.markdown("---")
                    
                    st.markdown("---")
                
                # Add download functionality
                st.markdown("### Export Analysis")
                if st.button("Download Analysis"):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Topics overview
                        topics_overview = pd.DataFrame([{
                            'Topic': t['label'],
                            'Description': t['description'],
                            'Prevalence': t['prevalence'],
                            'Top Terms': ', '.join([w['word'] for w in t['words'][:10]])
                        } for t in topics_data])
                        topics_overview.to_excel(writer, sheet_name='Topics Overview', index=False)
                        
                        # Terms by topic
                        for topic in topics_data:
                            terms_df = pd.DataFrame([{
                                'Term': w['word'],
                                'Relevance': f"{w['weight']*100:.2f}%",
                                'Occurrences': w['count'],
                                'Documents': w['documents']
                            } for w in topic['words']])
                            terms_df.to_excel(
                                writer, 
                                sheet_name=f"Terms_Topic_{topic['id']}",
                                index=False
                            )
                            
                            # Related reports
                            reports_df = pd.DataFrame([{
                                'Title': r['title'],
                                'Date': r['date'],
                                'Area': r['area'],
                                'Relevance': f"{r['topicRelevance']*100:.2f}%"
                            } for r in topic['relatedReports']])
                            reports_df.to_excel(
                                writer,
                                sheet_name=f"Reports_Topic_{topic['id']}",
                                index=False
                            )
                    
                    st.download_button(
                        "📥 Download Analysis (Excel)",
                        output.getvalue(),
                        "topic_analysis.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
        except Exception as e:
            st.error(f"Error during topic modeling: {str(e)}")
            logging.error(f"Topic modeling error: {e}", exc_info=True)



def main():
    initialize_session_state()
    st.title("UK Judiciary PFD Reports Analysis")
    st.markdown("""
    This application allows you to analyze Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    You can either scrape new reports or analyze existing data.
    """)
    
    # Create separate tab selection to avoid key conflicts
    current_tab = st.radio(
        "Select section:",
        ["🔍 Scrape Reports", "📊 Analysis", "🔬 Topic Modeling"],
        label_visibility="collapsed",
        horizontal=True,
        key="main_tab_selector"
    )
    
    st.markdown("---")  # Add separator
    
    # Handle tab content
    if current_tab == "🔍 Scrape Reports":
        render_scraping_tab()
    
    elif current_tab == "📊 Analysis":
        if hasattr(st.session_state, 'current_data'):
            render_analysis_tab(st.session_state.current_data)
        else:
            st.warning("No data available. Please scrape reports or upload a file.")
    
    elif current_tab == "🔬 Topic Modeling":
        if not hasattr(st.session_state, 'current_data') or st.session_state.current_data is None:
            st.warning("No data available. Please scrape reports or upload a file first.")
            return
            
        try:
            is_valid, message = validate_data(st.session_state.current_data, "topic_modeling")
            if is_valid:
                render_topic_modeling_tab(st.session_state.current_data)
            else:
                st.error(message)
        except Exception as e:
            st.error(f"Error in topic modeling: {e}")
            logging.error(f"Topic modeling error: {e}", exc_info=True)
    
    # Add footer
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center'>
        <p>Built with Streamlit • Data from UK Judiciary</p>
        </div>""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
