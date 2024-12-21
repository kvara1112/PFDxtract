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
import plotly.graph_objects as go

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
    order: str = "relevance",
    max_pages: Optional[int] = None
) -> List[Dict]:
    """
    Scrape PFD reports with enhanced progress tracking
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
        
        # Create containers for progress tracking
        progress_container = st.empty()
        status_container = st.empty()
        report_container = st.empty()
        page_reports = []
        total_reports_processed = 0
        
        for current_page in range(1, total_pages + 1):
            try:
                # Check if scraping should be stopped
                if hasattr(st.session_state, 'stop_scraping') and st.session_state.stop_scraping:
                    status_container.warning("Scraping stopped by user")
                    break
                
                # Update page progress
                page_progress = (current_page - 1) / total_pages
                progress_container.progress(page_progress)
                status_container.write(f"📄 Processing page {current_page} of {total_pages}")
                
                # Construct page URL
                page_url = construct_search_url(
                    base_url=base_url,
                    keyword=keyword,
                    category=category,
                    category_slug=category_slug,
                    page=current_page
                )
                
                # Scrape current page
                current_page_reports = scrape_page(page_url)
                
                if current_page_reports:
                    for report in current_page_reports:
                        total_reports_processed += 1
                        page_reports.append(report)
                        
                        # Update report progress
                        report_container.write(f"📑 Retrieved: {report.get('Title', 'Untitled Report')} ({total_reports_processed} reports processed)")
                        
                        # Update overall progress
                        overall_progress = (current_page - 1 + total_reports_processed/len(current_page_reports)) / total_pages
                        progress_container.progress(min(overall_progress, 1.0))
                else:
                    status_container.warning(f"No results found on page {current_page}")
                    continue
                
                # Add delay between pages
                time.sleep(2)
                
            except Exception as page_error:
                logging.error(f"Error processing page {current_page}: {page_error}")
                status_container.warning(f"Error on page {current_page}. Continuing with next page...")
                continue
        
        all_reports.extend(page_reports)
        
        # Clear progress indicators and show final status
        progress_container.empty()
        status_container.empty()
        report_container.empty()
        
        st.success(f"Successfully scraped {len(all_reports)} reports")
        
        # Sort results if specified
        if order != "relevance":
            all_reports = sort_reports(all_reports, order)
        
        return all_reports
        
    except Exception as e:
        logging.error(f"Error in scrape_pfd_reports: {e}")
        st.error(f"An error occurred while scraping reports: {e}")
        return all_reports

def filter_reports_by_date(reports: List[Dict], date_after: Optional[str] = None, date_before: Optional[str] = None) -> List[Dict]:
    """
    Filter reports by date range with improved date parsing
    
    Args:
        reports: List of report dictionaries
        date_after: Start date in DD/MM/YYYY format
        date_before: End date in DD/MM/YYYY format
        
    Returns:
        List of filtered reports
    """
    if not reports or (not date_after and not date_before):
        return reports
        
    filtered_reports = []
    
    try:
        # Convert filter dates to datetime objects
        after_date = None
        before_date = None
        
        if date_after:
            after_date = datetime.strptime(date_after, '%d/%m/%Y')
        if date_before:
            before_date = datetime.strptime(date_before, '%d/%m/%Y')
        
        for report in reports:
            # Extract report date
            content = str(report.get('Content', ''))
            date_match = re.search(r'Date of report:\s*(\d{1,2}(?:/|-)\d{1,2}(?:/|-)\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})', content)
            
            if not date_match:
                continue
                
            date_str = date_match.group(1)
            
            try:
                # Handle different date formats
                if '/' in date_str:
                    report_date = datetime.strptime(date_str, '%d/%m/%Y')
                else:
                    # Remove ordinal indicators
                    date_str = re.sub(r'(?<=\d)(st|nd|rd|th)', '', date_str)
                    try:
                        report_date = datetime.strptime(date_str, '%d %B %Y')
                    except ValueError:
                        report_date = datetime.strptime(date_str, '%d %b %Y')
                
                # Apply date filters
                if after_date and report_date < after_date:
                    continue
                if before_date and report_date > before_date:
                    continue
                    
                filtered_reports.append(report)
                
            except ValueError as e:
                logging.warning(f"Could not parse date '{date_str}': {e}")
                continue
                
        return filtered_reports
        
    except Exception as e:
        logging.error(f"Error in date filtering: {e}")
        return reports

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

def render_topic_modeling_tab(data: pd.DataFrame) -> None:
    """Enhanced topic modeling analysis for PFD reports."""
    st.header("Topic Modeling Analysis")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Model Parameters")
        
        num_topics = st.slider(
            "Number of Topics",
            min_value=2,
            max_value=20,
            value=8,
            help="Select number of topics to extract"
        )
        
        max_features = st.slider(
            "Maximum Features",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            help="Maximum number of terms to include"
        )
        
        with st.expander("Advanced Settings"):
            min_df = st.slider(
                "Minimum Document Frequency",
                min_value=1,
                max_value=10,
                value=2,
                help="Minimum number of documents a term must appear in"
            )
    
    # Run topic modeling
    if st.button("Extract Topics", type="primary"):
        try:
            with st.spinner("Analyzing document topics..."):
                # Prepare documents
                documents = []
                for idx, row in data.iterrows():
                    doc_text = combine_document_text(row)
                    if doc_text.strip():
                        documents.append(clean_text_for_modeling(doc_text))
                
                # Create document-term matrix
                vectorizer = CountVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    stop_words='english'
                )
                
                dtm = vectorizer.fit_transform(documents)
                feature_names = vectorizer.get_feature_names_out()
                
                # Train LDA model
                lda = LatentDirichletAllocation(
                    n_components=num_topics,
                    random_state=42,
                    n_jobs=-1
                )
                
                doc_topics = lda.fit_transform(dtm)
                
                # Store results
                st.session_state.topic_results = {
                    'model': lda,
                    'vectorizer': vectorizer,
                    'doc_topics': doc_topics,
                    'feature_names': feature_names,
                    'documents': documents
                }
                
                # Create visualization tabs
                topic_tab, docs_tab, network_tab = st.tabs([
                    "Topic Overview",
                    "Document Analysis",
                    "Topic Network"
                ])
                
                with topic_tab:
                    st.markdown("### Topics Overview")
                    
                    # Display topics
                    for topic_idx in range(num_topics):
                        # Get top words for topic
                        top_words = get_top_words(lda, feature_names, topic_idx)
                        
                        # Calculate topic prevalence
                        topic_prev = (doc_topics[:, topic_idx] > 0.2).mean() * 100
                        
                        with st.expander(f"Topic {topic_idx + 1}: {' - '.join(top_words[:3])}"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Show word cloud or bar chart
                                words_df = pd.DataFrame({
                                    'Word': top_words,
                                    'Weight': lda.components_[topic_idx][
                                        np.argsort(lda.components_[topic_idx])[-10:]
                                    ]
                                })
                                
                                fig = px.bar(
                                    words_df,
                                    x='Weight',
                                    y='Word',
                                    orientation='h',
                                    title='Top Terms'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.metric(
                                    "Topic Prevalence",
                                    f"{topic_prev:.1f}%"
                                )
                                
                                # Get representative docs
                                top_doc_indices = doc_topics[:, topic_idx].argsort()[-3:][::-1]
                                st.markdown("#### Representative Documents")
                                
                                for idx in top_doc_indices:
                                    doc_title = data.iloc[idx].get('Title', f'Document {idx}')
                                    doc_score = doc_topics[idx, topic_idx]
                                    st.markdown(f"- {doc_title} ({doc_score:.1%})")
                
                with docs_tab:
                    st.markdown("### Document-Topic Distribution")
                    
                    # Create heatmap
                    doc_ids = [f"Doc {i+1}" for i in range(len(documents))]
                    topic_labels = [f"Topic {i+1}" for i in range(num_topics)]
                    
                    fig = px.imshow(
                        doc_topics,
                        labels=dict(x="Topics", y="Documents", color="Weight"),
                        x=topic_labels,
                        y=doc_ids,
                        aspect="auto"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show document assignments
                    st.markdown("### Primary Topic Assignments")
                    assignments = pd.DataFrame({
                        'Document': data['Title'].values,
                        'Primary Topic': [f"Topic {i+1}" for i in doc_topics.argmax(axis=1)],
                        'Confidence': doc_topics.max(axis=1)
                    })
                    
                    st.dataframe(
                        assignments.sort_values('Confidence', ascending=False),
                        hide_index=True
                    )
                
                with network_tab:
                    st.markdown("### Topic Similarity Network")
                    
                    # Calculate topic similarities
                    topic_sims = cosine_similarity(lda.components_)
                    
                    # Create network graph
                    G = nx.Graph()
                    
                    # Add nodes
                    for i in range(num_topics):
                        top_words = get_top_words(lda, feature_names, i)
                        G.add_node(i, name=f"Topic {i+1}\n{' - '.join(top_words[:3])}")
                    
                    # Add edges
                    for i in range(num_topics):
                        for j in range(i+1, num_topics):
                            if topic_sims[i,j] > 0.2:  # Similarity threshold
                                G.add_edge(i, j, weight=float(topic_sims[i,j]))
                    
                    # Create network visualization
                    pos = nx.spring_layout(G)
                    
                    # Create plotly figure
                    edge_trace = create_edge_trace(G, pos)
                    node_trace = create_node_trace(G, pos)
                    
                    fig = go.Figure(
                        data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40)
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add export options
                st.markdown("### Export Results")
                
                # Prepare export data
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Topics overview
                    topics_df = pd.DataFrame([
                        {
                            'Topic': f"Topic {i+1}",
                            'Top Words': ', '.join(get_top_words(lda, feature_names, i)),
                            'Prevalence': (doc_topics[:, i] > 0.2).mean() * 100
                        }
                        for i in range(num_topics)
                    ])
                    topics_df.to_excel(writer, sheet_name='Topics Overview', index=False)
                    
                    # Document-topic distributions
                    doc_topics_df = pd.DataFrame(
                        doc_topics,
                        columns=[f"Topic {i+1}" for i in range(num_topics)]
                    )
                    doc_topics_df.to_excel(writer, sheet_name='Document-Topic Distribution')
                
                # Download button
                st.download_button(
                    "📥 Download Analysis (Excel)",
                    output.getvalue(),
                    "topic_analysis.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        except Exception as e:
            st.error(f"Error in topic modeling: {str(e)}")
            logging.error(f"Topic modeling error: {e}", exc_info=True)

def get_top_words(model, feature_names, topic_idx, n_words=10):
    """Get top words for a topic."""
    return [feature_names[i] for i in model.components_[topic_idx].argsort()[:-n_words-1:-1]]

def create_edge_trace(G, pos):
    """Create edge trace for network visualization."""
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    return go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

def create_node_trace(G, pos):
    """Create node trace for network visualization."""
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['name'])
    
    return go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=20,
            line=dict(width=2),
            color='lightblue'
        )
    )

def format_topics_for_display(topic_insights):
    """Format topic insights for the React component"""
    return [{
        'id': topic['id'],
        'label': topic['label'],
        'prevalence': topic['prevalence'],
        'description': topic['description'],
        'words': [{
            'word': word['word'],
            'weight': word['weight'],
            'count': word.get('docs', 0)
        } for word in topic['words']],
        'relatedDocs': [{
            'title': doc['title'],
            'date': doc['date'].strftime('%Y-%m-%d') if isinstance(doc['date'], pd.Timestamp) else str(doc['date']),
            'relevance': doc['relevance'],
            'summary': doc['summary']
        } for doc in topic['representativeDocs']]
    } for topic in topic_insights]

def export_topic_analysis(topic_insights, data):
    """Export topic analysis to Excel"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Overview sheet
        overview_data = []
        for topic in topic_insights:
            overview_data.append({
                'Topic': topic['label'],
                'Prevalence': topic['prevalence'],
                'Description': topic['description'],
                'Top Words': ', '.join([w['word'] for w in topic['words'][:10]])
            })
        
        pd.DataFrame(overview_data).to_excel(
            writer,
            sheet_name='Topics Overview',
            index=False
        )
        
        # Detailed sheets for each topic
        for topic in topic_insights:
            # Words sheet
            words_df = pd.DataFrame([{
                'Word': w['word'],
                'Weight': w['weight'],
                'Document Frequency': w.get('docs', 0)
            } for w in topic['words']])
            
            words_df.to_excel(
                writer,
                sheet_name=f'Topic_{topic["id"]}_Words',
                index=False
            )
            
            # Documents sheet
            docs_df = pd.DataFrame([{
                'Title': doc['title'],
                'Date': doc['date'],
                'Relevance': doc['relevance'],
                'Summary': doc['summary']
            } for doc in topic['representativeDocs']])
            
            docs_df.to_excel(
                writer,
                sheet_name=f'Topic_{topic["id"]}_Docs',
                index=False
            )
    
    return output.getvalue()

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


def generate_topic_label(topic_words):
    """Generate a meaningful label for a topic based on its top words"""
    return " & ".join([word for word, _ in topic_words[:3]]).title()


def export_to_excel(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to Excel bytes with proper formatting
    extract_topics_lda
    Args:
        df: DataFrame to export
        
    Returns:
        bytes: Excel file content as bytes
    """
    try:
        # Create output buffer
        output = io.BytesIO()
        
        # Create Excel writer
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Convert main data to Excel
            df.to_excel(writer, sheet_name='Reports', index=False)
            
            # Get the worksheet
            worksheet = writer.sheets['Reports']
            
            # Auto-adjust column widths based on content
            for idx, col in enumerate(df.columns, 1):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                # Add a little extra space and limit maximum width
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[chr(64 + idx)].width = adjusted_width
            
            # Add filters to header row
            worksheet.auto_filter.ref = worksheet.dimensions
            
            # Freeze the header row
            worksheet.freeze_panes = 'A2'
        
        # Get the bytes value
        excel_data = output.getvalue()
        
        return excel_data
        
    except Exception as e:
        logging.error(f"Error exporting to Excel: {e}")
        raise Exception(f"Failed to export data to Excel: {str(e)}")
        
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

def extract_topics_lda(data: pd.DataFrame, num_topics: int = 5, max_features: int = 1000):
    """Extract topics using LDA"""
    try:
        # Filter out rows with no content
        valid_data = data[data['Content'].notna()].copy()
        
        # Combine and clean texts
        texts = []
        for _, row in valid_data.iterrows():
            combined_text = combine_document_text(row)
            cleaned_text = clean_text_for_modeling(combined_text)
            if cleaned_text and len(cleaned_text.split()) > 3:
                texts.append(cleaned_text)
        
        # Configure and fit vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.95,
            stop_words='english',
            token_pattern=r'(?u)\b[a-z]{2,}\b'
        )
        dtm = vectorizer.fit_transform(texts)
        
        # Configure and fit LDA model
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            n_jobs=-1,
            max_iter=20,
            learning_method='batch',
            doc_topic_prior=0.1,
            topic_word_prior=0.01
        )
        
        # Fit model and get document-topic distributions
        doc_topic_dist = lda_model.fit_transform(dtm)
        
        # Prepare visualization data
        feature_names = vectorizer.get_feature_names_out()
        doc_lengths = np.array(dtm.sum(axis=1)).ravel()
        term_frequency = np.array(dtm.sum(axis=0)).ravel()
        
        prepared_data = pyLDAvis.prepare(
            lda_model.components_,
            doc_topic_dist,
            doc_lengths,
            feature_names,
            term_frequency,
            sort_topics=False,
            mds='mmds'
        )
        
        return lda_model, vectorizer, doc_topic_dist, prepared_data
        
    except Exception as e:
        st.error(f"Error in topic extraction: {str(e)}")
        logging.error(f"Topic extraction error: {e}", exc_info=True)
        raise e

def extract_advanced_topics(data: pd.DataFrame, num_topics: int = 5, max_features: int = 1000, min_df: int = 2, n_iterations: int = 20):
    """Extract topics with improved preprocessing and error handling"""
    try:
        # Create a copy of the data to avoid modifying the original
        working_data = data.copy()
        
        # Ensure 'Content' column exists
        if 'Content' not in working_data.columns:
            raise ValueError("DataFrame must contain a 'Content' column")
        
        # Filter valid data first
        valid_data = working_data[working_data['Content'].notna()].copy()
        if len(valid_data) == 0:
            raise ValueError("No valid content found in the dataset")
        
        # Combine and clean texts
        texts = []
        doc_mapping = []  # Keep track of original document indices
        
        for idx, row in valid_data.iterrows():
            combined_text = combine_document_text(row)
            cleaned_text = clean_text_for_modeling(combined_text)
            if cleaned_text and len(cleaned_text.split()) > 3:
                texts.append(cleaned_text)
                doc_mapping.append(idx)
        
        if not texts:
            raise ValueError("No valid documents found after cleaning")
        
        # Configure vectorizer with error checking
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min(min_df, len(texts)),  # Ensure min_df doesn't exceed document count
            max_df=0.95,
            stop_words='english',
            token_pattern=r'(?u)\b[a-z]{2,}\b'
        )
        
        # Create document-term matrix
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError as e:
            raise ValueError(f"Error in document vectorization: {str(e)}")
        
        # Verify we have enough features
        if dtm.shape[1] < num_topics:
            num_topics = max(2, dtm.shape[1] - 1)  # Adjust number of topics if necessary
            st.warning(f"Adjusted number of topics to {num_topics} due to limited vocabulary")
        
        # Configure and fit LDA model
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            n_jobs=-1,
            max_iter=n_iterations,
            learning_method='batch',
            doc_topic_prior=0.1,
            topic_word_prior=0.01
        )
        
        # Fit model and get document-topic distributions
        doc_topic_dist = lda_model.fit_transform(dtm)
        
        # Normalize components
        for idx in range(len(lda_model.components_)):
            lda_model.components_[idx] = lda_model.components_[idx] / lda_model.components_[idx].sum()
        
        # Prepare visualization data
        feature_names = vectorizer.get_feature_names_out()
        doc_lengths = np.array(dtm.sum(axis=1)).ravel()
        term_frequency = np.array(dtm.sum(axis=0)).ravel()
        
        # Format data for pyLDAvis
        prepared_data = pyLDAvis.prepare(
            lda_model.components_,
            doc_topic_dist,
            doc_lengths,
            feature_names,
            term_frequency,
            sort_topics=False,
            mds='mmds'
        )
        
        return lda_model, vectorizer, doc_topic_dist, prepared_data
        
    except Exception as e:
        logging.error(f"Error in topic extraction: {str(e)}", exc_info=True)
        raise Exception(f"Topic modeling failed: {str(e)}")

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



def format_topic_data(lda_model, vectorizer, doc_topics, df):
    """Format topic modeling results for display"""
    try:
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
                if len(word) > 1:
                    weight = float(topic[i])
                    count = doc_freq.get(word, 0)
                    topic_words.append({
                        'word': word,
                        'weight': weight,
                        'count': count,
                        'documents': doc_freq.get(word, 0)
                    })
            
            # Format topics data
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
                'prevalence': round((doc_topics[:, idx] > 0.05).mean() * 100, 1)
            })
        
        return topics_data
        
    except Exception as e:
        st.error(f"Error formatting topic data: {str(e)}")
        logging.error(f"Topic formatting error: {e}", exc_info=True)
        return []


def render_topic_visualization(vis_data: pyLDAvis._prepare.PreparedData) -> None:
    """
    Render the pyLDAvis visualization in Streamlit.
    
    Args:
        vis_data: Prepared visualization data from pyLDAvis
    """
    st.markdown("""
        ### Interactive Topic Visualization
        
        This visualization shows the relationships between topics and terms:
        - **Left panel**: Topic bubbles sized by prevalence
        - **Right panel**: Term frequencies and relevance
        - **Interactions**:
            - Click topics to see associated terms
            - Adjust λ slider to change term relevance weighting
            - Hover over terms for detailed statistics
    """)
    
    # Add custom CSS for better visualization display
    st.markdown("""
        <style>
            #ldavis_container {
                width: 100% !important;
                height: 800px !important;
                overflow: hidden;
                margin: 20px 0;
            }
            .ldavis_container svg {
                max-width: 100%;
                height: auto;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Convert visualization to HTML and render
    html_string = pyLDAvis.prepared_data_to_html(vis_data)
    components.html(html_string, width=1300, height=800)

def render_topic_modeling_tab(data: pd.DataFrame) -> None:
    """Enhanced topic modeling analysis for PFD reports."""
    st.header("Topic Modeling Analysis")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Model Parameters")
        
        num_topics = st.slider(
            "Number of Topics",
            min_value=2,
            max_value=20,
            value=8,
            help="Select number of topics to extract"
        )
        
        max_features = st.slider(
            "Maximum Features",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            help="Maximum number of terms to include"
        )
        
        with st.expander("Advanced Settings"):
            min_df = st.slider(
                "Minimum Document Frequency",
                min_value=1,
                max_value=10,
                value=2,
                help="Minimum number of documents a term must appear in"
            )
    
    # Run topic modeling
    if st.button("Extract Topics", type="primary"):
        try:
            with st.spinner("Analyzing document topics..."):
                # Prepare documents
                valid_data = data[data['Content'].notna()].copy()
                
                # Combine text from content and PDFs
                documents = []
                for _, row in valid_data.iterrows():
                    doc_text = combine_document_text(row)
                    if doc_text.strip():
                        cleaned_text = clean_text_for_modeling(doc_text)
                        if cleaned_text.strip():
                            documents.append(cleaned_text)
                
                if not documents:
                    st.error("No valid documents found for analysis")
                    return
                
                # Create document-term matrix
                vectorizer = CountVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    stop_words='english'
                )
                
                try:
                    dtm = vectorizer.fit_transform(documents)
                except ValueError as e:
                    st.error(f"Error creating document-term matrix: {str(e)}")
                    return
                
                feature_names = vectorizer.get_feature_names_out()
                
                # Train LDA model
                lda = LatentDirichletAllocation(
                    n_components=num_topics,
                    random_state=42,
                    n_jobs=-1
                )
                
                doc_topics = lda.fit_transform(dtm)
                
                # Store results
                st.session_state.topic_results = {
                    'model': lda,
                    'vectorizer': vectorizer,
                    'doc_topics': doc_topics,
                    'feature_names': feature_names,
                    'documents': documents,
                    'valid_indices': valid_data.index.tolist()
                }
                
                # Create visualization tabs
                topic_tab, docs_tab, network_tab = st.tabs([
                    "Topic Overview",
                    "Document Analysis",
                    "Topic Network"
                ])
                
                with topic_tab:
                    st.markdown("### Topics Overview")
                    
                    # Display topics
                    for topic_idx in range(num_topics):
                        # Get top words for topic
                        top_words = get_top_words(lda, feature_names, topic_idx)
                        
                        # Calculate topic prevalence
                        topic_prev = (doc_topics[:, topic_idx] > 0.2).mean() * 100
                        
                        with st.expander(f"Topic {topic_idx + 1}: {' - '.join(top_words[:3])}"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Show word distribution
                                word_weights = lda.components_[topic_idx]
                                top_word_indices = word_weights.argsort()[:-10-1:-1]
                                
                                words_df = pd.DataFrame({
                                    'Word': [feature_names[i] for i in top_word_indices],
                                    'Weight': word_weights[top_word_indices]
                                })
                                
                                fig = px.bar(
                                    words_df,
                                    x='Weight',
                                    y='Word',
                                    orientation='h',
                                    title='Top Terms'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.metric(
                                    "Topic Prevalence",
                                    f"{topic_prev:.1f}%"
                                )
                                
                                # Get representative docs
                                top_doc_indices = doc_topics[:, topic_idx].argsort()[-3:][::-1]
                                st.markdown("#### Representative Documents")
                                
                                for doc_idx in top_doc_indices:
                                    actual_idx = valid_data.index[doc_idx]
                                    doc_title = valid_data.loc[actual_idx, 'Title']
                                    doc_score = doc_topics[doc_idx, topic_idx]
                                    st.markdown(f"- {doc_title} ({doc_score:.1%})")
                
                with docs_tab:
                    st.markdown("### Document-Topic Distribution")
                    
                    # Create heatmap
                    doc_labels = [f"Doc {i+1}" for i in range(len(documents))]
                    topic_labels = [f"Topic {i+1}" for i in range(num_topics)]
                    
                    fig = px.imshow(
                        doc_topics,
                        labels=dict(x="Topics", y="Documents", color="Weight"),
                        x=topic_labels,
                        y=doc_labels,
                        aspect="auto"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show document assignments
                    assignments = pd.DataFrame({
                        'Document': valid_data['Title'].values,
                        'Primary Topic': [f"Topic {i+1}" for i in doc_topics.argmax(axis=1)],
                        'Confidence': doc_topics.max(axis=1)
                    })
                    
                    st.dataframe(
                        assignments.sort_values('Confidence', ascending=False),
                        hide_index=True
                    )
                
                with network_tab:
                    st.markdown("### Topic Similarity Network")
                    
                    # Calculate topic similarities
                    topic_sims = cosine_similarity(lda.components_)
                    
                    # Create network graph
                    G = nx.Graph()
                    
                    # Add nodes
                    for i in range(num_topics):
                        top_words = get_top_words(lda, feature_names, i)
                        G.add_node(i, name=f"Topic {i+1}\n{' - '.join(top_words[:3])}")
                    
                    # Add edges
                    for i in range(num_topics):
                        for j in range(i+1, num_topics):
                            if topic_sims[i,j] > 0.2:  # Similarity threshold
                                G.add_edge(i, j, weight=float(topic_sims[i,j]))
                    
                    # Create network visualization
                    pos = nx.spring_layout(G)
                    
                    # Create plotly figure
                    edge_trace = create_edge_trace(G, pos)
                    node_trace = create_node_trace(G, pos)
                    
                    fig = go.Figure(
                        data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40)
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add export options
                st.markdown("### Export Results")
                
                # Prepare export data
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Topics overview
                    topics_df = pd.DataFrame([
                        {
                            'Topic': f"Topic {i+1}",
                            'Top Words': ', '.join(get_top_words(lda, feature_names, i)),
                            'Prevalence': (doc_topics[:, i] > 0.2).mean() * 100
                        }
                        for i in range(num_topics)
                    ])
                    topics_df.to_excel(writer, sheet_name='Topics Overview', index=False)
                    
                    # Document-topic distributions
                    doc_topics_df = pd.DataFrame(
                        doc_topics,
                        columns=[f"Topic {i+1}" for i in range(num_topics)]
                    )
                    doc_topics_df['Document'] = valid_data['Title'].values
                    doc_topics_df.set_index('Document').to_excel(
                        writer,
                        sheet_name='Document-Topic Distribution'
                    )
                
                # Download button
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
