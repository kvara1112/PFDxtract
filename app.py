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
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
import torch
from collections import Counter

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
}

def make_request(url: str, retries: int = 3, delay: int = 2) -> Optional[requests.Response]:
    """Make HTTP request with retries and delay"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Referer': 'https://judiciary.uk/'
    }
    
    for attempt in range(retries):
        try:
            time.sleep(delay)  # Add delay between requests
            response = requests.get(url, headers=headers, verify=False, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"Request failed: {str(e)}")
                raise e
            time.sleep(delay * (attempt + 1))
    return None

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
        text = ''.join(char if char.isprintable() or char == '\n' else ' ' for char in text)
        
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
        # Extract date
        date_match = re.search(r'Date of report:\s*(\d{1,2}/\d{1,2}/\d{4})', content)
        if date_match:
            try:
                date_str = date_match.group(1)
                datetime.strptime(date_str, '%d/%m/%Y')  # Validate date format
                metadata['date_of_report'] = date_str
            except ValueError:
                logging.warning(f"Invalid date format found: {date_match.group(1)}")
        
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
            metadata['categories'] = [cat.strip() for cat in categories if cat.strip()]
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
        return metadata

def get_pfd_categories() -> List[str]:
    """Get all available PFD report categories"""
    return [
        "accident-at-work-and-health-and-safety-related-deaths",
        "alcohol-drug-and-medication-related-deaths",
        "care-home-health-related-deaths",
        "child-death-from-2015",
        "community-health-care-and-emergency-services-related-deaths",
        "emergency-services-related-deaths-2019-onwards",
        "hospital-death-clinical-procedures-and-medical-management-related-deaths",
        "mental-health-related-deaths",
        "other-related-deaths",
        "police-related-deaths",
        "product-related-deaths",
        "railway-related-deaths",
        "road-highways-safety-related-deaths",
        "service-personnel-related-deaths",
        "state-custody-related-deaths",
        "suicide-from-2015",
        "wales-prevention-of-future-deaths-reports-2019-onwards"
    ]

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
        
        # Extract text content
        paragraphs = content.find_all(['p', 'table'])
        webpage_text = '\n\n'.join(p.get_text(strip=True, separator=' ') for p in paragraphs)
        
        pdf_contents = []
        pdf_paths = []
        pdf_names = []
        
        # Find PDF links
        pdf_links = (
            soup.find_all('a', class_='related-content__link', href=re.compile(r'\.pdf$')) or
            soup.find_all('a', href=re.compile(r'\.pdf$'))
        )
        
        for pdf_link in pdf_links:
            pdf_url = pdf_link['href']
            
            # Handle relative URLs
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
                title_elem = card.find('h3', class_='card__title').find('a')
                if not title_elem:
                    continue
                
                title = clean_text(title_elem.text)
                card_url = title_elem['href']
                
                logging.info(f"Processing report: {title}")
                
                if not card_url.startswith(('http://', 'https://')):
                    card_url = f"https://www.judiciary.uk{card_url}"
                
                content_data = get_report_content(card_url)
                
                if content_data:
                    report = {
                        'Title': title,
                        'URL': card_url,
                        'Content': content_data['content']
                    }
                    
                    # Add PDF data
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

def get_total_pages(url: str) -> int:
    """Get total number of pages"""
    try:
        response = make_request(url)
        if not response:
            return 0
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check pagination
        pagination = soup.find('nav', class_='navigation pagination')
        if pagination:
            page_numbers = pagination.find_all('a', class_='page-numbers')
            numbers = [int(p.text.strip()) for p in page_numbers if p.text.strip().isdigit()]
            if numbers:
                return max(numbers)
        
        # Check if at least one page of results exists
        results = soup.find('ul', class_='search__list')
        if results and results.find_all('div', class_='card'):
            return 1
            
        return 0
        
    except Exception as e:
        logging.error(f"Error getting total pages: {e}")
        return 0

def scrape_pfd_reports(keyword: Optional[str] = None,
                      category: Optional[str] = None,
                      date_after: Optional[str] = None,
                      date_before: Optional[str] = None,
                      order: str = "relevance",
                      max_pages: Optional[int] = None) -> List[Dict]:
    """Scrape PFD reports with comprehensive filtering"""
    all_reports = []
    current_page = 1
    base_url = "https://www.judiciary.uk"
    
    # Build query parameters
    params = {
        'post_type': 'pfd',
        'order': order
    }
    
    if keyword and keyword.strip():
        params['s'] = keyword.strip()
    if category:
        params['pfd_report_type'] = category
    
    # Handle date parameters
    if date_after:
        try:
            day, month, year = date_after.split('/')
            params['after-year'] = year
            params['after-month'] = month
            params['after-day'] = day
        except ValueError as e:
            logging.error(f"Invalid date_after format: {e}")
            return []
    
    if date_before:
        try:
            day, month, year = date_before.split('/')
            params['before-year'] = year
            params['before-month'] = month
            params['before-day'] = day
        except ValueError as e:
            logging.error(f"Invalid date_before format: {e}")
            return []
    
    # Build initial URL
    param_strings = [f"{k}={v}" for k, v in params.items()]
    initial_url = f"{base_url}/?{'&'.join(param_strings)}"
    
    st.write(f"Searching URL: {initial_url}")
    
    try:
        total_pages = get_total_pages(initial_url)
        if total_pages == 0:
            st.warning("No results found")
            return []
            
        logging.info(f"Total pages to scrape: {total_pages}")
        
        if max_pages:
            total_pages = min(total_pages, max_pages)
        
        # Setup progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while current_page <= total_pages:
            # Build page URL
            page_url = initial_url if current_page == 1 else f"{base_url}/page/{current_page}/?{'&'.join(param_strings)}"
            
            # Update progress
            status_text.text(f"Scraping page {current_page} of {total_pages}...")
            progress_bar.progress(current_page / total_pages)
            
            # Scrape page
            reports = scrape_page(page_url)
            
            if reports:
                all_reports.extend(reports)
                logging.info(f"Found {len(reports)} reports on page {current_page}")
            else:
                logging.warning(f"No reports found on page {current_page}")
                if current_page > 1:
                    break
            
            current_page += 1
        
        progress_bar.progress(1.0)
        status_text.text(f"Completed! Total reports found: {len(all_reports)}")
        
        return all_reports
    
    except Exception as e:
        logging.error(f"Error in scrape_pfd_reports: {e}")
        st.error(f"An error occurred while scraping reports: {e}")
        return []

def process_scraped_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean scraped data"""
    try:
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Extract metadata
        metadata = df['Content'].fillna("").apply(extract_metadata)
        metadata_df = pd.DataFrame(metadata.tolist())
        
        # Combine with original data
        result = pd.concat([df, metadata_df], axis=1)
        
        # Convert dates to datetime
        try:
            result['date_of_report'] = pd.to_datetime(
                result['date_of_report'],
                format='%d/%m/%Y',
                errors='coerce'
            )
        except Exception as e:
            logging.error(f"Error converting dates: {e}")
        
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

def clean_text_for_modeling(text: str) -> str:
    """Clean text for topic modeling"""
    if not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        logging.error(f"Error cleaning text for modeling: {e}")
        return ""

def create_topic_model(documents: List[str], num_topics: int) -> Optional[Tuple[BERTopic, List[int], np.ndarray]]:
    """Create and train topic model"""
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # Initialize embedding model
        with st.spinner("Loading embedding model..."):
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Initialize BERTopic with parameters
        with st.spinner("Initializing topic model..."):
            topic_model = BERTopic(
                embedding_model=embedding_model,
                nr_topics=num_topics,
                low_memory=True,
                calculate_probabilities=True,
                verbose=True
            )
        
        # Fit the model and transform documents
        with st.spinner("Training topic model..."):
            topics, probs = topic_model.fit_transform(documents)
        
        return topic_model, topics, probs
        
    except Exception as e:
        logging.error(f"Error creating topic model: {e}")
        st.error(f"Topic modeling error: {str(e)}")
        return None

def plot_topic_distribution(topic_model: BERTopic, topics: List[int]) -> None:
    """Plot distribution of documents across topics"""
    topic_counts = pd.Series(topics).value_counts()
    
    fig = px.bar(
        x=topic_counts.index,
        y=topic_counts.values,
        labels={'x': 'Topic', 'y': 'Count'},
        title='Distribution of Documents Across Topics'
    )
    
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Number of Documents",
        bargap=0.2
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_topic_timeline(df: pd.DataFrame, topics: List[int]) -> None:
    """Plot topic evolution over time"""
    if 'date_of_report' not in df.columns:
        return
    
    topic_evolution = pd.DataFrame({
        'Date': df['date_of_report'],
        'Topic': topics
    })
    
    topic_evolution = topic_evolution.groupby([
        pd.Grouper(key='Date', freq='M'),
        'Topic'
    ]).size().reset_index(name='Count')
    
    fig = px.line(
        topic_evolution,
        x='Date',
        y='Count',
        color='Topic',
        title='Topic Evolution Over Time'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_topic_similarity(topic_model: BERTopic) -> None:
    """Plot topic similarity heatmap"""
    similarity_matrix = topic_model.get_topic_similarity_matrix()
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=[f'Topic {i}' for i in range(len(similarity_matrix))],
        y=[f'Topic {i}' for i in range(len(similarity_matrix))],
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Topic Similarity Heatmap',
        xaxis_title='Topics',
        yaxis_title='Topics'
    )
    
    st.plotly_chart(fig, use_container_width=True)
def process_scraped_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean scraped data"""
    try:
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Extract metadata
        metadata = df['Content'].fillna("").apply(extract_metadata)
        metadata_df = pd.DataFrame(metadata.tolist())
        
        # Combine with original data
        result = pd.concat([df, metadata_df], axis=1)
        
        # Convert dates to datetime
        try:
            result['date_of_report'] = pd.to_datetime(
                result['date_of_report'],
                format='%d/%m/%Y',
                errors='coerce'
            )
        except Exception as e:
            logging.error(f"Error converting dates: {e}")
        
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

def clean_text_for_modeling(text: str) -> str:
    """Clean text for topic modeling"""
    if not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        logging.error(f"Error cleaning text for modeling: {e}")
        return ""

def create_topic_model(documents: List[str], num_topics: int) -> Optional[Tuple[BERTopic, List[int], np.ndarray]]:
    """Create and train topic model"""
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # Initialize embedding model
        with st.spinner("Loading embedding model..."):
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Initialize BERTopic with parameters
        with st.spinner("Initializing topic model..."):
            topic_model = BERTopic(
                embedding_model=embedding_model,
                nr_topics=num_topics,
                low_memory=True,
                calculate_probabilities=True,
                verbose=True
            )
        
        # Fit the model and transform documents
        with st.spinner("Training topic model..."):
            topics, probs = topic_model.fit_transform(documents)
        
        return topic_model, topics, probs
        
    except Exception as e:
        logging.error(f"Error creating topic model: {e}")
        st.error(f"Topic modeling error: {str(e)}")
        return None

def plot_topic_distribution(topic_model: BERTopic, topics: List[int]) -> None:
    """Plot distribution of documents across topics"""
    topic_counts = pd.Series(topics).value_counts()
    
    fig = px.bar(
        x=topic_counts.index,
        y=topic_counts.values,
        labels={'x': 'Topic', 'y': 'Count'},
        title='Distribution of Documents Across Topics'
    )
    
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Number of Documents",
        bargap=0.2
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_topic_timeline(df: pd.DataFrame, topics: List[int]) -> None:
    """Plot topic evolution over time"""
    if 'date_of_report' not in df.columns:
        return
    
    topic_evolution = pd.DataFrame({
        'Date': df['date_of_report'],
        'Topic': topics
    })
    
    topic_evolution = topic_evolution.groupby([
        pd.Grouper(key='Date', freq='M'),
        'Topic'
    ]).size().reset_index(name='Count')
    
    fig = px.line(
        topic_evolution,
        x='Date',
        y='Count',
        color='Topic',
        title='Topic Evolution Over Time'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_topic_similarity(topic_model: BERTopic) -> None:
    """Plot topic similarity heatmap"""
    similarity_matrix = topic_model.get_topic_similarity_matrix()
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=[f'Topic {i}' for i in range(len(similarity_matrix))],
        y=[f'Topic {i}' for i in range(len(similarity_matrix))],
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Topic Similarity Heatmap',
        xaxis_title='Topics',
        yaxis_title='Topics'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_topic_modeling_tab() -> None:
    """Render the topic modeling tab"""
    st.header("Topic Modeling Analysis")
    
    # Get the scraped data
    df = st.session_state.scraped_data
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Topic Modeling Options")
        
        num_topics = st.slider(
            "Number of Topics",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            help="Select the number of topics to extract from the documents"
        )
        
        topic_type = st.selectbox(
            "Report Type for Topic Modeling",
            ["All Reports", "Prevention of Future Death", "Response to PFD"]
        )
    
    # Filter data based on type
    if topic_type == "Prevention of Future Death":
        filtered_df = df[df['Content'].str.contains(
            'Prevention of Future Death',
            case=False,
            na=False
        )]
    elif topic_type == "Response to PFD":
        filtered_df = df[df['Content'].str.contains(
            'Response to Prevention of Future Death',
            case=False,
            na=False
        )]
    else:
        filtered_df = df
    
    # Check if we have enough documents
    if len(filtered_df) < num_topics:
        st.error(f"Not enough documents ({len(filtered_df)}) for {num_topics} topics. Please reduce topic count or scrape more reports.")
        return
    
    # Run topic modeling button
    if st.button("Run Topic Modeling"):
        try:
            # Preprocess text
            documents = [clean_text_for_modeling(doc) for doc in filtered_df['Content'].fillna("")]
            documents = [doc for doc in documents if len(doc.split()) > 10]
            
            if not documents:
                st.error("No valid documents found after preprocessing")
                return
            
            # Create and train model
            result = create_topic_model(documents, num_topics)
            
            if result:
                topic_model, topics, probs = result
                
                # Display results
                st.header("Topic Modeling Results")
                
                # Topic distribution
                st.subheader("Topic Distribution")
                plot_topic_distribution(topic_model, topics)
                
                # Topic details
                st.subheader("Topic Details")
                topic_info = topic_model.get_topic_info()
                
                # Create DataFrame with topic details
                topics_df = pd.DataFrame([
                    {
                        'Topic': row['Topic'],
                        'Size': row['Count'],
                        'Top Keywords': ', '.join([word for word, _ in topic_model.get_topic(row['Topic'])[:10]])
                    }
                    for _, row in topic_info.iterrows() if row['Topic'] != -1
                ])
                
                st.dataframe(
                    topics_df,
                    column_config={
                        "Topic": st.column_config.NumberColumn("Topic ID"),
                        "Size": st.column_config.NumberColumn("Number of Documents"),
                        "Top Keywords": st.column_config.TextColumn("Top Keywords")
                    },
                    hide_index=True
                )
                
                # Topic similarity
                st.subheader("Topic Similarity Analysis")
                plot_topic_similarity(topic_model)
                
                # Topic timeline
                st.subheader("Topic Evolution Over Time")
                plot_topic_timeline(filtered_df, topics)
                
                # Example documents per topic
                st.subheader("Example Documents per Topic")
                for topic_id in range(num_topics):
                    with st.expander(f"Topic {topic_id}"):
                        # Get keywords for this topic
                        keywords = topics_df.loc[topics_df['Topic'] == topic_id, 'Top Keywords'].iloc[0]
                        st.markdown(f"**Keywords**: {keywords}")
                        
                        # Get example documents
                        topic_docs = [doc for doc, t in zip(filtered_df['Content'], topics) if t == topic_id]
                        st.markdown(f"**Number of documents**: {len(topic_docs)}")
                        
                        if topic_docs:
                            st.markdown("### Example Documents:")
                            for i, doc in enumerate(topic_docs[:3], 1):
                                st.markdown(f"**Document {i}**:")
                                st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                        else:
                            st.info("No documents found for this topic")
                
        except Exception as e:
            st.error(f"Error in topic modeling: {str(e)}")
            logging.error(f"Topic modeling error: {e}", exc_info=True)

def render_analysis_tab() -> None:
    """Render the analysis tab"""
    st.header("Reports Analysis")
    
    df = st.session_state.scraped_data
    
    # Filters sidebar
    with st.sidebar:
        st.header("Analysis Filters")
        
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=[df['date_of_report'].min(), df['date_of_report'].max()],
            key="analysis_date_range"
        )
        
        # Category filter
        all_categories = set()
        for cats in df['categories'].dropna():
            if isinstance(cats, list):
                all_categories.update(cats)
                
        selected_categories = st.multiselect(
            "Categories",
            options=sorted(all_categories)
        )
        
        # Coroner area filter
        coroner_areas = sorted(df['coroner_area'].dropna().unique())
        selected_areas = st.multiselect(
            "Coroner Areas",
            options=coroner_areas
        )
    
    # Apply filters
    mask = pd.Series(True, index=df.index)
    
    if len(date_range) == 2:
        mask &= (df['date_of_report'].dt.date >= date_range[0]) & \
                (df['date_of_report'].dt.date <= date_range[1])
    
    if selected_categories:
        mask &= df['categories'].apply(
            lambda x: any(cat in x for cat in selected_categories) if isinstance(x, list) else False
        )
    
    if selected_areas:
        mask &= df['coroner_area'].isin(selected_areas)
    
    filtered_df = df[mask]
    
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters.")
        return
    
    # Overview metrics
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reports", len(filtered_df))
    with col2:
        st.metric("Unique Coroner Areas", filtered_df['coroner_area'].nunique())
    with col3:
        st.metric("Categories", len(all_categories))
    with col4:
        date_range = (filtered_df['date_of_report'].max() - filtered_df['date_of_report'].min()).days
        avg_reports_month = len(filtered_df) / (date_range / 30) if date_range > 0 else len(filtered_df)
        st.metric("Avg Reports/Month", f"{avg_reports_month:.1f}")
    
    # Visualizations
    st.subheader("Visualizations")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Timeline", "Categories", "Coroner Areas"])
    
    with viz_tab1:
        plot_timeline(filtered_df)
    
    with viz_tab2:
        plot_category_distribution(filtered_df)
    
    with viz_tab3:
        plot_coroner_areas(filtered_df)
    
    # Raw Data View
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(
            filtered_df,
            column_config={
                "URL": st.column_config.LinkColumn("Report Link"),
                "date_of_report": st.column_config.DateColumn("Date of Report"),
                "categories": st.column_config.ListColumn("Categories")
            },
            hide_index=True
        )

def initialize_session_state():
    """Initialize all required session state variables"""
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = None
    if 'cleanup_scheduled' not in st.session_state:
        st.session_state.cleanup_scheduled = False

def main():
    try:
        # Initialize session state
        initialize_session_state()
        
        # App title
        st.title("UK Judiciary PFD Reports Analysis")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs([
            "🔍 Scrape Reports",
            "📊 Analyze Reports",
            "🔬 Topic Modeling"
        ])
        
        # Render tabs
        with tab1:
            render_scraping_tab()
        
        with tab2:
            if st.session_state.scraped_data is not None:
                render_analysis_tab()
            else:
                st.warning("Please scrape reports first in the 'Scrape Reports' tab")
        
        with tab3:
            if st.session_state.scraped_data is not None:
                render_topic_modeling_tab()
            else:
                st.warning("Please scrape reports first in the 'Scrape Reports' tab")
        
        # Footer
        st.markdown("---")
        st.markdown(
            """<div style='text-align: center'>
            <p>Built with Streamlit • Data from UK Judiciary</p>
            </div>""",
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error("An error occurred in the application. Please try again.")
        logging.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
