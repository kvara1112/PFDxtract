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
        # Convert to lowercase and remove non-ASCII characters
        text = ''.join([char for char in text if ord(char) < 128])
        text = text.lower()
        
        # Tokenize and filter
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        
        # Remove stopwords and keep meaningful tokens
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
        
        # POS tagging and filtering for nouns
        tagged_words = nltk.pos_tag(tokens)
        filtered_tokens = [word for word, pos in tagged_words if pos.startswith('NN')]
        
        return ' '.join(filtered_tokens)
    
    except Exception as e:
        logging.error(f"Error cleaning text for modeling: {e}")
        return ""

def extract_topics_lda(df: pd.DataFrame, num_topics: int = 5, max_features: int = 1000) -> Tuple[LatentDirichletAllocation, TfidfVectorizer, np.ndarray]:
    """Extract topics using LDA"""
    try:
        # Prepare text data
        texts = df['Content'].fillna('').apply(clean_text_for_modeling)
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
        
        # LDA Model
        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=20
        )
        
        # Fit model and get topic distribution
        doc_topic_dist = lda_model.fit_transform(tfidf_matrix)
        
        return lda_model, vectorizer, doc_topic_dist
    
    except Exception as e:
        logging.error(f"Error in topic extraction: {e}")
        raise e

def create_network_diagram(topic_words: List[str], 
                         tfidf_matrix: np.ndarray, 
                         similarity_threshold: float = 0.3) -> go.Figure:
    """Create network diagram for topic visualization"""
    try:
        # Calculate similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Create graph
        G = nx.Graph()
        
        # Add edges based on similarity threshold
        for i, word1 in enumerate(topic_words):
            for j, word2 in enumerate(topic_words[i+1:], i+1):
                similarity = similarities[i][j]
                if similarity >= similarity_threshold:
                    G.add_edge(word1, word2, weight=similarity)
        
        # Get positions for visualization
        pos = nx.spring_layout(G)
        
        # Create traces for plotly
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            text=[],
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20
            ))
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            
        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += (node,)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=0,l=0,r=0,t=0)
                     ))
        
        return fig
    
    except Exception as e:
        logging.error(f"Error creating network diagram: {e}")
        return None

def render_topic_modeling_tab():
    """Render the topic modeling tab"""
    st.header("Topic Modeling Analysis")
    
    if st.session_state.scraped_data is None:
        st.warning("Please scrape or upload data first")
        return
    
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
        
        min_doc_freq = st.slider(
            "Minimum Document Frequency",
            min_value=1,
            max_value=10,
            value=2,
            help="Minimum number of documents a word must appear in"
        )

        similarity_threshold = st.slider(
            "Word Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            help="Minimum similarity score for word connections"
        )
    
    # Run topic modeling
    if st.button("Extract Topics"):
        try:
            with st.spinner("Preprocessing text and extracting topics..."):
                # First, check if we have enough valid text
                valid_docs = st.session_state.scraped_data['Content'].dropna().str.strip().str.len() > 0
                if valid_docs.sum() < 2:
                    st.error("Not enough valid documents found. Please ensure you have scraped or uploaded documents with text content.")
                    return
                
                # Extract topics
                result = extract_topics_lda(
                    st.session_state.scraped_data,
                    num_topics=num_topics,
                    max_features=max_features
                )
                
                if result[0] is None:
                    return
                
                lda_model, vectorizer, doc_topics = result
                
                # Store results
                st.session_state.topic_model = {
                    'model': lda_model,
                    'vectorizer': vectorizer,
                    'doc_topics': doc_topics
                }
                
                st.success("Topic extraction complete!")
        
        except Exception as e:
            st.error(f"Error during topic extraction: {str(e)}")
            logging.error(f"Topic modeling error: {e}", exc_info=True)
            return
    
    # Display results if model exists
    if hasattr(st.session_state, 'topic_model') and st.session_state.topic_model:
        st.subheader("Topic Analysis Results")
        
        # Get topic words
        feature_names = st.session_state.topic_model['vectorizer'].get_feature_names_out()
        
        # Create tabs for different visualizations
        topic_tab, dist_tab, network_tab, doc_tab = st.tabs([
            "Topic Keywords",
            "Topic Distribution",
            "Word Networks",
            "Documents by Topic"
        ])
        
        with topic_tab:
            # Display top words per topic
            for idx, topic in enumerate(st.session_state.topic_model['model'].components_):
                # Get top words and their weights
                top_indices = topic.argsort()[:-11:-1]
                top_words = [feature_names[i] for i in top_indices]
                weights = [topic[i] for i in top_indices]
                
                # Create word-weight pairs
                word_weights = [f"{word} ({weight:.3f})" for word, weight in zip(top_words, weights)]
                
                st.write(f"**Topic {idx + 1}:** {', '.join(word_weights)}")
        
        with dist_tab:
            # Plot topic distribution
            topic_dist = np.sum(st.session_state.topic_model['doc_topics'], axis=0)
            topic_props = topic_dist / topic_dist.sum() * 100
            
            fig = px.bar(
                x=[f"Topic {i+1}" for i in range(num_topics)],
                y=topic_props,
                title="Topic Distribution Across Documents",
                labels={'x': 'Topic', 'y': 'Percentage of Documents (%)'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
        
        with network_tab:
            # Create network diagrams for each topic
            for idx, topic in enumerate(st.session_state.topic_model['model'].components_):
                with st.expander(f"Topic {idx + 1} Network"):
                    top_indices = topic.argsort()[:-11:-1]
                    top_words = [feature_names[i] for i in top_indices]
                    fig = create_network_diagram(
                        top_words,
                        st.session_state.topic_model['model'].components_[idx].reshape(1, -1),
                        similarity_threshold
                    )
                    if fig:
                        st.plotly_chart(fig)
        
        with doc_tab:
            # Show example documents for each topic
            doc_topics = st.session_state.topic_model['doc_topics']
            df = st.session_state.scraped_data
            
            for topic_idx in range(num_topics):
                with st.expander(f"Topic {topic_idx + 1} Documents"):
                    # Get documents where this topic has the highest probability
                    topic_docs = [i for i, doc_topic in enumerate(doc_topics) 
                                if np.argmax(doc_topic) == topic_idx]
                    
                    if topic_docs:
                        st.write(f"Number of documents: {len(topic_docs)}")
                        
                        # Show top 3 documents
                        for i, doc_idx in enumerate(topic_docs[:3], 1):
                            st.markdown(f"**Document {i}**")
                            st.markdown(f"*Title:* {df.iloc[doc_idx]['Title']}")
                            st.markdown(f"*URL:* {df.iloc[doc_idx]['URL']}")
                            
                            # Show document preview
                            content = df.iloc[doc_idx]['Content']
                            preview = content[:500] + "..." if len(content) > 500 else content
                            st.text_area(f"Content Preview {i}", preview, height=150)
                    else:
                        st.info("No documents found predominantly featuring this topic")
            
            # Add option to download topic assignments
            topic_assignments = pd.DataFrame({
                'Document': df['Title'],
                'URL': df['URL'],
                'Dominant_Topic': np.argmax(doc_topics, axis=1) + 1,
                'Topic_Probability': np.max(doc_topics, axis=1)
            })
            
            csv = topic_assignments.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Topic Assignments",
                csv,
                "topic_assignments.csv",
                "text/csv",
                key="download_topics"
            )
                        
def render_scraping_tab():
    """Render the scraping tab UI and functionality"""
    # Initialize directories if they don't exist
    os.makedirs('pdfs', exist_ok=True)
    
    st.markdown("""
    ## UK Judiciary PFD Reports Scraper
    This tool scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    You can search by keywords, categories, and date ranges.
    """)
    
    with st.form("search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            search_keyword = st.text_input("Search keywords:", "")
            category = st.selectbox("PFD Report type:", [""] + get_pfd_categories())
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
            with st.spinner("Searching for reports..."):
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
                    
                    st.success(f"Found {len(reports)} reports")
                    
                    # Display results
                    st.header("Results")
                    st.dataframe(
                        df,
                        column_config={
                            "URL": st.column_config.LinkColumn("Report Link"),
                            "date_of_report": st.column_config.DateColumn("Date of Report"),
                            "categories": st.column_config.ListColumn("Categories")
                        },
                        hide_index=True
                    )
                    
                    # Export options
                    st.header("Export Options")
                    
                    # Generate filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pfd_reports_{search_keyword}_{timestamp}"
                    
                    col1, col2 = st.columns(2)
                    
                    # CSV Export
                    with col1:
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Reports (CSV)",
                            csv,
                            f"{filename}.csv",
                            "text/csv",
                            key="download_csv"
                        )
                    
                    # Excel Export
                    with col2:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        excel_data = excel_buffer.getvalue()
                        st.download_button(
                            "📥 Download Reports (Excel)",
                            excel_data,
                            f"{filename}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_excel"
                        )
                    
                    # PDF Download
                    st.header("Download PDFs")
                    if st.button("Download all PDFs"):
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
                                    key="download_pdfs_zip"
                                )
                            
                            # Cleanup zip file
                            os.remove(pdf_zip_path)
                else:
                    st.warning("No reports found matching your search criteria")
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Scraping error: {e}")



def render_file_upload():
    """Render file upload section"""
    st.header("Upload Existing Data")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.scraped_data = df
            st.success("File uploaded successfully!")
            
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

def render_analysis_tab():
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
    if 'topic_model' not in st.session_state:
        st.session_state.topic_model = None
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
        
        with tab1:
            render_scraping_tab()
        
        with tab2:
            if st.session_state.scraped_data is not None:
                render_analysis_tab()
            else:
                st.warning("Please scrape or upload data first")
        
        with tab3:
            if st.session_state.scraped_data is not None:
                render_topic_modeling_tab()
            else:
                st.warning("Please scrape or upload data first")
        
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
