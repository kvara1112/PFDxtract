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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import json
import nltk
import traceback

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

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

class WebScraper:
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Referer': 'https://judiciary.uk/'
    }
    
    @staticmethod
    def make_request(url: str, retries: int = 3, delay: int = 2) -> Optional[requests.Response]:
        for attempt in range(retries):
            try:
                time.sleep(delay)
                response = requests.get(url, headers=WebScraper.HEADERS, verify=False, timeout=30)
                response.raise_for_status()
                return response
            except Exception as e:
                if attempt == retries - 1:
                    st.error(f"Request failed: {str(e)}")
                    raise e
                time.sleep(delay * (attempt + 1))
        return None

    @staticmethod
    def scrape_page(url: str) -> List[Dict]:
        try:
            response = WebScraper.make_request(url)
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
                    if not title_elem or not title_elem.find('a'):
                        continue
                        
                    title_link = title_elem.find('a')
                    title = DataProcessor.clean_text(title_link.text)
                    card_url = title_link['href']
                    
                    if not card_url.startswith(('http://', 'https://')):
                        card_url = f"https://www.judiciary.uk{card_url}"
                    
                    logging.info(f"Processing report: {title}")
                    content_data = WebScraper.get_report_content(card_url)
                    
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
                        
                except Exception as e:
                    logging.error(f"Error processing card: {e}")
                    continue
            
            return reports
            
        except Exception as e:
            logging.error(f"Error fetching page {url}: {e}")
            return []

    @staticmethod
    def get_report_content(url: str) -> Optional[Dict]:
        try:
            logging.info(f"Fetching content from: {url}")
            response = WebScraper.make_request(url)
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
                
                pdf_path, pdf_name = PDFHandler.save_pdf(pdf_url)
                
                if pdf_path:
                    pdf_content = PDFHandler.extract_pdf_content(pdf_path)
                    pdf_contents.append(pdf_content)
                    pdf_paths.append(pdf_path)
                    pdf_names.append(pdf_name)
            
            return {
                'content': DataProcessor.clean_text(webpage_text),
                'pdf_contents': pdf_contents,
                'pdf_paths': pdf_paths,
                'pdf_names': pdf_names
            }
            
        except Exception as e:
            logging.error(f"Error getting report content: {e}")
            return None

class DataProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text while preserving structure and metadata"""
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
                'Â': ' '
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

    @staticmethod
    def clean_text_for_modeling(text: str) -> str:
        """Clean text specifically for modeling purposes"""
        if not isinstance(text, str):
            return ""
        
        try:
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+|\b\w*\d+\w*\b|\b[a-z]\b|[^a-z\s]', ' ', text)
            return ' '.join(text.split())
            
        except Exception as e:
            logging.error(f"Error in text cleaning: {e}")
            return ""

    @staticmethod
    def extract_metadata(content: str) -> Dict:
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
            
            # Extract other fields
            metadata.update({
                'ref': DataProcessor._extract_field(content, r'Ref(?:erence)?:?\s*([-\d]+)'),
                'deceased_name': DataProcessor._extract_field(content, r'Deceased name:?\s*([^\n]+)'),
                'coroner_name': DataProcessor._extract_field(content, r'Coroner(?:\'?s)? name:?\s*([^\n]+)'),
                'coroner_area': DataProcessor._extract_field(content, r'Coroner(?:\'?s)? Area:?\s*([^\n]+)')
            })
            
            # Extract categories
            cat_match = re.search(r'Category:?\s*([^\n]+)', content)
            if cat_match:
                categories = cat_match.group(1).split('|')
                metadata['categories'] = [DataProcessor.clean_text(cat).strip() 
                                       for cat in categories if DataProcessor.clean_text(cat).strip()]
            
            return metadata
            
        except Exception as e:
            logging.error(f"Error extracting metadata: {e}")
            return metadata

    @staticmethod
    def _extract_field(content: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, content)
        return DataProcessor.clean_text(match.group(1)) if match else None

class PDFHandler:
    @staticmethod
    def save_pdf(pdf_url: str, base_dir: str = 'pdfs') -> Tuple[Optional[str], Optional[str]]:
        try:
            os.makedirs(base_dir, exist_ok=True)
            
            response = WebScraper.make_request(pdf_url)
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

    @staticmethod
    def extract_pdf_content(pdf_path: str, chunk_size: int = 10) -> str:
        try:
            filename = os.path.basename(pdf_path)
            text_chunks = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for i in range(0, len(pdf.pages), chunk_size):
                    chunk = pdf.pages[i:i+chunk_size]
                    chunk_text = "\n\n".join([page.extract_text() or "" for page in chunk])
                    text_chunks.append(chunk_text)
                    
            full_content = f"PDF FILENAME: {filename}\n\n{''.join(text_chunks)}"
            return DataProcessor.clean_text(full_content)
            
        except Exception as e:
            logging.error(f"Error extracting PDF text from {pdf_path}: {e}")
            return ""

class TopicModeling:
    def __init__(self, num_topics: int = 5, max_features: int = 1000):
        self.num_topics = num_topics
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english'
        )
        self.model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42
        )
    
    def fit_transform(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        dtm = self.vectorizer.fit_transform(texts)
        doc_topics = self.model.fit_transform(dtm)
        return doc_topics, self.vectorizer.get_feature_names_out()
    
    def get_top_words(self, topic_idx: int, n_words: int = 10) -> List[str]:
        return [
            self.vectorizer.get_feature_names_out()[i] 
            for i in self.model.components_[topic_idx].argsort()[:-n_words-1:-1]
        ]

class SemanticClustering:
    def __init__(self, min_cluster_size: int = 3, max_features: int = 5000):
        self.min_cluster_size = min_cluster_size
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def cluster_documents(self, texts: List[str]) -> Dict:
        if len(texts) < self.min_cluster_size:
            raise ValueError(f"Not enough documents. Need at least {self.min_cluster_size}")
            
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        max_clusters = min(int(len(texts) * 0.4), 20)
        best_n_clusters = 2
        best_score = -1
        
        for n_clusters in range(2, max_clusters + 1):
            if len(texts) < n_clusters * self.min_cluster_size:
                break
                
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='euclidean',
                linkage='average'
            )
            
            tfidf_dense = tfidf_matrix.toarray()
            cluster_labels = clustering.fit_predict(tfidf_dense)
            
            if len(set(cluster_labels)) > 1:
                score = silhouette_score(tfidf_dense, cluster_labels, metric='euclidean')
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
        
        final_clustering = AgglomerativeClustering(
            n_clusters=best_n_clusters,
            metric='euclidean',
            linkage='average'
        )
        
        labels = final_clustering.fit_predict(tfidf_matrix.toarray())
        
        clusters = []
        feature_names = self.vectorizer.get_feature_names_out()
        
        for cluster_id in range(best_n_clusters):
            cluster_docs = np.where(labels == cluster_id)[0]
            if len(cluster_docs) >= self.min_cluster_size:
                cluster_tfidf = tfidf_matrix[cluster_docs].toarray()
                centroid = cluster_tfidf.mean(axis=0)
                
                top_terms = []
                for idx in centroid.argsort()[:-20-1:-1]:
                    if centroid[idx] > 0:
                        top_terms.append({
                            'term': feature_names[idx],
                            'weight': float(centroid[idx])
                        })
                
                clusters.append({
                    'id': cluster_id,
                    'size': len(cluster_docs),
                    'documents': [texts[i] for i in cluster_docs],
                    'terms': top_terms
                })
        
        return {
            'n_clusters': len(clusters),
            'silhouette_score': float(best_score),
            'clusters': clusters
        }

class UI:
    @staticmethod
    def render_filters():
        with st.sidebar:
            st.header("Filters")
            
            # Date Range
            with st.expander("📅 Date Range", expanded=True):
                start_date = st.date_input(
                    "From",
                    value=datetime.now() - timedelta(days=30),
                    format="DD/MM/YYYY"
                )
                end_date = st.date_input(
                    "To",
                    value=datetime.now(),
                    format="DD/MM/YYYY"
                )
            
            # Categories
            categories = st.multiselect(
                "Categories",
                options=get_pfd_categories()
            )
            
            return start_date, end_date, categories
    
    @staticmethod
    def render_analysis(data: pd.DataFrame):
        if data.empty:
            st.warning("No data available for analysis")
            return
        
        st.header("Analysis Results")
        
        # Timeline
        st.subheader("Reports Timeline")
        timeline_data = data.groupby(
            pd.Grouper(key='date_of_report', freq='M')
        ).size().reset_index()
        timeline_data.columns = ['Date', 'Count']
        
        fig = px.line(timeline_data, 
                     x='Date', 
                     y='Count',
                     title='Reports Timeline')
        st.plotly_chart(fig, use_container_width=True)
        
        # Category Distribution
        st.subheader("Category Distribution")
        categories = []
        for cats in data['categories'].dropna():
            if isinstance(cats, list):
                categories.extend(cats)
        
        cat_counts = pd.Series(categories).value_counts()
        
        fig = px.bar(
            x=cat_counts.index,
            y=cat_counts.values,
            title='Category Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return fig

def get_pfd_categories() -> List[str]:
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

def initialize_session_state():
    if not hasattr(st.session_state, 'initialized'):
        st.session_state.data_source = None
        st.session_state.current_data = None
        st.session_state.scraped_data = None
        st.session_state.uploaded_data = None
        st.session_state.topic_model = None
        st.session_state.cleanup_done = False
        st.session_state.last_scrape_time = None
        st.session_state.last_upload_time = None
        st.session_state.initialized = True
        
        # Cleanup old PDFs
        if not st.session_state.cleanup_done:
            try:
                pdf_dir = 'pdfs'
                os.makedirs(pdf_dir, exist_ok=True)
                
                current_time = time.time()
                for file in os.listdir(pdf_dir):
                    file_path = os.path.join(pdf_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            if os.stat(file_path).st_mtime < current_time - 86400:
                                os.remove(file_path)
                    except Exception as e:
                        logging.warning(f"Error cleaning up file {file_path}: {e}")
                        continue
            except Exception as e:
                logging.error(f"Error during PDF cleanup: {e}")
            finally:
                st.session_state.cleanup_done = True

def main():
    initialize_session_state()
    
    st.title("UK Judiciary PFD Reports Analysis")
    
    # Tab selection
    current_tab = st.radio(
        "Select section:",
        ["🔍 Scrape Reports", "📊 Analysis", "🔬 Topic Modeling"],
        horizontal=True,
        key="main_tab_selector"
    )
    
    st.markdown("---")
    
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
            render_topic_modeling_tab(st.session_state.current_data)
        except Exception as e:
            st.error(f"Error in topic modeling: {e}")
            logging.error(f"Topic modeling error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
