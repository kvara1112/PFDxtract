import streamlit as st
import pandas as pd
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import time
import urllib3
import io
import pdfplumber
import tempfile
import logging
import os
import zipfile
import unicodedata
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

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

# Initialize Streamlit
st.set_page_config(page_title="UK Judiciary PFD Reports Analysis", layout="wide")

# Import local modules with error handling
try:
    from analysis_tab import render_analysis_tab
except ImportError as e:
    logging.error(f"Error importing analysis_tab: {e}")
    def render_analysis_tab():
        st.error("Analysis functionality not available. Please check installation.")

try:
    from topic_modeling_tab import render_topic_modeling_tab
except ImportError as e:
    logging.error(f"Error importing topic_modeling_tab: {e}")
    def render_topic_modeling_tab():
        st.error("Topic modeling functionality not available. Please check installation.")

def cleanup_temp_files(directory='pdfs', max_age_hours=24):
    """Clean up old temporary files"""
    try:
        current_time = time.time()
        directory_path = Path(directory)
        
        if not directory_path.exists():
            return
            
        for file_path in directory_path.glob('*'):
            if file_path.is_file():
                if current_time - file_path.stat().st_mtime > max_age_hours * 3600:
                    try:
                        file_path.unlink()
                        logging.info(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        logging.error(f"Error removing file {file_path}: {e}")
                        
    except Exception as e:
        logging.error(f"Error in cleanup_temp_files: {e}")

def clean_pdf_content(content: str) -> str:
    """Clean PDF content by removing headers and normalizing text"""
    if pd.isna(content) or not content:
        return ""
    
    try:
        content = str(content)
        
        # Remove PDF filename headers
        content = re.sub(r'PDF FILENAME:.*?\n', '', content)
        
        # Fix encoding issues
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€¦': '...',
            'â€"': '-',
            'â€¢': '•'
        }
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    except Exception as e:
        logging.error(f"Error cleaning PDF content: {e}")
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
                # Validate date format
                datetime.strptime(date_str, '%d/%m/%Y')
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

def process_scraped_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean scraped data"""
    try:
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Clean PDF content
        pdf_cols = [col for col in df.columns if col.endswith('_Content')]
        for col in pdf_cols:
            try:
                df[col] = df[col].fillna("").astype(str)
                df[col] = df[col].apply(clean_pdf_content)
            except Exception as e:
                logging.error(f"Error processing column {col}: {e}")
        
        # Extract metadata
        try:
            metadata = df['Content'].fillna("").apply(extract_metadata)
            metadata_df = pd.DataFrame(metadata.tolist())
            
            # Combine with original data
            result = pd.concat([df, metadata_df], axis=1)
            
            # Convert date strings to datetime
            try:
                result['date_of_report'] = pd.to_datetime(result['date_of_report'], 
                                                        format='%d/%m/%Y', 
                                                        errors='coerce')
            except Exception as e:
                logging.error(f"Error converting dates: {e}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error extracting metadata: {e}")
            return df
            
    except Exception as e:
        logging.error(f"Error in process_scraped_data: {e}")
        return df

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
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(pdf_url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        logging.info(f"Fetching content from: {url}")
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content = soup.find('div', class_='flow') or soup.find('article', class_='single__post')
        
        if not content:
            logging.warning(f"No content found at {url}")
            return None
        
        webpage_text = ""
        pdf_contents = []
        pdf_paths = []
        pdf_names = []
        
        # Extract text content
        paragraphs = content.find_all(['p', 'table'])
        webpage_text = '\n\n'.join(p.get_text(strip=True, separator=' ') for p in paragraphs)
        
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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        
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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        pagination = soup.find('nav', class_='navigation pagination')
        if pagination:
            page_numbers = pagination.find_all('a', class_='page-numbers')
            numbers = []
            for p in page_numbers:
                text = p.text.strip()
                if text.isdigit():
                    numbers.append(int(text))
            if numbers:
                return max(numbers)
        
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
    
    params = {
        'post_type': 'pfd',
        'order': order
    }
    
    if keyword and keyword.strip():
        params['s'] = keyword
    if category:
        params['pfd_report_type'] = category
    
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
    
    param_strings = [f"{k}={v}" for k, v in params.items()]
    initial_url = f"{base_url}/?{'&'.join(param_strings)}"
    
    try:
        total_pages = get_total_pages(initial_url)
        if total_pages == 0:
            st.warning("No results found")
            return []
            
        logging.info(f"Total pages to scrape: {total_pages}")
        
        if max_pages:
            total_pages = min(total_pages, max_pages)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while current_page <= total_pages:
            page_url = initial_url if current_page == 1 else f"{base_url}/page/{current_page}/?{'&'.join(param_strings)}"
            
            status_text.text(f"Scraping page {current_page} of {total_pages}...")
            progress_bar.progress(current_page / total_pages)
            
            reports = scrape_page(page_url)
            
            if reports:
                all_reports.extend(reports)
                logging.info(f"Found {len(reports)} reports on page {current_page}")
            else:
                logging.warning(f"No reports found on page {current_page}")
                if current_page > 1:
                    break
            
            current_page += 1
            time.sleep(1)  # Rate limiting
        
        progress_bar.progress(1.0)
        status_text.text(f"Completed! Total reports found: {len(all_reports)}")
        
        return all_reports
    
    except Exception as e:
        logging.error(f"Error in scrape_pfd_reports: {e}")
        st.error(f"An error occurred while scraping reports: {e}")
        return []

def scrape_all_categories() -> List[Dict]:
    """Scrape reports from all available categories"""
    all_reports = []
    categories = get_pfd_categories()
    
    for category in categories:
        try:
            st.info(f"Scraping category: {category}")
            reports = scrape_pfd_reports(category=category)
            all_reports.extend(reports)
            st.success(f"Found {len(reports)} reports in category {category}")
        except Exception as e:
            st.error(f"Error scraping category {category}: {e}")
            continue
    
    return all_reports

def render_scraping_tab():
    # Initialize directories if they don't exist
    os.makedirs('pdfs', exist_ok=True)
    
    # Schedule cleanup if not already done
    if 'cleanup_scheduled' not in st.session_state:
        cleanup_temp_files()
        st.session_state.cleanup_scheduled = True
    
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    You can search by keywords, categories, and date ranges.
    """)
    
    with st.form("search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            search_keyword = st.text_input("Search keywords:", "")
            category = st.selectbox("PFD Report type:", [""] + get_pfd_categories())
            order = st.selectbox("Sort by:", [
                ("relevance", "Relevance"),
                ("desc", "Newest first"),
                ("asc", "Oldest first")
            ], format_func=lambda x: x[1])
        
        with col2:
            date_after = st.date_input("Published after:", None)
            date_before = st.date_input("Published before:", None)
            max_pages = st.number_input("Maximum pages to scrape (0 for all):", 
                                      min_value=0, 
                                      value=0,
                                      help="Set to 0 to scrape all available pages")
        
        col3, col4 = st.columns(2)
        with col3:
            search_mode = st.radio("Search mode:",
                                 ["Search with filters", "Scrape all categories"],
                                 help="Choose whether to search with specific filters or scrape all categories")
        
        submitted = st.form_submit_button("Search Reports")
    
    if submitted:
        try:
            reports = []
            
            with st.spinner("Searching for reports..."):
                try:
                    if search_mode == "Search with filters":
                        date_after_str = date_after.strftime('%d/%m/%Y') if date_after else None
                        date_before_str = date_before.strftime('%d/%m/%Y') if date_before else None
                        
                        sort_order = order[0] if isinstance(order, tuple) else order
                        max_pages_val = None if max_pages == 0 else max_pages
                        
                        scraped_reports = scrape_pfd_reports(
                            keyword=search_keyword,
                            category=category if category else None,
                            date_after=date_after_str,
                            date_before=date_before_str,
                            order=sort_order,
                            max_pages=max_pages_val
                        )
                    else:
                        scraped_reports = scrape_all_categories()
                    
                    if scraped_reports:
                        reports.extend(scraped_reports)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logging.error(f"Scraping error: {e}")
            
            if reports:
                df = pd.DataFrame(reports)
                df = process_scraped_data(df)
                
                st.session_state.scraped_data = df
                st.success(f"Found {len(reports):,} reports")
                
                st.subheader("Reports Data")
                st.dataframe(
                    df,
                    column_config={
                        "URL": st.column_config.LinkColumn("Report Link"),
                        "date_of_report": st.column_config.DateColumn("Date of Report"),
                        "categories": st.column_config.ListColumn("Categories")
                    },
                    hide_index=True
                )
                
                st.subheader("Export Options")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pfd_reports_{search_keyword}_{timestamp}"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV Export
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Reports (CSV)",
                        csv,
                        f"{filename}.csv",
                        "text/csv",
                        key="download_csv"
                    )
                
                with col2:
                    # Excel Export
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
                
                st.subheader("Download PDFs")
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
                        
                        # Cleanup zip file after download
                        try:
                            os.remove(pdf_zip_path)
                        except Exception as e:
                            logging.error(f"Error removing zip file: {e}")
            
            else:
                if search_keyword or category:
                    st.warning("No reports found matching your search criteria")
                else:
                    st.info("Please enter search keywords or select a category to find reports")
                    
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            logging.error(f"Processing error: {e}", exc_info=True)

def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Scrape Reports", "Analyze Reports", "Topic Modeling"])
    
    # Initialize session state
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = None
    
    # Render tabs
    with tab1:
        render_scraping_tab()
    
    with tab2:
        render_analysis_tab()
    
    with tab3:
        render_topic_modeling_tab()

if __name__ == "__main__":
    main()

