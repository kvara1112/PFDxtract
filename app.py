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
from analysis_tab import render_analysis_tab

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Analysis", layout="wide")

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
            
            return result
        except Exception as e:
            logging.error(f"Error extracting metadata: {e}")
            return df
            
    except Exception as e:
        logging.error(f"Error in process_scraped_data: {e}")
        return df  # Return original dataframe if processing fails



def get_pfd_categories():
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

def clean_text(text):
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

def save_pdf(pdf_url, base_dir='pdfs'):
    """Download and save PDF, return local path and filename"""
    try:
        os.makedirs(base_dir, exist_ok=True)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, headers=headers, verify=False, timeout=10)
        
        filename = os.path.basename(pdf_url)
        filename = re.sub(r'[^\w\-_\. ]', '_', filename)
        local_path = os.path.join(base_dir, filename)
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        return local_path, filename
    
    except Exception as e:
        logging.error(f"Error saving PDF {pdf_url}: {e}")
        return None, None

def extract_pdf_content(pdf_path):
    """Extract text from PDF file"""
    try:
        filename = os.path.basename(pdf_path)
        
        with pdfplumber.open(pdf_path) as pdf:
            pdf_text = "\n\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        full_content = f"PDF FILENAME: {filename}\n\n{pdf_text}"
        
        return clean_text(full_content)
    
    except Exception as e:
        logging.error(f"Error extracting PDF text from {pdf_path}: {e}")
        return ""

def get_report_content(url):
    """Get full content from report page with multiple PDF handling"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        logging.info(f"Fetching content from: {url}")
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content = soup.find('div', class_='flow') or soup.find('article', class_='single__post')
        
        webpage_text = ""
        pdf_contents = []
        pdf_paths = []
        pdf_names = []
        
        if content:
            paragraphs = content.find_all(['p', 'table'])
            webpage_text = '\n\n'.join(p.get_text(strip=True, separator=' ') for p in paragraphs)
            
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

def scrape_page(url):
    """Scrape a single page of search results"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
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
                url = title_elem['href']
                
                logging.info(f"Processing report: {title}")
                
                if not url.startswith(('http://', 'https://')):
                    url = f"https://www.judiciary.uk{url}"
                
                content_data = get_report_content(url)
                
                if content_data:
                    report = {
                        'Title': title,
                        'URL': url,
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
def get_total_pages(url):
    """Get total number of pages"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
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

def scrape_pfd_reports(keyword=None, category=None, date_after=None, date_before=None, order="relevance", max_pages=None):
    """
    Scrape PFD reports with comprehensive filtering
    """
    all_reports = []
    current_page = 1
    base_url = "https://www.judiciary.uk"
    
    params = {
        'post_type': 'pfd',
        'order': order
    }
    
    if keyword:
        params['s'] = keyword
    if category:
        params['pfd_report_type'] = category
    
    if date_after:
        year, month, day = date_after.split('-')
        params['after-year'] = year
        params['after-month'] = month
        params['after-day'] = day
    
    if date_before:
        year, month, day = date_before.split('-')
        params['before-year'] = year
        params['before-month'] = month
        params['before-day'] = day
    
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
            if current_page == 1:
                page_url = initial_url
            else:
                page_url = f"{base_url}/page/{current_page}/?{'&'.join(param_strings)}"
            
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

def scrape_all_categories():
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
        reports = []  # Initialize reports list
        
        with st.spinner("Searching for reports..."):
            try:
                if search_mode == "Search with filters":
                    # Convert dates to string format if provided
                    date_after_str = date_after.strftime("%Y-%m-%d") if date_after else None
                    date_before_str = date_before.strftime("%Y-%m-%d") if date_before else None
                    
                    # Get the actual sort order value from the tuple
                    sort_order = order[0] if isinstance(order, tuple) else order
                    
                    # Set max_pages to None if 0 was selected
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
            df = process_scraped_data(df)  # Process the scraped data
            
            # Store in session state for analysis tab
            if 'scraped_data' not in st.session_state:
                st.session_state.scraped_data = df
            
            st.success(f"Found {len(reports):,} reports")
            
            # Show detailed data
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
            
            # Export options
            st.subheader("Export Options")
            export_format = st.selectbox("Export format:", ["CSV", "Excel"], key="export_format")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pfd_reports_{search_keyword}_{timestamp}"
            
            if export_format == "CSV":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Reports (CSV)",
                    csv,
                    f"{filename}.csv",
                    "text/csv",
                    key="download_csv"
                )
            else:
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
            
            # Option to download PDFs
            st.subheader("Download PDFs")
            if st.button("Download all PDFs"):
                with st.spinner("Preparing PDF download..."):
                    # Create a zip file of all PDFs
                    pdf_zip_path = f"{filename}_pdfs.zip"
                    
                    with zipfile.ZipFile(pdf_zip_path, 'w') as zipf:
                        # Collect all unique PDF paths
                        unique_pdfs = set()
                        pdf_columns = [col for col in df.columns if col.startswith('PDF_') and col.endswith('_Path')]
                        
                        for col in pdf_columns:
                            paths = df[col].dropna()
                            unique_pdfs.update(paths)
                        
                        # Add PDFs to zip
                        for pdf_path in unique_pdfs:
                            if pdf_path and os.path.exists(pdf_path):
                                zipf.write(pdf_path, os.path.basename(pdf_path))
                    
                    # Provide download button for ZIP
                    with open(pdf_zip_path, 'rb') as f:
                        st.download_button(
                            "📦 Download All PDFs (ZIP)",
                            f.read(),
                            pdf_zip_path,
                            "application/zip",
                            key="download_pdfs_zip"
                        )
        else:
            if search_keyword or category:
                st.warning("No reports found matching your search criteria")
            else:
                st.info("Please enter search keywords or select a category to find reports")

def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Scrape Reports", "Analyze Reports"])
    
    # Initialize session state for sharing data between tabs
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = None
    
    with tab1:
        render_scraping_tab()
    
    with tab2:
        render_analysis_tab()

if __name__ == "__main__":
    main()
