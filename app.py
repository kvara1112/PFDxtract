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

def clean_text(text):
    """
    Clean text while preserving structure and metadata formatting
    """
    if not text:
        return ""
    
    try:
        # Convert to string and handle potential non-string inputs
        text = str(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Replace problematic encoded characters
        replacements = {
            'â€™': "'",   # Smart single quote
            'â€œ': '"',   # Left double quote
            'â€': '"',    # Right double quote
            'â€¦': '...',  # Ellipsis
            'â€"': '-',   # Em dash
            'â€¢': '•',   # Bullet point
            'Â': '',      # Unwanted character
            '\u200b': '',  # Zero-width space
            '\uf0b7': '',  # Private use area character
        }
        
        for encoded, replacement in replacements.items():
            text = text.replace(encoded, replacement)
        
        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Preserve metadata field markers with proper spacing
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
        
        # Add newlines before metadata fields
        for field in key_fields:
            text = text.replace(field, f'\n{field}')
        
        # Remove non-printable characters while preserving newlines
        text = ''.join(char if char.isprintable() or char == '\n' else ' ' for char in text)
        
        # Clean up multiple spaces/newlines while preserving structure
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                # Preserve exact spacing after metadata field markers
                if any(field in line for field in key_fields):
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        lines.append(f"{parts[0]}: {parts[1].strip()}")
                else:
                    # For non-metadata lines, normalize spaces
                    lines.append(' '.join(line.split()))
        
        # Join lines with single newlines
        text = '\n'.join(lines)
        
        # Normalize quotation marks
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('"', '"').replace('"', '"')
        
        return text.strip()
    
    except Exception as e:
        logging.error(f"Error in clean_text: {e}")
        return ""

def extract_date(text):
    """
    Comprehensive date extraction method with multiple strategies
    """
    if not text:
        return None
    
    # Clean the text first
    cleaned_text = clean_text(text)
    
    # Comprehensive date patterns
    date_patterns = [
        r'Date\s*of\s*report\s*[:]*\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',  # Supports / . or -
        r'Date\s*of\s*report\s*:\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',
        r'Date\s*of\s*report\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})'
    ]
    
    # Additional text-based patterns (for edge cases)
    text_patterns = [
        r'\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})\b',  # Anywhere in text
        r'\b(\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b'   # Alternate format
    ]
    
    # Combine patterns
    all_patterns = date_patterns + text_patterns
    
    # Possible date separators
    separators = ['/', '-', '.']
    
    for pattern in all_patterns:
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            
            # Try different date formats
            formats_to_try = [
                '%d/%m/%Y',   # UK/EU format: Day/Month/Year
                '%m/%d/%Y',   # US format: Month/Day/Year
                '%Y/%m/%d',   # ISO format: Year/Month/Day
                '%d-%m-%Y',   # Alternative UK format
                '%m-%d-%Y',   # Alternative US format
                '%Y-%m-%d'    # Alternative ISO format
            ]
            
            for sep in separators:
                for fmt in formats_to_try:
                    reformatted_fmt = fmt.replace('/', sep).replace('-', sep)
                    try:
                        parsed_date = datetime.strptime(date_str.replace('/', sep).replace('-', sep), reformatted_fmt)
                        # Additional validation
                        if 2000 <= parsed_date.year <= datetime.now().year:
                            return parsed_date.strftime('%d/%m/%Y')
                    except ValueError:
                        continue
    
    logging.warning(f"Could not extract valid date from text: {text}")
    return None
    
def save_pdf(pdf_url, base_dir='pdfs'):
    """Download and save PDF, return local path and filename"""
    try:
        # Create PDFs directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Download PDF
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, headers=headers, verify=False, timeout=10)
        
        # Extract filename from URL
        filename = os.path.basename(pdf_url)
        
        # Ensure filename is valid
        filename = re.sub(r'[^\w\-_\. ]', '_', filename)
        
        # Full path to save PDF
        local_path = os.path.join(base_dir, filename)
        
        # Save PDF
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        return local_path, filename
    
    except Exception as e:
        logging.error(f"Error saving PDF {pdf_url}: {e}")
        return None, None

def extract_pdf_content(pdf_path):
    """Extract text from PDF file"""
    try:
        # Get filename
        filename = os.path.basename(pdf_path)
        
        with pdfplumber.open(pdf_path) as pdf:
            # Combine text from all pages
            pdf_text = "\n\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        # Prepend filename to the content
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
        
        # Try to find content in different possible locations
        content = soup.find('div', class_='flow')
        if not content:
            content = soup.find('article', class_='single__post')
        
        webpage_text = ""
        pdf_contents = []
        pdf_paths = []
        pdf_names = []
        
        if content:
            # Get webpage text
            paragraphs = content.find_all(['p', 'table'])
            webpage_text = '\n\n'.join(p.get_text(strip=True, separator=' ') for p in paragraphs)
            
            # Find PDF download links - look for multiple strategies
            pdf_links = (
                soup.find_all('a', class_='related-content__link', href=re.compile(r'\.pdf$')) or
                soup.find_all('a', href=re.compile(r'\.pdf$'))
            )
            
            for pdf_link in pdf_links:
                pdf_url = pdf_link['href']
                
                # Ensure full URL
                if not pdf_url.startswith(('http://', 'https://')):
                    pdf_url = f"https://www.judiciary.uk{pdf_url}" if not pdf_url.startswith('/') else f"https://www.judiciary.uk/{pdf_url}"
                
                # Save PDF
                pdf_path, pdf_name = save_pdf(pdf_url)
                
                if pdf_path:
                    # Extract PDF content
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
    print(f"Scraping URL: {url}")  # Debug print
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
                # Get title and URL
                title_elem = card.find('h3', class_='card__title').find('a')
                if not title_elem:
                    continue
                    
                title = clean_text(title_elem.text)
                url = title_elem['href']
                
                logging.info(f"Processing report: {title}")
                content_data = get_report_content(url)
                
                if content_data:
                    # Dynamically create columns for multiple PDFs
                    report = {
                        'Title': title,
                        'URL': url,
                        'Content': content_data['content']
                    }
                    
                    # Add PDF information dynamically
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
        
        pagination = soup.find('ul', class_='pagination')
        
        # Debug print
        print("Full HTML of pagination:", pagination)
        
        if pagination:
            page_numbers = pagination.find_all('a', class_='page-numbers')
            
            # Debug print
            print("Page number links:", [p.text for p in page_numbers])
            
            if page_numbers:
                numbers = [int(p.text) for p in page_numbers if p.text.isdigit()]
                
                # Debug print
                print("Extracted page numbers:", numbers)
                
                if numbers:
                    return max(numbers)
        return 1
    except Exception as e:
        logging.error(f"Error getting total pages: {e}")
        return 1

def scrape_pfd_reports(keyword=None):
    """Scrape reports with keyword search"""
    all_reports = []
    current_page = 1
    base_url = "https://www.judiciary.uk/"
    
    initial_url = f"{base_url}?s={keyword if keyword else ''}&post_type=pfd"
    
    try:
        total_pages = get_total_pages(initial_url)
        
        # Debug prints
        print(f"Initial URL: {initial_url}")
        print(f"Total pages detected: {total_pages}")
        
        logging.info(f"Total pages to scrape: {total_pages}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while current_page <= total_pages:
            # Debug print
            print(f"Scraping page {current_page}")
            
            url = f"{base_url}{'page/' + str(current_page) + '/' if current_page > 1 else ''}?s={keyword if keyword else ''}&post_type=pfd"
            
            status_text.text(f"Scraping page {current_page} of {total_pages}...")
            progress_bar.progress(current_page / total_pages)
            
            reports = scrape_page(url)
            
            # Debug print
            print(f"Reports found on page {current_page}: {len(reports)}")
            
            if reports:
                all_reports.extend(reports)
                logging.info(f"Found {len(reports)} reports on page {current_page}")
            else:
                logging.warning(f"No reports found on page {current_page}")
            
            current_page += 1
            time.sleep(1)  # Rate limiting
        
        progress_bar.progress(1.0)
        status_text.text(f"Completed! Total reports found: {len(all_reports)}")
        
        # Debug print
        print(f"Total reports scraped: {len(all_reports)}")
        
        return all_reports
    
    except Exception as e:
        logging.error(f"Error in scrape_pfd_reports: {e}")
        st.error(f"An error occurred while scraping reports: {e}")
        return []

def render_scraping_tab():
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Enter keywords to search for relevant reports.
    """)
    
    # Use form for input
    with st.form("search_form"):
        search_keyword = st.text_input("Search keywords:", "")
        submitted = st.form_submit_button("Search Reports")
    
    if submitted:
        reports = []  # Initialize reports list
        with st.spinner("Searching for reports..."):
            try:
                scraped_reports = scrape_pfd_reports(keyword=search_keyword)
                if scraped_reports:
                    reports.extend(scraped_reports)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logging.error(f"Scraping error: {e}")
        
        if reports:
            df = pd.DataFrame(reports)
            
            st.success(f"Found {len(reports):,} reports")
            
            # Show detailed data
            st.subheader("Reports Data")
            st.dataframe(
                df,
                column_config={
                    "URL": st.column_config.LinkColumn("Report Link")
                },
                hide_index=True
            )
            
            # Export options
            export_format = st.selectbox("Export format:", ["CSV", "Excel"], key="export_format")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pfd_reports_{search_keyword}_{timestamp}"
            
            if export_format == "CSV":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Reports",
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
                    "📥 Download Reports",
                    excel_data,
                    f"{filename}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
            
            # Option to download PDFs
            if st.button("Download all PDFs"):
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
                        "📦 Download All PDFs",
                        f.read(),
                        pdf_zip_path,
                        "application/zip",
                        key="download_pdfs_zip"
                    )
        else:
            if search_keyword:
                st.warning("No reports found matching your search criteria")
            else:
                st.info("Please enter search keywords to find reports")

def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Scrape Reports", "Analyze Reports"])
    
    with tab1:
        render_scraping_tab()
    
    with tab2:
        render_analysis_tab()

if __name__ == "__main__":
    main()
