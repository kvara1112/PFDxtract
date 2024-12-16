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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Analysis", layout="wide")

def clean_text(text):
    """Clean text by removing extra whitespace and special characters"""
    if not text:
        return ""
    
    try:
        # Replace special characters
        text = re.sub(r'[â€™]', "'", str(text))
        text = re.sub(r'[â€¦]', "...", text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logging.error(f"Error in clean_text: {e}")
        return ""

def extract_pdf_text(pdf_url):
    """Extract text from PDF URL"""
    try:
        # Download PDF
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, headers=headers, verify=False, timeout=10)
        
        # Create a temporary file to save the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf_path = temp_pdf.name
        
        # Extract text from PDF
        with pdfplumber.open(temp_pdf_path) as pdf:
            # Combine text from all pages
            pdf_text = "\n\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        return clean_text(pdf_text)
    
    except Exception as e:
        logging.error(f"Error extracting PDF text from {pdf_url}: {e}")
        return ""

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
                # Get title and URL
                title_elem = card.find('h3', class_='card__title').find('a')
                if not title_elem:
                    continue
                    
                title = clean_text(title_elem.text)
                url = title_elem['href']
                
                logging.info(f"Processing report: {title}")
                content = get_report_content(url)
                
                if content:
                    report = {
                        'Title': title,
                        'URL': url,
                        'Content': content
                    }
                    
                    reports.append(report)
                    logging.info(f"Successfully processed: {title}")
                
            except Exception as e:
                logging.error(f"Error processing card: {e}")
                continue
                
        return reports
        
    except Exception as e:
        logging.error(f"Error fetching page {url}: {e}")
        return []

def get_report_content(url):
    """Get full content from report page"""
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
        
        if content:
            # Get all text content preserving line breaks
            paragraphs = content.find_all(['p', 'table'])
            text_content = '\n\n'.join(p.get_text(strip=True, separator=' ') for p in paragraphs)
            if text_content:
                logging.info("Successfully extracted webpage content")
                
                # Find PDF download links
                pdf_links = soup.find_all('a', class_='related-content__link', href=re.compile(r'\.pdf$'))
                pdf_texts = []
                
                for pdf_link in pdf_links:
                    pdf_url = pdf_link['href']
                    pdf_text = extract_pdf_text(pdf_url)
                    if pdf_text:
                        pdf_texts.append(pdf_text)
                
                # Combine webpage and PDF texts
                full_text = text_content
                if pdf_texts:
                    full_text += "\n\n--- PDF CONTENT ---\n\n" + "\n\n".join(pdf_texts)
                
                return clean_text(full_text)
        
        logging.warning(f"No content found for: {url}")
        return None
        
    except Exception as e:
        logging.error(f"Error getting report content: {e}")
        return None

def get_total_pages(url):
    """Get total number of pages"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        pagination = soup.find('ul', class_='pagination')
        if pagination:
            page_numbers = pagination.find_all('a', class_='page-numbers')
            if page_numbers:
                numbers = [int(p.text) for p in page_numbers if p.text.isdigit()]
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
        logging.info(f"Total pages to scrape: {total_pages}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while current_page <= total_pages:
            url = f"{base_url}{'page/' + str(current_page) + '/' if current_page > 1 else ''}?s={keyword if keyword else ''}&post_type=pfd"
            
            status_text.text(f"Scraping page {current_page} of {total_pages}...")
            progress_bar.progress(current_page / total_pages)
            
            reports = scrape_page(url)
            if reports:
                all_reports.extend(reports)
                logging.info(f"Found {len(reports)} reports on page {current_page}")
            else:
                logging.warning(f"No reports found on page {current_page}")
            
            current_page += 1
            time.sleep(1)  # Rate limiting
        
        progress_bar.progress(1.0)
        status_text.text(f"Completed! Total reports found: {len(all_reports)}")
        
        return all_reports
    
    except Exception as e:
        logging.error(f"Error in scrape_pfd_reports: {e}")
        st.error(f"An error occurred while scraping reports: {e}")
        return []

# Rest of the code remains the same as previous implementation
def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
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
        else:
            if search_keyword:
                st.warning("No reports found matching your search criteria")
            else:
                st.info("Please enter search keywords to find reports")

if __name__ == "__main__":
    main()
