import requests
from bs4 import BeautifulSoup
import re
import urllib3
import logging
import os
import pdfplumber
import time
import streamlit as st

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def clean_text(text):
    """Clean text by removing extra whitespace and special characters"""
    if not text:
        return ""
    
    try:
        # Replace problematic characters and symbols
        text = re.sub(r'â€™', "'", str(text))
        text = re.sub(r'â€¦', "...", text)
        text = re.sub(r'â€"', "-", text)
        text = re.sub(r'â€œ', '"', text)
        text = re.sub(r'â€', '"', text)
        
        # Remove or replace other potential encoding issues
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove any non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        
        return text.strip()
    except Exception as e:
        logging.error(f"Error in clean_text: {e}")
        return ""

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
            
            # Find PDF download links
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
                    # Create report dictionary
                    report = {
                        'Title': title,
                        'URL': url,
                        'Content': content_data['content']
                    }
                    
                    # Add PDF information
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
