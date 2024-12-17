import requests
from bs4 import BeautifulSoup
import logging
import streamlit as st
import time
from typing import List, Dict, Optional
from pdf_utils import save_pdf, extract_pdf_content
from data_processing import clean_text

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

def get_report_content(url: str) -> Optional[Dict]:
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

def scrape_page(url: str) -> List[Dict]:
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
                    
            except Exception as e:
                logging.error(f"Error processing card: {e}")
                continue
        
        return reports
        
    except Exception as e:
        logging.error(f"Error fetching page {url}: {e}")
        return []

def scrape_pfd_reports(keyword: Optional[str] = None) -> List[Dict]:
    """
    Scrape PFD reports with keyword search
    """
    base_url = "https://www.judiciary.uk"
    params = {'post_type': 'pfd', 'order': 'relevance'}
    
    if keyword:
        params['s'] = keyword
    
    param_strings = [f"{k}={v}" for k, v in params.items()]
    search_url = f"{base_url}/?{'&'.join(param_strings)}"
    
    try:
        return scrape_page(search_url)
    except Exception as e:
        logging.error(f"Error in scrape_pfd_reports: {e}")
        st.error(f"An error occurred while scraping reports: {e}")
        return []
