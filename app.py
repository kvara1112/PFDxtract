import streamlit as st
import pandas as pd
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import time
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Scraper", layout="wide")

def clean_text(text):
    """Clean text by removing extra whitespace and newlines"""
    if text:
        return ' '.join(text.strip().split())
    return ""

def extract_metadata(text):
    """Extract metadata fields from text"""
    patterns = {
        'date_of_report': r'Date of report:?\s*([^\n]+)',
        'ref': r'Ref:?\s*([\w-]+)',
        'deceased_name': r'Deceased name:?\s*([^\n]+)',
        'coroner_name': r'Coroner(?:s)? name:?\s*([^\n]+)',
        'coroner_area': r'Coroner(?:s)? Area:?\s*([^\n]+)',
        'category': r'Category:?\s*([^|\n]+)',
        'sent_to': r'This report is being sent to:?\s*([^\n]+)'
    }
    
    info = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            info[key] = clean_text(match.group(1))
    return info

def extract_section_content(text):
    """Extract content from numbered sections in the report"""
    sections = {
        'CORONER': '',
        'LEGAL_POWERS': '',
        'INVESTIGATION': '',
        'CIRCUMSTANCES': '',
        'CONCERNS': '',
        'ACTION': '',
        'RESPONSE': '',
        'COPIES': '',
        'DATE_CORONER': ''
    }
    
    patterns = {
        'CORONER': r'1\s*CORONER\s*(.*?)(?=2\s*CORONER|$)',
        'LEGAL_POWERS': r'2\s*CORONER\'S LEGAL POWERS\s*(.*?)(?=3\s*INVESTIGATION|$)',
        'INVESTIGATION': r'3\s*INVESTIGATION\s*(.*?)(?=4\s*CIRCUMSTANCES|$)',
        'CIRCUMSTANCES': r'4\s*CIRCUMSTANCES OF THE DEATH\s*(.*?)(?=5\s*CORONER|$)',
        'CONCERNS': r'5\s*CORONER\'S CONCERNS\s*(.*?)(?=6\s*ACTION|$)',
        'ACTION': r'6\s*ACTION SHOULD BE TAKEN\s*(.*?)(?=7\s*YOUR RESPONSE|$)',
        'RESPONSE': r'7\s*YOUR RESPONSE\s*(.*?)(?=8\s*COPIES|$)',
        'COPIES': r'8\s*COPIES and PUBLICATION\s*(.*?)(?=9|$)',
        'DATE_CORONER': r'9\s*(.*?)(?=Related content|$)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = clean_text(match.group(1))
            
    return sections

def get_report_content(url):
    """Get detailed content from individual report page"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the main content
        content = soup.find('div', class_='entry-content')
        if not content:
            return None
            
        return content.get_text()
    except Exception as e:
        st.error(f"Error getting report content: {str(e)}")
        return None

def scrape_page(url):
    """Scrape a single page of search results"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, verify=False)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    results_list = soup.find('ul', class_='search__list')
    if not results_list:
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
            
            # Get metadata from card description
            desc = card.find('p', class_='card__description')
            if not desc:
                continue
                
            metadata = extract_metadata(desc.text)
            
            # Get categories from pills
            categories = []
            pills = card.find_all('a', href=re.compile(r'/pfd-types/'))
            for pill in pills:
                categories.append(clean_text(pill.text))
            
            # Get full report content
            report_content = get_report_content(url)
            if report_content:
                sections = extract_section_content(report_content)
            else:
                sections = {}
            
            report = {
                'Title': title,
                'URL': url,
                'Categories': ' | '.join(categories),
                **metadata,  # Add metadata
                **sections  # Add section content
            }
            
            reports.append(report)
            st.write(f"Processed report: {title}")
            
        except Exception as e:
            st.error(f"Error processing card: {str(e)}")
            continue
            
    return reports

def scrape_pfd_reports(keyword, max_pages=10):
    """Scrape multiple pages of reports"""
    all_reports = []
    current_page = 1
    
    base_url = "https://www.judiciary.uk/"
    
    while current_page <= max_pages:
        if current_page == 1:
            url = f"{base_url}?s={keyword}&post_type=pfd"
        else:
            url = f"{base_url}page/{current_page}/?s={keyword}&post_type=pfd"
            
        st.write(f"Scraping page {current_page}: {url}")
        
        try:
            reports = scrape_page(url)
            if not reports:
                break
                
            all_reports.extend(reports)
            st.write(f"Found {len(reports)} reports on page {current_page}")
            
            current_page += 1
            time.sleep(1)  # Be nice to the server
            
        except Exception as e:
            st.error(f"Error processing page {current_page}: {str(e)}")
            break
    
    return all_reports

def main():
    st.title("UK Judiciary PFD Reports Scraper")
    
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Enter keywords to search for relevant reports.
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_keyword = st.text_input("Enter search keywords:", "child")
    with col2:
        max_pages = st.number_input("Maximum pages to search:", min_value=1, max_value=50, value=10)
    
    if st.button("Search Reports"):
        with st.spinner("Searching for reports..."):
            reports = scrape_pfd_reports(search_keyword, max_pages)
            
            if reports:
                df = pd.DataFrame(reports)
                
                st.success(f"Found {len(reports)} reports")
                
                # Display results
                st.dataframe(
                    df,
                    column_config={
                        "URL": st.column_config.LinkColumn("Report Link")
                    },
                    hide_index=True
                )
                
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pfd_reports_{search_keyword}_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            else:
                st.warning("No reports found")

if __name__ == "__main__":
    main()
