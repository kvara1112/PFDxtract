import streamlit as st
import pandas as pd
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import time

st.set_page_config(
    page_title="UK Judiciary PFD Reports Scraper",
    page_icon="📚",
    layout="wide"
)

def clean_text(text):
    if text:
        return ' '.join(text.strip().split())
    return ""

def extract_section_content(text, section_number, next_section_number=None):
    """Extract content between section numbers"""
    if next_section_number:
        pattern = fr'{section_number}\s+(.+?)(?={next_section_number}|$)'
    else:
        pattern = fr'{section_number}\s+(.+?)$'
    match = re.search(pattern, text, re.DOTALL)
    return clean_text(match.group(1)) if match else ""

def get_report_details(url):
    """Get detailed information from individual report page"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Get the main content
        content = soup.find('div', class_='entry-content')
        if not content:
            return None
            
        text = content.get_text()
        
        # Extract sections
        sections = {
            'CORONER': extract_section_content(text, '1', '2'),
            'LEGAL_POWERS': extract_section_content(text, '2', '3'),
            'INVESTIGATION': extract_section_content(text, '3', '4'),
            'CIRCUMSTANCES': extract_section_content(text, '4', '5'),
            'CONCERNS': extract_section_content(text, '5', '6'),
            'ACTION': extract_section_content(text, '6', '7'),
            'RESPONSE': extract_section_content(text, '7', '8'),
            'DISTRIBUTION': extract_section_content(text, '8', '9'),
            'DATE': extract_section_content(text, '9')
        }
        
        return sections
        
    except Exception as e:
        st.error(f"Error fetching report details: {str(e)}")
        return None

def get_reports_by_keyword(keyword, max_pages=10):
    base_url = "https://www.judiciary.uk/"
    reports = []
    
    params = {
        's': keyword,
        'post_type': 'pfd',
        'order': 'relevance'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    st.write(f"Searching for reports with keyword: {keyword}")
    
    for page in range(1, max_pages + 1):
        try:
            if page > 1:
                params['paged'] = page
            
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find results header
            results_header = soup.find('div', class_='search__header')
            if results_header and page == 1:
                st.write(f"Found results: {results_header.text.strip()}")
            
            # Find all report entries
            entries = soup.select('.search-results article, .archive__listings article')
            
            st.write(f"Processing page {page} - Found {len(entries)} reports")
            
            for entry in entries:
                try:
                    # Get basic information
                    title_elem = entry.select_one('.entry-title a')
                    if not title_elem:
                        continue
                        
                    title = clean_text(title_elem.text)
                    url = title_elem['href']
                    
                    metadata = entry.find('p')
                    if not metadata:
                        continue
                        
                    metadata_text = clean_text(metadata.text)
                    
                    # Extract basic metadata
                    patterns = {
                        'Date': r'Date of report:?\s*(\d{2}/\d{2}/\d{4})',
                        'Reference': r'Ref:?\s*([\w-]+)',
                        'Deceased_Name': r'Deceased name:?\s*([^,\n]+)',
                        'Coroner_Name': r'Coroner name:?\s*([^,\n]+)',
                        'Coroner_Area': r'Coroner Area:?\s*([^,\n]+)',
                        'Category': r'Category:?\s*([^|\n]+)',
                        'Trust': r'This report is being sent to:\s*([^|\n]+)'
                    }
                    
                    report = {
                        'Title': title,
                        'URL': url
                    }
                    
                    for key, pattern in patterns.items():
                        match = re.search(pattern, metadata_text)
                        report[key] = clean_text(match.group(1)) if match else ""
                    
                    # Get detailed information
                    st.write(f"Fetching details for: {title}")
                    details = get_report_details(url)
                    if details:
                        report.update(details)
                    
                    reports.append(report)
                    
                except Exception as e:
                    st.error(f"Error processing entry: {str(e)}")
                    continue
            
            time.sleep(1)
            
        except Exception as e:
            st.error(f"Error processing page {page}: {str(e)}")
            break
    
    return reports

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
            reports = get_reports_by_keyword(search_keyword, max_pages)
            
            if reports:
                df = pd.DataFrame(reports)
                
                # Set column order
                columns = [
                    'Title', 'Date', 'Reference', 'Deceased_Name', 'Coroner_Name', 
                    'Coroner_Area', 'Category', 'Trust', 'CORONER', 'LEGAL_POWERS',
                    'INVESTIGATION', 'CIRCUMSTANCES', 'CONCERNS', 'ACTION', 
                    'RESPONSE', 'DISTRIBUTION', 'DATE', 'URL'
                ]
                
                df = df.reindex(columns=columns)
                
                st.success(f"Found {len(reports)} reports")
                
                # Display results
                st.dataframe(
                    df,
                    column_config={
                        "URL": st.column_config.LinkColumn("Report Link")
                    },
                    hide_index=True
                )
                
                # Prepare CSV download
                csv = df.to_csv(index=False).encode('utf-8')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pfd_reports_{search_keyword.replace(' ', '_')}_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    key='download-csv'
                )
                
                # Show statistics
                st.write("### Report Statistics")
                st.write(f"- Total reports found: {len(reports)}")
                if df['Category'].notna().any():
                    st.write("#### Categories:")
                    for category, count in df['Category'].value_counts().items():
                        st.write(f"- {category}: {count} reports")
            else:
                st.warning(f"No reports found for search: {search_keyword}")

if __name__ == "__main__":
    main()
