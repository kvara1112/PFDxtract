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
    """Clean extracted text by removing extra whitespace and newlines"""
    if text:
        return ' '.join(text.strip().split())
    return ""

def get_reports_by_keyword(keyword, max_pages=10):
    """
    Scrape PFD reports based on keyword search
    """
    base_url = "https://www.judiciary.uk/"
    reports = []
    
    params = {
        's': keyword,
        'post_type': 'pfd',
        'order': 'relevance'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
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
            
            # Find all entries using more specific selector
            entries = soup.select('.archive__listings article')
            if not entries:
                st.write(f"No more results found on page {page}")
                break
            
            st.write(f"Processing page {page} - Found {len(entries)} entries")
            
            for entry in entries:
                try:
                    # Find title and link
                    title_elem = entry.select_one('.entry-title a')
                    if not title_elem:
                        continue
                    
                    title = clean_text(title_elem.text)
                    url = title_elem['href']
                    
                    # Find metadata paragraph
                    metadata = entry.find('p')
                    if not metadata:
                        continue
                    
                    metadata_text = clean_text(metadata.text)
                    
                    # Extract fields using patterns
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
                    
                    reports.append(report)
                    
                except Exception as e:
                    st.error(f"Error processing entry: {str(e)}")
                    continue
            
            time.sleep(1)  # Pause between pages
            
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
                # Create DataFrame
                df = pd.DataFrame(reports)
                
                # Reorder columns
                columns = ['Title', 'Date', 'Reference', 'Deceased_Name', 'Coroner_Name', 
                          'Coroner_Area', 'Category', 'Trust', 'URL']
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
                
                # Add download button
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    key='download-csv'
                )
                
                # Show some statistics
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
