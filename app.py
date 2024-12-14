import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re

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

def scrape_pfd_reports(keyword):
    """
    Scrape Prevention of Future Death reports from judiciary.uk based on keyword
    """
    base_url = "https://www.judiciary.uk/"
    # Updated search URL to match the website's format
    search_url = f"{base_url}search/{keyword}/?post_type=pfd"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Add debug information
        st.write(f"Status Code: {response.status_code}")
        st.write(f"URL being searched: {search_url}")
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find main content area
        main_content = soup.find('main', {'id': 'main-content'})
        if not main_content:
            st.warning("Could not find main content area")
            return pd.DataFrame()
            
        # Look for all report entries in the main content
        report_entries = main_content.find_all('article') or main_content.find_all('div', class_='search-result')
        
        st.write(f"Found {len(report_entries)} potential entries")
        
        if not report_entries:
            st.warning("No reports found for the given keyword.")
            return pd.DataFrame()
        
        reports = []
        for entry in report_entries:
            try:
                # First try to find title with class
                title_elem = entry.find('h2', class_='search-result-title') or entry.find('h2')
                if not title_elem:
                    continue
                
                # Get the link and title
                link = title_elem.find('a')
                if not link:
                    continue
                    
                title = clean_text(link.text)
                url = link['href']
                
                # Try to find metadata in different possible locations
                metadata = None
                metadata_candidates = [
                    entry.find('div', class_='search-result-metadata'),
                    entry.find('div', class_='entry-content'),
                    entry.find('p'),
                    title_elem.find_next_sibling('p')
                ]
                
                for candidate in metadata_candidates:
                    if candidate and candidate.text.strip():
                        metadata = candidate
                        break
                
                if metadata:
                    metadata_text = clean_text(metadata.text)
                    
                    # Extract information using regex
                    patterns = {
                        'Date': r'Date of report:?\s*(\d{2}/\d{2}/\d{4})',
                        'Reference': r'Ref:?\s*([\w-]+)',
                        'Deceased_Name': r'Deceased name:?\s*([^,\n]+)',
                        'Coroner_Name': r'Coroner name:?\s*([^,\n]+)',
                        'Coroner_Area': r'Coroner Area:?\s*([^,\n]+)',
                        'Category': r'Category:?\s*([^|]+)'
                    }
                    
                    report = {
                        'Title': title,
                        'URL': url
                    }
                    
                    for key, pattern in patterns.items():
                        match = re.search(pattern, metadata_text)
                        report[key] = clean_text(match.group(1)) if match else ""
                    
                    reports.append(report)
                    st.write(f"Successfully parsed report: {title}")
                
            except Exception as e:
                st.error(f"Error processing entry: {str(e)}")
                continue
        
        if reports:
            df = pd.DataFrame(reports)
            
            # Reorder columns
            column_order = ['Title', 'Date', 'Reference', 'Deceased_Name', 'Coroner_Name', 'Coroner_Area', 'Category', 'URL']
            df = df[column_order]
            
            return df
        else:
            st.warning("No reports could be parsed from the page.")
            return pd.DataFrame()
    
    except requests.RequestException as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("UK Judiciary PFD Reports Scraper")
    
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Enter a keyword to search for relevant reports.
    """)
    
    with st.form("search_form"):
        keyword = st.text_input("Enter search keyword:", "maternity")
        submitted = st.form_submit_button("Search Reports")
        
        if submitted:
            with st.spinner("Searching for reports..."):
                df = scrape_pfd_reports(keyword)
                
                if not df.empty:
                    st.success(f"Found {len(df)} reports")
                    
                    # Display results in a clean table
                    st.dataframe(
                        df,
                        column_config={
                            "URL": st.column_config.LinkColumn("Report Link")
                        },
                        hide_index=True
                    )
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pfd_reports_{keyword}_{timestamp}.csv"
                    
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
