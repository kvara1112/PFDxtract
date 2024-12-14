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
    # Use the exact URL structure from the screenshot
    search_url = "https://www.judiciary.uk/"
    
    # Use the exact parameters we see in the screenshot
    params = {
        's': keyword,
        'pfd_report_type': '',
        'post_type': 'pfd',
        'order': 'relevance'
    }
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        st.write(f"Status Code: {response.status_code}")
        st.write(f"URL being searched: {response.url}")
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Debug: Let's look at what titles we can find
        all_h2s = soup.find_all('h2')
        st.write(f"Found {len(all_h2s)} h2 elements")
        for h2 in all_h2s[:5]:  # Show first 5 for debugging
            st.write(f"H2 text: {h2.text.strip()}")
            
        # Try to find the results section
        results_section = soup.find('div', class_='search-results')
        if results_section:
            st.write("Found search results section")
            
        # Look for entries in multiple ways
        entries = (
            soup.find_all('article') or 
            soup.find_all('div', class_='search-result') or
            soup.find_all('h2', class_='entry-title')
        )
        
        st.write(f"Found {len(entries)} potential entries")
        
        if not entries:
            st.warning("No reports found for the given keyword.")
            return pd.DataFrame()
        
        reports = []
        for entry in entries:
            try:
                # Get the title element
                title_elem = entry if entry.name == 'h2' else entry.find('h2')
                if not title_elem:
                    continue
                
                # Get the link
                link = title_elem.find('a')
                if not link:
                    continue
                
                title = clean_text(link.text)
                url = link['href']
                
                # Try to find metadata in various locations
                metadata = None
                for elem in [
                    entry.find_next('p'),
                    entry.find('div', class_='entry-content'),
                    title_elem.find_next_sibling('p'),
                    title_elem.parent.find_next_sibling('p')
                ]:
                    if elem and elem.text.strip():
                        metadata = elem
                        break
                
                if metadata:
                    metadata_text = clean_text(metadata.text)
                    
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
                    st.write(f"Successfully parsed: {title}")
                
            except Exception as e:
                st.error(f"Error processing entry: {str(e)}")
                continue
        
        if reports:
            df = pd.DataFrame(reports)
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
                    st.dataframe(
                        df,
                        column_config={
                            "URL": st.column_config.LinkColumn("Report Link")
                        },
                        hide_index=True
                    )
                    
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
