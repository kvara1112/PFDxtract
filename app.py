import streamlit as st
import pandas as pd
from datetime import datetime
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

st.set_page_config(
    page_title="UK Judiciary PFD Reports Scraper",
    page_icon="📚",
    layout="wide"
)

def clean_text(text):
    """Clean extracted text by removing extra whitespace and newlines"""
    if text is None:
        return ""
    try:
        return ' '.join(str(text).strip().split())
    except (AttributeError, TypeError):
        return ""

def get_driver():
    """Set up Chrome driver with appropriate options for Streamlit Cloud"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--remote-debugging-port=9222')
    # Add user agent to mimic real browser
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36')
    return webdriver.Chrome(options=options)

def scrape_pfd_reports(keyword):
    """
    Scrape Prevention of Future Death reports from judiciary.uk based on keyword
    """
    driver = None
    try:
        search_url = f"https://www.judiciary.uk/?s={keyword}&pfd_report_type=&post_type=pfd&order=relevance"
        st.write(f"Accessing URL: {search_url}")
        
        driver = get_driver()
        driver.get(search_url)
        
        # Wait for the page to load
        time.sleep(5)
        
        # Get the page source
        page_source = driver.page_source
        
        # Debug: Show a portion of the HTML
        st.code(page_source[:5000], language='html')
        
        soup = BeautifulSoup(page_source, 'lxml')
        
        # Debug: Look for specific elements
        main_content = soup.find('main', {'id': 'main-content'})
        if main_content:
            st.write("Found main content area")
            # Show the first part of main content
            st.code(str(main_content)[:1000], language='html')
        else:
            st.write("Could not find main content area")
        
        # Try different selectors
        entries = (
            soup.find_all('article') or 
            soup.find_all(['h2', 'div'], class_=['entry-title', 'search-result']) or
            soup.select('.search-results article') or
            soup.select('h2.entry-title')
        )
        
        st.write(f"Found {len(entries)} potential entries")
        
        # Debug: Show all h2 elements
        all_h2s = soup.find_all('h2')
        st.write(f"Found {len(all_h2s)} h2 elements")
        for h2 in all_h2s[:5]:
            st.write(f"H2 text: {h2.text.strip() if h2 else 'None'}")
        
        reports = []
        for entry in entries:
            try:
                # Find the title and link
                title_tag = entry if entry.name == 'h2' else entry.find('h2')
                if not title_tag:
                    continue
                    
                link = title_tag.find('a')
                if not link:
                    continue
                
                title = clean_text(link.text)
                url = link.get('href', '')
                
                # Initialize report
                report = {
                    'Title': title,
                    'URL': url,
                    'Date': '',
                    'Reference': '',
                    'Deceased_Name': '',
                    'Coroner_Name': '',
                    'Coroner_Area': '',
                    'Category': ''
                }
                
                # Look for metadata
                metadata = None
                if entry.name == 'h2':
                    metadata = entry.find_next('p')
                else:
                    metadata = entry.find('p')
                
                if metadata and metadata.text:
                    metadata_text = clean_text(metadata.text)
                    
                    patterns = {
                        'Date': r'Date of report:?\s*(\d{2}/\d{2}/\d{4})',
                        'Reference': r'Ref:?\s*([\w-]+)',
                        'Deceased_Name': r'Deceased name:?\s*([^,\n]+)',
                        'Coroner_Name': r'Coroner name:?\s*([^,\n]+)',
                        'Coroner_Area': r'Coroner Area:?\s*([^,\n]+)',
                        'Category': r'Category:?\s*([^|]+)'
                    }
                    
                    for key, pattern in patterns.items():
                        match = re.search(pattern, metadata_text)
                        if match:
                            report[key] = clean_text(match.group(1))
                
                reports.append(report)
                
            except Exception as e:
                st.error(f"Error processing entry: {str(e)}")
                continue
        
        if reports:
            df = pd.DataFrame(reports)
            df = df[['Title', 'Date', 'Reference', 'Deceased_Name', 'Coroner_Name', 'Coroner_Area', 'Category', 'URL']]
            return df
        
        st.warning("No reports could be extracted from the page.")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return pd.DataFrame()
    
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

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
