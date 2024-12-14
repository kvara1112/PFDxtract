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
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

st.set_page_config(
    page_title="UK Judiciary PFD Reports Scraper",
    page_icon="📚",
    layout="wide"
)

def clean_text(text):
    """Clean extracted text by removing extra whitespace and newlines"""
    if not text:
        return ""
    try:
        return ' '.join(str(text).strip().split())
    except Exception:
        return str(text).strip()

def setup_driver():
    """Set up and return the Chrome driver with appropriate options"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def scrape_pfd_reports(keyword):
    """
    Scrape Prevention of Future Death reports from judiciary.uk based on keyword
    """
    driver = None
    try:
        search_url = f"https://www.judiciary.uk/?s={keyword}&pfd_report_type=&post_type=pfd&order=relevance"
        st.write(f"Accessing URL: {search_url}")
        
        driver = setup_driver()
        driver.get(search_url)
        
        # Wait for content to load
        time.sleep(3)
        
        try:
            # Wait for content or timeout after 10 seconds
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
        except Exception as e:
            st.warning("Timed out waiting for results to load")
        
        # Get page source
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'lxml')
        
        # Debug output
        st.write("Page loaded successfully")
        
        # Find main content
        main_content = soup.find('main', id='main-content')
        if not main_content:
            st.warning("Could not find main content area")
            return pd.DataFrame()
        
        # Find all articles
        report_entries = main_content.find_all('article')
        st.write(f"Found {len(report_entries)} reports")
        
        if not report_entries:
            st.warning("No reports found")
            return pd.DataFrame()
        
        reports = []
        for entry in report_entries:
            try:
                # Find title
                title_elem = entry.find('h2', class_='entry-title')
                if not title_elem or not title_elem.find('a'):
                    continue
                
                link = title_elem.find('a')
                title = clean_text(link.text)
                url = link.get('href', '')
                
                # Find metadata
                metadata = entry.find('p')
                if not metadata:
                    continue
                    
                metadata_text = clean_text(metadata.text)
                
                # Create report dictionary
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
                
                # Extract metadata using patterns
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
                st.write(f"Processed report: {title}")
                
            except Exception as e:
                st.error(f"Error processing entry: {str(e)}")
                continue
        
        if reports:
            df = pd.DataFrame(reports)
            return df
        
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
                    
                    # Display results
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
