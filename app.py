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
    if text:
        return ' '.join(text.strip().split())
    return ""

@st.cache_resource
def get_chrome_driver():
    """Initialize and cache the Chrome driver"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scrape_pfd_reports(keyword):
    """
    Scrape Prevention of Future Death reports from judiciary.uk based on keyword
    """
    search_url = f"https://www.judiciary.uk/?s={keyword}&pfd_report_type=&post_type=pfd&order=relevance"
    
    try:
        driver = get_chrome_driver()
        
        st.write(f"Accessing URL: {search_url}")
        driver.get(search_url)
        
        # Wait for the results to load
        time.sleep(3)  # Give JavaScript time to execute
        
        # Wait for either the results or a no-results message
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "entry-title"))
        )
        
        # Get the page source after JavaScript has run
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'lxml')
        
        # Find all report entries
        report_entries = soup.find_all('article') or soup.find_all('div', class_='search-result')
        
        st.write(f"Found {len(report_entries)} potential entries")
        
        reports = []
        for entry in report_entries:
            try:
                # Get the title element
                title_elem = entry.find('h2', class_='entry-title')
                if not title_elem:
                    continue
                
                link = title_elem.find('a')
                if not link:
                    continue
                
                title = clean_text(link.text)
                url = link['href']
                
                # Get metadata
                metadata = entry.find('p')
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
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return pd.DataFrame()
    finally:
        driver.quit()

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
