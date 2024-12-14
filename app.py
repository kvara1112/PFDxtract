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
import time

st.set_page_config(
    page_title="UK Judiciary PFD Reports Scraper",
    page_icon="📚",
    layout="wide"
)

def get_driver():
    """Set up Chrome driver with appropriate options for Streamlit Cloud"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument("--window-size=1920,1080")
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36')
    return webdriver.Chrome(options=options)

def wait_for_presence(driver, selector, timeout=10):
    """Wait for element to be present and return it"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return element
    except Exception as e:
        return None

def clean_text(text):
    """Clean text by removing extra whitespace and newlines"""
    if text:
        return ' '.join(text.strip().split())
    return ""

def find_articles(driver):
    """Find all article elements using multiple methods"""
    try:
        # Wait for content to load
        content = wait_for_presence(driver, ".archive__listings")
        if not content:
            st.warning("Could not find content container")
            return []
            
        # Try multiple methods to find articles
        articles = driver.find_elements(By.CSS_SELECTOR, ".post")
        if not articles:
            articles = driver.find_elements(By.CSS_SELECTOR, "article")
        if not articles:
            articles = driver.find_elements(By.CSS_SELECTOR, ".search-result")
            
        return articles
    except Exception as e:
        st.error(f"Error finding articles: {str(e)}")
        return []

def extract_report_info(article):
    """Extract report information from an article element"""
    try:
        # Debug info
        st.write("Processing article...")
        
        # Get title and link
        title_elem = article.find_element(By.CSS_SELECTOR, "h2 a")
        if not title_elem:
            return None
            
        title = clean_text(title_elem.text)
        url = title_elem.get_attribute("href")
        
        st.write(f"Found title: {title}")
        
        # Get metadata
        metadata = article.find_element(By.TAG_NAME, "p")
        if not metadata:
            return None
            
        metadata_text = clean_text(metadata.text)
        st.write(f"Found metadata: {metadata_text[:100]}...")
        
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
            
        st.write(f"Extracted report info for: {title}")
        return report
        
    except Exception as e:
        st.error(f"Error extracting report info: {str(e)}")
        return None

def scrape_pfd_reports(keyword, max_results=50):
    driver = None
    try:
        search_url = f"https://www.judiciary.uk/?s={keyword}&post_type=pfd"
        st.write(f"Accessing URL: {search_url}")
        
        driver = get_driver()
        driver.get(search_url)
        time.sleep(3)
        
        # Find results header
        header = wait_for_presence(driver, ".search__header")
        if header:
            st.write(f"Found results header: {header.text}")
        
        reports = []
        processed_urls = set()
        scroll_count = 0
        max_scrolls = 10 if max_results > 0 else 20
        
        with st.spinner("Loading reports..."):
            while scroll_count < max_scrolls:
                # Find articles
                articles = find_articles(driver)
                initial_count = len(reports)
                
                st.write(f"Found {len(articles)} articles on current page")
                
                for article in articles:
                    report_info = extract_report_info(article)
                    
                    if report_info and report_info['URL'] not in processed_urls:
                        reports.append(report_info)
                        processed_urls.add(report_info['URL'])
                        
                        st.write(f"Added report: {report_info['Title']}")
                        
                        if max_results > 0 and len(reports) >= max_results:
                            break
                
                if len(reports) == initial_count:
                    scroll_count += 1
                else:
                    scroll_count = 0
                    
                # Scroll down
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                if max_results > 0 and len(reports) >= max_results:
                    break
        
        if reports:
            st.write(f"Total reports found: {len(reports)}")
            df = pd.DataFrame(reports)
            # Reorder columns
            columns = ['Title', 'Date', 'Reference', 'Deceased_Name', 'Coroner_Name', 
                      'Coroner_Area', 'Category', 'Trust', 'URL']
            df = df[columns]
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
        keyword = st.text_input("Enter search keyword:", "child")
        max_results = st.number_input("Maximum number of results to fetch (0 for all):", 
                                    value=50, min_value=0)
        submitted = st.form_submit_button("Search Reports")
        
        if submitted:
            df = scrape_pfd_reports(keyword, max_results)
            
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
