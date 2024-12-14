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
from bs4 import BeautifulSoup

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

def wait_for_element(driver, selector, timeout=10):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return element
    except Exception as e:
        return None

def clean_text(text):
    if text:
        return ' '.join(text.strip().split())
    return ""

def find_articles(driver):
    try:
        # Get page source and create BeautifulSoup object
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'lxml')
        
        # Debug: Print the structure we're looking for
        st.write("Looking for content in HTML...")
        
        # Find the main content area
        main_content = soup.find('div', class_='archive__listings')
        if main_content:
            st.write("Found main content area")
            # Debug: Show first part of content
            st.code(str(main_content)[:500], language='html')
        else:
            st.write("Could not find main content area")
            
        # Find all posts
        posts = soup.find_all(class_='search-results')
        st.write(f"Found {len(posts)} posts with class 'search-results'")
        
        # Try alternative selectors
        entries = soup.find_all(['article', 'div'], class_=['post', 'entry', 'search-result'])
        st.write(f"Found {len(entries)} entries with article/div tags")
        
        # Return Selenium elements that match our findings
        if entries:
            article_elements = driver.find_elements(By.CSS_SELECTOR, ".post, .entry, .search-result, article")
            return article_elements
        return []
        
    except Exception as e:
        st.error(f"Error finding articles: {str(e)}")
        return []

def extract_report_info(article):
    try:
        # Get HTML of the article for debugging
        article_html = article.get_attribute('outerHTML')
        soup = BeautifulSoup(article_html, 'lxml')
        
        # Debug: Print article HTML
        st.write("Processing article HTML:")
        st.code(article_html[:500], language='html')
        
        # Get title and link using BeautifulSoup
        title_elem = soup.find('h2', class_='entry-title')
        if not title_elem or not title_elem.find('a'):
            st.write("Could not find title element")
            return None
            
        link = title_elem.find('a')
        title = clean_text(link.text)
        url = link['href']
        
        st.write(f"Found title: {title}")
        
        # Get metadata
        metadata = soup.find('p')
        if not metadata:
            st.write("Could not find metadata")
            return None
            
        metadata_text = clean_text(metadata.text)
        st.write(f"Found metadata: {metadata_text[:100]}...")
        
        # Extract fields
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
            
        st.write(f"Successfully extracted report info for: {title}")
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
        
        # Check if we're on the right page
        header = wait_for_element(driver, ".search__header")
        if header:
            st.write(f"Found results header: {header.text}")
            
        # Debug: Print page title and URL
        st.write(f"Current page title: {driver.title}")
        st.write(f"Current URL: {driver.current_url}")
        
        reports = []
        processed_urls = set()
        scroll_count = 0
        max_scrolls = 10
        
        while scroll_count < max_scrolls:
            # Find articles
            articles = find_articles(driver)
            initial_count = len(reports)
            
            for article in articles:
                report_info = extract_report_info(article)
                
                if report_info and report_info['URL'] not in processed_urls:
                    reports.append(report_info)
                    processed_urls.add(report_info['URL'])
                    
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
            df = pd.DataFrame(reports)
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
