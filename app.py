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

def wait_for_element(driver, by, value, timeout=10):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except Exception as e:
        st.error(f"Timeout waiting for element: {value}")
        return None

def scrape_pfd_reports(keyword):
    driver = None
    try:
        # Use the exact URL format from the UI
        search_url = f"https://www.judiciary.uk/?s={keyword}&post_type=pfd"
        st.write(f"Accessing URL: {search_url}")
        
        driver = get_driver()
        driver.get(search_url)
        
        # Wait for page to load and show debug info
        results_header = wait_for_element(driver, By.CLASS_NAME, "search__header")
        if results_header:
            results_text = results_header.text
            st.write(f"Found results header: {results_text}")
            
            # Try to extract the number of results
            match = re.search(r'found (\d+) results', results_text)
            if match:
                expected_results = int(match.group(1))
                st.write(f"Expecting to find {expected_results} results")
        
        # Wait and scroll multiple times to ensure content loads
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        # Try multiple ways to find the articles
        listing_container = wait_for_element(driver, By.CLASS_NAME, "archive__listings")
        if not listing_container:
            st.error("Could not find listings container")
            return pd.DataFrame()
        
        # Debug: Print the HTML of the listing container
        st.write("Listing container HTML:")
        st.code(listing_container.get_attribute('innerHTML')[:1000])
        
        # Try different selectors to find articles
        articles = []
        selectors_to_try = [
            "article",
            ".search-result",
            ".entry",
            "h2.entry-title",
            ".post"
        ]
        
        for selector in selectors_to_try:
            articles = driver.find_elements(By.CSS_SELECTOR, selector)
            if articles:
                st.write(f"Found {len(articles)} articles using selector: {selector}")
                break
        
        if not articles:
            st.warning("Could not find any articles with any selector")
            return pd.DataFrame()
        
        reports = []
        for article in articles:
            try:
                # Try to get title both directly and through h2
                title_elem = None
                try:
                    title_elem = article.find_element(By.CLASS_NAME, "entry-title")
                except:
                    try:
                        title_elem = article.find_element(By.TAG_NAME, "h2")
                    except:
                        if article.tag_name == 'h2':
                            title_elem = article
                
                if not title_elem:
                    continue
                
                # Get link and title
                try:
                    link = title_elem.find_element(By.TAG_NAME, "a")
                    title = link.text.strip()
                    url = link.get_attribute("href")
                except:
                    continue
                
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
                
                # Try to find metadata in various ways
                try:
                    metadata = article.find_element(By.TAG_NAME, "p")
                    if metadata:
                        metadata_text = metadata.text.strip()
                        
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
                                report[key] = match.group(1).strip()
                except:
                    pass
                
                reports.append(report)
                st.write(f"Processed report: {title}")
            
            except Exception as e:
                st.error(f"Error processing article: {str(e)}")
                continue
        
        if reports:
            df = pd.DataFrame(reports)
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
