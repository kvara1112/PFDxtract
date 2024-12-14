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
from selenium.webdriver.common.action_chains import ActionChains

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
        return None

def scroll_and_collect(driver, max_scrolls=10):
    """Scroll through the page and collect articles"""
    collected_articles = set()
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0
    
    while scroll_count < max_scrolls:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for content to load
        
        # Get new page height
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        # Find all articles currently visible
        articles = driver.find_elements(By.CSS_SELECTOR, "article")
        current_count = len(collected_articles)
        
        # Add new articles to our set
        for article in articles:
            try:
                title = article.find_element(By.CSS_SELECTOR, "h2.entry-title").text
                collected_articles.add(title)
            except:
                continue
        
        # Debug output
        new_count = len(collected_articles)
        if new_count > current_count:
            st.write(f"Found {new_count - current_count} new articles. Total: {new_count}")
        
        # If no new height, we've reached the bottom
        if new_height == last_height:
            scroll_count += 1
        else:
            scroll_count = 0
            last_height = new_height
            
        # Try to click "Load more" button if it exists
        try:
            load_more = driver.find_element(By.CSS_SELECTOR, ".load-more, .pagination-next")
            if load_more and load_more.is_displayed():
                actions = ActionChains(driver)
                actions.move_to_element(load_more).click().perform()
                time.sleep(2)
        except:
            pass
    
    return driver.find_elements(By.CSS_SELECTOR, "article")

def scrape_pfd_reports(keyword):
    driver = None
    try:
        search_url = f"https://www.judiciary.uk/?s={keyword}&post_type=pfd"
        st.write(f"Accessing URL: {search_url}")
        
        driver = get_driver()
        driver.get(search_url)
        
        # Wait for results to load
        results_header = wait_for_element(driver, By.CLASS_NAME, "search__header")
        if results_header:
            results_text = results_header.text
            st.write(f"Found results header: {results_text}")
            
            # Extract expected number of results
            match = re.search(r'found (\d+) results', results_text)
            if match:
                expected_results = int(match.group(1))
                st.write(f"Expecting to find {expected_results} results")
        
        # Scroll and collect articles with progress bar
        st.write("Scrolling through results...")
        progress_bar = st.progress(0)
        
        articles = scroll_and_collect(driver)
        st.write(f"Found {len(articles)} articles after scrolling")
        
        reports = []
        for index, article in enumerate(articles):
            try:
                # Update progress
                progress_bar.progress((index + 1) / len(articles))
                
                # Get title and link
                title_elem = article.find_element(By.CSS_SELECTOR, "h2.entry-title a")
                title = title_elem.text.strip()
                url = title_elem.get_attribute("href")
                
                # Get metadata
                metadata = article.find_element(By.TAG_NAME, "p")
                metadata_text = metadata.text.strip()
                
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
                        report[key] = match.group(1).strip()
                
                reports.append(report)
                
                # Show progress
                if (index + 1) % 10 == 0:
                    st.write(f"Processed {index + 1} reports...")
                
            except Exception as e:
                st.error(f"Error processing article: {str(e)}")
                continue
        
        progress_bar.progress(1.0)
        
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
        max_results = st.number_input("Maximum number of results to fetch (0 for all):", 
                                    value=50, min_value=0)
        submitted = st.form_submit_button("Search Reports")
        
        if submitted:
            with st.spinner("Searching for reports..."):
                df = scrape_pfd_reports(keyword)
                
                if not df.empty:
                    if max_results > 0:
                        df = df.head(max_results)
                        
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
