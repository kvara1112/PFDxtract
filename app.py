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

def extract_report_info(article):
    """Extract report information from an article element"""
    try:
        # Get title and URL
        title_elem = article.find_element(By.CSS_SELECTOR, "h2.entry-title a")
        title = title_elem.text.strip()
        url = title_elem.get_attribute("href")
        
        # Get metadata paragraph
        metadata = article.find_element(By.TAG_NAME, "p").text.strip()
        
        # Extract information using patterns
        patterns = {
            'Date': r'Date of report:?\s*(\d{2}/\d{2}/\d{4})',
            'Reference': r'Ref:?\s*([\w-]+)',
            'Deceased_Name': r'Deceased name:?\s*([^,\n]+)',
            'Coroner_Name': r'Coroner name:?\s*([^,\n]+)',
            'Coroner_Area': r'Coroner Area:?\s*([^,\n]+)',
            'Category': r'Category:?\s*([^|\n]+)',
            'Hospital': r'This report is being sent to:\s*([^|\n]+)'
        }
        
        # Initialize report with default values
        report = {
            'Title': title,
            'URL': url
        }
        
        # Add extracted fields
        for key, pattern in patterns.items():
            match = re.search(pattern, metadata)
            report[key] = match.group(1).strip() if match else ""
            
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
        time.sleep(3)  # Wait for initial load
        
        # Find total number of results
        results_header = driver.find_element(By.CLASS_NAME, "search__header").text
        st.write(f"Found results header: {results_header}")
        
        reports = []
        processed_titles = set()
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        with st.progress(0) as progress_bar:
            while len(reports) < max_results or max_results == 0:
                # Find all current articles
                articles = driver.find_elements(By.TAG_NAME, "article")
                
                for article in articles:
                    # Extract information from article
                    report_info = extract_report_info(article)
                    
                    if report_info and report_info['Title'] not in processed_titles:
                        reports.append(report_info)
                        processed_titles.add(report_info['Title'])
                        
                        # Update progress
                        if max_results > 0:
                            progress_bar.progress(min(len(reports) / max_results, 1.0))
                        
                        # Show periodic updates
                        if len(reports) % 10 == 0:
                            st.write(f"Processed {len(reports)} reports...")
                        
                        if max_results > 0 and len(reports) >= max_results:
                            break
                
                # Scroll down
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Check if we've reached the bottom
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
        
        if reports:
            df = pd.DataFrame(reports)
            # Reorder columns
            columns = ['Title', 'Date', 'Reference', 'Deceased_Name', 'Coroner_Name', 
                      'Coroner_Area', 'Category', 'Hospital', 'URL']
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
        keyword = st.text_input("Enter search keyword:", "maternity")
        max_results = st.number_input("Maximum number of results to fetch (0 for all):", 
                                    value=50, min_value=0)
        submitted = st.form_submit_button("Search Reports")
        
        if submitted:
            with st.spinner("Searching for reports..."):
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
