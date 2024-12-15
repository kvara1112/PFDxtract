import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from time import sleep
import pandas as pd
from datetime import date
import urllib3
from tqdm.auto import tqdm
import io

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Scraper", layout="wide")

def get_url(url):
    """Get URL content with retries"""
    response = requests.get(url, verify=False)
    return BeautifulSoup(response.content, "html.parser")

def retries(record_url, tries=3):
    """Retry URL fetch with multiple attempts"""
    for i in range(tries):
        try:
            return get_url(record_url)
        except Exception:
            if i < tries - 1:
                sleep(2)
                continue
            else:
                return 'Con error'

def main():
    st.title("UK Judiciary PFD Reports Scraper")
    
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Enter keywords to search for relevant reports.
    """)
    
    # Initialize session state
    if 'record_text' not in st.session_state:
        st.session_state.record_text = []
    if 'pdf_urls' not in st.session_state:
        st.session_state.pdf_urls = []
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_keyword = st.text_input("Enter search keywords:", "")
    with col2:
        max_pages = st.number_input("Maximum pages to search:", min_value=1, max_value=50, value=10)
    
    if st.button("Search Reports"):
        with st.spinner("Gathering reports..."):
            # Reset lists
            record_text = []
            pdf_urls = []
            error_catching = []
            record_count = 0
            
            # Define text categories to extract
            text_cats = ['Date of report', 'Ref', 'Deceased name', 'Coroner name', 
                        'Coroner Area', 'Category', "This report is being sent to"]
            
            # Get all report URLs
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for page in range(1, max_pages + 1):
                try:
                    # Construct search URL
                    if search_keyword:
                        url = f"https://www.judiciary.uk/?s={search_keyword}&post_type=pfd&paged={page}"
                    else:
                        url = f"https://www.judiciary.uk/?post_type=pfd&paged={page}"
                    
                    soup = get_url(url)
                    articles = soup.select('.archive__listings article, .search-results article')
                    
                    if not articles:
                        break
                        
                    status_text.text(f"Processing page {page} - Found {len(articles)} reports")
                    
                    for article in articles:
                        try:
                            # Get URL of full report
                            link = article.select_one('.entry-title a')
                            if not link:
                                continue
                                
                            record_url = link['href']
                            
                            # Get full report content
                            report_soup = retries(record_url)
                            if report_soup == 'Con error':
                                continue
                                
                            # Extract report content
                            death_info = report_soup.find('div', {'class':'entry-content'})
                            if not death_info:
                                continue
                                
                            # Process metadata
                            blankdict = {}
                            paragraphs = death_info.find_all('p')
                            
                            for p in paragraphs:
                                text = p.text.strip()
                                if ':' in text:
                                    parts = text.split(':')
                                    key = parts[0].strip()
                                    if key in text_cats:
                                        dict_key = key.lower().replace(' ', '_')
                                        blankdict[dict_key] = parts[1].strip()
                            
                            blankdict['url'] = record_url
                            record_text.append(blankdict)
                            
                            # Get PDFs
                            pdf_list = []
                            pdfs = death_info.find_all('a', href=re.compile(r'\.pdf$'))
                            for pdf in pdfs:
                                pdf_list.append(pdf['href'])
                            pdf_urls.append(pdf_list)
                            
                            record_count += 1
                            
                        except Exception as e:
                            st.error(f"Error processing record: {str(e)}")
                            continue
                            
                    progress_bar.progress(page / max_pages)
                    sleep(1)  # Be nice to the server
                    
                except Exception as e:
                    st.error(f"Error processing page {page}: {str(e)}")
                    break
            
            # Save results
            if record_text:
                # Save to CSV
                df = pd.DataFrame(record_text)
                csv = df.to_csv(index=False).encode('utf-8')
                
                st.success(f"Found {len(record_text)} reports")
                
                # Display results
                st.dataframe(df)
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                # CSV download
                timestamp = date.today().strftime("%Y%m%d")
                filename = f"death_info_{timestamp}.csv"
                
                col1.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
                
                # PDFs download
                if pdf_urls:
                    pdfs_found = sum(len(urls) for urls in pdf_urls)
                    if pdfs_found > 0:
                        col2.write(f"Found {pdfs_found} PDFs")
                        st.info("PDF download feature to be implemented based on your requirements")
                
            else:
                st.warning("No reports found")
            
            # Clear progress
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()
