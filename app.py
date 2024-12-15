import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from time import sleep
import pandas as pd
from datetime import date
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Scraper", layout="wide")

def get_url(url):
    """Get URL content with retries"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, verify=False, headers=headers)
    return BeautifulSoup(response.content, "html.parser")

def get_report_urls(keyword, page=1):
    """Get all report URLs from a search page"""
    base_url = "https://www.judiciary.uk/"
    if keyword:
        params = {
            's': keyword,
            'post_type': 'pfd',
            'paged': page
        }
        url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    else:
        url = f"{base_url}subject/prevention-of-future-deaths/page/{page}/"
    
    st.write(f"Fetching from: {url}")
    soup = get_url(url)
    
    # Find results header to confirm we have results
    results_header = soup.find('div', class_='search__header')
    if results_header:
        header_text = results_header.text.strip()
        st.write(f"Found: {header_text}")
    
    # Get all article listings
    articles_section = soup.find('div', class_='archive__listings')
    if not articles_section:
        return []
        
    # Debug: Show the HTML structure we're working with
    st.write("HTML Structure:")
    st.code(str(articles_section)[:1000], language='html')
    
    articles = []
    # Look for articles after the search form
    search_form = articles_section.find('form', class_='search__form')
    if search_form:
        current_element = search_form.find_next_sibling()
        while current_element:
            if current_element.name == 'article':
                link = current_element.find('a', class_='view-more')
                if link:
                    articles.append({
                        'url': link['href'],
                        'title': link.text.strip()
                    })
            current_element = current_element.find_next_sibling()
    
    st.write(f"Found {len(articles)} articles")
    return articles

def extract_report_info(url):
    """Extract information from a single report"""
    soup = get_url(url)
    content = soup.find('div', class_='entry-content')
    if not content:
        return None
        
    info = {
        'url': url,
        'date_of_report': '',
        'ref': '',
        'deceased_name': '',
        'coroner_name': '',
        'coroner_area': '',
        'category': '',
        'this_report_is_being_sent_to': ''
    }
    
    paragraphs = content.find_all('p')
    for p in paragraphs:
        text = p.text.strip()
        
        patterns = {
            'date_of_report': r'Date of report:?\s*([^\n]+)',
            'ref': r'Ref:?\s*([\w-]+)',
            'deceased_name': r'Deceased name:?\s*([^\n]+)',
            'coroner_name': r'Coroner name:?\s*([^\n]+)',
            'coroner_area': r'Coroner Area:?\s*([^\n]+)',
            'category': r'Category:?\s*([^|\n]+)',
            'this_report_is_being_sent_to': r'This report is being sent to:?\s*([^\n]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                info[key] = match.group(1).strip()
    
    return info

def main():
    st.title("UK Judiciary PFD Reports Scraper")
    
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Enter keywords to search for relevant reports.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_keyword = st.text_input("Enter search keywords:", "child")
        
    with col2:
        max_pages = st.number_input("Maximum pages to search:", min_value=1, max_value=50, value=10)
    
    if st.button("Search Reports"):
        with st.spinner("Gathering reports..."):
            all_reports = []
            progress_bar = st.progress(0)
            
            for page in range(1, max_pages + 1):
                st.write(f"Processing page {page}")
                
                articles = get_report_urls(search_keyword, page)
                if not articles:
                    st.write(f"No more articles found on page {page}")
                    break
                
                for idx, article in enumerate(articles):
                    try:
                        report_info = extract_report_info(article['url'])
                        if report_info:
                            all_reports.append(report_info)
                            st.write(f"Processed: {article['title']}")
                    except Exception as e:
                        st.error(f"Error processing {article['url']}: {str(e)}")
                        continue
                
                progress_bar.progress(page / max_pages)
                sleep(1)
            
            if all_reports:
                df = pd.DataFrame(all_reports)
                
                st.success(f"Found {len(all_reports)} reports")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode('utf-8')
                timestamp = date.today().strftime("%Y%m%d")
                filename = f"death_info_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            else:
                st.warning("No reports found")
            
            progress_bar.empty()

if __name__ == "__main__":
    main()
