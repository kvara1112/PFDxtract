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
    if keyword:
        url = f"https://www.judiciary.uk/?s={keyword}&post_type=pfd&paged={page}"
    else:
        url = f"https://www.judiciary.uk/subject/prevention-of-future-deaths/page/{page}/"
    
    st.write(f"Fetching from: {url}")
    soup = get_url(url)
    
    # Debug: Print some HTML to see structure
    st.code(str(soup.select_one('.archive__listings'))[:1000], language='html')
    
    articles = []
    for article in soup.find_all(['article', 'div'], class_=['post', 'search-result']):
        link = article.find('a')
        if link and link.get('href'):
            articles.append({
                'url': link['href'],
                'title': link.text.strip()
            })
    
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
    
    # Get all paragraphs
    paragraphs = content.find_all('p')
    for p in paragraphs:
        text = p.text.strip()
        
        # Extract key information using patterns
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
        search_keyword = st.text_input("Enter search keywords:", "")
    with col2:
        max_pages = st.number_input("Maximum pages to search:", min_value=1, max_value=50, value=10)
    
    if st.button("Search Reports"):
        with st.spinner("Gathering reports..."):
            all_reports = []
            progress_bar = st.progress(0)
            
            for page in range(1, max_pages + 1):
                st.write(f"Processing page {page}")
                
                # Get report URLs from the page
                articles = get_report_urls(search_keyword, page)
                if not articles:
                    st.write(f"No more articles found on page {page}")
                    break
                
                st.write(f"Found {len(articles)} reports on page {page}")
                
                # Process each report
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
                sleep(1)  # Be nice to the server
            
            # Save and display results
            if all_reports:
                df = pd.DataFrame(all_reports)
                
                st.success(f"Found {len(all_reports)} reports")
                st.dataframe(df)
                
                # Download button
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
