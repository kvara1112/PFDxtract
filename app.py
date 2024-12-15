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

def get_url(url, debug=False):
    """Get URL content with retries"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        
        if debug:
            st.write(f"Response status: {response.status_code}")
            st.write("Full HTML structure:")
            html_content = response.text
            st.code(html_content, language='html')
        
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        st.error(f"Error fetching URL {url}: {str(e)}")
        return None

def find_reports(soup):
    """Find all report entries in the page"""
    reports = []
    
    # First look for the results header to confirm we have results
    header = soup.find('div', class_='search__header')
    if header:
        header_text = header.text.strip()
        st.write(f"Found header: {header_text}")
        
        # Check if we have results
        if 'found 0 results' in header_text.lower():
            return reports
        
        # Extract number of results if available
        match = re.search(r'found (\d+) results', header_text.lower())
        if match:
            expected_results = int(match.group(1))
            st.write(f"Expecting {expected_results} results")
    
    # Look for the main content area
    listings = soup.find('div', class_='archive__listings')
    if listings:
        st.write("Found listings container")
        st.code(str(listings), language='html')
        
        # Look for articles after the search form
        search_form = listings.find('form', class_='search__form')
        if search_form:
            st.write("Found search form, looking for articles after it")
            current = search_form.find_next_sibling()
            
            while current:
                if current.name == 'article':
                    st.write(f"Found article: {current}")
                    link = current.find('a', class_='entry-link')
                    if not link:
                        link = current.find('h2', class_='entry-title').find('a')
                    
                    if link:
                        reports.append({
                            'url': link['href'],
                            'title': link.text.strip()
                        })
                        st.write(f"Added report: {link.text.strip()}")
                current = current.find_next_sibling()
    
    st.write(f"Total reports found: {len(reports)}")
    return reports

def extract_metadata(url):
    """Extract metadata from a report URL"""
    try:
        soup = get_url(url)
        if not soup:
            return None
            
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
        full_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
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
            match = re.search(pattern, full_text)
            if match:
                info[key] = match.group(1).strip()
        
        return info
        
    except Exception as e:
        st.error(f"Error extracting metadata from {url}: {str(e)}")
        return None

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
        with st.spinner("Searching for reports..."):
            all_reports = []
            progress_bar = st.progress(0)
            
            for page in range(1, max_pages + 1):
                url = f"https://www.judiciary.uk/?s={search_keyword}&post_type=pfd"
                if page > 1:
                    url += f"&paged={page}"
                    
                st.write(f"Searching page {page}: {url}")
                
                soup = get_url(url, debug=True)
                if not soup:
                    continue
                
                page_reports = find_reports(soup)
                if not page_reports:
                    st.write(f"No reports found on page {page}")
                    break
                
                for idx, report in enumerate(page_reports):
                    try:
                        metadata = extract_metadata(report['url'])
                        if metadata:
                            metadata['title'] = report['title']
                            all_reports.append(metadata)
                            st.write(f"Processed: {report['title']}")
                    except Exception as e:
                        st.error(f"Error processing {report['url']}: {str(e)}")
                        continue
                    
                    progress = (page - 1 + (idx + 1) / len(page_reports)) / max_pages
                    progress_bar.progress(progress)
                
                sleep(1)
            
            if all_reports:
                df = pd.DataFrame(all_reports)
                
                st.success(f"Found {len(all_reports)} reports")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode('utf-8')
                timestamp = date.today().strftime("%Y%m%d")
                filename = f"pfd_reports_{search_keyword}_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            else:
                st.warning("No reports found")
            
            progress_bar.empty()

if __name__ == "__main__":
    main()
