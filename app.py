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
            st.write("Response headers:", response.headers)
        
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        st.error(f"Error fetching URL {url}: {str(e)}")
        return None

def find_reports(soup):
    """Find all report entries in the page"""
    reports = []
    
    # First try to find the main content area
    content_area = soup.find('main', {'id': 'main-content'})
    if not content_area:
        content_area = soup
        
    # Try different ways to find report entries
    entries = []
    
    # Method 1: Look for articles directly
    entries = content_area.find_all('article')
    
    # Method 2: Look for entries in search results
    if not entries:
        search_results = content_area.find('div', class_='search-results')
        if search_results:
            entries = search_results.find_all(['article', 'div'], class_=['post', 'search-result'])
    
    # Method 3: Look in archive listings
    if not entries:
        archive_listings = content_area.find('div', class_='archive__listings')
        if archive_listings:
            # Skip the search form
            search_form = archive_listings.find('form', class_='search__form')
            if search_form:
                current = search_form.find_next_sibling()
                while current:
                    if current.name == 'article' or (current.name == 'div' and 'post' in current.get('class', [])):
                        entries.append(current)
                    current = current.find_next_sibling()
    
    # Debug output
    st.write(f"Found {len(entries)} entries")
    
    # Process each entry
    for entry in entries:
        # Try to find title and link
        link = None
        title_elem = entry.find(['h2', 'h3', 'h4', 'h5'], class_='entry-title')
        if title_elem:
            link = title_elem.find('a')
        
        if not link:
            link = entry.find('a')
            
        if link and 'href' in link.attrs:
            reports.append({
                'url': link['href'],
                'title': link.text.strip()
            })
            st.write(f"Found report: {link.text.strip()}")
    
    return reports

def extract_metadata(content):
    """Extract metadata from report content"""
    info = {}
    
    # Find all paragraphs with potential metadata
    paragraphs = content.find_all('p')
    
    # Define patterns to match
    patterns = {
        'date_of_report': r'Date of report:?\s*([^\n]+)',
        'ref': r'Ref:?\s*([\w-]+)',
        'deceased_name': r'Deceased name:?\s*([^\n]+)',
        'coroner_name': r'Coroner name:?\s*([^\n]+)',
        'coroner_area': r'Coroner Area:?\s*([^\n]+)',
        'category': r'Category:?\s*([^|\n]+)',
        'this_report_is_being_sent_to': r'This report is being sent to:?\s*([^\n]+)'
    }
    
    # Look for metadata in each paragraph
    for p in paragraphs:
        text = p.get_text(strip=True)
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
        if not search_keyword:
            st.warning("Please enter a search keyword")
            return
            
        reports = []
        with st.spinner("Searching for reports..."):
            progress_bar = st.progress(0)
            
            for page in range(1, max_pages + 1):
                # Construct search URL
                url = f"https://www.judiciary.uk/?s={search_keyword}&post_type=pfd&paged={page}"
                st.write(f"Searching page {page}: {url}")
                
                # Get page content
                soup = get_url(url, debug=True)
                if not soup:
                    continue
                
                # Find report entries
                page_reports = find_reports(soup)
                if not page_reports:
                    st.write(f"No reports found on page {page}")
                    break
                
                # Process each report
                for idx, report in enumerate(page_reports):
                    try:
                        # Get report content
                        report_soup = get_url(report['url'])
                        if not report_soup:
                            continue
                            
                        # Find report content
                        content = report_soup.find('div', class_='entry-content')
                        if content:
                            # Extract metadata
                            metadata = extract_metadata(content)
                            metadata.update({
                                'title': report['title'],
                                'url': report['url']
                            })
                            reports.append(metadata)
                            st.write(f"Processed: {report['title']}")
                    except Exception as e:
                        st.error(f"Error processing report {report['url']}: {str(e)}")
                        continue
                    
                    # Update progress
                    progress = (page - 1 + (idx + 1) / len(page_reports)) / max_pages
                    progress_bar.progress(progress)
                
                sleep(1)  # Be nice to the server
            
            progress_bar.empty()
            
            # Create DataFrame and save results
            if reports:
                df = pd.DataFrame(reports)
                
                st.success(f"Found {len(reports)} reports")
                st.dataframe(df)
                
                # Download button
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

if __name__ == "__main__":
    main()
