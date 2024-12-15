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
        
        # Save HTML for debugging
        html_content = response.text
        if debug:
            st.write("First 1000 characters of HTML:")
            st.code(html_content[:1000], language='html')
        
        return BeautifulSoup(html_content, "html.parser")
    except Exception as e:
        st.error(f"Error fetching URL {url}: {str(e)}")
        return None

def find_reports(soup):
    """Find all report entries in the page"""
    reports = []
    
    # Find the search results container
    results_container = soup.find('div', {'id': 'listing'})
    if not results_container:
        st.write("No results container found")
        return reports
        
    # Skip past the search form
    search_form = results_container.find('form', {'class': 'search__form'})
    if search_form:
        # Get all elements after the search form
        current = search_form.find_next_sibling()
        while current:
            # Check if this is a report entry
            title_elem = current.find('h2', class_='entry-title')
            if title_elem:
                link = title_elem.find('a')
                if link and 'href' in link.attrs:
                    reports.append({
                        'url': link['href'],
                        'title': link.text.strip()
                    })
                    st.write(f"Found report: {link.text.strip()}")
            current = current.find_next_sibling()
    
    st.write(f"Found {len(reports)} reports")
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
        
        # Get all paragraphs
        paragraphs = content.find_all('p')
        full_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        # Extract information using patterns
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
                # Construct URL
                url = f"https://www.judiciary.uk/?s={search_keyword}&post_type=pfd"
                if page > 1:
                    url += f"&paged={page}"
                    
                st.write(f"Searching page {page}: {url}")
                
                # Get page content
                soup = get_url(url, debug=True)
                if not soup:
                    continue
                
                # Find reports on page
                page_reports = find_reports(soup)
                if not page_reports:
                    st.write(f"No reports found on page {page}")
                    break
                
                # Process each report
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
                    
                    # Update progress
                    progress = (page - 1 + (idx + 1) / len(page_reports)) / max_pages
                    progress_bar.progress(progress)
                
                sleep(1)  # Be nice to the server
            
            # Create DataFrame and save results
            if all_reports:
                df = pd.DataFrame(all_reports)
                
                st.success(f"Found {len(all_reports)} reports")
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
            
            progress_bar.empty()

if __name__ == "__main__":
    main()
