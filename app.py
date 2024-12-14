import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import time

st.set_page_config(
    page_title="Coroner Name Extractor",
    page_icon="📚",
    layout="wide"
)

def clean_text(text):
    if text:
        return ' '.join(text.strip().split())
    return ""

def get_coroner_name(url):
    """Extract the coroner's name from an individual report page."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Get the main content
        content = soup.find('div', class_='entry-content')
        if not content:
            return None
        
        text = content.get_text()

        # Use regex to extract the coroner's name
        pattern = r'Coroner name:?\s*([^,\n]+)'
        match = re.search(pattern, text)
        return clean_text(match.group(1)) if match else "Not Found"
    
    except Exception as e:
        st.error(f"Error fetching coroner's name: {str(e)}")
        return None

def get_reports_by_keyword(keyword, max_pages=10):
    base_url = "https://www.judiciary.uk/"
    reports = []
    
    params = {
        's': keyword,
        'post_type': 'pfd',
        'order': 'relevance'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    st.write(f"Searching for reports with keyword: {keyword}")
    
    for page in range(1, max_pages + 1):
        try:
            if page > 1:
                params['paged'] = page
            
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find all report entries
            entries = soup.select('.search-results article, .archive__listings article')
            
            st.write(f"Processing page {page} - Found {len(entries)} reports")
            
            for entry in entries:
                try:
                    # Get basic information
                    title_elem = entry.select_one('.entry-title a')
                    if not title_elem:
                        continue
                        
                    title = clean_text(title_elem.text)
                    url = title_elem['href']
                    
                    # Extract the coroner's name
                    st.write(f"Fetching coroner's name for: {title}")
                    coroner_name = get_coroner_name(url)
                    
                    reports.append({
                        'Title': title,
                        'Coroner_Name': coroner_name,
                        'URL': url
                    })
                    
                except Exception as e:
                    st.error(f"Error processing entry: {str(e)}")
                    continue
            
            time.sleep(1)
            
        except Exception as e:
            st.error(f"Error processing page {page}: {str(e)}")
            break
    
    return reports

def main():
    st.title("Coroner Name Extractor")
    
    st.markdown("""This app extracts coroner names from UK Judiciary PFD reports.""")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_keyword = st.text_input("Enter search keywords:", "child")
    with col2:
        max_pages = st.number_input("Maximum pages to search:", min_value=1, max_value=50, value=10)
    
    if st.button("Search Reports"):
        with st.spinner("Searching for reports..."):
            reports = get_reports_by_keyword(search_keyword, max_pages)
            
            if reports:
                # Display results
                st.success(f"Found {len(reports)} reports")
                st.write("### Extracted Coroner Names")
                
                for report in reports:
                    st.write(f"- **{report['Title']}**: {report['Coroner_Name']} ([Link]({report['URL']}))")
            else:
                st.warning(f"No reports found for search: {search_keyword}")

if __name__ == "__main__":
    main()
