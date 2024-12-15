import streamlit as st
import pandas as pd
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import time

st.set_page_config(page_title="UK Judiciary PFD Reports Scraper", layout="wide")

def clean_text(text):
    if text:
        return ' '.join(text.strip().split())
    return ""

def extract_metadata(text):
    patterns = {
        'date_of_report': r'Date of report:?\s*([^\n]+)',
        'ref': r'Ref:?\s*([\w-]+)',
        'deceased_name': r'Deceased name:?\s*([^\n]+)',
        'coroner_name': r'Coroner(?:s)? name:?\s*([^\n]+)',
        'coroner_area': r'Coroner(?:s)? Area:?\s*([^\n]+)',
        'category': r'Category:?\s*([^|\n]+)',
        'sent_to': r'This report is being sent to:?\s*([^\n]+)'
    }
    
    info = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            info[key] = clean_text(match.group(1))
    return info

def scrape_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, verify=False)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the search results list
    results_list = soup.find('ul', class_='search__list')
    if not results_list:
        return []
        
    reports = []
    # Find all card elements within the list
    cards = results_list.find_all('div', class_='card')
    
    for card in cards:
        try:
            # Get title and URL
            title_elem = card.find('h3', class_='card__title').find('a')
            if not title_elem:
                continue
                
            title = clean_text(title_elem.text)
            url = title_elem['href']
            
            # Get description containing metadata
            desc = card.find('p', class_='card__description')
            if not desc:
                continue
                
            # Get metadata
            metadata = extract_metadata(desc.text)
            
            # Get categories from pills
            categories = []
            pills = card.find_all('a', href=re.compile(r'/pfd-types/'))
            for pill in pills:
                categories.append(clean_text(pill.text))
            
            report = {
                'Title': title,
                'URL': url,
                'Categories': ' | '.join(categories),
                **metadata  # Add all extracted metadata
            }
            
            reports.append(report)
            
        except Exception as e:
            st.error(f"Error processing card: {str(e)}")
            continue
            
    return reports

def get_total_pages(soup):
    """Get total number of pages from pagination"""
    pagination = soup.find('div', class_='nav-links')
    if pagination:
        # Find the last page number
        last_page = pagination.find_all('a', class_='page-numbers')[-2].text
        return int(last_page)
    return 1

def scrape_pfd_reports(keyword, max_pages=10):
    all_reports = []
    current_page = 1
    
    base_url = "https://www.judiciary.uk/"
    
    while current_page <= max_pages:
        if current_page == 1:
            url = f"{base_url}?s={keyword}&post_type=pfd"
        else:
            url = f"{base_url}page/{current_page}/?s={keyword}&post_type=pfd"
            
        st.write(f"Scraping page {current_page}: {url}")
        
        try:
            reports = scrape_page(url)
            if not reports:
                break
                
            all_reports.extend(reports)
            st.write(f"Found {len(reports)} reports on page {current_page}")
            
            current_page += 1
            time.sleep(1)  # Be nice to the server
            
        except Exception as e:
            st.error(f"Error processing page {current_page}: {str(e)}")
            break
    
    return all_reports

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
            reports = scrape_pfd_reports(search_keyword, max_pages)
            
            if reports:
                df = pd.DataFrame(reports)
                
                st.success(f"Found {len(reports)} reports")
                
                # Display results
                st.dataframe(
                    df,
                    column_config={
                        "URL": st.column_config.LinkColumn("Report Link")
                    },
                    hide_index=True
                )
                
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
