import streamlit as st
import pandas as pd
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import time

st.set_page_config(
    page_title="UK Judiciary PFD Reports Scraper",
    page_icon="📚",
    layout="wide"
)

def clean_text(text):
    """Clean extracted text by removing extra whitespace and newlines"""
    if text:
        return ' '.join(text.strip().split())
    return ""

def get_reports_by_category(category=None, max_pages=10):
    """
    Scrape PFD reports and filter by category if specified
    """
    base_url = "https://www.judiciary.uk/"
    reports = []
    
    # Construct search parameters - using simple keyword instead of exact phrase
    params = {
        's': category if category and category != "All Categories" else "",
        'post_type': 'pfd',
        'order': 'relevance'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5'
    }
    
    st.write(f"Searching for reports{' with keyword: ' + category if category and category != 'All Categories' else ''}")
    
    for page in range(1, max_pages + 1):
        try:
            # Add page number to parameters if not first page
            if page > 1:
                params['page'] = page
            
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Check if we found the results header
            results_header = soup.find('div', class_='search__header')
            if results_header:
                header_text = results_header.text.strip()
                if page == 1:  # Only show the header for the first page
                    st.write(f"Found results: {header_text}")
            
            # Find the listings container
            listings = soup.find('div', class_='archive__listings')
            if not listings:
                break
            
            # Find all posts in the listings
            posts = listings.find_all(['article', 'div'], class_=['post', 'entry'])
            if not posts:
                posts = listings.find_all('article')  # Try alternative selector
            
            if not posts:
                st.write(f"No more results found on page {page}")
                break
            
            st.write(f"Processing page {page} - Found {len(posts)} posts")
            
            for post in posts:
                try:
                    # Get title and link
                    title_elem = post.find('h2', class_='entry-title')
                    if not title_elem or not title_elem.find('a'):
                        continue
                        
                    link = title_elem.find('a')
                    title = clean_text(link.text)
                    url = link['href']
                    
                    # Get metadata
                    metadata = post.find('p')
                    if not metadata:
                        continue
                        
                    metadata_text = clean_text(metadata.text)
                    
                    # Extract report info
                    patterns = {
                        'Date': r'Date of report:?\s*(\d{2}/\d{2}/\d{4})',
                        'Reference': r'Ref:?\s*([\w-]+)',
                        'Deceased_Name': r'Deceased name:?\s*([^,\n]+)',
                        'Coroner_Name': r'Coroner name:?\s*([^,\n]+)',
                        'Coroner_Area': r'Coroner Area:?\s*([^,\n]+)',
                        'Category': r'Category:?\s*([^|\n]+)',
                        'Trust': r'This report is being sent to:\s*([^|\n]+)'
                    }
                    
                    report = {
                        'Title': title,
                        'URL': url
                    }
                    
                    for key, pattern in patterns.items():
                        match = re.search(pattern, metadata_text)
                        report[key] = clean_text(match.group(1)) if match else ""
                    
                    reports.append(report)
                    st.write(f"Found report: {title}")
                    
                except Exception as e:
                    st.error(f"Error processing post: {str(e)}")
                    continue
            
            # Check if there are more pages
            next_page = soup.find('a', class_='next')
            if not next_page:
                st.write("Reached last page")
                break
                
            # Small delay between pages
            time.sleep(1)
            
        except Exception as e:
            st.error(f"Error processing page {page}: {str(e)}")
            break
    
    return reports

def main():
    st.title("UK Judiciary PFD Reports Scraper")
    
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Enter keywords to search for reports.
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_keyword = st.text_input("Enter search keywords:", "child death")
    with col2:
        max_pages = st.number_input("Maximum pages to search:", min_value=1, max_value=50, value=10)
    
    if st.button("Search Reports"):
        with st.spinner("Searching for reports..."):
            reports = get_reports_by_category(search_keyword, max_pages)
            
            if reports:
                df = pd.DataFrame(reports)
                
                # Reorder columns
                columns = ['Title', 'Date', 'Reference', 'Deceased_Name', 'Coroner_Name', 
                          'Coroner_Area', 'Category', 'Trust', 'URL']
                df = df[columns]
                
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
                csv = df.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pfd_reports_{search_keyword.replace(' ', '_')}_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            else:
                st.warning(f"No reports found for search: {search_keyword}")

if __name__ == "__main__":
    main()
