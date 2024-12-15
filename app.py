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
    html_content = response.text
    
    # Debug: Save full HTML
    st.session_state['last_html'] = html_content
    
    return BeautifulSoup(html_content, "html.parser")

def get_report_urls(keyword, page=1):
    """Get all report URLs from a search page"""
    base_url = "https://www.judiciary.uk/"
    params = {
        's': keyword,
        'post_type': 'pfd',
        'order': 'relevance'
    }
    if page > 1:
        params['paged'] = page
        
    url = f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    
    st.write(f"Fetching from: {url}")
    soup = get_url(url)
    
    # Find results header
    results_header = soup.find('div', class_='search__header')
    if results_header:
        header_text = results_header.text.strip()
        st.write(f"Found: {header_text}")
        
        # Extract number of results
        match = re.search(r'found (\d+) results', header_text)
        if match:
            total_results = int(match.group(1))
            st.write(f"Total results to process: {total_results}")
    
    # Find all posts in the results section
    articles = []
    posts = soup.find_all('div', class_='post')
    
    if not posts:
        # Try alternative selectors
        posts = soup.find_all(['article', 'div'], class_=['post', 'search-result'])
    
    st.write(f"Found {len(posts)} posts on page {page}")
    
    # Debug: Show HTML structure of a post if any found
    if posts:
        st.write("Example post structure:")
        st.code(str(posts[0])[:500], language='html')
    else:
        st.write("No posts found. Showing full page HTML for debugging:")
        st.code(str(soup)[:1000], language='html')
    
    # Process each post
    for post in posts:
        # Try different ways to find the link
        link = (post.find('h2', class_='entry-title').find('a') if post.find('h2', class_='entry-title') else None) or \
               post.find('a', class_='view-more') or \
               post.find('a')
               
        if link and 'href' in link.attrs:
            articles.append({
                'url': link['href'],
                'title': link.text.strip()
            })
    
    st.write(f"Successfully extracted {len(articles)} article URLs")
    return articles

def extract_report_info(url):
    """Extract information from a single report"""
    try:
        soup = get_url(url)
        content = soup.find('div', class_='entry-content')
        
        if not content:
            st.write(f"No content found for {url}")
            return None
        
        info = {
            'url': url,
            'title': '',
            'date_of_report': '',
            'ref': '',
            'deceased_name': '',
            'coroner_name': '',
            'coroner_area': '',
            'category': '',
            'this_report_is_being_sent_to': ''
        }
        
        # Get title
        title = soup.find('h1', class_='entry-title')
        if title:
            info['title'] = title.text.strip()
        
        # Get metadata from paragraphs
        paragraphs = content.find_all('p')
        
        for p in paragraphs:
            text = p.text.strip()
            if not text:
                continue
                
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
                match = re.search(pattern, text)
                if match:
                    info[key] = match.group(1).strip()
        
        return info
        
    except Exception as e:
        st.error(f"Error processing {url}: {str(e)}")
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
        with st.spinner("Gathering reports..."):
            all_reports = []
            progress_bar = st.progress(0)
            
            for page in range(1, max_pages + 1):
                st.write(f"Processing page {page}")
                
                articles = get_report_urls(search_keyword, page)
                
                if articles:
                    for idx, article in enumerate(articles):
                        try:
                            st.write(f"Processing article: {article['title']}")
                            report_info = extract_report_info(article['url'])
                            if report_info:
                                all_reports.append(report_info)
                                st.write(f"Successfully processed: {article['title']}")
                        except Exception as e:
                            st.error(f"Error processing {article['url']}: {str(e)}")
                            continue
                        
                        # Update progress
                        progress = (page - 1 + (idx + 1) / len(articles)) / max_pages
                        progress_bar.progress(progress)
                else:
                    st.write(f"No more articles found on page {page}")
                    break
                
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
