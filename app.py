import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from time import sleep
import pandas as pd
from datetime import datetime
import urllib3
import io
import zipfile

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Scraper", layout="wide")

def get_url(url):
    """Get URL content with error handling"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, verify=False, headers=headers)
    return BeautifulSoup(response.content, "html.parser")

def clean_text(text):
    """Clean extracted text"""
    if text:
        return ' '.join(text.strip().split())
    return ""

def scrape_pfd_reports(keyword=None, max_pages=10, include_pdfs=False):
    """Main scraping function"""
    reports = []
    pdfs = []
    base_url = "https://www.judiciary.uk/"
    
    params = {
        's': keyword if keyword else '',
        'post_type': 'pfd',
        'order': 'relevance'
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for page in range(1, max_pages + 1):
            try:
                if page > 1:
                    params['paged'] = page
                    
                soup = get_url(base_url + "?" + "&".join(f"{k}={v}" for k, v in params.items()))
                
                # Find all report entries
                entries = soup.select('.archive__listings article, .search-results article')
                if not entries:
                    break
                    
                status_text.text(f"Processing page {page} - Found {len(entries)} reports")
                
                for idx, entry in enumerate(entries):
                    try:
                        report = {}
                        
                        # Get title and URL
                        title_elem = entry.select_one('.entry-title a')
                        if title_elem:
                            report['Title'] = clean_text(title_elem.text)
                            report['URL'] = title_elem['href']
                            
                            # Get report content
                            report_soup = get_url(report['URL'])
                            content = report_soup.find('div', class_='entry-content')
                            
                            if content:
                                metadata_text = content.get_text()
                                
                                # Extract metadata using patterns
                                patterns = {
                                    'Date': r'Date of report:?\s*(\d{2}/\d{2}/\d{4})',
                                    'Reference': r'Ref:?\s*([\w-]+)',
                                    'Deceased_Name': r'Deceased name:?\s*([^,\n]+)',
                                    'Coroner_Name': r'Coroner name:?\s*([^,\n]+)',
                                    'Coroner_Area': r'Coroner Area:?\s*([^,\n]+)',
                                    'Category': r'Category:?\s*([^|\n]+)',
                                    'Trust': r'This report is being sent to:\s*([^|\n]+)'
                                }
                                
                                for key, pattern in patterns.items():
                                    match = re.search(pattern, metadata_text)
                                    report[key] = clean_text(match.group(1)) if match else ""
                                
                                # Extract sections
                                for i in range(1, 10):
                                    if i < 9:
                                        pattern = fr'{i}\s+(.*?)(?={i+1}\s+|$)'
                                    else:
                                        pattern = fr'{i}\s+(.*?)$'
                                    match = re.search(pattern, metadata_text, re.DOTALL)
                                    report[f'Section_{i}'] = clean_text(match.group(1)) if match else ""
                                
                                # Get PDF URLs if requested
                                if include_pdfs:
                                    pdf_links = content.find_all('a', href=re.compile(r'\.pdf$'))
                                    if pdf_links:
                                        pdf_urls = [link['href'] for link in pdf_links]
                                        pdfs.append({
                                            'Reference': report.get('Reference', ''),
                                            'URLs': pdf_urls
                                        })
                            
                            reports.append(report)
                            
                        # Update progress
                        progress = (page - 1 + (idx + 1) / len(entries)) / max_pages
                        progress_bar.progress(progress)
                        
                    except Exception as e:
                        st.error(f"Error processing entry: {str(e)}")
                        continue
                
                sleep(1)  # Be nice to the server
                
            except Exception as e:
                st.error(f"Error processing page {page}: {str(e)}")
                break
                
    finally:
        progress_bar.empty()
        status_text.empty()
    
    return reports, pdfs

def main():
    st.title("UK Judiciary PFD Reports Scraper")
    
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Enter keywords to search for relevant reports.
    """)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_keyword = st.text_input("Enter search keywords:", "child")
    with col2:
        max_pages = st.number_input("Maximum pages to search:", min_value=1, max_value=50, value=10)
    with col3:
        include_pdfs = st.checkbox("Include PDFs", value=False)
    
    if st.button("Search Reports"):
        with st.spinner("Searching for reports..."):
            reports, pdfs = scrape_pfd_reports(search_keyword, max_pages, include_pdfs)
            
            if reports:
                df = pd.DataFrame(reports)
                
                # Set column order
                base_columns = ['Title', 'Date', 'Reference', 'Deceased_Name', 'Coroner_Name', 
                              'Coroner_Area', 'Category', 'Trust']
                section_columns = [f'Section_{i}' for i in range(1, 10)]
                final_columns = base_columns + section_columns + ['URL']
                
                df = df.reindex(columns=final_columns)
                
                st.success(f"Found {len(reports)} reports")
                
                # Display results
                st.dataframe(
                    df,
                    column_config={
                        "URL": st.column_config.LinkColumn("Report Link")
                    },
                    hide_index=True
                )
                
                # Prepare downloads
                col1, col2 = st.columns(2)
                
                # CSV download
                csv = df.to_csv(index=False).encode('utf-8')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"pfd_reports_{search_keyword.replace(' ', '_')}_{timestamp}.csv"
                
                col1.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=csv_filename,
                    mime="text/csv",
                    key='download-csv'
                )
                
                # PDF download if included
                if include_pdfs and pdfs:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for pdf in pdfs:
                            for idx, url in enumerate(pdf['URLs']):
                                try:
                                    response = requests.get(url, verify=False)
                                    if response.status_code == 200:
                                        filename = f"{pdf['Reference']}_{idx+1}.pdf" if pdf['Reference'] else f"report_{idx+1}.pdf"
                                        zf.writestr(filename, response.content)
                                except Exception as e:
                                    st.error(f"Error downloading PDF {url}: {str(e)}")
                    
                    col2.download_button(
                        label="📥 Download PDFs",
                        data=zip_buffer.getvalue(),
                        file_name=f"pfd_pdfs_{search_keyword.replace(' ', '_')}_{timestamp}.zip",
                        mime="application/zip",
                        key='download-pdfs'
                    )
                
                # Show statistics
                st.write("### Report Statistics")
                st.write(f"- Total reports found: {len(reports)}")
                if df['Category'].notna().any():
                    st.write("#### Categories:")
                    for category, count in df['Category'].value_counts().items():
                        st.write(f"- {category}: {count} reports")
            else:
                st.warning(f"No reports found for search: {search_keyword}")

if __name__ == "__main__":
    main()
