import streamlit as st
import pandas as pd
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import time
import urllib3
from dateutil import parser
import io

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Analysis", layout="wide")

def clean_text(text):
    """Clean text by removing extra whitespace and newlines"""
    if text:
        text = re.sub(r'[â€™]', "'", text)
        text = re.sub(r'[â€¦]', "...", text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""

def extract_metadata_and_content(text):
    """Extract metadata fields and full content"""
    metadata = {
        'date_of_report': '',
        'ref': '',
        'deceased_name': '',
        'coroner_name': '',
        'coroner_area': '',
        'category': '',
        'sent_to': '',
        'content': ''
    }
    
    # Metadata patterns
    patterns = {
        'date_of_report': r'Date of report:?\s*([^\n]+)',
        'ref': r'Ref:?\s*([\w-]+)',
        'deceased_name': r'Deceased name:?\s*([^\n]+)',
        'coroner_name': r'Coroner name:?\s*([^\n]+)',
        'coroner_area': r'Coroner Area:?\s*([^\n]+)',
        'category': r'Category:?\s*([^|\n]+(?:\s*\|\s*[^|\n]+)*)',
        'sent_to': r'This report is being sent to:?\s*([^R\n]+)'
    }
    
    # Extract metadata fields
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = clean_text(match.group(1))
            metadata[key] = value
    
    # Split categories
    if metadata['category']:
        categories = [cat.strip() for cat in metadata['category'].split('|')]
        metadata['primary_category'] = categories[0] if categories else ''
        metadata['additional_categories'] = ' | '.join(categories[1:]) if len(categories) > 1 else ''
    
    # Extract full content starting from "REGULATION 28 REPORT"
    content_match = re.search(r'(REGULATION 28 REPORT.*?)$', text, re.DOTALL | re.IGNORECASE)
    if content_match:
        metadata['content'] = clean_text(content_match.group(1))
    
    return metadata

def get_report_content(url):
    """Get detailed content from individual report page"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try finding content in different possible locations
        content = soup.find('div', class_='flow')
        if not content:
            content = soup.find('article', class_='single__post')
            
        if content:
            # Get all text content
            text_content = content.get_text(separator='\n')
            text_content = clean_text(text_content)
            return text_content
            
        st.warning(f"No content found for: {url}")
        return None
            
    except Exception as e:
        st.error(f"Error getting report content: {str(e)}")
        return None

def scrape_page(url):
    """Scrape a single page of search results"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results_list = soup.find('ul', class_='search__list')
        if not results_list:
            return []
            
        reports = []
        cards = results_list.find_all('div', class_='card')
        
        for card in cards:
            try:
                # Get title and URL
                title_elem = card.find('h3', class_='card__title').find('a')
                if not title_elem:
                    continue
                    
                title = clean_text(title_elem.text)
                url = title_elem['href']
                
                # Get metadata and content
                st.write(f"Processing report: {title}")
                report_content = get_report_content(url)
                
                if report_content:
                    metadata = extract_metadata_and_content(report_content)
                    st.write(f"Successfully extracted content")
                else:
                    metadata = {}
                    st.warning(f"No content found for: {title}")
                
                # Construct report with separate metadata fields and combined content
                report = {
                    'Title': title,
                    'URL': url,
                    'Date_of_Report': metadata.get('date_of_report', ''),
                    'Reference': metadata.get('ref', ''),
                    'Deceased_Name': metadata.get('deceased_name', ''),
                    'Coroner_Name': metadata.get('coroner_name', ''),
                    'Coroner_Area': metadata.get('coroner_area', ''),
                    'Primary_Category': metadata.get('primary_category', ''),
                    'Additional_Categories': metadata.get('additional_categories', ''),
                    'Sent_To': metadata.get('sent_to', ''),
                    'Content': metadata.get('content', '')
                }
                
                reports.append(report)
                st.write(f"Completed processing: {title}")
                
            except Exception as e:
                st.error(f"Error processing card: {str(e)}")
                continue
                
        return reports
        
    except Exception as e:
        st.error(f"Error fetching page: {str(e)}")
        return []

def get_total_pages(url):
    """Get total number of pages"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        pagination = soup.find('ul', class_='pagination')
        if pagination:
            page_numbers = pagination.find_all('a', class_='page-numbers')
            if page_numbers:
                numbers = [int(p.text) for p in page_numbers if p.text.isdigit()]
                if numbers:
                    return max(numbers)
        return 1
    except Exception as e:
        st.error(f"Error getting total pages: {str(e)}")
        return 1

def scrape_pfd_reports(keyword=None, start_date=None, end_date=None):
    """Scrape reports with filtering options"""
    all_reports = []
    current_page = 1
    base_url = "https://www.judiciary.uk/"
    
    initial_url = f"{base_url}?s={keyword if keyword else ''}&post_type=pfd"
    total_pages = get_total_pages(initial_url)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while current_page <= total_pages:
        url = f"{base_url}{'page/' + str(current_page) + '/' if current_page > 1 else ''}?s={keyword if keyword else ''}&post_type=pfd"
        
        status_text.text(f"Scraping page {current_page} of {total_pages}...")
        progress_bar.progress(current_page / total_pages)
        
        try:
            reports = scrape_page(url)
            if reports:
                if start_date or end_date:
                    filtered_reports = []
                    for report in reports:
                        report_date = report.get('Date_of_Report')
                        try:
                            if report_date:
                                # Try to parse the date, assuming UK format
                                parsed_date = parser.parse(report_date, dayfirst=True)
                                if start_date and parsed_date.date() < start_date:
                                    continue
                                if end_date and parsed_date.date() > end_date:
                                    continue
                            filtered_reports.append(report)
                        except (ValueError, TypeError):
                            # If date parsing fails, include the report anyway
                            filtered_reports.append(report)
                    reports = filtered_reports
                
                all_reports.extend(reports)
                st.write(f"Found {len(reports)} reports on page {current_page}")
            else:
                break
                
            current_page += 1
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            st.error(f"Error processing page {current_page}: {str(e)}")
            continue
    
    progress_bar.progress(1.0)
    status_text.text(f"Completed! Total reports found: {len(all_reports)}")
    return all_reports

def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
    st.markdown("""
    This app scrapes and analyses Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Filter by keywords and date range to find relevant reports.
    """)
    
    # Use form for inputs
    with st.form("search_form"):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_keyword = st.text_input("Search keywords:", "")
        with col2:
            start_date = st.date_input("Start date:", format="DD/MM/YYYY")
        with col3:
            end_date = st.date_input("End date:", format="DD/MM/YYYY")
        
        submitted = st.form_submit_button("Search and Analyse Reports")
    
    if submitted:
        reports = []  # Initialize reports list
        with st.spinner("Searching for reports..."):
            scraped_reports = scrape_pfd_reports(
                keyword=search_keyword,
                start_date=start_date,
                end_date=end_date
            )
            if scraped_reports:
                reports.extend(scraped_reports)
        
        if reports:
            df = pd.DataFrame(reports)
            
            st.success(f"Found {len(reports):,} reports")
            
            # Show detailed data
            st.subheader("Detailed Reports Data")
            st.dataframe(
                df,
                column_config={
                    "URL": st.column_config.LinkColumn("Report Link")
                },
                hide_index=True
            )
            
            # Export options
            export_format = st.selectbox("Export format:", ["CSV", "Excel"], key="export_format")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pfd_reports_{search_keyword}_{timestamp}"
            
            if export_format == "CSV":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Reports",
                    csv,
                    f"{filename}.csv",
                    "text/csv",
                    key="download_csv"
                )
            else:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                excel_data = excel_buffer.getvalue()
                st.download_button(
                    "📥 Download Reports",
                    excel_data,
                    f"{filename}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
        else:
            if search_keyword:
                st.warning("No reports found matching your search criteria")
            else:
                st.info("Please enter search keywords to find reports")

if __name__ == "__main__":
    main()
