import streamlit as st
import pandas as pd
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import time
import urllib3
from dateutil import parser

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Analysis", layout="wide")

def clean_text(text):
    """Clean text by removing extra whitespace, newlines and special characters"""
    if text:
        # Remove special characters and normalize whitespace
        text = re.sub(r'[â€¦]', '...', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""

def parse_date(date_str):
    """Parse date string into datetime object"""
    try:
        return parser.parse(date_str)
    except:
        return None

def extract_metadata(text):
    """Extract and clean metadata fields from text"""
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
            value = clean_text(match.group(1))
            if key == 'date_of_report':
                info[key] = parse_date(value)
            else:
                info[key] = value
    return info

def extract_section_content(text):
    """Extract and clean content from numbered sections in the report"""
    sections = {
        'background': '',
        'investigation_details': '',
        'circumstances': '',
        'coroner_concerns': '',
        'recommended_actions': '',
        'response_deadline': '',
        'publication_info': ''
    }
    
    patterns = {
        'background': r'1\s*CORONER\s*(.*?)(?=2\s*CORONER|$)',
        'investigation_details': r'3\s*INVESTIGATION\s*(.*?)(?=4\s*CIRCUMSTANCES|$)',
        'circumstances': r'4\s*CIRCUMSTANCES OF THE DEATH\s*(.*?)(?=5\s*CORONER|$)',
        'coroner_concerns': r'5\s*CORONER\'S CONCERNS\s*(.*?)(?=6\s*ACTION|$)',
        'recommended_actions': r'6\s*ACTION SHOULD BE TAKEN\s*(.*?)(?=7\s*YOUR RESPONSE|$)',
        'response_deadline': r'7\s*YOUR RESPONSE\s*(.*?)(?=8\s*COPIES|$)',
        'publication_info': r'8\s*COPIES and PUBLICATION\s*(.*?)(?=9|$)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = clean_text(match.group(1))
            
    return sections

def get_report_content(url):
    """Get and parse detailed content from individual report page"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content = soup.find('div', class_='entry-content')
        if not content:
            return None
            
        return content.get_text()
    except Exception as e:
        st.error(f"Error getting report content: {str(e)}")
        return None

def scrape_pfd_reports(keyword=None, start_date=None, end_date=None, max_pages=50):
    """Scrape reports with filtering options"""
    all_reports = []
    current_page = 1
    base_url = "https://www.judiciary.uk/"
    
    while current_page <= max_pages:
        url = f"{base_url}{'page/' + str(current_page) + '/' if current_page > 1 else ''}?s={keyword if keyword else ''}&post_type=pfd"
        
        with st.spinner(f"Scraping page {current_page}..."):
            try:
                reports = scrape_page(url)
                if not reports:
                    break
                    
                # Filter by date if specified
                if start_date or end_date:
                    filtered_reports = []
                    for report in reports:
                        report_date = report.get('date_of_report')
                        if report_date:
                            if start_date and report_date < start_date:
                                continue
                            if end_date and report_date > end_date:
                                continue
                        filtered_reports.append(report)
                    reports = filtered_reports
                
                all_reports.extend(reports)
                st.write(f"Found {len(reports)} reports on page {current_page}")
                
                current_page += 1
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                st.error(f"Error processing page {current_page}: {str(e)}")
                break
    
    return all_reports

def analyze_reports(reports_df):
    """Generate analysis of the reports"""
    st.subheader("Reports Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reports", len(reports_df))
        
    with col2:
        if 'date_of_report' in reports_df.columns:
            date_range = f"{reports_df['date_of_report'].min():%Y-%m-%d} to {reports_df['date_of_report'].max():%Y-%m-%d}"
            st.metric("Date Range", date_range)
            
    with col3:
        if 'category' in reports_df.columns:
            categories = reports_df['category'].str.split('|').explode().str.strip().unique()
            st.metric("Unique Categories", len(categories))
    
    # Category breakdown
    if 'category' in reports_df.columns:
        st.subheader("Category Distribution")
        category_counts = (reports_df['category'].str.split('|')
                         .explode()
                         .str.strip()
                         .value_counts())
        st.bar_chart(category_counts)
    
    # Timeline analysis
    if 'date_of_report' in reports_df.columns:
        st.subheader("Reports Timeline")
        timeline = reports_df.set_index('date_of_report').resample('M').size()
        st.line_chart(timeline)

def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
    st.markdown("""
    This app scrapes and analyzes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Filter by keywords and date range to find relevant reports.
    """)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_keyword = st.text_input("Search keywords:", "")
    with col2:
        start_date = st.date_input("Start date:")
    with col3:
        end_date = st.date_input("End date:")
    
    if st.button("Search and Analyze Reports"):
        reports = scrape_pfd_reports(
            keyword=search_keyword,
            start_date=start_date,
            end_date=end_date
        )
        
        if reports:
            df = pd.DataFrame(reports)
            
            # Clean and format data
            if 'date_of_report' in df.columns:
                df['date_of_report'] = pd.to_datetime(df['date_of_report'])
            
            st.success(f"Found {len(reports)} reports")
            
            # Display and analyze results
            analyze_reports(df)
            
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
            export_format = st.selectbox(
                "Export format:",
                ["CSV", "Excel"]
            )
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pfd_reports_{search_keyword}_{timestamp}"
            
            if export_format == "CSV":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Reports",
                    csv,
                    f"{filename}.csv",
                    "text/csv"
                )
            else:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False)
                st.download_button(
                    "📥 Download Reports",
                    excel_buffer.getvalue(),
                    f"{filename}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.warning("No reports found")

if __name__ == "__main__":
    main()
