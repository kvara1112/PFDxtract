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
import locale

# Set locale to British English
try:
    locale.setlocale(locale.LC_ALL, 'en_GB.UTF-8')
except:
    pass  # Fallback if locale not available

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Analysis", layout="wide")

def format_uk_date(date):
    """Format date in UK style"""
    if pd.isna(date):
        return ""
    try:
        return date.strftime("%d/%m/%Y")
    except:
        return str(date)

def clean_text(text):
    """Clean text by removing extra whitespace, newlines and special characters"""
    if text:
        # Remove special characters and normalize whitespace
        text = re.sub(r'[â€¦]', '...', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""

def parse_date(date_str):
    """Parse date string into datetime object, handling UK formats"""
    try:
        # Try UK format first (dd/mm/yyyy)
        return parser.parse(date_str, dayfirst=True)
    except:
        return None

def get_total_pages(url):
    """Detect total number of pages available"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find pagination element
        pagination = soup.find('ul', class_='pagination')
        if pagination:
            # Find all page numbers
            page_numbers = pagination.find_all('a', class_='page-numbers')
            if page_numbers:
                # Get last page number
                last_page = max([int(p.text) for p in page_numbers if p.text.isdigit()])
                return last_page
        
        return 1  # Return 1 if no pagination found
    except Exception as e:
        st.error(f"Error detecting pagination: {str(e)}")
        return 1

[previous functions remain the same...]

def scrape_pfd_reports(keyword=None, start_date=None, end_date=None):
    """Scrape reports with filtering options"""
    all_reports = []
    current_page = 1
    base_url = "https://www.judiciary.uk/"
    
    # Get total pages
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
    
    progress_bar.progress(1.0)
    status_text.text(f"Completed! Total reports found: {len(all_reports)}")
    return all_reports

def analyze_reports(reports_df):
    """Generate analysis of the reports"""
    st.subheader("Reports Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reports", f"{len(reports_df):,}")
        
    with col2:
        if 'date_of_report' in reports_df.columns:
            date_range = f"{format_uk_date(reports_df['date_of_report'].min())} to {format_uk_date(reports_df['date_of_report'].max())}"
            st.metric("Date Range", date_range)
            
    with col3:
        if 'Categories' in reports_df.columns:
            categories = reports_df['Categories'].str.split('|').explode().str.strip().unique()
            st.metric("Unique Categories", len(categories))
    
    # Category breakdown
    if 'Categories' in reports_df.columns:
        st.subheader("Category Distribution")
        category_counts = (reports_df['Categories'].str.split('|')
                         .explode()
                         .str.strip()
                         .value_counts())
        st.bar_chart(category_counts)
    
    # Timeline analysis
    if 'date_of_report' in reports_df.columns:
        st.subheader("Reports Timeline")
        timeline = reports_df.set_index('date_of_report').resample('M').size()
        timeline.index = timeline.index.strftime('%B %Y')  # UK month format
        st.line_chart(timeline)

def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
    st.markdown("""
    This app scrapes and analyses Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Filter by keywords and date range to find relevant reports.
    """)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_keyword = st.text_input("Search keywords:", "")
    with col2:
        start_date = st.date_input("Start date:", format="DD/MM/YYYY")
    with col3:
        end_date = st.date_input("End date:", format="DD/MM/YYYY")
    
    if st.button("Search and Analyse Reports"):
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
            
            st.success(f"Found {len(reports):,} reports")
            
            # Display and analyze results
            analyze_reports(df)
            
            # Show detailed data
            st.subheader("Detailed Reports Data")
            
            # Format dates in UK style for display
            display_df = df.copy()
            if 'date_of_report' in display_df.columns:
                display_df['date_of_report'] = display_df['date_of_report'].apply(format_uk_date)
            
            st.dataframe(
                display_df,
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
