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
    """Format date in UK style, handling None and NaT values"""
    if pd.isna(date) or date is None:
        return "Unknown"
    try:
        return date.strftime("%d/%m/%Y")
    except:
        return "Unknown"

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
                numbers = [int(p.text) for p in page_numbers if p.text.isdigit()]
                if numbers:
                    return max(numbers)
        
        return 1  # Return 1 if no pagination found
    except Exception as e:
        st.error(f"Error detecting pagination: {str(e)}")
        return 1

def extract_structured_content(text):
    """Extract content from a structured PFD report"""
    sections = {
        'metadata': {},
        'legal_powers': '',
        'investigation': '',
        'circumstances': '',
        'concerns': '',
        'action': '',
        'response': '',
        'copies': '',
        'linked_content': {}
    }
    
    # Extract metadata
    metadata_patterns = {
        'date_of_report': r'Date of report:\s*(.*?)(?:\n|$)',
        'ref': r'Ref:\s*(.*?)(?:\n|$)',
        'deceased_name': r'Deceased name:\s*(.*?)(?:\n|$)',
        'coroner_name': r'Coroner name:\s*(.*?)(?:\n|$)',
        'coroner_area': r'Coroner Area:\s*(.*?)(?:\n|$)',
        'category': r'Category:\s*(.*?)(?:\n|$)',
        'sent_to': r'This report is being sent to:\s*(.*?)(?:\n|$)'
    }
    
    for key, pattern in metadata_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            sections['metadata'][key] = clean_text(match.group(1))
    
    # Extract main sections
    section_patterns = {
        'legal_powers': r'2\s*CORONER\'S LEGAL POWERS\s*(.*?)(?=3\s*INVESTIGATION|$)',
        'investigation': r'3\s*INVESTIGATION[^\n]*\s*(.*?)(?=4\s*CIRCUMSTANCES|$)',
        'circumstances': r'4\s*CIRCUMSTANCES OF THE DEATH\s*(.*?)(?=5\s*CORONER|$)',
        'concerns': r'5\s*CORONER\'S CONCERNS\s*(.*?)(?=6\s*ACTION|$)',
        'action': r'6\s*ACTION SHOULD BE TAKEN\s*(.*?)(?=7\s*YOUR RESPONSE|$)',
        'response': r'7\s*YOUR RESPONSE\s*(.*?)(?=8\s*COPIES|$)',
        'copies': r'8\s*COPIES[^\n]*\s*(.*?)(?=9|$)'
    }
    
    for key, pattern in section_patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = clean_text(match.group(1))
    
    # Extract URLs and their content
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    
    return sections, urls

def extract_table_from_html(html_content):
    """Extract tables from HTML content"""
    tables = []
    soup = BeautifulSoup(html_content, 'lxml')
    for table in soup.find_all('table'):
        try:
            df = pd.read_html(str(table))[0]
            tables.append(df)
        except:
            continue
    return tables

def get_linked_content(url):
    """Get content from linked websites"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Get tables if any
        tables = extract_table_from_html(response.text)
        
        return {
            'text': clean_text(text),
            'tables': tables
        }
    except Exception as e:
        return {
            'text': f'Error fetching content: {str(e)}',
            'tables': []
        }

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

def process_pfd_report(url):
    """Process a single PFD report and its linked content"""
    # Get main report content
    report_content = get_report_content(url)
    if not report_content:
        return None
        
    # Extract structured content
    structured_content, urls = extract_structured_content(report_content)
    
    # Process linked content
    linked_content = {}
    for linked_url in urls:
        linked_content[linked_url] = get_linked_content(linked_url)
    
    structured_content['linked_content'] = linked_content
    
    return structured_content

def scrape_page(url):
    """Scrape a single page of search results with enhanced content extraction"""
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
                title_elem = card.find('h3', class_='card__title').find('a')
                if not title_elem:
                    continue
                    
                title = clean_text(title_elem.text)
                url = title_elem['href']
                
                # Get enhanced report content
                report_content = process_pfd_report(url)
                if report_content:
                    report = {
                        'Title': title,
                        'URL': url,
                        **report_content['metadata'],
                        'legal_powers': report_content['legal_powers'],
                        'investigation': report_content['investigation'],
                        'circumstances': report_content['circumstances'],
                        'concerns': report_content['concerns'],
                        'action': report_content['action'],
                        'response': report_content['response'],
                        'copies': report_content['copies'],
                        'linked_content': report_content['linked_content']
                    }
                    reports.append(report)
                    st.write(f"Processed report: {title}")
                
            except Exception as e:
                st.error(f"Error processing card: {str(e)}")
                continue
                
        return reports
        
    except Exception as e:
        st.error(f"Error fetching page: {str(e)}")
        return []

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
            # Filter out NaN dates for min/max calculation
            valid_dates = reports_df['date_of_report'].dropna()
            if len(valid_dates) > 0:
                date_range = f"{format_uk_date(valid_dates.min())} to {format_uk_date(valid_dates.max())}"
            else:
                date_range = "No dates available"
            st.metric("Date Range", date_range)
            
    with col3:
        if 'category' in reports_df.columns:
            categories = reports_df['category'].fillna('').str.split('|').explode().str.strip()
            categories = categories[categories != ''].unique()
            st.metric("Unique Categories", len(categories))
    
    # Category breakdown
    if 'category' in reports_df.columns:
        st.subheader("Category Distribution")
        category_counts = (reports_df['category'].fillna('')
                         .str.split('|')
                         .explode()
                         .str.strip()
                         .value_counts())
        category_counts = category_counts[category_counts.index != '']
        if not category_counts.empty:
            st.bar_chart(category_counts)
        else:
            st.write("No category data available")
    
    # Timeline analysis
    if 'date_of_report' in reports_df.columns:
        st.subheader("Reports Timeline")
        valid_dates_df = reports_df[reports_df['date_of_report'].notna()].copy()
        if not valid_dates_df.empty:
            timeline = valid_dates_df.set_index('date_of_report').resample('M').size()
            timeline.index = timeline.index.strftime('%B %Y')
            st.line_chart(timeline)
        else:
            st.write("No timeline data available")

def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
    st.markdown("""
    This app scrapes and analyses Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Filter by keywords and date range to find relevant reports.
    """)
    
    # Input form
    with st.form("search_form"):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_keyword = st.text_input("Search keywords:", "")
        with col2:
            start_date = st.date_input("Start date:", format="DD/MM/YYYY")
        with col3:
            end_date = st.date_input("End date:", format="DD/MM/YYYY")
        
        # Submit button
        submitted = st.form_submit_button("Search and Analyse Reports")
    
    # Handle form submission
    if submitted:
        with st.spinner("Searching for reports..."):
            reports = scrape_pfd_reports(
                keyword=search_keyword,
                start_date=start_date,
                end_date=end_date
            )
        
        if reports:
            df = pd.DataFrame(reports)
            
            # Clean and format data
            if 'date_of_report' in df.columns:
                # Convert to datetime, coerce errors to NaT
                df['date_of_report'] = pd.to_datetime(df['date_of_report'], errors='coerce')
            
            # Fill NaN values with appropriate placeholders
            df = df.fillna({
                'Title': 'Untitled',
                'category': '',
                'ref': 'Unknown',
                'deceased_name': 'Unknown',
                'coroner_name': 'Unknown',
                'coroner_area': 'Unknown',
                'sent_to': 'Unknown',
                'legal_powers': '',
                'investigation': '',
                'circumstances': '',
                'concerns': '',
                'action': '',
                'response': '',
                'copies': ''
            })
            
            st.success(f"Found {len(reports):,} reports")
            
            try:
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
                
                # Export options container
                with st.container():
                    export_format = st.selectbox(
                        "Export format:",
                        ["CSV", "Excel"],
                        key="export_format"
                    )
                    
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
            except Exception as e:
                st.error(f"Error analyzing reports: {str(e)}")
                st.write("Raw data:")
                st.write(df)
        else:
            st.warning("No reports found")

if __name__ == "__main__":
    main()
