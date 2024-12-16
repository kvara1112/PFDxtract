import streamlit as st
import pandas as pd
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup
import time
import urllib3
import io
import pdfplumber
import tempfile

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Analysis", layout="wide")

def clean_text(text):
    """Clean text by removing extra whitespace and special characters"""
    if text:
        # Replace special characters
        text = re.sub(r'[â€™]', "'", text)
        text = re.sub(r'[â€¦]', "...", text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""

def extract_pdf_text(pdf_url):
    """Extract text from PDF URL"""
    try:
        # Download PDF
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, headers=headers, verify=False, timeout=10)
        
        # Create a temporary file to save the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf_path = temp_pdf.name
        
        # Extract text from PDF
        with pdfplumber.open(temp_pdf_path) as pdf:
            # Combine text from all pages
            pdf_text = "\n\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        return clean_text(pdf_text)
    
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return ""

def get_report_content(url):
    """Get full content from report page"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        st.write(f"Fetching content from: {url}")
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find content in different possible locations
        content = soup.find('div', class_='flow')
        if not content:
            content = soup.find('article', class_='single__post')
        
        if content:
            # Get all text content preserving line breaks
            paragraphs = content.find_all(['p', 'table'])
            text_content = '\n\n'.join(p.get_text(strip=True, separator=' ') for p in paragraphs)
            if text_content:
                st.write("Successfully extracted webpage content")
                
                # Find PDF download links
                pdf_links = soup.find_all('a', class_='related-content__link', href=re.compile(r'\.pdf$'))
                pdf_texts = []
                
                for pdf_link in pdf_links:
                    pdf_url = pdf_link['href']
                    pdf_text = extract_pdf_text(pdf_url)
                    if pdf_text:
                        pdf_texts.append(pdf_text)
                
                # Combine webpage and PDF texts
                full_text = text_content
                if pdf_texts:
                    full_text += "\n\n--- PDF CONTENT ---\n\n" + "\n\n".join(pdf_texts)
                
                return clean_text(full_text)
        
        st.warning(f"No content found for: {url}")
        return None
        
    except Exception as e:
        st.error(f"Error getting report content: {str(e)}")
        return None

# Rest of the code remains the same as in the previous implementation
# (scrape_page, get_total_pages, scrape_pfd_reports, main functions)

# Main function remains unchanged
def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Enter keywords to search for relevant reports.
    """)
    
    # Use form for input
    with st.form("search_form"):
        search_keyword = st.text_input("Search keywords:", "")
        submitted = st.form_submit_button("Search Reports")
    
    if submitted:
        reports = []  # Initialize reports list
        with st.spinner("Searching for reports..."):
            scraped_reports = scrape_pfd_reports(keyword=search_keyword)
            if scraped_reports:
                reports.extend(scraped_reports)
        
        if reports:
            df = pd.DataFrame(reports)
            
            st.success(f"Found {len(reports):,} reports")
            
            # Show detailed data
            st.subheader("Reports Data")
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
