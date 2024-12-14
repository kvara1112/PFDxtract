import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re

st.set_page_config(
    page_title="UK Judiciary PFD Reports Scraper",
    page_icon="📚",
    layout="wide"
)

def scrape_pfd_reports(keyword):
    """
    Scrape Prevention of Future Death reports from judiciary.uk based on keyword
    """
    base_url = "https://www.judiciary.uk/"
    search_url = f"{base_url}?s={keyword}&pfd_report_type=&post_type=pfd&order=relevance"
    
    try:
        # Add headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Use lxml parser for better performance
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find all report entries
        report_entries = soup.find_all('article')
        
        if not report_entries:
            st.warning("No reports found for the given keyword.")
            return pd.DataFrame()
        
        reports = []
        for entry in report_entries:
            try:
                # Extract title and clean it
                title_elem = entry.find('h2')
                title = title_elem.text.strip() if title_elem else "No title"
                
                # Extract metadata text
                metadata = entry.find('p')
                metadata_text = metadata.text.strip() if metadata else ""
                
                # Extract URL before h2 to avoid nested links
                url_elem = entry.find('a', href=True)
                url = url_elem['href'] if url_elem else ""
                
                # Parse metadata using regex
                patterns = {
                    'Date': r'Date of report: (\d{2}/\d{2}/\d{4})',
                    'Reference': r'Ref: ([\w-]+)',
                    'Deceased_Name': r'Deceased name: ([^,]+)',
                    'Coroner_Name': r'Coroner name: ([^,]+)',
                    'Coroner_Area': r'Coroner Area: ([^,]+)'
                }
                
                report = {'Title': title, 'URL': url}
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, metadata_text)
                    report[key] = match.group(1).strip() if match else ""
                
                reports.append(report)
                
            except Exception as e:
                st.error(f"Error processing entry: {str(e)}")
                continue
        
        return pd.DataFrame(reports)
    
    except requests.RequestException as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("UK Judiciary PFD Reports Scraper")
    
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Enter a keyword to search for relevant reports.
    """)
    
    with st.form("search_form"):
        keyword = st.text_input("Enter search keyword:", "maternity")
        submitted = st.form_submit_button("Search Reports")
        
        if submitted:
            with st.spinner("Searching for reports..."):
                df = scrape_pfd_reports(keyword)
                
                if not df.empty:
                    st.success(f"Found {len(df)} reports")
                    
                    # Display results in a clean table
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
                    filename = f"pfd_reports_{keyword}_{timestamp}.csv"
                    
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
