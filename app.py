import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re

def scrape_pfd_reports(keyword):
    """
    Scrape Prevention of Future Death reports from judiciary.uk based on keyword
    """
    base_url = "https://www.judiciary.uk/"
    search_url = f"{base_url}?s={keyword}&pfd_report_type=&post_type=pfd&order=relevance"
    
    # Initialize list to store report data
    reports = []
    
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all report entries
        report_entries = soup.find_all('article')
        
        for entry in report_entries:
            # Extract title
            title = entry.find('h2').text.strip() if entry.find('h2') else "No title"
            
            # Extract metadata text
            metadata = entry.find('p').text.strip() if entry.find('p') else ""
            
            # Parse metadata using regex
            date_match = re.search(r'Date of report: (\d{2}/\d{2}/\d{4})', metadata)
            ref_match = re.search(r'Ref: ([\w-]+)', metadata)
            deceased_match = re.search(r'Deceased name: ([^,]+)', metadata)
            coroner_match = re.search(r'Coroner name: ([^,]+)', metadata)
            area_match = re.search(r'Coroner Area: ([^,]+)', metadata)
            
            # Extract URL
            url = entry.find('a')['href'] if entry.find('a') else ""
            
            # Create report dictionary
            report = {
                'Title': title,
                'Date': date_match.group(1) if date_match else "",
                'Reference': ref_match.group(1) if ref_match else "",
                'Deceased_Name': deceased_match.group(1) if deceased_match else "",
                'Coroner_Name': coroner_match.group(1) if coroner_match else "",
                'Coroner_Area': area_match.group(1) if area_match else "",
                'URL': url,
            }
            
            reports.append(report)
            
        return pd.DataFrame(reports)
    
    except Exception as e:
        st.error(f"Error occurred while scraping: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("UK Judiciary PFD Reports Scraper")
    
    # Add description
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Enter a keyword to search for relevant reports.
    """)
    
    # Input field for keyword
    keyword = st.text_input("Enter search keyword:", "maternity")
    
    if st.button("Search Reports"):
        with st.spinner("Scraping reports..."):
            # Perform scraping
            df = scrape_pfd_reports(keyword)
            
            if not df.empty:
                # Display results
                st.subheader(f"Found {len(df)} reports")
                st.dataframe(df)
                
                # Download button
                csv = df.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pfd_reports_{keyword}_{timestamp}.csv"
                
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            else:
                st.warning("No reports found or error occurred during scraping.")

if __name__ == "__main__":
    main()
