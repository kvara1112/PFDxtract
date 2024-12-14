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

def clean_text(text):
    """Clean extracted text by removing extra whitespace and newlines"""
    if text:
        return ' '.join(text.strip().split())
    return ""

def scrape_pfd_reports(category=None):
    """
    Scrape Prevention of Future Death reports based on category
    """
    base_url = "https://www.judiciary.uk/"
    
    # If category is specified, add it to the search parameters
    if category:
        search_url = f"{base_url}?post_type=pfd&s=Child+Death"
    else:
        search_url = f"{base_url}?post_type=pfd"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find all entries - look for divs containing report information
        report_entries = soup.find_all('div', class_='entry-content')
        
        if not report_entries:
            st.warning("No reports found.")
            return pd.DataFrame()
        
        reports = []
        for entry in report_entries:
            try:
                # Find the title link
                title_elem = entry.find_previous('h2', class_='entry-title')
                if not title_elem:
                    continue
                    
                link = title_elem.find('a')
                if not link:
                    continue
                
                title = clean_text(link.text)
                url = link['href']
                
                # Get the metadata text
                metadata_text = clean_text(entry.text)
                
                # Check if this is a Child Death report
                if 'Child Death' not in metadata_text:
                    continue
                
                # Extract information using regex
                patterns = {
                    'Date': r'Date of report: (\d{2}/\d{2}/\d{4})',
                    'Reference': r'Ref: ([\w-]+)',
                    'Deceased_Name': r'Deceased name: ([^,]+)',
                    'Coroner_Name': r'Coroner name: ([^,]+)',
                    'Coroner_Area': r'Coroner Area: ([^,]+)',
                    'Category': r'Category: ([^|]+)',
                }
                
                report = {
                    'Title': title,
                    'URL': url
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, metadata_text)
                    report[key] = clean_text(match.group(1)) if match else ""
                
                reports.append(report)
                
            except Exception as e:
                st.error(f"Error processing entry: {str(e)}")
                continue
        
        if reports:
            df = pd.DataFrame(reports)
            
            # Reorder columns
            column_order = ['Title', 'Date', 'Reference', 'Deceased_Name', 'Coroner_Name', 'Coroner_Area', 'Category', 'URL']
            df = df[column_order]
            
            return df
        else:
            st.warning("No Child Death reports found.")
            return pd.DataFrame()
        
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
    Currently set to find reports categorized as Child Death cases.
    """)
    
    if st.button("Search for Child Death Reports"):
        with st.spinner("Searching for Child Death reports..."):
            df = scrape_pfd_reports(category="Child Death")
            
            if not df.empty:
                st.success(f"Found {len(df)} Child Death reports")
                
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
                filename = f"child_death_reports_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
