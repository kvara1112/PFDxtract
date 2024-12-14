import streamlit as st
import pandas as pd
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="UK Judiciary PFD Reports Scraper",
    page_icon="📚",
    layout="wide"
)

def get_reports_by_category(category=None):
    """
    Scrape PFD reports and filter by category if specified
    """
    url = "https://www.judiciary.uk/"
    params = {
        'post_type': 'pfd',
        'pfd_report_type': '',
        'order': 'relevance'
    }
    
    if category:
        # Add category to search if specified
        params['s'] = category
    
    st.write(f"Searching for reports{'in category: ' + category if category else ''}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find all report entries
        entries = soup.find_all(['article', 'div'], class_=['post', 'entry', 'search-result'])
        
        reports = []
        for entry in entries:
            try:
                # Extract title and link
                title_elem = entry.find('h2', class_='entry-title')
                if not title_elem or not title_elem.find('a'):
                    continue
                    
                link = title_elem.find('a')
                title = link.text.strip()
                url = link['href']
                
                # Get metadata
                metadata = entry.find('p')
                if not metadata:
                    continue
                    
                metadata_text = metadata.text.strip()
                
                # Extract report info
                patterns = {
                    'Date': r'Date of report:?\s*(\d{2}/\d{2}/\d{4})',
                    'Reference': r'Ref:?\s*([\w-]+)',
                    'Deceased_Name': r'Deceased name:?\s*([^,\n]+)',
                    'Coroner_Name': r'Coroner name:?\s*([^,\n]+)',
                    'Coroner_Area': r'Coroner Area:?\s*([^,\n]+)',
                    'Category': r'Category:?\s*([^|\n]+)',
                    'Trust': r'This report is being sent to:\s*([^|\n]+)'
                }
                
                report = {
                    'Title': title,
                    'URL': url
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, metadata_text)
                    if match:
                        report[key] = match.group(1).strip()
                    else:
                        report[key] = ""
                
                # Only add report if it matches the selected category
                if not category or (report['Category'] and category.lower() in report['Category'].lower()):
                    reports.append(report)
                
            except Exception as e:
                st.error(f"Error processing entry: {str(e)}")
                continue
        
        return reports
    
    except Exception as e:
        st.error(f"Error fetching reports: {str(e)}")
        return []

def main():
    st.title("UK Judiciary PFD Reports Scraper")
    
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    Select a category to view relevant reports.
    """)
    
    # Predefined categories based on common report types
    categories = [
        "All Categories",
        "Child Death",
        "Hospital Death",
        "Mental Health",
        "Alcohol, Drug and Medication",
        "Accident at Work",
        "Road Traffic Death",
        "Care Home Health",
        "Emergency Services",
        "Other"
    ]
    
    selected_category = st.selectbox("Select Category:", categories)
    
    if st.button("Search Reports"):
        with st.spinner("Searching for reports..."):
            # If "All Categories" is selected, don't filter by category
            category_filter = None if selected_category == "All Categories" else selected_category
            reports = get_reports_by_category(category_filter)
            
            if reports:
                df = pd.DataFrame(reports)
                
                # Reorder columns
                columns = ['Title', 'Date', 'Reference', 'Deceased_Name', 'Coroner_Name', 
                          'Coroner_Area', 'Category', 'Trust', 'URL']
                df = df[columns]
                
                st.success(f"Found {len(reports)} reports")
                
                # Display results
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
                filename = f"pfd_reports_{selected_category}_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
            else:
                st.warning(f"No reports found for category: {selected_category}")

if __name__ == "__main__":
    main()
