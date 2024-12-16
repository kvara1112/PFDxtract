# main.py
import streamlit as st
from scraper import scrape_pfd_reports
from data_processor import load_and_process_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

st.set_page_config(page_title="UK Judiciary PFD Reports Analysis", layout="wide")

def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Scrape Reports", "Process Data"])
    
    with tab1:
        st.markdown("""
        ## Scrape Prevention of Future Deaths Reports
        Enter keywords to search for and scrape relevant reports from the UK Judiciary website.
        """)
        
        # Use form for input
        with st.form("search_form"):
            search_keyword = st.text_input("Search keywords:", "")
            save_path = st.text_input("Save file path (include .csv extension):", "scraped_reports.csv")
            submitted = st.form_submit_button("Search and Save Reports")
        
        if submitted:
            if not save_path.endswith('.csv'):
                st.error("Save path must end with .csv")
                return
                
            reports = []
            with st.spinner("Searching for reports..."):
                try:
                    scraped_reports = scrape_pfd_reports(keyword=search_keyword)
                    if scraped_reports:
                        reports.extend(scraped_reports)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logging.error(f"Scraping error: {e}")
            
            if reports:
                try:
                    import pandas as pd
                    df = pd.DataFrame(reports)
                    df.to_csv(save_path, index=False)
                    st.success(f"Successfully saved {len(reports)} reports to {save_path}")
                except Exception as e:
                    st.error(f"Error saving data: {e}")
            else:
                if search_keyword:
                    st.warning("No reports found matching your search criteria")
                else:
                    st.info("Please enter search keywords to find reports")
    
    with tab2:
        st.markdown("""
        ## Process Saved Reports
        Load and analyze previously scraped report data.
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
        if uploaded_file is not None:
            load_and_process_data(uploaded_file)

if __name__ == "__main__":
    main()
