import streamlit as st
import pandas as pd
from datetime import datetime
import logging
import urllib3
import io
from scraper import scrape_pfd_reports, get_pfd_categories
from data_processing import process_scraped_data
from analysis_tab import render_analysis_tab

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="UK Judiciary PFD Reports Analysis", layout="wide")

def render_scraping_tab():
    st.markdown("""
    This app scrapes Prevention of Future Deaths (PFD) reports from the UK Judiciary website.
    You can search by keywords to find relevant reports.
    """)
    
    with st.form("search_form"):
        search_keyword = st.text_input("Search keywords:", "")
        submitted = st.form_submit_button("Search Reports")
    
    if submitted:
        if not search_keyword:
            st.warning("Please enter search keywords to find reports")
            return
            
        with st.spinner("Searching for reports..."):
            try:
                reports = scrape_pfd_reports(keyword=search_keyword)
                
                if reports:
                    df = pd.DataFrame(reports)
                    df = process_scraped_data(df)
                    
                    # Store in session state for analysis tab
                    st.session_state.scraped_data = df
                    
                    st.success(f"Found {len(reports):,} reports")
                    
                    # Show detailed data
                    st.subheader("Reports Data")
                    st.dataframe(
                        df,
                        column_config={
                            "URL": st.column_config.LinkColumn("Report Link"),
                            "date_of_report": st.column_config.DateColumn("Date of Report"),
                            "categories": st.column_config.ListColumn("Categories")
                        },
                        hide_index=True
                    )
                    
                    # Export options
                    st.subheader("Export Options")
                    export_format = st.selectbox("Export format:", ["CSV", "Excel"])
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pfd_reports_{search_keyword}_{timestamp}"
                    
                    if export_format == "CSV":
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Reports (CSV)",
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
                            "📥 Download Reports (Excel)",
                            excel_data,
                            f"{filename}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_excel"
                        )
                else:
                    st.warning("No reports found matching your search criteria")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logging.error(f"Scraping error: {e}")

def main():
    st.title("UK Judiciary PFD Reports Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Scrape Reports", "Analyze Reports"])
    
    # Initialize session state for sharing data between tabs
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = None
    
    with tab1:
        render_scraping_tab()
    
    with tab2:
        render_analysis_tab()

if __name__ == "__main__":
    main()
