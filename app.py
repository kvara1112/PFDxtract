#!/usr/bin/env python3
"""
UK Judiciary PFD Reports Analysis Tool - Main Application
A comprehensive tool for analyzing Prevention of Future Deaths (PFD) reports.
"""

import streamlit as st
import logging
import time
from datetime import datetime
from streamlit_modal import Modal
import csv
import os
import pandas as pd

import requests

# Configure Streamlit page
st.set_page_config(
    page_title="UK Judiciary PFD Reports Analysis",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our custom modules
from modules.streamlit_components import (
    initialize_session_state,
    check_app_password,
    render_footer,
    validate_data_state,
    handle_no_data_state,
    handle_error,
    upload_reports,
    render_scraping_tab,
    render_bert_file_merger,
    render_bert_analysis_tab,
    render_theme_analysis_dashboard,
    non_pfd_tab
)

from modules.vectorizer_models import render_topic_summary_tab
from modules.visualization import (
    plot_timeline,
    plot_monthly_distribution,
    plot_yearly_comparison,
    plot_category_distribution,
    plot_coroner_areas,
    analyze_data_quality
)

from modules.core_utils import (
    filter_by_categories,
    is_response_document,
    export_to_excel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)



def render_analysis_tab(data=None):
    """Render the analysis tab with improved filters and visualizations"""
    st.subheader("Data Analysis & Visualization")
    
    # Add file upload section at the top
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file", 
        type=['csv', 'xlsx'],
        help="Upload previously exported data"
    )
    
    if uploaded_file is not None:
        try:
            import pandas as pd
            from modules.core_utils import process_scraped_data
            
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Process uploaded data
            data = process_scraped_data(data)
            st.success("File uploaded and processed successfully!")
            
            # Update session state
            st.session_state.uploaded_data = data.copy()
            st.session_state.data_source = 'uploaded'
            st.session_state.current_data = data.copy()
        
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            logging.error(f"File upload error: {e}", exc_info=True)
            return
    
    # Use either uploaded data or passed data
    if data is None:
        data = st.session_state.get('current_data')
    
    if data is None or len(data) == 0:
        st.warning("No data available. Please upload a file or scrape reports first.")
        return
        
    try:
        import pandas as pd
        
        # Ensure date column is datetime
        if 'date_of_report' in data.columns:
            data['date_of_report'] = pd.to_datetime(data['date_of_report'], errors='coerce')
        
        # Get date range for the data
        min_date = data['date_of_report'].min().date()
        max_date = data['date_of_report'].max().date()
        
        # Filters sidebar
        with st.sidebar:
            st.header("Filters")
            
            # Date Range
            with st.expander("📅 Date Range", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "From",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="start_date_filter",
                        format="DD/MM/YYYY"
                    )
                with col2:
                    end_date = st.date_input(
                        "To",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="end_date_filter",
                        format="DD/MM/YYYY"
                    )
            
            # Document Type Filter
            doc_type = st.multiselect(
                "Document Type",
                ["Report", "Response"],
                default=[],
                key="doc_type_filter",
                help="Filter by document type"
            )
            
            # Categories
            if 'categories' in data.columns:
                all_categories = set()
                for cats in data['categories'].dropna():
                    if isinstance(cats, list):
                        all_categories.update(str(cat).strip() for cat in cats if cat)
                    elif isinstance(cats, str):
                        all_categories.update(str(cat).strip() for cat in cats.split(',') if cat)
                
                all_categories = {cat for cat in all_categories if cat.strip()}
                
                if all_categories:
                    selected_categories = st.multiselect(
                        "Categories",
                        options=sorted(all_categories),
                        key="categories_filter"
                    )
            
            # Reset Filters Button
            if st.button("🔄 Reset Filters"):
                for key in st.session_state:
                    if key.endswith('_filter'):
                        del st.session_state[key]
                st.rerun()

        # Apply filters
        filtered_df = data.copy()

        # Date filter
        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['date_of_report'].dt.date >= start_date) &
                (filtered_df['date_of_report'].dt.date <= end_date)
            ]

        # Document type filter
        if doc_type:
            if "Response" in doc_type and "Report" not in doc_type:
                filtered_df = filtered_df[filtered_df.apply(is_response_document, axis=1)]
            elif "Report" in doc_type and "Response" not in doc_type:
                filtered_df = filtered_df[~filtered_df.apply(is_response_document, axis=1)]

        # Categories filter
        if 'selected_categories' in locals() and selected_categories:
            filtered_df = filter_by_categories(filtered_df, selected_categories)

        # Display results
        st.subheader("Results")
        st.write(f"Showing {len(filtered_df)} of {len(data)} reports")

        if len(filtered_df) > 0:
            # Display the dataframe
            st.dataframe(
                filtered_df,
                column_config={
                    "URL": st.column_config.LinkColumn("Report Link"),
                    "date_of_report": st.column_config.DateColumn(
                        "Date of Report",
                        format="DD/MM/YYYY"
                    ),
                    "categories": st.column_config.ListColumn("Categories"),
                },
                hide_index=True
            )

            # Create tabs for different analyses
            st.markdown("---")
            quality_tab, temporal_tab, distribution_tab = st.tabs([
                "📊 Data Quality Analysis",
                "📅 Temporal Analysis", 
                "📍 Distribution Analysis"
            ])

            # Data Quality Analysis Tab
            with quality_tab:
                analyze_data_quality(filtered_df)

            # Temporal Analysis Tab
            with temporal_tab:
                st.subheader("Reports Timeline")
                plot_timeline(filtered_df)
                
                st.subheader("Monthly Distribution")
                plot_monthly_distribution(filtered_df)
                
                st.subheader("Year-over-Year Comparison")
                plot_yearly_comparison(filtered_df)

            # Distribution Analysis Tab
            with distribution_tab:
                st.subheader("Reports by Category")
                plot_category_distribution(filtered_df)

                st.subheader("Reports by Coroner Area")
                plot_coroner_areas(filtered_df)

            # Export options
            st.markdown("---")
            st.subheader("Export Options")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pfd_reports_analysis_{timestamp}"
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Reports (CSV)",
                    csv_data,
                    f"{filename}.csv",
                    "text/csv"
                )
            
            with col2:
                # Excel Export
                try:
                    excel_data = export_to_excel(filtered_df)
                    st.download_button(
                        "📥 Download Reports (Excel)",
                        excel_data,
                        f"{filename}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error preparing Excel export: {str(e)}")

        else:
            st.warning("No reports match your filter criteria. Try adjusting the filters.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Analysis error: {e}", exc_info=True)

if "button_clicked" not in st.session_state:
       st.session_state.button_clicked = None

def go_to_page(page):
    st.session_state.button_clicked = page
def main():
    """Updated main application entry point."""
    initialize_session_state()
    
    # Check authentication first
    if not check_app_password():
        # Render the footer even when not authenticated
        render_footer()
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.title("⚖️")
    with col4:
        # Initialize session state
        if "show_help" not in st.session_state:
            st.session_state.show_help = False

        # Custom CSS for just-icon button
        st.markdown(
            """
            <style>
            /* Make all buttons inside col4 just an icon */
            .help-btn button {
                background: none;
                border: none;
                padding: 0;
                font-size: 1.5rem;
                cursor: pointer;
            }
            .help-btn button:hover {
                color: #555;  /* optional hover effect */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Place the button in a div with class "help-btn"
        if st.button("🛈︎", key="help_icon", help="Click for help"):
            st.session_state.show_help = not st.session_state.show_help

        # Display help message
        if st.session_state.show_help:
            st.info("""
            This app has two functions:

            1) **Analyse PFD reports**  
            These are prevention of future death reports that follow a certain command. 
            You can either upload your own or search and filter for them.

            2) **Analyse other healthcare documents**  
            This performs a similar analysis, but for any other healthcare documents you may want to analyse.

            Click the button again to hide this message.
            """)
            
    if st.session_state.button_clicked is None or st.session_state.button_clicked == "dash" or st.session_state.button_clicked == "dash2":
        st.markdown(
                """
                <h1 style='text-align:center;'>Welcome to the Analyser</h1>""",
                unsafe_allow_html = True
            )
        button_style = """
                <style>
                div.stButton > button:first-child {
                    background-color: #0084B4;
                    color: white;
                    height: 200px;
                    width: 300px;
                    font-size: 60px;
                    border-radius: 30px;
                    border: 2px solid white;
                    border-color: white
                }
                div.stButton {
                    display: flex;
                    justify-content: center;
                    gap: 40px;
                }
                </style>
            """
        st.markdown(button_style, unsafe_allow_html=True)

        
        

        col1, col2, col3, col4 = st.columns(4)
    

        with col2:
            st.button("Analyse PFD Reports", key="page1_btn", on_click=go_to_page, args=("page1",))

        with col3:
            st.button("Analyse Other Healthcare Documents", key="page2_btn", on_click=go_to_page, args=("page2",))


    

    elif st.session_state.button_clicked == "page1":
        st.title("UK Judiciary PFD Reports Analysis")
        
        # Add main description
        st.markdown(
            """
            This application analyses Prevention of Future Deaths (PFD) reports from the UK Judiciary website to uncover patterns, themes, and insights.
            """
        )


        # Add collapsible help section
        with st.expander("💡 How to Use This Tool"):
            pdf_url = "https://raw.githubusercontent.com/gcosma/PFDxtract/main/Usability%20Guide%20PFDxtract.pdf"
            st.markdown(f"""
                        <a href = "{pdf_url}" target="_blank">
                            <button style="background-color: #0083B3;
                                            color: white;
                                            padding: 10px 20px;
                                            font-size: 16px;
                                            border: none;
                                            border-radius: 8px;
                                            cursor: pointer;">Open Usability Guide 📝</button>
                        </a>
                        """, unsafe_allow_html = True)
            st.markdown(
                
                """
                ### Complete Analysis Pipeline:
                
                1. **(Step 1) 🔍 Scrape Reports**: Start by collecting PFD reports from the UK Judiciary website
                2. **(Step 2) 📂 Scraped File Preparation**: Process and merge your scraped reports
                3. **(Step 3) 📊 Scraped File Analysis**: Visualise and analyse basic report patterns
                4. **(Step 4) 📝 Topic Analysis & Summaries**: Generate basic themes from report content
                5. **(Step 5) 🔬 Concept Annotation**: Conduct advanced theme analysis with AI
                6. **(Step 6) 📈 Theme Analysis Dashboard**: Explore comprehensive theme visualisations
                7. **(Step 7) 📃 Non PFD Analysis**: Analyse reports that are not PFD reports 
                
                Select each numbered tab in sequence to move through the complete analysis pipeline.
                
                ### Tips:
                - Each step generates outputs that can be used in subsequent steps
                - You can upload previously generated files at any step
                - Use the "Clear All Data" button in the sidebar to reset the application
                """
            )
            

        # The radio button selection remains outside the expander
        current_tab = st.radio(
            "Select section:",
            [
                "(1)🔍 Scrape Reports",
                "(2)📂 Scraped File Preparation",
                "(3)📊 Scraped File Analysis",
                "(4)📝 Topic Analysis & Summaries", 
                "(5)🔬 Concept Annotation",
                "(6)📈 Theme Analysis Dashboard"
            ],
            label_visibility="collapsed",
            horizontal=True,
            key="main_tab_selector",
        )
        st.markdown("---")

        try:
            if current_tab == "(1)🔍 Scrape Reports":
                # Add tab-specific description here
                st.markdown(
                    """
                    Search tool for Prevention of Future Deaths (PFD) reports from the UK Judiciary website.

                    - Extract detailed PFD reports with metadata, full content, and associated PDFs
                    - Filtering by keywords, categories, and date ranges
                    - Export options in CSV and Excel formats

                    Handling Large Result Sets: For extensive search results, use the 'Start page' and 'End page' number inputs to download reports in manageable batches.
                    Upload your own PFD reports of filter and search for PFD reports
                    """
                )
                st.subheader("Upload PFD Reports")
                upload_reports(True)
                
                st.subheader("Scrape PFD Reports")
                
                render_scraping_tab()
            
            elif current_tab == "(2)📂 Scraped File Preparation":
                # Add tab-specific description here
                st.markdown(
                    """
                    This tool merges multiple scraped files into a single dataset. It prepares the data for steps (3) - (5).
                    
                    - Run this step even if you only have one scraped file. This step extracts the year and applies other processing.
                    - Combine data from multiple CSV or Excel files (files starting with pfd_reports_scraped/uploaded_reportID_)
                    - Extract missing concerns from PDF content and fill empty Content fields
                    - Extract year information from date fields
                    - Remove duplicate records
                    - Export full or reduced datasets with essential columns
                    """
                )
                render_bert_file_merger()
            
            elif current_tab == "(3)📊 Scraped File Analysis":
                # Add tab-specific description here
                st.markdown(
                    """
                    Analyse and explore your prepared Prevention of Future Deaths (PFD) reports.
                    - Upload processed files from Scraped File Preparation (file starting with merged_)
                    - Data visualisation
                    - Report insights and export options

                    Upload your prepared CSV or Excel file from Step 2 to begin analysis.
                    """
                )
                if not validate_data_state():
                    handle_no_data_state("analysis")
                else:
                    render_analysis_tab(st.session_state.current_data)
            
            elif current_tab == "(4)📝 Topic Analysis & Summaries":
                # Add tab-specific description here
                st.markdown(
                    """
                    Basic thematic analysis of Prevention of Future Deaths (PFD) reports.
                    - Automatically identify key themes across document collections
                    - Cluster similar documents (adjust the parameters to identify optimal clusters)
                    - Generate summaries for each identified theme
                    - Visualise relationships between key concepts and topics
                    """
                )
                if not validate_data_state():
                    handle_no_data_state("topic_summary")
                else:
                    render_topic_summary_tab(True, st.session_state.current_data)
            
            elif current_tab == "(5)🔬 Concept Annotation":
                # Add tab-specific description here
                st.markdown(
                    """
                    Advanced AI-powered thematic analysis.
                    - Upload the merged Prevention of Future Deaths (PFD) reports file from step (2) 
                    - Automatic extraction of important themes from Prevention of Future Deaths (PFD) reports (using 4 frameworks)
                    - Download detailed results in structured tables
                    - Download colour highlighted sentences based on theme colours in a html report
                    """
                )
                render_bert_analysis_tab(True, st.session_state.current_data)
                
            elif current_tab == "(6)📈 Theme Analysis Dashboard":
                # Add tab-specific description here
                st.markdown(
                    """
                    Interactive Theme Analysis Dashboard
                    
                    - Upload theme analysis results from step (5) (file named annotated_theme_analysis_*.xlsx)
                    - Navigate through multiple visualization tabs: framework heatmaps, distribution charts, temporal analysis and more
                    - Filter results by framework, year, coroner area, and confidence level
                    - Discover relationships between themes using correlation analysis
                    """
                )
                render_theme_analysis_dashboard(True, st.session_state.current_data)
            # Sidebar data management
            with st.sidebar:
                st.header("Data Management")
            
                if hasattr(st.session_state, "data_source"):
                    st.info(f"Current data: {st.session_state.data_source}")

                #
                if st.button("Clear All Data", key="clear_all_data_button"):
                    # Define a comprehensive list of keys to clear
                    keys_to_clear = [
                        # Core data keys
                        "current_data",
                        "scraped_data", 
                        "uploaded_data",
                        "topic_model",
                        "data_source",
                        
                        # BERT-specific keys
                        "bert_results",
                        "bert_initialized",
                        "bert_merged_data",
                        
                        # Dashboard specific keys
                        "dashboard_data",
                        "theme_analysis_dashboard_uploader",
                        
                        # File upload keys
                        "bert_file_uploader",
                        "bert_content_column",
                        "bert_analysis_type",
                        "bert_selected_indices",
                        "bert_similarity_threshold",
                        
                        # BERT merger settings keys
                        "drop_duplicates_static",
                        "extract_year_static",
                        "extract_from_pdf_static",
                        "fill_empty_content_static",
                        "duplicate_columns_static",
                        "merge_files_button_static",
                    ]
                    
                    # Clear each key if it exists
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Force re-initialization of key values
                    # These explicit resets ensure clean state
                    st.session_state.current_data = None
                    st.session_state.uploaded_data = None
                    st.session_state.scraped_data = None
                    st.session_state.data_source = None
                    st.session_state.bert_results = {}
                    st.session_state.bert_initialized = False
                    st.session_state.bert_merged_data = None
                    st.session_state.dashboard_data = None
                    
                    # Clear all filter-related keys
                    for key in list(st.session_state.keys()):
                        if key.startswith('filter_'):
                            del st.session_state[key]
                    
                    # Clear any cached file information
                    if 'last_uploaded_file_hash' in st.session_state:
                        del st.session_state.last_uploaded_file_hash
                    
                    # Generate a unique key for file uploaders to force reload
                    if "reset_counter" not in st.session_state:
                        st.session_state.reset_counter = 0
                    st.session_state.reset_counter += 1
                    
                    # Give feedback and rerun
                    st.success("All data cleared successfully")
                    time.sleep(0.5)  # Brief pause to ensure UI updates
                    st.rerun()
                st.button("Back to Dashboard", key="dash_btn", on_click=go_to_page, args=("dash",) )
                    
        
                
                
                # Add logout button
                if st.button("Logout"):
                    st.session_state.authenticated = False
                    st.session_state.button_clicked = None
                    st.rerun()

            render_footer()

        except Exception as e:
            handle_error(e)
            
            # Render footer even when an exception occurs
            render_footer()
    elif st.session_state.button_clicked == "page2":
        st.title("Healthcare Document Analyser")
        
        st.markdown(
                """
                Analyse any medical report

                - Upload a folder of your reports
                - Perform cleaning and formatting into one document
                - Complete a topic analysis on the reports
                - Concept anntoate your reports by framework
                - Visualise annotations and results
                """
            )
        non_pfd_tab(st.session_state.current_data)
        with st.sidebar:
            st.header("Data Management")
            st.button("Back to Dashboard", key="dash_btn", on_click=go_to_page, args=("dash",) )

            # Add logout button
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.button_clicked = None
                st.rerun()
    
    

        
    
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Critical Error")
        st.error(str(e))
        logging.critical(f"Application crash: {e}", exc_info=True)
