import streamlit as st
import pandas as pd
import logging
import time
import os
import io
import zipfile
import random
import string
from datetime import datetime
from typing import Dict, List, Optional
from openpyxl.utils import get_column_letter
import pytz

# Import our modules
from core_utils import (
    process_scraped_data, 
    validate_data, 
    export_to_excel,
    filter_by_categories,
    filter_by_areas,
    filter_by_coroner_names,
    filter_by_document_type,
    is_response_document
)
from web_scraping import (
    get_pfd_categories,
    get_sort_options,
    scrape_pfd_reports,
    get_total_pages,
    construct_search_url,
    estimate_scraping_time
)
from vectorizer_models import render_topic_summary_tab
from bert_analysis import BERTResultsAnalyzer, ThemeAnalyzer
from visualization import (
    plot_category_distribution,
    plot_coroner_areas,
    plot_timeline,
    plot_monthly_distribution,
    plot_yearly_comparison,
    analyze_data_quality,
    render_framework_heatmap
)

# Add this to your initialize_session_state function
def initialize_session_state():
    """Initialize all required session state variables"""
    # Check if we need to perform first-time initialization
    if "initialized" not in st.session_state:
        # Clear all existing session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
            
        # Set new session state variables
        st.session_state.initialized = True
        st.session_state.data_source = None
        st.session_state.current_data = None
        st.session_state.scraped_data = None
        st.session_state.uploaded_data = None
        st.session_state.topic_model = None
        st.session_state.cleanup_done = False
        st.session_state.last_scrape_time = None
        st.session_state.last_upload_time = None
        st.session_state.reset_counter = 0  # Add this counter for file uploader resets
        
        # BERT-related session state variables
        st.session_state.bert_results = {}
        st.session_state.bert_initialized = False
        st.session_state.bert_merged_data = None
        
        # Analysis filters
        st.session_state.analysis_filters = {
            "date_range": None,
            "selected_categories": None,
            "selected_areas": None,
        }
        
        # Topic model settings
        st.session_state.topic_model_settings = {
            "num_topics": 5,
            "max_features": 1000,
            "similarity_threshold": 0.3,
        }
    
    # Perform PDF cleanup if not done
    if not st.session_state.get("cleanup_done", False):
        try:
            pdf_dir = "pdfs"
            os.makedirs(pdf_dir, exist_ok=True)
            current_time = time.time()
            cleanup_count = 0
            
            for file in os.listdir(pdf_dir):
                file_path = os.path.join(pdf_dir, file)
                try:
                    if os.path.isfile(file_path):
                        if os.stat(file_path).st_mtime < current_time - 86400:
                            os.remove(file_path)
                            cleanup_count += 1
                except Exception as e:
                    logging.warning(f"Error cleaning up file {file_path}: {e}")
                    continue
                    
            if cleanup_count > 0:
                logging.info(f"Cleaned up {cleanup_count} old PDF files")
        except Exception as e:
            logging.error(f"Error during PDF cleanup: {e}")
        finally:
            st.session_state.cleanup_done = True
            
def initialize_session_state2():
    """Initialize all required session state variables"""
    # Initialize basic state variables if they don't exist
    if not hasattr(st.session_state, "initialized"):
        # Clear all existing session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Set new session state variables
        st.session_state.data_source = None
        st.session_state.current_data = None
        st.session_state.scraped_data = None
        st.session_state.uploaded_data = None
        st.session_state.topic_model = None
        st.session_state.cleanup_done = False
        st.session_state.last_scrape_time = None
        st.session_state.last_upload_time = None
        st.session_state.analysis_filters = {
            "date_range": None,
            "selected_categories": None,
            "selected_areas": None,
        }
        st.session_state.topic_model_settings = {
            "num_topics": 5,
            "max_features": 1000,
            "similarity_threshold": 0.3,
        }
        st.session_state.initialized = True

    # Perform PDF cleanup if not done
    if not st.session_state.cleanup_done:
        try:
            pdf_dir = "pdfs"
            os.makedirs(pdf_dir, exist_ok=True)

            current_time = time.time()
            cleanup_count = 0

            for file in os.listdir(pdf_dir):
                file_path = os.path.join(pdf_dir, file)
                try:
                    if os.path.isfile(file_path):
                        if os.stat(file_path).st_mtime < current_time - 86400:
                            os.remove(file_path)
                            cleanup_count += 1
                except Exception as e:
                    logging.warning(f"Error cleaning up file {file_path}: {e}")
                    continue

            if cleanup_count > 0:
                logging.info(f"Cleaned up {cleanup_count} old PDF files")
        except Exception as e:
            logging.error(f"Error during PDF cleanup: {e}")
        finally:
            st.session_state.cleanup_done = True
            
def check_app_password():
    """Check if user has entered the correct password to access the app"""
    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # If already authenticated, continue
    if st.session_state.authenticated:
        return True
    
    # Otherwise show login screen
    st.title("UK Judiciary PFD Reports Analysis")
    st.markdown("### Authentication Required")
    st.markdown("Please enter the password to access the application.")

    correct_password = "amazing2"
    # Password input as a form to accept enter key - Jamie L
    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if password == correct_password:
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
                return True
            else:
                st.error("Incorrect password. Please try again.")
                return False
    
    # Old code for password input - Jamie L
    # # Submit button
    # if st.button("Login"):
    #     # Get correct password from secrets.toml
    #     correct_password = "amazing2"
        
    #     if password == correct_password:
    #         st.session_state.authenticated = True
    #         st.success("Login successful!")
    #         st.rerun()
    #         return True
    #     else:
    #         st.error("Incorrect password. Please try again.")
    #         return False
    
    return False

def render_footer():
    """Render footer with timestamp in UK time (GMT/BST)."""
    # Get file modification time (UTC by default on Streamlit Cloud)
    file_path = os.path.abspath(__file__)
    last_modified_timestamp = os.path.getmtime(file_path)
    last_modified_datetime_utc = datetime.fromtimestamp(last_modified_timestamp, tz=pytz.utc)
    
    # Convert to UK time (handles GMT/BST automatically)
    uk_tz = pytz.timezone("Europe/London")
    last_modified_datetime_uk = last_modified_datetime_utc.replace(tzinfo=pytz.utc).astimezone(uk_tz)
    formatted_time = last_modified_datetime_uk.strftime("%d %B %Y %H:%M (UK Time)")

    # Display footer
    st.markdown("---")
    st.markdown(
        f"""<div style='text-align: center'>
        <p>Built with Streamlit • Data Source: UK Judiciary • Copyright © 2025 Loughborough University • Developer: Georgina Cosma • 
        Contact: g.cosma@lboro.ac.uk • All rights reserved. Last update:{formatted_time}</p>
        </div>""",
        unsafe_allow_html=True,
    )

def validate_data_state():
    """Validate current data state"""
    return (st.session_state.get("current_data") is not None and 
            len(st.session_state.current_data) > 0)

def handle_no_data_state(tab_type: str):
    """Handle when no data is available"""
    st.warning("No data available. Please complete previous steps first.")
    
    if tab_type == "analysis":
        st.info("To analyze data, you need to either:")
        st.markdown("- Scrape reports using the 'Scrape Reports' tab")
        st.markdown("- Upload existing data using the file uploader below")
        
        # Add file upload option
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file", 
            type=['csv', 'xlsx'],
            help="Upload previously exported data"
        )
        
        if uploaded_file is not None:
            try:
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
                
                # Trigger rerun to show data
                st.rerun()
            
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
                logging.error(f"File upload error: {e}", exc_info=True)

def handle_error(error):
    """Handle application errors gracefully"""
    st.error("An unexpected error occurred")
    with st.expander("Error Details"):
        st.code(str(error))
    logging.error(f"Application error: {error}", exc_info=True)

def render_scraping_tab():
    """Render the scraping tab with batch saving options and date filters"""
    st.subheader("Scrape PFD Reports")

    # Initialize default values if not in session state
    if "init_done" not in st.session_state:
        st.session_state.init_done = True
        st.session_state["search_keyword_default"] = ""
        st.session_state["category_default"] = ""
        st.session_state["order_default"] = "relevance"
        st.session_state["start_page_default"] = 1
        st.session_state["end_page_default"] = None
        st.session_state["auto_save_batches_default"] = True
        st.session_state["batch_size_default"] = 5

    if "scraped_data" in st.session_state and st.session_state.scraped_data is not None:
        st.success(f"Found {len(st.session_state.scraped_data)} reports")

        st.subheader("Results")
        st.dataframe(
            st.session_state.scraped_data,
            column_config={
                "URL": st.column_config.LinkColumn("Report Link"),
                "date_of_report": st.column_config.DateColumn(
                    "Date of Report", format="DD/MM/YYYY"
                ),
                "categories": st.column_config.ListColumn("Categories"),
            },
            hide_index=True,
        )

        show_export_options(st.session_state.scraped_data, "scraped")

    # Create the search form with page range selection and batch options
    with st.form("scraping_form"):
        # Create rows for the main search criteria
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        # First row - Main search criteria
        with row1_col1:
            search_keyword = st.text_input(
                "Search keywords:",
                value=st.session_state.get("search_keyword_default", ""),
                key="search_keyword",
                help="Do not leave empty, use 'report' or another search term",
            )

        with row1_col2:
            category = st.selectbox(
                "PFD Report type:",
                [""] + get_pfd_categories(),
                index=0,
                key="category",
                format_func=lambda x: x if x else "Select element",
            )

        # Second row - Sort by
        with row2_col1:
            order = st.selectbox(
                "Sort by:",
                ["relevance", "desc", "asc"],
                index=0,
                key="order",
                format_func=lambda x: {
                    "relevance": "Relevance",
                    "desc": "Newest first",
                    "asc": "Oldest first",
                }[x],
            )
            
        # Date filter section
        st.markdown("### Filter search")
        
        # Published on or after
        st.markdown("**Published on or after**")
        st.markdown("For example, 27 3 2007")
        after_day_col, after_month_col, after_year_col = st.columns(3)
        
        with after_day_col:
            after_day = st.number_input(
                "Day",
                min_value=0,
                max_value=31,
                value=0,
                key="after_day"
            )
        
        with after_month_col:
            after_month = st.number_input(
                "Month",
                min_value=0,
                max_value=12,
                value=0,
                key="after_month"
            )
        
        with after_year_col:
            after_year = st.number_input(
                "Year",
                min_value=0,
                max_value=2025,
                value=0,
                key="after_year"
            )
        
        # Published on or before
        st.markdown("**Published on or before**")
        st.markdown("For example, 27 3 2007")
        before_day_col, before_month_col, before_year_col = st.columns(3)
        
        with before_day_col:
            before_day = st.number_input(
                "Day",
                min_value=0,
                max_value=31,
                value=0,
                key="before_day"
            )
        
        with before_month_col:
            before_month = st.number_input(
                "Month",
                min_value=0,
                max_value=12,
                value=0,
                key="before_month"
            )
        
        with before_year_col:
            before_year = st.number_input(
                "Year",
                min_value=0,
                max_value=2025,
                value=0,
                key="before_year"
            )

        # Create date filter strings for preview
        after_date = None
        if after_day > 0 and after_month > 0 and after_year > 0:
            after_date = f"{after_day}-{after_month}-{after_year}"
            
        before_date = None
        if before_day > 0 and before_month > 0 and before_year > 0:
            before_date = f"{before_day}-{before_month}-{before_year}"

        # Display preview results count with date filters
        if search_keyword or category or after_date or before_date:
            base_url = "https://www.judiciary.uk/"

            # Prepare category slug
            category_slug = None
            if category:
                category_slug = (
                    category.lower()
                    .replace(" ", "-")
                    .replace("&", "and")
                    .replace("--", "-")
                    .strip("-")
                )

            # Create preview URL with date filters
            preview_url = construct_search_url(
                base_url=base_url,
                keyword=search_keyword,
                category=category,
                category_slug=category_slug,
                after_date=after_date,
                before_date=before_date,
            )

            try:
                with st.spinner("Checking total pages..."):
                    total_pages, total_results = get_total_pages(preview_url)
                    if total_pages > 0:
                        st.info(
                            f"After filtering, this search has {total_pages} pages with {total_results} results"
                        )
                        st.session_state["total_pages_preview"] = total_pages
                    else:
                        st.warning("No results found for this search with the current filters")
                        st.session_state["total_pages_preview"] = 0
            except Exception as e:
                st.error(f"Error checking pages: {str(e)}")
                st.session_state["total_pages_preview"] = 0
        else:
            st.session_state["total_pages_preview"] = 0
            
        # Page settings AFTER filter search
        row3_col1, row3_col2 = st.columns(2)
        row4_col1, row4_col2 = st.columns(2)

        # Page range row - MOVED to after filter search
        with row3_col1:
            start_page = st.number_input(
                "Start page:",
                min_value=1,
                value=st.session_state.get("start_page_default", 1),
                key="start_page",
                help="First page to scrape (minimum 1)",
            )

        with row3_col2:
            end_page = st.number_input(
                "End page (Optimal: 10 pages per extraction):",
                min_value=0,
                value=st.session_state.get("end_page_default", 0),
                key="end_page",
                help="Last page to scrape (0 for all pages)",
            )

        # Batch options row - MOVED to after filter search
        with row4_col1:
            auto_save_batches = st.checkbox(
                "Auto-save batches",
                value=st.session_state.get("auto_save_batches_default", True),
                key="auto_save_batches",
                help="Automatically save results in batches as they are scraped",
            )

        with row4_col2:
            batch_size = st.number_input(
                "Pages per batch: (ideally set to 5)",
                min_value=1,
                max_value=10,
                value=st.session_state.get("batch_size_default", 5),
                key="batch_size",
                help="Number of pages to process before saving a batch",
            )

        # Action buttons in a row
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            submitted = st.form_submit_button("Search Reports")
        with button_col2:
            stop_scraping = st.form_submit_button("Stop Scraping")

    # Handle stop scraping
    if stop_scraping:
        st.session_state.stop_scraping = True
        st.warning("Scraping will be stopped after the current page completes...")
        return

    if submitted:
        try:
            # Store search parameters in session state
            st.session_state.last_search_params = {
                "keyword": search_keyword,
                "category": category,
                "order": order,
                "start_page": start_page,
                "end_page": end_page,
                "auto_save_batches": auto_save_batches,
                "batch_size": batch_size,
                "after_day": after_day,
                "after_month": after_month,
                "after_year": after_year,
                "before_day": before_day,
                "before_month": before_month,
                "before_year": before_year,
            }

            # Initialize stop_scraping flag
            st.session_state.stop_scraping = False

            # Convert end_page=0 to None (all pages)
            end_page_val = None if end_page == 0 else end_page
            
            # Create date filter strings
            after_date = None
            if after_day > 0 and after_month > 0 and after_year > 0:
                after_date = f"{after_day}-{after_month}-{after_year}"
                
            before_date = None
            if before_day > 0 and before_month > 0 and before_year > 0:
                before_date = f"{before_day}-{before_month}-{before_year}"

            # Perform scraping with batch options and date filters
            reports = scrape_pfd_reports(
                keyword=search_keyword,
                category=category if category else None,
                order=order,
                start_page=start_page,
                end_page=end_page_val,
                auto_save_batches=auto_save_batches,
                batch_size=batch_size,
                after_date=after_date,
                before_date=before_date,
            )

            if reports:
                # Process the data
                df = pd.DataFrame(reports)
                df = process_scraped_data(df)

                # Store in session state
                st.session_state.scraped_data = df
                st.session_state.data_source = "scraped"
                st.session_state.current_data = df

                # Trigger a rerun to refresh the page
                st.rerun()
            else:
                st.warning("No reports found matching your search criteria")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Scraping error: {e}")
            return False

def render_bert_file_merger():
    """Render the BERT file merger interface"""
    analyzer = BERTResultsAnalyzer()
    analyzer.render_analyzer_ui()

def render_bert_analysis_tab(data: pd.DataFrame = None):
    """Render the BERT analysis interface"""
    st.subheader("BERT Theme Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload merged file from Step 2",
        type=["csv", "xlsx"],
        help="Upload the merged file from the previous step",
        key="bert_analysis_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Process data
            data = process_scraped_data(data)
            st.success(f"File loaded successfully with {len(data)} records")
            
            # Initialize theme analyzer
            if "theme_analyzer" not in st.session_state:
                with st.spinner("Initializing BERT model..."):
                    st.session_state.theme_analyzer = ThemeAnalyzer()
            
            # Analysis options
            with st.expander("Analysis Settings", expanded=True):
                content_column = st.selectbox(
                    "Content Column",
                    options=[col for col in data.columns if "content" in col.lower()],
                    key="bert_content_column"
                )
                
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.65,
                    step=0.05,
                    key="bert_similarity_threshold"
                )
            
            # Run analysis
            if st.button("🔬 Run Theme Analysis", key="run_bert_analysis"):
                try:
                    with st.spinner("Analyzing themes..."):
                        analyzer = st.session_state.theme_analyzer
                        analyzer.config["base_similarity_threshold"] = similarity_threshold
                        
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in data.iterrows():
                            # Update progress
                            progress = (idx + 1) / len(data)
                            progress_bar.progress(progress)
                            
                            # Analyze document
                            content = str(row.get(content_column, ""))
                            if content and len(content.strip()) > 50:
                                themes, highlights = analyzer.analyze_document(content)
                                
                                # Process results
                                for framework, framework_themes in themes.items():
                                    for theme_data in framework_themes:
                                        result_row = {
                                            "Record ID": row.get("Record ID", idx),
                                            "Title": row.get("Title", ""),
                                            "Framework": framework,
                                            "Theme": theme_data["theme"],
                                            "Combined Score": theme_data["combined_score"],
                                            "Semantic Similarity": theme_data["semantic_similarity"],
                                            "Matched Keywords": theme_data["matched_keywords"],
                                            "Keyword Count": theme_data["keyword_count"],
                                        }
                                        
                                        # Add original row data
                                        for col in ["year", "coroner_area", "deceased_name"]:
                                            if col in row:
                                                result_row[col] = row[col]
                                        
                                        results.append(result_row)
                        
                        progress_bar.empty()
                        
                        if results:
                            results_df = pd.DataFrame(results)
                            st.success(f"Analysis complete! Found {len(results)} theme matches")
                            
                            # Show results
                            st.subheader("Analysis Results")
                            st.dataframe(results_df)
                            
                            # Download options
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"annotated_theme_analysis_{timestamp}.xlsx"
                            
                            excel_buffer = io.BytesIO()
                            results_df.to_excel(excel_buffer, index=False, engine="openpyxl")
                            excel_buffer.seek(0)
                            
                            st.download_button(
                                "📥 Download Results (Excel)",
                                data=excel_buffer.getvalue(),
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.warning("No themes found. Try adjusting the similarity threshold.")
                            
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    logging.error(f"BERT analysis error: {e}", exc_info=True)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("Please upload a file to begin theme analysis.")

def render_theme_analysis_dashboard(data: pd.DataFrame = None):
    """Render the theme analysis dashboard"""
    st.subheader("Theme Analysis Dashboard")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload theme analysis results from Step 5",
        type=["xlsx", "csv"],
        help="Upload annotated_theme_analysis_*.xlsx file from Step 5",
        key="dashboard_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.success(f"Loaded {len(data)} theme analysis results")
            
            # Sidebar filters
            with st.sidebar:
                st.header("Dashboard Filters")
                
                # Framework filter
                frameworks = data["Framework"].unique()
                selected_frameworks = st.multiselect(
                    "Frameworks",
                    options=frameworks,
                    default=frameworks,
                    key="framework_filter"
                )
                
                # Year filter
                if "year" in data.columns:
                    years = sorted(data["year"].dropna().unique())
                    selected_years = st.multiselect(
                        "Years",
                        options=years,
                        default=years,
                        key="year_filter"
                    )
                
                # Confidence filter
                if "Combined Score" in data.columns:
                    min_confidence = st.slider(
                        "Minimum Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        key="confidence_filter"
                    )
            
            # Apply filters
            filtered_data = data.copy()
            if selected_frameworks:
                filtered_data = filtered_data[filtered_data["Framework"].isin(selected_frameworks)]
            if "year" in data.columns and selected_years:
                filtered_data = filtered_data[filtered_data["year"].isin(selected_years)]
            if "Combined Score" in data.columns:
                filtered_data = filtered_data[filtered_data["Combined Score"] >= min_confidence]
            
            # Dashboard tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Overview",
                "🔥 Framework Heatmap", 
                "📈 Trends",
                "🎯 Theme Details"
            ])
            
            with tab1:
                st.subheader("Analysis Overview")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Documents", filtered_data["Record ID"].nunique())
                with col2:
                    st.metric("Total Themes", len(filtered_data))
                with col3:
                    st.metric("Frameworks", len(selected_frameworks))
                with col4:
                    if "Combined Score" in filtered_data.columns:
                        avg_confidence = filtered_data["Combined Score"].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                
                # Framework distribution
                from visualization import plot_framework_comparison
                plot_framework_comparison(filtered_data)
            
            with tab2:
                st.subheader("Framework Heatmap by Year")
                
                if "year" in filtered_data.columns and not filtered_data["year"].isna().all():
                    fig = render_framework_heatmap(filtered_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Unable to create heatmap - insufficient year data")
                else:
                    st.warning("Year data not available for heatmap")
            
            with tab3:
                st.subheader("Theme Trends")
                
                from visualization import (
                    plot_themes_by_year,
                    plot_theme_confidence_distribution
                )
                
                if "year" in filtered_data.columns:
                    plot_themes_by_year(filtered_data)
                
                plot_theme_confidence_distribution(filtered_data)
            
            with tab4:
                st.subheader("Theme Details")
                
                # Theme selection
                themes = filtered_data["Theme"].unique()
                selected_theme = st.selectbox("Select Theme", themes)
                
                if selected_theme:
                    theme_data = filtered_data[filtered_data["Theme"] == selected_theme]
                    st.write(f"Found {len(theme_data)} documents with this theme")
                    st.dataframe(theme_data)
            
        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")
            logging.error(f"Dashboard error: {e}", exc_info=True)
    else:
        st.info("Please upload theme analysis results to view the dashboard.") 

def render_topic_modeling_tab(data: pd.DataFrame):
    """Render the topic modeling tab with enhanced visualization options"""
    st.subheader("Topic Modeling Analysis")

    if data is None or len(data) == 0:
        st.warning("No data available. Please scrape or upload data first.")
        return

    with st.sidebar:
        st.subheader("Vectorization Settings")
        vectorizer_type = st.selectbox(
            "Vectorization Method",
            ["tfidf", "bm25", "weighted"],
            help="Choose how to convert text to numerical features",
        )

        # BM25 specific parameters
        if vectorizer_type == "bm25":
            k1 = st.slider(
                "k1 parameter", 0.5, 3.0, 1.5, 0.1, help="Term saturation parameter"
            )
            b = st.slider(
                "b parameter",
                0.0,
                1.0,
                0.75,
                0.05,
                help="Length normalization parameter",
            )

        # Weighted TF-IDF parameters
        elif vectorizer_type == "weighted":
            tf_scheme = st.selectbox(
                "TF Weighting Scheme",
                ["raw", "log", "binary", "augmented"],
                help="How to weight term frequencies",
            )
            idf_scheme = st.selectbox(
                "IDF Weighting Scheme",
                ["smooth", "standard", "probabilistic"],
                help="How to weight inverse document frequencies",
            )

    # Analysis parameters
    st.subheader("Analysis Parameters")
    col1, col2 = st.columns(2)

    with col1:
        num_topics = st.slider(
            "Number of Topics",
            min_value=2,
            max_value=20,
            value=5,
            help="Number of distinct topics to identify",
        )

    with col2:
        max_features = st.slider(
            "Maximum Features",
            min_value=500,
            max_value=5000,
            value=1000,
            help="Maximum number of terms to consider",
        )

    # Get vectorizer parameters
    vectorizer_params = {}
    if vectorizer_type == "bm25":
        vectorizer_params.update({"k1": k1, "b": b})
    elif vectorizer_type == "weighted":
        vectorizer_params.update({"tf_scheme": tf_scheme, "idf_scheme": idf_scheme})

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Performing topic analysis..."):
            try:
                # Create vectorizer
                vectorizer = get_vectorizer(
                    vectorizer_type=vectorizer_type,
                    max_features=max_features,
                    min_df=2,
                    max_df=0.95,
                    **vectorizer_params,
                )

                # Process text data
                docs = data["Content"].fillna("").apply(clean_text_for_modeling)

                # Create document-term matrix
                dtm = vectorizer.fit_transform(docs)
                feature_names = vectorizer.get_feature_names_out()

                # Fit LDA model
                lda = LatentDirichletAllocation(
                    n_components=num_topics, random_state=42, n_jobs=-1
                )

                doc_topics = lda.fit_transform(dtm)

                # Store model results
                st.session_state.topic_model = {
                    "lda": lda,
                    "vectorizer": vectorizer,
                    "feature_names": feature_names,
                    "doc_topics": doc_topics,
                }

                # Display results
                st.success("Topic analysis complete!")

                # Show topic words
                st.subheader("Topic Keywords")
                for idx, topic in enumerate(lda.components_):
                    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                    st.markdown(f"**Topic {idx+1}:** {', '.join(top_words)}")

                # Display network visualization
                st.subheader("Topic Similarity Network")
                display_topic_network(lda, feature_names)

                # Show topic distribution
                st.subheader("Topic Distribution")
                topic_dist = doc_topics.mean(axis=0)
                topic_df = pd.DataFrame(
                    {
                        "Topic": [f"Topic {i+1}" for i in range(num_topics)],
                        "Proportion": topic_dist,
                    }
                )

                fig = px.bar(
                    topic_df,
                    x="Topic",
                    y="Proportion",
                    title="Topic Distribution Across Documents",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Export options
                st.subheader("Export Results")
                if st.download_button(
                    "Download Topic Analysis Results",
                    data=export_topic_results(
                        lda, vectorizer, feature_names, doc_topics
                    ).encode(),
                    file_name="topic_analysis_results.json",
                    mime="application/json",
                ):
                    st.success("Results downloaded successfully!")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                logging.error(f"Topic modeling error: {e}", exc_info=True)
