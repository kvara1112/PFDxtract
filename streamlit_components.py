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
        st.session_state.reset_counter = 0
        
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

def check_app_password():
    """Check if user has entered correct password"""
    # Get password from environment variable or use default
    correct_password = os.environ.get('STREAMLIT_PASSWORD', 'amazing2')
    
    # Check if already authenticated
    if st.session_state.get('authenticated', False):
        return True
    
    # Show password input
    st.title("🔒 Access Required")
    st.markdown("Please enter the password to access the PFD Analysis Tool.")
    
    password = st.text_input("Password", type="password", key="password_input")
    
    if st.button("Login", key="login_button"):
        if password == correct_password:
            st.session_state.authenticated = True
            st.success("✅ Access granted! Redirecting...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
    
    return False

def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666; font-size: 0.8em; padding: 20px;'>
            UK Judiciary PFD Reports Analysis Tool<br>
            Developed for academic research and healthcare safety analysis
        </div>
        """,
        unsafe_allow_html=True
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
    """Render the web scraping interface"""
    st.header("UK Judiciary PFD Reports Scraper")
    
    # Create form for search parameters
    with st.form("scraping_form"):
        # Search parameters
        col1, col2 = st.columns(2)
        
        with col1:
            search_keyword = st.text_input(
                "Search keyword (optional):",
                value=st.session_state.get("search_keyword_default", ""),
                help="Leave empty to search all reports",
                key="search_keyword",
            )
            
            category = st.selectbox(
                "Category:",
                options=get_pfd_categories(),
                index=0,
                key="category",
                help="Filter by report category",
            )
            
            order = st.selectbox(
                "Sort order:",
                options=get_sort_options(),
                index=0,  # Default to date_desc
                format_func=lambda x: "Newest first" if x == "date_desc" else "Oldest first",
                key="order",
            )
        
        with col2:
            # Date filters
            st.subheader("Date Filters (Optional)")
            
            # After date
            after_col1, after_col2, after_col3 = st.columns(3)
            with after_col1:
                after_day = st.number_input("After Day", min_value=0, max_value=31, value=0, key="after_day")
            with after_col2:
                after_month = st.number_input("After Month", min_value=0, max_value=12, value=0, key="after_month")
            with after_col3:
                after_year = st.number_input("After Year", min_value=0, max_value=2030, value=0, key="after_year",
                                           help="Enter 0 to ignore date filter")
            
            # Before date
            before_col1, before_col2, before_col3 = st.columns(3)
            with before_col1:
                before_day = st.number_input("Before Day", min_value=0, max_value=31, value=0, key="before_day")
            with before_col2:
                before_month = st.number_input("Before Month", min_value=0, max_value=12, value=0, key="before_month")
            with before_col3:
                before_year = st.number_input("Before Year", min_value=0, max_value=2030, value=0, key="before_year",
                                            help="Enter 0 to ignore date filter")
        
        # Preview search results
        if st.form_submit_button("Preview Search Results"):
            try:
                # Create date filter strings
                after_date = None
                if after_day > 0 and after_month > 0 and after_year >= 2000:
                    after_date = f"{after_day}-{after_month}-{after_year}"
                    
                before_date = None
                if before_day > 0 and before_month > 0 and before_year >= 2000:
                    before_date = f"{before_day}-{before_month}-{before_year}"
                
                # Convert category to slug
                category_slug = ""
                if category:
                    category_slug = (
                        category.lower()
                        .replace(" ", "-")
                        .replace("&", "and")
                        .replace("--", "-")
                        .strip("-")
                    )
                
                # Create preview URL
                base_url = "https://www.judiciary.uk/prevention-of-future-death-reports/"
                preview_url = construct_search_url(
                    base_url=base_url,
                    keyword=search_keyword,
                    category=category,
                    category_slug=category_slug,
                    after_date=after_date,
                    before_date=before_date,
                    order=order,
                )
                
                # Get total pages
                total_pages, total_results = get_total_pages(preview_url)
                if total_pages > 0:
                    st.info(f"Search found {total_pages} pages with {total_results} results")
                    st.session_state["total_pages_preview"] = total_pages
                    
                    # Show estimated time
                    est_time = estimate_scraping_time(1, min(total_pages, 10))
                    st.info(f"Estimated time for 10 pages: {est_time}")
                else:
                    st.warning("No results found for this search")
                    st.session_state["total_pages_preview"] = 0
                    
            except Exception as e:
                st.error(f"Error checking search results: {str(e)}")
                st.session_state["total_pages_preview"] = 0
        
        # Page settings
        st.subheader("Scraping Settings")
        row3_col1, row3_col2 = st.columns(2)
        row4_col1, row4_col2 = st.columns(2)
        
        with row3_col1:
            start_page = st.number_input(
                "Start page:",
                min_value=1,
                value=1,
                key="start_page",
                help="First page to scrape (minimum 1)",
            )
        
        with row3_col2:
            end_page = st.number_input(
                "End page (0 for all):",
                min_value=0,
                value=10,
                key="end_page",
                help="Last page to scrape (0 for all pages)",
            )
        
        with row4_col1:
            auto_save_batches = st.checkbox(
                "Auto-save batches",
                value=True,
                key="auto_save_batches",
                help="Automatically save results in batches",
            )
        
        with row4_col2:
            batch_size = st.number_input(
                "Pages per batch:",
                min_value=1,
                max_value=10,
                value=5,
                key="batch_size",
                help="Number of pages to process before saving",
            )
        
        # Action buttons
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            submitted = st.form_submit_button("🔍 Start Scraping")
        with button_col2:
            stop_scraping = st.form_submit_button("⏹️ Stop Scraping")
    
    # Handle stop scraping
    if stop_scraping:
        st.session_state.stop_scraping = True
        st.warning("Scraping will be stopped after the current page completes...")
        return
    
    # Handle scraping
    if submitted:
        try:
            # Initialize stop flag
            st.session_state.stop_scraping = False
            
            # Create date filters
            after_date = None
            if after_day > 0 and after_month > 0 and after_year >= 2000:
                after_date = f"{after_day}-{after_month}-{after_year}"
                
            before_date = None
            if before_day > 0 and before_month > 0 and before_year >= 2000:
                before_date = f"{before_day}-{before_month}-{before_year}"
            
            # Convert end_page=0 to None (all pages)
            end_page_val = None if end_page == 0 else end_page
            
            # Perform scraping
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
                
                st.success(f"Successfully scraped {len(reports)} reports!")
                
                # Show preview
                st.subheader("Scraped Data Preview")
                st.dataframe(df.head())
                
        except Exception as e:
            st.error(f"Scraping error: {str(e)}")
            logging.error(f"Scraping error: {e}", exc_info=True)

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