import pandas as pd
import logging
import time
import os
import io
import zipfile
import random
import string
import math
from datetime import datetime
from typing import Dict, List, Optional
from openpyxl.utils import get_column_letter
import pytz
import plotly.express as px
from sklearn.decomposition import LatentDirichletAllocation
import json
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import streamlit as st

# Import our modules
from core_utils import (
    process_scraped_data, 
    clean_text_for_modeling, 
    export_topic_results,
    export_to_excel,
    filter_by_categories,
    save_dashboard_images_as_zip,
)
from web_scraping import (
    get_pfd_categories,
    scrape_pfd_reports,
    get_total_pages,
    construct_search_url,
)
from vectorizer_models import get_vectorizer
from bert_analysis import BERTResultsAnalyzer, ThemeAnalyzer
from visualization import (
    plot_category_distribution,
    plot_coroner_areas,
    plot_timeline,
    plot_monthly_distribution,
    plot_yearly_comparison,
    display_topic_network,
    improved_truncate_text,
    analyze_data_quality
)
from file_prep import ( 
    render_filter_data_tab,
    show_export_options
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

    

    # Password input as a form to accept enter key - Jamie L
    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if password == st.secrets.get("app_password"):
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
    """Render the BERT file merger tab in the Streamlit app with custom initialization."""
    # Create an instance of the analyzer
    analyzer = BERTResultsAnalyzer()
    
    # Add tabs for Merger and Filter functionality
    merger_tab, filter_tab = st.tabs(["Merge & Process Files", "Filter & Explore Data"])
    
    with merger_tab:
        # Skip the standard render_analyzer_ui and call the file upload directly
        analyzer._render_multiple_file_upload()
    
    with filter_tab:
        render_filter_data_tab()

def render_bert_analysis_tab(data: pd.DataFrame = None):
    """Modified render_bert_analysis_tab function to include framework selection and custom framework upload"""
    
    # Ensure the bert_results dictionary exists in session state
    if "bert_results" not in st.session_state:
        st.session_state.bert_results = {}
    
    # Track if BERT model is initialized
    if "bert_initialized" not in st.session_state:
        st.session_state.bert_initialized = False
    
    # Initialize custom frameworks dictionary if not present
    if "custom_frameworks" not in st.session_state:
        st.session_state.custom_frameworks = {}
        
    # Safer initialization with validation
    if "selected_frameworks" not in st.session_state:
        # Only include frameworks that actually exist
        default_frameworks = ["I-SIRch", "House of Commons", "Extended Analysis"]
        st.session_state.selected_frameworks = default_frameworks
    else:
        # Validate existing selections against available options
        available_frameworks = ["I-SIRch", "House of Commons", "Extended Analysis"] + list(st.session_state.get("custom_frameworks", {}).keys())
        st.session_state.selected_frameworks = [f for f in st.session_state.selected_frameworks if f in available_frameworks]

    # File upload section
    st.subheader("Upload Data")
    reset_counter = st.session_state.get("reset_counter", 0)
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file for BERT Analysis",
        type=["csv", "xlsx"],
        help="Upload a file with reports for theme analysis",
        key="bert_file_uploader",
    )

    # If a file is uploaded, process it
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                uploaded_data = pd.read_csv(uploaded_file)
            else:
                uploaded_data = pd.read_excel(uploaded_file)

            # Process the uploaded data
            uploaded_data = process_scraped_data(uploaded_data)

            # Update the data reference
            data = uploaded_data

            st.success("File uploaded and processed successfully!")
                
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            return

    # Check if data is available
    if data is None or len(data) == 0:
        st.warning(
            "No data available. Please upload a file or ensure existing data is loaded."
        )
        return

    # Framework selection section
    st.subheader("Select Frameworks")
    
    # Create columns for the framework selection and custom framework upload
    frame_col1, frame_col2 = st.columns([2, 1])
    
    with frame_col1:
        # Get all available framework options
        available_frameworks = ["I-SIRch", "House of Commons", "Extended Analysis"]
        if "custom_frameworks" in st.session_state:
            available_frameworks.extend(list(st.session_state.custom_frameworks.keys()))
        
        # Predefined framework selection - use a unique key
        st.session_state.selected_frameworks = st.multiselect(
            "Choose Frameworks to Use",
            options=available_frameworks,
            default=st.session_state.selected_frameworks,
            help="Select which conceptual frameworks to use for theme analysis",
            key=f"framework_select_{reset_counter}"
        )
    
    with frame_col2:
        # Custom framework upload
        custom_framework_file = st.file_uploader(
            "Upload Custom Framework",
            type=["json", "txt"],
            help="Upload a JSON file containing custom framework definitions",
            key=f"custom_framework_uploader_{reset_counter}"
        )
        
        if custom_framework_file is not None:
            try:
                # Read framework definition
                custom_framework_content = custom_framework_file.read().decode("utf-8")
                custom_framework_data = json.loads(custom_framework_content)
                
                # Validate framework structure
                if isinstance(custom_framework_data, list) and all(isinstance(item, dict) and "name" in item and "keywords" in item for item in custom_framework_data):
                    # Framework name input
                    custom_framework_name = st.text_input(
                        "Custom Framework Name", 
                        f"Custom Framework {len(st.session_state.custom_frameworks) + 1}",
                        key=f"custom_framework_name_{reset_counter}"
                    )
                    
                    # Add button for the custom framework
                    if st.button("Add Custom Framework", key=f"add_custom_framework_{reset_counter}"):
                        # Check if name already exists
                        if custom_framework_name in st.session_state.custom_frameworks:
                            st.warning(f"A framework with the name '{custom_framework_name}' already exists. Please choose a different name.")
                        else:
                            # Add to session state
                            st.session_state.custom_frameworks[custom_framework_name] = custom_framework_data
                            
                            # Add to selected frameworks if not already there
                            if custom_framework_name not in st.session_state.selected_frameworks:
                                st.session_state.selected_frameworks.append(custom_framework_name)
                            
                            st.success(f"Custom framework '{custom_framework_name}' with {len(custom_framework_data)} themes added successfully")
                            st.rerun()  # Refresh to update UI
                else:
                    st.error("Invalid framework format. Each item must have 'name' and 'keywords' fields.")
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your file.")
            except Exception as e:
                st.error(f"Error processing custom framework: {str(e)}")
                logging.error(f"Custom framework error: {e}", exc_info=True)
    
    # Display currently loaded custom frameworks
    if "custom_frameworks" in st.session_state and st.session_state.custom_frameworks:
        st.subheader("Loaded Custom Frameworks")
        for name, framework in st.session_state.custom_frameworks.items():
            with st.expander(f"{name} ({len(framework)} themes)"):
                # Display the first few themes as an example
                for i, theme in enumerate(framework[:5]):
                    st.markdown(f"**{theme['name']}**: {', '.join(theme['keywords'][:5])}...")
                    if i >= 4 and len(framework) > 5:
                        st.markdown(f"*... and {len(framework) - 5} more themes*")
                        break
                
                # Add option to remove this framework
                if st.button("Remove Framework", key=f"remove_{name}_{reset_counter}"):
                    del st.session_state.custom_frameworks[name]
                    if name in st.session_state.selected_frameworks:
                        st.session_state.selected_frameworks.remove(name)
                    st.success(f"Removed framework '{name}'")
                    st.rerun()  # Refresh to update UI

    # Column selection for analysis
    st.subheader("Select Analysis Column")

    # Find text columns (object/string type)
    text_columns = data.select_dtypes(include=["object"]).columns.tolist()

    # If no text columns found
    if not text_columns:
        st.error("No text columns found in the dataset.")
        return

    # Column selection with dropdown
    content_column = st.selectbox(
        "Choose the column to analyse:",
        options=text_columns,
        index=text_columns.index("Content") if "Content" in text_columns else 0,
        help="Select the column containing the text you want to analyse",
        key="bert_content_column",
    )

    # Filtering options
    st.subheader("Select Documents to Analyse")

    # Option to select all or specific records
    analysis_type = st.radio(
        "Analysis Type",
        ["All Reports", "Selected Reports"],
        horizontal=True,
        key="bert_analysis_type",
    )

    if analysis_type == "Selected Reports":
        # Multi-select for reports
        selected_indices = st.multiselect(
            "Choose specific reports to analyse",
            options=list(range(len(data))),
            format_func=lambda x: f"{data.iloc[x]['Title']} ({data.iloc[x]['date_of_report'].strftime('%d/%m/%Y') if pd.notna(data.iloc[x]['date_of_report']) else 'No date'})",
            key="bert_selected_indices",
        )
        selected_data = data.iloc[selected_indices] if selected_indices else None
    else:
        selected_data = data

    # Analysis parameters
    st.subheader("Analysis Parameters")
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.65,
        step=0.05,
        help="Minimum similarity score for theme detection (higher = more strict)",
        key="bert_similarity_threshold",
    )

    # Analysis button
    run_analysis = st.button(
        "Run Analysis", type="primary", key="bert_run_analysis"
    )

    # Run analysis if button is clicked
    if run_analysis:
        with st.spinner("Performing Theme Analysis..."):
            try:
                # Validate data selection
                if selected_data is None or len(selected_data) == 0:
                    st.warning("No documents selected for analysis.")
                    return

                # Initialize the theme analyzer (with loading message in a spinner)
                with st.spinner("Loading annotation model and tokenizer..."):
                    # Initialize the analyzer
                    theme_analyzer = ThemeAnalyzer(
                        model_name="emilyalsentzer/Bio_ClinicalBERT"
                    )
                    
                    # Mark as initialized
                    st.session_state.bert_initialized = True
                
                # Set custom configuration
                theme_analyzer.config[
                    "base_similarity_threshold"
                ] = similarity_threshold
                
                # Filter frameworks based on user selection
                filtered_frameworks = {}
                
                # Add selected built-in frameworks
                for framework in st.session_state.selected_frameworks:
                    if framework == "I-SIRch":
                        filtered_frameworks["I-SIRch"] = theme_analyzer._get_isirch_framework()
                    elif framework == "House of Commons":
                        filtered_frameworks["House of Commons"] = theme_analyzer._get_house_of_commons_themes()
                    elif framework == "Extended Analysis":
                        filtered_frameworks["Extended Analysis"] = theme_analyzer._get_extended_themes()
                    elif framework in st.session_state.custom_frameworks:
                        # Add custom framework
                        filtered_frameworks[framework] = st.session_state.custom_frameworks[framework]
                
                # Set the filtered frameworks
                theme_analyzer.frameworks = filtered_frameworks
                
                # If no frameworks selected, show error
                if not filtered_frameworks:
                    st.error("Please select at least one framework for analysis.")
                    return

                # Use the enhanced create_detailed_results method
                (
                    results_df,
                    highlighted_texts,
                ) = theme_analyzer.create_detailed_results(
                    selected_data, content_column=content_column
                )

                # Save results to session state to ensure persistence
                st.session_state.bert_results["results_df"] = results_df
                st.session_state.bert_results["highlighted_texts"] = highlighted_texts

                st.success(f"Analysis complete using {len(filtered_frameworks)} frameworks!")

            except Exception as e:
                st.error(f"Error during annotation analysis: {str(e)}")
                logging.error(f"Annotation analysis error: {e}", exc_info=True)

    # Display results if they exist
    if "bert_results" in st.session_state and st.session_state.bert_results.get("results_df") is not None:
        results_df = st.session_state.bert_results["results_df"]
        
        # Summary stats
        st.subheader("Results")
        
        # Show framework distribution
        if "Framework" in results_df.columns:
            framework_counts = results_df["Framework"].value_counts()
            
            # Create columns for framework distribution metrics
            framework_cols = st.columns(len(framework_counts))
            
            for i, (framework, count) in enumerate(framework_counts.items()):
                with framework_cols[i]:
                    st.metric(framework, count, help=f"Number of theme identifications from {framework} framework")
        
        st.write(f"Total Theme Identifications: {len(results_df)}")
        
        # Clean up the results DataFrame to display only the essential columns
        display_cols = ["Record ID", "Title", "Framework", "Theme", "Confidence", "Combined Score", "Matched Keywords"]
        
        # Add metadata columns if available
        for col in ["coroner_name", "coroner_area", "year", "date_of_report"]:
            if col in results_df.columns:
                display_cols.append(col)
        
        # Add matched sentences at the end
        if "Matched Sentences" in results_df.columns:
            display_cols.append("Matched Sentences")
        
        # Create the display DataFrame with only existing columns
        valid_cols = [col for col in display_cols if col in results_df.columns]
        clean_df = results_df[valid_cols].copy()
        
        # Display the results table
        st.dataframe(
            clean_df,
            use_container_width=True,
            column_config={
                "Title": st.column_config.TextColumn("Document Title"),
                "Framework": st.column_config.TextColumn("Framework"),
                "Theme": st.column_config.TextColumn("Theme"),
                "Confidence": st.column_config.TextColumn("Confidence"),
                "Combined Score": st.column_config.NumberColumn("Score", format="%.3f"),
                "Matched Keywords": st.column_config.TextColumn("Keywords"),
                "Matched Sentences": st.column_config.TextColumn("Matched Sentences"),
                "coroner_name": st.column_config.TextColumn("Coroner Name"),
                "coroner_area": st.column_config.TextColumn("Coroner Area"),
                "year": st.column_config.NumberColumn("Year"),
                "date_of_report": st.column_config.DateColumn("Date of Report", format="DD/MM/YYYY")
            }
        )
        
        # Add download options
        st.subheader("Export Results")
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create columns for download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel download button using the enhanced export_to_excel function
            excel_data = export_to_excel(clean_df)
            st.download_button(
                "📥 Download Results Table",
                data=excel_data,
                file_name=f"annotated_theme_analysis_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="bert_excel_download",
            )
        
        with col2:
            # Always regenerate HTML report when results are available
            if "results_df" in st.session_state.bert_results and "highlighted_texts" in st.session_state.bert_results:
                # Generate fresh HTML report based on current results
                theme_analyzer = ThemeAnalyzer()
                
                # Set custom frameworks if they exist
                if st.session_state.custom_frameworks:
                    for name, framework in st.session_state.custom_frameworks.items():
                        if name in st.session_state.selected_frameworks:
                            theme_analyzer.frameworks[name] = framework
                
                html_content = theme_analyzer._create_integrated_html_for_pdf(
                    results_df, st.session_state.bert_results["highlighted_texts"]
                )
                html_filename = f"theme_analysis_report_{timestamp}.html"
                
                with open(html_filename, "w", encoding="utf-8") as f:
                    f.write(html_content)
                    
                st.session_state.bert_results["html_filename"] = html_filename
                
                # Provide download button for fresh HTML
                with open(html_filename, "rb") as f:
                    html_data = f.read()
                
                st.download_button(
                    "📄 Download Annotated Reports (HTML)",
                    data=html_data,
                    file_name=os.path.basename(html_filename),
                    mime="text/html",
                    key="bert_html_download",
                )
            else:
                st.warning("HTML report not available")

def render_bert_analysis_tabworking(data: pd.DataFrame = None):
    """Modified render_bert_analysis_tab function to include enhanced metadata in results"""
    
    # Ensure the bert_results dictionary exists in session state
    if "bert_results" not in st.session_state:
        st.session_state.bert_results = {}
    
    # Track if BERT model is initialized
    if "bert_initialized" not in st.session_state:
        st.session_state.bert_initialized = False

    # File upload section
    st.subheader("Upload Data")
    reset_counter = st.session_state.get("reset_counter", 0)
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file for BERT Analysis",
        type=["csv", "xlsx"],
        help="Upload a file with reports for theme analysis",
        key="bert_file_uploader",
    )

    # If a file is uploaded, process it
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                uploaded_data = pd.read_csv(uploaded_file)
            else:
                uploaded_data = pd.read_excel(uploaded_file)

            # Process the uploaded data
            uploaded_data = process_scraped_data(uploaded_data)

            # Update the data reference
            data = uploaded_data

            st.success("File uploaded and processed successfully!")
                
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            return

    # Check if data is available
    if data is None or len(data) == 0:
        st.warning(
            "No data available. Please upload a file or ensure existing data is loaded."
        )
        return

    # Column selection for analysis
    st.subheader("Select Analysis Column")

    # Find text columns (object/string type)
    text_columns = data.select_dtypes(include=["object"]).columns.tolist()

    # If no text columns found
    if not text_columns:
        st.error("No text columns found in the dataset.")
        return

    # Column selection with dropdown
    content_column = st.selectbox(
        "Choose the column to analyse:",
        options=text_columns,
        index=text_columns.index("Content") if "Content" in text_columns else 0,
        help="Select the column containing the text you want to analyse",
        key="bert_content_column",
    )

    # Filtering options
    st.subheader("Select Documents to Analyse")

    # Option to select all or specific records
    analysis_type = st.radio(
        "Analysis Type",
        ["All Reports", "Selected Reports"],
        horizontal=True,
        key="bert_analysis_type",
    )

    if analysis_type == "Selected Reports":
        # Multi-select for reports
        selected_indices = st.multiselect(
            "Choose specific reports to analyse",
            options=list(range(len(data))),
            format_func=lambda x: f"{data.iloc[x]['Title']} ({data.iloc[x]['date_of_report'].strftime('%d/%m/%Y') if pd.notna(data.iloc[x]['date_of_report']) else 'No date'})",
            key="bert_selected_indices",
        )
        selected_data = data.iloc[selected_indices] if selected_indices else None
    else:
        selected_data = data

    # Analysis parameters
    st.subheader("Analysis Parameters")
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.65,
        step=0.05,
        help="Minimum similarity score for theme detection (higher = more strict)",
        key="bert_similarity_threshold",
    )

    # Analysis button
    run_analysis = st.button(
        "Run Analysis", type="primary", key="bert_run_analysis"
    )

    # Run analysis if button is clicked
    if run_analysis:
        with st.spinner("Performing Theme Analysis..."):
            try:
                # Validate data selection
                if selected_data is None or len(selected_data) == 0:
                    st.warning("No documents selected for analysis.")
                    return

                # Initialize the theme analyzer (with loading message in a spinner)
                with st.spinner("Loading annotation model and tokenizer..."):
                    # Initialize the analyzer
                    theme_analyzer = ThemeAnalyzer(
                        model_name="emilyalsentzer/Bio_ClinicalBERT"
                    )
                    
                    # Mark as initialized
                    st.session_state.bert_initialized = True
                
                # Set custom configuration
                theme_analyzer.config[
                    "base_similarity_threshold"
                ] = similarity_threshold

                # Use the enhanced create_detailed_results method
                (
                    results_df,
                    highlighted_texts,
                ) = theme_analyzer.create_detailed_results(
                    selected_data, content_column=content_column
                )

                # Save results to session state to ensure persistence
                st.session_state.bert_results["results_df"] = results_df
                st.session_state.bert_results["highlighted_texts"] = highlighted_texts

                st.success("Analysis complete!")

            except Exception as e:
                st.error(f"Error during annotation analysis: {str(e)}")
                logging.error(f"Annotation analysis error: {e}", exc_info=True)

    # Display results if they exist
    if "bert_results" in st.session_state and st.session_state.bert_results.get("results_df") is not None:
        results_df = st.session_state.bert_results["results_df"]
        
        # Summary stats
        st.subheader("Results")
        st.write(f"Total Theme Identifications: {len(results_df)}")
        
        # Clean up the results DataFrame to display only the essential columns
        display_cols = ["Record ID", "Title", "Framework", "Theme", "Confidence", "Combined Score", "Matched Keywords"]
        
        # Add metadata columns if available
        for col in ["coroner_name", "coroner_area", "year", "date_of_report"]:
            if col in results_df.columns:
                display_cols.append(col)
        
        # Add matched sentences at the end
        if "Matched Sentences" in results_df.columns:
            display_cols.append("Matched Sentences")
        
        # Create the display DataFrame with only existing columns
        valid_cols = [col for col in display_cols if col in results_df.columns]
        clean_df = results_df[valid_cols].copy()
        
        # Display the results table
        st.dataframe(
            clean_df,
            use_container_width=True,
            column_config={
                "Title": st.column_config.TextColumn("Document Title"),
                "Framework": st.column_config.TextColumn("Framework"),
                "Theme": st.column_config.TextColumn("Theme"),
                "Confidence": st.column_config.TextColumn("Confidence"),
                "Combined Score": st.column_config.NumberColumn("Score", format="%.3f"),
                "Matched Keywords": st.column_config.TextColumn("Keywords"),
                "Matched Sentences": st.column_config.TextColumn("Matched Sentences"),
                "coroner_name": st.column_config.TextColumn("Coroner Name"),
                "coroner_area": st.column_config.TextColumn("Coroner Area"),
                "year": st.column_config.NumberColumn("Year"),
                "date_of_report": st.column_config.DateColumn("Date of Report", format="DD/MM/YYYY")
            }
        )
        
        # Add download options
        st.subheader("Export Results")
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create columns for download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel download button using the enhanced export_to_excel function
            excel_data = export_to_excel(clean_df)
            st.download_button(
                "📥 Download Results Table",
                data=excel_data,
                file_name=f"annotated_theme_analysis_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="bert_excel_download",
            )
        
        with col2:
            # Always regenerate HTML report when results are available
            if "results_df" in st.session_state.bert_results and "highlighted_texts" in st.session_state.bert_results:
                # Generate fresh HTML report based on current results
                theme_analyzer = ThemeAnalyzer()
                html_content = theme_analyzer._create_integrated_html_for_pdf(
                    results_df, st.session_state.bert_results["highlighted_texts"]
                )
                html_filename = f"theme_analysis_report_{timestamp}.html"
                
                with open(html_filename, "w", encoding="utf-8") as f:
                    f.write(html_content)
                    
                st.session_state.bert_results["html_filename"] = html_filename
                
                # Provide download button for fresh HTML
                with open(html_filename, "rb") as f:
                    html_data = f.read()
                
                st.download_button(
                    "📄 Download Annotated Reports (HTML)",
                    data=html_data,
                    file_name=os.path.basename(html_filename),
                    mime="text/html",
                    key="bert_html_download",
                )
            else:
                st.warning("HTML report not available")


def render_theme_analysis_dashboard(data: pd.DataFrame = None):
    """
    Render a comprehensive dashboard for analyzing themes by various metadata fields
    
    Args:
        data: Optional DataFrame containing theme analysis results
    """
    #st.title("Theme Analysis Dashboard")
    
    # Check for existing data in session state
    if data is None:
        if "dashboard_data" in st.session_state:
            data = st.session_state.dashboard_data
    
    # File upload section with persistent state
    upload_key = "theme_analysis_dashboard_uploader"
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file for Dashboard Analysis",
        type=["csv", "xlsx"],
        key=upload_key
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Load the file based on its type
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Process the data to ensure it's clean
            data = process_scraped_data(data)
            
            # Store in session state
            st.session_state.dashboard_data = data
            
            st.success(f"File uploaded successfully! Found {len(data)} records.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return
    
    # If no data is available after upload
    if data is None or len(data) == 0:
        with st.expander("💡 How to get theme analysis data?"):
            st.markdown("""
            #### To get theme analysis data:
        
            1. **Upload Existing Results**
               - Use the file uploader above to load previously saved theme analysis results
            
            2. **Run New Theme Analysis**
               - Go to the 'Concept Annotation' tab 
               - Upload your merged PFD reports file
               - Run a new theme analysis
            """)
            
        return  # Exit the function if no data
    
    # Validate required columns
    required_cols = ["Framework", "Theme"]
    recommended_cols = ["coroner_area", "coroner_name", "year"]
    
    missing_required = [col for col in required_cols if col not in data.columns]
    missing_recommended = [col for col in recommended_cols if col not in data.columns]
    
    if missing_required:
        st.error(f"Missing required columns: {', '.join(missing_required)}")
        return
    
    if missing_recommended:
        st.warning(f"Some recommended columns are missing: {', '.join(missing_recommended)}")
    
    # Data Overview
    st.subheader("Data Overview")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Total Theme Identifications", len(data))
    with metrics_col2:
        st.metric("Unique Themes", data["Theme"].nunique())
    
    with metrics_col3:
        if "coroner_area" in data.columns and not data["coroner_area"].isna().all():
            st.metric("Coroner Areas", data["coroner_area"].nunique())
        else:
            st.metric("Coroner Areas", "N/A")
    
    with metrics_col4:
        if "year" in data.columns and not data["year"].isna().all():
            years_count = data["year"].dropna().nunique()
            year_text = f"{years_count}" if years_count > 0 else "N/A"
            st.metric("Years Covered", year_text)
        else:
            st.metric("Years Covered", "N/A")
            
    results_df = data.copy() if data is not None else pd.DataFrame()
    
    #
    
    # Sidebar filters
    # Find the sidebar filter section in the render_theme_analysis_dashboard function 
    # and replace it with this improved version:
    
    # Sidebar filters
    st.sidebar.header("Dashboard Filters")
    
    # Framework filter
    frameworks = ["All"] + sorted(results_df["Framework"].unique().tolist())
    selected_framework = st.sidebar.selectbox("Filter by Framework", frameworks)
    
    # Year filter - Modified to handle single year selection properly
    years = sorted(results_df["year"].dropna().unique().tolist())
    if years:
        min_year, max_year = min(years), max(years)
        
        # If only one year available, provide a checkbox instead of slider
        if min_year == max_year:
            include_year = st.sidebar.checkbox(f"Include year {min_year}", value=True)
            if include_year:
                selected_years = (min_year, max_year)
            else:
                selected_years = None
        else:
            # For multiple years, keep using the slider
            selected_years = st.sidebar.slider(
                "Year Range", min_year, max_year, (min_year, max_year)
            )
    else:
        selected_years = None
    
    # Coroner area filter - MODIFIED: Multi-select instead of single select
    areas = sorted(results_df["coroner_area"].dropna().unique().tolist())
    area_options = ["All Areas"] + areas
    # Default to "All Areas" if no specific selection
    area_filter_type = st.sidebar.radio("Coroner Area Filter Type", ["All Areas", "Select Specific Areas"])
    if area_filter_type == "All Areas":
        selected_areas = areas  # Include all areas
    else:
        # Multi-select for specific areas
        selected_areas = st.sidebar.multiselect(
            "Select Areas", 
            options=areas,
            default=None,
            help="Select one or more specific coroner areas to include"
        )
        # If nothing selected, default to all areas
        if not selected_areas:
            st.sidebar.warning("No areas selected. Showing all areas.")
            selected_areas = areas
    
    # Coroner name filter - MODIFIED: Multi-select instead of single select
    names = sorted(results_df["coroner_name"].dropna().unique().tolist())
    name_filter_type = st.sidebar.radio("Coroner Name Filter Type", ["All Coroners", "Select Specific Coroners"])
    if name_filter_type == "All Coroners":
        selected_names = names  # Include all coroner names
    else:
        # Multi-select for specific coroner names
        selected_names = st.sidebar.multiselect(
            "Select Coroners", 
            options=names,
            default=None,
            help="Select one or more specific coroners to include"
        )
        # If nothing selected, default to all names
        if not selected_names:
            st.sidebar.warning("No coroners selected. Showing all coroners.")
            selected_names = names
    
    # Number of top themes to display
    top_n_themes = st.sidebar.slider("Number of Top Themes", 5, 20, 10)
    
    # Confidence filter - MODIFIED: Multi-select instead of minimum
    confidence_levels = ["High", "Medium", "Low"]
    confidence_filter_type = st.sidebar.radio("Confidence Filter Type", ["All Confidence Levels", "Select Specific Levels"])
    if confidence_filter_type == "All Confidence Levels":
        selected_confidence_levels = confidence_levels  # Include all confidence levels
    else:
        # Multi-select for specific confidence levels
        selected_confidence_levels = st.sidebar.multiselect(
            "Select Confidence Levels", 
            options=confidence_levels,
            default=["High", "Medium"],  # Default to high and medium
            help="Select one or more confidence levels to include"
        )
        # If nothing selected, default to all confidence levels
        if not selected_confidence_levels:
            st.sidebar.warning("No confidence levels selected. Showing all levels.")
            selected_confidence_levels = confidence_levels
    
    # Apply filters - UPDATED to handle new multi-select filters
    filtered_df = results_df.copy()
    
    if selected_framework != "All":
        filtered_df = filtered_df[filtered_df["Framework"] == selected_framework]
    
    if selected_years:
        # Handle edge case of single year (where both values are the same)
        if selected_years[0] == selected_years[1]:
            filtered_df = filtered_df[filtered_df["year"] == selected_years[0]]
        else:
            filtered_df = filtered_df[(filtered_df["year"] >= selected_years[0]) & 
                                    (filtered_df["year"] <= selected_years[1])]
    
    # Apply multi-select area filter
    filtered_df = filtered_df[filtered_df["coroner_area"].isin(selected_areas)]
    
    # Apply multi-select coroner name filter
    filtered_df = filtered_df[filtered_df["coroner_name"].isin(selected_names)]
    
    # Apply multi-select confidence level filter
    filtered_df = filtered_df[filtered_df["Confidence"].isin(selected_confidence_levels)]
    
    # Display filter summary
    active_filters = []
    if selected_framework != "All":
        active_filters.append(f"Framework: {selected_framework}")
    if selected_years:
        if selected_years[0] == selected_years[1]:
            active_filters.append(f"Year: {selected_years[0]}")
        else:
            active_filters.append(f"Years: {selected_years[0]}-{selected_years[1]}")
    if area_filter_type == "Select Specific Areas" and selected_areas:
        if len(selected_areas) <= 3:
            active_filters.append(f"Areas: {', '.join(selected_areas)}")
        else:
            active_filters.append(f"Areas: {len(selected_areas)} selected")
    if name_filter_type == "Select Specific Coroners" and selected_names:
        if len(selected_names) <= 3:
            active_filters.append(f"Coroners: {', '.join(selected_names)}")
        else:
            active_filters.append(f"Coroners: {len(selected_names)} selected")
    if confidence_filter_type == "Select Specific Levels" and selected_confidence_levels:
        active_filters.append(f"Confidence: {', '.join(selected_confidence_levels)}")
    
    if active_filters:
        st.info("Active filters: " + " | ".join(active_filters))
    
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters. Please adjust your filters.")
        return
        
    # Continue with the rest of the dashboard code...
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Framework Heatmap", 
        "Theme Distribution", 
        "Temporal Analysis", 
        "Area Comparison",
        "Correlation Analysis"
    ])
    
    # === TAB 1: FRAMEWORK HEATMAP ===
    with tab1:
        st.subheader("Framework Theme Heatmap by Year")
        
        if "year" not in filtered_df.columns or filtered_df["year"].isna().all():
            st.warning("No year data available for temporal analysis.")
        else:
            # Handle special case where we only have one year
            if filtered_df["year"].nunique() == 1:
                st.info(f"Showing data for year {filtered_df['year'].iloc[0]}")
                
                # Create a simplified categorical count visualization for single year
                theme_counts = filtered_df.groupby(['Framework', 'Theme']).size().reset_index(name='Count')
                
                # Sort by framework and count
                theme_counts = theme_counts.sort_values(['Framework', 'Count'], ascending=[True, False])
                
                # Process theme names for better display using improved function
                theme_counts['Display_Theme'] = theme_counts['Theme'].apply(
                    lambda x: improved_truncate_text(x, max_length=40)
                )
                
                # Display as a horizontal bar chart grouped by framework
                fig = px.bar(
                    theme_counts,
                    y='Display_Theme',  # Use formatted theme names
                    x='Count',
                    color='Framework',
                    title=f"Theme Distribution for Year {filtered_df['year'].iloc[0]}",
                    height=max(500, len(theme_counts) * 30),
                    color_discrete_map={
                        "I-SIRch": "orange",
                        "House of Commons": "royalblue",
                        "Extended Analysis": "firebrick"
                    }
                )
                
                fig.update_layout(
                    xaxis_title="Number of Reports",
                    yaxis_title="Theme",
                    yaxis={'categoryorder': 'total ascending'},
                    font=dict(family="Arial, sans-serif", color="white"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=250, r=40, t=80, b=60)  # Increased left margin for theme labels
                )
                
                # Update axes for dark mode
                fig.update_xaxes(
                    title_font=dict(color="white"),
                    tickfont=dict(color="white"),
                    gridcolor="rgba(255,255,255,0.1)"
                )
                
                fig.update_yaxes(
                    title_font=dict(color="white"),
                    tickfont=dict(color="white"),
                    automargin=True  # Enable automargin to ensure labels fit
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Regular heatmap code for multiple years
                # Create combined framework:theme field
                filtered_df['Framework_Theme'] = filtered_df['Framework'] + ': ' + filtered_df['Theme']
                
                # Count reports per year (for denominator)
                # Assuming Record ID is the unique identifier for reports
                id_column = 'Record ID' if 'Record ID' in filtered_df.columns else filtered_df.columns[0]
                reports_per_year = filtered_df.groupby('year')[id_column].nunique()
                
                # Count unique report IDs per theme per year
                counts = filtered_df.groupby(['year', 'Framework', 'Framework_Theme'])[id_column].nunique().reset_index()
                counts.columns = ['year', 'Framework', 'Framework_Theme', 'Count']
                
                # Calculate percentages
                counts['Total'] = counts['year'].map(reports_per_year)
                counts['Percentage'] = (counts['Count'] / counts['Total'] * 100).round(1)
                
                # Get frameworks in the filtered data
                frameworks_present = filtered_df['Framework'].unique()
                
                # Get top themes by framework (5 per framework)
                top_themes = []
                for framework in frameworks_present:
                    framework_counts = counts[counts['Framework'] == framework]
                    theme_totals = framework_counts.groupby('Framework_Theme')['Count'].sum().sort_values(ascending=False)
                    top_themes.extend(theme_totals.head(5).index.tolist())
                
                # Filter to top themes
                counts = counts[counts['Framework_Theme'].isin(top_themes)]
                
                # Create pivot table for heatmap
                pivot = counts.pivot_table(
                    index='Framework_Theme',
                    columns='year',
                    values='Percentage',
                    fill_value=0
                )
                
                # Create pivot for counts
                count_pivot = counts.pivot_table(
                    index='Framework_Theme',
                    columns='year',
                    values='Count',
                    fill_value=0
                )
                
                # Sort by framework then by total count
                theme_totals = counts.groupby('Framework_Theme')['Count'].sum()
                theme_frameworks = {theme: theme.split(':')[0] for theme in theme_totals.index}
                
                # Sort first by framework, then by count within framework
                sorted_themes = sorted(
                    theme_totals.index,
                    key=lambda x: (theme_frameworks[x], -theme_totals[x])
                )
                
                # Apply the sort order
                pivot = pivot.reindex(sorted_themes)
                count_pivot = count_pivot.reindex(sorted_themes)
                
                # Create color mapping for frameworks
                framework_colors = {
                    "I-SIRch": "orange",
                    "House of Commons": "royalblue",
                    "Extended Analysis": "firebrick"
                }
                
                # Default colors for any frameworks not specifically mapped
                other_colors = ["forestgreen", "purple", "darkred"]
                for i, framework in enumerate(frameworks_present):
                    if framework not in framework_colors:
                        framework_colors[framework] = other_colors[i % len(other_colors)]
                
                # Create a visually distinctive dataframe for plotting
                # For each theme, create a dict with clean name and framework
                theme_display_data = []
                
                for theme in pivot.index:
                    framework = theme.split(':')[0].strip()
                    theme_name = theme.split(':', 1)[1].strip()
                    
                    # Use improved_truncate_text for line breaking instead of manual handling
                    formatted_theme = improved_truncate_text(theme_name, max_length=40)
                    
                    theme_display_data.append({
                        'original': theme,
                        'clean_name': formatted_theme,
                        'framework': framework,
                        'color': framework_colors[framework]
                    })
                    
                theme_display_df = pd.DataFrame(theme_display_data)
                
                # Add year count labels
                year_labels = [f"{math.floor(year)}<br>n={reports_per_year[year]}" for year in pivot.columns]
                
                # Create heatmap using plotly
                fig = go.Figure()
                
                # Add heatmap
                heatmap = go.Heatmap(
                    z=pivot.values,
                    x=year_labels,
                    y=theme_display_df['clean_name'],
                    colorscale=[
                        [0, '#f7fbff'],      # Lightest blue (almost white) for zero values
                        [0.2, '#deebf7'],    # Very light blue
                        [0.4, '#9ecae1'],    # Light blue
                        [0.6, '#4292c6'],    # Medium blue
                        [0.8, '#2171b5'],    # Deep blue
                        [1.0, '#084594']     # Darkest blue
                    ],
                    zmin=0,
                    zmax=min(100, pivot.values.max() * 1.2),  # Cap at 100% or 20% higher than max
                    colorbar=dict(
                        title=dict(text="Percentage (%)", font=dict(color="white", size=12)),
                        tickfont=dict(color="white", size=10),
                        outlinecolor="rgba(255,255,255,0.3)",
                        outlinewidth=1
                    ),
                    hoverongaps=False,
                    text=count_pivot.values,  # This will show the count in the hover
                    hovertemplate='Year: %{x}<br>Theme: %{y}<br>Percentage: %{z}%<br>Count: %{text}<extra></extra>'
                )
                
                fig.add_trace(heatmap)
                
                # Add count annotations
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        if pivot.iloc[i, j] > 0:
                            count = count_pivot.iloc[i, j]
                            
                            # Calculate appropriate text color based on cell value
                            bg_intensity = pivot.iloc[i, j] / 100  # Normalize to 0-1
                            text_color = 'white' if bg_intensity > 0.4 else 'black'
                            
                            # Add an annotation with outline for better visibility
                            fig.add_annotation(
                                x=j,
                                y=i,
                                text=f"({math.floor(count)})",
                                font=dict(size=9, color=text_color),
                                showarrow=False,
                                xanchor="center",
                                yanchor="middle",
                                bordercolor="rgba(0,0,0,0.2)",
                                borderwidth=1,
                                borderpad=2
                            )
                
                # Improved framework indicator styling
                for framework, color in framework_colors.items():
                    # Find rows corresponding to this framework
                    framework_rows = theme_display_df[theme_display_df['framework'] == framework]
                    
                    if len(framework_rows) > 0:
                        # Get the indices in the sorted display order
                        indices = framework_rows.index.tolist()
                        min_idx = min(indices)
                        max_idx = max(indices)
                        
                        # Add a colored rectangle with higher contrast
                        fig.add_shape(
                            type="rect",
                            x0=-1.3,  # Further to the left
                            x1=-0.7,  # Wider rectangle
                            y0=min_idx - 0.5,  # Align with the heatmap cells
                            y1=max_idx + 0.5,
                            fillcolor=color,
                            opacity=0.85,  # More visible
                            layer="below",
                            line=dict(width=1, color='rgba(255,255,255,0.5)')  # White border
                        )
                        
                        # Add framework label with black or white text based on background color
                        if len(framework_rows) > 1:  # Only add text if multiple themes in framework
                            # Determine text color (black for light backgrounds, white for dark)
                            text_color = "black" if framework in ["I-SIRch"] else "white"
                            
                            fig.add_annotation(
                                x=-1.0,
                                y=(min_idx + max_idx) / 2,
                                text=framework,
                                showarrow=False,
                                textangle=90,  # Vertical text
                                font=dict(size=12, color=text_color, family="Arial, sans-serif"),
                                xanchor="center",
                                yanchor="middle"
                            )
                
                # Update layout for better readability on dark background
                fig.update_layout(
                    title="Framework Theme Heatmap by Year",
                    font=dict(family="Arial, sans-serif", color="white"),  # White font for dark background
                    title_font=dict(size=16, color="white"),  # Larger title with white color (fixed small size)
                    xaxis_title="Year (number of reports)",
                    yaxis_title="Theme",
                    height=max(650, len(pivot.index) * 35),  # Increased height
                    width=900,  # Set explicit width
                    margin=dict(l=250, r=60, t=80, b=80),  # Increased left margin further
                    paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
                    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.05,  # Moved up for more space
                        xanchor="center",
                        x=0.5,
                        bgcolor="rgba(50,50,50,0.8)",  # Dark semi-transparent background
                        bordercolor="rgba(255,255,255,0.3)",
                        borderwidth=1,
                        font=dict(color="white")  # White text for legend
                    )
                )
                
                # Set y-axis formatting for dark mode
                fig.update_layout(
                    yaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(theme_display_df))),
                        ticktext=theme_display_df['clean_name'],
                        tickfont=dict(
                            size=11,
                            color='white'  # Changed to white from black for consistency
                        ),
                        automargin=True  # Added to ensure labels fit properly
                    )
                )
                # Improve x-axis formatting for dark mode
                fig.update_xaxes(
                    tickangle=-0,  # Horizontal labels
                    title_font=dict(size=14, color="white"),  # White color for axis title
                    tickfont=dict(size=12, color="white"),  # White color for tick labels
                    gridcolor="rgba(255,255,255,0.1)"  # Very subtle grid
                )
                
                # Improve y-axis formatting for dark mode
                fig.update_yaxes(
                    title_font=dict(size=14, color="white"),  # White color for axis title
                    tickfont=dict(size=11, color="white"),  # White color for tick labels
                    automargin=True  # Ensure labels fit properly
                )
                
                # Add framework legend
                for i, (framework, color) in enumerate(framework_colors.items()):
                    if framework in frameworks_present:  # Only show legends for frameworks present in data
                        fig.add_trace(go.Scatter(
                            x=[None],
                            y=[None],
                            mode='markers',
                            marker=dict(size=10, color=color),
                            name=framework,
                            showlegend=True
                        ))
                
                # Use st.plotly_chart's config parameter for better sizing
                st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    config={
                        'displayModeBar': True,
                        'responsive': True,
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'theme_heatmap',
                            'height': 800,
                            'width': 1200,
                            'scale': 2  # Higher resolution
                        }
                    }
                )
                
    # === TAB 2: THEME DISTRIBUTION ===
    with tab2:
        st.subheader("Theme Distribution Analysis")
        
        # Get top themes by count
        theme_counts = filtered_df["Theme"].value_counts().head(top_n_themes)
        
        # Use improved_truncate_text for better label formatting
        formatted_themes = [improved_truncate_text(theme, max_length=40) for theme in theme_counts.index]
        
        # Create a bar chart with formatted theme names
        fig = px.bar(
            x=formatted_themes,
            y=theme_counts.values,
            labels={"x": "Theme", "y": "Count"},
            title="Top 10 Themes by Occurrence",
            height=600,  # Increased height to accommodate multi-line labels
            color_discrete_sequence=['#4287f5']  # Consistent blue color
        )
        
        # Improve layout with better spacing for multi-line text
        fig.update_layout(
            xaxis_title="Theme",
            yaxis_title="Number of Occurrences",
            xaxis={'categoryorder':'total descending'},
            xaxis_tickangle=-30,  # Less extreme angle for better readability with multi-line text
            margin=dict(l=50, r=50, b=150, t=80),  # Increased bottom margin for labels
            bargap=0.2,  # Add some gap between bars
            font=dict(family="Arial, sans-serif", color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        # Update axes for dark mode
        fig.update_xaxes(
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255,255,255,0.1)",
            automargin=True  # Ensure labels fit properly without overlap
        )
        
        fig.update_yaxes(
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255,255,255,0.1)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
        # Theme by confidence
        st.subheader("Theme Confidence Breakdown")
        
        # Group by theme and confidence
        theme_confidence = filtered_df.groupby(["Theme", "Confidence"]).size().reset_index(name="Count")
        
        # Filter for top themes only
        top_themes = theme_counts.index.tolist()
        theme_confidence = theme_confidence[theme_confidence["Theme"].isin(top_themes)]
        
        # Create a mapping dictionary for theme display names
        theme_display_map = {theme: improved_truncate_text(theme, max_length=40) for theme in top_themes}
        
        # Apply the formatting to the DataFrame
        theme_confidence["Display_Theme"] = theme_confidence["Theme"].map(theme_display_map)
        
        # Create a grouped bar chart with formatted theme names
        fig = px.bar(
            theme_confidence, 
            x="Display_Theme",  # Use the formatted theme names
            y="Count", 
            color="Confidence",
            barmode="group",
            color_discrete_map={"High": "#4CAF50", "Medium": "#FFC107", "Low": "#F44336"},
            category_orders={
                "Confidence": ["High", "Medium", "Low"],
                "Display_Theme": [theme_display_map[theme] for theme in top_themes]
            },
            title="Confidence Distribution by Theme",
            height=600  # Increased height for better readability
        )
        
        # Improve layout for multi-line labels
        fig.update_layout(
            xaxis_title="Theme",
            yaxis_title="Number of Reports",
            xaxis_tickangle=-30,  # Less extreme angle for better readability
            margin=dict(l=50, r=50, b=150, t=80),  # Increased bottom margin
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="center", 
                x=0.5,
                title=None,
                font=dict(color="white")
            ),
            font=dict(family="Arial, sans-serif", color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        # Update axes for dark mode and ensure labels fit
        fig.update_xaxes(
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255,255,255,0.1)",
            automargin=True  # Ensure labels fit properly without overlap
        )
        
        fig.update_yaxes(
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255,255,255,0.1)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # === TAB 3: TEMPORAL ANALYSIS ===
    with tab3:
        st.subheader("Temporal Analysis")
        
        if "year" not in filtered_df.columns or filtered_df["year"].isna().all():
            st.warning("No year data available for temporal analysis.")
        else:
            # Create a time series of themes by year
            year_theme_counts = filtered_df.groupby(["year", "Theme"]).size().reset_index(name="Count")
            
            # Filter for top themes only - get top themes by total count
            all_theme_counts = filtered_df["Theme"].value_counts()
            top_themes = all_theme_counts.head(top_n_themes).index.tolist()
            
            year_theme_counts = year_theme_counts[year_theme_counts["Theme"].isin(top_themes)]
            
            # Create a mapping dictionary for formatted theme names
            theme_display_map = {theme: improved_truncate_text(theme, max_length=40) for theme in top_themes}
            
            # Apply the formatting to the DataFrame
            year_theme_counts["Display_Theme"] = year_theme_counts["Theme"].map(theme_display_map)
            
            # Create a line chart - important fix: convert year to string to treat as categorical
            year_theme_counts['year_str'] = year_theme_counts['year'].astype(str)
            year_theme_counts['year_int'] = year_theme_counts['year'].astype(int).astype(str)


            fig = px.line(
                year_theme_counts,
                x="year_int",  # Use string version of year
                y="Count",
                color="Display_Theme",  # Use formatted theme names
                markers=True,
                title="Theme Trends Over Time",
                height=600,  # Increased height
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of Occurrences",
                xaxis=dict(
                    type='category',  # Force categorical x-axis
                    tickmode="array",
                    tickvals=sorted(year_theme_counts['year_int'].unique()),
                    ticktext=sorted(year_theme_counts['year_int'].unique()),
                ),
                # Move legend below the chart for more horizontal space and prevent overlap
                legend=dict(
                    orientation="h", 
                    yanchor="top", 
                    y=-0.2,  # Position below the chart
                    xanchor="center", 
                    x=0.5,
                    title=None  # Remove legend title
                ),
                margin=dict(l=50, r=50, b=150, t=80),  # Increase bottom margin for legend
                font=dict(color="white"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            
            # Update axes for dark mode
            fig.update_xaxes(
                title_font=dict(color="white"),
                tickfont=dict(color="white"),
                gridcolor="rgba(255,255,255,0.1)"
            )
            
            fig.update_yaxes(
                title_font=dict(color="white"),
                tickfont=dict(color="white"),
                gridcolor="rgba(255,255,255,0.1)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap of themes by year
            st.subheader("Theme Prevalence by Year")
            
            # Create a pivot table
            pivot_df = year_theme_counts.pivot(index="Theme", columns="year_str", values="Count").fillna(0)
            
            # Convert to a normalized heatmap (percentage)
            # Calculate the total themes per year
            year_theme_totals = pivot_df.sum(axis=0)
            normalized_pivot = pivot_df.div(year_theme_totals, axis=1) * 100
            
            # Format the theme names for better display
            formatted_themes = [improved_truncate_text(theme, max_length=40) for theme in normalized_pivot.index]
            
            # Create a heatmap - ensure years are in correct order
            year_order = sorted(year_theme_counts['year'].unique())
            year_order_str = [str(y) for y in year_order]
            
            fig = px.imshow(
                normalized_pivot[year_order_str],  # Ensure columns are in correct order
                labels=dict(x="Year", y="Theme", color="% of Themes"),
                x=year_order_str,  # Use sorted string years
                y=formatted_themes,  # Use formatted theme names
                color_continuous_scale=[
                    [0, '#f7fbff'],      # Lightest blue (almost white) for zero values
                    [0.2, '#deebf7'],    # Very light blue
                    [0.4, '#9ecae1'],    # Light blue
                    [0.6, '#4292c6'],    # Medium blue
                    [0.8, '#2171b5'],    # Deep blue
                    [1.0, '#084594']     # Darkest blue
                ],
                title="Theme Prevalence by Year (%)",
                height=600,
                aspect="auto",
                text_auto=".1f"  # Show percentage to 1 decimal place
            )
            
            fig.update_layout(
                xaxis=dict(
                    type='category',  # Force categorical x-axis
                    tickmode="array",
                    tickvals=year_order_str,
                    ticktext=year_order_str,
                ),
                margin=dict(l=250, r=50, t=80, b=50),  # Increased left margin for theme labels
                font=dict(color="white"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            
            # Update axes and colorbar for dark mode
            fig.update_xaxes(
                title_font=dict(color="white"),
                tickfont=dict(color="white"),
                automargin=True  # Ensure labels don't overlap
            )
            
            fig.update_yaxes(
                title_font=dict(color="white"),
                tickfont=dict(color="white"),
                automargin=True  # Ensure y-axis labels fit
            )
            
            fig.update_traces(
                colorbar=dict(
                    title=dict(text="% of Themes", font=dict(color="white")),
                    tickfont=dict(color="white")
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # === TAB 4: AREA COMPARISON ===
    with tab4:
        st.subheader("Coroner Area Comparison")
        
        if "coroner_area" not in filtered_df.columns or filtered_df["coroner_area"].isna().all():
            st.warning("No coroner area data available for area comparison.")
        else:
            # Get the top areas by theme count
            area_counts = filtered_df["coroner_area"].value_counts().head(10)
            top_areas = area_counts.index.tolist()
            
            # Format area names for better display
            formatted_areas = [improved_truncate_text(area, max_length=40) for area in area_counts.index]
            
            # Create a mapping for display names
            area_display_map = dict(zip(area_counts.index, formatted_areas))
            
            # Create a bar chart of top areas with formatted names
            fig = px.bar(
                x=formatted_areas,
                y=area_counts.values,
                labels={"x": "Coroner Area", "y": "Count"},
                title="Theme Identifications by Coroner Area",
                height=500,
                color_discrete_sequence=['#ff9f40']  # Orange color for areas
            )
            
            fig.update_layout(
                xaxis_title="Coroner Area",
                yaxis_title="Number of Theme Identifications",
                xaxis_tickangle=-30,  # Less extreme angle for readability
                margin=dict(l=50, r=50, b=150, t=80),  # Increased bottom margin
                font=dict(color="white"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            
            # Update axes for dark mode and ensure labels fit
            fig.update_xaxes(
                title_font=dict(color="white"),
                tickfont=dict(color="white"),
                gridcolor="rgba(255,255,255,0.1)",
                automargin=True  # Ensure labels fit without overlap
            )
            
            fig.update_yaxes(
                title_font=dict(color="white"),
                tickfont=dict(color="white"),
                gridcolor="rgba(255,255,255,0.1)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Multi-area theme comparison
            st.subheader("Theme Distribution Across Top Coroner Areas")
            
            # Calculate total records per area (for normalization)
            area_totals = {}
            for area in top_areas:
                area_totals[area] = len(filtered_df[filtered_df["coroner_area"] == area])
            
            # Get theme distribution for each area
            area_theme_data = []
            
            # Get top themes overall for comparison
            all_theme_counts = filtered_df["Theme"].value_counts()
            top_themes = all_theme_counts.head(top_n_themes).index.tolist()
            
            # Create a mapping for formatted theme names
            theme_display_map = {theme: improved_truncate_text(theme, max_length=40) for theme in top_themes}
            
            for area in top_areas:
                area_df = filtered_df[filtered_df["coroner_area"] == area]
                area_themes = area_df["Theme"].value_counts()
                
                # Calculate percentage for each top theme
                for theme in top_themes:
                    count = area_themes.get(theme, 0)
                    percentage = (count / area_totals[area] * 100) if area_totals[area] > 0 else 0
                    
                    area_theme_data.append({
                        "Coroner Area": area,
                        "Display_Area": area_display_map[area],
                        "Theme": theme,
                        "Display_Theme": theme_display_map[theme],
                        "Count": count,
                        "Percentage": round(percentage, 1)
                    })
            
            area_theme_df = pd.DataFrame(area_theme_data)
            
            # Create heatmap using formatted names
            pivot_df = area_theme_df.pivot(
                index="Display_Area", 
                columns="Display_Theme", 
                values="Percentage"
            ).fillna(0)
            
            # Ensure we have data to display
            if not pivot_df.empty:
                fig = px.imshow(
                    pivot_df,
                    labels=dict(x="Theme", y="Coroner Area", color="Percentage"),
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    color_continuous_scale="YlGnBu",
                    title="Theme Distribution by Coroner Area (%)",
                    height=700,  # Increased height
                    aspect="auto",
                    text_auto=".1f"  # Show to 1 decimal place
                )
                
                fig.update_layout(
                    xaxis_title="Theme",
                    yaxis_title="Coroner Area",
                    xaxis_tickangle=-30,  # Reduce angle
                    coloraxis_colorbar=dict(
                        title=dict(text="% of Cases", font=dict(color="white")),
                        tickfont=dict(color="white")
                    ),
                    margin=dict(l=250, r=70, b=180, t=80),  # Increased margins
                    font=dict(color="white"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                
                # Enable automargin to ensure labels fit
                fig.update_xaxes(automargin=True)
                fig.update_yaxes(automargin=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data to create area-theme heatmap.")
                
            # Radar chart option for areas
            st.subheader("Theme Radar Comparison")
            
            # Select areas for radar chart
            radar_areas = st.multiselect(
                "Select Areas to Compare (2-5 recommended)",
                options=top_areas,
                default=top_areas[:3] if len(top_areas) >= 3 else top_areas
            )
            
            if radar_areas and len(radar_areas) >= 2:
                # Filter data for selected areas and top themes
                radar_data = area_theme_df[
                    (area_theme_df["Coroner Area"].isin(radar_areas)) & 
                    (area_theme_df["Theme"].isin(top_themes[:8]))  # Limit to 8 themes for readability
                ]
                
                # Create radar chart
                fig = go.Figure()
                
                # Add traces for each area
                for area in radar_areas:
                    area_data = radar_data[radar_data["Coroner Area"] == area]
                    # Sort by theme to ensure consistency
                    area_data = area_data.set_index("Theme").reindex(top_themes[:8]).reset_index()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=area_data["Percentage"],
                        theta=area_data["Display_Theme"],  # Use formatted theme names
                        fill="toself",
                        name=area_display_map.get(area, area)  # Use formatted area names
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(radar_data["Percentage"]) * 1.1],
                            tickfont=dict(color="white")
                        ),
                        angularaxis=dict(
                            tickfont=dict(color="white")
                        )
                    ),
                    showlegend=True,
                    legend=dict(font=dict(color="white")),
                    title=dict(
                        text="Theme Distribution Radar Chart",
                        font=dict(color="white")
                    ),
                    height=700,  # Increased height
                    margin=dict(l=80, r=80, t=100, b=80),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least 2 areas for radar comparison.")
            
    # === TAB 5: CORRELATION ANALYSIS ===
    with tab5:
        st.subheader("Theme Correlation Analysis")
        
        # Create correlation explanation
        st.markdown("""
        This analysis reveals relationships between themes. A high correlation suggests that when one theme appears, 
        the other is likely to appear in the same documents as well.
        """)
        
        # Calculate correlation between themes
        # First, pivot the data to get a binary matrix of themes by report
        id_column = 'Record ID' if 'Record ID' in filtered_df.columns else filtered_df.columns[0]
        
        # Create a binary pivot table: 1 if theme exists for a report, 0 otherwise
        theme_pivot = pd.crosstab(
            index=filtered_df[id_column], 
            columns=filtered_df['Theme'],
            values=filtered_df.get('Combined Score', 1),  # Use the score if we want weighted values
            aggfunc='max'  # Take the maximum score for each theme in a report
        ).fillna(0)
        
        # Convert to binary (1 if theme exists, 0 otherwise)
        theme_pivot = (theme_pivot > 0).astype(int)
        
        # Calculate correlation between themes
        theme_corr = theme_pivot.corr()
        
        # Get only the top themes for clarity
        top_theme_corr = theme_corr.loc[top_themes, top_themes]
        
        # Create a mapping dictionary for theme display names
        theme_display_map = {theme: improved_truncate_text(theme, max_length=40) for theme in top_themes}
        
        # Format column and index labels
        formatted_themes = [theme_display_map[theme] for theme in top_theme_corr.columns]
        
        # Create the correlation matrix visualization
        fig_corr_matrix = px.imshow(
            top_theme_corr,
            color_continuous_scale=px.colors.diverging.RdBu_r,  # Red-Blue diverging colorscale
            color_continuous_midpoint=0,
            labels=dict(x="Theme", y="Theme", color="Correlation"),
            title="Theme Correlation Matrix",
            height=800,  # Increased height
            width=850,   # Increased width
            text_auto=".2f",  # Show correlation values with 2 decimal places
            x=formatted_themes,
            y=formatted_themes
        )
        
        # Improved layout with better label positioning
        fig_corr_matrix.update_layout(
            margin=dict(l=200, r=80, b=220, t=80),  # Dramatically increased bottom margin
            font=dict(color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                side="bottom",  # Place labels at the bottom
                tickangle=90,   # Vertical text instead of angled
                automargin=True # Auto-adjust margins
            ),
            yaxis=dict(
                automargin=True # Auto-adjust margins
            )
        )
        
        # Update axes for dark mode and ensure labels fit
        fig_corr_matrix.update_xaxes(
            title_font=dict(color="white"),
            tickfont=dict(color="white", size=11),
            gridcolor="rgba(255,255,255,0.1)",
            automargin=True  # Ensure labels fit properly
        )
        
        fig_corr_matrix.update_yaxes(
            title_font=dict(color="white"),
            tickfont=dict(color="white", size=11),
            gridcolor="rgba(255,255,255,0.1)",
            automargin=True  # Ensure labels fit properly
        )
        
        # Update colorbar for dark mode
        fig_corr_matrix.update_traces(
            colorbar=dict(
                title=dict(text="Correlation", font=dict(color="white")),
                tickfont=dict(color="white")
            )
        )
        
        # Display the correlation matrix with a unique key
        st.plotly_chart(fig_corr_matrix, use_container_width=True, key="theme_correlation_matrix")
        
        # Network graph of correlations
        st.subheader("Theme Connection Network (THIS WILL BE IMPROVED)")
        
        # Correlation threshold slider
        corr_threshold = st.slider(
            "Correlation Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.05,
            help="Minimum correlation value to show connections between themes",
            key="correlation_threshold_slider"
        )
        
        # Create network from correlation matrix
        G = nx.Graph()
        
        # Add nodes (themes) with formatted display names
        for theme in top_theme_corr.columns:
            G.add_node(theme, display_name=theme_display_map[theme])
        
        # Add edges (correlations above threshold)
        for i, theme1 in enumerate(top_theme_corr.columns):
            for j, theme2 in enumerate(top_theme_corr.columns):
                if i < j:  # Only process each pair once
                    correlation = top_theme_corr.loc[theme1, theme2]
                    if correlation >= corr_threshold:
                        G.add_edge(theme1, theme2, weight=correlation)
        
        # Check if we have any edges
        if len(G.edges()) == 0:
            st.warning(f"No connections found with correlation threshold of {corr_threshold}. Try lowering the threshold.")
        else:
            # Calculate positions using the Fruchterman-Reingold force-directed algorithm
            pos = nx.spring_layout(G, seed=42)  # For reproducibility
            
            # Create a network visualization
            edge_trace = []
            
            # Add edges with width proportional to correlation
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = G[edge[0]][edge[1]]['weight']
                
                edge_trace.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(width=weight*5, color=f'rgba(150,150,150,{weight})'),
                        hoverinfo='none',
                        mode='lines'
                    )
                )
            
            # Add nodes
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_hover = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Use formatted name for display
                display_name = improved_truncate_text(node.split(':')[0] if ':' in node else node, max_length=20)
                node_text.append(display_name)
                
                # Calculate node size based on number of connections
                size = len(list(G.neighbors(node))) * 10 + 20
                node_size.append(size)
                
                # Create node text for hover
                neighbors = list(G.neighbors(node))
                connections = [f"{theme_display_map[neighbor]} (r={G[node][neighbor]['weight']:.2f})" for neighbor in neighbors]
                connection_text = "<br>".join(connections)
                node_hover.append(f"{theme_display_map[node]}<br>Connections: {len(connections)}<br>{connection_text}")
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                marker=dict(
                    size=node_size,
                    color='skyblue',
                    line=dict(width=1, color='white')
                ),
                hoverinfo='text',
                hovertext=node_hover,
                textfont=dict(color="white")
            )
            
            # Create the figure
            fig_network = go.Figure(
                data=edge_trace + [node_trace],
                layout=go.Layout(
                    title=dict(
                        text=f'Theme Connection Network (r ≥ {corr_threshold})',
                        font=dict(color="white")
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    width=800,
                    height=800,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
            )
            
            # Display the network graph with a unique key
            st.plotly_chart(fig_network, key="theme_network_graph")
        
            # Create a co-occurrence frequency table
            st.subheader("Theme Co-occurrence Table")
            
            # Create a matrix to count co-occurrences
            co_occurrence_matrix = np.zeros((len(top_themes), len(top_themes)))
            
            # Iterate through each document to count co-occurrences
            for doc_id in theme_pivot.index:
                # Get themes present in this document
                doc_themes = theme_pivot.columns[theme_pivot.loc[doc_id] == 1].tolist()
                # Only consider top themes
                doc_themes = [t for t in doc_themes if t in top_themes]
                
                # Count pairs of co-occurring themes
                for i, theme1 in enumerate(doc_themes):
                    idx1 = top_themes.index(theme1)
                    for theme2 in doc_themes:
                        idx2 = top_themes.index(theme2)
                        co_occurrence_matrix[idx1, idx2] += 1
        
            # Create a formatted DataFrame for display in the UI
            display_co_occurrence = pd.DataFrame(
                co_occurrence_matrix,
                index=[theme_display_map[theme] for theme in top_themes],
                columns=[theme_display_map[theme] for theme in top_themes]
            )
            
            # Keep original for CSV export
            co_occurrence_df = pd.DataFrame(
                co_occurrence_matrix,
                index=top_themes,
                columns=top_themes
            )
            
            # Display the co-occurrence table with formatted names
            st.dataframe(
                display_co_occurrence,
                use_container_width=True,
                height=400,
                key="co_occurrence_table"
            )
            
            # Add explanation
            st.markdown("""
            This table shows the number of documents where each pair of themes co-occurs. 
            The diagonal represents the total count of each theme.
            """)
    
            # Create a heatmap of the co-occurrence matrix
            fig_cooccur = px.imshow(
                co_occurrence_matrix,
                x=[theme_display_map[theme] for theme in top_themes],
                y=[theme_display_map[theme] for theme in top_themes],
                labels=dict(x="Theme", y="Theme", color="Co-occurrences"),
                title="Theme Co-occurrence Heatmap",
                color_continuous_scale="Viridis",
                text_auto=".0f"  # Show integer values
            )
            
            fig_cooccur.update_layout(
                margin=dict(l=200, r=80, b=220, t=80),
                font=dict(color="white"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    side="bottom",
                    tickangle=90,
                    automargin=True
                ),
                yaxis=dict(
                    automargin=True
                )
            )
            
            # Update axes for dark mode
            fig_cooccur.update_xaxes(
                title_font=dict(color="white"),
                tickfont=dict(color="white", size=11),
                gridcolor="rgba(255,255,255,0.1)",
                automargin=True
            )
            
            fig_cooccur.update_yaxes(
                title_font=dict(color="white"),
                tickfont=dict(color="white", size=11),
                gridcolor="rgba(255,255,255,0.1)",
                automargin=True
            )
            
            # Update colorbar for dark mode
            fig_cooccur.update_traces(
                colorbar=dict(
                    title=dict(text="Co-occurrences", font=dict(color="white")),
                    tickfont=dict(color="white")
                )
            )
            
            # Display the co-occurrence heatmap with a unique key
            st.plotly_chart(fig_cooccur, use_container_width=True, key="cooccurrence_heatmap")
        
        # Show detailed data table
        with st.expander("View Detailed Data"):
            st.dataframe(
                filtered_df,
                column_config={
                    "Title": st.column_config.TextColumn("Document Title"),
                    "Framework": st.column_config.TextColumn("Framework"),
                    "Theme": st.column_config.TextColumn("Theme"),
                    "Confidence": st.column_config.TextColumn("Confidence"),
                    "Combined Score": st.column_config.NumberColumn("Score", format="%.3f"),
                    "Matched Keywords": st.column_config.TextColumn("Keywords"),
                    "coroner_name": st.column_config.TextColumn("Coroner Name"),
                    "coroner_area": st.column_config.TextColumn("Coroner Area"),
                    "year": st.column_config.NumberColumn("Year"),
                },
                use_container_width=True,
                key="detailed_data_table"
            )
            
        # Export options
        st.subheader("Export Filtered Data")
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create columns for download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"theme_analysis_export_{timestamp}.csv",
                mime="text/csv",
                key=f"download_csv_{timestamp}",
            )
        
        with col2:
            # Excel Export
            excel_data = export_to_excel(filtered_df)
            st.download_button(
                "📥 Download Filtered Data (Excel)",
                data=excel_data,
                file_name=f"theme_analysis_export_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_excel_{timestamp}",
            )
    
        with col3:
            # All Images Export
            try:
                # Get the zip file and image count
                images_zip, image_count = save_dashboard_images_as_zip(filtered_df)
                
                # Update button text to show number of images
                st.download_button(
                    f"📥 Download {image_count} Visualizations (ZIP)",
                    data=images_zip,
                    file_name=f"theme_analysis_images_{timestamp}.zip",
                    mime="application/zip",
                    key=f"download_images_{timestamp}",
                )
            except Exception as e:
                st.error(f"Error creating visualization zip: {e}")
                logging.error(f"Visualization zip error: {e}", exc_info=True)


def render_analysis_tab(data: pd.DataFrame = None):
    """Render the analysis tab with improved filters, file upload functionality, and analysis sections"""

    # Add file upload section at the top
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
            
            # Reference Number
            ref_numbers = sorted(data['ref'].dropna().unique())
            selected_refs = st.multiselect(
                "Reference Numbers",
                options=ref_numbers,
                key="ref_filter"
            )
            
            # Deceased Name - Changed to dropdown instead of text input
            deceased_names = sorted(set(
                str(name).strip() for name in data['deceased_name'].dropna().unique()
            ))
            selected_deceased = st.multiselect(
                "Deceased Name",
                options=deceased_names,
                key="deceased_filter",
                help="Select one or more deceased names"
            )
            
            # Coroner Name
            # Normalize coroner names for selection and ensure uniqueness
            coroner_names = sorted(set(
                str(name).strip() for name in data['coroner_name'].dropna().unique()
            ))
            selected_coroners = st.multiselect(
                "Coroner Names",
                options=coroner_names,
                key="coroner_filter"
            )
            
            # Coroner Area
            # Normalize coroner areas for selection and ensure uniqueness
            coroner_areas = sorted(set(
                str(area).strip() for area in data['coroner_area'].dropna().unique()
            ))
            selected_areas = st.multiselect(
                "Coroner Areas",
                options=coroner_areas,
                key="areas_filter"
            )
            
            # Categories - Improved handling of both list and string formats
            all_categories = set()
            for cats in data['categories'].dropna():
                if isinstance(cats, list):
                    all_categories.update(str(cat).strip() for cat in cats if cat)
                elif isinstance(cats, str):
                    # Handle comma-separated strings
                    all_categories.update(str(cat).strip() for cat in cats.split(',') if cat)
            
            # Remove any empty strings
            all_categories = {cat for cat in all_categories if cat.strip()}
            
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
                # Only responses
                filtered_df = filtered_df[filtered_df.apply(is_response, axis=1)]
            elif "Report" in doc_type and "Response" not in doc_type:
                # Only reports
                filtered_df = filtered_df[~filtered_df.apply(is_response, axis=1)]

        # Reference number filter
        if selected_refs:
            filtered_df = filtered_df[filtered_df['ref'].isin(selected_refs)]

        # Deceased name filter - changed to use dropdown selection
        if selected_deceased:
            # Normalize selected deceased names and create a case-insensitive filter
            selected_deceased_norm = [str(name).lower().strip() for name in selected_deceased]
            filtered_df = filtered_df[
                filtered_df['deceased_name'].fillna('').str.lower().apply(
                    lambda x: any(selected_name in x or x in selected_name for selected_name in selected_deceased_norm)
                )
            ]

        # Coroner name filter - case-insensitive partial match
        if selected_coroners:
            # Normalize selected coroners and create a case-insensitive filter
            selected_coroners_norm = [str(name).lower().strip() for name in selected_coroners]
            filtered_df = filtered_df[
                filtered_df['coroner_name'].fillna('').str.lower().apply(
                    lambda x: any(selected_name in x or x in selected_name for selected_name in selected_coroners_norm)
                )
            ]

        # Coroner area filter - case-insensitive partial match
        if selected_areas:
            # Normalize selected areas and create a case-insensitive filter
            selected_areas_norm = [str(area).lower().strip() for area in selected_areas]
            filtered_df = filtered_df[
                filtered_df['coroner_area'].fillna('').str.lower().apply(
                    lambda x: any(selected_area in x or x in selected_area for selected_area in selected_areas_norm)
                )
            ]

        # Categories filter - use the improved filter_by_categories function
        if selected_categories:
            filtered_df = filter_by_categories(filtered_df, selected_categories)

        # Show active filters
        active_filters = []
        if start_date != min_date or end_date != max_date:
            active_filters.append(f"Date: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
        if doc_type:
            active_filters.append(f"Document Types: {', '.join(doc_type)}")
        if selected_refs:
            active_filters.append(f"References: {', '.join(selected_refs)}")
        if selected_deceased:
            if len(selected_deceased) <= 3:
                active_filters.append(f"Deceased Names: {', '.join(selected_deceased)}")
            else:
                active_filters.append(f"Deceased Names: {len(selected_deceased)} selected")
        if selected_coroners:
            if len(selected_coroners) <= 3:
                active_filters.append(f"Coroners: {', '.join(selected_coroners)}")
            else:
                active_filters.append(f"Coroners: {len(selected_coroners)} selected")
        if selected_areas:
            if len(selected_areas) <= 3:
                active_filters.append(f"Areas: {', '.join(selected_areas)}")
            else:
                active_filters.append(f"Areas: {len(selected_areas)} selected")
        if selected_categories:
            if len(selected_categories) <= 3:
                active_filters.append(f"Categories: {', '.join(selected_categories)}")
            else:
                active_filters.append(f"Categories: {len(selected_categories)} selected")

        if active_filters:
            st.info("Active filters:\n" + "\n".join(f"• {filter_}" for filter_ in active_filters))

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
                    "Document Type": st.column_config.TextColumn(
                        "Document Type",
                        help="Type of document based on PDF filename"
                    )
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
                # Timeline of reports
                st.subheader("Reports Timeline")
                plot_timeline(filtered_df)
                
                # Monthly distribution
                st.subheader("Monthly Distribution")
                plot_monthly_distribution(filtered_df)
                
                # Year-over-year comparison
                st.subheader("Year-over-Year Comparison")
                plot_yearly_comparison(filtered_df)
                
                # Seasonal patterns
                st.subheader("Seasonal Patterns")
                seasonal_counts = filtered_df['date_of_report'].dt.month.value_counts().sort_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                fig = px.line(
                    x=month_names,
                    y=[seasonal_counts.get(i, 0) for i in range(1, 13)],
                    markers=True,
                    labels={'x': 'Month', 'y': 'Number of Reports'},
                    title='Seasonal Distribution of Reports'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Distribution Analysis Tab
            with distribution_tab:
                st.subheader("Reports by Category")
                plot_category_distribution(filtered_df)
  
                st.subheader("Reports by Coroner Area")
                plot_coroner_areas(filtered_df)

            # Export options
            st.markdown("---")
            show_export_options(filtered_df, "analysis")

        else:
            st.warning("No reports match your filter criteria. Try adjusting the filters.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Analysis error: {e}", exc_info=True)



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
