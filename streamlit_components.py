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
import plotly.express as px
from sklearn.decomposition import LatentDirichletAllocation
import json

# Import our modules
from core_utils import (
    process_scraped_data, 
    clean_text_for_modeling, 
    export_topic_results,
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
from vectorizer_models import get_vectorizer
from bert_analysis import BERTResultsAnalyzer, ThemeAnalyzer
from visualization import (
    plot_category_distribution,
    plot_coroner_areas,
    plot_timeline,
    plot_monthly_distribution,
    plot_yearly_comparison,
    display_topic_network,
    render_framework_heatmap
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
