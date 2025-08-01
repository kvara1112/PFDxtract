import logging
import pandas as pd
import numpy as np
import streamlit as st
import io
import time
import random
import string
import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm
import os
import shutil
import streamlit as st

# Optional WeasyPrint import (only needed for PDF generation)
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    # Handle both import errors and OS-level library errors (like GTK on Windows)
    WEASYPRINT_AVAILABLE = False
    HTML = None
    CSS = None


# Import our core utilities
from .core_utils import (
    export_to_excel, 
    extract_concern_text, 
    format_date_uk,
    is_response_document
)

######################################
class BERTResultsAnalyzer:
    """Enhanced class for merging BERT theme analysis results files with specific column outputs and year extraction."""

    
    def __init__(self):
        """Initialize the analyzer with default settings."""
        self.data = None
        # Define the essential columns you want in the reduced output
        self.essential_columns = [
            "Title",
            "URL",
            "Content",
            "date_of_report",
            "ref", 
            "deceased_name",
            "coroner_name",
            "coroner_area",
            "categories",
            "Report ID",
            "Deceased Name",
            "Death Type",
            "Year",
            "year",
            "Extracted_Concerns",
        ]
        
    def render_analyzer_ui(self):
        """Render the file merger UI."""
        st.subheader("Scraped File Merger")
        st.markdown(
            """
            This tool merges multiple scraped files into a single dataset. It prepares the data for steps (3) - (5).
            
            - Run this step even if you only have one scraped file. This step extracts the year and applies other processing as described in the bullets below. 
            - Combine data from multiple CSV or Excel files (the name of these files starts with pfd_reports_scraped_reportID_ )
            - Extract missing concerns from PDF content and fill empty Content fields
            - Extract year information from date fields
            - Remove duplicate records
            - Export full or reduced datasets with essential columns
            
            Use the options below to control how your files will be processed.
            """
        )

        # File upload section
        self._render_multiple_file_upload()

    def _render_multiple_file_upload(self):
        """Render interface for multiple file upload and merging."""
        # Initialize session state for processed data if not already present
        if "bert_merged_data" not in st.session_state:
            st.session_state.bert_merged_data = None

        # Use a dynamic key for file uploader based on reset counter
        reset_counter = st.session_state.get("reset_counter", 0)
        uploaded_files = st.file_uploader(
            "Upload CSV or Excel files exported from the scraper tool",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            help="Upload multiple CSV or Excel files to merge them",
            key="bert_multi_uploader_static",
        )

        if uploaded_files and len(uploaded_files) > 0:
            st.info(f"Uploaded {len(uploaded_files)} files")

            # Allow user to specify merge settings
            with st.expander("Merge Settings", expanded=True):
                st.info("These settings control how the files will be merged.")

                # Option to remove duplicates - static key
                drop_duplicates = st.checkbox(
                    "Remove Duplicate Records",
                    value=True,
                    help="If checked, duplicate records will be removed after merging",
                    key="drop_duplicates_static",
                )

                # Option to add year from date_of_report
                extract_year = st.checkbox(
                    "Extract Year from date_of_report",
                    value=True,
                    help="If checked, a 'year' column will be added based on the date_of_report",
                    key="extract_year_static",
                )

                # Option to attempt extraction from PDF content when concerns are missing
                extract_from_pdf = st.checkbox(
                    "Extract Missing Concerns from PDF Content",
                    value=True,
                    help="If checked, will attempt to extract missing concerns from PDF content",
                    key="extract_from_pdf_static",
                )

                # Option to fill empty Content from PDF content
                fill_empty_content = st.checkbox(
                    "Fill Empty Content from PDF Content",
                    value=True,
                    help="If checked, will fill empty Content fields from PDF content",
                    key="fill_empty_content_static",
                )

                duplicate_columns = "Record ID"
                if drop_duplicates:
                    duplicate_columns = st.text_input(
                        "Columns for Duplicate Check",
                        value="ref",
                        help="Comma-separated list of columns to check for duplicates",
                        key="duplicate_columns_static",
                    )

            # Button to process the files - static key
            if st.button("Merge and/or Process Files", key="merge_files_button_static"):
                try:
                    with st.spinner("Processing and merging files..."):
                        # Stack files
                        duplicate_cols = None
                        if drop_duplicates:
                            duplicate_cols = [
                                col.strip() for col in duplicate_columns.split(",")
                            ]

                        self._merge_files_stack(uploaded_files, duplicate_cols)

                        # Now processed_data is in self.data
                        if self.data is not None:
                            # Fill empty Content from PDF if requested
                            if fill_empty_content:
                                before_count = self.data["Content"].notna().sum()
                                self.data = self._fill_empty_content_from_pdf(self.data)
                                after_count = self.data["Content"].notna().sum()
                                newly_filled = after_count - before_count

                                if newly_filled > 0:
                                    st.success(
                                        f"Filled empty Content from PDF content for {newly_filled} records."
                                    )
                                else:
                                    st.info(
                                        "No Content fields could be filled from PDF content."
                                    )

                            # Extract year from date_of_report if requested
                            if extract_year and "date_of_report" in self.data.columns:
                                self.data = self._add_year_column(self.data)
                                with_year = self.data["year"].notna().sum()
                                st.success(
                                    f"Added year data to {with_year} out of {len(self.data)} reports."
                                )

                            # Extract missing concerns from PDF content if requested
                            if extract_from_pdf:
                                before_count = (
                                    self.data["Extracted_Concerns"].notna().sum()
                                )
                                self.data = self._extract_missing_concerns_from_pdf(
                                    self.data
                                )
                                after_count = (
                                    self.data["Extracted_Concerns"].notna().sum()
                                )
                                newly_extracted = after_count - before_count

                                if newly_extracted > 0:
                                    st.success(
                                        f"Extracted missing concerns from PDF content for {newly_extracted} reports."
                                    )
                                else:
                                    st.info(
                                        "No additional concerns could be extracted from PDF content."
                                    )

                            st.success(
                                f"Files merged successfully! Final dataset has {len(self.data)} records."
                            )

                            # Show a preview of the data
                            st.subheader("Preview of Merged Data")
                            st.dataframe(self.data.head(5))

                            # Save merged data to session state
                            st.session_state.bert_merged_data = self.data.copy()
                        else:
                            st.error(
                                "File merging resulted in empty data. Please check your files."
                            )

                except Exception as e:
                    st.error(f"Error merging files: {str(e)}")
                    logging.error(f"File merging error: {e}", exc_info=True)

        # Show download options if we have processed data
        # Use either the current instance data or data from session state
        show_download_options = False

        if hasattr(self, "data") and self.data is not None and len(self.data) > 0:
            show_download_options = True
        elif st.session_state.bert_merged_data is not None:
            self.data = st.session_state.bert_merged_data
            show_download_options = True

        if show_download_options:
            self._provide_download_options()

    def _fill_empty_content_from_pdf(self, df):
        """
        Fill empty Content fields from PDF_1_Content only.
        
        Args:
            df: DataFrame with merged data
        
        Returns:
            DataFrame with filled Content fields
        """
        if df is None or len(df) == 0:
            return df
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Identify records with missing Content
        missing_content_mask = processed_df["Content"].isna() | (
            processed_df["Content"].astype(str).str.strip() == ""
        )
        missing_content_count = missing_content_mask.sum()
        
        if missing_content_count == 0:
            return processed_df
        
        # Add a progress bar for processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(
            f"Filling empty Content fields from PDF_1_Content for {missing_content_count} records..."
        )
        
        # Check if PDF_1_Content column exists
        if "PDF_1_Content" not in processed_df.columns:
            progress_bar.empty()
            status_text.empty()
            return processed_df
        
        # Process each record with missing Content
        missing_indices = processed_df[missing_content_mask].index
        filled_count = 0
        
        for i, idx in enumerate(missing_indices):
            # Update progress
            progress = (i + 1) / len(missing_indices)
            progress_bar.progress(progress)
            
            # Check if PDF_1_Content exists and has content
            if pd.notna(processed_df.at[idx, "PDF_1_Content"]) and processed_df.at[idx, "PDF_1_Content"].strip() != "":
                # Use the PDF content as the main Content
                processed_df.at[idx, "Content"] = processed_df.at[idx, "PDF_1_Content"]
                processed_df.at[idx, "Content_Source"] = "PDF_1_Content"  # Track where content came from
                filled_count += 1
            
            # Update status
            status_text.text(
                f"Filled Content for {filled_count} of {i+1}/{missing_content_count} records..."
            )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return processed_df
    
    
    def _extract_missing_concerns_from_pdf(self, df):
        """
        Extract concerns from PDF content for records with missing Extracted_Concerns.

        Args:
            df: DataFrame with merged data

        Returns:
            DataFrame with additional extracted concerns
        """
        if df is None or len(df) == 0:
            return df

        # Make a copy to avoid modifying the original
        processed_df = df.copy()

        # Identify records with missing concerns
        missing_concerns = self._identify_missing_concerns(processed_df)

        if len(missing_concerns) == 0:
            return processed_df

        # Add a progress bar for extraction
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(
            f"Extracting concerns from PDF content for {len(missing_concerns)} records..."
        )

        # Check each PDF content column for missing concerns
        pdf_columns = [
            col
            for col in processed_df.columns
            if col.startswith("PDF_") and col.endswith("_Content")
        ]
        count_extracted = 0

        for i, (idx, row) in enumerate(missing_concerns.iterrows()):
            # Update progress
            progress = (i + 1) / len(missing_concerns)
            progress_bar.progress(progress)

            # Try to extract concerns from each PDF content column
            for pdf_col in pdf_columns:
                if pd.notna(row.get(pdf_col)) and row.get(pdf_col) != "":
                    # Extract concerns using existing function
                    concern_text = extract_concern_text(row[pdf_col])

                    # If we found concerns, update the main dataframe
                    if (
                        concern_text and len(concern_text.strip()) > 20
                    ):  # Ensure meaningful text
                        processed_df.at[idx, "Extracted_Concerns"] = concern_text
                        processed_df.at[
                            idx, "Concern_Source"
                        ] = pdf_col  # Track where concerns came from
                        count_extracted += 1
                        break  # Move to next record once concerns are found

            # Update status
            status_text.text(
                f"Extracted concerns for {count_extracted} of {i+1}/{len(missing_concerns)} records..."
            )

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        return processed_df

    def _extract_report_year(self, date_val):
        """
        Optimized function to extract year from dd/mm/yyyy date format.
        Works with both string representations and datetime objects.
        """

        # Return None for empty inputs
        if pd.isna(date_val):
            return None

        # Handle datetime objects directly
        if isinstance(date_val, (datetime.datetime, pd.Timestamp)):
            return date_val.year

        # Handle string dates with dd/mm/yyyy format
        if isinstance(date_val, str):
            # Direct regex match for dd/mm/yyyy pattern (faster than datetime parsing)
            match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_val)
            if match:
                return int(match.group(3))  # Year is in the third capture group

        # Handle numeric or other non-string types by converting to string
        try:
            date_str = str(date_val)
            match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str)
            if match:
                return int(match.group(3))
        except:
            pass

        return None

    # block3
    def _add_missing_years_from_content(self, concern_sections, df=None):
        """Try to extract years from content when date_of_report is missing."""


        missing_year_count = 0

        # Try to use dataframe if available for additional data sources
        index_to_date = {}
        if df is not None:
            date_related_columns = []
            for col in df.columns:
                if any(term in col.lower() for term in ["date", "year", "time"]):
                    date_related_columns.append(col)

            if date_related_columns:
                print(
                    f"Checking {len(date_related_columns)} date-related columns for missing years"
                )
                for idx, row in df.iterrows():
                    for col in date_related_columns:
                        if pd.notna(row.get(col)):
                            val = row.get(col)
                            # Convert to string if needed
                            if not isinstance(val, str):
                                val = str(val)
                            year = self._extract_report_year(val)
                            if year:
                                index_to_date[idx] = year
                                break

        for section in concern_sections:
            if section.get("year") is None:
                # Check if we have a year for this index in our lookup
                if (
                    df is not None
                    and "index" in section
                    and section["index"] in index_to_date
                ):
                    section["year"] = index_to_date[section["index"]]
                    section["year_source"] = "other_date_column"
                    missing_year_count += 1
                    continue

                # Try to extract year from title first
                title = section.get("title", "") or section.get("Title", "")

                # Look for years in the title
                title_year_match = re.search(r"\b(19|20)\d{2}\b", title)
                if title_year_match:
                    section["year"] = int(title_year_match.group(0))
                    section["year_source"] = "extracted_from_title"
                    missing_year_count += 1
                    continue

                # Then try content
                content = (
                    section.get("Content", "")
                    or section.get("content", "")
                    or section.get("concern_text", "")
                    or section.get("Extracted_Concerns", "")
                )
                if content:
                    # Look for date patterns first
                    date_matches = re.findall(
                        r"\b\d{1,2}[/-]\d{1,2}[/-](19|20)\d{2}\b", content
                    )
                    if date_matches:
                        # Extract year from the first date pattern found
                        full_date = re.search(
                            r"\b\d{1,2}[/-]\d{1,2}[/-](19|20)\d{2}\b", content
                        ).group(0)
                        year_str = re.search(r"(19|20)\d{2}", full_date).group(0)
                        section["year"] = int(year_str)
                        section["year_source"] = "date_in_content"
                        missing_year_count += 1
                        continue

                    # Otherwise look for any years
                    year_matches = re.findall(r"\b(19|20)\d{2}\b", content)
                    if year_matches:
                        # Get all years mentioned
                        years = [int(y) for y in year_matches]
                        # Use the most frequent year
                        most_common_year = Counter(years).most_common(1)[0][0]
                        section["year"] = most_common_year
                        section["year_source"] = "year_in_content"
                        missing_year_count += 1

                        # Extra check for future years which may be errors
                        current_year = datetime.datetime.now().year
                        if section["year"] > current_year:
                            # If it's a future year, try to find a more plausible one
                            plausible_years = [y for y in years if y <= current_year]
                            if plausible_years:
                                section["year"] = max(plausible_years)

        if missing_year_count > 0:
            print(
                f"Added year data to {missing_year_count} reports using content analysis"
            )

        return concern_sections

    def _add_year_column(self, df):
        """
        Add a 'year' column to the DataFrame extracted from the date_of_report column.
        If date_of_report is missing, try to extract year from content.

        Args:
            df: DataFrame with at least date_of_report column

        Returns:
            DataFrame with additional 'year' column
        """
        # Create a copy to avoid modifying the original DataFrame
        processed_df = df.copy()

        # Check if DataFrame has date_of_report column
        if "date_of_report" not in processed_df.columns:
            st.warning(
                "No 'date_of_report' column found in the data. Year extraction may be incomplete."
            )
            processed_df["year"] = None
            return processed_df

        # Create a new column for year extracted from date_of_report
        processed_df["year"] = processed_df["date_of_report"].apply(
            self._extract_report_year
        )

        # Count how many years were successfully extracted
        extracted_count = processed_df["year"].notna().sum()

        # For rows with missing years, try to extract from content
        if processed_df["year"].isna().any():
            # Create a list of dicts from rows with missing years
            missing_year_rows = []
            for idx, row in processed_df[processed_df["year"].isna()].iterrows():
                section_dict = row.to_dict()
                section_dict["index"] = idx  # Add index for tracking
                missing_year_rows.append(section_dict)

            # Try to extract years from content
            if missing_year_rows:
                missing_year_rows = self._add_missing_years_from_content(
                    missing_year_rows, processed_df
                )

                # Update the original DataFrame with extracted years
                for section in missing_year_rows:
                    if (
                        "year" in section
                        and section["year"] is not None
                        and "index" in section
                    ):
                        processed_df.at[section["index"], "year"] = section["year"]
                        # Optionally add the source of extraction
                        if "year_source" in section:
                            processed_df.at[section["index"], "year_source"] = section[
                                "year_source"
                            ]

        # Count final results
        final_count = processed_df["year"].notna().sum()
        if final_count > extracted_count:
            print(
                f"Added {final_count - extracted_count} more years from content analysis"
            )

        return processed_df


# block4

    def _provide_download_options(self):
        """Provide options to download the current data."""
        if self.data is None or len(self.data) == 0:
            return
        
        st.subheader("Download Merged Data")
        
        # Generate timestamp and random suffix for truly unique keys
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        unique_id = f"{timestamp}_{random_suffix}"
        
        # Deduplicate data by Record ID before download if requested
        dedup_download = st.checkbox(
            "Remove duplicate Record IDs before download (keep only first occurrence)",
            value=True,
            key=f"dedup_checkbox_{unique_id}",
        )
        
        download_data = self.data
        if dedup_download and "Record ID" in self.data.columns:
            download_data = self.data.drop_duplicates(subset=["Record ID"], keep="first")
            st.info(
                f"Download will contain {len(download_data)} rows after removing duplicate Record IDs (original had {len(self.data)} rows)"
            )
        
        # Prepare the reduced dataset with essential columns
        reduced_data = download_data.copy()
        
        # Get list of available essential columns
        available_essential_cols = [
            col for col in self.essential_columns if col in reduced_data.columns
        ]
        
        # Ensure 'ref' is in the available columns if it exists in the data
        if "ref" in reduced_data.columns and "ref" not in available_essential_cols:
            available_essential_cols.append("ref")
            
        if available_essential_cols:
            reduced_data = reduced_data[available_essential_cols]
            st.success(
                f"Reduced dataset includes these columns: {', '.join(available_essential_cols)}"
            )
        else:
            st.warning(
                "None of the essential columns found in the data. Will provide full dataset only."
            )
            reduced_data = None
        
        # Generate filename prefix
        filename_prefix = f"merged_{timestamp}"
        
        # Full Dataset Section
        st.markdown("### Full Dataset")
        full_col1, full_col2 = st.columns(2)
        
        # CSV download button for full data
        with full_col1:
            try:
                # Create export copy with formatted dates
                df_csv = download_data.copy()
                if (
                    "date_of_report" in df_csv.columns
                    and pd.api.types.is_datetime64_any_dtype(df_csv["date_of_report"])
                ):
                    df_csv["date_of_report"] = df_csv["date_of_report"].dt.strftime(
                        "%d/%m/%Y"
                    )
    
                csv_data = df_csv.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Full Dataset (CSV)",
                    data=csv_data,
                    file_name=f"{filename_prefix}_full.csv",
                    mime="text/csv",
                    key=f"download_full_csv_{unique_id}",
                )
            except Exception as e:
                st.error(f"Error preparing CSV export: {str(e)}")
        
        # Excel download button for full data
        with full_col2:
            try:
                excel_buffer_full = io.BytesIO()
                download_data.to_excel(excel_buffer_full, index=False, engine="openpyxl")
                excel_buffer_full.seek(0)
                st.download_button(
                    "📥 Download Full Dataset (Excel)",
                    data=excel_buffer_full,
                    file_name=f"{filename_prefix}_full.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_full_excel_{unique_id}",
                )
            except Exception as e:
                st.error(f"Error preparing Excel export: {str(e)}")
        
        # Only show reduced dataset options if we have essential columns
        if reduced_data is not None:
            st.markdown("### Reduced Dataset (Essential Columns)")
            reduced_col1, reduced_col2 = st.columns(2)
        
            # CSV download button for reduced data
            with reduced_col1:
                try:
                    # Create export copy with formatted dates
                    df_csv_reduced = reduced_data.copy()
                    if (
                        "date_of_report" in df_csv_reduced.columns
                        and pd.api.types.is_datetime64_any_dtype(
                            df_csv_reduced["date_of_report"]
                        )
                    ):
                        df_csv_reduced["date_of_report"] = df_csv_reduced[
                            "date_of_report"
                        ].dt.strftime("%d/%m/%Y")
    
                    reduced_csv_data = df_csv_reduced.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Reduced Dataset (CSV)",
                        data=reduced_csv_data,
                        file_name=f"{filename_prefix}_reduced.csv",
                        mime="text/csv",
                        key=f"download_reduced_csv_{unique_id}",
                    )
                except Exception as e:
                    st.error(f"Error preparing reduced CSV export: {str(e)}")
        
            # Excel download button for reduced data
            with reduced_col2:
                try:
                    excel_buffer_reduced = io.BytesIO()
                    reduced_data.to_excel(
                        excel_buffer_reduced, index=False, engine="openpyxl"
                    )
                    excel_buffer_reduced.seek(0)
                    st.download_button(
                        "📥 Download Reduced Dataset (Excel)",
                        data=excel_buffer_reduced,
                        file_name=f"{filename_prefix}_reduced.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_reduced_excel_{unique_id}",
                    )
                except Exception as e:
                    st.error(f"Error preparing reduced Excel export: {str(e)}")
        
        # NEW: Add section to display and download reports with missing concerns
        st.markdown("### Reports Without Extracted Concerns")
        
        # Find records without concerns
        missing_concerns_df = self._identify_missing_concerns(download_data)
        
        if len(missing_concerns_df) > 0:
            st.warning(
                f"Found {len(missing_concerns_df)} reports without properly extracted concerns."
            )
        
            # Display the dataframe with missing concerns
            essential_columns_for_display = [
                col
                for col in ["Title", "URL", "date_of_report", "year", "deceased_name", "ref"]
                if col in missing_concerns_df.columns
            ]
            if essential_columns_for_display:
                st.dataframe(
                    missing_concerns_df[essential_columns_for_display],
                    use_container_width=True,
                )
        
            # Download options for missing concerns
            missing_col1, missing_col2 = st.columns(2)
        
            # CSV download
            with missing_col1:
                try:
                    missing_csv = missing_concerns_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Missing Concerns Data (CSV)",
                        data=missing_csv,
                        file_name=f"{filename_prefix}_missing_concerns.csv",
                        mime="text/csv",
                        key=f"download_missing_csv_{unique_id}",
                    )
                except Exception as e:
                    st.error(f"Error preparing missing concerns CSV: {str(e)}")
        
            # Excel download
            with missing_col2:
                try:
                    excel_buffer_missing = io.BytesIO()
                    missing_concerns_df.to_excel(
                        excel_buffer_missing, index=False, engine="openpyxl"
                    )
                    excel_buffer_missing.seek(0)
                    st.download_button(
                        "📥 Download Missing Concerns Data (Excel)",
                        data=excel_buffer_missing,
                        file_name=f"{filename_prefix}_missing_concerns.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_missing_excel_{unique_id}",
                    )
                except Exception as e:
                    st.error(f"Error preparing missing concerns Excel: {str(e)}")
        else:
            st.success("All reports have properly extracted concerns.")
    
        # block5
        def _is_response(self, row):
            """Check if a row represents a response document."""
            # Check title for response indicators
            title = str(row.get("Title", "")).lower()
            title_response = any(
                word in title for word in ["response", "reply", "answered"]
            )
    
            # Check PDF types if available
            for i in range(1, 5):  # Check PDF_1 to PDF_4
                pdf_type = str(row.get(f"PDF_{i}_Type", "")).lower()
                if pdf_type == "response":
                    return True
    
            # Check PDF names as backup
            for i in range(1, 5):
                pdf_name = str(row.get(f"PDF_{i}_Name", "")).lower()
                if "response" in pdf_name or "reply" in pdf_name:
                    return True
    
            # Check content as final fallback if available
            content = str(row.get("Content", "")).lower()
            content_response = any(
                phrase in content
                for phrase in [
                    "in response to",
                    "responding to",
                    "reply to",
                    "response to",
                    "following the regulation 28",
                    "following receipt of the regulation 28",
                ]
            )
    
            return title_response or content_response

    def _filter_out_responses(self, df):
        """Filter out response documents, keeping only reports."""
        return df[~df.apply(self._is_response, axis=1)]

    def _merge_files_stack(self, files, duplicate_cols=None):
        """Merge multiple files by stacking (appending) them."""
        dfs = []
    
        for file_index, file in enumerate(files):
            try:
                # Read file
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
    
                # Display file information
                st.info(
                    f"Processing file {file_index+1}: {file.name} ({len(df)} rows, {len(df.columns)} columns)"
                )
    
                # Add source filename
                df["Source File"] = file.name
    
                # Add to the list of dataframes
                dfs.append(df)
    
            except Exception as e:
                st.warning(f"Error processing file {file.name}: {str(e)}")
                continue
    
        if not dfs:
            raise ValueError("No valid files to merge")
    
        # Combine all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
    
        # Remove duplicates if specified
        if duplicate_cols:
            valid_dup_cols = [col for col in duplicate_cols if col in merged_df.columns]
            if valid_dup_cols:
                before_count = len(merged_df)
                merged_df = merged_df.drop_duplicates(
                    subset=valid_dup_cols, keep="first"
                )
                after_count = len(merged_df)
    
                if before_count > after_count:
                    st.success(
                        f"Removed {before_count - after_count} duplicate records based on {', '.join(valid_dup_cols)}"
                    )
            else:
                st.warning(
                    f"Specified duplicate columns {duplicate_cols} not found in the merged data"
                )
    
        # ALWAYS remove duplicate Record IDs, keeping only the first occurrence
        if "Record ID" in merged_df.columns:
            before_count = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=["Record ID"], keep="first")
            after_count = len(merged_df)
    
            if before_count > after_count:
                st.success(
                    f"Removed {before_count - after_count} records with duplicate Record IDs (keeping first occurrence)"
                )
    
        # Clean coroner_name column
        if "coroner_name" in merged_df.columns:
            before_cleaning = merged_df["coroner_name"].copy()
            merged_df = self._clean_coroner_names(merged_df)
            
            # Count changes made
            changes_made = sum(before_cleaning != merged_df["coroner_name"])
            if changes_made > 0:
                st.success(f"Cleaned {changes_made} coroner name entries")
                
                # Show the first few changes as an example
                example_changes = []
                for i, (old, new) in enumerate(zip(before_cleaning, merged_df["coroner_name"])):
                    if old != new and len(example_changes) < 3 and isinstance(old, str) and isinstance(new, str):
                        example_changes.append(f"'{old}' → '{new}'")
                
                if example_changes:
                    st.info("Examples of cleaned coroner names:\n" + "\n".join(example_changes))
    
        # Clean coroner_area column
        if "coroner_area" in merged_df.columns:
            before_cleaning = merged_df["coroner_area"].copy()
            merged_df = self._clean_coroner_areas(merged_df)
            
            # Count changes made
            changes_made = sum(before_cleaning != merged_df["coroner_area"])
            if changes_made > 0:
                st.success(f"Cleaned {changes_made} coroner area entries")
                
                # Show the first few changes as an example
                example_changes = []
                for i, (old, new) in enumerate(zip(before_cleaning, merged_df["coroner_area"])):
                    if old != new and len(example_changes) < 3 and isinstance(old, str) and isinstance(new, str):
                        example_changes.append(f"'{old}' → '{new}'")
                
                if example_changes:
                    st.info("Examples of cleaned coroner areas:\n" + "\n".join(example_changes))

        ##her
        # Remove duplicate Record IDs, keeping only the first occurrence
        if "Record ID" in merged_df.columns:
            before_count = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=["Record ID"], keep="first")
            after_count = len(merged_df)

            if before_count > after_count:
                st.success(
                    f"Removed {before_count - after_count} records with duplicate Record IDs (keeping first occurrence)"
                )
        
        # Clean deceased names
        before_cleaning = merged_df["deceased_name"].copy()
        merged_df = self._clean_deceased_name(merged_df)
        
        # Count changes made to deceased names
        changes_made = sum(before_cleaning != merged_df["deceased_name"])
        if changes_made > 0:
            st.success(f"Cleaned {changes_made} deceased name entries")
            
            # Show the first few changes as an example
            example_changes = []
            for i, (old, new) in enumerate(zip(before_cleaning, merged_df["deceased_name"])):
                if old != new and len(example_changes) < 3 and isinstance(old, str) and isinstance(new, str):
                    example_changes.append(f"'{old}' → '{new}'")
            
            if example_changes:
                st.info("Examples of cleaned deceased names:\n" + "\n".join(example_changes))

        
        # Clean categories column
        if "categories" in merged_df.columns:
            # Save the original values for comparison
            original_categories = merged_df["categories"].copy()
            
            # Apply cleaning
            merged_df = self._clean_categories(merged_df)
            
            # Count changes - this is more complex since categories can be lists
            changes_made = 0
            example_changes = []
            
            # Check each row for changes
            for i, (old, new) in enumerate(zip(original_categories, merged_df["categories"])):
                # Handle list case
                if isinstance(old, list) and isinstance(new, list):
                    # Consider it changed if any element changed
                    if any(o != n for o, n in zip(old, new) if isinstance(o, str) and isinstance(n, str)):
                        changes_made += 1
                        # Add example if we don't have many yet
                        if len(example_changes) < 3:
                            old_str = ", ".join(old) if all(isinstance(x, str) for x in old) else str(old)
                            new_str = ", ".join(new) if all(isinstance(x, str) for x in new) else str(new)
                            example_changes.append(f"'{old_str}' → '{new_str}'")
                # Handle string case
                elif isinstance(old, str) and isinstance(new, str) and old != new:
                    changes_made += 1
                    if len(example_changes) < 3:
                        example_changes.append(f"'{old}' → '{new}'")
            
            # Report changes
            if changes_made > 0:
                st.success(f"Cleaned {changes_made} categories entries")
                if example_changes:
                    st.info("Examples of cleaned categories:\n" + "\n".join(example_changes))
                    
        # Store the result
        self.data = merged_df
    
        # Show summary of the merged data
        st.subheader("Merged Data Summary")
        st.write(f"Total rows: {len(merged_df)}")
        st.write(f"Columns: {', '.join(merged_df.columns)}")

    def _identify_missing_concerns(self, df):
        """
        Identify records without extracted concerns

        Args:
            df: DataFrame with BERT results

        Returns:
            DataFrame containing only records without extracted concerns
        """
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # Check for concerns column
        concerns_column = None
        for col_name in ["Extracted_Concerns", "extracted_concerns", "concern_text"]:
            if col_name in df.columns:
                concerns_column = col_name
                break

        if concerns_column is None:
            st.warning("No concerns column found in the data.")
            return pd.DataFrame()

        # Filter out records with missing or empty concerns
        missing_concerns = df[
            df[concerns_column].isna()
            | (df[concerns_column].astype(str).str.strip() == "")
            | (
                df[concerns_column].astype(str).str.len() < 20
            )  # Very short extracts likely failed
        ].copy()

        return missing_concerns

    #
    def _clean_deceased_name(self, df):
        """
        Clean deceased name column by removing coroner-related text and normalizing
        
        Args:
            df (pd.DataFrame): DataFrame containing 'deceased_name' column
        
        Returns:
            pd.DataFrame: DataFrame with cleaned deceased names
        """
        if df is None or len(df) == 0 or 'deceased_name' not in df.columns:
            return df
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        def clean_name(name_text):
            # Return if input is not a string or is NaN
            if pd.isna(name_text) or not isinstance(name_text, str):
                return name_text
            
            # Convert to string and strip whitespace
            name = str(name_text).strip()
            
            # Remove common coroner-related prefixes and labels
            coroner_patterns = [
                r'^deceased name:?\s*',  # Remove "Deceased Name:" at start
                r'^coroner\'?s?\s*(name)?:?\s*',  # Remove "Coroner" or "Coroner's Name:" at start
                r'^deceased person:?\s*',  # Remove "Deceased Person:" at start
                r'^name of deceased:?\s*',  # Remove "Name of Deceased:" at start
            ]
            
            for pattern in coroner_patterns:
                name = re.sub(pattern, '', name, flags=re.IGNORECASE)
            
            # Remove any text after known trigger words
            name = re.sub(r'\s*(?:coroner|ref|reference).*$', '', name, flags=re.IGNORECASE)
            
            # Remove common unwanted suffixes or additional text
            name = re.sub(r'\s*\(.*\)$', '', name)  # Remove text in parentheses at end
            
            # Remove multiple spaces and extra whitespace
            name = re.sub(r'\s+', ' ', name).strip()
            
            return name
        
        # Save original values for comparison
        original_names = cleaned_df["deceased_name"].copy()
        
        # Apply cleaning to deceased name column
        cleaned_df['deceased_name'] = cleaned_df['deceased_name'].apply(clean_name)
        
        # Count and report changes
        changes_made = sum(original_names != cleaned_df['deceased_name'])
        
        # Optionally log or display changes (you can customize this part)
        if changes_made > 0:
            # Collect a few example changes
            example_changes = []
            for old, new in zip(original_names, cleaned_df['deceased_name']):
                if old != new and isinstance(old, str) and isinstance(new, str):
                    example_changes.append(f"'{old}' → '{new}'")
                    if len(example_changes) >= 5:  # Limit to 5 examples
                        break
            
            # Log or display changes
            print(f"Cleaned {changes_made} deceased name entries")
            for change in example_changes:
                print(change)
        
        return cleaned_df

    #

    def _clean_coroner_names(self, df): 
        """
        Clean coroner_name column by removing titles/prefixes and standardizing format
        
        Args:
            df (pd.DataFrame): DataFrame containing a 'coroner_name' column
            
        Returns:
            pd.DataFrame: DataFrame with cleaned 'coroner_name' column
        """
        if df is None or len(df) == 0 or 'coroner_name' not in df.columns:
            return df
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Common titles and prefixes to remove
        titles = [
            'Dr\\.?\\s+', 'Doctor\\s+', 
            'Mr\\.?\\s+', 'Mrs\\.?\\s+', 'Ms\\.?\\s+', 'Miss\\.?\\s+',
            'Prof\\.?\\s+', 'Professor\\s+',
            'Sir\\s+', 'Dame\\s+',
            'HM\\s+', 'HM\\s+Senior\\s+',
            'Hon\\.?\\s+', 'Honorable\\s+',
            'Justice\\s+', 'Judge\\s+',
            'QC\\s+', 'KC\\s+'
        ]
        
        import re
    
        def clean_name(name_text):
            if pd.isna(name_text) or not isinstance(name_text, str):
                return name_text
            
            name = name_text.strip()
    
            # Remove any titles/prefixes from the beginning
            pattern = r'^(' + '|'.join(titles) + r')+'
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
            # Remove known suffixes
            name = re.sub(r'\s+(QC|KC|Esq\.?|Jr\.?|Sr\.?)$', '', name, flags=re.IGNORECASE)
    
            # Remove content in parentheses
            name = re.sub(r'\(.*?\)', '', name)
    
            # Remove punctuation at the end and normalize whitespace
            name = re.sub(r'[;:,\.]$', '', name)
            name = re.sub(r'\s+', ' ', name).strip()
    
            # Remove repeated titles like "Dr Dr"
            name = re.sub(r'^(Dr\s+){2,}', '', name, flags=re.IGNORECASE)
    
            # Final cleanup: remove all text starting from "Coroner"
            name = re.sub(r'\s*Coroner.*$', '', name, flags=re.IGNORECASE)
    
            return name.strip()
        
        cleaned_df['coroner_name'] = cleaned_df['coroner_name'].apply(clean_name)
        return cleaned_df

    
    def _clean_coroner_areas(self, df):
        """
        Clean coroner_area column by:
        1. Converting everything to lowercase
        2. Removing brackets (but keeping their content)
        3. Replacing & with the word "and"
        4. Removing hyphens
        5. Removing the word "the"
        6. Making specific replacements for known locations
        
        Args:
            df (pd.DataFrame): DataFrame containing a 'coroner_area' column
            
        Returns:
            pd.DataFrame: DataFrame with cleaned 'coroner_area' column
        """
        if df is None or len(df) == 0 or 'coroner_area' not in df.columns:
            return df
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Define the cleaning function
        def clean_area(area_text):
            if pd.isna(area_text) or not isinstance(area_text, str):
                return area_text
            
            import re
            
            # Convert to lowercase
            area = area_text.lower()
            
            # Remove brackets but keep their content
            # For example, "bbbb (aaa)" becomes "bbbb aaa"
            area = re.sub(r'\(', ' ', area)  # Replace opening brackets with space
            area = re.sub(r'\)', ' ', area)  # Replace closing brackets with space
            area = re.sub(r'\[', ' ', area)  # Replace opening square brackets with space
            area = re.sub(r'\]', ' ', area)  # Replace closing square brackets with space
            
            # Replace & with 'and'
            area = area.replace('&', ' and ')  # This ensures the ampersand is properly replaced
            
            # Remove hyphens
            area = area.replace('-', ' ')  # Replace hyphens with spaces
            
            # Remove the word "the" - both standalone and as part of other words
            area = re.sub(r'\bthe\b', ' ', area)  # Remove standalone "the" with word boundaries
            
            # Replace multiple spaces with a single space
            area = re.sub(r'\s+', ' ', area)
            
            # Specific replacements for known variations - do these AFTER other cleanings
            # so they catch all variations including those with dashes or different spacing
            area = re.sub(r'\bisle of scilly\b', 'isles of scilly', area)  # Change "isle of scilly" to "isles of scilly"
            area = re.sub(r'\beast riding of yorkshire\b', 'east riding', area)  # Change "east riding of yorkshire" to "east riding"
            area = re.sub(r'\b(city of )?kingston upon hull\b', 'kingston upon hull', area)  # Remove "city of" from "kingston upon hull"
            
            # Remove common patterns that indicate the end of the coroner area
            end_patterns = [
                "coroner's concerns", 
                "matters of concern",
                "the matters of concern",
                "this report is being sent to:",
                "these reports are being sent to:",
                "the report is being sent to:",
                "this report",
                "these reports",
                "the report",
                "coroner",
                "category"
            ]
            
            # Find the earliest position of any pattern
            earliest_pos = len(area)
            for pattern in end_patterns:
                pos = area.find(pattern)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
            
            # If a pattern was found, truncate
            if earliest_pos != len(area):
                area = area[:earliest_pos]
            
            # Find the position of 'Category'
            category_pos = area.find('category')
            if category_pos != -1:
                area = area[:category_pos]
            
            # Try with just a pipe character, which often separates coroner area from categories
            pipe_pos = area.find('|')
            if pipe_pos != -1:
                area = area[:pipe_pos]
                
            # Remove any special characters at the beginning or end, but keep alphanumeric and spaces
            area = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', area)
            
            # Final cleanup - replace multiple spaces again and strip
            area = re.sub(r'\s+', ' ', area).strip()
            
            return area
        
        # Apply the cleaning function
        cleaned_df['coroner_area'] = cleaned_df['coroner_area'].apply(clean_area)
        
        return cleaned_df
      

    def _clean_categories(self, df):
        """
        Clean categories column by removing "These reports are being sent to:" and any text that follows
        
        Args:
            df (pd.DataFrame): DataFrame containing a 'categories' column
            
        Returns:
            pd.DataFrame: DataFrame with cleaned 'categories' column
        """
        if df is None or len(df) == 0 or 'categories' not in df.columns:
            return df
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Define the cleaning function for a single value
        def clean_categories_value(categories_text):
            if pd.isna(categories_text) or not isinstance(categories_text, str):
                return categories_text
            
            # Convert to lowercase for case-insensitive matching
            categories_text_lower = categories_text.lower()
            
            # Patterns to look for and remove
            report_patterns = [
                "these reports are being sent to:",
                "this report is being sent to:",
                "the report is being sent to:",
                "this report",
                "these reports",
                "the report",
                "this is being"
            ]
            
            # Find the earliest position of any report-related pattern
            earliest_pos = len(categories_text)
            for pattern in report_patterns:
                pos = categories_text_lower.find(pattern)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
            
            # If a report pattern was found, truncate
            if earliest_pos != len(categories_text):
                categories_text = categories_text[:earliest_pos].strip()
            
            # Normalize text: remove brackets, convert '&' and 'and' to a standard form
            # Remove brackets
            categories_text = re.sub(r'\(.*?\)', '', categories_text).strip()
            
            # Replace variations of conjunctions
            categories_text = re.sub(r'\s*&\s*', ' and ', categories_text)
            
            return categories_text.strip()
        
        # Apply the cleaning function to the DataFrame
        # Handle both string values and list values
        before_cleaning = cleaned_df["categories"].copy()
        def log_to_console(message: str) -> None:
            js_code = f"""
        <script>
            console.log({json.dumps(message)});
        </script>
        """
            components.html(js_code)  
        # Process based on data type
        for idx, value in enumerate(cleaned_df["categories"]):
            if isinstance(value, list):
                # For list values, we need to check each element
                cleaned_list = []
                
                os.write(1, f"{cleaned_list}\n".encode()) 
                st.write(cleaned_list)
                log_to_console("test2")
                for item in value:
                    if isinstance(item, str):
                        cleaned_list.append(clean_categories_value(item))
                    else:
                        cleaned_list.append(item)
                cleaned_df.at[idx, "categories"] = cleaned_list
            elif isinstance(value, str):
                # For string values, clean directly
                cleaned_df.at[idx, "categories"] = clean_categories_value(value)

        # Remove duplicates from the categories column
        os.write(1, f"{cleaned_df['categories']}\n".encode()) 
        st.write(cleaned_df['categories'])

        
        return cleaned_df



    # End of BERTResultsAnalyzer class

###########################
class ThemeAnalyzer:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        """Initialize the BERT-based theme analyzer with sentence highlighting capabilities"""
        # Initialize transformer model and tokenizer
        st.info("Loading annotation model and tokenizer... This may take a moment.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Configuration settings
        self.config = {
            "base_similarity_threshold": 0.65,
            "keyword_match_weight": 0.3,
            "semantic_similarity_weight": 0.7,
            "max_themes_per_framework": 5,
            "context_window_size": 200,
        }

        # Initialize frameworks with themes
        self.frameworks = {
            "I-SIRch": self._get_isirch_framework(),
            "House of Commons": self._get_house_of_commons_themes(),
            "Extended Analysis": self._get_extended_themes(),
            "Yorkshire Contributory":self._get_yorkshire_framework(),
        }

        # Color mapping for themes
        self.theme_color_map = {}
        self.theme_colors = [
            "#FFD580",  # Light orange
            "#FFECB3",  # Light amber
            "#E1F5FE",  # Light blue
            "#E8F5E9",  # Light green
            "#F3E5F5",  # Light purple
            "#FFF3E0",  # Light orange
            "#E0F7FA",  # Light cyan
            "#F1F8E9",  # Light lime
            "#FFF8E1",  # Light yellow
            "#E8EAF6",  # Light indigo
            "#FCE4EC",  # Light pink
            "#F5F5DC",  # Beige
            "#E6E6FA",  # Lavender
            "#FFFACD",  # Lemon chiffon
            "#D1E7DD",  # Mint
            "#F8D7DA",  # Light red
            "#D1ECF1",  # Teal light
            "#FFF3CD",  # Light yellow
            "#D6D8D9",  # Light gray
            "#CFF4FC",  # Info light
        ]

        # Pre-assign colors to frameworks
        self._preassign_framework_colors()

    def _preassign_framework_colors(self):
        """Preassign colors to each framework for consistent coloring"""
        # Create a dictionary to track colors used for each framework
        framework_colors = {}

        # Assign colors to each theme in each framework
        for framework, themes in self.frameworks.items():
            for i, theme in enumerate(themes):
                theme_key = f"{framework}_{theme['name']}"
                # Assign color from the theme_colors list, cycling if needed
                color_idx = i % len(self.theme_colors)
                self.theme_color_map[theme_key] = self.theme_colors[color_idx]

    def get_bert_embedding(self, text, max_length=512):
        """Generate BERT embedding for text"""
        if not isinstance(text, str) or not text.strip():
            return np.zeros(768)

        # Tokenize with truncation
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use CLS token for sentence representation
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    def _get_contextual_embedding(self, text, keyword, window_size=100):
        """Get embedding for text surrounding the keyword occurrence"""
        if not isinstance(text, str) or not text.strip() or keyword not in text.lower():
            return self.get_bert_embedding(keyword)

        text_lower = text.lower()
        position = text_lower.find(keyword.lower())

        # Get context window
        start = max(0, position - window_size)
        end = min(len(text), position + len(keyword) + window_size)

        # Get contextual text
        context = text[start:end]
        return self.get_bert_embedding(context)

    def _calculate_combined_score(
        self, semantic_similarity, keyword_count, text_length
    ):
        """Calculate combined score that balances semantic similarity and keyword presence"""
        # Normalize keyword count by text length
        normalized_keyword_density = min(1.0, keyword_count / (text_length / 1000))

        # Weighted combination
        keyword_component = (
            normalized_keyword_density * self.config["keyword_match_weight"]
        )
        semantic_component = (
            semantic_similarity * self.config["semantic_similarity_weight"]
        )

        return keyword_component + semantic_component

    def _find_sentence_positions(self, text, keywords):
        """Find sentences containing keywords and return their positions"""
        if not isinstance(text, str) or not text.strip():
            return []

        # Split text into sentences
        sentence_endings = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_endings, text)

        # Track character positions and matched sentences
        positions = []
        current_pos = 0

        for sentence in sentences:
            if not sentence.strip():
                current_pos += len(sentence)
                continue

            # Check if any keyword is in this sentence
            sentence_lower = sentence.lower()
            matched_keywords = []

            for keyword in keywords:
                if keyword and len(keyword) >= 3 and keyword.lower() in sentence_lower:
                    # Check if it's a whole word using word boundaries
                    keyword_lower = keyword.lower()
                    pattern = r"\b" + re.escape(keyword_lower) + r"\b"
                    if re.search(pattern, sentence_lower):
                        matched_keywords.append(keyword)

            # If sentence contains any keywords, add to positions
            if matched_keywords:
                start_pos = current_pos
                end_pos = current_pos + len(sentence)
                # Join all matched keywords
                keywords_str = ", ".join(matched_keywords)
                positions.append((start_pos, end_pos, keywords_str, sentence))

            # Move to next position
            current_pos += len(sentence)

            # Account for sentence ending characters and whitespace
            if current_pos < len(text) and text[current_pos - 1] in ".!?":
                # Check for any whitespace after sentence ending
                space_count = 0
                while (
                    current_pos + space_count < len(text)
                    and text[current_pos + space_count].isspace()
                ):
                    space_count += 1
                current_pos += space_count

        return sorted(positions)
######
    def create_highlighted_html(self, text, theme_highlights):
        """Create HTML with sentences highlighted by theme with improved color consistency"""
        if not text or not theme_highlights:
            return text
        
        # Convert highlights to a flat list of positions
        all_positions = []
        for theme_key, positions in theme_highlights.items():
            theme_color = self._get_theme_color(theme_key)
            for pos_info in positions:
                # position format: (start_pos, end_pos, keywords_str, sentence)
                all_positions.append((
                    pos_info[0],  # start position
                    pos_info[1],  # end position
                    theme_key,    # theme key
                    pos_info[2],  # keywords string
                    pos_info[3],  # original sentence
                    theme_color   # theme color
                ))
        
        # Sort positions by start position
        all_positions.sort()
        
        # Merge overlapping sentences using primary theme's color
        merged_positions = []
        if all_positions:
            current = all_positions[0]
            for i in range(1, len(all_positions)):
                if all_positions[i][0] <= current[1]:  # Overlap
                    # Create a meaningful theme name combination
                    combined_theme = current[2] + " + " + all_positions[i][2]
                    combined_keywords = current[3] + " + " + all_positions[i][3]
                    
                    # Use the first theme's color for consistency
                    combined_color = current[5]
                    
                    # Update current with merged information
                    current = (
                        current[0],                # Keep original start position
                        max(current[1], all_positions[i][1]),  # Take the later end position
                        combined_theme,            # Combined theme names
                        combined_keywords,         # Combined keywords
                        current[4],                # Keep original sentence
                        combined_color             # Use the first theme's color
                    )
                else:
                    merged_positions.append(current)
                    current = all_positions[i]
            merged_positions.append(current)
        
        # Create highlighted text
        result = []
        last_end = 0
        
        for start, end, theme_key, keywords, sentence, color in merged_positions:
            # Add text before this highlight
            if start > last_end:
                result.append(text[last_end:start])
            
            # Simple styling with solid background color
            style = f"background-color:{color}; border:1px solid #666; border-radius:2px; padding:1px 2px;"
            tooltip = f"Theme: {theme_key}\nKeywords: {keywords}"
            result.append(f'<span style="{style}" title="{tooltip}">{text[start:end]}</span>')
            
            last_end = end
        
        # Add remaining text
        if last_end < len(text):
            result.append(text[last_end:])
        
        # Create HTML structure
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Highlighted Document Analysis</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                    font-weight: bold;
                }
                .paragraph-container {
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin-bottom: 20px;
                    background-color: #f9f9f9;
                }
            </style>
        </head>
        <body>
            <h2>Document Theme Analysis</h2>
            
            <div class="paragraph-container">
                <h3>Highlighted Text</h3>
                <p>
        """
        
        # Add the highlighted text
        html_content += ''.join(result)
        
        html_content += """
                </p>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Framework</th>
                        <th>Theme</th>
                        <th>Color</th>
                        <th>Matched Keywords</th>
                        <th>Extracted Sentence</th>
                    </tr>
                </thead>
                <tbody>
        """
    
        # Add rows to HTML with color-coded backgrounds
        for start, end, theme_key, keywords, sentence, color in merged_positions:
            # Split theme key into framework and theme
            framework, theme = theme_key.split('_', 1) if '_' in theme_key else (theme_key, theme_key)
            
            html_content += f"""
                    <tr>
                        <td>{framework}</td>
                        <td>{theme}</td>
                        <td style="background-color: {color};">{color}</td>
                        <td>{keywords}</td>
                        <td>{sentence}</td>
                    </tr>
            """
    
        # Close HTML structure
        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
    
        return html_content
    
    def convert_html_to_pdf(self, html_content, output_filename=None):
        """
        Convert the generated HTML to a PDF file
        
        Args:
            html_content (str): HTML content to convert
            output_filename (str, optional): Filename for the PDF. 
                                             If None, generates a timestamped filename.
        
        Returns:
            str: Path to the generated PDF file
        """
        if not WEASYPRINT_AVAILABLE:
            st.warning("WeasyPrint is not available. PDF conversion is disabled on this platform.")
            return None
            
        try:
            # Generate default filename if not provided
            if output_filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"theme_analysis_{timestamp}.pdf"
            
            # Ensure the filename ends with .pdf
            if not output_filename.lower().endswith('.pdf'):
                output_filename += '.pdf'
            
            # Additional CSS to ensure proper PDF rendering
            additional_css = """
            @page {
                size: A4;
                margin: 1cm;
            }
            body {
                font-family: Arial, sans-serif;
                font-size: 12pt;
                line-height: 1.6;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .paragraph-container {
                page-break-inside: avoid;
            }
            """
            
            # Create PDF
            HTML(string=html_content).write_pdf(
                output_filename, 
                stylesheets=[CSS(string=additional_css)]
            )
            
            return output_filename
        
        except Exception as e:
            # Handle other potential errors
            st.error(f"Error converting HTML to PDF: {str(e)}")
            return None
        
    def _create_integrated_html_for_pdf(self, results_df, highlighted_texts):
        """
        Create a single integrated HTML file with all highlighted records, themes, and framework information
        that can be easily converted to PDF
        """
    
    
    
        # Map report IDs to their themes
        report_themes = defaultdict(list)
        
        # Ensure all themes have unique colors
        self._ensure_unique_theme_colors(results_df)
    
        # Build the report data with consistent colors
        for _, row in results_df.iterrows():
            if "Record ID" in row and "Theme" in row and "Framework" in row:
                record_id = row["Record ID"]
                framework = row["Framework"]
                theme = row["Theme"]
                confidence = row.get("Confidence", "")
                score = row.get("Combined Score", 0)
                matched_keywords = row.get("Matched Keywords", "")
    
                # Get theme color from our mapping
                theme_key = f"{framework}_{theme}"
                theme_color = self._get_theme_color(theme_key)
    
                report_themes[record_id].append({
                    "framework": framework,
                    "theme": theme,
                    "confidence": confidence,
                    "score": score,
                    "keywords": matched_keywords,
                    "color": theme_color,
                    "theme_key": theme_key
                })
    
        # Create HTML content with modern styling
        html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Theme Analysis Report</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        line-height: 1.6; 
                        margin: 0;
                        padding: 20px;
                        color: #333;
                        background-color: #f9f9f9;
                    }
                    h1 { 
                        color: #2c3e50; 
                        border-bottom: 3px solid #3498db; 
                        padding-bottom: 10px; 
                        margin-top: 30px;
                        font-weight: 600;
                    }
                    h2 { 
                        color: #2c3e50; 
                        margin-top: 30px; 
                        border-bottom: 2px solid #bdc3c7; 
                        padding-bottom: 5px; 
                        font-weight: 600;
                    }
                    h3 {
                        color: #34495e;
                        font-weight: 600;
                        margin-top: 20px;
                    }
                    .record-container { 
                        margin-bottom: 40px; 
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                        padding: 20px;
                        page-break-after: always; 
                    }
                    .highlighted-text { 
                        margin: 15px 0; 
                        padding: 15px; 
                        border-radius: 4px;
                        border: 1px solid #ddd; 
                        background-color: #fff; 
                        line-height: 1.7;
                    }
                    .theme-info { margin: 15px 0; }
                    .theme-info table { 
                        border-collapse: collapse; 
                        width: 100%; 
                        margin-top: 15px;
                        border-radius: 4px;
                        overflow: hidden;
                    }
                    .theme-info th, .theme-info td { 
                        border: 1px solid #ddd; 
                        padding: 12px; 
                        text-align: left; 
                    }
                    .theme-info th { 
                        background-color: #3498db; 
                        color: white;
                        font-weight: 600;
                    }
                    .theme-info tr:nth-child(even) { background-color: #f9f9f9; }
                    .theme-info tr:hover { background-color: #f1f1f1; }
                    .high-confidence { background-color: #D5F5E3; }  /* Light green */
                    .medium-confidence { background-color: #FCF3CF; } /* Light yellow */
                    .low-confidence { background-color: #FADBD8; }   /* Light red */
                    .report-header {
                        background-color: #3498db;
                        color: white;
                        padding: 30px;
                        text-align: center;
                        border-radius: 8px;
                        margin-bottom: 30px;
                    }
                    .summary-card {
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                        padding: 20px;
                        margin-bottom: 30px;
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                    }
                    .summary-box {
                        flex: 1;
                        min-width: 200px;
                        padding: 15px;
                        text-align: center;
                        border-right: 1px solid #eee;
                    }
                    .summary-box:last-child {
                        border-right: none;
                    }
                    .summary-number {
                        font-size: 36px;
                        font-weight: bold;
                        color: #3498db;
                        margin-bottom: 10px;
                    }
                    .summary-label {
                        font-size: 14px;
                        color: #7f8c8d;
                        text-transform: uppercase;
                    }
                    .theme-color-box {
                        display: inline-block;
                        width: 20px;
                        height: 20px;
                        margin-right: 5px;
                        vertical-align: middle;
                        border: 1px solid #999;
                    }
                    .legend-container {
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                        padding: 15px;
                        margin-bottom: 20px;
                    }
                    .legend-title {
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    .legend-item {
                        display: flex;
                        align-items: center;
                        margin-bottom: 5px;
                    }
                    @media print {
                        .record-container { page-break-after: always; }
                        body { background-color: white; }
                        .record-container, .summary-card { box-shadow: none; }
                    }
                    
                    /* Define theme-specific CSS classes for consistency */
        """
    
        # Add dynamic CSS classes for each theme
        theme_keys = set()
        for _, row in results_df.iterrows():
            if "Framework" in row and "Theme" in row:
                theme_keys.add(f"{row['Framework']}_{row['Theme']}")
        
        for theme_key in theme_keys:
            color = self._get_theme_color(theme_key)
            safe_class_name = "theme-" + theme_key.replace(" ", "-").replace("(", "").replace(")", "").replace(",", "").replace(".", "").lower()
            html_content += f"""
                    .{safe_class_name} {{
                        background-color: {color} !important;
                    }}
            """
    
        html_content += """
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1>Theme Analysis Results</h1>
                    <p>Generated on """ + datetime.datetime.now().strftime("%d %B %Y, %H:%M") + """</p>
                </div>
                
                <div class="summary-card">
                    <div class="summary-box">
                        <div class="summary-number">""" + str(len(highlighted_texts)) + """</div>
                        <div class="summary-label">Documents Analyzed</div>
                    </div>
                    <div class="summary-box">
                        <div class="summary-number">""" + str(len(results_df)) + """</div>
                        <div class="summary-label">Theme Identifications</div>
                    </div>
                    <div class="summary-box">
                        <div class="summary-number">""" + str(len(results_df["Framework"].unique())) + """</div>
                        <div class="summary-label">Frameworks</div>
                    </div>
                </div>
                
                <!-- Add legend explaining gradients -->
                <div class="legend-container">
                    <div class="legend-title">Theme Color Guide</div>
                    <div>When text contains multiple themes, a gradient background is used to show all applicable themes. Check the tooltip for details.</div>
                </div>
            """
    
        # Add framework summary
        html_content += """
                <h2>Framework Summary</h2>
                <table class="theme-info">
                    <tr>
                        <th>Framework</th>
                        <th>Number of Themes</th>
                        <th>Number of Documents</th>
                    </tr>
            """
    
        for framework in results_df["Framework"].unique():
            framework_results = results_df[results_df["Framework"] == framework]
            num_themes = len(framework_results["Theme"].unique())
            num_docs = len(framework_results["Record ID"].unique())
    
            html_content += f"""
                    <tr>
                        <td>{framework}</td>
                        <td>{num_themes}</td>
                        <td>{num_docs}</td>
                    </tr>
                """
    
        html_content += """
                </table>
            """
    
        # Add each record with its themes and highlighted text
        html_content += "<h2>Document Analysis</h2>"
    
        for record_id, themes in report_themes.items():
            if record_id in highlighted_texts:
                record_title = next(
                    (row["Title"] for _, row in results_df.iterrows() if row.get("Record ID") == record_id),
                    f"Document {record_id}"
                )
    
                html_content += f"""
                    <div class="record-container">
                        <h2>Document: {record_title}</h2>
                        
                        <div class="theme-info">
                            <h3>Identified Themes</h3>
                            <table>
                                <tr>
                                    <th>Framework</th>
                                    <th>Theme</th>
                                    <th>Confidence</th>
                                    <th>Score</th>
                                    <th>Matched Keywords</th>
                                    <th>Color</th>
                                </tr>
                    """
    
                # Add theme rows with consistent styling and no gradients
                for theme_info in sorted(themes, key=lambda x: (x["framework"], -x.get("score", 0))):
                    theme_color = theme_info["color"]
                    
                    html_content += f"""
                                <tr style="background-color: {theme_color};">
                                    <td>{theme_info['framework']}</td>
                                    <td>{theme_info['theme']}</td>
                                    <td>{theme_info.get('confidence', '')}</td>
                                    <td>{round(theme_info.get('score', 0), 3)}</td>
                                    <td>{theme_info.get('keywords', '')}</td>
                                    <td><div class="theme-color-box" style="background-color:{theme_color};"></div></td>
                                </tr>
                        """
    
                html_content += """
                            </table>
                        </div>
                        
                        <div class="highlighted-text">
                            <h3>Text with Highlighted Keywords</h3>
                    """
    
                # Add highlighted text
                html_content += highlighted_texts[record_id]
    
                html_content += """
                        </div>
                    </div>
                    """
    
        html_content += """
            </body>
            </html>
            """
    
        return html_content
    
    def _ensure_unique_theme_colors(self, results_df):
        """Ensure all themes have unique colors by checking and reassigning if needed"""
        from collections import defaultdict
        
        # First collect all theme keys
        theme_keys = set()
        for _, row in results_df.iterrows():
            if "Framework" in row and "Theme" in row:
                theme_key = f"{row['Framework']}_{row['Theme']}"
                theme_keys.add(theme_key)
        
        # Assign colors for any missing themes
        for theme_key in theme_keys:
            if theme_key not in self.theme_color_map:
                self._assign_unique_theme_color(theme_key)
        
        # Check for duplicate colors and fix them
        color_to_themes = defaultdict(list)
        for theme_key in theme_keys:
            color = self.theme_color_map[theme_key]
            color_to_themes[color].append(theme_key)
        
        # Reassign colors for themes with duplicates
        for color, duplicate_themes in color_to_themes.items():
            if len(duplicate_themes) > 1:
                # Keep the first theme's color, reassign others
                for theme_key in duplicate_themes[1:]:
                    self._assign_unique_theme_color(theme_key)
    
    def _assign_unique_theme_color(self, theme_key):
        """Assign a unique color to a theme, ensuring no duplicates"""
        # Get currently used colors
        used_colors = set(self.theme_color_map.values())
        
        # Find an unused color from our palette
        for color in self.theme_colors:
            if color not in used_colors:
                self.theme_color_map[theme_key] = color
                return

        
        def random_hex_color():
            """Generate a random pastel color that's visually distinct"""
            # Higher base value (200) ensures lighter/pastel colors
            r = random.randint(180, 240)
            g = random.randint(180, 240)
            b = random.randint(180, 240)
            return f"#{r:02x}{g:02x}{b:02x}"
        
        # Generate colors until we find one that's not too similar to existing ones
        while True:
            new_color = random_hex_color()
            if new_color not in used_colors:
                self.theme_color_map[theme_key] = new_color
                break

    
    def _create_gradient_css(self, colors):
        """Create a CSS gradient string from a list of colors
        
        For 2 colors: simple diagonal gradient
        For 3+ colors: striped gradient with equal divisions
        """
        if len(colors) == 2:
            # Simple diagonal gradient for 2 colors
            return f"linear-gradient(135deg, {colors[0]} 50%, {colors[1]} 50%)"
        else:
            # Create striped gradient for 3+ colors
            stops = []
            segment_size = 100.0 / len(colors)
            
            for i, color in enumerate(colors):
                start = i * segment_size
                end = (i + 1) * segment_size
                
                # Add color stop
                stops.append(f"{color} {start:.1f}%")
                stops.append(f"{color} {end:.1f}%")
            
            return f"linear-gradient(135deg, {', '.join(stops)})"




######



    

    
    def _get_theme_color(self, theme_key):
        """Get a consistent color for a specific theme"""
        # If this theme already has an assigned color, use it
        if theme_key in self.theme_color_map:
            return self.theme_color_map[theme_key]

        # Extract framework and theme from the theme_key (format: "framework_theme")
        parts = theme_key.split("_", 1)
        framework = parts[0] if len(parts) > 0 else "unknown"

        # Count existing colors for this framework
        framework_count = sum(
            1
            for existing_key in self.theme_color_map
            if existing_key.startswith(framework + "_")
        )

        # Assign the next available color from our palette
        color_idx = framework_count % len(self.theme_colors)
        assigned_color = self.theme_colors[color_idx]

        # Store the assignment for future consistency
        self.theme_color_map[theme_key] = assigned_color
        return assigned_color

    def analyze_document(self, text):
        """Analyze document text for themes and highlight sentences containing theme keywords"""
        if not isinstance(text, str) or not text.strip():
            return {}, {}

        # Get full document embedding
        document_embedding = self.get_bert_embedding(text)
        text_length = len(text.split())

        framework_themes = {}
        theme_highlights = {}

        for framework_name, framework_theme_list in self.frameworks.items():
            # Track keyword matches across the entire document
            all_keyword_matches = []

            # First pass: identify all keyword matches and their contexts
            theme_matches = []
            for theme in framework_theme_list:
                # Find all sentence positions containing any matching keywords
                sentence_positions = self._find_sentence_positions(
                    text, theme["keywords"]
                )

                # Extract keywords from sentence positions
                keyword_matches = []
                match_contexts = []

                for _, _, keywords_str, _ in sentence_positions:
                    for keyword in keywords_str.split(", "):
                        if keyword not in keyword_matches:
                            keyword_matches.append(keyword)

                            # Get contextual embeddings for each keyword occurrence
                            context_embedding = self._get_contextual_embedding(
                                text, keyword, self.config["context_window_size"]
                            )
                            match_contexts.append(context_embedding)

                # Calculate semantic similarity with theme description
                theme_description = theme["name"] + ": " + ", ".join(theme["keywords"])
                theme_embedding = self.get_bert_embedding(theme_description)
                theme_doc_similarity = cosine_similarity(
                    [document_embedding], [theme_embedding]
                )[0][0]

                # Calculate context similarities if available
                context_similarities = []
                if match_contexts:
                    for context_emb in match_contexts:
                        sim = cosine_similarity([context_emb], [theme_embedding])[0][0]
                        context_similarities.append(sim)

                # Use max context similarity if available, otherwise use document similarity
                max_context_similarity = (
                    max(context_similarities) if context_similarities else 0
                )
                semantic_similarity = max(theme_doc_similarity, max_context_similarity)

                # Calculate combined score
                combined_score = self._calculate_combined_score(
                    semantic_similarity, len(keyword_matches), text_length
                )

                if (
                    keyword_matches
                    and combined_score >= self.config["base_similarity_threshold"]
                ):
                    theme_matches.append(
                        {
                            "theme": theme["name"],
                            "semantic_similarity": round(semantic_similarity, 3),
                            "combined_score": round(combined_score, 3),
                            "matched_keywords": ", ".join(keyword_matches),
                            "keyword_count": len(keyword_matches),
                            "sentence_positions": sentence_positions,  # Store sentence positions for highlighting
                        }
                    )

                    all_keyword_matches.extend(keyword_matches)

            # Sort by combined score
            theme_matches.sort(key=lambda x: x["combined_score"], reverse=True)

            # Limit number of themes
            top_theme_matches = theme_matches[: self.config["max_themes_per_framework"]]

            # Store theme matches and their highlighting info
            if top_theme_matches:
                # Count keywords to identify potential overlaps
                keyword_counter = Counter(all_keyword_matches)

                # Filter out themes with high keyword overlap and lower scores
                final_themes = []
                used_keywords = set()

                for theme_match in top_theme_matches:
                    # Check if this theme adds unique keywords
                    theme_keywords = set(theme_match["matched_keywords"].split(", "))
                    unique_keywords = theme_keywords - used_keywords

                    # If theme adds unique keywords or has high score, include it
                    if unique_keywords or theme_match["combined_score"] > 0.75:
                        # Store the theme data
                        theme_match_data = {
                            "theme": theme_match["theme"],
                            "semantic_similarity": theme_match["semantic_similarity"],
                            "combined_score": theme_match["combined_score"],
                            "matched_keywords": theme_match["matched_keywords"],
                            "keyword_count": theme_match["keyword_count"],
                        }
                        final_themes.append(theme_match_data)

                        # Store the highlighting positions separately
                        theme_key = f"{framework_name}_{theme_match['theme']}"
                        theme_highlights[theme_key] = theme_match["sentence_positions"]

                        used_keywords.update(theme_keywords)

                framework_themes[framework_name] = final_themes
            else:
                framework_themes[framework_name] = []

        return framework_themes, theme_highlights
    def _get_yorkshire_framework(self):
        """Yorkshire Contributory factors framework themes mapped exactly to the official framework structure"""
        return [{
                    "name":"Situational- Team Factors",
                    "keywords":[
                        "conflicting team goals",
                        "poor delegation",
                        "poor feedback",
                        "team function",
                        "poor communication"
                    ]
                },
                {
                    "name":"Situational- Individual Staff Factors ",
                    "keywords":[
                        "tiredness",
                        "worker was stressed",
                        "distraction",
                        "inexperience",
                        "unfamiliar with the equipment",
                        "new to the role",
                        "hadn't had a break",
                        "felt overwhelmed",
                        "preoccupied",
                        "staff were sick"
                    ]
                },
                {
                    "name":"Situational- Task Characteristics",
                    "keywords":[
                        "unfamiliar task",
                        "difficult task",
                        "monotonous task",
                        "unclear task",
                        "multiple steps",
                        "had to multitask",
                        "rare procedure"
                    ]
                },
                {
                    "name":"Situational- Patient Factors",
                    "keywords":[
                        "language barrier",
                        "uncooperative",
                        "medical history",
                        "unusual physiology",
                        "intoxicated",
                        "high-risk",
                        "unstable",
                        "speech impairment",
                        "dementia",
                        "aggressive behaviour",
                        "elderly",
                        "fragile",
                        "too unwell",
                        "not suitable"
                    ]
                },
                {
                    "name":"Local Working Conditions- Workload and Staffing Issues",
                    "keywords":[
                        "high workload",
                        "heavy workload",
                        "insufficient staff",
                        "staff sickness",
                        "inexperienced staff"
                    ]
                },
                {
                    "name":"Local Working Conditions- Supervision and Leadership",
                    "keywords":[
                        "inappropriate delegation",
                        "remote supervision",
                        "manager was absent",
                        "no senior presence",
                        "limited supervision",
                        "no supervision",
                        "lack of regulation"
                    ]
                },
                {
                    "name":"Local Working Conditions- Drugs, Equipment and Supplies",
                    "keywords":[
                        "unavailable drugs",
                        "stock issues",
                        "faulty equipment",
                        "poor maintenance",
                        "no supplies",
                        "no clear labelling",
                        "battery was flat",
                        "not restocked",
                        "missing equipment",
                        "mechanism",
                        "material"
                    ]
                },
                {
                    "name":"Local Working Conditions- Lines of Responsibility",
                    "keywords":[
                        "assumed task already done",
                        "no team leader",
                        "overlap in responsibility",
                        "not sure who",
                        "mutlple allocation",
                        "no allocation"
                    ]
                },
                {
                    "name":"Local Working Conditions- Management of Staff and Staffing Levels",
                    "keywords":[
                        "short-staffed",
                        "under pressure",
                        "too busy",
                        "no time",
                        "no cover",
                        "no one available",
                        "no support",
                        "poor communication"
                    ]
                },
                {
                    "name":"Organisational Factors- Physical Environment",
                    "keywords":[
                        "poor layout",
                        "poor visibility",
                        "lack of space",
                        "excessive noise",
                        "too hot",
                        "too cold",
                        "poor lighting",
                        "poor access to patient",
                        "dust",
                        "exposure to",
                        "unsafe conditions",
                        "working conditions",
                        "smoke detector",
                        "smoky",
                        "smoke filled"
                    ]
                },
                {
                    "name":"Organisational Factors- Support from other departments",
                    "keywords":[
                        "IT support",
                        "HR",
                        "clinical services",
                        "radiology",
                        "pharmacy",
                        "blood bank",
                        "medical departmant",
                        "GP",
                        "ambulance was delayed",
                        "ambulances facilities"
                    ]
                },
                {
                    "name":"Organisational Factors- Scheduling and Bed Management",
                    "keywords":[
                        "delay in the provision of care",
                        "difficulties finding a bed",
                        "ward transfer",
                        "poor out of hours support",
                        "no beds available",
                        "hospital in full capacity",
                        "discharged to make space"
                    ]
                },
                {
                    "name":"Organisational Factors- Staff Training and Education",
                    "keywords":[
                        "inadequate training",
                        "less teaching time",
                        "no regular updates",
                        "training not standardised",
                        "lack of knowledge",
                        "no formal training",
                        "training overdue",
                        "no risk assessment",
                        "outside the experience",
                        "outside their capabilities",
                        "trained"
                    ]
                },
                {
                    "name":"Organisational Factors- Policies and Procedures",
                    "keywords":[
                        "no policy",
                        "policy not followed",
                        "unclear procedure",
                        "lack of guidance",
                        "inconsitent policies",
                        "complex procedure",
                        "conflicting instructions",
                        "outdated policy"
                    ]
                },
                {
                    "name": "Organisational Factors- Escalation/referral factor",
                    "keywords": [
                        "escalation/referral factor",
                        "escalation/referral",
                        "including fresh eyes reviews",
                        "specialist referral",
                        "delay in escalation",
                        "specialist review",
                        "senior input",
                        "interdisciplinary referral",
                        "escalation delay",
                        "consultant opinion"
                    ]
                },
                {
                    "name":"External Factors- Design of Equipment Supplies and Drugs",
                    "keywords":[
                        "complicated equipment design",
                        "equipment not fit for purpose",
                        "similar drug names",
                        "unclear labelling",
                        "equipment",
                        "design",
                        "mechanism",
                        "equipped"
                    ]
                },
                {
                    "name":"External Factors- National Policies",
                    "keywords":[
                        "commisioned resources",
                        "national screening policy",
                        "government organisations",
                        "inteference",
                        "national medical standards",
                        "emergencey department",
                        "locally",
                        "nationally",
                        "due to the rules and regulations"
                    ]
                },
                {
                    "name":"Communication and Culture- Safety Culture",
                    "keywords":[
                        "safety awareness",
                        "fear of documenting errors",
                        "risk management",
                        "afraid of senior",
                        "mistakes punishment",
                        "no risk assessment",
                        "safety notices",
                        "no safety"
                    ]
                },
                {
                    "name":"Communication and Culture- Verbal and Written Communication",
                    "keywords":[
                        "poor communication",
                        "inappropriate abbreviations",
                        "handover issues",
                        "lack of notes",
                        "unable to read notes",
                        "legibility",
                        "notes availability"
                    ]
                },
                {
                    "name":"Human Error- Slips or Lapses",
                    "keywords":[
                        "wrong medicine",
                        "wrong syringe",
                        "wrong option",
                        "wrong button",
                        "wrong patient",
                        "entered incorrect data",
                        "missed a step",
                        "forgot",
                        "overlooked alert",
                        "wrong decision",
                        "misunderstanding",
                        "misjudged risk",
                        "wrong protocol"
                    ]
                },
                {
                    "name":"Human Error- Violations",
                    "keywords":[
                        "didn't follow it",
                        "done it before",
                        "skipped a step",
                        "short on time",
                        "quicker this way",
                        "skipping checks",
                        "skipped double check",
                        "knew the policy",
                        "presumed",
                        "not followed"
                    ]
                }
                # {
                #     "name":"Situational- Team Factors",
                #     "keywords":[
                #         "conflicting team goals",
                #         "poor delegation",
                #         "respect",
                #         "poor feedback",
                #         "team function",
                #         "poor communication"
                #     ]
                # },
                # {
                #     "name":"Situational- Individual Staff Factors ",
                #     "keywords":[
                #         "tiredness",
                #         "worker was stressed",
                #         "distraction",
                #         "inexperience",
                #         "unfamiliar with the equipment",
                #         "new to the role",
                #         "hadn't had a break",
                #         "felt overwhelmed",
                #         "preoccupied",
                #         "staff were sick"
                #     ]
                # },
                # {
                #     "name":"Situational- Task Characteristics",
                #     "keywords":[
                #         "unfamiliar task",
                #         "difficult task",
                #         "monotonous task",
                #         "unclear task",
                #         "multiple steps",
                #         "had to multitask",
                #         "rare procedure"
                #     ]
                # },
                # {
                #     "name":"Situational- Patient Factors",
                #     "keywords":[
                #         "language barrier",
                #         "uncooperative",
                #         "medical history",
                #         "unusual physiology",
                #         "intoxicated",
                #         "high-risk",
                #         "unstable",
                #         "speech impairment",
                #         "dementia",
                #         "aggressive behaviour",
                #         "elderly",
                #         "fragile",
                #         "too unwell",
                #         "not suitable"
                #     ]
                # },
                # {
                #     "name":"Local Working Conditions- Workload and Staffing Issues",
                #     "keywords":[
                #         "high workload",
                #         "heavy workload",
                #         "insufficient staff",
                #         "staff sickness",
                #         "inexperienced staff"
                #     ]
                # },
                # {
                #     "name":"Local Working Conditions- Supervision and Leadership",
                #     "keywords":[
                #         "inappropriate delegation",
                #         "remote supervision",
                #         "manager was absent",
                #         "no senior presence",
                #         "limited supervision",
                #         "no supervision",
                #         "lack of regulation"
                #     ]
                # },
                # {
                #     "name":"Local Working Conditions- Drugs, Equipment and Supplies",
                #     "keywords":[
                #         "unavailable drugs",
                #         "stock issues",
                #         "faulty equipment",
                #         "poor maintenance",
                #         "no supplies",
                #         "no clear labelling",
                #         "battery was flat",
                #         "not restocked",
                #         "missing equipment",
                #         "mechanism"
                #     ]
                # },
                # {
                #     "name":"Local Working Conditions- Lines of Responsibility",
                #     "keywords":[
                #         "assumed task already done",
                #         "no team leader",
                #         "overlap in responsibility",
                #         "not sure who",
                #         "mutlple allocation",
                #         "no allocation"
                #     ]
                # },
                # {
                #     "name":"Local Working Conditions- Management of Staff and Staffing Levels",
                #     "keywords":[
                #         "short-staffed",
                #         "under pressure",
                #         "too busy",
                #         "no time",
                #         "no cover",
                #         "no one available",
                #         "no support",
                #         "poor communication"
                #     ]
                # },
                # {
                #     "name":"Organisational Factors- Physical Environment",
                #     "keywords":[
                #         "poor layout",
                #         "poor visibility",
                #         "lack of space",
                #         "excessive noise",
                #         "too hot",
                #         "too cold",
                #         "poor lighting",
                #         "poor access to patient",
                #         "dust",
                #         "exposure to",
                #         "unsafe conditions",
                #         "working conditions",
                #         "smoke detector",
                #         "smoky",
                #         "smoke filled"
                #     ]
                # },
                # {
                #     "name":"Organisational Factors- Support from other departments",
                #     "keywords":[
                #         "IT support",
                #         "HR",
                #         "clinical services",
                #         "radiology",
                #         "pharmacy",
                #         "blood bank",
                #         "medical departmant",
                #         "GP",
                #         "ambulance was delayed",
                #         "ambulances facilities"
                #     ]
                # },
                # {
                #     "name":"Organisational Factors- Scheduling and Bed Management",
                #     "keywords":[
                #         "delay in the provision of care",
                #         "difficulties finding a bed",
                #         "ward transfer",
                #         "poor out of hours support",
                #         "no beds available",
                #         "hospital in full capacity",
                #         "discharged to make space"
                #     ]
                # },
                # {
                #     "name":"Organisational Factors- Staff Training and Education",
                #     "keywords":[
                #         "inadequate training",
                #         "less teaching time",
                #         "no regular updates",
                #         "training not standardised",
                #         "lack of knowledge",
                #         "no formal training",
                #         "training overdue",
                #         "no risk assessment",
                #         "outside the experience",
                #         "outside their capabilities",
                #         "trained"
                #     ]
                # },
                # {
                #     "name":"Organisational Factors- Policies and Procedures",
                #     "keywords":[
                #         "no policy",
                #         "policy not followed",
                #         "unclear procedure",
                #         "lack of guidance",
                #         "inconsitent policies",
                #         "complex procedure",
                #         "conflicting instructions",
                #         "outdated policy"
                #     ]
                # },
                # {
                #     "name":"External Factors- Design of Equipment Supplies and Drugs",
                #     "keywords":[
                #         "complicated equipment design",
                #         "equipment not fit for purpose",
                #         "similar drug names",
                #         "unclear labelling",
                #         "equipment",
                #         "design",
                #         "mechanism"
                #     ]
                # },
                # {
                #     "name":"External Factors- National Policies",
                #     "keywords":[
                #         "commisioned resources",
                #         "national screening policy",
                #         "government organisations",
                #         "inteference",
                #         "national medical standards",
                #         "emergencey department",
                #         "locally",
                #         "nationally",
                #         "regulation"
                #     ]
                # },
                # {
                #     "name":"Communication and Culture- Safety Culture",
                #     "keywords":[
                #         "safety awareness",
                #         "fear of documenting errors",
                #         "risk management",
                #         "afraid of senior",
                #         "mistakes punishment",
                #         "no risk assessment",
                #         "safety notices",
                #         "no safety"
                #     ]
                # },
                # {
                #     "name":"Communication and Culture- Verbal and Written Communication",
                #     "keywords":[
                #         "poor communication",
                #         "inappropriate abbreviations",
                #         "handover issues",
                #         "lack of notes",
                #         "unable to read notes",
                #         "legibility",
                #         "notes availability"
                #     ]
                # },
                # {
                #     "name":"Active Failures- Mistakes",
                #     "keywords":[
                #         "wrong decision",
                #         "thought it was okay",
                #         "presumed",
                #         "incorrect interpretation",
                #         "misunderstanding",
                #         "misjudged risk",
                #         "wrong protocol"
                #     ]
                # },
                # {
                #     "name":"Active Failures- Slips and Lapses",
                #     "keywords":[
                #         "wrong medicine",
                #         "wrong syringe",
                #         "wrong option",
                #         "wrong button",
                #         "wrong patient",
                #         "entered incorrect data",
                #         "missed a step",
                #         "forgot",
                #         "overlooked alert"
                #     ]
                # },
                # {
                #     "name":"Active Failures- Violations",
                #     "keywords":[
                #         "didn't follow it",
                #         "done it before",
                #         "skipped a step",
                #         "short on time",
                #         "quicker this way",
                #         "skipping checks",
                #         "skipped double check",
                #         "knew the policy"
                #     ]
                # }
            
            ]
        
    def _get_isirch_framework(self):
        """I-SIRCh framework themes mapped exactly to the official framework structure"""
        return [
            {
                "name": "External - Policy factor",
                "keywords": ["policy factor", "policy", "factor"],
            },
            {
                "name": "External - Societal factor",
                "keywords": ["societal factor", "societal", "factor"],
            },
            {
                "name": "External - Economic factor",
                "keywords": ["economic factor", "economic", "factor"],
            },
            {"name": "External - COVID ✓", "keywords": ["covid ✓", "covid"]},
            {
                "name": "External - Geographical factor (e.g. Location of patient)",
                "keywords": [
                    "geographical factor",
                    "geographical",
                    "factor",
                    "location of patient",
                ],
            },
            {
                "name": "Internal - Physical layout and Environment",
                "keywords": [
                    "physical layout and environment",
                    "physical",
                    "layout",
                    "environment",
                ],
            },
            {
                "name": "Internal - Acuity (e.g., capacity of the maternity unit as a whole)",
                "keywords": ["acuity", "capacity of the maternity unit as a whole"],
            },
            {
                "name": "Internal - Availability (e.g., operating theatres)",
                "keywords": ["availability", "operating theatres"],
            },
            {
                "name": "Internal - Time of day (e.g., night working or day of the week)",
                "keywords": ["time of day", "time", "night working or day of the week"],
            },
            {
                "name": "Organisation - Team culture factor (e.g., patient safety culture)",
                "keywords": [
                    "team culture factor",
                    "team",
                    "culture",
                    "factor",
                    "patient safety culture",
                ],
            },
            {
                "name": "Organisation - Incentive factor (e.g., performance evaluation)",
                "keywords": [
                    "incentive factor",
                    "incentive",
                    "factor",
                    "performance evaluation",
                ],
            },
            {"name": "Organisation - Teamworking", "keywords": ["teamworking"]},
            {
                "name": "Organisation - Communication factor",
                "keywords": ["communication factor", "communication", "factor"],
            },
            {
                "name": "Organisation - Communication factor - Between staff",
                "keywords": ["between staff", "between", "staff"],
            },
            {
                "name": "Organisation - Communication factor - Between staff and patient (verbal)",
                "keywords": [
                    "between staff and patient",
                    "between",
                    "staff",
                    "patient",
                    "verbal",
                ],
            },
            {"name": "Organisation - Documentation", "keywords": ["documentation"]},
            {
                "name": "Organisation - Escalation/referral factor (including fresh eyes reviews)",
                "keywords": [
                    "escalation/referral factor",
                    "escalation/referral",
                    "factor",
                    "including fresh eyes reviews",
                    "specialist referral",
                    "delay in escalation",
                    "specialist review",
                    "senior input",
                    "interdisciplinary referral",
                    "escalation delay",
                    "consultant opinion",
                ],
            },
            {
                "name": "Organisation - National and/or local guidance",
                "keywords": [
                    "national and/or local guidance",
                    "national",
                    "local",
                    "guidance",
                    "national screening",
                    "screening program",
                    "standard implementation",
                    "standardized screening",
                    "protocol adherence",
                ],
            },
            {
                "name": "Organisation - Language barrier",
                "keywords": ["language barrier", "language", "barrier"],
            },
            {
                "name": "Jobs/Task - Assessment, investigation, testing, screening (e.g., holistic review)",
                "keywords": [
                    "assessment, investigation, testing, screening",
                    "assessment,",
                    "investigation,",
                    "testing,",
                    "screening",
                    "holistic review",
                    "specimen",
                    "sample",
                    "laboratory",
                    "test result",
                    "abnormal finding",
                    "test interpretation",
                ],
            },
            {
                "name": "Jobs/Task - Care planning",
                "keywords": ["care planning", "care", "planning"],
            },
            {
                "name": "Jobs/Task - Dispensing, administering",
                "keywords": [
                    "dispensing, administering",
                    "dispensing,",
                    "administering",
                ],
            },
            {"name": "Jobs/Task - Monitoring", "keywords": ["monitoring"]},
            {
                "name": "Jobs/Task - Risk assessment",
                "keywords": ["risk assessment", "risk", "assessment"],
            },
            {
                "name": "Jobs/Task - Situation awareness (e.g., loss of helicopter view)",
                "keywords": [
                    "situation awareness",
                    "situation",
                    "awareness",
                    "loss of helicopter view",
                ],
            },
            {
                "name": "Jobs/Task - Obstetric review",
                "keywords": ["obstetric review", "obstetric", "review"],
            },
            {"name": "Technologies - Issues", "keywords": ["issues"]},
            {
                "name": "Technologies - Interpretation (e.g., CTG)",
                "keywords": ["interpretation", "ctg"],
            },
            {
                "name": "Person - Patient (characteristics and performance)",
                "keywords": ["patient", "characteristics and performance"],
            },
            {
                "name": "Person - Patient (characteristics and performance) - Characteristics",
                "keywords": ["characteristics", "patient characteristics"],
            },
            {
                "name": "Person - Patient (characteristics and performance) - Characteristics - Physical characteristics",
                "keywords": ["physical characteristics", "physical", "characteristics"],
            },
            {
                "name": "Person - Patient (characteristics and performance) - Characteristics - Psychological characteristics (e.g., stress, mental health)",
                "keywords": [
                    "psychological characteristics",
                    "psychological",
                    "characteristics",
                    "stress",
                    "mental health",
                ],
            },
            {
                "name": "Person - Patient (characteristics and performance) - Characteristics - Language competence (English)",
                "keywords": [
                    "language competence",
                    "language",
                    "competence",
                    "english",
                ],
            },
            {
                "name": "Person - Patient (characteristics and performance) - Characteristics - Disability (e.g., hearing problems)",
                "keywords": ["disability", "hearing problems"],
            },
            {
                "name": "Person - Patient (characteristics and performance) - Characteristics - Training and education (e.g., attendance at ante-natal classes)",
                "keywords": [
                    "training and education",
                    "training",
                    "education",
                    "attendance at ante-natal classes",
                ],
            },
            {
                "name": "Person - Patient (characteristics and performance) - Characteristics - Record of attendance (e.g., failure to attend antenatal classes)",
                "keywords": [
                    "record of attendance",
                    "record",
                    "attendance",
                    "failure to attend antenatal classes",
                ],
            },
            {
                "name": "Person - Patient (characteristics and performance) - Performance",
                "keywords": ["performance", "patient performance"],
            },
            {
                "name": "Person - Staff (characteristics and performance)",
                "keywords": ["staff", "characteristics and performance"],
            },
            {
                "name": "Person - Staff (characteristics and performance) - Characteristics",
                "keywords": ["characteristics", "staff characteristics"],
            },
            {
                "name": "Person - Staff (characteristics and performance) - Performance",
                "keywords": ["performance", "staff performance"],
            },
        ]

    def _get_house_of_commons_themes(self):
        """House of Commons themes mapped exactly to the official document"""
        return [
            {
                "name": "Communication",
                "keywords": [
                    "communication",
                    "dismissed",
                    "listened",
                    "concerns not taken seriously",
                    "concerns",
                    "seriously",
                ],
            },
            {
                "name": "Fragmented care",
                "keywords": [
                    "fragmented care",
                    "fragmented",
                    "care",
                    "spread",
                    "poorly",
                    "communicating",
                    "providers",
                    "no clear coordination",
                    "clear",
                    "coordination",
                ],
            },
            {
                "name": "Guidance gaps",
                "keywords": [
                    "guidance gaps",
                    "guidance",
                    "gaps",
                    "information",
                    "needs",
                    "optimal",
                    "minority",
                ],
            },
            {
                "name": "Pre-existing conditions and comorbidities",
                "keywords": [
                    "pre-existing conditions and comorbidities",
                    "pre-existing",
                    "conditions",
                    "comorbidities",
                    "overrepresented",
                    "ethnic",
                    "minority",
                    "contribute",
                    "higher",
                    "mortality",
                ],
            },
            {
                "name": "Inadequate maternity care",
                "keywords": [
                    "inadequate maternity care",
                    "inadequate",
                    "maternity",
                    "care",
                    "individualized",
                    "culturally",
                    "sensitive",
                ],
            },
            {
                "name": "Care quality and access issues",
                "keywords": [
                    "microaggressions and racism",
                    "microaggressions",
                    "racism",
                    "implicit/explicit",
                    "impacts",
                    "access",
                    "treatment",
                    "quality",
                    "stereotyping",
                ],
            },
            {
                "name": "Socioeconomic factors and deprivation",
                "keywords": [
                    "socioeconomic factors and deprivation",
                    "socioeconomic",
                    "factors",
                    "deprivation",
                    "links to poor outcomes",
                    "links",
                    "outcomes",
                    "minority",
                    "overrepresented",
                    "deprived",
                    "areas",
                ],
            },
            {
                "name": "Biases and stereotyping",
                "keywords": [
                    "biases and stereotyping",
                    "biases",
                    "stereotyping",
                    "perpetuation",
                    "stereotypes",
                    "providers",
                ],
            },
            {
                "name": "Consent/agency",
                "keywords": [
                    "consent/agency",
                    "consent",
                    "agency",
                    "informed consent",
                    "agency over care decisions",
                    "informed",
                    "decisions",
                ],
            },
            {
                "name": "Dignity/respect",
                "keywords": [
                    "dignity/respect",
                    "dignity",
                    "respect",
                    "neglectful",
                    "lacking",
                    "discrimination faced",
                    "discrimination",
                    "faced",
                ],
            },
        ]

    def _get_extended_themes(self):
        """Extended Analysis themes with unique concepts not covered in I-SIRCh or House of Commons frameworks"""
        return [
            {
                "name": "Procedural and Process Failures",
                "keywords": [
                    "procedure failure",
                    "process breakdown",
                    "protocol breach",
                    "standard violation",
                    "workflow issue",
                    "operational failure",
                    "process gap",
                    "procedural deviation",
                    "system failure",
                    "process error",
                    "workflow disruption",
                    "task failure",
                ],
            },
            {
                "name": "Medication safety",
                "keywords": [
                    "medication safety",
                    "medication",
                    "safety",
                    "drug error",
                    "prescription",
                    "drug administration",
                    "medication error",
                    "adverse reaction",
                    "medication reconciliation",
                ],
            },
            {
                "name": "Resource allocation",
                "keywords": [
                    "resource allocation",
                    "resource",
                    "allocation",
                    "resource management",
                    "resource constraints",
                    "prioritisation",
                    "resource distribution",
                    "staffing levels",
                    "staff shortage",
                    "budget constraints",
                ],
            },
            {
                "name": "Facility and Equipment Issues",
                "keywords": [
                    "facility",
                    "equipment",
                    "maintenance",
                    "infrastructure",
                    "device failure",
                    "equipment malfunction",
                    "equipment availability",
                    "technical failure",
                    "equipment maintenance",
                    "facility limitations",
                ],
            },
            {
                "name": "Emergency preparedness",
                "keywords": [
                    "emergency preparedness",
                    "emergency protocol",
                    "emergency response",
                    "crisis management",
                    "contingency planning",
                    "disaster readiness",
                    "emergency training",
                    "rapid response",
                ],
            },
            {
                "name": "Staff Wellbeing and Burnout",
                "keywords": [
                    "burnout",
                    "staff wellbeing",
                    "resilience",
                    "psychological safety",
                    "stress management",
                    "compassion fatigue",
                    "work-life balance",
                    "staff support",
                    "mental health",
                    "emotional burden",
                ],
            },
            {
                "name": "Ethical considerations",
                "keywords": [
                    "ethical dilemma",
                    "ethical decision",
                    "moral distress",
                    "ethical conflict",
                    "value conflict",
                    "ethics committee",
                    "moral judgment",
                    "conscientious objection",
                    "ethical framework",
                ],
            },
            {
                "name": "Diagnostic process",
                "keywords": [
                    "diagnostic error",
                    "misdiagnosis",
                    "delayed diagnosis",
                    "diagnostic uncertainty",
                    "diagnostic reasoning",
                    "differential diagnosis",
                    "diagnostic testing",
                    "diagnostic accuracy",
                    "test interpretation",
                ],
            },
            {
                "name": "Post-Event Learning and Improvement",
                "keywords": [
                    "incident learning",
                    "corrective action",
                    "improvement plan",
                    "feedback loop",
                    "lessons learned",
                    "action tracking",
                    "improvement verification",
                    "learning culture",
                    "incident review",
                    "recommendation implementation",
                    "systemic improvement",
                    "organisational learning",
                ],
            },
            {
                "name": "Electronic Health Record Issues",
                "keywords": [
                    "electronic health record",
                    "ehr issue",
                    "alert fatigue",
                    "interface design",
                    "copy-paste error",
                    "dropdown selection",
                    "clinical decision support",
                    "digital documentation",
                    "system integration",
                    "information retrieval",
                    "data entry error",
                    "electronic alert",
                ],
            },
            {
                "name": "Time-Critical Interventions",
                "keywords": [
                    "time-critical",
                    "delayed intervention",
                    "response time",
                    "golden hour",
                    "deterioration recognition",
                    "rapid response",
                    "timely treatment",
                    "intervention delay",
                    "time sensitivity",
                    "critical timing",
                    "delayed recognition",
                    "prompt action",
                    "urgent intervention",
                    "emergency response",
                    "time-sensitive decision",
                    "immediate action",
                    "rapid assessment",
                ],
            },
            {
                "name": "Human Factors and Cognitive Aspects",
                "keywords": [
                    "cognitive bias",
                    "situational awareness",
                    "attention management",
                    "visual perception",
                    "cognitive overload",
                    "decision heuristic",
                    "tunnel vision",
                    "confirmation bias",
                    "fixation error",
                    "anchoring bias",
                    "memory limitation",
                    "cognitive fatigue",
                    "isolation decision-making",
                    "clinical confidence",
                    "professional authority",
                    "hierarchical barriers",
                    "professional autonomy",
                ],
            },
            {
                "name": "Service Design and Patient Flow",
                "keywords": [
                    "service design",
                    "patient flow",
                    "care pathway",
                    "bottleneck",
                    "patient journey",
                    "waiting time",
                    "system design",
                    "process mapping",
                    "patient transfer",
                    "capacity planning",
                    "workflow design",
                    "service bottleneck",
                ],
            },
            {
                "name": "Maternal and Neonatal Risk Factors",
                "keywords": [
                    "maternal risk",
                    "pregnancy complication",
                    "obstetric risk",
                    "neonatal risk",
                    "fetal risk",
                    "gestational diabetes",
                    "preeclampsia",
                    "placental issue",
                    "maternal age",
                    "parity",
                    "previous cesarean",
                    "multiple gestation",
                    "fetal growth restriction",
                    "prematurity",
                    "congenital anomaly",
                    "birth asphyxia",
                    "maternal obesity",
                    "maternal hypertension",
                    "maternal infection",
                    "obstetric hemorrhage",
                    "maternal cardiac",
                    "thromboembolism",
                ],
            },
            {
                "name": "Private vs. NHS Care Integration",
                "keywords": [
                    "private care",
                    "private midwife",
                    "private provider",
                    "NHS interface",
                    "care transition",
                    "private-public interface",
                    "independent provider",
                    "private consultation",
                    "private-NHS coordination",
                    "privately arranged care",
                    "independent midwife",
                    "cross-system communication",
                ],
            },
            {
                "name": "Peer Support and Supervision",
                "keywords": [
                    "peer support",
                    "collegial support",
                    "professional isolation",
                    "clinical supervision",
                    "peer review",
                    "case discussion",
                    "professional feedback",
                    "unsupported decision",
                    "lack of collegiality",
                    "professional network",
                    "mentoring",
                    "supervision",
                ],
            },
            {
                "name": "Diagnostic Testing and Specimen Handling",
                "keywords": [
                    "specimen",
                    "sample",
                    "test result",
                    "laboratory",
                    "analysis",
                    "interpretation",
                    "abnormal finding",
                    "discolored",
                    "contamination",
                    "collection",
                    "processing",
                    "transportation",
                    "storage",
                    "labeling",
                    "amniocentesis",
                    "blood sample",
                ],
            },
        ]

    # New methods to add

    def _get_confidence_label(self, score):
        """Convert numerical score to confidence label"""
        if score >= 0.7:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"


    def create_detailed_results(self, data, content_column="Content"):
        """
        Analyze multiple documents and create detailed results with progress tracking.
        Enhanced to include additional metadata (coroner_name, coroner_area, year).
    
        Args:
            data (pd.DataFrame): DataFrame containing documents
            content_column (str): Name of the column containing text to analyze
    
        Returns:
            Tuple[pd.DataFrame, Dict]: (Results DataFrame, Dictionary of highlighted texts)
        """
    
        results = []
        highlighted_texts = {}
    
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        doc_count_text = st.empty()
    
        # Calculate total documents to process
        total_docs = len(data)
        doc_count_text.text(f"Processing 0/{total_docs} documents")
    
        # Process each document
        for idx, (i, row) in enumerate(data.iterrows()):
            # Update progress
            progress = (idx + 1) / total_docs
            progress_bar.progress(progress)
            status_text.text(
                f"Analyzing document {idx + 1}/{total_docs}: {row.get('Title', f'Document {i}')}"
            )
    
            # Skip empty content
            if pd.isna(row[content_column]) or row[content_column] == "":
                continue
    
            content = str(row[content_column])
    
            # Analyze themes and get highlights
            framework_themes, theme_highlights = self.analyze_document(content)
    
            # Create highlighted HTML for this document
            highlighted_html = self.create_highlighted_html(content, theme_highlights)
            highlighted_texts[i] = highlighted_html
    
            # Store results for each theme
            theme_count = 0
            for framework_name, themes in framework_themes.items():
                for theme in themes:
                    theme_count += 1
    
                    # Extract matched sentences for this theme
                    matched_sentences = []
                    theme_key = f"{framework_name}_{theme['theme']}"
                    if theme_key in theme_highlights:
                        for (
                            start_pos,
                            end_pos,
                            keywords_str,
                            sentence,
                        ) in theme_highlights[theme_key]:
                            matched_sentences.append(sentence)
    
                    # Join sentences if there are any
                    matched_text = (
                        "; ".join(matched_sentences) if matched_sentences else ""
                    )
    
                    # Prepare the base result dictionary with theme information
                    result_dict = {
                        "Record ID": i,
                        "Title": row.get("Title", f"Document {i}"),
                        "Framework": framework_name,
                        "Theme": theme["theme"],
                        "Confidence": self._get_confidence_label(
                            theme["combined_score"]
                        ),
                        "Combined Score": theme["combined_score"],
                        "Semantic_Similarity": theme["semantic_similarity"],
                        "Matched Keywords": theme["matched_keywords"],
                        "Matched Sentences": matched_text,
                    }
                    
                    # Add additional metadata fields from the original data
                    # Only add if they exist in the input data
                    if "coroner_name" in row:
                        result_dict["coroner_name"] = row.get("coroner_name", "")
                    
                    if "coroner_area" in row:
                        result_dict["coroner_area"] = row.get("coroner_area", "")
                    
                    if "year" in row:
                        result_dict["year"] = row.get("year", "")
                    elif "Year" in row:
                        result_dict["year"] = row.get("Year", "")
                    
                    # Add date_of_report if available
                    if "date_of_report" in row:
                        result_dict["date_of_report"] = row.get("date_of_report", "")
    
                    results.append(result_dict)
    
                # Update documents processed count with theme info
                doc_count_text.text(
                    f"Processed {idx + 1}/{total_docs} documents. Found {theme_count} themes in current document."
                )
    
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
    
        # Final count update
        if results:
            doc_count_text.text(
                f"Completed analysis of {total_docs} documents. Found {len(results)} total themes."
            )
        else:
            doc_count_text.text(
                f"Completed analysis, but no themes were identified in the documents."
            )
    
        # Create results DataFrame
        results_df = pd.DataFrame(results) if results else pd.DataFrame()
    
        return results_df, highlighted_texts

    def create_comprehensive_pdf(
        self, results_df, highlighted_texts, output_filename=None
    ):
        """
        Create a comprehensive PDF report with analysis results

        Args:
            results_df (pd.DataFrame): Results DataFrame
            highlighted_texts (Dict): Dictionary of highlighted texts
            output_filename (str, optional): Output filename

        Returns:
            str: Path to the created PDF file
        """

        # Generate default filename if not provided
        if output_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"theme_analysis_report_{timestamp}.pdf"

        # Use a tempfile for matplotlib to avoid file conflicts
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmpfile:
            temp_pdf_path = tmpfile.name

        # Create PDF with matplotlib
        with PdfPages(temp_pdf_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(12, 10))
            plt.text(
                0.5,
                0.6,
                "Theme Analysis Report",
                fontsize=28,
                ha="center",
                va="center",
                weight="bold",
            )
            plt.text(
                0.5,
                0.5,
                f"Generated on {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
                fontsize=16,
                ha="center",
                va="center",
            )

            # Add a decorative header bar
            plt.axhline(y=0.75, xmin=0.1, xmax=0.9, color="#3366CC", linewidth=3)
            plt.axhline(y=0.35, xmin=0.1, xmax=0.9, color="#3366CC", linewidth=3)

            # Add framework names
            frameworks = self.frameworks.keys()
            framework_text = "Frameworks analyzed: " + ", ".join(frameworks)
            plt.text(
                0.5,
                0.3,
                framework_text,
                fontsize=14,
                ha="center",
                va="center",
                style="italic",
            )

            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Summary statistics page
            if not results_df.empty:
                # Create a summary page with charts
                fig = plt.figure(figsize=(12, 10))
                gs = gridspec.GridSpec(3, 2, height_ratios=[1, 2, 2])

                # Header
                ax_header = plt.subplot(gs[0, :])
                ax_header.text(
                    0.5,
                    0.5,
                    "Analysis Summary",
                    fontsize=20,
                    ha="center",
                    va="center",
                    weight="bold",
                )
                ax_header.axis("off")

                # Document count and metrics
                ax_metrics = plt.subplot(gs[1, 0])
                doc_count = len(highlighted_texts)
                theme_count = len(results_df)
                frameworks_count = len(results_df["Framework"].unique())

                metrics_text = (
                    f"Total Documents Analyzed: {doc_count}\n"
                    f"Total Theme Predictions: {theme_count}\n"
                    f"Unique Frameworks: {frameworks_count}\n"
                )

                if "Confidence" in results_df.columns:
                    confidence_counts = results_df["Confidence"].value_counts()
                    metrics_text += "\nConfidence Levels:\n"
                    for conf, count in confidence_counts.items():
                        metrics_text += f"  {conf}: {count} themes\n"

                ax_metrics.text(
                    0.1, 0.9, metrics_text, fontsize=12, va="top", linespacing=2
                )
                ax_metrics.axis("off")

                # Framework distribution chart
                ax_framework = plt.subplot(gs[1, 1])
                if not results_df.empty:
                    framework_counts = results_df["Framework"].value_counts()
                    bars = ax_framework.bar(
                        framework_counts.index,
                        framework_counts.values,
                        color=["#3366CC", "#DC3912", "#FF9900"],
                    )
                    ax_framework.set_title(
                        "Theme Distribution by Framework", fontsize=14
                    )
                    ax_framework.set_ylabel("Number of Themes")

                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax_framework.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + 0.1,
                            f"{height:d}",
                            ha="center",
                            fontsize=10,
                        )

                    ax_framework.spines["top"].set_visible(False)
                    ax_framework.spines["right"].set_visible(False)
                    plt.setp(ax_framework.get_xticklabels(), rotation=30, ha="right")

                # Themes by confidence chart
                ax_confidence = plt.subplot(gs[2, :])
                if "Confidence" in results_df.columns and "Theme" in results_df.columns:
                    # Prepare data for stacked bar chart
                    theme_conf_data = pd.crosstab(
                        results_df["Theme"], results_df["Confidence"]
                    )

                    # Select top themes by total count
                    top_themes = (
                        theme_conf_data.sum(axis=1)
                        .sort_values(ascending=False)
                        .head(10)
                        .index
                    )
                    theme_conf_data = theme_conf_data.loc[top_themes]

                    # Plot stacked bar chart
                    confidence_colors = {
                        "High": "#4CAF50",
                        "Medium": "#FFC107",
                        "Low": "#F44336",
                    }

                    # Get confidence levels present in the data
                    confidence_levels = list(theme_conf_data.columns)
                    colors = [
                        confidence_colors.get(level, "#999999")
                        for level in confidence_levels
                    ]

                    theme_conf_data.plot(
                        kind="barh",
                        stacked=True,
                        ax=ax_confidence,
                        color=colors,
                        figsize=(10, 6),
                    )

                    ax_confidence.set_title(
                        "Top Themes by Confidence Level", fontsize=14
                    )
                    ax_confidence.set_xlabel("Number of Documents")
                    ax_confidence.set_ylabel("Theme")

                    # Create custom legend
                    patches = [
                        Patch(
                            color=confidence_colors.get(level, "#999999"), label=level
                        )
                        for level in confidence_levels
                    ]
                    ax_confidence.legend(
                        handles=patches, title="Confidence", loc="upper right"
                    )

                    ax_confidence.spines["top"].set_visible(False)
                    ax_confidence.spines["right"].set_visible(False)
                else:
                    ax_confidence.axis("off")
                    ax_confidence.text(
                        0.5,
                        0.5,
                        "Confidence data not available",
                        fontsize=14,
                        ha="center",
                        va="center",
                    )

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # Framework-specific pages
            for framework_name in self.frameworks.keys():
                # Filter results for this framework
                framework_results = results_df[
                    results_df["Framework"] == framework_name
                ]

                if not framework_results.empty:
                    # Create a new page for the framework
                    fig = plt.figure(figsize=(12, 10))

                    # Title
                    plt.suptitle(
                        f"{framework_name} Framework Analysis",
                        fontsize=20,
                        y=0.95,
                        weight="bold", 
                    )

                    # Theme counts
                    theme_counts = framework_results["Theme"].value_counts().head(15)

                    if not theme_counts.empty:
                        plt.subplot(111)
                        bars = plt.barh(
                            theme_counts.index[::-1],
                            theme_counts.values[::-1],
                            color="#5975A4",
                            alpha=0.8,
                        )

                        # Add value labels
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            plt.text(
                                width + 0.3,
                                bar.get_y() + bar.get_height() / 2,
                                f"{width:d}",
                                va="center",
                                fontsize=10,
                            )

                        plt.xlabel("Number of Documents")
                        plt.ylabel("Theme")
                        plt.title(f"Top Themes in {framework_name}", pad=20)
                        plt.grid(axis="x", linestyle="--", alpha=0.7)
                        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

            # Sample highlighted documents (text descriptions only)
            if highlighted_texts:
                # Create document summaries page
                fig = plt.figure(figsize=(12, 10))
                plt.suptitle(
                    "Document Analysis Summaries", fontsize=20, y=0.95, weight="bold"
                )

                # We'll show summaries of a few documents
                max_docs_to_show = min(3, len(highlighted_texts))
                docs_to_show = list(highlighted_texts.keys())[:max_docs_to_show]

                # Get theme counts for each document
                doc_summaries = []
                for doc_id in docs_to_show:
                    doc_themes = results_df[results_df["Record ID"] == doc_id]
                    theme_count = len(doc_themes)
                    frameworks = doc_themes["Framework"].unique()
                    doc_summaries.append(
                        {
                            "doc_id": doc_id,
                            "theme_count": theme_count,
                            "frameworks": ", ".join(frameworks),
                        }
                    )

                # Format as a table-like display
                plt.axis("off")
                table_text = "Document Analysis Summaries:\n\n"
                for i, summary in enumerate(doc_summaries):
                    doc_id = summary["doc_id"]
                    table_text += f"Document {i+1} (ID: {doc_id}):\n"
                    table_text += f"  • Identified Themes: {summary['theme_count']}\n"
                    table_text += f"  • Frameworks: {summary['frameworks']}\n\n"

                plt.text(0.1, 0.8, table_text, fontsize=12, va="top", linespacing=1.5)

                # Also add a note about the full HTML version
                note_text = (
                    "Note: A detailed HTML report with highlighted text excerpts has been\n"
                    "created alongside this PDF. The HTML version provides interactive\n"
                    "highlighting of theme-related sentences in each document."
                )
                plt.text(
                    0.1,
                    0.3,
                    note_text,
                    fontsize=12,
                    va="top",
                    linespacing=1.5,
                    style="italic",
                    bbox=dict(
                        facecolor="#F0F0F0",
                        edgecolor="#CCCCCC",
                        boxstyle="round,pad=0.5",
                    ),
                )

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # Copy the temp file to the desired output filename

        shutil.copy2(temp_pdf_path, output_filename)

        # Clean up temp file
        try:
            os.unlink(temp_pdf_path)
        except:
            pass

        return output_filename

    def _preassign_framework_colors(self):
        """Preassign colors to each framework for consistent coloring"""
        # Create a dictionary to track colors used for each framework
        framework_colors = {}

        # Assign colors to each theme in each framework
        for framework, themes in self.frameworks.items():
            for i, theme in enumerate(themes):
                theme_key = f"{framework}_{theme['name']}"
                # Assign color from the theme_colors list, cycling if needed
                color_idx = i % len(self.theme_colors)
                self.theme_color_map[theme_key] = self.theme_colors[color_idx]

