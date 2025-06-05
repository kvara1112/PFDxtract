import logging
import pandas as pd
import numpy as np
import streamlit as st
import io
import time
import random
import string
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm

# Import our core utilities
from core_utils import (
    clean_text, 
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
            "Title", "URL", "Content", "date_of_report", "ref", 
            "deceased_name", "coroner_name", "coroner_area", "categories",
            "Report ID", "Deceased Name", "Death Type", "Year", "year",
            "Extracted_Concerns",
        ]
        
    def render_analyzer_ui(self):
        """Render the file merger UI."""
        st.subheader("Scraped File Merger")
        st.markdown("""
            This tool merges multiple scraped files into a single dataset. It prepares the data for steps (3) - (5).
            
            - Run this step even if you only have one scraped file. This step extracts the year and applies other processing as described in the bullets below. 
            - Combine data from multiple CSV or Excel files (the name of these files starts with pfd_reports_scraped_reportID_ )
            - Extract missing concerns from PDF content and fill empty Content fields
            - Extract year information from date fields
            - Remove duplicate records
            - Export full or reduced datasets with essential columns
            
            Use the options below to control how your files will be processed.
        """)

        # File upload section
        self._render_multiple_file_upload()

    def _render_multiple_file_upload(self):
        """Render interface for multiple file upload and merging."""
        # Initialize session state for processed data if not already present
        if "bert_merged_data" not in st.session_state:
            st.session_state.bert_merged_data = None

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
                                    st.success(f"Filled empty Content from PDF content for {newly_filled} records.")
                                else:
                                    st.info("No Content fields could be filled from PDF content.")

                            # Extract year from date_of_report if requested
                            if extract_year and "date_of_report" in self.data.columns:
                                self.data = self._add_year_column(self.data)
                                with_year = self.data["year"].notna().sum()
                                st.success(f"Added year data to {with_year} out of {len(self.data)} reports.")

                            # Extract missing concerns from PDF content if requested
                            if extract_from_pdf:
                                before_count = (self.data["Extracted_Concerns"].notna().sum())
                                self.data = self._extract_missing_concerns_from_pdf(self.data)
                                after_count = (self.data["Extracted_Concerns"].notna().sum())
                                newly_extracted = after_count - before_count

                                if newly_extracted > 0:
                                    st.success(f"Extracted missing concerns from PDF content for {newly_extracted} reports.")
                                else:
                                    st.info("No additional concerns could be extracted from PDF content.")

                            st.success(f"Files merged successfully! Final dataset has {len(self.data)} records.")

                            # Show a preview of the data
                            st.subheader("Preview of Merged Data")
                            st.dataframe(self.data.head(5))

                            # Save merged data to session state
                            st.session_state.bert_merged_data = self.data.copy()
                        else:
                            st.error("File merging resulted in empty data. Please check your files.")

                except Exception as e:
                    st.error(f"Error merging files: {str(e)}")
                    logging.error(f"File merging error: {e}", exc_info=True)

        # Show download options if we have processed data
        show_download_options = False
        if hasattr(self, "data") and self.data is not None and len(self.data) > 0:
            show_download_options = True
        elif st.session_state.bert_merged_data is not None:
            self.data = st.session_state.bert_merged_data
            show_download_options = True

        if show_download_options:
            self._provide_download_options()

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
                st.info(f"Processing file {file_index+1}: {file.name} ({len(df)} rows, {len(df.columns)} columns)")
    
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
                merged_df = merged_df.drop_duplicates(subset=valid_dup_cols, keep="first")
                after_count = len(merged_df)
    
                if before_count > after_count:
                    st.success(f"Removed {before_count - after_count} duplicate records based on {', '.join(valid_dup_cols)}")
            else:
                st.warning(f"Specified duplicate columns {duplicate_cols} not found in the merged data")
        
        # ALWAYS remove duplicate Record IDs, keeping only the first occurrence
        if "Record ID" in merged_df.columns:
            before_count = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=["Record ID"], keep="first")
            after_count = len(merged_df)

            if before_count > after_count:
                st.success(f"Removed {before_count - after_count} records with duplicate Record IDs (keeping first occurrence)")

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

        # Clean deceased names
        if "deceased_name" in merged_df.columns:
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

    def _provide_download_options(self):
        """Provide options to download the current data."""
        if self.data is None or len(self.data) == 0:
            return
        
        st.subheader("Download Merged Data")
        
        # Generate timestamp and random suffix for truly unique keys
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        unique_id = f"{timestamp}_{random_suffix}"
        
        # Generate filename prefix
        filename_prefix = f"merged_{timestamp}"
        
        # Full Dataset Section
        st.markdown("### Full Dataset")
        full_col1, full_col2 = st.columns(2)
        
        # CSV download button for full data
        with full_col1:
            try:
                # Create export copy with formatted dates
                df_csv = self.data.copy()
                if ("date_of_report" in df_csv.columns and pd.api.types.is_datetime64_any_dtype(df_csv["date_of_report"])):
                    df_csv["date_of_report"] = df_csv["date_of_report"].dt.strftime("%d/%m/%Y")
    
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

    def _clean_coroner_names(self, df):
        """Clean coroner names by removing common prefixes and standardizing format."""
        if "coroner_name" not in df.columns:
            return df
        
        processed_df = df.copy()
        
        def clean_name(name):
            if pd.isna(name) or not isinstance(name, str):
                return name
            
            # Remove common prefixes
            prefixes_to_remove = [
                "coroner ", "senior coroner ", "assistant coroner ", "area coroner ",
                "deputy coroner ", "acting coroner ", "his honour ", "her honour ",
                "judge ", "dr ", "professor ", "prof ", "mr ", "mrs ", "ms ", "miss "
            ]
            
            cleaned = name.strip().lower()
            for prefix in prefixes_to_remove:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            # Title case the result
            return cleaned.title() if cleaned else name
        
        processed_df["coroner_name"] = processed_df["coroner_name"].apply(clean_name)
        return processed_df

    def _clean_coroner_areas(self, df):
        """Clean coroner area names by standardizing format."""
        if "coroner_area" not in df.columns:
            return df
        
        processed_df = df.copy()
        
        def clean_area(area):
            if pd.isna(area) or not isinstance(area, str):
                return area
            
            # Remove common prefixes/suffixes
            area = area.strip()
            
            # Remove common patterns
            patterns_to_remove = [
                r"\bcoroner'?s?\s+area\b",
                r"\bcoroner'?s?\s+district\b", 
                r"\bcoroner'?s?\s+jurisdiction\b",
                r"\barea\s+of\b",
                r"\bdistrict\s+of\b"
            ]
            
            for pattern in patterns_to_remove:
                area = re.sub(pattern, "", area, flags=re.IGNORECASE).strip()
            
            # Clean up multiple spaces
            area = re.sub(r'\s+', ' ', area)
            
            return area.title() if area else area
        
        processed_df["coroner_area"] = processed_df["coroner_area"].apply(clean_area)
        return processed_df

    def _clean_deceased_name(self, df):
        """Clean deceased names by standardizing format."""
        if "deceased_name" not in df.columns:
            return df
        
        processed_df = df.copy()
        
        def clean_name(name):
            if pd.isna(name) or not isinstance(name, str):
                return name
            
            # Remove extra whitespace and normalize
            name = re.sub(r'\s+', ' ', name.strip())
            
            # Remove common prefixes that sometimes appear
            prefixes_to_remove = ["deceased: ", "name: ", "the late "]
            name_lower = name.lower()
            
            for prefix in prefixes_to_remove:
                if name_lower.startswith(prefix):
                    name = name[len(prefix):].strip()
                    break
            
            return name.title() if name else name
        
        processed_df["deceased_name"] = processed_df["deceased_name"].apply(clean_name)
        return processed_df

    def _clean_categories(self, df):
        """Clean categories by standardizing format and removing duplicates."""
        if "categories" not in df.columns:
            return df
        
        processed_df = df.copy()
        
        def clean_category_list(categories):
            if pd.isna(categories):
                return categories
            
            if isinstance(categories, str):
                # Split string categories
                cat_list = [cat.strip() for cat in categories.split(',')]
            elif isinstance(categories, list):
                cat_list = [str(cat).strip() for cat in categories if cat]
            else:
                return categories
            
            # Clean each category
            cleaned_cats = []
            for cat in cat_list:
                if cat and len(cat.strip()) > 0:
                    # Remove extra whitespace
                    cleaned_cat = re.sub(r'\s+', ' ', cat.strip())
                    # Standardize case
                    cleaned_cat = cleaned_cat.title()
                    
                    # Remove duplicates
                    if cleaned_cat not in cleaned_cats:
                        cleaned_cats.append(cleaned_cat)
            
            return cleaned_cats if cleaned_cats else categories
        
        processed_df["categories"] = processed_df["categories"].apply(clean_category_list)
        return processed_df

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
            st.warning("No 'date_of_report' column found in the data. Year extraction may be incomplete.")
            processed_df["year"] = None
            return processed_df

        # Create a new column for year extracted from date_of_report
        processed_df["year"] = processed_df["date_of_report"].apply(self._extract_report_year)

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
            logging.info(f"Added {final_count - extracted_count} more years from content analysis")

        return processed_df

    def _add_missing_years_from_content(self, concern_sections, df=None):
        """Try to extract years from content when date_of_report is missing."""
        import re
        import datetime
        from collections import Counter

        missing_year_count = 0

        # Try to use dataframe if available for additional data sources
        index_to_date = {}
        if df is not None:
            date_related_columns = []
            for col in df.columns:
                if any(term in col.lower() for term in ["date", "year", "time"]):
                    date_related_columns.append(col)

            if date_related_columns:
                logging.info(f"Checking {len(date_related_columns)} date-related columns for missing years")
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
            logging.info(f"Added year data to {missing_year_count} reports using content analysis")

        return concern_sections

    def _fill_empty_content_from_pdf(self, df):
        """
        Fill empty Content fields from PDF content columns.

        Args:
            df: DataFrame with merged data

        Returns:
            DataFrame with Content fields filled from PDF content
        """
        if df is None or len(df) == 0:
            return df

        # Make a copy to avoid modifying the original
        processed_df = df.copy()

        # Find PDF content columns
        pdf_columns = [
            col
            for col in processed_df.columns
            if col.startswith("PDF_") and col.endswith("_Content")
        ]

        if not pdf_columns:
            logging.info("No PDF content columns found")
            return processed_df

        # Identify rows with missing or empty Content
        empty_content_mask = (
            processed_df["Content"].isna() |
            (processed_df["Content"].astype(str).str.strip() == "") |
            (processed_df["Content"].astype(str).str.len() < 50)
        )

        empty_content_indices = processed_df[empty_content_mask].index
        filled_count = 0

        for idx in empty_content_indices:
            row = processed_df.loc[idx]
            
            # Try to find content from PDF columns
            for pdf_col in pdf_columns:
                pdf_content = row.get(pdf_col)
                if pd.notna(pdf_content) and len(str(pdf_content).strip()) > 50:
                    processed_df.at[idx, "Content"] = str(pdf_content)
                    filled_count += 1
                    break  # Use the first available PDF content

        if filled_count > 0:
            logging.info(f"Filled Content for {filled_count} records from PDF content")

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

    def _extract_report_year(self, date_val):
        """Optimized function to extract year from dd/mm/yyyy date format."""
        import re
        import datetime
        import pandas as pd

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
        }

        # Color mapping for themes
        self.theme_color_map = {}
        self.theme_colors = [
            "#FFD580", "#FFECB3", "#E1F5FE", "#E8F5E9", "#F3E5F5",
            "#FFF3E0", "#E0F7FA", "#F1F8E9", "#FFF8E1", "#E8EAF6",
            "#FCE4EC", "#F5F5DC", "#E6E6FA", "#FFFACD", "#D1E7DD",
            "#F8D7DA", "#D1ECF1", "#FFF3CD", "#D6D8D9", "#CFF4FC",
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
                sentence_positions = self._find_sentence_positions(text, theme["keywords"])

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

                if (keyword_matches and combined_score >= self.config["base_similarity_threshold"]):
                    theme_matches.append({
                        "theme": theme["name"],
                        "semantic_similarity": round(semantic_similarity, 3),
                        "combined_score": round(combined_score, 3),
                        "matched_keywords": ", ".join(keyword_matches),
                        "keyword_count": len(keyword_matches),
                        "sentence_positions": sentence_positions,  # Store sentence positions for highlighting
                    })

                    all_keyword_matches.extend(keyword_matches)

            # Sort by combined score
            theme_matches.sort(key=lambda x: x["combined_score"], reverse=True)

            # Limit number of themes
            top_theme_matches = theme_matches[: self.config["max_themes_per_framework"]]

            # Store theme matches and their highlighting info
            if top_theme_matches:
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

    def _get_isirch_framework(self):
        """I-SIRCh framework themes mapped exactly to the official framework structure"""
        return [
            {"name": "External - Policy factor", "keywords": ["policy factor", "policy", "factor"]},
            {"name": "External - Societal factor", "keywords": ["societal factor", "societal", "factor"]},
            {"name": "External - Economic factor", "keywords": ["economic factor", "economic", "factor"]},
            {"name": "External - COVID ✓", "keywords": ["covid ✓", "covid"]},
            {"name": "Organisation - Communication factor", "keywords": ["communication factor", "communication", "factor"]},
            {"name": "Organisation - Documentation", "keywords": ["documentation"]},
            {"name": "Organisation - Teamworking", "keywords": ["teamworking"]},
            {"name": "Jobs/Task - Assessment, investigation, testing, screening", "keywords": ["assessment", "investigation", "testing", "screening", "specimen", "sample", "laboratory"]},
            {"name": "Jobs/Task - Care planning", "keywords": ["care planning", "care", "planning"]},
            {"name": "Jobs/Task - Monitoring", "keywords": ["monitoring"]},
            {"name": "Person - Patient (characteristics and performance)", "keywords": ["patient", "characteristics and performance"]},
            {"name": "Person - Staff (characteristics and performance)", "keywords": ["staff", "characteristics and performance"]},
        ]

    def _get_house_of_commons_themes(self):
        """House of Commons themes mapped exactly to the official document"""
        return [
            {"name": "Communication", "keywords": ["communication", "dismissed", "listened", "concerns not taken seriously"]},
            {"name": "Fragmented care", "keywords": ["fragmented care", "fragmented", "care", "spread", "poorly", "communicating", "providers"]},
            {"name": "Guidance gaps", "keywords": ["guidance gaps", "guidance", "gaps", "information", "needs", "optimal"]},
            {"name": "Pre-existing conditions and comorbidities", "keywords": ["pre-existing conditions", "comorbidities", "overrepresented", "ethnic", "minority"]},
            {"name": "Inadequate maternity care", "keywords": ["inadequate maternity care", "inadequate", "maternity", "care", "individualized", "culturally", "sensitive"]},
            {"name": "Care quality and access issues", "keywords": ["microaggressions", "racism", "implicit", "impacts", "access", "treatment", "quality"]},
            {"name": "Socioeconomic factors and deprivation", "keywords": ["socioeconomic factors", "deprivation", "links to poor outcomes"]},
            {"name": "Biases and stereotyping", "keywords": ["biases", "stereotyping", "perpetuation", "stereotypes", "providers"]},
            {"name": "Consent/agency", "keywords": ["consent", "agency", "informed consent", "agency over care decisions"]},
            {"name": "Dignity/respect", "keywords": ["dignity", "respect", "neglectful", "lacking", "discrimination"]},
        ]

    def _get_extended_themes(self):
        """Extended Analysis themes with unique concepts not covered in other frameworks"""
        return [
            {"name": "Procedural and Process Failures", "keywords": ["procedure failure", "process breakdown", "protocol breach", "standard violation", "workflow issue"]},
            {"name": "Medication safety", "keywords": ["medication safety", "medication", "drug error", "prescription", "drug administration"]},
            {"name": "Resource allocation", "keywords": ["resource allocation", "resource", "allocation", "resource management", "staffing levels", "staff shortage"]},
            {"name": "Facility and Equipment Issues", "keywords": ["facility", "equipment", "maintenance", "infrastructure", "device failure", "equipment malfunction"]},
            {"name": "Emergency preparedness", "keywords": ["emergency preparedness", "emergency protocol", "emergency response", "crisis management"]},
            {"name": "Staff Wellbeing and Burnout", "keywords": ["burnout", "staff wellbeing", "resilience", "psychological safety", "stress management"]},
            {"name": "Ethical considerations", "keywords": ["ethical dilemma", "ethical decision", "moral distress", "ethical conflict", "value conflict"]},
            {"name": "Diagnostic process", "keywords": ["diagnostic error", "misdiagnosis", "delayed diagnosis", "diagnostic uncertainty", "diagnostic reasoning"]},
            {"name": "Post-Event Learning and Improvement", "keywords": ["incident learning", "corrective action", "improvement plan", "feedback loop", "lessons learned"]},
            {"name": "Electronic Health Record Issues", "keywords": ["electronic health record", "ehr issue", "alert fatigue", "interface design", "copy-paste error"]},
            {"name": "Time-Critical Interventions", "keywords": ["time-critical", "delayed intervention", "response time", "golden hour", "deterioration recognition"]},
            {"name": "Human Factors and Cognitive Aspects", "keywords": ["cognitive bias", "situational awareness", "attention management", "visual perception", "cognitive overload"]},
            {"name": "Service Design and Patient Flow", "keywords": ["service design", "patient flow", "care pathway", "bottleneck", "patient journey"]},
            {"name": "Maternal and Neonatal Risk Factors", "keywords": ["maternal risk", "pregnancy complication", "obstetric risk", "neonatal risk", "fetal risk"]},
        ]

    def _calculate_combined_score(self, semantic_similarity, keyword_count, text_length):
        """Calculate combined score that balances semantic similarity and keyword presence"""
        # Normalize keyword count by text length
        normalized_keyword_density = min(1.0, keyword_count / (text_length / 1000))

        # Weighted combination
        keyword_component = normalized_keyword_density * self.config["keyword_match_weight"]
        semantic_component = semantic_similarity * self.config["semantic_similarity_weight"]

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

        return sorted(positions)

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