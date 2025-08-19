import pandas as pd
import streamlit as st
import logging
import zipfile
import io
import os
import re
import random
import string
from datetime import datetime
from .core_utils import (
    perform_advanced_keyword_search,
    export_to_excel,
)

def render_filter_data_tab():
    """Render a filtering tab within the Scraped File Preparation section with layout similar to Scrape Reports tab"""
    st.subheader("Filter Data")

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"],
        help="Upload your PFD reports dataset",
        key="filter_file_uploader",
    )
    
    # If no file is uploaded, show instructions
    if uploaded_file is None:
        st.info("Please upload a PFD reports dataset (CSV or Excel file) to begin filtering.")
        
        with st.expander("📋 File Requirements", expanded=False):
            st.markdown(
                """
            ## Recommended Columns
            
            For optimal filtering, your file should include these columns:
            
            - **Title**: Report title
            - **URL**: Link to the original report
            - **date_of_report**: Date in format DD/MM/YYYY
            - **ref**: Reference number
            - **deceased_name**: Name of deceased person (if applicable)
            - **coroner_name**: Name of the coroner
            - **coroner_area**: Coroner jurisdiction area
            - **categories**: Report categories
            - **Content**: Report text content
            - **Extracted_Concerns**: Extracted coroner concerns text
            
            Files created from the File Merger tab should contain all these columns.
            """
            )
        return
    
    # Process uploaded file
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Basic data cleaning
        data = data.dropna(how="all")  # Remove completely empty rows

        # Convert date_of_report to datetime if it exists
        if "date_of_report" in data.columns:
            try:
                # Try multiple date formats
                data["date_of_report"] = pd.to_datetime(
                    data["date_of_report"], format="%d/%m/%Y", errors="coerce"
                )
                # Fill in any parsing failures with other formats
                mask = data["date_of_report"].isna()
                if mask.any():
                    data.loc[mask, "date_of_report"] = pd.to_datetime(
                        data.loc[mask, "date_of_report"], format="%Y-%m-%d", errors="coerce"
                    )
                # Final attempt with pandas' smart parsing
                mask = data["date_of_report"].isna()
                if mask.any():
                    data.loc[mask, "date_of_report"] = pd.to_datetime(
                        data.loc[mask, "date_of_report"],
                        infer_datetime_format=True,
                        errors="coerce",
                    )
            except Exception as e:
                st.warning(
                    "Some date values could not be converted. Date filtering may not work completely."
                )
        
        # Create a form for filter settings
        with st.form("filter_settings_form"):
            st.subheader("Filter Settings")
            
            # Create rows with two columns each
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)
            row3_col1, row3_col2 = st.columns(2)
            
            # First row
            with row1_col1:
                # Date Range Filter
                if "date_of_report" in data.columns and pd.api.types.is_datetime64_any_dtype(data["date_of_report"]):
                    min_date = data["date_of_report"].min().date()
                    max_date = data["date_of_report"].max().date()
                    
                    st.markdown("**Date Range**")
                    start_date = st.date_input(
                        "From",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="filter_start_date",
                        format="DD/MM/YYYY",
                    )
            
            with row1_col2:
                # Date Range (continued)
                if "date_of_report" in data.columns and pd.api.types.is_datetime64_any_dtype(data["date_of_report"]):
                    st.markdown("&nbsp;")  # Add some spacing to align with "From"
                    end_date = st.date_input(
                        "To",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="filter_end_date",
                        format="DD/MM/YYYY",
                    )
                # Year Filter if date_of_report not available
                elif "year" in data.columns:
                    years = sorted(data["year"].dropna().unique())
                    if years:
                        min_year, max_year = min(years), max(years)
                        st.markdown("**Year Range**")
                        selected_years = st.slider(
                            "Select years",
                            min_year,
                            max_year,
                            (min_year, max_year),
                            key="filter_years",
                        )
            
            # Second row
            with row2_col1:
                # Coroner Area Filter
                if "coroner_area" in data.columns:
                    st.markdown("**Coroner Areas**")
                    coroner_areas = sorted(data["coroner_area"].dropna().unique())
                    selected_areas = st.multiselect(
                        "Select coroner areas",
                        options=coroner_areas,
                        key="filter_coroner_areas",
                        help="Select coroner areas to include"
                    )
            
            with row2_col2:
                # Coroner Name Filter
                if "coroner_name" in data.columns:
                    st.markdown("**Coroner Names**")
                    coroner_names = sorted(data["coroner_name"].dropna().unique())
                    selected_coroners = st.multiselect(
                        "Select coroner names",
                        options=coroner_names,
                        key="filter_coroner_names",
                        help="Select coroner names to include"
                    )
            
            # Third row
            with row3_col1:
                # Categories Filter
                if "categories" in data.columns:
                    st.markdown("**Categories**")
                    # Extract unique categories by splitting and cleaning
                    all_categories = set()
                    for cats in data["categories"].dropna():
                        # Split by comma and strip whitespace
                        if isinstance(cats, str):
                            split_cats = [cat.strip() for cat in cats.split(",")]
                            all_categories.update(split_cats)
                        elif isinstance(cats, list):
                            all_categories.update(
                                [cat.strip() for cat in cats if isinstance(cat, str)]
                            )

                    # Sort and remove any empty strings
                    sorted_categories = sorted(cat for cat in all_categories if cat)

                    # Create multiselect for categories
                    selected_categories = st.multiselect(
                        "Select categories",
                        options=sorted_categories,
                        key="filter_categories",
                        help="Select categories to include"
                    )
            
            with row3_col2:
                # Advanced content search
                if "content" in data.columns or "Content" in data.columns:
                    content_col = "content" if "content" in data.columns else "Content"
                    st.markdown("**Content Search**")
                    keyword_search = st.text_input(
                        "Search in content",
                        key="filter_keyword_search",
                        help="Use 'term1 and term2' for AND, 'term1 or term2' for OR",
                        placeholder="Enter search terms"
                    )
            
            # Option to exclude records without extracted concerns
            if "extracted_concerns" in data.columns or "Extracted_Concerns" in data.columns:
                concerns_col = (
                    "extracted_concerns"
                    if "extracted_concerns" in data.columns
                    else "Extracted_Concerns"
                )
                exclude_no_concerns = st.checkbox(
                    "Exclude records without extracted concerns",
                    value=False,
                    key="filter_exclude_no_concerns",
                    help="Show only records with extracted coroner concerns",
                )
            
            # Submission buttons
            submitted = st.form_submit_button("Apply Filters & Search", type="primary")
            
        # Reset Filters Button (outside the form)
        if st.button("Reset Filters"):
            for key in list(st.session_state.keys()):
                if key.startswith("filter_"):
                    del st.session_state[key]
            st.rerun()
        
        # Display results if form submitted or if this is a rerun with active filters
        show_results = submitted or any(k.startswith("filter_") for k in st.session_state.keys())
        
        if show_results:
            # Apply filters to data
            filtered_df = data.copy()
            active_filters = []
            
            # Date filter
            if "date_of_report" in filtered_df.columns and "filter_start_date" in st.session_state and "filter_end_date" in st.session_state:
                start_date = st.session_state.filter_start_date
                end_date = st.session_state.filter_end_date
                min_date = filtered_df["date_of_report"].min().date()
                max_date = filtered_df["date_of_report"].max().date()
                
                if start_date != min_date or end_date != max_date:
                    filtered_df = filtered_df[
                        (filtered_df["date_of_report"].dt.date >= start_date)
                        & (filtered_df["date_of_report"].dt.date <= end_date)
                    ]
                    active_filters.append(
                        f"Date: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}"
                    )

            # Year filter (if date_of_report not available)
            if "year" in filtered_df.columns and "filter_years" in st.session_state:
                selected_years = st.session_state.filter_years
                years = sorted(data["year"].dropna().unique())
                min_year, max_year = min(years), max(years)
                
                if selected_years != (min_year, max_year):
                    filtered_df = filtered_df[
                        (filtered_df["year"] >= selected_years[0])
                        & (filtered_df["year"] <= selected_years[1])
                    ]
                    active_filters.append(f"Year: {selected_years[0]} to {selected_years[1]}")

            # Coroner name filter
            if "coroner_name" in filtered_df.columns and "filter_coroner_names" in st.session_state and st.session_state.filter_coroner_names:
                selected_coroners = st.session_state.filter_coroner_names
                filtered_df = filtered_df[filtered_df["coroner_name"].isin(selected_coroners)]
                
                if len(selected_coroners) <= 3:
                    active_filters.append(f"Coroners: {', '.join(selected_coroners)}")
                else:
                    active_filters.append(f"Coroners: {len(selected_coroners)} selected")

            # Coroner area filter
            if "coroner_area" in filtered_df.columns and "filter_coroner_areas" in st.session_state and st.session_state.filter_coroner_areas:
                selected_areas = st.session_state.filter_coroner_areas
                filtered_df = filtered_df[filtered_df["coroner_area"].isin(selected_areas)]
                
                if len(selected_areas) <= 3:
                    active_filters.append(f"Areas: {', '.join(selected_areas)}")
                else:
                    active_filters.append(f"Areas: {len(selected_areas)} selected")

            # Categories filter
            if "categories" in filtered_df.columns and "filter_categories" in st.session_state and st.session_state.filter_categories:
                selected_categories = st.session_state.filter_categories
                
                # Handle both string and list categories
                if isinstance(filtered_df["categories"].iloc[0] if len(filtered_df) > 0 else "", list):
                    # List case
                    filtered_df = filtered_df[
                        filtered_df["categories"].apply(
                            lambda x: isinstance(x, list)
                            and any(cat in x for cat in selected_categories)
                        )
                    ]
                else:
                    # String case
                    filtered_df = filtered_df[
                        filtered_df["categories"]
                        .fillna("")
                        .astype(str)
                        .apply(lambda x: any(cat in x for cat in selected_categories))
                    ]
                
                if len(selected_categories) <= 3:
                    active_filters.append(f"Categories: {', '.join(selected_categories)}")
                else:
                    active_filters.append(f"Categories: {len(selected_categories)} selected")

            # Content keyword search with advanced AND/OR operators
            if "filter_keyword_search" in st.session_state and st.session_state.filter_keyword_search:
                keyword_search = st.session_state.filter_keyword_search
                content_col = "content" if "content" in filtered_df.columns else "Content"
                
                if content_col in filtered_df.columns:
                    before_count = len(filtered_df)
                    
                    #

                    # Apply the advanced search with AND/OR operators
                    filtered_df = filtered_df[
                        filtered_df[content_col]
                        .fillna("")
                        .astype(str)
                        .apply(lambda x: perform_advanced_keyword_search(x, keyword_search))
                    ]
                    
                    after_count = len(filtered_df)
                    
                    # Add to active filters
                    if " and " in keyword_search.lower():
                        search_desc = f"Content contains all terms: '{keyword_search}'"
                    elif " or " in keyword_search.lower():
                        search_desc = f"Content contains any term: '{keyword_search}'"
                    else:
                        search_desc = f"Content contains: '{keyword_search}'"
                    
                    active_filters.append(f"{search_desc} ({after_count}/{before_count} records)")

            # Apply filter for records with extracted concerns
            if "filter_exclude_no_concerns" in st.session_state and st.session_state.filter_exclude_no_concerns:
                concerns_col = (
                    "extracted_concerns"
                    if "extracted_concerns" in filtered_df.columns
                    else "Extracted_Concerns"
                )
                
                if concerns_col in filtered_df.columns:
                    before_count = len(filtered_df)
                    filtered_df = filtered_df[
                        filtered_df[concerns_col].notna()
                        & (filtered_df[concerns_col].astype(str).str.strip() != "")
                        & (filtered_df[concerns_col].astype(str).str.len() > 20)  # Ensure meaningful content
                    ]
                    after_count = len(filtered_df)
                    removed_count = before_count - after_count
                    active_filters.append(f"Excluding records without concerns (-{removed_count} records)")

            # Display active filters in a clean info box if there are any
            if active_filters:
                st.info(
                    "Active filters:\n" + "\n".join(f"• {filter_}" for filter_ in active_filters)
                )
            
            # Update session state with filtered data
            st.session_state.filtered_data = filtered_df
            
            # Display results
            st.subheader("Results")
            st.write(f"Showing {len(filtered_df)} of {len(data)} reports")
            
            if len(filtered_df) > 0:
                # Display the dataframe with formatted columns
                column_config = {}
                
                # Configure special columns
                if "date_of_report" in filtered_df.columns:
                    column_config["date_of_report"] = st.column_config.DateColumn(
                        "Date of Report", format="DD/MM/YYYY"
                    )
                
                if "URL" in filtered_df.columns:
                    column_config["URL"] = st.column_config.LinkColumn("Report Link")
                
                if "categories" in filtered_df.columns:
                    column_config["categories"] = st.column_config.ListColumn("Categories")
                
                # Display results in a Streamlit dataframe
                st.dataframe(
                    filtered_df,
                    column_config=column_config,
                    hide_index=True,
                    use_container_width=True
                )
                
                # Show export options
                show_export_options(filtered_df, "filtered")
            else:
                st.warning("No reports match your filter criteria. Try adjusting the filters.")
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logging.error(f"File processing error: {e}", exc_info=True)

def show_export_options(df: pd.DataFrame, prefix: str):
    """Show export options for the data with descriptive filename and unique keys"""
    try:
        st.subheader("Export Options")

        # Generate timestamp and random suffix to create unique keys
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        unique_id = f"{timestamp}_{random_suffix}"
        if prefix == "uploaded_other":
            filename = f"other_reports_{timestamp}"
        else:
            filename = f"pfd_reports_{prefix}_{timestamp}"

        col1, col2 = st.columns(2)

        # CSV Export
        with col1:
            try:
                # Create export copy with formatted dates
                df_csv = df.copy()
                if "date_of_report" in df_csv.columns:
                    df_csv["date_of_report"] = df_csv["date_of_report"].dt.strftime("%d/%m/%Y")

                csv = df_csv.to_csv(index=False).encode('utf-8')
                csv_key = f"download_csv_{prefix}_{unique_id}"
                st.download_button(
                    "📥 Download Reports (CSV)",
                    csv,
                    f"{filename}.csv",
                    "text/csv",
                    key=csv_key
                )
            except Exception as e:
                st.error(f"Error preparing CSV export: {str(e)}")

        # Excel Export
        with col2:
            try:
                excel_data = export_to_excel(df)
                excel_key = f"download_excel_{prefix}_{unique_id}"
                st.download_button(
                    "📥 Download Reports (Excel)",
                    excel_data,
                    f"{filename}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=excel_key
                )
            except Exception as e:
                st.error(f"Error preparing Excel export: {str(e)}")
        ##
        # PDF Export section
        # PDF Export section
        # PDF Export section
        if any(col.startswith("PDF_") and col.endswith("_Path") for col in df.columns):
            st.subheader("Download PDFs")
            try:
                # Create the ZIP file in memory
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "w") as zipf:
                    pdf_columns = [col for col in df.columns if col.startswith("PDF_") and col.endswith("_Path")]
                    added_files = set()
                    pdf_count = 0
                    folders_created = set()
                    
                    # Process each PDF file
                    for idx, row in df.iterrows():
                        # Build folder name using ref and deceased_name
                        folder_parts = []
                        
                        # Add reference number if available
                        if "ref" in row and pd.notna(row["ref"]):
                            folder_parts.append(str(row["ref"]))
                        
                        # Add deceased name if available
                        if "deceased_name" in row and pd.notna(row["deceased_name"]):
                            # Clean up deceased name for folder name
                            deceased_name = str(row["deceased_name"])
                            # Remove invalid characters for folder names
                            clean_name = re.sub(r'[<>:"/\\|?*]', '_', deceased_name)
                            # Limit folder name length
                            clean_name = clean_name[:50].strip()
                            folder_parts.append(clean_name)
                        
                        # Create folder name from parts
                        if folder_parts:
                            folder_name = "_".join(folder_parts)
                        else:
                            # Fallback if no ref or deceased name
                            record_id = str(row.get("Record ID", idx))
                            folder_name = f"report_{record_id}"
                        
                        # Add year to folder if available
                        if "year" in row and pd.notna(row["year"]):
                            folder_name = f"{folder_name}_{row['year']}"
                        
                        # Keep track of created folders
                        folders_created.add(folder_name)
                        
                        # Process each PDF for this row
                        for col in pdf_columns:
                            pdf_path = row.get(col)
                            if pd.isna(pdf_path) or not pdf_path or not os.path.exists(pdf_path) or pdf_path in added_files:
                                continue
                            
                            # Get the original filename without any modifications
                            original_filename = os.path.basename(pdf_path)
                            
                            # Create archive path with folder structure
                            zip_path = f"{folder_name}/{original_filename}"
                            
                            # Read the file content and write it to the ZIP
                            with open(pdf_path, 'rb') as file:
                                zipf.writestr(zip_path, file.read())
                            
                            added_files.add(pdf_path)
                            pdf_count += 1
                
                # Reset buffer position
                zip_buffer.seek(0)
                
                # PDF Download Button with Unique Key
                pdf_key = f"download_pdfs_{prefix}_{unique_id}"
                st.download_button(
                    f"📦 Download All PDFs ({pdf_count} files in {len(folders_created)} case folders)",
                    zip_buffer,
                    f"{filename}_pdfs.zip",
                    "application/zip",
                    key=pdf_key
                )
                    
            except Exception as e:
                st.error(f"Error preparing PDF download: {str(e)}")
                logging.error(f"PDF download error: {e}", exc_info=True)
        
        ##
       
    except Exception as e:
        st.error(f"Error setting up export options: {str(e)}")
        logging.error(f"Export options error: {e}", exc_info=True)

        
def show_export_options2(df: pd.DataFrame, prefix: str):
    """Show export options for the data with descriptive filename and unique keys"""
    try:
        st.subheader("Export Options")

        # Generate timestamp and random suffix for unique keys
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )
        unique_id = f"{timestamp}_{random_suffix}"
        filename = f"pfd_reports_{prefix}_{timestamp}"

        col1, col2 = st.columns(2)

        # CSV Export
        with col1:
            try:
                # Create export copy with formatted dates
                df_csv = df.copy()
                if "date_of_report" in df_csv.columns:
                    df_csv["date_of_report"] = df_csv["date_of_report"].dt.strftime(
                        "%d/%m/%Y"
                    )

                csv = df_csv.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Reports (CSV)",
                    csv,
                    f"{filename}.csv",
                    "text/csv",
                    key=f"download_csv_{prefix}_{unique_id}",
                )
            except Exception as e:
                st.error(f"Error preparing CSV export: {str(e)}")

        # Excel Export
        with col2:
            try:
                excel_data = export_to_excel(df)
                st.download_button(
                    "📥 Download Reports (Excel)",
                    excel_data,
                    f"{filename}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_excel_{prefix}_{unique_id}",
                )
            except Exception as e:
                st.error(f"Error preparing Excel export: {str(e)}")

        # PDF Download
        if any(col.startswith("PDF_") and col.endswith("_Path") for col in df.columns):
            st.subheader("Download PDFs")
            if st.button(f"Download all PDFs", key=f"pdf_button_{prefix}_{unique_id}"):
                with st.spinner("Preparing PDF download..."):
                    try:
                        pdf_zip_path = f"{filename}_pdfs.zip"

                        with zipfile.ZipFile(pdf_zip_path, "w") as zipf:
                            pdf_columns = [
                                col
                                for col in df.columns
                                if col.startswith("PDF_") and col.endswith("_Path")
                            ]
                            added_files = set()

                            for col in pdf_columns:
                                paths = df[col].dropna()
                                for pdf_path in paths:
                                    if (
                                        pdf_path
                                        and os.path.exists(pdf_path)
                                        and pdf_path not in added_files
                                    ):
                                        zipf.write(pdf_path, os.path.basename(pdf_path))
                                        added_files.add(pdf_path)

                        with open(pdf_zip_path, "rb") as f:
                            st.download_button(
                                "📦 Download All PDFs (ZIP)",
                                f.read(),
                                pdf_zip_path,
                                "application/zip",
                                key=f"download_pdfs_zip_{prefix}_{unique_id}",
                            )

                        # Cleanup zip file
                        os.remove(pdf_zip_path)
                    except Exception as e:
                        st.error(f"Error preparing PDF download: {str(e)}")

    except Exception as e:
        st.error(f"Error setting up export options: {str(e)}")
        logging.error(f"Export options error: {e}", exc_info=True)


