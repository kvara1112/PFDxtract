import streamlit as st
import pandas as pd
import re
from datetime import datetime

def extract_metadata(content):
    """Extract metadata from report content using regex patterns"""
    metadata = {
        'date_of_report': None,
        'reference': None,
        'deceased_name': None,
        'coroner_name': None,
        'coroner_area': None,
        'categories': None,
        'sent_to': None
    }
    
    patterns = {
        'date_of_report': r'Date of report:\s*(\d{2}/\d{2}/\d{4})',
        'reference': r'Ref:\s*([\w-]+)',
        'deceased_name': r'Deceased name:\s*([^\n]+)',
        'coroner_name': r'Coroner name:\s*([^\n]+)',
        'coroner_area': r'Coroner Area:\s*([^\n]+)',
        'categories': r'Category:\s*([^\n]+)',
        'sent_to': r'This report is being sent to:\s*([^\n]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metadata[key] = match.group(1).strip()
            
    # Split categories into list if found
    if metadata['categories']:
        metadata['categories'] = [cat.strip() for cat in metadata['categories'].split('|')]
    
    return metadata

def process_data(df):
    """Process the dataframe to extract metadata from content"""
    # Create new dataframe for metadata
    metadata_rows = []
    
    for _, row in df.iterrows():
        # Extract metadata from main content
        metadata = extract_metadata(row['Content'])
        
        # Extract metadata from PDF contents if available
        pdf_columns = [col for col in df.columns if col.startswith('PDF_') and col.endswith('_Content')]
        for pdf_col in pdf_columns:
            if pd.notna(row[pdf_col]):
                pdf_metadata = extract_metadata(row[pdf_col])
                # Update metadata if new information found
                metadata.update({k: v for k, v in pdf_metadata.items() if v is not None})
        
        # Add original title and URL
        metadata['title'] = row['Title']
        metadata['url'] = row['URL']
        
        metadata_rows.append(metadata)
    
    processed_df = pd.DataFrame(metadata_rows)
    
    # Convert date string to datetime
    processed_df['date_of_report'] = pd.to_datetime(processed_df['date_of_report'], format='%d/%m/%Y', errors='coerce')
    
    return processed_df

def render_analysis_tab():
    st.header("Reports Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload previously exported reports (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Load the data
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
            
            # Process the data
            processed_df = process_data(raw_df)
            
            # Show data processing tabs
            data_tab1, data_tab2 = st.tabs(["Raw Data", "Processed Data"])
            
            with data_tab1:
                st.subheader("Raw Imported Data")
                st.dataframe(
                    raw_df,
                    column_config={
                        "URL": st.column_config.LinkColumn("Report Link")
                    },
                    hide_index=True
                )
            
            with data_tab2:
                st.subheader("Processed Metadata")
                st.dataframe(
                    processed_df,
                    column_config={
                        "url": st.column_config.LinkColumn("Report Link"),
                        "date_of_report": st.column_config.DateColumn("Date of Report"),
                        "categories": st.column_config.ListColumn("Categories"),
                    },
                    hide_index=True
                )
            
            # Display filters
            st.subheader("Filter Processed Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Date range filter
                min_date = processed_df['date_of_report'].min()
                max_date = processed_df['date_of_report'].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = st.date_input(
                        "Date range",
                        value=(min_date.date(), max_date.date()),
                        key="date_range"
                    )
            
            with col2:
                # Coroner area filter
                areas = sorted(processed_df['coroner_area'].dropna().unique())
                selected_area = st.multiselect("Coroner Area", areas)
            
            with col3:
                # Category filter
                all_categories = set()
                for cats in processed_df['categories'].dropna():
                    if isinstance(cats, list):
                        all_categories.update(cats)
                selected_categories = st.multiselect("Categories", sorted(all_categories))
            
            # Additional filters
            col1, col2 = st.columns(2)
            with col1:
                # Reference number filter
                ref_numbers = sorted(processed_df['reference'].dropna().unique())
                selected_ref = st.multiselect("Reference Numbers", ref_numbers)
            
            with col2:
                # Coroner name filter
                coroners = sorted(processed_df['coroner_name'].dropna().unique())
                selected_coroner = st.multiselect("Coroner Names", coroners)
            
            # Text search
            search_text = st.text_input("Search in deceased name or organizations:", "")
            
            # Apply filters
            filtered_df = processed_df.copy()
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df['date_of_report'].dt.date >= start_date) &
                    (filtered_df['date_of_report'].dt.date <= end_date)
                ]
            
            if selected_area:
                filtered_df = filtered_df[filtered_df['coroner_area'].isin(selected_area)]
            
            if selected_categories:
                filtered_df = filtered_df[
                    filtered_df['categories'].apply(
                        lambda x: any(cat in x for cat in selected_categories) if isinstance(x, list) else False
                    )
                ]
            
            if selected_ref:
                filtered_df = filtered_df[filtered_df['reference'].isin(selected_ref)]
                
            if selected_coroner:
                filtered_df = filtered_df[filtered_df['coroner_name'].isin(selected_coroner)]
            
            if search_text:
                search_mask = (
                    filtered_df['deceased_name'].str.contains(search_text, case=False, na=False) |
                    filtered_df['sent_to'].str.contains(search_text, case=False, na=False)
                )
                filtered_df = filtered_df[search_mask]
            
            # Display analysis
            st.subheader("Filtered Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Reports", len(filtered_df))
            with col2:
                st.metric("Unique Coroner Areas", filtered_df['coroner_area'].nunique())
            with col3:
                st.metric("Reports This Year", 
                         len(filtered_df[filtered_df['date_of_report'].dt.year == datetime.now().year]))
            with col4:
                if len(filtered_df) > 0:
                    date_range = (filtered_df['date_of_report'].max() - filtered_df['date_of_report'].min()).days
                    avg_reports_month = len(filtered_df) / (date_range / 30) if date_range > 0 else len(filtered_df)
                    st.metric("Avg Reports/Month", f"{avg_reports_month:.1f}")
            
            # Display filtered data
            st.dataframe(
                filtered_df,
                column_config={
                    "url": st.column_config.LinkColumn("Report Link"),
                    "date_of_report": st.column_config.DateColumn("Date of Report"),
                    "categories": st.column_config.ListColumn("Categories"),
                },
                hide_index=True
            )
            
            # Export filtered data
            if st.button("Export Filtered Data"):
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Filtered Data",
                    csv,
                    "filtered_reports.csv",
                    "text/csv",
                    key="download_filtered"
                )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a file to begin analysis")

if __name__ == "__main__":
    render_analysis_tab()
