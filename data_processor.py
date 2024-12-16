import pandas as pd
import streamlit as st
from datetime import datetime
import io
import re

def extract_structured_data(content):
    """Extract structured data from the content column"""
    extracted_data = {
        'Date_of_Report': None,
        'Reference': None,
        'Deceased_Name': None,
        'Coroners_Name': None,
        'Coroners_Area': None,
        'Category': None
    }
    
    # Date of Report
    date_match = re.search(r'Date of report:\s*(\d{2}/\d{2}/\d{4})', content)
    if date_match:
        extracted_data['Date_of_Report'] = date_match.group(1)
    
    # Reference Number
    ref_match = re.search(r'Ref:\s*(\S+)', content)
    if ref_match:
        extracted_data['Reference'] = ref_match.group(1)
    
    # Deceased Name
    deceased_match = re.search(r'Deceased name:\s*([^\n]+)', content)
    if deceased_match:
        extracted_data['Deceased_Name'] = deceased_match.group(1).strip()
    
    # Coroner's Name
    coroner_name_match = re.search(r'Coroners name:\s*([^\n]+)', content)
    if coroner_name_match:
        extracted_data['Coroners_Name'] = coroner_name_match.group(1).strip()
    
    # Coroner's Area
    coroner_area_match = re.search(r'Coroners Area:\s*([^\n]+)', content)
    if coroner_area_match:
        extracted_data['Coroners_Area'] = coroner_area_match.group(1).strip()
    
    # Category
    category_match = re.search(r'Category:\s*([^\n]+)', content)
    if category_match:
        extracted_data['Category'] = category_match.group(1).strip()
    
    return extracted_data

def process_dataframe(df):
    """Process the DataFrame to add new columns from content"""
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    # Apply extraction to each row
    extracted_data = processed_df['Content'].apply(extract_structured_data)
    
    # Expand the extracted data into new columns
    for column in ['Date_of_Report', 'Reference', 'Deceased_Name', 'Coroners_Name', 'Coroners_Area', 'Category']:
        processed_df[column] = extracted_data.apply(lambda x: x[column])
    
    # Truncate the content column
    processed_df['Content'] = processed_df['Content'].apply(lambda x: re.sub(
        r'Date of report:.*?Category:.*?(?=This report is being sent to:|\Z)', 
        '', x, flags=re.DOTALL
    ).strip())
    
    return processed_df

def load_and_process_data(uploaded_file):
    """Load and process uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        df = process_dataframe(df)
        
        st.success(f"Loaded {len(df):,} reports")
        
        # Add filtering options
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_areas = st.multiselect(
                "Filter by Coroner's Area", 
                sorted(df['Coroners_Area'].dropna().unique())
            )
        
        with col2:
            selected_categories = st.multiselect(
                "Filter by Category", 
                sorted(df['Category'].dropna().unique())
            )
        
        with col3:
            df['Date_of_Report'] = pd.to_datetime(df['Date_of_Report'], format='%d/%m/%Y', errors='coerce')
            min_date = df['Date_of_Report'].min()
            max_date = df['Date_of_Report'].max()
            
            date_range = st.date_input(
                "Filter by Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Apply filters
        filtered_df = df.copy()
        if selected_areas:
            filtered_df = filtered_df[filtered_df['Coroners_Area'].isin(selected_areas)]
        
        if selected_categories:
            filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]
        
        if date_range:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['Date_of_Report'].dt.date >= start_date) & 
                (filtered_df['Date_of_Report'].dt.date <= end_date)
            ]
        
        # Show filtered data
        st.subheader(f"Reports Data (Showing {len(filtered_df)} of {len(df)} reports)")
        st.dataframe(
            filtered_df,
            column_config={
                "URL": st.column_config.LinkColumn("Report Link"),
                "Date_of_Report": st.column_config.DateColumn("Date of Report")
            },
            hide_index=True
        )
        
        # Export options
        export_format = st.selectbox("Export format:", ["CSV", "Excel"], key="export_format")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_reports_{timestamp}"
        
        if export_format == "CSV":
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Filtered Reports as CSV",
                csv,
                f"{filename}.csv",
                "text/csv",
                key="download_csv"
            )
        else:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False)
            excel_data = excel_buffer.getvalue()
            st.download_button(
                "📥 Download Filtered Reports as Excel",
                excel_data,
                f"{filename}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel"
            )
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
