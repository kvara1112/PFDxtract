import pandas as pd
import re

def extract_structured_data(content):
    """
    Extract structured data from the content column
    """
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
    """
    Process the DataFrame to add new columns from content
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Apply extraction to each row
    extracted_data = processed_df['Content'].apply(extract_structured_data)
    
    # Expand the extracted data into new columns
    for column in ['Date_of_Report', 'Reference', 'Deceased_Name', 'Coroners_Name', 'Coroners_Area', 'Category']:
        processed_df[column] = extracted_data.apply(lambda x: x[column])
    
    # Optionally, truncate the content column to remove extracted information
    processed_df['Content'] = processed_df['Content'].apply(lambda x: re.sub(
        r'Date of report:.*?Category:.*?(?=This report is being sent to:|\Z)', 
        '', x, flags=re.DOTALL
    ).strip()
    
    return processed_df
