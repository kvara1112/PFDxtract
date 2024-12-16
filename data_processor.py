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
