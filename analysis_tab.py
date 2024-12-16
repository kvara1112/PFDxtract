import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional
import nltk
from nltk.tokenize import sent_tokenize
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class MetadataExtractor:
    """Advanced metadata extraction class with robust parsing capabilities"""
    
    METADATA_PATTERNS = {
        'date_of_report': [
            r'Date of report:\s*(\d{2}/\d{2}/\d{4})',
            r'Date of report\s*(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})',
            r'Report date:\s*(\d{2}/\d{2}/\d{4})',
        ],
        'reference': [
            r'Ref:\s*([\w-]+)',
            r'Reference:\s*([\w-]+)',
            r'Reference Number:\s*([\w-]+)',
        ],
        'deceased_name': [
            r'Deceased name:\s*([^\n]+)',
            r'Name of deceased:\s*([^\n]+)',
            r'Name of the deceased:\s*([^\n]+)',
        ],
        'coroner_name': [
            r'Coroner name:\s*([^\n]+)',
            r'Coroner:\s*([^\n]+)',
            r'Name of coroner:\s*([^\n]+)',
        ],
        'coroner_area': [
            r'Coroner Area:\s*([^\n]+)',
            r'Coroner\'s Area:\s*([^\n]+)',
            r'Area:\s*([^\n]+)',
        ],
        'categories': [
            r'Category:\s*([^\n]+)',
            r'Categories:\s*([^\n]+)',
            r'Type:\s*([^\n]+)',
        ],
        'sent_to': [
            r'This report is being sent to:\s*([^\n]+)',
            r'Report sent to:\s*([^\n]+)',
            r'Sent to:\s*([^\n]+)',
        ]
    }
    
    DATE_FORMATS = [
        '%d/%m/%Y',
        '%d %B %Y',
        '%d %b %Y',
        '%B %d %Y',
        '%b %d %Y',
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better metadata extraction"""
        if not text:
            return ""
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Ensure proper spacing after colons
        text = re.sub(r':\s*', ': ', text)
        
        # Remove unnecessary Unicode characters
        text = re.sub(r'[\u200b\ufeff]', '', text)
        
        return text.strip()
    
    def _extract_with_patterns(self, text: str, patterns: List[str]) -> Optional[str]:
        """Try multiple patterns to extract metadata"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string using multiple formats"""
        if not date_str:
            return None
            
        # Remove ordinal indicators
        date_str = re.sub(r'(\d)(st|nd|rd|th)', r'\1', date_str)
        
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    
    def _clean_categories(self, categories: str) -> List[str]:
        """Clean and split categories"""
        if not categories:
            return []
            
        # Split on multiple possible delimiters
        cats = re.split(r'\s*[|;,]\s*', categories)
        
        # Clean individual categories
        cats = [cat.strip() for cat in cats if cat.strip()]
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in cats if not (x in seen or seen.add(x))]
    
    def _extract_from_section(self, text: str, start_marker: str, end_markers: List[str]) -> Optional[str]:
        """Extract content between markers"""
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return None
            
        start_idx += len(start_marker)
        min_end_idx = len(text)
        
        for end_marker in end_markers:
            end_idx = text.find(end_marker, start_idx)
            if end_idx != -1 and end_idx < min_end_idx:
                min_end_idx = end_idx
        
        return text[start_idx:min_end_idx].strip()
    
    def extract_metadata(self, content: str) -> Dict:
        """Extract metadata from report content"""
        metadata = {
            'date_of_report': None,
            'reference': None,
            'deceased_name': None,
            'coroner_name': None,
            'coroner_area': None,
            'categories': None,
            'sent_to': None
        }
        
        if not content:
            return metadata
        
        # Preprocess content
        processed_text = self._preprocess_text(content)
        
        # Split into sentences for better context
        sentences = sent_tokenize(processed_text)
        
        # Try pattern-based extraction first
        for field, patterns in self.METADATA_PATTERNS.items():
            value = self._extract_with_patterns(processed_text, patterns)
            if value:
                if field == 'categories':
                    metadata[field] = self._clean_categories(value)
                elif field == 'date_of_report':
                    parsed_date = self._parse_date(value)
                    metadata[field] = parsed_date.strftime('%d/%m/%Y') if parsed_date else value
                else:
                    metadata[field] = value
        
        # Try contextual extraction for missing fields
        if not metadata['sent_to']:
            sent_to = self._extract_from_section(
                processed_text,
                "This report is being sent to",
                ["CIRCUMSTANCES OF THE DEATH", "CORONER'S CONCERNS", "\n\n"]
            )
            if sent_to:
                metadata['sent_to'] = sent_to
        
        # Post-process specific fields
        if metadata['categories'] and isinstance(metadata['categories'], list):
            # Remove category prefixes if present
            metadata['categories'] = [
                re.sub(r'^(Category:\s*|Type:\s*)', '', cat) 
                for cat in metadata['categories']
            ]
        
        return metadata

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the dataframe to extract metadata from content"""
    extractor = MetadataExtractor()
    metadata_rows = []
    
    for _, row in df.iterrows():
        # Extract metadata from main content
        metadata = extractor.extract_metadata(row['Content'])
        
        # Extract metadata from PDF contents if available
        pdf_columns = [col for col in df.columns if col.startswith('PDF_') and col.endswith('_Content')]
        for pdf_col in pdf_columns:
            if pd.notna(row[pdf_col]):
                pdf_metadata = extractor.extract_metadata(row[pdf_col])
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
            data_tab1, data_tab2, data_tab3 = st.tabs(["Raw Data", "Processed Data", "Data Quality"])
            
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
            
            with data_tab3:
                st.subheader("Data Quality Metrics")
                col1, col2, col3 = st.columns(3)
                
                # Calculate completeness percentages
                completeness = {
                    field: (processed_df[field].notna().sum() / len(processed_df) * 100)
                    for field in processed_df.columns
                }
                
                with col1:
                    st.metric("Date Extraction Rate", f"{completeness['date_of_report']:.1f}%")
                with col2:
                    st.metric("Category Extraction Rate", f"{completeness['categories']:.1f}%")
                with col3:
                    st.metric("Overall Completeness", 
                             f"{sum(completeness.values()) / len(completeness):.1f}%")
            
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
            st.subheader("Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Reports", len(filtered_df))
            with col2:
                st.metric("Unique Coroner Areas", filtered_df['coroner_area'].nunique())
            with col3:
                current_year = datetime.now().year
                st.metric("Reports This Year", 
                         len(filtered_df[filtered_df['date_of_report'].dt.year == current_year]))
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
            
            # Add visualization section
            st.subheader("Data Visualization")
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Timeline", "Categories", "Coroner Areas"])
            
            with viz_tab1:
                st.subheader("Reports Timeline")
                # Group by month and count
                timeline_data = filtered_df.groupby(
                    pd.Grouper(key='date_of_report', freq='M')
                ).size().reset_index()
                timeline_data.columns = ['Date', 'Count']
                
                # Create line chart
                st.line_chart(timeline_data.set_index('Date'))
            
            with viz_tab2:
                st.subheader("Category Distribution")
                # Flatten categories and count
                all_cats = []
                for cats in filtered_df['categories'].dropna():
                    if isinstance(cats, list):
                        all_cats.extend(cats)
                
                cat_counts = pd.Series(all_cats).value_counts()
                st.bar_chart(cat_counts)
            
            with viz_tab3:
                st.subheader("Reports by Coroner Area")
                area_counts = filtered_df['coroner_area'].value_counts()
                st.bar_chart(area_counts)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logging.error(f"Analysis error: {str(e)}", exc_info=True)
    else:
        st.info("Please upload a file to begin analysis")

if __name__ == "__main__":
    render_analysis_tab()
