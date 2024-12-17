import streamlit as st
import pandas as pd
from datetime import datetime
import re
from typing import Dict, List, Optional
import logging
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def clean_text_for_topic_modeling(text: str) -> str:
    """Enhanced text cleaning for topic modeling"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).strip()
    
    # Remove common headers and footers
    text = re.sub(r'PDF FILENAME:.*?\n', '', text)
    text = re.sub(r'REGULATION 28.*?FUTURE DEATHS?', '', text, flags=re.IGNORECASE|re.MULTILINE)
    
    # Clean special characters
    replacements = {
        'â€™': "'", 'â€˜': "'", 'â€œ': '"', 'â€': '"',
        'â€¦': '...', 'â€"': '-', 'â€¢': '•', 'Â': '',
        '\u200b': '', '\uf0b7': ''
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove email addresses and URLs
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def classify_document_type(filename: str, content: str) -> str:
    """Classify document as either PFD Report or Response"""
    filename = filename.lower() if filename else ""
    if 'response' in filename or 'reply' in filename or 'letter' in filename:
        return 'Response'
    elif any(x in filename for x in ['prevention-of-future-deaths', 'pfd']) and 'report' in filename:
        return 'PFD Report'
    
    # Fallback to content analysis if filename is inconclusive
    content = content.lower() if content else ""
    if 'regulation 28: report to prevent future deaths' in content:
        return 'PFD Report'
    elif 'response to regulation 28' in content or 'response to prevention of future death' in content:
        return 'Response'
    
    return 'Unknown'

class TopicModelAnalyzer:
    """Class for handling topic modeling analysis"""
    def __init__(self):
        self.model = None
        self.topics = None
        self.topic_info = None
    
    def train_model(self, texts: List[str], n_topics: int = "auto"):
        """Train BERTopic model"""
        # Initialize model with parameters suitable for Streamlit
        self.model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",  # Smaller, faster model
            min_topic_size=2,
            n_gram_range=(1, 2),
            nr_topics=n_topics,
            verbose=True,
            calculate_probabilities=True
        )
        
        # Fit model
        topics, probs = self.model.fit_transform(texts)
        self.topics = topics
        self.topic_info = self.model.get_topic_info()
        
        return self.topic_info
    
    def get_topic_visualization(self):
        """Get interactive topic visualization"""
        if self.model is None:
            return None
        
        return self.model.visualize_topics()
    
    def get_topic_distributions(self):
        """Get topic distribution visualization"""
        if self.model is None:
            return None
        
        return self.model.visualize_distribution()
   
def analyze_documents(df: pd.DataFrame):
    """Main analysis function to separate and analyze documents"""
    # Separate documents by type
    doc_types = {}
    for idx, row in df.iterrows():
        # Check main content
        if pd.notna(row.get('Content')):
            doc_type = classify_document_type(row.get('Title', ''), row['Content'])
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append({
                'content': clean_text_for_topic_modeling(row['Content']),
                'title': row.get('Title', ''),
                'url': row.get('URL', ''),
                'metadata': {
                    'date': row.get('date_of_report'),
                    'coroner_area': row.get('coroner_area'),
                    'categories': row.get('categories')
                }
            })
        
        # Check PDF contents
        pdf_cols = [col for col in df.columns if col.endswith('_Content')]
        for pdf_col in pdf_cols:
            if pd.notna(row.get(pdf_col)):
                name_col = pdf_col.replace('_Content', '_Name')
                doc_type = classify_document_type(
                    row.get(name_col, ''),
                    row[pdf_col]
                )
                if doc_type not in doc_types:
                    doc_types[doc_type] = []
                doc_types[doc_type].append({
                    'content': clean_text_for_topic_modeling(row[pdf_col]),
                    'title': row.get(name_col, ''),
                    'url': row.get('URL', ''),
                    'metadata': {
                        'date': row.get('date_of_report'),
                        'coroner_area': row.get('coroner_area'),
                        'categories': row.get('categories')
                    }
                })
    
    return doc_types

def render_data_quality_tab(df: pd.DataFrame):
    """Render data quality analysis"""
    st.subheader("Data Quality Metrics")
    col1, col2, col3 = st.columns(3)
    
    # Calculate completeness percentages
    completeness = {
        field: (df[field].notna().sum() / len(df) * 100)
        for field in ['date_of_report', 'reference', 'deceased_name', 'coroner_name', 
                     'coroner_area', 'categories', 'sent_to']
    }
    
    with col1:
        st.metric("Date Extraction Rate", f"{completeness['date_of_report']:.1f}%")
        st.metric("Reference Extraction Rate", f"{completeness['reference']:.1f}%")
        st.metric("Name Extraction Rate", f"{completeness['deceased_name']:.1f}%")
    
    with col2:
        st.metric("Coroner Name Rate", f"{completeness['coroner_name']:.1f}%")
        st.metric("Coroner Area Rate", f"{completeness['coroner_area']:.1f}%")
    
    with col3:
        st.metric("Category Extraction Rate", f"{completeness['categories']:.1f}%")
        st.metric("Sent To Extraction Rate", f"{completeness['sent_to']:.1f}%")

def render_filtering_options(df: pd.DataFrame):
    """Render filtering options"""
    st.subheader("Filter Data")
    col1, col2, col3 = st.columns(3)
    
    filters = {}
    
    with col1:
        # Date range filter
        min_date = df['date_of_report'].min()
        max_date = df['date_of_report'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            filters['date_range'] = st.date_input(
                "Date range",
                value=(min_date.date(), max_date.date()),
                key="date_range"
            )
    
    with col2:
        # Coroner area filter
        areas = sorted(df['coroner_area'].dropna().unique())
        filters['selected_area'] = st.multiselect("Coroner Area", areas)
    
    with col3:
        # Category filter
        all_categories = set()
        for cats in df['categories'].dropna():
            if isinstance(cats, list):
                all_categories.update(cats)
        filters['selected_categories'] = st.multiselect("Categories", sorted(all_categories))
    
    # Additional filters
    col1, col2 = st.columns(2)
    with col1:
        # Reference number filter
        ref_numbers = sorted(df['reference'].dropna().unique())
        filters['selected_ref'] = st.multiselect("Reference Numbers", ref_numbers)
    
    with col2:
        # Coroner name filter
        coroners = sorted(df['coroner_name'].dropna().unique())
        filters['selected_coroner'] = st.multiselect("Coroner Names", coroners)
    
    # Text search
    filters['search_text'] = st.text_input("Search in deceased name or organizations:", "")
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    if 'date_range' in filters and len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['date_of_report'].dt.date >= start_date) &
            (filtered_df['date_of_report'].dt.date <= end_date)
        ]
    
    if filters.get('selected_area'):
        filtered_df = filtered_df[filtered_df['coroner_area'].isin(filters['selected_area'])]
    
    if filters.get('selected_categories'):
        filtered_df = filtered_df[
            filtered_df['categories'].apply(
                lambda x: any(cat in x for cat in filters['selected_categories']) if isinstance(x, list) else False
            )
        ]
    
    if filters.get('selected_ref'):
        filtered_df = filtered_df[filtered_df['reference'].isin(filters['selected_ref'])]
    
    if filters.get('selected_coroner'):
        filtered_df = filtered_df[filtered_df['coroner_name'].isin(filters['selected_coroner'])]
    
    if filters.get('search_text'):
        search_mask = (
            filtered_df['deceased_name'].str.contains(filters['search_text'], case=False, na=False) |
            filtered_df['sent_to'].str.contains(filters['search_text'], case=False, na=False)
        )
        filtered_df = filtered_df[search_mask]
    
    return filtered_df

def render_visualizations(df: pd.DataFrame):
    """Render standard visualizations"""
    st.subheader("Data Visualization")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Timeline", "Categories", "Coroner Areas"])
    
    with viz_tab1:
        st.subheader("Reports Timeline")
        timeline_data = df.groupby(
            pd.Grouper(key='date_of_report', freq='M')
        ).size().reset_index()
        timeline_data.columns = ['Date', 'Count']
        fig = px.line(timeline_data, x='Date', y='Count', 
                     title='Reports Over Time',
                     labels={'Count': 'Number of Reports'})
        st.plotly_chart(fig)
    
    with viz_tab2:
        st.subheader("Category Distribution")
        all_cats = []
        for cats in df['categories'].dropna():
            if isinstance(cats, list):
                all_cats.extend(cats)
        cat_counts = pd.Series(all_cats).value_counts()
        fig = px.bar(x=cat_counts.index, y=cat_counts.values,
                    title='Distribution of Categories',
                    labels={'x': 'Category', 'y': 'Count'})
        st.plotly_chart(fig)
    
    with viz_tab3:
        st.subheader("Reports by Coroner Area")
        area_counts = df['coroner_area'].value_counts()
        fig = px.bar(x=area_counts.index, y=area_counts.values,
                    title='Reports by Coroner Area',
                    labels={'x': 'Coroner Area', 'y': 'Count'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

def render_topic_analysis(documents: List[Dict], title: str):
    """Render topic modeling analysis"""
    st.subheader(f"Topic Analysis: {title}")
    
    if not documents:
        st.warning(f"No {title} found in the data")
        return
    
    # Topic modeling controls
    col1, col2 = st.columns(2)
    with col1:
        n_topics = st.slider(
            f"Number of topics for {title}", 
            min_value=2, 
            max_value=20, 
            value=5,
            key=f"n_topics_{title}"
        )
    
    with col2:
        min_topic_size = st.slider(
            "Minimum topic size",
            min_value=2,
            max_value=20,
            value=5,
            key=f"min_topic_size_{title}"
        )
    
    # Extract texts
    texts = [doc['content'] for doc in documents]
    
    with st.spinner("Training topic model... This may take a few minutes."):
        try:
            # Initialize and train model
            analyzer = TopicModelAnalyzer()
            topic_info = analyzer.train_model(texts, n_topics)
            
            # Show topics
            st.write("### Discovered Topics")
            for idx, row in topic_info.iterrows():
                if row['Name'] != '-1':  # Skip outlier topic
                    st.write(f"**Topic {row['Name']}:** {', '.join(row['Keywords'])}")
            
            # Show visualizations
            viz_tab1, viz_tab2 = st.tabs(["Topic Map", "Topic Distribution"])
            
            with viz_tab1:
                st.write("### Topic Visualization")
                fig = analyzer.get_topic_visualization()
                if fig:
                    st.plotly_chart(fig)
            
            with viz_tab2:
                st.write("### Topic Distribution")
                fig = analyzer.get_topic_distributions()
                if fig:
                    st.plotly_chart(fig)
            
            # Document examples
            st.write("### Example Documents per Topic")
            topics = analyzer.topics
            for topic_id in set(topics):
                if topic_id != -1:  # Skip outlier topic
                    st.write(f"**Topic {topic_id}**")
                    topic_docs = [doc for doc, t in zip(documents, topics) if t == topic_id][:3]
                    for doc in topic_docs:
                        with st.expander(f"{doc['title'][:100]}..."):
                            st.write(doc['content'][:500] + "...")
                            st.write(f"[Link to full document]({doc['url']})")
        
        except Exception as e:
            st.error(f"Error in topic modeling: {str(e)}")
            logging.error(f"Topic modeling error: {str(e)}", exc_info=True)

def render_analysis_tab():
    """Main function to render the analysis tab"""
    st.title("Document Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload previously exported reports (CSV/Excel)", 
        type=['csv', 'xlsx']
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Create main tabs
            data_tab, quality_tab, viz_tab, topic_tab = st.tabs([
                "Data Overview",
                "Data Quality",
                "Visualizations",
                "Topic Analysis"
            ])
            
            with data_tab:
                # Show raw data
                st.subheader("Raw Data")
                st.dataframe(
                    df,
                    column_config={
                        "URL": st.column_config.LinkColumn("Report Link"),
                        "date_of_report": st.column_config.DateColumn("Date of Report"),
                        "categories": st.column_config.ListColumn("Categories")
                    },
                    hide_index=True
                )
                
                # Filtering options
                filters = render_filtering_options(df)
                filtered_df = apply_filters(df, filters)
                
                # Show filtered data
                st.subheader(f"Filtered Data (Showing {len(filtered_df)} of {len(df)} reports)")
                st.dataframe(
                    filtered_df,
                    column_config={
                        "URL": st.column_config.LinkColumn("Report Link"),
                        "date_of_report": st.column_config.DateColumn("Date of Report"),
                        "categories": st.column_config.ListColumn("Categories")
                    },
                    hide_index=True
                )
                
                # Export options
                if st.button("Export Filtered Data"):
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Filtered Data",
                        csv,
                        "filtered_reports.csv",
                        "text/csv",
                        key="download_filtered"
                    )
            
            with quality_tab:
                render_data_quality_tab(filtered_df)
            
            with viz_tab:
                render_visualizations(filtered_df)
            
            with topic_tab:
                # Analyze documents and separate by type
                doc_types = analyze_documents(filtered_df)
                
                # Create tabs for different document types
                pfd_tab, response_tab = st.tabs([
                    "Prevention of Future Deaths Reports",
                    "Responses to Reports"
                ])
                
                with pfd_tab:
                    render_topic_analysis(
                        doc_types.get('PFD Report', []),
                        "Prevention of Future Deaths Reports"
                    )
                
                with response_tab:
                    render_topic_analysis(
                        doc_types.get('Response', []),
                        "Responses to Reports"
                    )
            
        except Exception as e:
            st.error(f"Error analyzing documents: {str(e)}")
            logging.error(f"Analysis error: {str(e)}", exc_info=True)
    else:
        st.info("Please upload a file to begin analysis")

if __name__ == "__main__":
    render_analysis_tab()    
  
