import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List
import logging

def render_data_quality_tab(df: pd.DataFrame):
    """Render data quality analysis"""
    st.subheader("Data Quality Metrics")
    col1, col2, col3 = st.columns(3)
    
    completeness = {
        field: (df[field].notna().sum() / len(df) * 100)
        for field in ['date_of_report', 'reference', 'deceased_name', 'coroner_name', 
                     'coroner_area', 'categories']
    }
    
    with col1:
        st.metric("Date Extraction Rate", f"{completeness['date_of_report']:.1f}%")
        st.metric("Reference Extraction Rate", f"{completeness['reference']:.1f}%")
    
    with col2:
        st.metric("Name Extraction Rate", f"{completeness['deceased_name']:.1f}%")
        st.metric("Coroner Name Rate", f"{completeness['coroner_name']:.1f}%")
    
    with col3:
        st.metric("Coroner Area Rate", f"{completeness['coroner_area']:.1f}%")
        st.metric("Category Extraction Rate", f"{completeness['categories']:.1f}%")

def render_visualizations(df: pd.DataFrame):
    """Render standard visualizations"""
    st.subheader("Data Visualization")
    
    if 'date_of_report' in df.columns:
        try:
            # Convert date strings to datetime
            df['date_of_report'] = pd.to_datetime(df['date_of_report'], format='%d/%m/%Y')
            
            # Timeline visualization
            timeline_data = df.groupby(
                pd.Grouper(key='date_of_report', freq='M')
            ).size().reset_index()
            timeline_data.columns = ['Date', 'Count']
            
            fig = px.line(timeline_data, x='Date', y='Count',
                         title='Reports Over Time',
                         labels={'Count': 'Number of Reports'})
            st.plotly_chart(fig)
            
        except Exception as e:
            logging.error(f"Error creating timeline visualization: {e}")
    
    if 'categories' in df.columns:
        try:
            # Category distribution
            all_cats = []
            for cats in df['categories'].dropna():
                if isinstance(cats, list):
                    all_cats.extend(cats)
            
            if all_cats:
                cat_counts = pd.Series(all_cats).value_counts()
                fig = px.bar(x=cat_counts.index, y=cat_counts.values,
                            title='Distribution of Categories',
                            labels={'x': 'Category', 'y': 'Count'})
                st.plotly_chart(fig)
                
        except Exception as e:
            logging.error(f"Error creating category visualization: {e}")

def render_analysis_tab():
    """Main function to render the analysis tab"""
    st.title("Document Analysis")
    
    if 'scraped_data' in st.session_state and st.session_state.scraped_data is not None:
        df = st.session_state.scraped_data
        
        # Create tabs for different analyses
        data_tab, quality_tab, viz_tab = st.tabs([
            "Data Overview",
            "Data Quality",
            "Visualizations"
        ])
        
        with data_tab:
            st.dataframe(
                df,
                column_config={
                    "URL": st.column_config.LinkColumn("Report Link"),
                    "date_of_report": st.column_config.DateColumn("Date of Report"),
                    "categories": st.column_config.ListColumn("Categories")
                },
                hide_index=True
            )
        
        with quality_tab:
            render_data_quality_tab(df)
        
        with viz_tab:
            render_visualizations(df)
    
    else:
        st.info("Please scrape some reports first using the Scrape Reports tab.")
