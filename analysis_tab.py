import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Optional
import logging

def plot_timeline(df: pd.DataFrame) -> None:
    """Plot timeline of reports"""
    timeline_data = df.groupby(
        pd.Grouper(key='date_of_report', freq='M')
    ).size().reset_index()
    timeline_data.columns = ['Date', 'Count']
    
    fig = px.line(timeline_data, x='Date', y='Count',
                  title='Reports Timeline',
                  labels={'Count': 'Number of Reports'})
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Reports",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_category_distribution(df: pd.DataFrame) -> None:
    """Plot category distribution"""
    all_cats = []
    for cats in df['categories'].dropna():
        if isinstance(cats, list):
            all_cats.extend(cats)
    
    cat_counts = pd.Series(all_cats).value_counts()
    
    fig = px.bar(
        x=cat_counts.index,
        y=cat_counts.values,
        title='Category Distribution',
        labels={'x': 'Category', 'y': 'Count'}
    )
    
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Number of Reports",
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_coroner_areas(df: pd.DataFrame) -> None:
    """Plot coroner areas distribution"""
    area_counts = df['coroner_area'].value_counts().head(20)
    
    fig = px.bar(
        x=area_counts.index,
        y=area_counts.values,
        title='Top 20 Coroner Areas',
        labels={'x': 'Area', 'y': 'Count'}
    )
    
    fig.update_layout(
        xaxis_title="Coroner Area",
        yaxis_title="Number of Reports",
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def generate_trends_analysis(df: pd.DataFrame) -> None:
    """Generate trends analysis"""
    st.subheader("Trends Analysis")
    
    if not df['date_of_report'].empty:
        # Calculate recent reports
        recent_mask = df['date_of_report'] >= (datetime.now() - timedelta(days=365))
        recent_reports = df[recent_mask]
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Reports", len(df))
        
        with col2:
            st.metric("Reports Last Year", len(recent_reports))
        
        with col3:
            avg_monthly = len(recent_reports) / 12 if len(recent_reports) > 0 else 0
            st.metric("Average Monthly", f"{avg_monthly:.1f}")
        
        # Plot timeline
        plot_timeline(df)
        
        # Monthly trends
        monthly_counts = df.groupby(
            [df['date_of_report'].dt.year, df['date_of_report'].dt.month]
        ).size().reset_index()
        monthly_counts.columns = ['Year', 'Month', 'Count']
        
        st.subheader("Monthly Report Trends")
        fig = px.bar(monthly_counts, x='Month', y='Count',
                    color='Year', barmode='group',
                    title='Reports by Month and Year')
        st.plotly_chart(fig, use_container_width=True)

def render_analysis_tab() -> None:
    """Render the analysis tab"""
    st.header("Reports Analysis")
    
    # Check if we have data
    if 'scraped_data' not in st.session_state or st.session_state.scraped_data is None:
        st.warning("No data available. Please scrape some reports first.")
        return
    
    df = st.session_state.scraped_data
    
    # Filters sidebar
    st.sidebar.header("Analysis Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=[df['date_of_report'].min(), df['date_of_report'].max()],
        key="analysis_date_range"
    )
    
    # Category filter
    all_categories = set()
    for cats in df['categories'].dropna():
        if isinstance(cats, list):
            all_categories.update(cats)
            
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=sorted(all_categories)
    )
    
    # Coroner area filter
    coroner_areas = sorted(df['coroner_area'].dropna().unique())
    selected_areas = st.sidebar.multiselect(
        "Coroner Areas",
        options=coroner_areas
    )
    
    # Apply filters
    mask = pd.Series(True, index=df.index)
    
    if len(date_range) == 2:
        mask &= (df['date_of_report'].dt.date >= date_range[0]) & \
                (df['date_of_report'].dt.date <= date_range[1])
    
    if selected_categories:
        mask &= df['categories'].apply(
            lambda x: any(cat in x for cat in selected_categories) if isinstance(x, list) else False
        )
    
    if selected_areas:
        mask &= df['coroner_area'].isin(selected_areas)
    
    filtered_df = df[mask]
    
    # Display analysis
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters.")
        return
    
    # Overview metrics
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reports", len(filtered_df))
    with col2:
        st.metric("Unique Coroner Areas", filtered_df['coroner_area'].nunique())
    with col3:
        st.metric("Categories", len(all_categories))
    with col4:
        date_range = (filtered_df['date_of_report'].max() - filtered_df['date_of_report'].min()).days
        avg_reports_month = len(filtered_df) / (date_range / 30) if date_range > 0 else len(filtered_df)
        st.metric("Avg Reports/Month", f"{avg_reports_month:.1f}")
    
    # Visualizations
    st.subheader("Visualizations")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Timeline", "Categories", "Coroner Areas"])
    
    with viz_tab1:
        plot_timeline(filtered_df)
    
    with viz_tab2:
        plot_category_distribution(filtered_df)
    
    with viz_tab3:
        plot_coroner_areas(filtered_df)
    
    # Trends Analysis
    generate_trends_analysis(filtered_df)
    
    # Raw Data View
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(
            filtered_df,
            column_config={
                "URL": st.column_config.LinkColumn("Report Link"),
                "date_of_report": st.column_config.DateColumn("Date of Report"),
                "categories": st.column_config.ListColumn("Categories")
            },
            hide_index=True
        )

if __name__ == "__main__":
    st.warning("This is a module and should not be run directly. Please run app.py instead.")
