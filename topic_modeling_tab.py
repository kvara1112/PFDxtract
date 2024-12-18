import streamlit as st
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import List, Dict, Optional, Tuple
import logging
from collections import Counter

def clean_text_for_modeling(text: str) -> str:
    """Clean text for topic modeling"""
    if not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        logging.error(f"Error cleaning text for modeling: {e}")
        return ""

def preprocess_text(df: pd.DataFrame, text_column: str = 'Content') -> List[str]:
    """Preprocess text data for topic modeling"""
    try:
        # Remove very short or empty documents
        documents = df[text_column].dropna().astype(str)
        documents = [clean_text_for_modeling(doc) for doc in documents]
        documents = [doc for doc in documents if len(doc.split()) > 10]
        return documents
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        return []

def create_topic_model(documents: List[str], num_topics: int) -> Optional[Tuple[BERTopic, List[int], np.ndarray]]:
    """Create and train topic model"""
    try:
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize BERTopic with parameters
        topic_model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=num_topics,
            low_memory=True,
            calculate_probabilities=True,
            verbose=True
        )
        
        # Fit the model and transform documents
        topics, probs = topic_model.fit_transform(documents)
        
        return topic_model, topics, probs
        
    except Exception as e:
        logging.error(f"Error creating topic model: {e}")
        return None

def plot_topic_distribution(topic_counts: pd.Series) -> None:
    """Plot distribution of documents across topics"""
    fig = px.bar(
        x=topic_counts.index,
        y=topic_counts.values,
        labels={'x': 'Topic Number', 'y': 'Number of Documents'},
        title='Distribution of Documents Across Topics'
    )
    
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Number of Documents",
        bargap=0.2
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_similarity_matrix(similarity_matrix: np.ndarray, num_topics: int) -> None:
    """Plot topic similarity heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=[f'Topic {i}' for i in range(num_topics)],
        y=[f'Topic {i}' for i in range(num_topics)],
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Topic Similarity Heatmap',
        xaxis_title='Topics',
        yaxis_title='Topics',
        width=700,
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_topic_modeling_tab() -> None:
    """Render the topic modeling tab"""
    st.header("Topic Modeling Analysis")
    
    # Check if data is available
    if 'scraped_data' not in st.session_state or st.session_state.scraped_data is None:
        st.warning("Please scrape reports first in the 'Scrape Reports' tab")
        return
    
    # Get the scraped data
    df = st.session_state.scraped_data
    
    # Sidebar configuration
    st.sidebar.header("Topic Modeling Options")
    
    num_topics = st.sidebar.slider(
        "Number of Topics",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        help="Select the number of topics to extract from the documents"
    )
    
    # Topic modeling type
    topic_type = st.sidebar.selectbox(
        "Report Type for Topic Modeling",
        ["All Reports", "Prevention of Future Death", "Response to PFD"]
    )
    
    # Filter data based on type
    if topic_type == "Prevention of Future Death":
        filtered_df = df[df['Content'].str.contains(
            'Prevention of Future Death',
            case=False,
            na=False
        )]
    elif topic_type == "Response to PFD":
        filtered_df = df[df['Content'].str.contains(
            'Response to Prevention of Future Death',
            case=False,
            na=False
        )]
    else:
        filtered_df = df
    
    # Check if we have enough documents
    if len(filtered_df) < num_topics:
        st.error(f"Not enough documents ({len(filtered_df)}) for {num_topics} topics. Please reduce topic count or scrape more reports.")
        return
    
    # Run topic modeling button
    if st.button("Run Topic Modeling"):
        with st.spinner("Performing Topic Modeling..."):
            try:
                # Preprocess text
                documents = preprocess_text(filtered_df)
                
                if not documents:
                    st.error("No valid documents found after preprocessing")
                    return
                
                # Create and train model
                result = create_topic_model(documents, num_topics)
                
                if not result:
                    st.error("Failed to create topic model")
                    return
                
                topic_model, topics, probs = result
                
                # Display results
                st.header("Topic Modeling Results")
                
                # Topic distribution
                topic_counts = pd.Series(topics).value_counts()
                plot_topic_distribution(topic_counts)
                
                # Topic keywords
                st.subheader("Top Keywords per Topic")
                topic_info = topic_model.get_topic_info()
                
                keywords_df = pd.DataFrame([
                    {
                        'Topic': row['Topic'],
                        'Size': row['Count'],
                        'Keywords': ', '.join([word for word, _ in topic_model.get_topic(row['Topic'])[:10]])
                    }
                    for _, row in topic_info.iterrows() if row['Topic'] != -1
                ])
                
                st.dataframe(
                    keywords_df,
                    column_config={
                        "Topic": st.column_config.NumberColumn("Topic"),
                        "Size": st.column_config.NumberColumn("Documents"),
                        "Keywords": st.column_config.TextColumn("Top Keywords")
                    }
                )
                
                # Topic similarity visualization
                st.subheader("Topic Similarity")
                similarity_matrix = topic_model.get_topic_similarity_matrix()
                plot_similarity_matrix(similarity_matrix, num_topics)
                
                # Example documents
                st.subheader("Example Documents per Topic")
                for topic_id in range(num_topics):
                    with st.expander(f"Topic {topic_id}"):
                        # Get documents for this topic
                        topic_docs = [doc for doc, topic in zip(documents, topics) if topic == topic_id]
                        
                        if topic_docs:
                            st.markdown(f"**Keywords**: {keywords_df.loc[keywords_df['Topic'] == topic_id, 'Keywords'].iloc[0]}")
                            st.markdown(f"**Number of documents**: {len(topic_docs)}")
                            st.markdown("### Example Documents:")
                            
                            for i, doc in enumerate(topic_docs[:3], 1):
                                st.markdown(f"**Document {i}**:")
                                st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                        else:
                            st.info("No documents found for this topic")
                
                # Topic evolution over time if dates are available
                if 'date_of_report' in filtered_df.columns:
                    st.subheader("Topic Evolution Over Time")
                    
                    # Create time-based visualization
                    topic_evolution = pd.DataFrame({
                        'Date': filtered_df['date_of_report'],
                        'Topic': topics
                    })
                    
                    topic_evolution = topic_evolution.groupby([
                        pd.Grouper(key='Date', freq='M'),
                        'Topic'
                    ]).size().reset_index(name='Count')
                    
                    fig = px.line(
                        topic_evolution,
                        x='Date',
                        y='Count',
                        color='Topic',
                        title='Topic Evolution Over Time'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in topic modeling: {str(e)}")
                logging.error(f"Topic modeling error: {e}", exc_info=True)

if __name__ == "__main__":
    st.warning("This is a module and should not be run directly. Please run app.py instead.")
