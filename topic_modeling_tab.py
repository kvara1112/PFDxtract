import streamlit as st
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import List, Dict, Optional
import logging

def clean_text_for_modeling(text: str) -> str:
    """Clean text for topic modeling"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_text(df: pd.DataFrame, text_column: str = 'Content') -> List[str]:
    """Preprocess text data for topic modeling"""
    documents = df[text_column].dropna().astype(str)
    documents = [clean_text_for_modeling(doc) for doc in documents]
    documents = [doc for doc in documents if len(doc.split()) > 10]
    return documents

def create_topic_model(documents: List[str], num_topics: int) -> Optional[BERTopic]:
    """Create and train topic model"""
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=num_topics,
            low_memory=True,
            calculate_probabilities=True
        )
        
        # Fit the model
        topics, probs = topic_model.fit_transform(documents)
        return topic_model
        
    except Exception as e:
        logging.error(f"Error creating topic model: {e}")
        return None

def render_topic_modeling_tab() -> None:
    st.header("Topic Modeling Analysis")
    
    if 'scraped_data' not in st.session_state or st.session_state.scraped_data is None:
        st.warning("Please scrape reports first in the 'Scrape Reports' tab")
        return
    
    # Get the scraped data
    df = st.session_state.scraped_data
    
    # Sidebar for configuration
    st.sidebar.header("Topic Modeling Options")
    
    num_topics = st.sidebar.slider(
        "Number of Topics",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        help="Select the number of topics to extract"
    )
    
    # Topic modeling type
    topic_type = st.sidebar.selectbox(
        "Report Type for Topic Modeling",
        ["All Reports", "Prevention of Future Death", "Response to PFD"]
    )
    
    # Filter data based on type
    if topic_type == "Prevention of Future Death":
        filtered_df = df[df['Content'].str.contains('Prevention of Future Death', case=False, na=False)]
    elif topic_type == "Response to PFD":
        filtered_df = df[df['Content'].str.contains('Response to Prevention of Future Death', case=False, na=False)]
    else:
        filtered_df = df
    
    if len(filtered_df) < num_topics:
        st.error(f"Not enough documents for {num_topics} topics. Please reduce topic count or scrape more reports.")
        return
    
    if st.button("Run Topic Modeling"):
        with st.spinner("Performing Topic Modeling..."):
            try:
                # Preprocess text
                documents = preprocess_text(filtered_df)
                
                # Create and train model
                topic_model = create_topic_model(documents, num_topics)
                
                if topic_model is None:
                    st.error("Failed to create topic model")
                    return
                
                # Display results
                st.subheader("Topic Modeling Results")
                
                # Topic distribution
                topics, _ = topic_model.transform(documents)
                topic_counts = pd.Series(topics).value_counts()
                
                fig_dist = px.bar(
                    x=topic_counts.index,
                    y=topic_counts.values,
                    labels={'x': 'Topic Number', 'y': 'Number of Documents'},
                    title='Distribution of Documents Across Topics'
                )
                st.plotly_chart(fig_dist)
                
                # Topic keywords
                st.subheader("Top Keywords per Topic")
                topic_info = topic_model.get_topic_info()
                
                keywords_df = pd.DataFrame([
                    {
                        'Topic': row['Topic'],
                        'Keywords': ', '.join([word for word, _ in topic_model.get_topic(row['Topic'])[:5]])
                    }
                    for _, row in topic_info.iterrows() if row['Topic'] != -1
                ])
                
                st.dataframe(keywords_df)
                
                # Topic similarity visualization
                st.subheader("Topic Similarity")
                similarity_matrix = topic_model.get_topic_similarity_matrix()
                
                fig_similarity = go.Figure(data=go.Heatmap(
                    z=similarity_matrix,
                    x=[f'Topic {i}' for i in range(num_topics)],
                    y=[f'Topic {i}' for i in range(num_topics)],
                    colorscale='Viridis'
                ))
                fig_similarity.update_layout(
                    title='Topic Similarity Heatmap',
                    xaxis_title='Topics',
                    yaxis_title='Topics'
                )
                st.plotly_chart(fig_similarity)
                
                # Document examples
                st.subheader("Example Documents per Topic")
                for topic_id in range(num_topics):
                    with st.expander(f"Topic {topic_id}"):
                        topic_docs = [doc for doc, topic in zip(documents, topics) if topic == topic_id]
                        for i, doc in enumerate(topic_docs[:3], 1):
                            st.text(f"Document {i}: {doc[:200]}...")
                
            except Exception as e:
                st.error(f"Error in topic modeling: {e}")
                logging.error(f"Topic modeling error: {e}", exc_info=True)
