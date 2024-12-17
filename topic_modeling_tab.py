import streamlit as st
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objs as go
import re

def clean_text_for_modeling(text):
    """
    Clean text for topic modeling
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_text(df, text_column='Content'):
    """
    Preprocess text data for topic modeling
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Column containing text to model
    
    Returns:
        list: Cleaned text documents
    """
    # Remove very short or empty documents
    documents = df[text_column].dropna().astype(str)
    documents = [clean_text_for_modeling(doc) for doc in documents]
    documents = [doc for doc in documents if len(doc.split()) > 10]
    
    return documents

def render_topic_modeling_tab():
    st.header("Topic Modeling Analysis")
    
    # Check if data is available
    if 'scraped_data' not in st.session_state or st.session_state.scraped_data is None:
        st.warning("Please scrape reports first in the 'Scrape Reports' tab")
        return
    
    # Get the scraped data
    df = st.session_state.scraped_data
    
    # Sidebar for topic modeling configuration
    st.sidebar.header("Topic Modeling Options")
    
    # Number of topics selection
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
    
    # Filter data based on type if needed
    if topic_type == "Prevention of Future Death":
        filtered_df = df[df['Content'].str.contains('Prevention of Future Death', case=False, na=False)]
    elif topic_type == "Response to PFD":
        filtered_df = df[df['Content'].str.contains('Response to Prevention of Future Death', case=False, na=False)]
    else:
        filtered_df = df
    
    # Disable topic modeling if not enough documents
    if len(filtered_df) < num_topics:
        st.error(f"Not enough documents for {num_topics} topics. Please reduce topic count or scrape more reports.")
        return
    
    # Perform topic modeling
    with st.spinner("Performing Topic Modeling..."):
        try:
            # Preprocess text
            documents = preprocess_text(filtered_df)
            
            # Use SentenceTransformer for embeddings
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize BERTopic
            topic_model = BERTopic(
                embedding_model=embedding_model,
                nr_topics=num_topics,
                low_memory=True,
                calculate_probabilities=True
            )
            
            # Fit the model
            topics, probs = topic_model.fit_transform(documents)
            
            # Create results section
            st.subheader("Topic Modeling Results")
            
            # Topic distribution
            topic_counts = pd.Series(topics).value_counts()
            
            # Plotly bar chart for topic distribution
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
            
            # Create an interactive table for topic keywords
            keywords_df = pd.DataFrame([
                {
                    'Topic': row['Topic'], 
                    'Keywords': ', '.join([word for word, _ in row['KeywordSet'][:5]])
                } 
                for _, row in topic_info.iterrows() if row['Topic'] != -1
            ])
            
            st.dataframe(keywords_df)
            
            # Representative documents for each topic
            st.subheader("Representative Documents")
            
            # Function to get representative documents for each topic
            def get_representative_docs(documents, topics, topic_id):
                topic_docs = [doc for doc, topic in zip(documents, topics) if topic == topic_id]
                return topic_docs[:3]  # Get first 3 representative docs
            
            # Expander for representative documents
            with st.expander("View Representative Documents"):
                for topic in range(num_topics):
                    st.write(f"### Topic {topic}")
                    rep_docs = get_representative_docs(documents, topics, topic)
                    for i, doc in enumerate(rep_docs, 1):
                        st.text(f"Document {i}: {doc}")
            
            # Topic Similarity Visualization
            st.subheader("Topic Similarity")
            
            # Compute topic similarity matrix
            similarity_matrix = topic_model.get_topic_similarity_matrix()
            
            # Create heatmap of topic similarities
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
            
        except Exception as e:
            st.error(f"Error in topic modeling: {e}")
            st.error(f"Detailed error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# If you want to add this as a separate function for tab creation
def render_topic_modeling_section():
    render_topic_modeling_tab()
