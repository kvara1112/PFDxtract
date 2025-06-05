import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st

# Import our core utilities
from core_utils import (
    clean_text_for_modeling,
    initialize_nltk,
    perform_advanced_keyword_search
)

class WeightedTfIdfVectorizer(BaseEstimator, TransformerMixin):
    """Enhanced TF-IDF vectorizer with configurable weighting schemes"""

    def __init__(
        self,
        max_features=5000,
        min_df=2,
        max_df=0.95,
        tf_scheme="raw",
        idf_scheme="smooth",
        ngram_range=(1, 2),
        stop_words="english",
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.tf_scheme = tf_scheme
        self.idf_scheme = idf_scheme
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.feature_names_ = None
        self.vocabulary_ = None
        self.idf_ = None

    def _compute_tf(self, X_count):
        """Compute term frequency with different schemes"""
        if self.tf_scheme == "raw":
            return X_count.astype(float)
        elif self.tf_scheme == "log":
            return np.log1p(X_count.astype(float))
        elif self.tf_scheme == "binary":
            return (X_count > 0).astype(float)
        elif self.tf_scheme == "augmented":
            # Augmented frequency: 0.5 + 0.5 * (tf / max_tf_in_doc)
            max_tf = np.array(X_count.max(axis=1)).flatten()
            max_tf[max_tf == 0] = 1  # Avoid division by zero
            return 0.5 + 0.5 * X_count.multiply(1.0 / max_tf[:, np.newaxis])
        else:
            raise ValueError(f"Unknown tf_scheme: {self.tf_scheme}")

    def _compute_idf(self, X_count):
        """Compute inverse document frequency with different schemes"""
        n_docs = X_count.shape[0]
        df = np.array((X_count > 0).sum(axis=0)).flatten()

        if self.idf_scheme == "smooth":
            idf = np.log(n_docs / (1 + df)) + 1
        elif self.idf_scheme == "standard":
            idf = np.log(n_docs / df)
        elif self.idf_scheme == "probabilistic":
            idf = np.log((n_docs - df) / df)
        else:
            raise ValueError(f"Unknown idf_scheme: {self.idf_scheme}")

        return idf

    def fit(self, documents):
        """Fit the vectorizer to documents"""
        # Use CountVectorizer to get initial counts and vocabulary
        count_vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
        )

        X_count = count_vectorizer.fit_transform(documents)
        self.vocabulary_ = count_vectorizer.vocabulary_
        self.feature_names_ = count_vectorizer.get_feature_names_out()

        # Compute IDF weights
        self.idf_ = self._compute_idf(X_count)

        return self

    def transform(self, documents):
        """Transform documents to TF-IDF matrix"""
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer has not been fitted yet")

        # Get term counts
        count_vectorizer = CountVectorizer(
            vocabulary=self.vocabulary_,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
        )
        X_count = count_vectorizer.transform(documents)

        # Compute TF
        X_tf = self._compute_tf(X_count)

        # Apply IDF weights
        X_tfidf = X_tf.multiply(self.idf_)

        return X_tfidf

    def fit_transform(self, documents):
        """Fit and transform documents"""
        return self.fit(documents).transform(documents)

    def get_feature_names_out(self):
        """Get feature names"""
        return self.feature_names_


class BM25Vectorizer(BaseEstimator, TransformerMixin):
    """BM25 vectorizer implementation"""

    def __init__(
        self,
        max_features=5000,
        min_df=2,
        max_df=0.95,
        k1=1.5,
        b=0.75,
        ngram_range=(1, 2),
        stop_words="english",
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.k1 = k1
        self.b = b
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.feature_names_ = None
        self.vocabulary_ = None
        self.idf_ = None
        self.doc_len_ = None
        self.avgdl_ = None

    def fit(self, documents):
        """Fit the BM25 vectorizer"""
        # Use CountVectorizer to get initial setup
        count_vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
        )

        X_count = count_vectorizer.fit_transform(documents)
        self.vocabulary_ = count_vectorizer.vocabulary_
        self.feature_names_ = count_vectorizer.get_feature_names_out()

        # Compute IDF for BM25
        n_docs = X_count.shape[0]
        df = np.array((X_count > 0).sum(axis=0)).flatten()
        self.idf_ = np.log((n_docs - df + 0.5) / (df + 0.5))

        # Compute document lengths
        self.doc_len_ = np.array(X_count.sum(axis=1)).flatten()
        self.avgdl_ = np.mean(self.doc_len_)

        return self

    def transform(self, documents):
        """Transform documents using BM25"""
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer has not been fitted yet")

        # Get term counts
        count_vectorizer = CountVectorizer(
            vocabulary=self.vocabulary_,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
        )
        X_count = count_vectorizer.transform(documents)

        # Compute document lengths for new documents
        doc_len = np.array(X_count.sum(axis=1)).flatten()

        # Convert to dense for easier computation
        X_count_dense = X_count.toarray()
        n_docs, n_features = X_count_dense.shape

        # BM25 calculation
        bm25_matrix = np.zeros((n_docs, n_features))

        for i in range(n_docs):
            for j in range(n_features):
                tf = X_count_dense[i, j]
                if tf > 0:
                    # BM25 score calculation
                    score = (
                        self.idf_[j]
                        * tf
                        * (self.k1 + 1)
                        / (tf + self.k1 * (1 - self.b + self.b * doc_len[i] / self.avgdl_))
                    )
                    bm25_matrix[i, j] = score

        return sp.csr_matrix(bm25_matrix)

    def fit_transform(self, documents):
        """Fit and transform documents"""
        return self.fit(documents).transform(documents)

    def get_feature_names_out(self):
        """Get feature names"""
        return self.feature_names_


def create_vectorizer(vectorizer_type="tfidf", **params):
    """Factory function to create appropriate vectorizer"""
    if vectorizer_type == "tfidf":
        return TfidfVectorizer(
            max_features=params.get("max_features", 5000),
            min_df=params.get("min_df", 2),
            max_df=params.get("max_df", 0.95),
            ngram_range=params.get("ngram_range", (1, 2)),
            stop_words="english",
        )
    elif vectorizer_type == "weighted":
        return WeightedTfIdfVectorizer(
            max_features=params.get("max_features", 5000),
            min_df=params.get("min_df", 2),
            max_df=params.get("max_df", 0.95),
            tf_scheme=params.get("tf_scheme", "raw"),
            idf_scheme=params.get("idf_scheme", "smooth"),
            ngram_range=params.get("ngram_range", (1, 2)),
        )
    elif vectorizer_type == "bm25":
        return BM25Vectorizer(
            max_features=params.get("max_features", 5000),
            min_df=params.get("min_df", 2),
            max_df=params.get("max_df", 0.95),
            k1=params.get("k1", 1.5),
            b=params.get("b", 0.75),
            ngram_range=params.get("ngram_range", (1, 2)),
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")


def perform_semantic_clustering(
    data: pd.DataFrame,
    min_cluster_size: int = 3,
    max_features: int = 5000,
    min_df: float = 0.01,
    max_df: float = 0.95,
    similarity_threshold: float = 0.3,
) -> Dict:
    """
    Perform semantic clustering with improved cluster selection
    """
    try:
        # Initialize NLTK resources
        initialize_nltk()

        # Validate input data
        if "Content" not in data.columns:
            raise ValueError("Input data must contain 'Content' column")

        processed_texts = data["Content"].apply(clean_text_for_modeling)
        valid_mask = processed_texts.notna() & (processed_texts != "")
        processed_texts = processed_texts[valid_mask]

        if len(processed_texts) == 0:
            raise ValueError("No valid text content found after preprocessing")

        # Keep the original data for display
        display_data = data[valid_mask].copy()

        # Calculate optimal parameters based on dataset size
        n_docs = len(processed_texts)
        min_clusters = max(2, min(3, n_docs // 20))  # More conservative minimum
        max_clusters = max(3, min(8, n_docs // 10))  # More conservative maximum

        # Get vectorization parameters from session state
        vectorizer_type = st.session_state.get("vectorizer_type", "tfidf")
        vectorizer_params = {}

        if vectorizer_type == "bm25":
            vectorizer_params.update(
                {
                    "k1": st.session_state.get("bm25_k1", 1.5),
                    "b": st.session_state.get("bm25_b", 0.75),
                }
            )
        elif vectorizer_type == "weighted":
            vectorizer_params.update(
                {
                    "tf_scheme": st.session_state.get("tf_scheme", "raw"),
                    "idf_scheme": st.session_state.get("idf_scheme", "smooth"),
                }
            )

        # Create the vectorizer
        vectorizer = get_vectorizer(
            vectorizer_type=vectorizer_type,
            max_features=max_features,
            min_df=max(min_df, 3 / len(processed_texts)),
            max_df=min(max_df, 0.7),
            **vectorizer_params,
        )

        # Create document vectors
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()

        # Find optimal number of clusters
        best_n_clusters, best_labels = find_optimal_clusters(
            tfidf_matrix,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            min_cluster_size=min_cluster_size,
        )

        # Calculate final clustering quality
        silhouette_avg = silhouette_score(
            tfidf_matrix.toarray(), best_labels, metric="euclidean"
        )

        # Calculate similarities using similarity threshold
        similarity_matrix = cosine_similarity(tfidf_matrix)
        similarity_matrix[similarity_matrix < similarity_threshold] = 0

        # Extract cluster information
        clusters = []
        for cluster_id in range(best_n_clusters):
            cluster_indices = np.where(best_labels == cluster_id)[0]

            # Skip if cluster is too small
            if len(cluster_indices) < min_cluster_size:
                continue

            # Calculate cluster terms
            cluster_tfidf = tfidf_matrix[cluster_indices].toarray()
            centroid = np.mean(cluster_tfidf, axis=0)

            # Get important terms with improved distinctiveness
            term_scores = []
            for idx, score in enumerate(centroid):
                if score > 0:
                    term = feature_names[idx]
                    cluster_freq = np.mean(cluster_tfidf[:, idx] > 0)
                    total_freq = np.mean(tfidf_matrix[:, idx].toarray() > 0)
                    distinctiveness = cluster_freq / (total_freq + 1e-10)

                    term_scores.append(
                        {
                            "term": term,
                            "score": float(score * distinctiveness),
                            "cluster_frequency": float(cluster_freq),
                            "total_frequency": float(total_freq),
                        }
                    )

            term_scores.sort(key=lambda x: x["score"], reverse=True)
            top_terms = term_scores[:20]

            # Get representative documents
            doc_similarities = []
            for idx in cluster_indices:
                doc_vector = tfidf_matrix[idx].toarray().flatten()
                sim_to_centroid = cosine_similarity(
                    doc_vector.reshape(1, -1), centroid.reshape(1, -1)
                )[0][0]

                doc_info = {
                    "title": display_data.iloc[idx]["Title"],
                    "date": display_data.iloc[idx]["date_of_report"],
                    "similarity": float(sim_to_centroid),
                    "summary": display_data.iloc[idx]["Content"][:500],
                }
                doc_similarities.append((idx, sim_to_centroid, doc_info))

            # Sort by similarity and get representative docs
            doc_similarities.sort(key=lambda x: x[1], reverse=True)
            representative_docs = [item[2] for item in doc_similarities]

            # Calculate cluster cohesion
            cluster_similarities = similarity_matrix[cluster_indices][
                :, cluster_indices
            ]
            cohesion = float(np.mean(cluster_similarities))

            clusters.append(
                {
                    "id": len(clusters),
                    "size": len(cluster_indices),
                    "cohesion": cohesion,
                    "terms": top_terms,
                    "documents": representative_docs,
                    "balance_ratio": max(
                        len(cluster_indices)
                        for cluster_indices in [
                            np.where(best_labels == i)[0]
                            for i in range(best_n_clusters)
                        ]
                    )
                    / min(
                        len(cluster_indices)
                        for cluster_indices in [
                            np.where(best_labels == i)[0]
                            for i in range(best_n_clusters)
                        ]
                    ),
                }
            )

        # Add cluster quality metrics to results
        metrics = {
            "silhouette_score": float(silhouette_avg),
            "calinski_score": float(
                calinski_harabasz_score(tfidf_matrix.toarray(), best_labels)
            ),
            "davies_score": float(
                davies_bouldin_score(tfidf_matrix.toarray(), best_labels)
            ),
            "balance_ratio": float(
                max(len(c["documents"]) for c in clusters)
                / min(len(c["documents"]) for c in clusters)
            ),
        }

        return {
            "n_clusters": len(clusters),
            "total_documents": len(processed_texts),
            "silhouette_score": float(silhouette_avg),
            "clusters": clusters,
            "vectorizer_type": vectorizer_type,
            "quality_metrics": metrics,
        }

    except Exception as e:
        logging.error(f"Error in semantic clustering: {e}", exc_info=True)
        raise


def generate_cluster_summary(documents: List[str], top_terms: List[str], max_length: int = 200) -> str:
    """Generate a natural language summary for a cluster"""
    try:
        # Use the top terms to create a thematic summary
        if len(top_terms) >= 3:
            primary_themes = top_terms[:3]
            summary = f"This cluster focuses on {', '.join(primary_themes[:-1])} and {primary_themes[-1]}."
            
            # Add more context based on document count
            doc_count = len(documents)
            if doc_count > 10:
                summary += f" This is a major theme appearing in {doc_count} documents."
            elif doc_count > 5:
                summary += f" This theme appears in {doc_count} documents."
            else:
                summary += f" This is a smaller theme with {doc_count} documents."
                
            # Add additional themes if available
            if len(top_terms) > 3:
                additional_themes = top_terms[3:6]
                summary += f" Related concepts include {', '.join(additional_themes)}."
                
            return summary[:max_length]
        else:
            return f"Small cluster with {len(documents)} documents focusing on general themes."
            
    except Exception as e:
        logging.error(f"Error generating cluster summary: {e}")
        return f"Cluster containing {len(documents)} documents."


def calculate_topic_coherence(lda_model, vectorizer, documents: List[str]) -> float:
    """Calculate topic coherence score"""
    try:
        # This is a simplified coherence calculation
        # For production use, consider using gensim's coherence models
        
        feature_names = vectorizer.get_feature_names_out()
        coherence_scores = []
        
        for topic_idx in range(lda_model.n_components):
            # Get top words for topic
            top_words_idx = lda_model.components_[topic_idx].argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            
            # Calculate pairwise co-occurrence
            word_pairs = [(top_words[i], top_words[j]) 
                         for i in range(len(top_words)) 
                         for j in range(i+1, len(top_words))]
            
            pair_scores = []
            for word1, word2 in word_pairs:
                co_occurrence = sum(1 for doc in documents 
                                  if word1 in doc.lower() and word2 in doc.lower())
                if co_occurrence > 0:
                    pair_scores.append(co_occurrence)
            
            if pair_scores:
                coherence_scores.append(np.mean(pair_scores))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
        
    except Exception as e:
        logging.error(f"Error calculating topic coherence: {e}")
        return 0.0


def optimize_cluster_parameters(
    df: pd.DataFrame, 
    param_ranges: Dict = None
) -> Dict:
    """
    Optimize clustering parameters using grid search
    
    Args:
        df: DataFrame with Content column
        param_ranges: Dictionary of parameter ranges to test
        
    Returns:
        Dictionary with optimal parameters and scores
    """
    if param_ranges is None:
        param_ranges = {
            "n_topics": [3, 5, 7, 10],
            "min_cluster_size": [2, 3, 5],
            "similarity_threshold": [0.2, 0.3, 0.4],
            "max_features": [3000, 5000, 7000],
        }
    
    best_score = -1
    best_params = {}
    results = []
    
    try:
        # Get all parameter combinations (simplified for performance)
        from itertools import product
        
        param_combinations = list(product(
            param_ranges["n_topics"],
            param_ranges["min_cluster_size"], 
            param_ranges["similarity_threshold"],
            param_ranges["max_features"]
        ))
        
        # Limit combinations for performance
        param_combinations = param_combinations[:20]  # Test max 20 combinations
        
        for n_topics, min_cluster_size, similarity_threshold, max_features in param_combinations:
            try:
                result = perform_semantic_clustering(
                    df,
                    min_cluster_size=min_cluster_size,
                    max_features=max_features,
                    similarity_threshold=similarity_threshold,
                    n_topics=n_topics,
                )
                
                if "error" not in result:
                    # Calculate composite score
                    metrics = result.get("metrics", {})
                    n_clusters = result.get("n_clusters", 0)
                    
                    # Favor solutions with reasonable number of clusters
                    cluster_penalty = abs(n_clusters - n_topics) * 0.1
                    
                    # Use silhouette score as primary metric
                    silhouette = metrics.get("silhouette_score", 0)
                    
                    # Composite score
                    score = silhouette - cluster_penalty
                    
                    results.append({
                        "params": {
                            "n_topics": n_topics,
                            "min_cluster_size": min_cluster_size,
                            "similarity_threshold": similarity_threshold,
                            "max_features": max_features,
                        },
                        "score": score,
                        "metrics": metrics,
                        "n_clusters": n_clusters,
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "n_topics": n_topics,
                            "min_cluster_size": min_cluster_size,
                            "similarity_threshold": similarity_threshold,
                            "max_features": max_features,
                        }
                        
            except Exception as e:
                logging.warning(f"Failed parameter combination: {e}")
                continue
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": sorted(results, key=lambda x: x["score"], reverse=True),
        }
        
    except Exception as e:
        logging.error(f"Error in parameter optimization: {e}")
        return {"error": f"Parameter optimization failed: {str(e)}"}


def render_topic_summary_tab(data: pd.DataFrame = None):
    """Render the topic analysis and summary tab"""
    st.subheader("Topic Analysis & Document Summaries")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload processed file from Step 2",
        type=["csv", "xlsx"],
        help="Upload the merged file from the Scraped File Preparation step",
        key="topic_analysis_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.success(f"File loaded successfully with {len(data)} records")
            
            # Analysis settings
            with st.expander("Topic Analysis Settings", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    n_topics = st.slider(
                        "Number of Topics",
                        min_value=3,
                        max_value=15,
                        value=5,
                        key="n_topics_slider"
                    )
                    
                    max_features = st.slider(
                        "Maximum Features",
                        min_value=500,
                        max_value=5000,
                        value=1000,
                        step=100,
                        key="max_features_slider"
                    )
                
                with col2:
                    min_cluster_size = st.slider(
                        "Minimum Cluster Size",
                        min_value=2,
                        max_value=10,
                        value=3,
                        key="min_cluster_size_slider"
                    )
                    
                    similarity_threshold = st.slider(
                        "Similarity Threshold",
                        min_value=0.1,
                        max_value=0.8,
                        value=0.3,
                        step=0.05,
                        key="similarity_threshold_slider"
                    )
            
            # Run clustering analysis
            if st.button("🔬 Run Topic Analysis", key="run_topic_analysis"):
                try:
                    with st.spinner("Performing semantic clustering and topic modeling..."):
                        # Initialize NLTK
                        initialize_nltk()
                        
                        # Run clustering
                        clustering_result = perform_semantic_clustering(
                            data,
                            min_cluster_size=min_cluster_size,
                            max_features=max_features,
                            similarity_threshold=similarity_threshold,
                            n_topics=n_topics,
                        )
                        
                        if "error" in clustering_result:
                            st.error(f"Clustering failed: {clustering_result['error']}")
                        else:
                            st.success("Topic analysis completed successfully!")
                            
                            # Display results
                            st.subheader("Analysis Results")
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Documents Analyzed", clustering_result.get("n_documents", 0))
                            with col2:
                                st.metric("Clusters Found", clustering_result.get("n_clusters", 0))
                            with col3:
                                st.metric("Topics Generated", clustering_result.get("n_topics", 0))
                            
                            # Display clusters
                            clusters = clustering_result.get("clusters", {})
                            if clusters:
                                st.subheader("Document Clusters")
                                
                                for cluster_id, cluster_info in clusters.items():
                                    with st.expander(f"Cluster {cluster_id} - {cluster_info['size']} documents"):
                                        st.write(f"**Top Terms:** {', '.join(cluster_info['top_terms'][:10])}")
                                        st.write(f"**Dominant Topic:** {cluster_info['dominant_topic']}")
                                        
                                        # Show some example documents
                                        st.write("**Example Documents:**")
                                        for i, doc in enumerate(cluster_info['documents'][:3]):
                                            st.write(f"{i+1}. {doc['title'][:100]}...")
                            
                            # Display topics
                            topic_terms = clustering_result.get("topic_terms", {})
                            if topic_terms:
                                st.subheader("Topic Terms")
                                
                                cols = st.columns(min(len(topic_terms), 3))
                                for i, (topic_id, terms) in enumerate(topic_terms.items()):
                                    with cols[i % len(cols)]:
                                        st.write(f"**Topic {topic_id}**")
                                        st.write(", ".join(terms[:8]))
                            
                            # Clustering metrics
                            metrics = clustering_result.get("metrics", {})
                            if metrics:
                                st.subheader("Clustering Quality Metrics")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if "silhouette_score" in metrics:
                                        st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
                                with col2:
                                    if "calinski_harabasz_score" in metrics:
                                        st.metric("Calinski-Harabasz Score", f"{metrics['calinski_harabasz_score']:.0f}")
                                with col3:
                                    if "davies_bouldin_score" in metrics:
                                        st.metric("Davies-Bouldin Score", f"{metrics['davies_bouldin_score']:.3f}")
                            
                            # Download results
                            st.subheader("Download Results")
                            
                            # Prepare summary data
                            summary_data = []
                            for cluster_id, cluster_info in clusters.items():
                                for doc in cluster_info['documents']:
                                    summary_data.append({
                                        "Document_Index": doc['index'],
                                        "Title": doc['title'],
                                        "Cluster_ID": cluster_id,
                                        "Cluster_Size": cluster_info['size'],
                                        "Top_Terms": ", ".join(cluster_info['top_terms'][:5]),
                                        "Dominant_Topic": cluster_info['dominant_topic'],
                                    })
                            
                            if summary_data:
                                summary_df = pd.DataFrame(summary_data)
                                
                                # Merge with original data
                                if "index" in data.columns or data.index.name:
                                    # Reset index to get numeric index
                                    data_reset = data.reset_index()
                                    merged_df = summary_df.merge(
                                        data_reset, 
                                        left_on="Document_Index", 
                                        right_index=True, 
                                        how="left"
                                    )
                                else:
                                    merged_df = summary_df
                                
                                # Create Excel output
                                from datetime import datetime
                                import io
                                
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"topic_analysis_results_{timestamp}.xlsx"
                                
                                excel_buffer = io.BytesIO()
                                merged_df.to_excel(excel_buffer, index=False, engine="openpyxl")
                                excel_buffer.seek(0)
                                
                                st.download_button(
                                    "📥 Download Topic Analysis Results (Excel)",
                                    data=excel_buffer.getvalue(),
                                    file_name=filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                
                except Exception as e:
                    st.error(f"Error in topic analysis: {str(e)}")
                    logging.error(f"Topic analysis error: {e}", exc_info=True)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("Please upload a file to begin topic analysis.") 