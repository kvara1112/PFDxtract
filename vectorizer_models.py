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
    df: pd.DataFrame,
    min_cluster_size: int = 3,
    max_features: int = 5000,
    min_df: float = 0.01,
    max_df: float = 0.95,
    similarity_threshold: float = 0.3,
    n_topics: int = 5,
) -> Dict:
    """
    Perform semantic clustering on documents with enhanced configurability
    
    Args:
        df: DataFrame with Content column
        min_cluster_size: Minimum documents per cluster
        max_features: Maximum number of features for vectorization
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        similarity_threshold: Similarity threshold for clustering
        n_topics: Number of topics for LDA
        
    Returns:
        Dictionary containing clustering results
    """
    try:
        initialize_nltk()

        # Clean and filter content
        docs = df["Content"].fillna("").astype(str)
        processed_docs = [clean_text_for_modeling(doc) for doc in docs]
        
        # Filter out empty documents
        valid_docs = [(i, doc) for i, doc in enumerate(processed_docs) if len(doc.split()) > 10]
        
        if len(valid_docs) < min_cluster_size:
            return {"error": "Not enough valid documents for clustering"}

        indices, cleaned_docs = zip(*valid_docs)

        # Vectorization based on session state settings
        vectorizer_type = st.session_state.get("vectorizer_type", "tfidf")
        
        # Create vectorizer with parameters from session state
        vectorizer_params = {
            "max_features": max_features,
            "min_df": max(2, int(min_df * len(cleaned_docs)) if min_df < 1 else min_df),
            "max_df": max_df,
            "ngram_range": (1, 2),
        }
        
        # Add specific parameters based on vectorizer type
        if vectorizer_type == "weighted":
            vectorizer_params.update({
                "tf_scheme": st.session_state.get("tf_scheme", "raw"),
                "idf_scheme": st.session_state.get("idf_scheme", "smooth")
            })
        elif vectorizer_type == "bm25":
            vectorizer_params.update({
                "k1": st.session_state.get("k1", 1.5),
                "b": st.session_state.get("b", 0.75)
            })

        vectorizer = create_vectorizer(vectorizer_type, **vectorizer_params)
        
        # Fit and transform documents
        doc_vectors = vectorizer.fit_transform(cleaned_docs)
        feature_names = vectorizer.get_feature_names_out()

        # Perform LDA topic modeling
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method="batch",
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
        )
        
        doc_topic_probs = lda.fit_transform(doc_vectors)

        # Hierarchical clustering for document similarity
        try:
            # Normalize vectors for cosine similarity
            normalized_vectors = normalize(doc_vectors, norm="l2")
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(normalized_vectors)
            
            # Convert to distance matrix
            distance_matrix = 1 - similarity_matrix
            
            # Perform hierarchical clustering
            n_clusters = min(n_topics, len(cleaned_docs) // min_cluster_size)
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, 
                metric="precomputed", 
                linkage="average"
            )
            
            cluster_labels = clusterer.fit_predict(distance_matrix)
            
        except Exception as e:
            logging.warning(f"Clustering failed, using topic assignments: {e}")
            # Fallback to using dominant topics as clusters
            cluster_labels = np.argmax(doc_topic_probs, axis=1)

        # Generate cluster summaries
        clusters = defaultdict(list)
        for i, (doc_idx, doc) in enumerate(valid_docs):
            clusters[cluster_labels[i]].append({
                "index": doc_idx,
                "content": doc,
                "title": df.iloc[doc_idx]["Title"] if "Title" in df.columns else f"Document {doc_idx}",
                "topic_probs": doc_topic_probs[i],
                "dominant_topic": np.argmax(doc_topic_probs[i]),
            })

        # Create cluster summaries
        cluster_summaries = {}
        for cluster_id, docs in clusters.items():
            if len(docs) >= min_cluster_size:
                # Get top terms for this cluster
                cluster_vectors = doc_vectors[[doc["index"] for doc in docs if doc["index"] < len(doc_vectors)]]
                
                if cluster_vectors.shape[0] > 0:
                    # Calculate average TF-IDF scores for cluster
                    avg_scores = np.mean(cluster_vectors, axis=0).A1
                    top_indices = avg_scores.argsort()[-10:][::-1]
                    top_terms = [feature_names[i] for i in top_indices if avg_scores[i] > 0]
                    
                    cluster_summaries[cluster_id] = {
                        "size": len(docs),
                        "documents": docs,
                        "top_terms": top_terms[:10],
                        "dominant_topic": Counter([doc["dominant_topic"] for doc in docs]).most_common(1)[0][0],
                        "avg_topic_probs": np.mean([doc["topic_probs"] for doc in docs], axis=0),
                    }

        # Get topic terms from LDA
        topic_terms = {}
        for topic_idx in range(n_topics):
            top_words_idx = lda.components_[topic_idx].argsort()[-10:][::-1]
            topic_terms[topic_idx] = [feature_names[i] for i in top_words_idx]

        # Calculate clustering metrics if we have enough clusters
        clustering_metrics = {}
        if len(set(cluster_labels)) > 1 and len(cluster_labels) > 3:
            try:
                clustering_metrics = {
                    "silhouette_score": silhouette_score(doc_vectors.toarray(), cluster_labels),
                    "calinski_harabasz_score": calinski_harabasz_score(doc_vectors.toarray(), cluster_labels),
                    "davies_bouldin_score": davies_bouldin_score(doc_vectors.toarray(), cluster_labels),
                }
            except Exception as e:
                logging.warning(f"Could not compute clustering metrics: {e}")

        return {
            "clusters": cluster_summaries,
            "topic_terms": topic_terms,
            "lda_model": lda,
            "vectorizer": vectorizer,
            "feature_names": feature_names,
            "doc_topic_probs": doc_topic_probs,
            "cluster_labels": cluster_labels,
            "valid_indices": indices,
            "metrics": clustering_metrics,
            "n_documents": len(valid_docs),
            "n_clusters": len(cluster_summaries),
            "n_topics": n_topics,
        }

    except Exception as e:
        logging.error(f"Error in semantic clustering: {e}", exc_info=True)
        return {"error": f"Clustering failed: {str(e)}"}


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