import re
import logging
import pandas as pd
import numpy as np
import scipy.sparse as sp
import streamlit as st
from typing import Dict, List, Tuple, Optional
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
from typing import Union
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Import vectorizer utilities from separate module
from vectorizer_utils import (
    WeightedTfIdfVectorizer,
    BM25Vectorizer,
    get_vectorizer,
    create_vectorizer
)

# Import our core utilities
from core_utils import (
    clean_text_for_modeling,
    initialize_nltk,
    perform_advanced_keyword_search,
    process_scraped_data,
    filter_by_categories
)

def find_optimal_clusters(
    tfidf_matrix: sp.csr_matrix,
    min_clusters: int = 2,
    max_clusters: int = 10,
    min_cluster_size: int = 3,
) -> Tuple[int, np.ndarray]:
    """Find optimal number of clusters with relaxed constraints"""

    best_score = -1
    best_n_clusters = min_clusters
    best_labels = None

    # Store metrics for each clustering attempt
    metrics = []

    # Try different numbers of clusters
    for n_clusters in range(min_clusters, max_clusters + 1):
        try:
            # Perform clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="euclidean", linkage="ward"
            )

            labels = clustering.fit_predict(tfidf_matrix.toarray())

            # Calculate cluster sizes
            cluster_sizes = np.bincount(labels)

            # Skip if any cluster is too small
            if min(cluster_sizes) < min_cluster_size:
                continue

            # Calculate balance ratio (smaller is better)
            balance_ratio = max(cluster_sizes) / min(cluster_sizes)

            # Skip only if clusters are extremely imbalanced
            if balance_ratio > 10:  # Relaxed from 5 to 10
                continue

            # Calculate clustering metrics
            sil_score = silhouette_score(
                tfidf_matrix.toarray(), labels, metric="euclidean"
            )

            # Simplified scoring focused on silhouette and basic balance
            combined_score = sil_score * (
                1 - (balance_ratio / 20)
            )  # Relaxed balance penalty

            metrics.append(
                {
                    "n_clusters": n_clusters,
                    "silhouette": sil_score,
                    "balance_ratio": balance_ratio,
                    "combined_score": combined_score,
                    "labels": labels,
                }
            )

        except Exception as e:
            logging.warning(f"Error trying {n_clusters} clusters: {str(e)}")
            continue

    # If no configurations met the strict criteria, try to find the best available
    if not metrics:
        # Try again with minimal constraints
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters, metric="euclidean", linkage="ward"
                )

                labels = clustering.fit_predict(tfidf_matrix.toarray())
                sil_score = silhouette_score(
                    tfidf_matrix.toarray(), labels, metric="euclidean"
                )

                if sil_score > best_score:
                    best_score = sil_score
                    best_n_clusters = n_clusters
                    best_labels = labels

            except Exception as e:
                continue

        if best_labels is None:
            # If still no valid configuration, use minimum number of clusters
            clustering = AgglomerativeClustering(
                n_clusters=min_clusters, metric="euclidean", linkage="ward"
            )
            best_labels = clustering.fit_predict(tfidf_matrix.toarray())
            best_n_clusters = min_clusters
    else:
        # Use the best configuration from metrics
        best_metric = max(metrics, key=lambda x: x["combined_score"])
        best_n_clusters = best_metric["n_clusters"]
        best_labels = best_metric["labels"]

    return best_n_clusters, best_labels


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


def generate_abstractive_summary(terms, documents):
    """Generate a natural language summary from cluster terms and documents"""
    try:
        if not terms:
            return "No significant terms found for this cluster."
        
        # Extract top term names
        if isinstance(terms[0], dict):
            top_terms = [term["term"] for term in terms[:5]]
        else:
            top_terms = terms[:5]
        
        # Create a basic summary
        if len(top_terms) >= 3:
            summary = f"This cluster focuses on {', '.join(top_terms[:-1])} and {top_terms[-1]}."
        elif len(top_terms) == 2:
            summary = f"This cluster focuses on {top_terms[0]} and {top_terms[1]}."
        elif len(top_terms) == 1:
            summary = f"This cluster focuses on {top_terms[0]}."
        else:
            summary = "This cluster contains miscellaneous documents."
        
        # Add document count information
        doc_count = len(documents)
        if doc_count > 10:
            summary += f" This is a major theme appearing in {doc_count} documents."
        elif doc_count > 5:
            summary += f" This theme appears in {doc_count} documents."
        else:
            summary += f" This is a smaller theme with {doc_count} documents."
        
        return summary
        
    except Exception as e:
        logging.error(f"Error generating abstractive summary: {e}")
        return f"Cluster containing {len(documents) if documents else 0} documents."


def render_summary_tab(cluster_results: Dict, original_data: pd.DataFrame) -> None:
    """Render cluster summaries and records with flexible column handling"""
    if not cluster_results or "clusters" not in cluster_results:
        st.warning("No cluster results available.")
        return

    st.write(
        f"Found {cluster_results['total_documents']} total documents in {cluster_results['n_clusters']} clusters"
    )

    for cluster in cluster_results["clusters"]:
        st.markdown(f"### Cluster {cluster['id']+1} ({cluster['size']} documents)")

        # Overview
        st.markdown("#### Overview")
        abstractive_summary = generate_abstractive_summary(
            cluster["terms"], cluster["documents"]
        )
        st.write(abstractive_summary)

        # Key terms table
        st.markdown("#### Key Terms")
        terms_df = pd.DataFrame(
            [
                {
                    "Term": term["term"],
                    "Frequency": f"{term['cluster_frequency']*100:.0f}%",
                }
                for term in cluster["terms"][:10]
            ]
        )
        st.dataframe(terms_df, hide_index=True)

        # Records
        st.markdown("#### Records")
        st.success(f"Showing {len(cluster['documents'])} matching documents")

        # Get the full records from original data
        doc_titles = [doc.get("title", "") for doc in cluster["documents"]]
        cluster_docs = original_data[original_data["Title"].isin(doc_titles)].copy()

        # Sort to match the original order
        title_to_position = {title: i for i, title in enumerate(doc_titles)}
        cluster_docs["sort_order"] = cluster_docs["Title"].map(title_to_position)
        cluster_docs = cluster_docs.sort_values("sort_order").drop("sort_order", axis=1)

        # Determine available columns
        available_columns = []
        column_config = {}

        # Always include URL and Title if available
        if "URL" in cluster_docs.columns:
            available_columns.append("URL")
            column_config["URL"] = st.column_config.LinkColumn("Report Link")

        if "Title" in cluster_docs.columns:
            available_columns.append("Title")
            column_config["Title"] = st.column_config.TextColumn("Title")

        # Add date if available
        if "date_of_report" in cluster_docs.columns:
            available_columns.append("date_of_report")
            column_config["date_of_report"] = st.column_config.DateColumn(
                "Date of Report", format="DD/MM/YYYY"
            )

        # Add optional columns if available
        optional_columns = [
            "ref",
            "deceased_name",
            "coroner_name",
            "coroner_area",
            "categories",
        ]
        for col in optional_columns:
            if col in cluster_docs.columns:
                available_columns.append(col)
                if col == "categories":
                    column_config[col] = st.column_config.ListColumn("Categories")
                else:
                    column_config[col] = st.column_config.TextColumn(
                        col.replace("_", " ").title()
                    )

        # Display the dataframe with available columns
        if available_columns:
            st.dataframe(
                cluster_docs[available_columns],
                column_config=column_config,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("No displayable columns found in the data")

        st.markdown("---")


def render_topic_summary_tab(data: pd.DataFrame = None) -> None:
    """Topic analysis with weighting schemes and essential controls"""
    st.subheader("Topic Analysis & Summaries")
    
    # Start with file upload, ignoring any previously loaded data
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file for Topic Analysis",
        type=["csv", "xlsx"],
        help="Upload a preprocessed file containing report content",
        key="topic_analysis_uploader"
    )

    # Only proceed with analysis if a file is uploaded
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
                
            # Process the data
            data = process_scraped_data(data)
            
            # Validate that we have the needed content column
            if "Content" not in data.columns:
                st.error("The uploaded file does not contain a 'Content' column needed for topic analysis.")
                return
                
            st.success(f"File loaded successfully with {len(data)} rows.")
            
            # Text Processing options
            st.subheader("Analysis Settings")
            col1, col2 = st.columns(2)

            with col1:
                # Vectorization method
                vectorizer_type = st.selectbox(
                    "Vectorization Method",
                    options=["tfidf", "bm25", "weighted"],
                    help="Choose how to convert text to numerical features",
                )

                # Weighting Schemes
                if vectorizer_type == "weighted":
                    tf_scheme = st.selectbox(
                        "Term Frequency Scheme",
                        options=["raw", "log", "binary", "augmented"],
                        help="How to count term occurrences",
                    )
                    idf_scheme = st.selectbox(
                        "Document Frequency Scheme",
                        options=["smooth", "standard", "probabilistic"],
                        help="How to weight document frequencies",
                    )
                elif vectorizer_type == "bm25":
                    k1 = st.slider(
                        "Term Saturation (k1)",
                        min_value=0.5,
                        max_value=3.0,
                        value=1.5,
                        help="Controls term frequency impact",
                    )
                    b = st.slider(
                        "Length Normalization (b)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.75,
                        help="Document length impact",
                    )

            with col2:
                # Clustering Parameters
                min_cluster_size = st.slider(
                    "Minimum Group Size",
                    min_value=2,
                    max_value=10,
                    value=3,
                    help="Minimum documents per theme",
                )

                max_features = st.slider(
                    "Maximum Features",
                    min_value=1000,
                    max_value=10000,
                    value=5000,
                    step=1000,
                    help="Number of terms to consider",
                )

            # Date range selection
            st.subheader("Date Range")
            date_col1, date_col2 = st.columns(2)
            
            # Only show date selector if date_of_report column exists
            if "date_of_report" in data.columns and pd.api.types.is_datetime64_any_dtype(data["date_of_report"]):
                with date_col1:
                    start_date = st.date_input(
                        "From",
                        value=data["date_of_report"].min().date(),
                        min_value=data["date_of_report"].min().date(),
                        max_value=data["date_of_report"].max().date(),
                    )

                with date_col2:
                    end_date = st.date_input(
                        "To",
                        value=data["date_of_report"].max().date(),
                        min_value=data["date_of_report"].min().date(),
                        max_value=data["date_of_report"].max().date(),
                    )
                
                # Apply date filter
                data = data[
                    (data["date_of_report"].dt.date >= start_date)
                    & (data["date_of_report"].dt.date <= end_date)
                ]
            else:
                st.info("No date column found. Date filtering is not available.")

            # Category selection
            if "categories" in data.columns:
                all_categories = set()
                for cats in data["categories"].dropna():
                    if isinstance(cats, list):
                        all_categories.update(cats)
                    elif isinstance(cats, str):
                        # Handle comma-separated strings
                        all_categories.update(cat.strip() for cat in cats.split(","))

                # Remove any empty strings
                all_categories = {cat for cat in all_categories if cat and isinstance(cat, str)}

                if all_categories:
                    categories = st.multiselect(
                        "Filter by Categories (Optional)",
                        options=sorted(all_categories),
                        help="Select specific categories to analyse",
                    )
                    
                    # Apply category filter if needed
                    if categories:
                        data = filter_by_categories(data, categories)
            else:
                st.info("No categories column found. Category filtering is not available.")

            # Analysis button
            analyze_clicked = st.button(
                "🔍 Analyse Documents", type="primary", use_container_width=True
            )

            # Run analysis if button is clicked
            if analyze_clicked:
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Initialize
                    progress_bar.progress(0.2)
                    status_text.text("Processing documents...")
                    initialize_nltk()

                    # Remove empty content
                    filtered_df = data[
                        data["Content"].notna()
                        & (data["Content"].str.strip() != "")
                    ]

                    if len(filtered_df) < min_cluster_size:
                        progress_bar.empty()
                        status_text.empty()
                        st.warning(
                            f"Not enough documents match the criteria. Found {len(filtered_df)}, need at least {min_cluster_size}."
                        )
                        return

                    # Process content
                    progress_bar.progress(0.4)
                    status_text.text("Identifying themes...")

                    processed_df = pd.DataFrame(
                        {
                            "Content": filtered_df["Content"],
                            "Title": filtered_df["Title"],
                            "date_of_report": filtered_df["date_of_report"] if "date_of_report" in filtered_df.columns else None,
                            "URL": filtered_df["URL"] if "URL" in filtered_df.columns else None,
                            "categories": filtered_df["categories"] if "categories" in filtered_df.columns else None,
                        }
                    )

                    progress_bar.progress(0.6)
                    status_text.text("Analyzing patterns...")

                    # Prepare vectorizer parameters
                    vectorizer_params = {}
                    if vectorizer_type == "weighted":
                        vectorizer_params.update(
                            {"tf_scheme": tf_scheme, "idf_scheme": idf_scheme}
                        )
                    elif vectorizer_type == "bm25":
                        vectorizer_params.update({"k1": k1, "b": b})

                    # Store vectorization settings in session state
                    st.session_state.vectorizer_type = vectorizer_type
                    st.session_state.update(vectorizer_params)

                    # Perform clustering
                    cluster_results = perform_semantic_clustering(
                        processed_df,
                        min_cluster_size=min_cluster_size,
                        max_features=max_features,
                        min_df=2 / len(processed_df),
                        max_df=0.95,
                        similarity_threshold=0.3,
                    )

                    progress_bar.progress(0.8)
                    status_text.text("Generating summaries...")

                    # Store results
                    st.session_state.topic_model = cluster_results

                    progress_bar.progress(1.0)
                    status_text.text("Analysis complete!")

                    progress_bar.empty()
                    status_text.empty()

                    # Display results
                    render_summary_tab(cluster_results, processed_df)

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Analysis error: {str(e)}")
                    logging.error(f"Analysis error: {e}", exc_info=True)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Show instructions when no file is uploaded
        st.info("Please upload a file to begin topic analysis.")
        
        with st.expander("📋 File Requirements"):
            st.markdown("""
            ## Required Columns
            
            For topic analysis, your file should include:
            
            - **Content**: The text content to analyze (required)
            - **Title**: Report titles (recommended)
            - **date_of_report**: Report dates (optional, for filtering)
            - **categories**: Report categories (optional, for filtering)
            
            Files prepared from Step 2 "Scraped File Preparation" are ideal for this analysis.
            """)
            
        # Show a sample of what to expect
        with st.expander("🔍 What to Expect"):
            st.markdown("""
            ## Topic Analysis Results
            
            The analysis will generate:
            
            1. **Topic clusters**: Groups of similar documents
            2. **Key terms**: Important words in each topic
            3. **Topic summaries**: Brief overview of each topic's content
            4. **Network visualizations**: Showing relationships between terms
            
            The quality of results depends on having enough documents with good text content.
            """)
