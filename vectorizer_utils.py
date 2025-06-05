import numpy as np
import scipy.sparse as sp
from typing import Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


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
        # Use CountVectorizer to get basic counts
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

        # Compute IDF
        self.idf_ = self._compute_idf(X_count)

        return self

    def transform(self, documents):
        """Transform documents using weighted TF-IDF"""
        if self.vocabulary_ is None:
            raise ValueError("Vectorizer has not been fitted yet")

        # Get term counts using the fitted vocabulary
        count_vectorizer = CountVectorizer(
            vocabulary=self.vocabulary_,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
        )
        X_count = count_vectorizer.transform(documents)

        # Compute TF
        X_tf = self._compute_tf(X_count)

        # Apply IDF
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
        self.vocabulary_ = None
        self.feature_names_ = None
        self.idf_ = None
        self.avgdl_ = None

    def fit(self, documents):
        """Fit the BM25 vectorizer"""
        # Use CountVectorizer to get basic counts and vocabulary
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

        # Calculate document frequencies for IDF
        n_docs = X_count.shape[0]
        df = np.array((X_count > 0).sum(axis=0)).flatten()
        self.idf_ = np.log((n_docs - df + 0.5) / (df + 0.5))

        # Calculate average document length
        doc_lengths = np.array(X_count.sum(axis=1)).flatten()
        self.avgdl_ = np.mean(doc_lengths)

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
    """Create a vectorizer instance based on type and parameters"""
    if vectorizer_type == "tfidf":
        return TfidfVectorizer(
            max_features=params.get("max_features", 5000),
            min_df=params.get("min_df", 2),
            max_df=params.get("max_df", 0.95),
            ngram_range=params.get("ngram_range", (1, 2)),
            stop_words=params.get("stop_words", "english"),
        )
    elif vectorizer_type == "weighted":
        return WeightedTfIdfVectorizer(
            max_features=params.get("max_features", 5000),
            min_df=params.get("min_df", 2),
            max_df=params.get("max_df", 0.95),
            tf_scheme=params.get("tf_scheme", "raw"),
            idf_scheme=params.get("idf_scheme", "smooth"),
            ngram_range=params.get("ngram_range", (1, 2)),
            stop_words=params.get("stop_words", "english"),
        )
    elif vectorizer_type == "bm25":
        return BM25Vectorizer(
            max_features=params.get("max_features", 5000),
            min_df=params.get("min_df", 2),
            max_df=params.get("max_df", 0.95),
            k1=params.get("k1", 1.5),
            b=params.get("b", 0.75),
            ngram_range=params.get("ngram_range", (1, 2)),
            stop_words=params.get("stop_words", "english"),
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")


def get_vectorizer(
    vectorizer_type: str, max_features: int, min_df: float, max_df: float, **kwargs
) -> Union[TfidfVectorizer, WeightedTfIdfVectorizer, BM25Vectorizer]:
    """
    Get vectorizer instance based on type and parameters.
    """
    
    if vectorizer_type == "bm25":
        return BM25Vectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            **kwargs
        )
    elif vectorizer_type == "weighted":
        return WeightedTfIdfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            **kwargs
        )
    else:  # Default to TfidfVectorizer
        return TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            stop_words="english",
            **kwargs
        ) 