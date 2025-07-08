import numpy as np
import scipy.sparse as sp
from typing import Union, List, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize


class WeightedTfidfVectorizer(BaseEstimator, TransformerMixin):
    """TF-IDF vectorizer with configurable weighting schemes"""

    def __init__(
        self,
        tf_scheme: str = "raw",
        idf_scheme: str = "smooth",
        norm: Optional[str] = "l2",
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
    ):
        self.tf_scheme = tf_scheme
        self.idf_scheme = idf_scheme
        self.norm = norm
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
        )

    def _compute_tf(self, X: sp.csr_matrix) -> sp.csr_matrix:
        if self.tf_scheme == "raw":
            return X
        elif self.tf_scheme == "log":
            X.data = np.log1p(X.data)
        elif self.tf_scheme == "binary":
            X.data = np.ones_like(X.data)
        elif self.tf_scheme == "augmented":
            max_tf = X.max(axis=1).toarray().flatten()
            max_tf[max_tf == 0] = 1
            for i in range(X.shape[0]):
                start = X.indptr[i]
                end = X.indptr[i + 1]
                X.data[start:end] = 0.5 + 0.5 * (X.data[start:end] / max_tf[i])
        return X

    def _compute_idf(self, X: sp.csr_matrix) -> np.ndarray:
        n_samples = X.shape[0]
        df = np.bincount(X.indices, minlength=X.shape[1])
        df = np.maximum(df, 1)

        if self.idf_scheme == "smooth":
            return np.log((n_samples + 1) / (df + 1)) + 1
        elif self.idf_scheme == "standard":
            return np.log(n_samples / df) + 1
        elif self.idf_scheme == "probabilistic":
            return np.log((n_samples - df) / df)

    def fit(self, raw_documents: List[str], y=None):
        X = self.count_vectorizer.fit_transform(raw_documents)
        self.idf_ = self._compute_idf(X)
        return self

    def transform(self, raw_documents: List[str]) -> sp.csr_matrix:
        X = self.count_vectorizer.transform(raw_documents)
        X = self._compute_tf(X)
        X = X.multiply(self.idf_)

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    def get_feature_names_out(self):
        return self.count_vectorizer.get_feature_names_out()



class BM25Vectorizer(BaseEstimator, TransformerMixin):
    """BM25 vectorizer implementation"""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
    ):
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
        )

    def fit(self, raw_documents: List[str], y=None):
        X = self.count_vectorizer.fit_transform(raw_documents)

        # Calculate document lengths
        self.doc_lengths = np.array(X.sum(axis=1)).flatten()
        self.avg_doc_length = np.mean(self.doc_lengths)

        # Calculate IDF scores
        n_samples = X.shape[0]
        df = np.bincount(X.indices, minlength=X.shape[1])
        df = np.maximum(df, 1)
        self.idf = np.log((n_samples - df + 0.5) / (df + 0.5) + 1.0)

        return self

    def transform(self, raw_documents: List[str]) -> sp.csr_matrix:
        X = self.count_vectorizer.transform(raw_documents)
        doc_lengths = np.array(X.sum(axis=1)).flatten()

        X = sp.csr_matrix(X)

        # Calculate BM25 scores
        for i in range(X.shape[0]):
            start_idx = X.indptr[i]
            end_idx = X.indptr[i + 1]

            freqs = X.data[start_idx:end_idx]
            length_norm = 1 - self.b + self.b * doc_lengths[i] / self.avg_doc_length

            # BM25 formula
            X.data[start_idx:end_idx] = (
                ((self.k1 + 1) * freqs) / (self.k1 * length_norm + freqs)
            ) * self.idf[X.indices[start_idx:end_idx]]

        return X

    def get_feature_names_out(self):
        return self.count_vectorizer.get_feature_names_out()
    

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
        return WeightedTfidfVectorizer(
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
) -> Union[TfidfVectorizer, BM25Vectorizer, WeightedTfidfVectorizer]:
    """Create and configure the specified vectorizer type"""

    if vectorizer_type == "tfidf":
        return TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
        )
    elif vectorizer_type == "bm25":
        return BM25Vectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            k1=kwargs.get("k1", 1.5),
            b=kwargs.get("b", 0.75),
        )
    elif vectorizer_type == "weighted":
        return WeightedTfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            tf_scheme=kwargs.get("tf_scheme", "raw"),
            idf_scheme=kwargs.get("idf_scheme", "smooth"),
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")

