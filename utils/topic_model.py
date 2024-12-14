import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora

class TopicModeler:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, texts):
        """Preprocess text for topic modeling"""
        processed_docs = []
        for doc in texts:
            # Tokenize and clean
            tokens = word_tokenize(doc.lower())
            tokens = [token for token in tokens 
                      if token.isalnum() and token not in self.stop_words]
            processed_docs.append(tokens)
        return processed_docs

    def lda_sklearn(self, texts, num_topics=5):
        """Perform LDA topic modeling using scikit-learn"""
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tfidf = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_output = lda.fit_transform(tfidf)
        
        # Get top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[:-10 - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append(top_features)
        
        return lda_output, topics

    def lda_gensim(self, texts, num_topics=5):
        """Perform LDA topic modeling using Gensim"""
        processed_docs = self.preprocess_text(texts)
        
        # Create Dictionary
        dictionary = corpora.Dictionary(processed_docs)
        
        # Create Corpus
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        
        # Train LDA model
        lda_model = gensim.models.LdaMulticore(
            corpus=corpus, 
            id2word=dictionary, 
            num_topics=num_topics
        )
        
        return lda_model, dictionary
