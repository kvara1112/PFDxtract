import nltk
import spacy

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Download Spacy model (small English model for efficiency)
try:
    spacy.cli.download("en_core_web_sm")
except:
    pass  # Skip if already installed
