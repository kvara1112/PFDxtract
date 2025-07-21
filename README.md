# UK Judiciary PFD Reports Analysis v Private

A Streamlit application for scraping and analysing Prevention of Future Deaths (PFD) reports from the UK Judiciary website.

## Features

- Scrape PFD reports with customisable filters
- Download reports in CSV or Excel format
- Download associated PDFs
- Interactive data analysis and visualisation
- Topic modelling of report content
- Timeline analysis
- Category distribution analysis
- Advanced AI-powered thematic analysis with BERT
- Network visualisation of report relationships

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/uk-judiciary-pfd-analysis.git
cd uk-judiciary-pfd-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv streamlit_env
source streamlit_env/bin/activate  # On Windows, use `streamlit_env\Scripts\activate`
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLP models (run in Python terminal):
```python
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
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

```
PFDxtract/
├── app.py                    # Main application entry point
├── modules/                  # Core application modules
│   ├── __init__.py          # Package initialisation
│   ├── streamlit_components.py  # UI components and main tabs
│   ├── web_scraping.py      # Web scraping functionality
│   ├── core_utils.py        # Core utility functions
│   ├── file_prep.py         # File preparation and merging
│   ├── visualization.py     # Data visualisation functions
│   ├── vectorizer_models.py # Topic modelling functionality
│   ├── vectorizer_utils.py  # Vectorisation utilities
│   └── bert_analysis.py     # BERT-based theme analysis
├── outputs/                  # Generated files (PDFs, HTML, etc.)
├── archive/                  # Unused/old files
├── requirements.txt          # Python package requirements
└── README.md                # This file
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
