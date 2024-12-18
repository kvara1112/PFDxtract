# UK Judiciary PFD Reports Analysis

A Streamlit application for scraping and analyzing Prevention of Future Deaths (PFD) reports from the UK Judiciary website.

## Features

- Scrape PFD reports with customizable filters
- Download reports in CSV or Excel format
- Download associated PDFs
- Interactive data analysis and visualization
- Topic modeling of report content
- Timeline analysis
- Category distribution analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/uk-judiciary-pfd-analysis.git
cd uk-judiciary-pfd-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install $(cat packages.txt)
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

```
project_root/
├── app.py                 # Main application file
├── analysis_tab.py        # Analysis functionality
├── topic_modeling_tab.py  # Topic modeling functionality
├── requirements.txt       # Python package requirements
├── packages.txt          # System package requirements
├── __init__.py           # Package initialization
└── pdfs/                 # Created automatically for PDF storage
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
