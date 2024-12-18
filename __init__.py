"""
UK Judiciary PFD Reports Analysis Tool

Required project structure:
project_root/
    ├── app.py                 # Main application file
    ├── analysis_tab.py        # Analysis functionality
    ├── topic_modeling_tab.py  # Topic modeling functionality
    ├── requirements.txt       # Python package requirements
    ├── packages.txt          # System package requirements
    ├── __init__.py           # This file
    └── pdfs/                 # Created automatically for PDF storage
"""

import os
import logging
from pathlib import Path
import sys

# Ensure the application root is in the Python path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Create necessary directories
PDFS_DIR = ROOT_DIR / 'pdfs'
PDFS_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(ROOT_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)

# Version info
__version__ = '1.0.0'
__author__ = 'Your Name'
__license__ = 'MIT'
