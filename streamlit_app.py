"""
Streamlit Cloud deployment entry point.
This file is used when deploying to Streamlit Cloud.
"""

import sys
from pathlib import Path

# Ensure modules are in path
sys.path.append(str(Path(__file__).parent))

# Import and run main app
from app import main

if __name__ == "__main__":
    main()
