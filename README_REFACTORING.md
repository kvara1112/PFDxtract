# UK Judiciary PFD Reports Analysis Tool - Refactored Structure

## Overview

The original `app.py` file (13,056 lines) has been successfully refactored into a modular structure with 7 separate Python files. This improves code organization, maintainability, and readability.

## New File Structure

### 1. `core_utils.py`
**Purpose**: Core utility functions and data processing operations
**Key Components**:
- Text cleaning and processing functions (`clean_text`, `clean_text_for_modeling`)
- Metadata extraction (`extract_metadata`, `extract_concern_text`)
- Data processing and filtering functions
- Date formatting and utility functions
- Category and area filtering functions

### 2. `vectorizer_models.py`
**Purpose**: Text vectorization, clustering, and topic modeling
**Key Components**:
- `WeightedTfIdfVectorizer` class with configurable weighting schemes
- `BM25Vectorizer` class for enhanced text vectorization
- `perform_semantic_clustering` function for document clustering
- Topic modeling and optimization functions
- Model evaluation metrics

### 3. `web_scraping.py`
**Purpose**: Web scraping functionality for UK Judiciary website
**Key Components**:
- `scrape_pfd_reports` main scraping function
- PDF download and text extraction functions
- URL construction and pagination handling
- Report data extraction and processing
- Batch saving and progress tracking

### 4. `bert_analysis.py`
**Purpose**: BERT-based theme analysis and file merging
**Key Components**:
- `BERTResultsAnalyzer` class for file merging and processing
- `ThemeAnalyzer` class for AI-powered theme extraction
- Framework definitions (I-SIRch, House of Commons, Extended Analysis)
- BERT embedding generation and document analysis
- Theme highlighting and visualization support

### 5. `visualization.py`
**Purpose**: All plotting and data visualization functions
**Key Components**:
- Statistical plots (`plot_timeline`, `plot_category_distribution`)
- Data quality analysis visualizations
- Framework heatmaps and theme analysis charts
- Network visualization for topic modeling
- LDA visualization integration

### 6. `streamlit_components.py`
**Purpose**: Streamlit UI components and session management
**Key Components**:
- Session state initialization and management
- Authentication and security functions
- Tab rendering functions for each analysis step
- File upload and data validation handlers
- Error handling and user feedback

### 7. `main_app.py`
**Purpose**: Main application entry point
**Key Components**:
- Application orchestration and navigation
- Module imports and initialization
- Main analysis tab implementation
- User interface coordination

## How to Use the Refactored Code

### Option 1: Use the New Modular Structure (Recommended)
```bash
# Run the new main application
streamlit run main_app.py
```

### Option 2: Keep Using the Original File
Your original `app.py` file remains unchanged and functional:
```bash
# Continue using the original file
streamlit run app.py
```

## Benefits of the Refactored Structure

### 1. **Improved Maintainability**
- Each module has a specific responsibility
- Easier to locate and modify specific functionality
- Reduced complexity in individual files

### 2. **Better Code Organization**
- Related functions grouped together
- Clear separation of concerns
- Logical module hierarchy

### 3. **Enhanced Readability**
- Smaller, focused files are easier to understand
- Clear imports show dependencies between modules
- Better documentation and comments

### 4. **Easier Development**
- Multiple developers can work on different modules simultaneously
- Faster loading and testing of individual components
- Simplified debugging and error tracking

### 5. **Reusability**
- Individual modules can be imported into other projects
- Functions can be easily tested in isolation
- Components can be extended without affecting others

## Dependencies Between Modules

```
main_app.py
├── streamlit_components.py
│   ├── core_utils.py
│   ├── web_scraping.py
│   ├── vectorizer_models.py
│   ├── bert_analysis.py
│   └── visualization.py
├── vectorizer_models.py
│   └── core_utils.py
├── visualization.py
│   └── core_utils.py
├── bert_analysis.py
│   └── core_utils.py
└── web_scraping.py
    └── core_utils.py
```

## Migration Guide

### If You Have Customizations
1. **Identify your changes**: Note any modifications you've made to the original `app.py`
2. **Map to new modules**: Determine which new module contains the functionality you modified
3. **Apply changes**: Make the same modifications to the appropriate new module
4. **Test thoroughly**: Ensure your customizations work in the new structure

### Common Customization Locations
- **UI changes**: Look in `streamlit_components.py` or `main_app.py`
- **Data processing**: Look in `core_utils.py`
- **Scraping modifications**: Look in `web_scraping.py`
- **Analysis changes**: Look in `bert_analysis.py` or `vectorizer_models.py`
- **Visualization updates**: Look in `visualization.py`

## Installation and Setup

The refactored code uses the same dependencies as the original application. Ensure you have all required packages installed:

```bash
pip install streamlit pandas numpy scikit-learn transformers plotly beautifulsoup4 pdfplumber openpyxl torch nltk pyLDAvis networkx requests
```

## Running the Application

### Production Use
```bash
streamlit run main_app.py
```

### Development and Testing
Each module can be imported and tested individually:
```python
from core_utils import clean_text, extract_metadata
from web_scraping import scrape_pfd_reports
from bert_analysis import ThemeAnalyzer
```

## Troubleshooting

### Import Errors
If you encounter import errors:
1. Ensure all files are in the same directory
2. Check that file names match exactly (case-sensitive)
3. Verify no circular imports exist

### Missing Functionality
If certain features don't work:
1. Check that the function exists in the expected module
2. Verify imports are correct in the calling module
3. Compare with the original `app.py` for reference

## Future Development

### Adding New Features
1. **Identify the appropriate module** for your new functionality
2. **Add the function/class** to that module
3. **Update imports** in files that need to use the new feature
4. **Test thoroughly** to ensure no breaking changes

### Best Practices
- Keep functions focused and single-purpose
- Maintain consistent coding style across modules
- Add proper documentation and type hints
- Write unit tests for new functionality

## Support

If you encounter issues with the refactored code:
1. **Compare behavior** with the original `app.py` file
2. **Check the logs** for specific error messages
3. **Test individual modules** to isolate problems
4. **Refer to this documentation** for structural understanding

The refactored structure maintains 100% functional compatibility with the original application while providing a much more maintainable and extensible codebase. 