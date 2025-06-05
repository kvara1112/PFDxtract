import logging
import os
import re
import time
import urllib3
import requests
import zipfile
import io
import pdfplumber
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
import streamlit as st
import pandas as pd

# Import our core utilities
from core_utils import clean_text, extract_metadata, process_scraped_data

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)

def get_pfd_categories() -> List[str]:
    """Return list of available PFD report categories"""
    return [
        "",  # Empty option for all categories
        "Accident at Work and Health and Safety related deaths",
        "Alcohol drug and medication related deaths",
        "Care Home Health related deaths",
        "Child Death from 2015",
        "Community health care and emergency services related deaths",
        "Emergency services related deaths 2019 onwards",
        "Hospital Death Clinical Procedures and medical management related deaths",
        "Mental Health related deaths",
        "Other related deaths",
        "Police related deaths",
        "Product related deaths",
        "Railway related deaths",
        "Road Highways Safety related deaths",
        "Service Personnel related deaths",
        "State Custody related deaths",
        "Suicide from 2015",
        "Wales prevention of future deaths reports 2019 onwards",
    ]

def get_sort_options() -> List[str]:
    """Return list of available sort options"""
    return ["date_desc", "date_asc"]

def construct_search_url(
    base_url: str = "https://www.judiciary.uk/prevention-of-future-death-reports/",
    keyword: str = "",
    category: str = "",
    category_slug: str = "",
    after_date: str = "",
    before_date: str = "",
    order: str = "date_desc"
) -> str:
    """
    Construct search URL with filters
    
    Args:
        base_url: Base URL for the search
        keyword: Search keyword
        category: Category name
        category_slug: URL-friendly category slug
        after_date: Date filter (after this date)
        before_date: Date filter (before this date)
        order: Sort order
    
    Returns:
        Constructed URL string
    """
    url = base_url
    params = []
    
    if keyword:
        params.append(f"keyword={keyword}")
    
    if category and category_slug:
        params.append(f"pfd-category={category_slug}")
    
    if after_date:
        params.append(f"after={after_date}")
    
    if before_date:
        params.append(f"before={before_date}")
    
    if order:
        params.append(f"order={order}")
    
    if params:
        url += "?" + "&".join(params)
    
    return url

def get_total_pages(url: str) -> Tuple[int, int]:
    """
    Get total number of pages and results from search URL
    
    Args:
        url: Search URL
        
    Returns:
        Tuple of (total_pages, total_results)
    """
    try:
        response = requests.get(url, verify=False, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Look for pagination info
        pagination = soup.find("div", class_="pagination")
        if pagination:
            # Find the last page number
            page_links = pagination.find_all("a", class_="page-numbers")
            if page_links:
                # Get all numeric page numbers
                page_numbers = []
                for link in page_links:
                    try:
                        page_num = int(link.get_text().strip())
                        page_numbers.append(page_num)
                    except ValueError:
                        continue
                
                if page_numbers:
                    total_pages = max(page_numbers)
                else:
                    total_pages = 1
            else:
                total_pages = 1
        else:
            total_pages = 1
        
        # Look for results count
        results_info = soup.find("div", class_="results-info")
        total_results = 0
        
        if results_info:
            # Extract number from text like "Showing 1-10 of 234 results"
            text = results_info.get_text()
            match = re.search(r"of\s+(\d+)\s+results", text)
            if match:
                total_results = int(match.group(1))
        
        # If no results info found, count the reports on current page
        if total_results == 0:
            reports = soup.find_all("article", class_="report")
            total_results = len(reports) * total_pages  # Rough estimate
        
        return total_pages, total_results
        
    except Exception as e:
        logging.error(f"Error getting total pages: {e}")
        return 0, 0

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logging.warning(f"Error extracting text from page: {e}")
                    continue
            
            full_text = "\n".join(text_parts)
            return clean_text(full_text)
            
    except Exception as e:
        logging.error(f"Error extracting PDF text from {pdf_path}: {e}")
        return ""

def download_pdf(pdf_url: str, save_dir: str = "pdfs") -> Optional[str]:
    """
    Download PDF from URL and return local path
    
    Args:
        pdf_url: URL of PDF to download
        save_dir: Directory to save PDF
        
    Returns:
        Local file path if successful, None otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename from URL
        filename = os.path.basename(pdf_url.split("?")[0])
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        
        # Create unique filename to avoid conflicts
        timestamp = str(int(time.time() * 1000))
        base_name, ext = os.path.splitext(filename)
        unique_filename = f"{base_name}_{timestamp}{ext}"
        
        file_path = os.path.join(save_dir, unique_filename)
        
        # Download PDF
        response = requests.get(pdf_url, verify=False, timeout=30)
        response.raise_for_status()
        
        # Save to file
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        logging.info(f"Downloaded PDF: {unique_filename}")
        return file_path
        
    except Exception as e:
        logging.error(f"Error downloading PDF from {pdf_url}: {e}")
        return None

def extract_report_data(soup: BeautifulSoup, report_element) -> Dict:
    """
    Extract data from a single report element
    
    Args:
        soup: BeautifulSoup object of the page
        report_element: HTML element containing report data
        
    Returns:
        Dictionary containing extracted report data
    """
    try:
        report_data = {
            "Title": "",
            "URL": "",
            "Content": "",
            "Record ID": "",
            "date_of_report": None,
            "ref": None,
            "deceased_name": None,
            "coroner_name": None,
            "coroner_area": None,
            "categories": [],
        }
        
        # Extract title and URL
        title_link = report_element.find("h2", class_="entry-title")
        if title_link:
            link = title_link.find("a")
            if link:
                report_data["Title"] = clean_text(link.get_text())
                report_data["URL"] = link.get("href", "")
                
                # Extract Record ID from URL
                url_match = re.search(r"record-id-(\d+)", report_data["URL"])
                if url_match:
                    report_data["Record ID"] = url_match.group(1)
        
        # Extract content/summary
        content_div = report_element.find("div", class_="entry-content")
        if content_div:
            # Remove any nested elements we don't want
            for unwanted in content_div.find_all(["script", "style"]):
                unwanted.decompose()
            
            report_data["Content"] = clean_text(content_div.get_text())
        
        # Extract metadata from content
        if report_data["Content"]:
            metadata = extract_metadata(report_data["Content"])
            report_data.update(metadata)
        
        # Look for PDF links
        pdf_links = report_element.find_all("a", href=True)
        pdf_count = 0
        
        for link in pdf_links:
            href = link.get("href", "")
            if href.lower().endswith(".pdf"):
                pdf_count += 1
                
                # Store PDF information
                pdf_name = link.get_text().strip() or f"PDF_{pdf_count}"
                
                report_data[f"PDF_{pdf_count}_Name"] = clean_text(pdf_name)
                report_data[f"PDF_{pdf_count}_URL"] = href
                
                # Determine PDF type based on filename/text
                pdf_name_lower = pdf_name.lower()
                if "response" in pdf_name_lower or "reply" in pdf_name_lower:
                    report_data[f"PDF_{pdf_count}_Type"] = "Response"
                else:
                    report_data[f"PDF_{pdf_count}_Type"] = "Report"
                
                # Download and extract PDF content
                try:
                    pdf_path = download_pdf(href)
                    if pdf_path:
                        report_data[f"PDF_{pdf_count}_Path"] = pdf_path
                        
                        # Extract text content
                        pdf_text = extract_pdf_text(pdf_path)
                        if pdf_text:
                            report_data[f"PDF_{pdf_count}_Content"] = pdf_text
                            
                            # If main content is empty, use PDF content
                            if not report_data["Content"] and pdf_text:
                                report_data["Content"] = pdf_text
                                
                                # Re-extract metadata from PDF content
                                pdf_metadata = extract_metadata(pdf_text)
                                for key, value in pdf_metadata.items():
                                    if not report_data.get(key) and value:
                                        report_data[key] = value
                        
                except Exception as e:
                    logging.warning(f"Error processing PDF {href}: {e}")
                    continue
        
        # Extract categories
        category_elements = report_element.find_all("span", class_="category")
        categories = []
        for cat_elem in category_elements:
            cat_text = clean_text(cat_elem.get_text())
            if cat_text:
                categories.append(cat_text)
        
        if categories:
            report_data["categories"] = categories
        
        return report_data
        
    except Exception as e:
        logging.error(f"Error extracting report data: {e}")
        return {}

def scrape_page(url: str, page_num: int = 1) -> List[Dict]:
    """
    Scrape a single page of reports
    
    Args:
        url: URL to scrape
        page_num: Page number for logging
        
    Returns:
        List of report dictionaries
    """
    try:
        # Add page parameter to URL if not page 1
        if page_num > 1:
            separator = "&" if "?" in url else "?"
            page_url = f"{url}{separator}page={page_num}"
        else:
            page_url = url
        
        logging.info(f"Scraping page {page_num}: {page_url}")
        
        # Make request
        response = requests.get(page_url, verify=False, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find all report articles
        reports = soup.find_all("article", class_="report")
        
        if not reports:
            # Try alternative selectors
            reports = soup.find_all("div", class_="report-item")
            
        if not reports:
            # Try generic article tags
            reports = soup.find_all("article")
        
        page_reports = []
        
        for i, report in enumerate(reports):
            try:
                # Check for stop signal
                if st.session_state.get("stop_scraping", False):
                    logging.info("Stopping scraping as requested by user")
                    break
                
                report_data = extract_report_data(soup, report)
                
                if report_data and report_data.get("Title"):
                    page_reports.append(report_data)
                    logging.info(f"Extracted report {i+1}: {report_data['Title'][:50]}...")
                
                # Small delay between reports
                time.sleep(0.5)
                
            except Exception as e:
                logging.error(f"Error processing report {i+1} on page {page_num}: {e}")
                continue
        
        logging.info(f"Page {page_num}: Extracted {len(page_reports)} reports")
        return page_reports
        
    except Exception as e:
        logging.error(f"Error scraping page {page_num}: {e}")
        return []

def save_batch_results(reports: List[Dict], batch_num: int, keyword: str = "") -> str:
    """
    Save batch of reports to file
    
    Args:
        reports: List of report dictionaries
        batch_num: Batch number
        keyword: Search keyword for filename
        
    Returns:
        Filename of saved file
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        keyword_part = f"_{keyword}" if keyword else ""
        filename = f"pfd_reports_scraped_reportID{keyword_part}_batch_{batch_num}_{timestamp}.xlsx"
        
        # Convert to DataFrame and process
        df = pd.DataFrame(reports)
        df = process_scraped_data(df)
        
        # Save to Excel
        df.to_excel(filename, index=False, engine="openpyxl")
        
        logging.info(f"Saved batch {batch_num} to {filename}")
        return filename
        
    except Exception as e:
        logging.error(f"Error saving batch {batch_num}: {e}")
        return ""

def scrape_pfd_reports(
    keyword: str = "",
    category: str = None,
    order: str = "date_desc",
    start_page: int = 1,
    end_page: Optional[int] = None,
    auto_save_batches: bool = True,
    batch_size: int = 5,
    after_date: str = "",
    before_date: str = "",
) -> List[Dict]:
    """
    Main function to scrape PFD reports
    
    Args:
        keyword: Search keyword
        category: Report category
        order: Sort order
        start_page: Starting page number
        end_page: Ending page number (None for all)
        auto_save_batches: Whether to auto-save batches
        batch_size: Number of pages per batch
        after_date: Date filter (after)
        before_date: Date filter (before)
        
    Returns:
        List of all scraped reports
    """
    try:
        # Initialize session state
        if "stop_scraping" not in st.session_state:
            st.session_state.stop_scraping = False
        
        # Convert category to slug
        category_slug = ""
        if category:
            category_slug = (
                category.lower()
                .replace(" ", "-")
                .replace("&", "and")
                .replace("--", "-")
                .strip("-")
            )
        
        # Construct search URL
        base_url = "https://www.judiciary.uk/prevention-of-future-death-reports/"
        search_url = construct_search_url(
            base_url=base_url,
            keyword=keyword,
            category=category,
            category_slug=category_slug,
            after_date=after_date,
            before_date=before_date,
            order=order,
        )
        
        # Get total pages if end_page not specified
        if end_page is None:
            total_pages, total_results = get_total_pages(search_url)
            end_page = total_pages
            st.info(f"Found {total_pages} pages with {total_results} total results")
        
        # Validate page range
        if start_page > end_page:
            st.error("Start page cannot be greater than end page")
            return []
        
        # Initialize tracking variables
        all_reports = []
        batch_reports = []
        batch_num = 1
        saved_files = []
        
        # Progress tracking
        total_pages_to_scrape = end_page - start_page + 1
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Scrape pages
        for page_num in range(start_page, end_page + 1):
            # Check for stop signal
            if st.session_state.get("stop_scraping", False):
                st.warning("Scraping stopped by user request")
                break
            
            # Update progress
            progress = (page_num - start_page) / total_pages_to_scrape
            progress_bar.progress(progress)
            status_text.text(f"Scraping page {page_num} of {end_page}...")
            
            # Scrape page
            page_reports = scrape_page(search_url, page_num)
            
            if page_reports:
                all_reports.extend(page_reports)
                batch_reports.extend(page_reports)
                
                st.success(f"Page {page_num}: Found {len(page_reports)} reports")
            else:
                st.warning(f"Page {page_num}: No reports found")
            
            # Save batch if needed
            if auto_save_batches and len(batch_reports) >= len(page_reports) * batch_size:
                if batch_reports:
                    filename = save_batch_results(batch_reports, batch_num, keyword)
                    if filename:
                        saved_files.append(filename)
                        st.info(f"Saved batch {batch_num}: {filename}")
                    
                    batch_reports = []
                    batch_num += 1
            
            # Delay between pages
            time.sleep(1)
        
        # Save final batch if needed
        if auto_save_batches and batch_reports:
            filename = save_batch_results(batch_reports, batch_num, keyword)
            if filename:
                saved_files.append(filename)
                st.info(f"Saved final batch {batch_num}: {filename}")
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.text("Scraping complete!")
        
        # Summary
        st.success(f"Scraping completed! Found {len(all_reports)} total reports")
        
        if saved_files:
            st.info(f"Saved {len(saved_files)} batch files:")
            for filename in saved_files:
                st.text(f"• {filename}")
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return all_reports
        
    except Exception as e:
        logging.error(f"Error in scraping: {e}")
        st.error(f"Error during scraping: {str(e)}")
        return []

def validate_date_format(date_str: str) -> bool:
    """
    Validate date format (DD-MM-YYYY)
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid format, False otherwise
    """
    try:
        datetime.strptime(date_str, "%d-%m-%Y")
        return True
    except ValueError:
        return False

def cleanup_old_pdfs(max_age_hours: int = 24) -> None:
    """
    Clean up old PDF files to save disk space
    
    Args:
        max_age_hours: Maximum age of files to keep (in hours)
    """
    try:
        pdf_dir = "pdfs"
        if not os.path.exists(pdf_dir):
            return
        
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        deleted_count = 0
        for filename in os.listdir(pdf_dir):
            file_path = os.path.join(pdf_dir, filename)
            
            try:
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        deleted_count += 1
            except Exception as e:
                logging.warning(f"Error deleting file {file_path}: {e}")
                continue
        
        if deleted_count > 0:
            logging.info(f"Cleaned up {deleted_count} old PDF files")
            
    except Exception as e:
        logging.error(f"Error in PDF cleanup: {e}")

def estimate_scraping_time(start_page: int, end_page: int, batch_size: int = 5) -> str:
    """
    Estimate scraping time based on page range
    
    Args:
        start_page: Starting page number
        end_page: Ending page number
        batch_size: Pages per batch
        
    Returns:
        Estimated time string
    """
    try:
        total_pages = end_page - start_page + 1
        
        # Estimate 30-60 seconds per page (including PDF downloads)
        avg_time_per_page = 45  # seconds
        total_seconds = total_pages * avg_time_per_page
        
        # Convert to human readable format
        if total_seconds < 60:
            return f"~{total_seconds} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            return f"~{minutes} minutes"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"~{hours}h {minutes}m"
            
    except Exception:
        return "Unable to estimate" 