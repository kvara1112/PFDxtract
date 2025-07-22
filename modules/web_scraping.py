import logging
import os
import re
import time
import urllib3
import requests
import pdfplumber
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st
import pandas as pd
import logging  
import os
import re
import time
import urllib3
import requests

from .core_utils import (
    clean_text, 
    extract_metadata, 
    process_scraped_data
)

# Global headers for all requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Referer": "https://judiciary.uk/",
}



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
    base_url: str,
    keyword: Optional[str] = None,
    category: Optional[str] = None,
    category_slug: Optional[str] = None,
    page: Optional[int] = None,
    after_date: Optional[str] = None,
    before_date: Optional[str] = None,
) -> str:
    """Constructs proper search URL with pagination and date filters"""
    # Start with base search URL
    url = f"{base_url}?s=&post_type=pfd"

    # Add category filter
    if category and category_slug:
        url += f"&pfd_report_type={category_slug}"

    # Add keyword search
    if keyword:
        url = f"{base_url}?s={keyword}&post_type=pfd"
        if category and category_slug:
            url += f"&pfd_report_type={category_slug}"

    # Add pagination
    if page and page > 1:
        url += f"&paged={page}"  # Changed from &page= to &paged= for proper pagination
        
    # Add date filters
    if after_date:
        # Parse the date parts
        parts = after_date.split("-")
        if len(parts) == 3:
            day, month, year = parts
            url += f"&after-day={day}&after-month={month}&after-year={year}"
            
    if before_date:
        # Parse the date parts
        parts = before_date.split("-")
        if len(parts) == 3:
            day, month, year = parts
            url += f"&before-day={day}&before-month={month}&before-year={year}"

    return url


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

def download_pdf(pdf_url: str, save_dir: str = "outputs") -> Optional[str]:
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

def scrape_page(url: str) -> List[Dict]:
    """Scrape a single page with improved PDF handling"""
    reports = []
    try:
        response = make_request(url)
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        results_list = soup.find("ul", class_="search__list")

        if not results_list:
            logging.warning(f"No results list found on page: {url}")
            return []

        cards = results_list.find_all("div", class_="card")

        for card in cards:
            try:
                title_elem = card.find("h3", class_="card__title")
                if not title_elem:
                    continue

                title_link = title_elem.find("a")
                if not title_link:
                    continue

                title = clean_text(title_link.text)
                card_url = title_link["href"]

                if not card_url.startswith(("http://", "https://")):
                    card_url = f"https://www.judiciary.uk{card_url}"

                logging.info(f"Processing report: {title}")
                content_data = get_report_content(card_url)

                if content_data:
                    report = {
                        "Title": title,
                        "URL": card_url,
                        "Content": content_data["content"],
                    }

                    # Add PDF details with type classification
                    for i, (name, content, path, pdf_type) in enumerate(
                        zip(
                            content_data["pdf_names"],
                            content_data["pdf_contents"],
                            content_data["pdf_paths"],
                            content_data["pdf_types"],
                        ),
                        1,
                    ):
                        report[f"PDF_{i}_Name"] = name
                        report[f"PDF_{i}_Content"] = content
                        report[f"PDF_{i}_Path"] = path
                        report[f"PDF_{i}_Type"] = pdf_type

                    reports.append(report)
                    logging.info(f"Successfully processed: {title}")

            except Exception as e:
                logging.error(f"Error processing card: {e}")
                continue

        return reports

    except Exception as e:
        logging.error(f"Error fetching page {url}: {e}")
        return []

def get_report_content(url: str) -> Optional[Dict]:
    """Get full content from report page with improved PDF and response handling"""
    try:
        logging.info(f"Fetching content from: {url}")
        response = make_request(url)
        if not response:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_="flow") or soup.find(
            "article", class_="single__post"
        )

        if not content:
            logging.warning(f"No content found at {url}")
            return None

        # Extract main report content
        paragraphs = content.find_all(["p", "table"])
        webpage_text = "\n\n".join(
            p.get_text(strip=True, separator=" ") for p in paragraphs
        )

        pdf_contents = []
        pdf_paths = []
        pdf_names = []
        pdf_types = []  # Track if PDF is main report or response

        # Find all PDF links with improved classification
        pdf_links = soup.find_all("a", href=re.compile(r"\.pdf$"))

        for pdf_link in pdf_links:
            pdf_url = pdf_link["href"]
            pdf_text = pdf_link.get_text(strip=True).lower()

            # Determine PDF type
            is_response = any(
                word in pdf_text.lower() for word in ["response", "reply"]
            )
            pdf_type = "response" if is_response else "report"

            if not pdf_url.startswith(("http://", "https://")):
                pdf_url = (
                    f"https://www.judiciary.uk{pdf_url}"
                    if not pdf_url.startswith("/")
                    else f"https://www.judiciary.uk/{pdf_url}"
                )

            pdf_path, pdf_name = save_pdf(pdf_url)

            if pdf_path:
                pdf_content = extract_pdf_content(pdf_path)
                pdf_contents.append(pdf_content)
                pdf_paths.append(pdf_path)
                pdf_names.append(pdf_name)
                pdf_types.append(pdf_type)

        return {
            "content": clean_text(webpage_text),
            "pdf_contents": pdf_contents,
            "pdf_paths": pdf_paths,
            "pdf_names": pdf_names,
            "pdf_types": pdf_types,
        }

    except Exception as e:
        logging.error(f"Error getting report content: {e}")
        return None
    
def save_pdf(
    pdf_url: str, base_dir: str = "outputs"
) -> Tuple[Optional[str], Optional[str]]:
    """Download and save PDF, return local path and filename"""
    try:
        os.makedirs(base_dir, exist_ok=True)

        response = make_request(pdf_url)
        if not response:
            return None, None

        filename = os.path.basename(pdf_url)
        filename = re.sub(r"[^\w\-_\. ]", "_", filename)
        local_path = os.path.join(base_dir, filename)

        with open(local_path, "wb") as f:
            f.write(response.content)

        return local_path, filename

    except Exception as e:
        logging.error(f"Error saving PDF {pdf_url}: {e}")
        return None, None
    
def extract_pdf_content(pdf_path: str, chunk_size: int = 10) -> str:
    """Extract text from PDF file with memory management"""
    try:
        filename = os.path.basename(pdf_path)
        text_chunks = []

        with pdfplumber.open(pdf_path) as pdf:
            for i in range(0, len(pdf.pages), chunk_size):
                chunk = pdf.pages[i : i + chunk_size]
                chunk_text = "\n\n".join([page.extract_text() or "" for page in chunk])
                text_chunks.append(chunk_text)

        full_content = f"PDF FILENAME: {filename}\n\n{''.join(text_chunks)}"
        return clean_text(full_content)

    except Exception as e:
        logging.error(f"Error extracting PDF text from {pdf_path}: {e}")
        return ""

def get_total_pages(url: str) -> Tuple[int, int]:
    """
    Get total number of pages and total results count

    Returns:
        Tuple[int, int]: (total_pages, total_results)
    """
    try:
        response = make_request(url)
        if not response:
            logging.error(f"No response from URL: {url}")
            return 0, 0

        soup = BeautifulSoup(response.text, "html.parser")

        # First check for total results count
        total_results = 0
        results_header = soup.find("div", class_="search__header")
        if results_header:
            results_text = results_header.get_text()
            match = re.search(r"found (\d+) results?", results_text, re.IGNORECASE)
            if match:
                total_results = int(match.group(1))
                total_pages = (total_results + 9) // 10  # 10 results per page
                return total_pages, total_results

        # If no results header, check pagination
        pagination = soup.find("nav", class_="navigation pagination")
        if pagination:
            page_numbers = pagination.find_all("a", class_="page-numbers")
            numbers = [
                int(p.text.strip()) for p in page_numbers if p.text.strip().isdigit()
            ]
            if numbers:
                return max(numbers), len(numbers) * 10  # Approximate result count

        # If no pagination but results exist
        results = soup.find("ul", class_="search__list")
        if results and results.find_all("div", class_="card"):
            cards = results.find_all("div", class_="card")
            return 1, len(cards)

        return 0, 0

    except Exception as e:
        logging.error(f"Error in get_total_pages: {str(e)}")
        return 0, 0

def make_request(
    url: str, retries: int = 3, delay: int = 2
) -> Optional[requests.Response]:
    """Make HTTP request with retries and delay"""
    for attempt in range(retries):
        try:
            time.sleep(delay)
            response = requests.get(url, headers=HEADERS, verify=False, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"Request failed: {str(e)}")
                raise e
            time.sleep(delay * (attempt + 1))
    return None



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
    keyword: Optional[str] = None,
    category: Optional[str] = None,
    order: str = "relevance",
    start_page: int = 1,
    end_page: Optional[int] = None,
    auto_save_batches: bool = True,
    batch_size: int = 5,
    after_date: Optional[str] = None,
    before_date: Optional[str] = None,
) -> List[Dict]:
    """
    Scrape PFD reports with enhanced progress tracking, proper pagination, date filters, and automatic batch saving
    
    Args:
        keyword: Optional keyword to search for
        category: Optional category to filter by
        order: Sort order ("relevance", "desc", "asc")
        start_page: First page to scrape
        end_page: Last page to scrape (None for all pages)
        auto_save_batches: Whether to automatically save batches of results
        batch_size: Number of pages per batch
        after_date: Date filter for reports published after (format: day-month-year)
        before_date: Date filter for reports published before (format: day-month-year)
        
    Returns:
        List of report dictionaries
    """
    all_reports = []
    base_url = "https://www.judiciary.uk/"
    batch_number = 1

    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        report_count_text = st.empty()
        batch_status = st.empty()

        # Validate and prepare category
        category_slug = None
        if category:
            category_slug = (
                category.lower()
                .replace(" ", "-")
                .replace("&", "and")
                .replace("--", "-")
                .strip("-")
            )
            logging.info(f"Using category: {category}, slug: {category_slug}")

        # Construct initial search URL with date filters
        base_search_url = construct_search_url(
            base_url=base_url,
            keyword=keyword,
            category=category,
            category_slug=category_slug,
            after_date=after_date,
            before_date=before_date,
        )

        st.info(f"Searching at: {base_search_url}")

        # Get total pages and results count
        total_pages, total_results = get_total_pages(base_search_url)

        if total_results == 0:
            st.warning("No results found matching your search criteria")
            return []

        st.info(f"Found {total_results} matching reports across {total_pages} pages")

        # Apply page range limits
        start_page = max(1, start_page)  # Ensure start_page is at least 1
        if end_page is None:
            end_page = total_pages
        else:
            end_page = min(
                end_page, total_pages
            )  # Ensure end_page doesn't exceed total_pages

        if start_page > end_page:
            st.warning(f"Invalid page range: {start_page} to {end_page}")
            return []

        st.info(f"Scraping pages {start_page} to {end_page}")
        
        # Variables for batch processing
        batch_reports = []
        current_batch_start = start_page
        batch_end = min(start_page + batch_size - 1, end_page)

        # Process each page in the specified range
        for current_page in range(start_page, end_page + 1):
            try:
                # Check if scraping should be stopped
                if (
                    hasattr(st.session_state, "stop_scraping")
                    and st.session_state.stop_scraping
                ):
                    # Save the current batch before stopping
                    if auto_save_batches and batch_reports:
                        save_batch(
                            batch_reports, 
                            batch_number, 
                            keyword, 
                            category, 
                            current_batch_start, 
                            current_page - 1
                        )
                    st.warning("Scraping stopped by user")
                    break

                # Update progress
                progress = (current_page - start_page) / (end_page - start_page + 1)
                progress_bar.progress(progress)
                status_text.text(
                    f"Processing page {current_page} of {end_page} (out of {total_pages} total pages)"
                )

                # Construct current page URL with date filters
                page_url = construct_search_url(
                    base_url=base_url,
                    keyword=keyword,
                    category=category,
                    category_slug=category_slug,
                    page=current_page,
                    after_date=after_date,
                    before_date=before_date,
                )

                # Scrape current page
                page_reports = scrape_page(page_url)

                if page_reports:
                    # Deduplicate based on title and URL
                    existing_reports = {(r["Title"], r["URL"]) for r in all_reports}
                    existing_batch_reports = {(r["Title"], r["URL"]) for r in batch_reports}
                    
                    new_reports = [
                        r
                        for r in page_reports
                        if (r["Title"], r["URL"]) not in existing_reports 
                        and (r["Title"], r["URL"]) not in existing_batch_reports
                    ]

                    # Add to both all_reports and batch_reports
                    all_reports.extend(new_reports)
                    batch_reports.extend(new_reports)
                    
                    report_count_text.text(
                        f"Retrieved {len(all_reports)} unique reports so far..."
                    )

                # Check if we've reached the end of a batch
                if auto_save_batches and (current_page == batch_end or current_page == end_page):
                    if batch_reports:
                        # Automatically save the batch
                        saved_file = save_batch(
                            batch_reports, 
                            batch_number, 
                            keyword, 
                            category, 
                            current_batch_start, 
                            current_page
                        )
                        batch_status.success(
                            f"Saved batch #{batch_number} (pages {current_batch_start}-{current_page}) to {saved_file}"
                        )
                        
                        # Reset for next batch
                        batch_reports = []
                        batch_number += 1
                        current_batch_start = current_page + 1
                        batch_end = min(current_batch_start + batch_size - 1, end_page)
                
                # Add delay between pages
                time.sleep(2)

            except Exception as e:
                logging.error(f"Error processing page {current_page}: {e}")
                st.warning(
                    f"Error on page {current_page}. Continuing with next page..."
                )
                continue

        # Sort results if specified
        if order != "relevance":
            all_reports = sort_reports(all_reports, order)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        report_count_text.empty()

        if all_reports:
            st.success(f"Successfully scraped {len(all_reports)} unique reports")
            
            # Final report
            if auto_save_batches:
                st.info(f"Reports were automatically saved in {batch_number} batches")
        else:
            st.warning("No reports were successfully retrieved")

        return all_reports

    except Exception as e:
        logging.error(f"Error in scrape_pfd_reports: {e}")
        st.error(f"An error occurred while scraping reports: {e}")
        # Save any unsaved reports if an error occurs
        if auto_save_batches and batch_reports:
            save_batch(
                batch_reports, 
                batch_number, 
                keyword, 
                category, 
                current_batch_start, 
                "error"
            )
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
        pdf_dir = "outputs"
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
    
    

def sort_reports(reports: List[Dict], order: str) -> List[Dict]:
    """Sort reports based on specified order"""
    if order == "date_desc":
        return sorted(
            reports,
            key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"),
            reverse=True,
        )
    elif order == "date_asc":
        return sorted(reports, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d"))
    return reports

def save_batch(
    reports: List[Dict], 
    batch_number: int, 
    keyword: Optional[str], 
    category: Optional[str], 
    start_page: int, 
    end_page: Union[int, str]
) -> str:
    """
    Save a batch of reports to Excel file with appropriate naming
    
    Args:
        reports: List of report dictionaries to save
        batch_number: Current batch number
        keyword: Search keyword used (for filename)
        category: Category used (for filename)
        start_page: Starting page of this batch
        end_page: Ending page of this batch (or "error" if saving due to error)
        
    Returns:
        Filename of the saved file
    """
    if not reports:
        return ""
    
    # Process the data
    df = pd.DataFrame(reports)
    df = process_scraped_data(df)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive filename parts
    keyword_part = f"kw_{keyword.replace(' ', '_')}" if keyword else "no_keyword"
    category_part = f"cat_{category.replace(' ', '_')}" if category else "no_category"
    page_part = f"pages_{start_page}_to_{end_page}"
    
    # Generate filename
    filename = f"pfd_reports_scraped_batch{batch_number}_{keyword_part}_{category_part}_{page_part}_{timestamp}.xlsx"
    
    # Ensure filename is valid (remove any problematic characters)
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    
    # Create directory if it doesn't exist
    os.makedirs("scraped_reports", exist_ok=True)
    file_path = os.path.join("scraped_reports", filename)
    
    # Save to Excel
    df.to_excel(file_path, index=False, engine="openpyxl")
    
    logging.info(f"Saved batch {batch_number} to {file_path}")
    return filename
