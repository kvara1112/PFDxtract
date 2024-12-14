import requests
from bs4 import BeautifulSoup
import pandas as pd

class JudiciaryReportScraper:
    def __init__(self, base_url='https://www.judiciary.uk/'):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_search_url(self, search_term='maternity', 
                       report_type='', 
                       order='relevance', 
                       start_date=None, 
                       end_date=None):
        """Construct the search URL with provided parameters"""
        params = {
            's': search_term,
            'pfd_report_type': report_type,
            'post_type': 'pfd',
            'order': order
        }

        # Handle date filtering if provided
        if start_date:
            params.update({
                'after-day': start_date.day,
                'after-month': start_date.month,
                'after-year': start_date.year
            })

        if end_date:
            params.update({
                'before-day': end_date.day,
                'before-month': end_date.month,
                'before-year': end_date.year
            })

        # Convert params to URL query string
        query_string = '&'.join(f'{k}={v}' for k, v in params.items())
        return f'{self.base_url}?{query_string}'

    def scrape_report_links(self, url):
        """Scrape report links from the search results page"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Target the specific selector for report links
            report_links = soup.select('h3.entry-title a')
            
            # Extract link details
            report_details = []
            for link in report_links:
                report_details.append({
                    'title': link.text.strip(),
                    'url': link['href']
                })
            
            return report_details
        
        except Exception as e:
            print(f"Error scraping report links: {e}")
            return []

    def extract_report_text(self, url):
        """Extract full text from a report page"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content - adjust selector as needed
            content = soup.select_one('div.entry-content')
            
            if content:
                return content.get_text(strip=True)
            return ""
        
        except Exception as e:
            print(f"Error extracting text from {url}: {e}")
            return ""
