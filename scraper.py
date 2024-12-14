# scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd

class WebScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_report_links(self, selector):
        """
        Scrape report links from the website
        
        :param selector: CSS selector to find report links
        :return: List of report URLs
        """
        try:
            response = requests.get(self.base_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            links = [a['href'] for a in soup.select(selector) if a.has_attr('href')]
            return links
        except Exception as e:
            print(f"Error scraping links: {e}")
            return []

    def extract_report_text(self, url):
        """
        Extract text from a specific report URL
        
        :param url: URL of the report
        :return: Extracted text
        """
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Modify this based on the specific website's HTML structure
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
            return text
        except Exception as e:
            print(f"Error extracting text from {url}: {e}")
            return ""
