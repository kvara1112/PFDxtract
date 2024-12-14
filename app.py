import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import time

# Set the page title
st.title("UK Judiciary Maternity Reports Analysis")

# Function to scrape maternity reports from the judiciary website
def scrape_reports():
    url = "https://www.judiciary.uk/?s=maternity&pfd_report_type=&post_type=pfd&order=relevance"
    headers = {'User-Agent': 'Mozilla/5.0'}

    reports = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            st.error(f"Failed to fetch the website. Status code: {response.status_code}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', class_='search-result')

        if not articles:
            st.warning("No articles found. The website structure might have changed.")
            return pd.DataFrame()

        for article in articles:
            title = article.find('h2').text.strip() if article.find('h2') else 'No Title'
            link = article.find('a')['href'] if article.find('a') else 'No Link'
            date = article.find('time')['datetime'] if article.find('time') else 'No Date'
            
            # Fetch content from individual report links
            content = ""
            try:
                if link != 'No Link':
                    report_response = requests.get(link, headers=headers, timeout=10)
                    report_soup = BeautifulSoup(report_response.text, 'html.parser')
                    content_div = report_soup.find('div', class_='content')
                    content = content_div.get_text(strip=True) if content_div else 'No Content'
                    time.sleep(1)  # Delay to avoid overloading the server
            except Exception as e:
                st.warning(f"Failed to fetch report content from {link}. Error: {str(e)}")
            
            reports.append({
                'title': title,
                'date': date,
                'url': link,
                'content': content
            })
        return pd.DataFrame(reports)
    except Exception as e:
        st.error(f"Error occurred while scraping data: {str(e)}")
        return pd.DataFrame()

# Initialize session state for storing reports
if 'reports_df' not in st.session_state:
    st.session_state.reports_df = None

# Button to scrape reports
if st.button("Scrape Reports"):
    st.session_state.reports_df = scrape_reports()

# Display reports and perform analysis if data is available
if st.session_state.reports_df is not None:
    df = st.session_state.reports_df
    st.write(f"Found {len(df)} reports.")
    st.dataframe(df[['title', 'date', 'url']])

    # Ensure there is content for analysis
    if df['content'].isnull().all() or df['content'].str.strip().eq("").all():
        st.error("No valid content available for analysis.")
    else:
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(df['content'].fillna(''))

        # Topic Modeling with LDA
        num_topics = st.slider("Select Number of Topics", 2, 10, 5)
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        doc_topics = lda.fit_transform(doc_term_matrix)

        # Display Top Words for Each Topic
        feature_names = vectorizer.get_feature_names_out()
        for idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            st.write(f"**Topic {idx + 1}:** {', '.join(top_words)}")

        # Visualize Topic Distribution
        topic_dist = pd.DataFrame(doc_topics).mean(axis=0)
        fig = px.bar(x=[f"Topic {i+1}" for i in range(num_topics)], y=topic_dist,
                     labels={'x': "Topics", 'y': "Proportion"},
                     title="Topic Distribution")
        st.plotly_chart(fig)
