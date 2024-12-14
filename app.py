import streamlit as st
import requests
from bs4 import BeautifulSoup 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import re

st.title("UK Judiciary Maternity Reports Analysis")

def scrape_reports():
    url = "https://www.judiciary.uk/?s=maternity&pfd_report_type=&post_type=pfd&order=relevance"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', class_='search-result')
        
        reports = []
        for article in articles:
            title = article.find('h2').text.strip() if article.find('h2') else ''
            link = article.find('a')['href'] if article.find('a') else ''
            date = article.find('time')['datetime'] if article.find('time') else ''
            
            if link:
                report_response = requests.get(link, headers=headers)
                report_soup = BeautifulSoup(report_response.text, 'html.parser')
                content = report_soup.find('div', class_='content')
                text = content.get_text(strip=True) if content else ''
            else:
                text = ''
                
            reports.append({
                'title': title,
                'date': date,
                'url': link,
                'content': text
            })
        
        return pd.DataFrame(reports)
    except Exception as e:
        st.error(f"Error scraping data: {str(e)}")
        return None

if 'reports_df' not in st.session_state:
    st.session_state.reports_df = None

if st.button("Scrape Reports"):
    st.session_state.reports_df = scrape_reports()

if st.session_state.reports_df is not None:
    df = st.session_state.reports_df
    st.write(f"Found {len(df)} reports")
    st.dataframe(df[['title', 'date', 'url']])
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df['content'])
    
    num_topics = st.slider("Number of Topics", 2, 10, 5)
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    doc_topics = lda.fit_transform(doc_term_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
        st.write(f"Topic {idx + 1}: {', '.join(top_words)}")
    
    topic_dist = pd.DataFrame(doc_topics).mean()
    fig = px.bar(x=[f"Topic {i+1}" for i in range(num_topics)], y=topic_dist)
    st.plotly_chart(fig)
