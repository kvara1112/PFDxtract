import streamlit as st
import requests
import bs4
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px

# Page config
st.set_page_config(page_title="Judiciary Maternity Reports", layout="wide")

# Title
st.title("UK Judiciary Maternity Reports Analysis")
st.write("Analyzes maternity-related reports from the UK judiciary website")

# Initialize session state
if 'reports_df' not in st.session_state:
    st.session_state.reports_df = None

def scrape_reports():
    url = "https://www.judiciary.uk/?s=maternity&pfd_report_type=&post_type=pfd&order=relevance"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        # Scrape main page
        response = requests.get(url, headers=headers)
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article', class_='search-result')
        
        reports = []
        progress_bar = st.progress(0)
        
        # Process each article
        for i, article in enumerate(articles):
            title = article.find('h2').text.strip() if article.find('h2') else ''
            link = article.find('a')['href'] if article.find('a') else ''
            date = article.find('time')['datetime'] if article.find('time') else ''
            
            # Get full report content
            if link:
                report_response = requests.get(link, headers=headers)
                report_soup = bs4.BeautifulSoup(report_response.text, 'html.parser')
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
            
            # Update progress
            progress_bar.progress((i + 1) / len(articles))
            
        return pd.DataFrame(reports)
    
    except Exception as e:
        st.error(f"Error scraping data: {str(e)}")
        return None

def analyze_topics(df, num_topics=5):
    # Prepare text data
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2
    )
    
    # Create document-term matrix
    doc_term_matrix = vectorizer.fit_transform(df['content'])
    
    # Create and fit LDA model
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )
    
    doc_topics = lda.fit_transform(doc_term_matrix)
    
    # Get top terms for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_terms_idx = topic.argsort()[:-10-1:-1]
        top_terms = [feature_names[i] for i in top_terms_idx]
        topics.append({
            'Topic': f'Topic {topic_idx + 1}',
            'Terms': ', '.join(top_terms)
        })
    
    return pd.DataFrame(topics), doc_topics

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Scrape New Reports"):
        with st.spinner("Scraping reports..."):
            st.session_state.reports_df = scrape_reports()
            if st.session_state.reports_df is not None:
                st.success("Reports scraped successfully!")

with col2:
    num_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5)

# If we have data, analyze and display it
if st.session_state.reports_df is not None:
    df = st.session_state.reports_df
    
    # Display basic stats
    st.subheader("Dataset Overview")
    st.write(f"Total Reports: {len(df)}")
    st.write(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    
    # Perform topic analysis
    topics_df, doc_topics = analyze_topics(df, num_topics)
    
    # Display topics
    st.subheader("Discovered Topics")
    st.dataframe(topics_df)
    
    # Create topic distribution visualization
    topic_dist = pd.DataFrame(doc_topics).mean()
    fig = px.bar(
        x=[f"Topic {i+1}" for i in range(num_topics)],
        y=topic_dist,
        title="Topic Distribution Across Documents"
    )
    st.plotly_chart(fig)
    
    # Display reports table
    st.subheader("Reports")
    st.dataframe(
        df[['title', 'date', 'url']],
        use_container_width=True
    )
    
    # Add download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Reports CSV",
        csv,
        "judiciary_reports.csv",
        "text/csv",
        key='download-csv'
    )
