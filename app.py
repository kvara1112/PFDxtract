import streamlit as st
from utils.scraper import JudiciaryReportScraper
from utils.topic_model import TopicModeler

def main():
    st.title("Judiciary UK Report Analyzer")
    
    # Sidebar for configuration
    st.sidebar.header("Scraping Configuration")
    search_term = st.sidebar.text_input("Search Term", "maternity")
    
    # Date range selection
    start_date = st.sidebar.date_input("Start Date", None)
    end_date = st.sidebar.date_input("End Date", None)
    
    # Order selection
    order = st.sidebar.selectbox(
        "Sort Order", 
        ["relevance", "date"]
    )
    
    # Number of topics
    num_topics = st.sidebar.slider("Number of Topics", 2, 10, 5)
    
    # Scrape button
    if st.sidebar.button("Analyze Reports"):
        with st.spinner("Scraping and Analyzing Reports..."):
            # Initialize scraper
            scraper = JudiciaryReportScraper()
            
            # Construct search URL
            search_url = scraper.get_search_url(
                search_term=search_term, 
                order=order,
                start_date=start_date,
                end_date=end_date
            )
            
            # Scrape report links
            report_links = scraper.scrape_report_links(search_url)
            
            if not report_links:
                st.error("No reports found!")
                return
            
            # Extract texts
            texts = [scraper.extract_report_text(link['url']) for link in report_links]
            
            # Perform topic modeling
            modeler = TopicModeler()
            
            # Scikit-learn LDA
            st.subheader("Scikit-learn LDA Topic Model")
            lda_output, topics = modeler.lda_sklearn(texts, num_topics)
            
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx + 1}: {', '.join(topic)}")
            
            # Display report links
            st.subheader(f"Found {len(report_links)} Reports")
            for link in report_links:
                st.write(f"[{link['title']}]({link['url']})")

if __name__ == "__main__":
    main()
