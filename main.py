import streamlit as st
from scraper import WebScraper
from topic_model import TopicModeler

def main():
    st.title("Web Report Topic Modeling Tool")
    
    # Sidebar for configuration
    st.sidebar.header("Scraping Configuration")
    base_url = st.sidebar.text_input("Website URL", "https://example.com/reports")
    link_selector = st.sidebar.text_input("CSS Selector for Report Links", "a.report-link")
    num_topics = st.sidebar.slider("Number of Topics", 2, 10, 5)
    
    # Scrape button
    if st.sidebar.button("Scrape Reports"):
        with st.spinner("Scraping reports..."):
            scraper = WebScraper(base_url)
            report_links = scraper.get_report_links(link_selector)
            
            if not report_links:
                st.error("No report links found!")
                return
            
            # Extract texts
            texts = [scraper.extract_report_text(link) for link in report_links]
            
            # Perform topic modeling
            modeler = TopicModeler()
            
            # Scikit-learn LDA
            st.subheader("Scikit-learn LDA Topic Model")
            lda_output, topics = modeler.lda_sklearn(texts, num_topics)
            
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx + 1}: {', '.join(topic)}")
            
            # Gensim LDA (optional)
            st.subheader("Gensim LDA Topic Model")
            gensim_model, dictionary = modeler.lda_gensim(texts, num_topics)
            
            # Display Gensim topics
            for idx, topic in gensim_model.print_topics():
                st.write(f"Topic {idx}: {topic}")

if __name__ == "__main__":
    main()
