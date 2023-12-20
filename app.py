import os
import pandas as pd
from gnews import GNews
from datetime import datetime
from langid import classify
from joblib import Parallel, delayed
import streamlit as st
import xlsxwriter
from io import BytesIO
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from docx import Document
from nltk.tokenize import word_tokenize
import subprocess

@st.cache_resource
def download_en_core_web_sm():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')

# Function to check if the article is in English
def is_english(text):
    lang, _ = classify(text)
    return lang == 'en'

# Function to get full article using newspaper3k and perform NLP
def get_full_article_with_nlp(url):
    article = GNews().get_full_article(url)

    # Perform NLP on the article
    try:
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        print(f"Error performing NLP on the article: {e}")

    return {
        'title': article.title,
        'text': article.text,
        'authors': article.authors,
        'summary': article.summary,
        'keywords': article.keywords,
    }

def process_article(article):
    source = article.get('source', 'N/A')
    if source == 'N/A':
        # Check if the source is still not available and try to extract it from the URL
        url = article.get('url', '')
        if url:
            source_from_url = url.split('/')[2]  # Extract domain from the URL
            source = source_from_url if source_from_url else 'N/A'

    if is_english(article.get('title', '')):
        title = article.get('title', 'N/A')

        # Extract information after " ... - " or " - " in the title
        if ' ... - ' in title:
            title_parts = title.split(' ... - ')
        elif ' - ' in title:
            title_parts = title.split(' - ')
        else:
            title_parts = [title]

        article_source = title_parts[1].strip() if len(title_parts) > 1 else 'N/A'

        result = {
            'Title': article.get('title', 'N/A'),
            'Source': source,
            'Article_Source': article_source,
            'URL': article.get('url', 'N/A'),
        }

        try:
            # Get the full article using newspaper3k and perform NLP
            full_article = get_full_article_with_nlp(article.get('url'))
            result.update(full_article)
        except Exception as e:
            print(f"Error processing article: {e}")

        return result
    else:
        return None

# Streamlit UI
st.title("Keyword-driven News Aggregator.")
st.sidebar.header("Search Parameters")

# User input
query = st.sidebar.text_input("Enter Keywords (separated by space)", 'aws cloud')
start_date = st.sidebar.date_input("Enter Start Date", datetime(2023, 11, 1))
end_date = st.sidebar.date_input("Enter End Date", datetime(2023, 12, 30))

# Set the language filter
GNews.language = 'en'

# Initialize GNews object with query parameters
google_news = GNews(country='US', max_results=100)

# Search button
if st.sidebar.button("Search"):
    with st.spinner("Please wait while scraping the web..."):
        # Set the start and end dates based on UI input
        google_news.start_date = start_date
        google_news.end_date = end_date

        # Get news results
        news_results = google_news.get_news(query)

        # Parallel processing using joblib
        results_list = Parallel(n_jobs=-1)(delayed(process_article)(article) for article in news_results)

        # Filter out None values (articles not in English)
        results_list = [result for result in results_list if result is not None]

        # Create a DataFrame from the list of results
        df = pd.DataFrame(results_list)

        # Download button
        # Save the filtered DataFrame to an Excel file
        excel_file_path_filtered = f"{query.split()[0]}_{query.split()[1]}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_filtered.xlsx"
        df.to_excel(excel_file_path_filtered, index=False, engine='openpyxl')

        # Create an in-memory Excel file for the filtered DataFrame
        output_filtered = BytesIO()
        df.to_excel(output_filtered, index=False, engine='openpyxl')

        # Download button for the filtered DataFrame
        st.sidebar.download_button(
            label="Download Filtered Results",
            data=output_filtered.getvalue(),
            file_name=f"{query.split()[0]}_{query.split()[1]}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="filtered_excel",
            help="Click to download the filtered results"
        )

        # Display results
        st.subheader("Results")
        st.subheader("Click Download Filtered Results near the search button to save results to Excel File")
        st.dataframe(df)

        # Sample data loading
        data = results_list  # Use the scraped data directly
        stop_words_list = list(ENGLISH_STOP_WORDS)
        stop_words = list(ENGLISH_STOP_WORDS)

        # Data preprocessing for clustering
        df['text'] = df['text'].fillna('')
        df = df[~df['text'].str.contains("Save my User ID and Password")]
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))

        # Text vectorization using TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words_list)
        X = vectorizer.fit_transform(df['text'])

        # Clustering using K-Means
        num_clusters = 5  # Adjust the number of clusters based on your dataset
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)

        # Extractive summarization for each cluster using sumy
        cluster_summaries = []
        cluster_keywords = []
        for i in range(num_clusters):
            cluster_articles = df[df['cluster'] == i]

            if not cluster_articles.empty:
                cluster_text = ' '.join(cluster_articles['text'])

                # Using LSA (Latent Semantic Analysis) Summarizer
                parser = PlaintextParser.from_string(cluster_text, Tokenizer("english"))
                summarizer = LsaSummarizer()

                # Adjust the number of sentences based on your preference
                sentences_count = 3
                cluster_summary = ' '.join(str(sentence) for sentence in summarizer(parser.document, sentences_count))

                # Get keywords using TF-IDF
                terms = vectorizer.get_feature_names_out()
                cluster_tfidf_values = X[cluster_articles.index].toarray()
                avg_tfidf_scores = cluster_tfidf_values.mean(axis=0)
                top_keywords_idx = avg_tfidf_scores.argsort()[-5:][::-1]  # Adjust the number of keywords to display
                cluster_top_keywords = [terms[idx] for idx in top_keywords_idx]

                cluster_summaries.append(cluster_summary)
                cluster_keywords.append(cluster_top_keywords)

                # Print cluster information
                st.write(f"\nCluster {i} Summary:")
                st.write(f"Number of Articles: {len(cluster_articles)}")
                st.write(f"Cluster Summary: {cluster_summary}")
                st.write(f"Cluster Keywords: {', '.join(cluster_top_keywords)}")

        # Create a Word document and add content
        doc_cluster = Document()

        # Add cluster summaries and keywords to the document
        for i, (summary, keywords) in enumerate(zip(cluster_summaries, cluster_keywords), 1):
            doc_cluster.add_heading(f'Cluster {i} Summary', level=2)
            doc_cluster.add_paragraph(summary)
            doc_cluster.add_heading(f'Cluster {i} Keywords', level=2)
            doc_cluster.add_paragraph(', '.join(keywords))

        # Save the clustering document
        doc_cluster_path = 'cluster_summary_document.docx'
        doc_cluster.save(doc_cluster_path)

        # Download button for the clustering document
        st.sidebar.download_button(
            label="Download Cluster Summary",
            data=doc_cluster_path,
            file_name='cluster_summary_document.docx',
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="cluster_summary_doc",
            help="Click to download the cluster summary document"
        )
