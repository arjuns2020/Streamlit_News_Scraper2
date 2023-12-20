import os
import pandas as pd
from gnews import GNews
from datetime import datetime
from langid import classify
from joblib import Parallel, delayed
import streamlit as st
import xlsxwriter
from io import BytesIO
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
from nltk.tokenize import word_tokenize
from docx import Document
import nltk
nltk.download('punkt')
import subprocess

@st.cache_resource
def download_en_core_web_sm():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    
nlp = spacy.load("en_core_web_sm")
# Load spaCy model
model_name = 'en_core_web_lg'
nlp = spacy.load(model_name)

# Function to check if the article is in English
def is_english(text):
    lang, _ = classify(text)
    return lang == 'en'

# Function to get full article using newspaper3k and perform NLP
def get_full_article_with_nlp(url):
    # ... (same as before)

# Function to process the article
def process_article(article):
    # ... (same as before)

# Sample data loading
data = df
stop_words_list = list(ENGLISH_STOP_WORDS)
stop_words = list(ENGLISH_STOP_WORDS)

# Data preprocessing for clustering
data['text'] = data['text'].fillna('')
data = data[~data['text'].str.contains("Save my User ID and Password")]
data = data.dropna(subset=['text'])
data['text'] = data['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))

# Streamlit UI
st.title("News Scraper and Analyzer")
st.sidebar.header("Search Parameters")

# User input
query = st.sidebar.text_input("Enter Keywords (separated by space)", 'datacentre cloud')
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

        # Data processing for clustering
        data['text'] = data['text'].fillna('')
        data = data[~data['text'].str.contains("Save my User ID and Password")]
        data = data.dropna(subset=['text'])
        data['text'] = data['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))

        # Text vectorization using TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words_list)
        X = vectorizer.fit_transform(data['text'])

        # Clustering using K-Means
        num_clusters = 5  # You can adjust this number based on your dataset
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)

        # Download button for the entire DataFrame
        entire_df_output = BytesIO()
        df.to_csv(entire_df_output, index=False)
        st.sidebar.download_button(
            label="Download Entire DataFrame",
            data=entire_df_output.getvalue(),
            file_name="entire_dataframe.csv",
            mime="text/csv",
            key="entire_dataframe",
            help="Click to download the entire DataFrame"
        )

# Display results
st.subheader("Results")
st.dataframe(df)

# Display cluster summaries
st.subheader("Cluster Summaries")
for i, (summary, keywords) in enumerate(zip(cluster_summaries, cluster_keywords), 1):
    st.write(f"\nCluster {i} Summary:")
    st.write(f"Cluster Summary: {summary}")
    st.write(f"Cluster Keywords: {', '.join(keywords)}")

# Download button for the cluster summaries
summary_doc = Document()
for i, (summary, keywords) in enumerate(zip(cluster_summaries, cluster_keywords), 1):
    summary_doc.add_heading(f'Cluster {i} Summary', level=2)
    summary_doc.add_paragraph(summary)
    summary_doc.add_heading(f'Cluster {i} Keywords', level=2)
    summary_doc.add_paragraph(', '.join(keywords))

# Save the document
output_summary = BytesIO()
summary_doc.save(output_summary)
st.sidebar.download_button(
    label="Download Cluster Summaries",
    data=output_summary.getvalue(),
    file_name="cluster_summaries.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    key="cluster_summaries_doc",
    help="Click to download the cluster summaries"
)
