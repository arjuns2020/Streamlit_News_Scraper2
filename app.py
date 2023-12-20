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
 

def is_english(text):
    lang, _ = classify(text)
    return lang == 'en'

def get_full_article_with_nlp(url):
    article = GNews().get_full_article(url)

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
        url = article.get('url', '')
        if url:
            source_from_url = url.split('/')[2]
            source = source_from_url if source_from_url else 'N/A'

    if is_english(article.get('title', '')):
        title = article.get('title', 'N/A')

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
            full_article = get_full_article_with_nlp(article.get('url'))
            result.update(full_article)
        except Exception as e:
            print(f"Error processing article: {e}")

        return result
    else:
        return None

# Sample data loading
data = df
stop_words_list = list(ENGLISH_STOP_WORDS)
stop_words = list(ENGLISH_STOP_WORDS)

data['text'] = data['text'].fillna('')
data = data[~data['text'].str.contains("Save my User ID and Password")]
data = data.dropna(subset=['text'])
data['text'] = data['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))

# Streamlit UI
st.title("News Scraper and Analyzer")
st.sidebar.header("Search Parameters")

query = st.sidebar.text_input("Enter Keywords (separated by space)", 'datacentre cloud')
start_date = st.sidebar.date_input("Enter Start Date", datetime(2023, 11, 1))
end_date = st.sidebar.date_input("Enter End Date", datetime(2023, 12, 30))

GNews.language = 'en'
google_news = GNews(country='US', max_results=100)

if st.sidebar.button("Search"):
    with st.spinner("Please wait while scraping the web..."):
        google_news.start_date = start_date
        google_news.end_date = end_date

        news_results = google_news.get_news(query)

        results_list = Parallel(n_jobs=-1)(delayed(process_article)(article) for article in news_results)
        results_list = [result for result in results_list if result is not None]
        df = pd.DataFrame(results_list)

        data['text'] = data['text'].fillna('')
        data = data[~data['text'].str.contains("Save my User ID and Password")]
        data = data.dropna(subset=['text'])
        data['text'] = data['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]))

        vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words_list)
        X = vectorizer.fit_transform(data['text'])

        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)

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

st.subheader("Results")
st.dataframe(df)

st.subheader("Cluster Summaries")
for i, (summary, keywords) in enumerate(zip(cluster_summaries, cluster_keywords), 1):
    st.write(f"\nCluster {i} Summary:")
    st.write(f"Cluster Summary: {summary}")
    st.write(f"Cluster Keywords: {', '.join(keywords)}")

summary_doc = Document()
for i, (summary, keywords) in enumerate(zip(cluster_summaries, cluster_keywords), 1):
    summary_doc.add_heading(f'Cluster {i} Summary', level=2)
    summary_doc.add_paragraph(summary)
    summary_doc.add_heading(f'Cluster {i} Keywords', level=2)
    summary_doc.add_paragraph(', '.join(keywords))

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
