#Necessary imports
import streamlit as st
#import nltk_download_utils
import pandas as pd
from matplotlib import pyplot as plt
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from gensim.summarization.summarizer import summarize 
import re
import spacy
import os
from datetime import datetime,timedelta
from transformers import pipeline
from transformers import pipeline, MarianMTModel, MarianTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import warnings
warnings.filterwarnings('ignore')
# Set Streamlit theme to dark

# Function to update the visit info in a file
def update_visit_info():
    file_path = "visit_info.txt"

    # Get the current timestamp
    current_time = datetime.now()

    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file doesn't exist, initialize the count and start time
        visit_count = 0
        start_time = current_time
        end_time = current_time

        # Write initial data to the file
        with open(file_path, "w") as file:
            file.write(f"{visit_count},{start_time.strftime('%Y-%m-%d %H:%M')},{end_time.strftime('%Y-%m-%d %H:%M')},{timedelta(0)}\n")
    else:
        # Read existing data from the file
        with open(file_path, "r") as file:
            data = file.read().split(",")

        # Check if the data has the expected length
        if len(data) >= 4:
            visit_count = int(data[0])
            start_time_str = data[1]
            end_time_str = data[2]

            # Extract only date and time without seconds
            start_time_str = ':'.join(start_time_str.split(':')[:2])
            end_time_str = ':'.join(end_time_str.split(':')[:2])

            # Convert string times to datetime objects
            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M")
        else:
            # If the data is incomplete, initialize the count and start time
            visit_count = 0
            start_time = current_time
            end_time = current_time

    # Increment the visit count
    visit_count += 1

    # Update end time to the current time
    end_time = current_time

    # Calculate total time spent online
    total_time_online = end_time - start_time

    # Append new data to the file
    with open(file_path, "a") as file:
        file.write(f"{visit_count},{start_time.strftime('%Y-%m-%d %H:%M')},{end_time.strftime('%Y-%m-%d %H:%M')},{total_time_online}\n")

    return visit_count, start_time, end_time, total_time_online

# Update visit info on each session
visit_info = st.session_state.get('visit_info', {'count': 0, 'start_time': datetime.now(), 'end_time': datetime.now(), 'total_time_online': timedelta()})

# Increment the visit count and get the current timestamps
visit_info['count'], visit_info['start_time'], visit_info['end_time'], visit_info['total_time_online'] = update_visit_info()

# Update session state
st.session_state.visit_info = visit_info

nlp = spacy.load("en_core_web_sm")

# Get the current datetime for timestamp
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

from collections import Counter
from wordcloud import WordCloud

# Function to save summary to a file
def save_summary_to_file(text, summary, timestamp):
    archive_folder = "results_archive"
    os.makedirs(archive_folder, exist_ok=True)
    folder_path = os.path.join(archive_folder, f"results_{timestamp}_Summary")
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, "summary.txt"), "w", encoding="utf-8") as file:
        file.write(f"Text: {text}\n")
        file.write(f"Summary: {summary}\n")

# Function to save entities to a file
def save_entities_to_file(text, entities, timestamp):
    archive_folder = "results_archive"
    os.makedirs(archive_folder, exist_ok=True)
    folder_path = os.path.join(archive_folder, f"results_{timestamp}_Entities") 
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, "entities.txt"), "w", encoding="utf-8") as file:
        file.write(f"Text: {text}\n")
        for ent_type, ent_list in entities.items():
            file.write(f"{ent_type} Entity: {', '.join(ent_list)}\n")

        
# Function to save word cloud image to a file
def save_word_cloud_to_file(wordcloud, timestamp):
    archive_folder = "results_archive"
    os.makedirs(archive_folder, exist_ok=True)
    folder_path =  os.path.join(archive_folder, f"results_{timestamp}_wordcloud") 
    os.makedirs(folder_path, exist_ok=True)

    # Save word cloud image as a PNG file
    wordcloud.to_file(os.path.join(folder_path, "word_cloud.png"))


#Headings for Web Application
st.title("Natural Language Processing Web Application")

#Textbox for text user is entering
st.subheader("Please Enter / Copy & Paste the text you'd like to analyze & press enter key ")
text = st.text_input('Enter text') #text is stored in this variable
source_en_text=text

st.subheader("What type of NLP service would you like to use ? ")

#Picking what NLP task you want to do
option = st.selectbox('NLP Service',('Sentiment Analysis', 'Entity Extraction', 'Text Summarization','Word Cloud','Translation')) #option is stored in this variable

# Language selection for translation
# Load local translation model
if option == 'Translation':
    st.info("Please wait. The machine learning model is working on translation...")
    model_path = '/home/arjun.murthy/NLP_SUM/model_fb'
    model = M2M100ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = M2M100Tokenizer.from_pretrained(model_path)

    # Set source language to English
    source_language = 'en'

    # Translate text from English to Korean
    tokenizer.src_lang = source_language
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("ko"))
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    st.write(f"Translated Text (Korean): {translated_text}")

#Display results of the NLP task
st.header("Results")

#Function to take in dictionary of entities, type of entity, and returns specific entities of specific type
def entRecognizer(entDict, typeEnt):
    entList = [ent for ent in entDict if entDict[ent] == typeEnt]
    return entList

#Word Cloud
if option == 'Word Cloud':

    wordcloud = WordCloud(background_color = 'lightcyan',
                      width = 1200,
                      height = 700).generate(str(text))

    # Display the generated image:
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(fig)
    # Save word cloud image to a file
    save_word_cloud_to_file(wordcloud, timestamp)

#Sentiment Analysis
if option == 'Sentiment Analysis':

    #Creating graph for sentiment across each sentence in the text inputted
    sents = sent_tokenize(text)
    entireText = TextBlob(text)
    sentScores = []
    for sent in sents:
        text = TextBlob(sent)
        score = text.sentiment[0]
        sentScores.append(score)

    #Plotting sentiment scores per sentence in line graph
    st.subheader('Sentiment Scores per Sentence Graph')
    st.line_chart(sentScores)

    #Polarity and Subjectivity of the entire text inputted
    sentimentTotal = entireText.sentiment
    st.write("The sentiment average polarity and subjectivity scores of the overall text below.")
    st.write(sentimentTotal)


# Load local translation model


#Named Entity Recognition
# Named Entity Recognition
elif option == 'Entity Extraction':
    # Getting Entity and type of Entity
    entities = []            # List to store extracted entities
    entity_labels = []       # List to store corresponding entity labels
    doc = nlp(text)          # Processing the text with spaCy NLP model

    # Add email entity extraction logic
    email_pattern = re.compile(r'\S+@\S+\.(com|net)')  # Simple pattern for email addresses

    for token in doc:
        if email_pattern.match(token.text):
            entities.append(token.text)
            entity_labels.append('EMAIL')

    # Add phone number entity extraction logic
    phone_pattern = re.compile(r'\+\d{1,4}\s?\d{1,4}[-.\s]?\d{1,12}')  # Updated pattern for phone numbers with optional country code and separators

    for token in doc:
        if phone_pattern.match(token.text):
            entities.append(token.text)
            entity_labels.append('PHONE')

    # Add address entity extraction logic
    address_pattern = re.compile(r'\b\d+\s\S+\s\S+,\s\S+,\s\S+\s\d{5}\b')  # Basic pattern for addresses with street address, city, state, and ZIP code

    for token in doc:
        if address_pattern.match(token.text):
            entities.append(token.text)
            entity_labels.append('ADDRESS')

    # Add web link recognition logic
    link_pattern = re.compile(r'https?://\S+|www\.\S+')  # Simple pattern for web links

    for token in doc:
        if link_pattern.match(token.text):
            entities.append(token.text)
            entity_labels.append('LINK')

    # Iterating over the entities identified by spaCy
    for ent in doc.ents:
        entities.append(ent.text)          # Storing the entity text
        entity_labels.append(ent.label_)   # Storing the entity label

    # Creating a dictionary with entities and their corresponding entity types
    ent_dict = dict(zip(entities, entity_labels))

    # List of entity types to extract
    entity_types_of_interest = ['ORG', 'CARDINAL', 'DATE', 'GPE', 'PERSON', 'MONEY', 'PRODUCT', 'TIME', 'PERCENT',
                                'WORK_OF_ART', 'QUANTITY', 'NORP', 'LOC', 'EVENT', 'ORDINAL', 'FAC', 'LAW', 'LANGUAGE', 'EMAIL', 'PHONE', 'ADDRESS', 'LINK']

    # Filter entities based on entity types of interest
    filtered_entities = {ent: ent_type for ent, ent_type in ent_dict.items() if ent_type in entity_types_of_interest}

     # Display the filtered entities using Streamlit
    grouped_entities = {}

    for ent, ent_type in filtered_entities.items():
        if ent_type in grouped_entities:
            grouped_entities[ent_type].append(ent)
        else:
            grouped_entities[ent_type] = [ent]

    for ent_type, ent_list in grouped_entities.items():
        #st.write(f"{ent_type} Entity: {' ,  '.join(ent_list)}")
        st.write(f"<b><u>{ent_type} Entity:</b></u> {', '.join(ent_list)}", unsafe_allow_html=True)

        # Save results to a file
    save_entities_to_file(text, grouped_entities, timestamp)

#Text Summarization
else:
    summWords = summarize(text)
    st.subheader("Summary")
    st.write(summWords)
    # Save results to a file
    save_summary_to_file(text, summWords, timestamp)


st.text("")
st.text("")
st.text("")
st.markdown("***")
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** spacy, gensim, matplotlib, pandas, nltk, textblob, huggingface
                     \n
pip install nltkcontact: arjun.murthy@samsung.com or vinay1.s@samsung.com if you need additional functionalities or face any issues


""")
