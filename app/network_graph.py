import json
import base64
import pandas as pd
import numpy as np
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
from pyvis.network import Network
import webbrowser
from bertopic import BERTopic
import umap
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import re


EMBEDDINGS_FILE = 'umap_embeddings.pkl'
TOPICS_FILE = 'topics.pkl'

CLIENT_SECRETS_FILE = 'client_secret_902701714108-2a6goqcd7bqmdosgms4k495bt4j09a9j.apps.googleusercontent.com.json'
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
TOKEN_JSON_FILE = 'token.json'
ENRICHED_JSON_FILE = 'files/enriched_email_interactions.json'

def get_email_interactions():
    creds = None
    if os.path.exists(TOKEN_JSON_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_JSON_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_JSON_FILE, 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    query = 'from:nebaird.sail@gmail.com'
    email_data = []
    next_page_token = None

    while True:
        results = service.users().messages().list(userId='me', q=query, pageToken=next_page_token).execute()
        messages = results.get('messages', [])
        next_page_token = results.get('nextPageToken')

        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            headers = msg['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), '')
            sender = next((header['value'] for header in headers if header['name'] == 'From'), '')
            receivers = [header['value'] for header in headers if header['name'] == 'To']
            body = ''
            if 'parts' in msg['payload']:
                for part in msg['payload']['parts']:
                    if part['mimeType'] == 'text/plain':
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
            
            email_data.append({
                'id': msg['id'],
                'subject': subject,
                'body': body,
                'sender': sender,
                'receivers': ', '.join(receivers)
            })

        if not next_page_token:
            break
    
    return email_data

def initialize_network_graph():
    nt = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    nt.toggle_physics(False)
    return nt

def calculate_strength_of_ties(email_df):
    email_df['receivers'] = email_df['receivers'].str.split(', ')
    interactions = email_df.explode('receivers')
    tie_strength = interactions.groupby(['sender', 'receivers']).size().reset_index(name='count')
    return tie_strength

def generate_bert_embeddings(texts):
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(TOPICS_FILE):
        print("Loading existing embeddings and topics...")
        umap_embeddings = joblib.load(EMBEDDINGS_FILE)
        topics = joblib.load(TOPICS_FILE)
        topic_model = BERTopic.load("topic_model")
    else:
        print("Generating new embeddings and topics...")
        topic_model = BERTopic(verbose=True)
        topics, probabilities = topic_model.fit_transform(texts)
        
        # Ensure probabilities is a 2D array
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        
        umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
        umap_embeddings = umap_model.fit_transform(probabilities)

        joblib.dump(umap_embeddings, EMBEDDINGS_FILE)
        joblib.dump(topics, TOPICS_FILE)
        topic_model.save("topic_model")
        print("Embeddings and topics saved.")
        
    return topics, umap_embeddings, topic_model

def extract_key_phrases(texts, topics):
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    topic_labels = {}
    for topic in set(topics):
        if topic == -1:  # Skip the outlier topic
            continue
        topic_texts = [texts[i] for i, t in enumerate(topics) if t == topic]
        X_topic = vectorizer.transform(topic_texts)
        
        # Sum the counts of each n-gram
        ngram_sums = X_topic.sum(axis=0)
        ngrams_freq = [(ngram, ngram_sums[0, idx]) for ngram, idx in vectorizer.vocabulary_.items()]
        sorted_ngrams = sorted(ngrams_freq, key=lambda x: x[1], reverse=True)
        
        # Create label from the top n-grams
        top_ngrams = [ngram for ngram, _ in sorted_ngrams[:5]]
        topic_labels[topic] = ' | '.join(top_ngrams)
    
    return topic_labels

def clean_labels(labels):
    cleaned_labels = {}
    for topic, label in labels.items():
        # Remove any emails and URLs from the labels
        cleaned_label = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', label)
        cleaned_label = re.sub(r'http\S+', '', cleaned_label)
        cleaned_labels[topic] = cleaned_label.strip()
    return cleaned_labels

def create_custom_labels(topic_model):
    topic_info = topic_model.get_topic_info()
    labels = {}
    for topic in topic_info['Topic'].values:
        if topic != -1:  # -1 is usually the outlier topic
            words = topic_model.get_topic(topic)
            label = ', '.join([word for word, _ in words[:5]])
            labels[topic] = label
    return labels

def integrate_email_data_with_topics(nt, email_df, umap_embeddings, edges_added):
    email_df['x'] = umap_embeddings[:, 0]
    email_df['y'] = umap_embeddings[:, 1]

    for index, row in email_df.iterrows():
        sender = row['sender']
        receivers = row['receivers'].split(', ')

        if sender not in nt.node_ids:
            nt.add_node(sender, label=sender, color="#f54242", title=f"Sender: {sender}", x=row['x'], y=row['y'])

        for receiver in receivers:
            if receiver not in nt.node_ids:
                receiver_x = email_df[email_df['sender'] == receiver]['x'].values[0] if not email_df[email_df['sender'] == receiver].empty else row['x']
                receiver_y = email_df[email_df['sender'] == receiver]['y'].values[0] if not email_df[email_df['sender'] == receiver].empty else row['y']
                nt.add_node(receiver, label=receiver, color="#4287f5", title=f"Receiver: {receiver}", x=receiver_x, y=receiver_y)

            if (sender, receiver) not in edges_added:
                nt.add_edge(sender, receiver, title=f"{sender} -> {receiver}", width=2)
                edges_added.add((sender, receiver))

def add_legend(nt):
    nt.add_node("Legend", label="Legend", shape="box", title="Legend: Node Colors<br>Red: Sender<br>Blue: Receivers")

def display_graph(nt, filename):
    nt.show_buttons(filter_=['physics'])
    nt.repulsion(node_distance=800, central_gravity=0.1, spring_length=350, spring_strength=0.05)
    nt.save_graph(filename)
    html_path = os.path.abspath(filename)
    print(f"Graph saved to {html_path}")
    webbrowser.open('file://' + html_path)

def visualize_topics_with_key_phrases(texts, topics, topic_model):
    fig = topic_model.visualize_topics()
    
    # Get key phrases for each topic
    labels = extract_key_phrases(texts, topics)
    cleaned_labels = clean_labels(labels)
    
    # Add labels to the plot
    for trace in fig.data:
        if trace.name in cleaned_labels:
            trace.text = cleaned_labels[trace.name]
    
    fig.show()

def main():
    nt = initialize_network_graph()
    edges_added = set()

    email_df = None

    if os.path.exists(ENRICHED_JSON_FILE):
        with open(ENRICHED_JSON_FILE, 'r') as file:
            data = [json.loads(line) for line in file]
        email_df = pd.DataFrame(data)
    else:
        email_data = get_email_interactions()
        email_df = pd.DataFrame(email_data)
        email_df.to_json(ENRICHED_JSON_FILE, orient='records', lines=True)

    texts = email_df['subject'] + ' ' + email_df['body']
    topics, umap_embeddings, topic_model = generate_bert_embeddings(texts)

    integrate_email_data_with_topics(nt, email_df, umap_embeddings, edges_added)

    add_legend(nt)
    
    display_graph(nt, "network.html")

    visualize_topics_with_key_phrases(texts, topics, topic_model)

if __name__ == "__main__":
    main()