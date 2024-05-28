import json
import base64
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os
from pyvis.network import Network
import webbrowser
import joblib
import re
from dotenv import load_dotenv
from textblob import TextBlob
from openai import OpenAI
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Load environment variables
load_dotenv()

CLIENT_SECRETS_FILE = os.getenv('CLIENT_SECRETS_FILE')
SCOPES = [os.getenv('SCOPES')]
TOKEN_JSON_FILE = os.getenv('TOKEN_JSON_FILE')
ENRICHED_JSON_FILE = os.getenv('ENRICHED_JSON_FILE')
EMBEDDINGS_FILE = 'umap_embeddings.pkl'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

# Ensure you have the necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# Function to clean and tokenize email content
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    tokens = word_tokenize(text.lower())  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stop words
    return ' '.join(tokens)

def sanitize_for_json(text):
    return json.dumps(text)

def ensure_utf8(text):
    # Decode using UTF-8, replacing invalid bytes
    clean_text = text.encode('utf-8', 'replace').decode('utf-8', 'replace')
    # Remove substrings enclosed in <| and |> (special tokens)
    clean_text = re.sub(r'<\|.*?\|>', '', clean_text)
    # Remove extra spaces
    clean_text = ' '.join(clean_text.split())
    return clean_text

def sanitize_text(text):
    if not text:
        return " "
    return ensure_utf8(text)

def divide_list_into_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]
        
def extract_top_keywords(texts, n_keywords=3):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
    top_n = tfidf_sorting[:n_keywords]
    
    return [terms[i] for i in top_n]

def label_clusters(texts, labels):
    cluster_texts = {}
    for text, label in zip(texts, labels):
        if label not in cluster_texts:
            cluster_texts[label] = []
        cluster_texts[label].append(text)
    
    cluster_keywords = {}
    for label, texts in cluster_texts.items():
        keywords = extract_top_keywords(texts)
        cluster_keywords[label] = ' '.join(keywords)
    
    return cluster_keywords

def visualize_topics(embeddings, labels, texts):
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
    umap_embeddings = umap_model.fit_transform(embeddings)
    
    cluster_keywords = label_clusters(texts, labels)
    
    # Plotting the clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar()
    plt.title('UMAP projection of the email embeddings')
    
    # Annotate clusters
    for label, keywords in cluster_keywords.items():
        x, y = np.mean(umap_embeddings[labels == label], axis=0)
        plt.text(x, y, keywords, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.show()
    
    return umap_embeddings

# Function to get OpenAI embeddings
def get_openai_embeddings(texts):
    embeddings = []
    sanitized_texts = [sanitize_text(text) for text in texts]
   
    for batch in divide_list_into_batches(sanitized_texts, batch_size=100):  # Adjust batch size if needed
        # Ensure batch is a list of non-empty strings
        batch = [text for text in batch if text.strip()]
        
        # Check token count
        total_tokens = sum(len(text.split()) for text in batch)
        if total_tokens > 8192:
            print(f"Skipping batch with {total_tokens} tokens, which exceeds the limit.")
            continue
        
        if not batch:
            continue
        
        try:
            response = client.embeddings.create(input=batch, model="text-embedding-3-small")
            embeddings.extend(np.array(item.embedding) for item in response.data)
        except Exception as e:
            print(f"Error processing batch: {e}")
            print(f"Batch content: {batch}")
        
    return embeddings

def generate_or_load_embeddings(texts, force_regenerate=False):
    if os.path.exists(EMBEDDINGS_FILE) and not force_regenerate:
        print("Loading existing embeddings...")
        umap_embeddings = joblib.load(EMBEDDINGS_FILE)
    else:
        print("Generating new embeddings...")
        embeddings = np.vstack([get_openai_embeddings([text]) for text in texts])
        
        umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine')
        umap_embeddings = umap_model.fit_transform(embeddings)

        joblib.dump(umap_embeddings, EMBEDDINGS_FILE)
        print("Embeddings saved.")
        
    return umap_embeddings

# Function to get sentiment score
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to get email interactions
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

# Initialize the network graph
def initialize_network_graph():
    nt = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    nt.toggle_physics(False)
    return nt

# Calculate strength of ties from email data
def calculate_strength_of_ties(email_df):
    email_df['receivers'] = email_df['receivers'].str.split(', ')
    interactions = email_df.explode('receivers')
    tie_strength = interactions.groupby(['sender', 'receivers']).size().reset_index(name='count')
    return tie_strength

# Function to integrate email data with relationship strength into the network graph
def integrate_email_data_with_topics(nt, email_df, umap_embeddings, labels, edges_added):
    email_df['x'] = umap_embeddings[:, 0]
    email_df['y'] = umap_embeddings[:, 1]
    email_df['topic'] = labels
    
    for index, row in email_df.iterrows():
        sender = row['sender']
        receivers = row['receivers'].split(', ')
        topic = row['topic']

        if sender not in nt.node_ids:
            nt.add_node(sender, label=sender, color="#f54242", title=f"Sender: {sender}, Topic: {topic}", x=row['x'], y=row['y'])

        for receiver in receivers:
            if receiver not in nt.node_ids:
                receiver_x = email_df[email_df['sender'] == receiver]['x'].values[0] if not email_df[email_df['sender'] == receiver].empty else row['x']
                receiver_y = email_df[email_df['sender'] == receiver]['y'].values[0] if not email_df[email_df['sender'] == receiver].empty else row['y']
                nt.add_node(receiver, label=receiver, color="#4287f5", title=f"Receiver: {receiver}, Topic: {topic}", x=receiver_x, y=receiver_y)

            if (sender, receiver) not in edges_added:
                nt.add_edge(sender, receiver, title=f"{sender} -> {receiver} (Strength: {row['relationship_strength']})", width=2)
                edges_added.add((sender, receiver))

# Function to add a legend to the network graph
def add_legend(nt):
    nt.add_node("Legend", label="Legend", shape="box", title="Legend: Node Colors<br>Red: Sender<br>Blue: Receivers")

# Function to display the network graph
def display_graph(nt, filename):
    nt.show_buttons(filter_=['physics'])
    nt.repulsion(node_distance=800, central_gravity=0.1, spring_length=350, spring_strength=0.05)
    nt.save_graph(filename)
    html_path = os.path.abspath(filename)
    print(f"Graph saved to {html_path}")
    webbrowser.open('file://' + html_path)

# Cluster topics
def cluster_topics(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_, kmeans.cluster_centers_

# Visualize topics
def visualize_topics(embeddings, labels):
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine')
    umap_embeddings = umap_model.fit_transform(embeddings)
    
    # Plotting the clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar()
    plt.title('UMAP projection of the email embeddings')
    plt.show()
    
    return umap_embeddings

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
    
    # Limit to a larger set of emails for meaningful clustering
    email_df = email_df.tail(100)

    # Preprocess email content
    email_df['cleaned_content'] = email_df['body'].apply(preprocess_text)

    # Generate embeddings
    embeddings = np.vstack([get_openai_embeddings(text) for text in email_df['cleaned_content']])

    # Cluster topics
    labels, _ = cluster_topics(embeddings)

    # Visualize topics and get 2D embeddings
    umap_embeddings = visualize_topics(embeddings, labels, email_df['cleaned_content'])

    # Calculate sentiment and relationship strength
    email_df['sentiment'] = email_df['body'].apply(get_sentiment)
    interaction_count = email_df.groupby(['sender', 'receivers']).size().reset_index(name='frequency')
    email_df = pd.merge(email_df, interaction_count, on=['sender', 'receivers'])
    email_df['relationship_strength'] = email_df['sentiment'] * email_df['frequency']

    # Integrate email data with relationship strength and topics into the network graph
    integrate_email_data_with_topics(nt, email_df, umap_embeddings, labels, edges_added)

    add_legend(nt)

    display_graph(nt, "network.html")

if __name__ == "__main__":
    main()